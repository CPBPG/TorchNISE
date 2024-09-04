import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import torchnise

from torchnise.pytorch_utility import renorm

from torchnise.averaging_and_lifetimes import (
    estimate_lifetime,
    averaging
    )

from torchnise.absorption import (
    absorption_time_domain,
    absorb_time_to_freq
    )

from torchnise.fft_noise_gen import gen_noise



                

def NISE_propagate(Hfull,realizations,psi0,total_time,dt,T,save_Interval=1,T_correction="None",
                   device="cpu",saveU=False,saveCoherence=False,MLNISE_inputs=None,MLNISE_Model=None):
    """
    Propagate the quantum state using the NISE algorithm with optional thermal corrections.

    Args:
        Hfull (torch.Tensor): Hamiltonian of the system over time for different realizations.
        realizations (int): Number of noise realizations to simulate.
        psi0 (torch.Tensor): Initial state of the system.
        total_time (float): Total time for the simulation.
        dt (float): Time step size.
        T (float): Temperature for thermal corrections.
        save_Interval (int, optional): Interval for saving results. Defaults to 1.
        T_correction (str, optional): Method for thermal correction ("None", "TNISE", "MLNISE"). Defaults to "None".
        device (str, optional): Device for computation ("cpu" or "cuda"). Defaults to "cpu".
        saveU (bool, optional): If True, save time evolution operators. Defaults to False.
        saveCoherence (bool, optional): If True, save coherences. Defaults to False.
        MLNISE_inputs (tuple, optional): Inputs for MLNISE model. Defaults to None.
        MLNISE_Model (nn.Module, optional): Machine learning model for MLNISE corrections. Defaults to None.

    Returns:
        tuple: (torch.Tensor, torch.Tensor, torch.Tensor) - Populations, coherences, and time evolution operators.
    """
    n_sites=Hfull.shape[-1] #Number of Sites
    factor=1j* 1/torchnise.units.hbar*dt*torchnise.units.t_unit
    kBT=T*torchnise.units.k
    
    total_steps=int(total_time/dt)+1
    total_steps_saved=int(total_time/dt/save_Interval)+1
    PSLOC=torch.zeros((realizations,total_steps_saved,n_sites),device=device)
    
    if saveCoherence:
        CohLoc=torch.zeros((realizations,total_steps_saved,n_sites,n_sites),
                           device=device,dtype=torch.complex64)
    H=Hfull[:,0,:,:].clone().to(device=device) #grab the 0th timestep [all realizations : 0th timestep  : all sites : all sites]
    E,C=torch.linalg.eigh(H) #get initial eigenenergies and transition matrix from eigen to site basis.
    # Since the Hamiltonian is hermitian we van use eigh
    # H contains the hamiltonians of all realizations. To our advantage The eigh torch function (like almost all torch functions)
    # is setup so that it can efficently calculate the results for a whole batch of inputs
    Cold=C
    Eold=E
    psi0=psi0.repeat(realizations,1)
    psi0=psi0.unsqueeze(-1)
    pop0=(psi0[:,:,0]**2)
    PSLOC[:,0,:]=pop0 #Save the population of the first timestep
    phiB=Cold.transpose(1,2).to(dtype=torch.complex64).bmm(psi0.to(dtype=torch.complex64)) #Use the transition Matrix to transfer to the eigenbasis. Bmm is a batched matrix multiplication, so it does the matrix multiplication for all batches at once
    if saveCoherence:
        for i in range(n_sites):
            CohLoc[:,0,i,i]=pop0[:,i]
    if saveU:
        ULOC=torch.zeros((realizations,total_steps_saved,n_sites,n_sites),device=device,dtype=torch.complex64)        
        identiy=torch.eye(n_sites,dtype=torch.complex64)
        identiy=identiy.reshape(1,n_sites,n_sites)
        ULOC[:,0,:,:]= identiy.repeat(realizations,1,1)
        UB=Cold.transpose(1,2).to(dtype=torch.complex64).bmm(ULOC[:,0,:,:])
        UB=UB.to(dtype=torch.complex64).to(device=device)
        
    #Now we get to the step by step timepropagation. We start at 1 and not 0 since we have already filled the first slot of our population dynamics
    for t in tqdm.tqdm(range(1,total_steps)):
        H=Hfull[:,t,:,:].clone().to(device=device)   #grab the t'th timestep [all realizations : t'th timestep  : all sites : all sites]
        E,v_eps=torch.linalg.eigh(H)
        C=v_eps
        S=torch.matmul(C.transpose(1,2),Cold) #multiply the old with the new transition matrix to get the non-adiabatic coupling
        if T_correction.lower() in ["mlnise","tnise"]:
            S=t_correction(S,n_sites,realizations,device,E,Eold,T_correction,kBT,MLNISE_Model,phiB)
        U=torch.diag_embed(torch.exp(-E[:,:]*factor)).bmm(S.to(dtype=torch.complex64))  #Make the Time Evolution operator
        phiB=U.bmm(phiB) #Apply the time evolution operator
        if saveU:
            UB=U.bmm(UB)
        if T_correction.lower() in ["mlnise","tnise"]:
            phiB=renorm(phiB,dim=1) #Renormalize
        Cold=C #Set the new variables to the old variables for the next step
        Eold=E #

        C=C.to(dtype=torch.complex64)  #Make the transition matrix complex

        PhiBinLocBase=C.bmm(phiB) #Tranision to the site basis
        
        if t%save_Interval==0:
            PSLOC[:,t//save_Interval,:]=((PhiBinLocBase.abs()**2)[:,:,0]) #Save the result in the site Basis
            if saveU:
                if T_correction.lower() in ["mlnise","tnise"]:
                    for i in range(0,n_sites):
                        UB_Norm_Row=renorm(UB[:,:,i],dim=1)
                        UB[:,:,i]=UB_Norm_Row[:,:]
                ULOC[:,t//save_Interval,:,:]=(C.bmm(UB))              
            if saveCoherence:
                CohLoc[:,t//save_Interval,:,:]= PhiBinLocBase.squeeze(-1)[:, :, None] * PhiBinLocBase.squeeze(-1)[:, None, :].conj()
    if not saveCoherence:
        CohLoc = None
    else:
        CohLoc = CohLoc.cpu()
    if not saveU:
        ULOC = None
    else:
        ULOC = ULOC.cpu()
    return PSLOC.cpu(), CohLoc ,ULOC


def t_correction(S,n_sites,realizations,device,E,Eold,T_correction,kBT,MLNISE_Model,phiB):
    """
    Apply thermal corrections to the non-adiabatic coupling matrix.

    Args:
        S (torch.Tensor): Non-adiabatic coupling matrix.
        n_sites (int): Number of sites in the system.
        realizations (int): Number of noise realizations.
        device (str): Device for computation ("cpu" or "cuda").
        E (torch.Tensor): Eigenvalues of the Hamiltonian at the current time step.
        Eold (torch.Tensor): Eigenvalues of the Hamiltonian at the previous time step.
        T_correction (str): Method for thermal correction ("TNISE", "MLNISE").
        kBT (float): Thermal energy (k_B * T).
        MLNISE_Model (nn.Module): Machine learning model for MLNISE corrections.
        phiB (torch.Tensor): Wavefunction in the eigenbasis.
 
    Returns:
        torch.Tensor: Corrected non-adiabatic coupling matrix.
    """
    for ii in range(0,n_sites):
        maxC=torch.max(S[:,:,ii].real**2,1) #The order of the eigenstates is not well defined and might be flipped from one transition matrix to the transition matrix
        #to find the eigenstate that matches the previous eigenstate we find the eigenvectors that overlapp the most and we use the index with the highest overlap (kk)
        #instead of the original index ii to index the non adiabatic coupling correctly
        kk=maxC.indices
        CD=torch.zeros(realizations,device=device)
        for jj in range(0,n_sites):
            DE=E[:,jj]-Eold[:,ii]
            DE[jj==kk]=0 #if they are the same state the energy difference is 0
            if T_correction=="TNISE":
                correction=torch.exp(-DE/kBT/4)
            else:
                correction=MLNISE_Model(DE,kBT,phiB,S,jj,ii,realizations,device=device)
            S[:,jj,ii]=S[:,jj,ii].clone()*correction
            AddCD=S[:,jj,ii].clone()**2
            AddCD[jj==kk]=0
            CD=CD.clone()+AddCD
        #The renormalization procedure broken into smaller steps, 
        #because previously some errors showed
        #should probably be simplified
        norm=torch.abs(S[torch.arange(realizations),kk,ii].clone())
        dummy1=torch.ones(realizations,device=device)
        dummy1[1-CD>0]=torch.sqrt(1-CD[1-CD>0])
        dummy2=torch.zeros(realizations,device=device)
        S_clone=S[torch.arange(realizations),kk,ii].clone()
        dummy2[1-CD>0]=S_clone.clone()[1-CD>0]/norm[1-CD>0]
        dummy3=dummy1*dummy2
        S[1-CD>0,kk[1-CD>0],ii]=dummy3[1-CD>0]
        return S

def NISE_averaging(Hfull,realizations,psi0,total_time,dt,T,save_Interval=1,T_correction="None",averaging_method="standard",
         lifetime_factor=5,device="cpu", saveCoherence=False,saveU=False,MLNISE_inputs=None):
    """
    Run NISE propagation with different averaging methods to calculate averaged population dynamics.

    Args:
        Hfull (torch.Tensor): Hamiltonian of the system over time for different realizations.
        realizations (int): Number of noise realizations to simulate.
        psi0 (torch.Tensor): Initial state of the system.
        total_time (float): Total time for the simulation.
        dt (float): Time step size.
        T (float): Temperature for thermal corrections.
        save_Interval (int, optional): Interval for saving results. Defaults to 1.
        T_correction (str, optional): Method for thermal correction ("None", "TNISE", "MLNISE"). Defaults to "None".
        averaging_method (str, optional): Method for averaging results ("standard", "boltzmann", "interpolated"). Defaults to "standard".
        lifetime_factor (int, optional): Factor to scale estimated lifetimes. Defaults to 5.
        device (str, optional): Device for computation ("cpu" or "cuda"). Defaults to "cpu".
        saveCoherence (bool, optional): If True, save coherences. Defaults to False.
        saveU (bool, optional): If True, save time evolution operators. Defaults to False.
        MLNISE_inputs (tuple, optional): Inputs for MLNISE model. Defaults to None.

    Returns:
        tuple: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) - Averaged populations, coherences, time evolution operators, and lifetimes.
    """
    with torch.no_grad():

        n_sites=Hfull.shape[-1] # get the number of sites from the size of the hamiltonian
        lifetimes=None
        
        #run NISE without T correction
        if averaging_method.lower() in ["boltzmann","interpolated"]:
            population, coherence ,U= NISE_propagate(Hfull.to(device),realizations,psi0.to(device),total_time,dt,T,
                                                     save_Interval=save_Interval,T_correction="None",
                                                     device=device,saveU=True,saveCoherence=True)
            lifetimes=estimate_lifetime(U, dt*save_Interval)
            lifetimes=lifetimes*lifetime_factor
        else:
            lifetimes=None
        
        population, coherence ,U= NISE_propagate(Hfull.to(device),realizations,psi0.to(device),total_time,dt,T,
                                                 save_Interval=save_Interval,T_correction=T_correction,
                                                 device=device,saveU=saveU,saveCoherence=True) 
            
        population_averaged,coherence_ave=averaging(population,averaging_method,lifetimes=lifetimes,
                                                    step=dt,coherence=coherence)

        return population_averaged,coherence_ave,U, lifetimes



        

def run_NISE(H,realizations, total_time,dt, initialState,T, spectral_funcs,save_Interval=1,
               T_correction="None",mode="Population",mu=None, aborption_padding=10000,
               averaging_method="Standard", lifetime_factor=5,maxreps=100000,MLNISE_inputs=None,
               device="cpu"):
    """
    Main function to run NISE simulations for population dynamics or absorption spectra.
    
    Args:
        H (torch.Tensor): Hamiltonian of the system over time or single Hamiltonian with noise.
        realizations (int): Number of noise realizations to simulate.
        total_time (float): Total time for the simulation.
        dt (float): Time step size.
        initialState (torch.Tensor): Initial state of the system.
        T (float): Temperature for thermal corrections.
        spectral_funcs (callable): Spectral density functions for noise generation.
        save_Interval (int, optional): Interval for saving results. Defaults to 1.
        T_correction (str, optional): Method for thermal correction ("None", "TNISE", "MLNISE"). Defaults to "None".
        mode (str, optional): Simulation mode ("Population" or "Absorption"). Defaults to "Population".
        mu (torch.Tensor, optional): Dipole moments for absorption calculations. Defaults to None.
        aborption_padding (int, optional): Padding for absorption spectra calculation. Defaults to 10000.
        averaging_method (str, optional): Method for averaging results ("Standard", "Boltzmann", "Interpolated"). Defaults to "Standard".
        lifetime_factor (int, optional): Factor to scale estimated lifetimes. Defaults to 5.
        maxreps (int, optional): Maximum number of realizations per chunk. Defaults to 100000.
        MLNISE_inputs (tuple, optional): Inputs for MLNISE model. Defaults to None.
        device (str, optional): Device for computation ("cpu" or "cuda"). Defaults to "cpu".
    
    Returns:
        tuple: Depending on mode, returns either (np.ndarray, np.ndarray) for absorption spectrum and frequency axis,
               or (torch.Tensor, torch.Tensor) for averaged populations and time axis.
    """   
    total_steps = int((total_time + dt) / dt)
    save_steps  = int((total_time + dt*save_Interval) / (dt*save_Interval))
    n_state = H.shape[-1] #H_0.shape[1] if time_dependent_H else H_0.shape[0]
    window=1
    time_dependent_H= len(H.shape)>=3
    avg_absorb_time = None
    x_axis = None
    absorb_f = None
    avg_output = None
    if time_dependent_H:
        trajectory_steps = H.shape[0]
        if realizations > 1:
            window = int((trajectory_steps - total_steps) / (realizations - 1))
            print(f"window is {window * dt} {torchnise.units.current_t_unit}")

    def generate_Hfull_chunk(chunk_size, start_index=0,window=1):
        if time_dependent_H:
            chunk_Hfull = torch.zeros((chunk_size, total_steps, n_state, n_state))
            for j in range(chunk_size):
                H_index = start_index + j
                chunk_Hfull[j, :, :, :] = torch.tensor(H[window * H_index:window * H_index + total_steps, :, :])
            return chunk_Hfull 
        else:
            chunk_Hfull = torch.zeros((chunk_size, total_steps, n_state, n_state))
            print("generating noise")
            mynoise = gen_noise(spectral_funcs,dt,(chunk_size,total_steps,n_state))    
            print("building H")
            chunk_Hfull[:] = H
            for i in range(n_state):
                chunk_Hfull[:, :, i, i] += mynoise[:, :, i]
            return chunk_Hfull

    if realizations > maxreps:
        num_chunks = (realizations + maxreps - 1) // maxreps  # This ensures rounding up
        print("splitting calculation into ", num_chunks, " chunks")
    else:
        num_chunks=1
    chunk_size = (realizations + num_chunks - 1) // num_chunks  # This ensures each chunk is not greater than reals

    if mode.lower()=="population" and T_correction.lower() in ["mlnise","tnise"] and averaging_method in ["interpolated","boltzmann"]:
        all_coherence = torch.zeros(num_chunks,save_steps,n_state,n_state)
        all_lifetimes = torch.zeros(num_chunks,n_state)
    elif mode.lower() =="absorption":
        all_absorb_time=[]
    saveU = mode.lower() =="absorption"

    weights = []
    all_output = torch.zeros(num_chunks,save_steps,n_state)
    for i in range(0, realizations, chunk_size):
        chunk_reps = min(chunk_size, realizations - i)
        weights.append(chunk_reps)      
        chunk_Hfull = generate_Hfull_chunk(chunk_reps, start_index=i,window=window)
        print("running calculation")
        population_averaged,coherence_ave,U, lifetimes = NISE_averaging(chunk_Hfull, chunk_reps,initialState,total_time,
                                                                        dt,T,save_Interval=save_Interval,T_correction=T_correction,
                                                                        averaging_method=averaging_method,lifetime_factor=lifetime_factor,
                                                                        device=device,saveU=saveU,saveCoherence=True,MLNISE_inputs=None)

        if mode.lower() =="population" and T_correction.lower() in ["mlnise","tnise"] and averaging_method in ["interpolated","boltzmann"]:
            all_coherence[i//chunk_size,:,:,:]= coherence_ave
            all_lifetimes[i//chunk_size,:] = lifetimes
        elif mode.lower() == "absorption":
            absorb_time=absorption_time_domain(U,mu)
            all_absorb_time.append(absorb_time)
        all_output[i//chunk_size,:,:]=population_averaged
    if mode.lower() =="population" and T_correction.lower() in ["mlnise","tnise"] and averaging_method in ["interpolated","boltzmann"]:
        lifetimes=torch.mean(all_lifetimes,dim=0)
        print(f"lifetimes multipiled by lieftime factor are {lifetimes}")
        avg_output,_ = averaging(all_output,averaging_method,lifetimes=lifetimes,step=dt,coherence=all_coherence,weight=torch.tensor(weights,dtype=torch.float))#np.average(all_oldave, axis=0, weights=weights)
    else:
        lifetimes=None
        avg_output = np.average(all_output, axis=0, weights=weights)
    

    if mode.lower() =="absorption":
        avg_absorb_time = np.average(all_absorb_time, axis=0, weights=weights)
        pad=int(aborption_padding/(dt*torchnise.units.t_unit))
        absorb_config = {
            "total_time":total_time,
            "dt": dt,
            "pad": pad,
            "smoothdamp": True,
            "smoothdamp_start_percent": 10
        }
        absorb_f, x_axis = absorb_time_to_freq(avg_absorb_time, absorb_config)
        return absorb_f, x_axis
    else:
        return avg_output, torch.linspace(0,total_time,avg_output.shape[0])
    

   
class MLNISE_model(nn.Module):
    """
    Predict correction factors for non-adiabatic coupling based on input features.

    Args:
        MLNISE_inputs (tuple): Inputs for MLNISE model.
        DE (torch.Tensor): Energy differences between states.
        kBT (float): Thermal energy (k_B * T).
        phiB (torch.Tensor): Wavefunction in the eigenbasis.
        S (torch.Tensor): Non-adiabatic coupling matrix.
        jj (int): Index of the target state.
        ii (int): Index of the current state.
        realizations (int): Number of noise realizations.
        device (str, optional): Device for computation ("cpu" or "cuda"). Defaults to "cpu".

    Returns:
        torch.Tensor: Correction factor for the non-adiabatic coupling matrix.
    """
    #init contains all the free parameters of our method. In case of the neural networks its the layers of the neural networks
    def __init__(self):
        super(MLNISE_model, self).__init__()
        self.fc1 = nn.Linear(8, 75)
        self.fc2 = nn.Linear(75, 75)
        self.fc3 = nn.Linear(75, 75)
        self.fc4 = nn.Linear(75, 1)
    def forward(self,MLNISE_inputs,DE,kBT,phiB,S,jj,ii,realizations,device="cpu"):
        #Initialize the input Vector for the neural network
        input_Vec=torch.zeros(realizations,8,device=device)

        #Currently it contains the following features:

        #Feature 0  is the energy difference
        input_Vec[:,0]=DE

        #Feature 1 is the current population ratio between eigenstate i and j

        #this first part checks if any ratio (from any of the batches) is bigger than 100x to avoid exploding gradients. Those ratios are then set to 100

        if ((phiB.real[:,ii]**2+phiB.imag[:,ii]**2)<0.01*(phiB.real[:,jj]**2+phiB.imag[:,jj]**2)).any():
            mask_input_vec =((phiB.real[:,ii]**2+phiB.imag[:,ii]**2)>0.01*(phiB.real[:,jj]**2+phiB.imag[:,jj]**2))
            mask_input_vec= mask_input_vec.squeeze()
            input_Vec[:,1]=100
            input_Vec[mask_input_vec,1]=((phiB.real[mask_input_vec,jj]**2+phiB.imag[mask_input_vec,jj]**2)/(phiB.real[mask_input_vec,ii]**2+phiB.imag[mask_input_vec,ii]**2)).squeeze()


        # if all ratios are <100 simply use the ratios

        else:
            input_Vec[:,1]=((phiB.real[:,jj]**2+phiB.imag[:,jj]**2)/(phiB.real[:,ii]**2+phiB.imag[:,ii]**2)).squeeze()
        #Feature 2 is the original value of the Nonadiabatic coupling
        input_Vec[:,2]=S[:,jj,ii]
        #Feature 3 is the Temperature
        input_Vec[:,3]=kBT
        #Feature 4 is the reorganization energy
        Er=MLNISE_inputs[0]
        input_Vec[:,4]=Er
        #Feature 5 is the correlation time
        cortim=MLNISE_inputs[1]
        input_Vec[:,5]=cortim
        #Feature 6 is the time since the excitation currently inactaive
        #input_Vec[:,6]=t*dt
        input_Vec[:,7]=torch.exp(DE/kBT)
        #Note set of input features will change with ongoing development. Reorganizationenergy and correlation time can not be used if the different sites are coupled to baths with
        #different features

        #Apply the neural network parameters
        res1=self.fc1(input_Vec)
        res1=F.elu(res1)

        res2=self.fc2(res1)
        res2=F.elu(res2)


        res3=self.fc3(res2)
        res3=F.elu(res3)

        correction=self.fc4(res3)
        correction=F.elu(correction)+1
        correction=correction.squeeze()
        return correction