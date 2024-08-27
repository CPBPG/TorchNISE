import numpy as np
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from mlnise.pytorch_utility import (
    batch_trace,
    create_empty_mmap_tensor,
    matrix_logh,
    renorm,
    tensor_to_mmap
    )


#constants
icm2ifs=2.99792458e-5 # cm/fs  conversion factor
kB=0.6950389 #1/cmK
hbar=5308.8459 #cm-1fs
def spectral_Drude_J(w,gamma,E_r): #single peak drude power spectrum. Recall S(w) =/ J(w)
        J = 2*gamma*E_r*w/(np.pi*(w**2+gamma**2))
        return J

def golden_section_search(func, a, b, tol):
    golden_ratio = (1 + 5 ** 0.5) / 2

    c = b - (b - a) / golden_ratio
    d = a + (b - a) / golden_ratio

    while abs(c - d) > tol:
        if func(c) < func(d):
            b = d
        else:
            a = c

        c = b - (b - a) / golden_ratio
        d = a + (b - a) / golden_ratio

    return (b + a) / 2

def blend_and_normalize_populations(pop1, pop2, lifetimes, delta_t):
    timesteps, sites = pop1.shape
    time_array = torch.arange(timesteps, dtype=torch.float) * delta_t

    blended_population = torch.zeros_like(pop1)

    for site in range(sites):
        # Exponential blending factor
        lifetime = lifetimes[site]
        blending_factor = torch.exp(-time_array / lifetime)

        # Blend populations
        blended_population[:, site] = blending_factor * pop1[:, site] + (1 - blending_factor) * pop2[:, site]

    # Renormalize the population at each timestep
    total_population = torch.sum(blended_population, dim=1, keepdim=True)
    normalized_population = blended_population / total_population

    return normalized_population

def estimate_lifetime(U, delta_t):
    # Number of states and time steps
    n = U.shape[2]
    timesteps = U.shape[1]
    time_array = torch.arange(timesteps, device=U.device, dtype=torch.float) * delta_t

    # Lifetime estimates
    lifetimes = torch.zeros(n, device=U.device, dtype=torch.float)

    # Objective function: MSE between population and exponential model
    def objective(tau, i):
        fit = (1 - 1/n) * torch.exp(-time_array/tau) + 1/n
        population = torch.mean(torch.abs(U[:,:,i,i])**2, dim=0).real
        reverse_cummax, _ = torch.flip(population, [0]).cummax(dim=0)

        # Reverse the cumulative maximum back to the original order
        reverse_cummax = torch.flip(reverse_cummax, [0])

        # Select values where the element is equal to its reverse cumulative maximum
        selected_values = population == reverse_cummax
        times=time_array[selected_values]
        weights=times[1:]-times[:-1]

        mse = torch.mean(((population[selected_values][1:] - fit[selected_values][1:])*weights) ** 2)
        return mse.item()

    # Estimate lifetime for each state
    for i in range(n):
        
        func = lambda tau: objective(tau, i)
        tolerance = 0.1 * delta_t
        tau_opt = golden_section_search(func, delta_t, 100 * timesteps * delta_t, tolerance)
        #plt.plot(torch.mean(torch.abs(U[:,:,i,i])**2, dim=0).real,label="NISE")
        #plt.plot((1 - 1/n) * torch.exp(-time_array/tau_opt) + 1/n,label="fit")
        #plt.legend()
        #plt.show()
        lifetimes[i] = tau_opt
        print(f"lifetime state {i+1} {tau_opt} fs")
    return lifetimes


def averaging(population,averiging_type,lifetimes=None,coherence=None):
    averiging_types= ["original", "boltzmann", "blend"]
    
    if averiging_type.lower() not in averiging_types:
        raise NotImplementedError(f"{averiging_type} not implemented only {averiging_types} are available")
    if averiging_type.lower() in ["blend","boltzmann"]:
        assert not coherence == None
        if averiging_type.lower()=="blend":
            assert not lifetimes==None
    
    

#The way Pytorch is build we need a Class for your ML model. This class does not only require the Neural Network but the
#entire machine learning model. In our case that is the Full NISE method.
class MLNISE(nn.Module):
    #init contains all the free parameters of our method. In case of the neural networks its the layers of the neural networks
    def __init__(self):
        super(MLNISE, self).__init__()


        self.fc1 = nn.Linear(8, 75)
        self.fc2 = nn.Linear(75, 75)
        self.fc3 = nn.Linear(75, 75)
        self.fc4 = nn.Linear(75, 1)

        # we found that reducing the initial weights slightly increased stability during training.
        with torch.no_grad():
            self.fc1.weight/=10
            self.fc2.weight/=10
            self.fc3.weight/=10
            self.fc4.weight/=10

    # This is the method that we want to train. As input come the system Parameters and as output the Population dynamics
    # To work well with pytorch this is setup to take a whole batch of input parameters and return a whole batch of population dynamics
    # For each set of input paramers it calculates only 1 realization
    def forward(self,T,Er,cortim,total_time,dt,reps, psi0,Hfull,device="cpu",T_correction="ML",saveCoherence=False,explicitTemp="None",saveU=False,memory_mapped=False,save_Interval=10):
        if device=="cuda":
            torch.backends.cuda.preferred_linalg_library(backend="magma")
        corrections=["ml","jansen","none"]
        if not T_correction.lower() in corrections:
            raise NotImplementedError(f"Correction {T_correction} not implemented, available are {corrections}")



        N=Hfull.size()[2] #Number of Sites

        if memory_mapped:
            Hfull=tensor_to_mmap(Hfull)

        batchsize=reps.size()[0] #Batchsize


        factor=1j* icm2ifs*2*np.pi*dt.unsqueeze(1)# for the time evolution operator
        kBT=T*kB #cm-1





        #Create an empy tensor of size (batches, total timesteps, number of sites) for the storage of the population dynamics


        if memory_mapped:
            PSLOC=create_empty_mmap_tensor((batchsize,int(total_time[0]/dt[0]/save_Interval)+1,N))
        else:
            PSLOC=torch.zeros(batchsize,int(total_time[0]/dt[0]/save_Interval)+1,N,device="cpu")







        if saveCoherence:

            if memory_mapped:
                CohLoc=create_empty_mmap_tensor((batchsize,int(total_time[0]/dt[0]/save_Interval)+1,N,N),dtype=torch.complex64)
            else:
                CohLoc=torch.zeros(batchsize,int(total_time[0]/dt[0]/save_Interval)+1,N,N,device="cpu",dtype=torch.complex64)

        H=Hfull[:,0,:,:].clone().to(device=device) #grab the 0th timestep [all batches : 0th timestep  : all sites : all sites]

        E,C=torch.linalg.eigh(H) #get initial eigenenergies and transition matrix from eigen to site basis.
        # Since the hamiltonian is symetric we van use symeig
        # H contains a whole batch of hamiltonians. To our advantage The symeig torch function (like almost all torch functions)
        # is setup so that it can efficently calculate the results for a whole batch of Hamiltonians

        Cold=C
        Eold=E
        PSLOC[:,0,:]=(psi0[:,:,0]**2).cpu(); #Save the population of the first timestep
        phiB=Cold.transpose(1,2).to(dtype=torch.complex64).bmm(psi0.to(dtype=torch.complex64)) #Use the transition Matrix to transfer to the eigenbasis. Bmm is a batched matrix multiplication, so it does the matrix multiplication for all batches at once

        #phiB#=MakeComplex(phiB) #We need complex numbers. We used our own implementation of complex numbers since Pytorch did not suport complex numbers at the time development of this project started
        if saveU:

            if memory_mapped:
                ULOC=create_empty_mmap_tensor((batchsize,int(total_time[0]/dt[0]/save_Interval)+1,N,N),dtype=torch.complex64)
            else:
                ULOC=torch.zeros(batchsize,int(total_time[0]/dt[0]/save_Interval)+1,N,N,device="cpu",dtype=torch.complex64)
            identiy=torch.eye(N,dtype=torch.complex64)
            identiy=identiy.reshape(1,N,N)
            ULOC[:,0,:,:]= identiy.repeat(batchsize,1,1)
            UB=Cold.transpose(1,2).cpu().to(dtype=torch.complex64).bmm(ULOC[:,0,:,:])

            UB=UB.to(dtype=torch.complex64).to(device=device) #=MakeComplex(UB)

        #Now we get to the step by step timepropagation. We start at 1 and not 0 since we have already filled the first slot of our population dynamics
        for t in tqdm.tqdm(range(1,int(total_time[0]/dt[0])+1)):
            H=Hfull[:,t,:,:].clone().to(device=device)   #grab the t'th timestep [all batches : t'th timestep  : all sites : all sites]
            E,v_eps=torch.linalg.eigh(H)
            C=v_eps
            S=torch.matmul(C.transpose(1,2),Cold) #multiply the old with the new transition matrix to get the non-adiabatic coupling

            if T_correction.lower() in ["ml","jansen"]:
                for ii in range(0,N):
                    maxC=torch.max(S[:,:,ii].real**2,1) #The order of the eigenstates is not well defined and might be flipped from one transition matrix to the transition matrix
                    #to find the eigenstate that matches the previous eigenstate we find the eigenvectors that overlapp the most and we use the index with the highest overlap (kk)
                    #instead of the original index ii to index the non adiabatic coupling correctly
                    kk=maxC.indices
                    CD=torch.zeros(batchsize,device=device)
                    for jj in range(0,N):
                        DE=E[:,jj]-Eold[:,ii]
                        DE[jj==kk]=0 #if they are the same state the energy difference is 0
                        if T_correction=="ML":
                            #Initialize the input Vector for the neural network
                            input_Vec=torch.zeros(batchsize,8,device=device)

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
                            if explicitTemp=="None":
                                input_Vec[:,3]=kBT
                            elif explicitTemp in ["Added","Ignore"]:
                                input_Vec[:,3]=300*kB
                            #Feature 4 is the reorganization energy
                            input_Vec[:,4]=Er
                            #Feature 5 is the correlation time
                            input_Vec[:,5]=cortim
                            #Feature 6 is the time since the excitation
                            #input_Vec[:,6]=t*dt
                            if explicitTemp in ["None","Ignore"]:
                                input_Vec[:,7]=torch.exp(DE/kBT)
                            elif explicitTemp in ["Added"]:
                                input_Vec[:,7]=torch.exp(DE/(300*kB))
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
                            if explicitTemp in ["Added","Trained"]:
                                correction=correction*torch.exp(-DE/kBT/4)
                                if explicitTemp == "Added":
                                    correction=correction/torch.exp(-DE/(300*kB)/4)
                        else: # Jansen correction
                            correction=torch.exp(-DE/kBT/4)

                        S[:,jj,ii]=S[:,jj,ii].clone()*correction
                        AddCD=S[:,jj,ii].clone()**2
                        AddCD[jj==kk]=0
                        CD=CD.clone()+AddCD
                    #The renormalization procedure broken into smaller steps, could probably be simplified
                    norm=torch.abs(S[torch.arange(batchsize),kk,ii].clone())
                    dummy1=torch.ones(batchsize,device=device)
                    dummy1[1-CD>0]=torch.sqrt(1-CD[1-CD>0])
                    dummy2=torch.zeros(batchsize,device=device)
                    S_clone=S[torch.arange(batchsize),kk,ii].clone()
                    dummy2[1-CD>0]=S_clone.clone()[1-CD>0]/norm[1-CD>0]
                    dummy3=dummy1*dummy2

                    S[1-CD>0,kk[1-CD>0],ii]=dummy3[1-CD>0]

            U=torch.diag_embed(torch.exp(-E[:,:]*factor)).bmm(S.to(dtype=torch.complex64))  #Make the Time Evolution operator

            #print(torch.sum(phiB.abs()[0,:,0]-UB.abs()[0,:,0]))
            phiB=U.bmm(phiB) #Apply the time evolution operator

            if saveU:
                UB=U.bmm(UB)
            #print(torch.sum(phiB.abs()[0,:,0]-UB.abs()[0,:,0]))
            phiB=renorm(phiB,dim=1) #Renormalize

            Cold=C #Set the new variables to the old variables for the next step
            Eold=E #

            EigenTEnsor=C.to(dtype=torch.complex64)  #Make the transition matrix complex

            PhiBinLocBase=EigenTEnsor.bmm(phiB) #Tranision to the site basis
            if t%save_Interval==0:
                PSLOC[:,t//save_Interval,:]=((PhiBinLocBase.abs()**2)[:,:,0]).cpu() #Save the result in the site Basis
                if saveU:
                    #UB_real_orig=UB.real.clone()
                    #UB_imag_orig=UB.imag.clone()
                    for i in range(0,N):
                        UB_Norm_Row=renorm(UB[:,:,i],dim=1)
                        UB[:,:,i]=UB_Norm_Row[:,:]
                        #UB.real[:,:,i]=UB_Norm_Row.real[:,:,0]
                        #UB.imag[:,:,i]=UB_Norm_Row.imag[:,:,0]
                    ULOC[:,t//save_Interval,:,:]=(EigenTEnsor.bmm(UB)).cpu()
                    #ULOC_real[:,t,:,:]=ULOC.real
                    #ULOC_imag[:,t,:,:]=ULOC.imag
                if saveCoherence:
                    CohLoc[:,t//save_Interval,:,:]= PhiBinLocBase.squeeze(-1)[:, :, None] * PhiBinLocBase.squeeze(-1)[:, None, :].conj()

                    #for i in range(0,N):
                    #    for j in range(0,N):
                    #        ac = PhiBinLocBase.real[:,i,0]*PhiBinLocBase.real[:,j,0]
                    #        bd = PhiBinLocBase.imag[:,i,0]*(-1*PhiBinLocBase.imag[:,j,0])
                    #        ad = PhiBinLocBase.real[:,i,0]*(-1*PhiBinLocBase.imag[:,j,0])
                    #        bc = PhiBinLocBase.imag[:,i,0]*PhiBinLocBase.real[:,j,0]
                            #real=ac - bd
                            #imag=ad + bc
                    #        CohLoc.real[:,t,i,j] = ac - bd
                    #        CohLoc.imag[:,t,i,j] = ad + bc
                            #CohLoc.real[:,t,i,j]=ComplexTensor(real,imag).abs()
        if not saveCoherence:
            CohLoc = None
        if not saveU:
            ULOC = None
        return PSLOC, CohLoc ,ULOC

    #The default forward method calculates one realization for every set of input parameters given
    #Therefore we need a simulation mode where one set of input parameters is used and run and avaraged over many realizations
    #This is not used for training but only for actually using the method
    def simulate(self, dw0, V,T,Er,cortim,total_time,step,reps,initially_excited_site,Hfull,device="cpu",T_correction="ML",saveCoherence=False,explicitTemp="None",saveU=False,memory_mapped=False,save_Interval=10):
        #Since this is not for training we don't need to keep track of the gradients
        with torch.no_grad():

            N=np.size(Hfull,3) # get the number of sites from the size of the hamiltonian


            totalres=0


            #Basically Create a batch of input parameters that are all the same
            dw0=torch.zeros(reps)+dw0
            V=torch.zeros(reps)+V
            T=torch.zeros(reps)+T
            Er=torch.zeros(reps)+Er#sig_square/(2*kBT)
            cortim=torch.zeros(reps)+cortim
            total_time=torch.zeros(reps)+total_time
            step=torch.zeros(reps)+step
            psi0=torch.zeros(reps,N,1)
            psi0[:,initially_excited_site,:]=1
            inputreps=torch.ones(reps)
            lifetimes=None

                ####run NISE without T correction
            if T_correction in ["Jansen","ML"]:
                res, coherence ,U= self.forward( T.to(device), Er.to(device), cortim.to(device), total_time.to(device), step.to(device), inputreps.to(device), psi0.to(device), Hfull,device,"None",True,explicitTemp,True,memory_mapped=memory_mapped,save_Interval=save_Interval)
                lifetimes=estimate_lifetime(U, step[0]*save_Interval)
                lifetimes=lifetimes*5
                print(lifetimes)



            res, coherence ,U = self.forward( T.to(device), Er.to(device), cortim.to(device), total_time.to(device), step.to(device), inputreps.to(device), psi0.to(device), Hfull,device,T_correction,True,explicitTemp,saveU,memory_mapped=memory_mapped,save_Interval=save_Interval)
            # Now the return will be a batch of population dynamics and we only need to take the average of those



            totalres+=res


            meanres_orig=torch.mean(res,dim=0)
            meanres_arr=torch.zeros_like(res)
            if T_correction in ["Jansen","ML"]:
                logres_Matrix=torch.mean(matrix_logh(coherence),dim=0)
                meanExp_Matrix=torch.linalg.matrix_exp(logres_Matrix)
                #renormalize
                meanres_Matrix=meanExp_Matrix/batch_trace(meanExp_Matrix) 
                meanres_orig_Matrix=torch.mean(coherence,dim=0)#(coherence.to_torch_complex(),dim=0) #torch.mean(coherence.to_torch_complex(),dim=0) #


            for i in range(0,res.shape[0]):
                meanres_arr[i,:,:]=meanres_orig

            MSE=F.mse_loss(meanres_arr,res) #the internal MSE of a single realization wiht respect to the average

            if T_correction in ["Jansen","ML"]:
                lifetimes=lifetimes*5
                print(lifetimes)
                mytime=torch.arange(0,total_time[0]+step[0]*save_Interval,step[0]*save_Interval)
                expratio=torch.exp(-mytime/lifetimes[0]).unsqueeze(1).unsqueeze(1)
                
                
                newresH=meanres_Matrix.clone()

                log_meanres_orig_Matrix=matrix_logh(meanres_orig_Matrix)
                log_meanres_Matrix=matrix_logh(meanres_Matrix)
                
                newresH[1:,:,:]=log_meanres_orig_Matrix[1:,:,:]*expratio[1:,:,:]+log_meanres_Matrix[1:,:,:]*(1-expratio[1:,:,:])

                newresH_exp=torch.linalg.matrix_exp(newresH)

                newres_Matrix=torch.clone(meanres_orig)
                for i in range(0,N):
                    newres_Matrix[1:,i]=newresH_exp[1:,i,i]/batch_trace(newresH_exp[1:,:,:])


                population_blend=blend_and_normalize_populations(meanres_orig,torch.diagonal(meanres_Matrix, dim1=1, dim2=2),lifetimes,step[0])



                result=population_blend
                old_res=meanres_orig
                matrix_ave=newres_Matrix


            else:
                result=meanres_orig
                old_res=None
                matrix_ave=None
            if not saveCoherence:
                meancoherence=None
            else:
                meancoherence=torch.mean(coherence.real,dim=0).abs()
            return result,MSE, meancoherence,U, old_res, matrix_ave,lifetimes