"""
This file implements various Running MLNISE for different options from command line.
"""
import argparse
import functools

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import correlate

from mlnise.example_spectral_functions import spectral_Drude
from mlnise.fft_noise_gen import noise_algorithm
from mlnise.mlnise_model import MLNISE,averaging
from mlnise.pytorch_utility import create_empty_mmap_tensor, tensor_to_mmap

k = 8.6173303E-5 # in eV/K.
hbar = 0.658211951 #in eV fs
cm_to_eV=1.23984E-4
k=k/cm_to_eV

def Absorption_time_domain(U,mu):
    pigments=U.shape[-1]
    realizations=U.shape[0]
    timesteps=U.shape[1]

    if len(mu.shape)==2: ## time dependence is not supplied
        mu=np.tile(mu,(realizations,timesteps,1,1)) ##copy the same mu vector along the time domain

    absorption_time_domain=0
    damp=1#np.exp(-np.arange(0,timesteps)/1000)
    for xyz in range(0,3):
        for real in range(0,realizations):
            for m in range (0,pigments):
                for n in range(0,pigments):

                    absorption_time_domain+=U[real,:,m,n]*mu[real,:,m,xyz]*mu[real,0,n,xyz]/realizations*damp
    return absorption_time_domain

def smooth_Damp_to_zero(f_init,start,end):
    f=f_init.copy()
    f[end:]=0
    def expdamp_helper(a):
        x=a.copy()
        x[x<=0]=0
        x[x>0]=np.exp(-1/x[x>0])
        return x
    damprange=np.arange(end-start,dtype=float)[::-1]/(end-start)
    f[start:end]=f[start:end]*expdamp_helper(damprange)/(expdamp_helper(damprange)+expdamp_helper(1-damprange))
    return f

def absorb_time_to_freq(absorb_time,pad,total_time,dt):
    absorb=np.pad(absorb_time,(0,pad))
    absorb=smooth_Damp_to_zero(absorb,int(total_time//dt-total_time//(dt*10)),int(total_time/dt)-1)
    absorb_f=np.fft.fftshift(np.fft.fft(absorb))
    hbar = 0.658211951 #ev fs
    hbar_cm= 8065.54*hbar
    x_axis=-hbar_cm*2*np.pi*np.fft.fftshift(np.fft.fftfreq(int((total_time+dt)/dt)+pad, d=dt))
    absorb_f=(absorb_f.real-absorb_f.real[0])/np.max(absorb_f.real-absorb_f.real[0])
    return absorb_f,x_axis


def gen_noise(spectral_funcs,dt,shape,memory_mapped=False):
    if len(shape)!=3:
        raise ValueError(f"gen_noise requres a shape tuple with (reals,steps,n_sites) but a tuple of size {len(shape)} was given")
    reals,steps,n_sites = shape
    if len(spectral_funcs)==1:
        if memory_mapped:
            noise= create_empty_mmap_tensor(shape)
        else:
            noise=torch.zeros(shape)
        for i in range(shape[-1]):
            noise[:,:,i]= torch.tensor(noise_algorithm((reals,steps),dt,spectral_funcs[0],axis=1))
        return noise
    elif len(spectral_funcs)==n_sites:
        if memory_mapped:
            noise= create_empty_mmap_tensor(shape)
        else:
            noise=torch.zeros(shape)
        for i in range(shape[-1]):
            noise[:,:,i]= torch.tensor(noise_algorithm((reals,steps),dt,spectral_funcs[i],axis=1))
        return noise
    else:
        raise ValueError(f"len of spectral_funcs was {len(spectral_funcs)}, but must either be 1 or match number of sites ({n_sites})")


def RunNiseOptions(model, reals, H_0, tau, Er, T, dt, initiallyExcitedState, total_time, spectral_funcs, trajectory_time=None, T_correction='None', maxreps=1000000, use_filter=False, filter_order=10, filter_cutoff=0.1, mode="population",mu=None,device="cpu",memory_mapped=False,save_Interval=10):
    hbar = 0.658211951

    total_steps = int((total_time + dt) / dt)
    save_steps  = int((total_time + dt*save_Interval) / (dt*save_Interval))
    n_state = H_0.shape[-1] #H_0.shape[1] if time_dependent_H else H_0.shape[0]
    window=1
    time_dependent_H= len(H_0.shape)>=3
    avg_absorb_time = None
    x_axis = None
    absorb_f = None
    avg_oldave = None
    avg_newave = None
    avg_output = None


    if time_dependent_H:
        if memory_mapped:
            H_0= tensor_to_mmap(H_0)
        trajectory_steps = H_0.shape[0]
        if reals > 1:
            window = int((trajectory_steps - total_steps) / (reals - 1))
            print("window is", window * dt, "fs")

    def generate_Hfull_chunk(chunk_size, start_index=0, window=0):
        if memory_mapped:
            chunk_Hfull = create_empty_mmap_tensor((chunk_size,total_steps,n_state,n_state), dtype=torch.float32)
        else:
            chunk_Hfull = torch.zeros((chunk_size, total_steps, n_state, n_state))
        if time_dependent_H:
            for j in range(chunk_size):
                H_index = start_index + j
                chunk_Hfull[j, :, :, :] = torch.tensor(H_0[window * H_index:window * H_index + total_steps, :, :])
        else:
            print("generating noise")
            mynoise = gen_noise(spectral_funcs,dt,(chunk_size,total_steps,n_state),memory_mapped=memory_mapped)
            #if memory_mapped:
            #    mynoise = tensor_to_mmap(torch.tensor(gen_noise(spectral_funcs,dt,(chunk_size,total_steps,n_state),memory_mapped=memory_mapped)))
            #else:
            #    mynoise = torch.tensor(gen_noise(spectral_funcs,dt,(chunk_size,total_steps,n_state)))
            print("building H")
            chunk_Hfull[:] = H_0
            for i in range(n_state):
                #print(f"adding noise state {i}")
                """print(mynoise.shape)
                print(mynoise[0, 0:10, i])
                plt.plot(mynoise[0, :, i])
                plt.show()
                plt.close()"""
                chunk_Hfull[:, :, i, i] += mynoise[:, :, i]
                """
                plt.plot(chunk_Hfull[0, :, i, i])
                plt.show()
                plt.close()"""

            if use_filter:
                filter_cutoff_omega = filter_cutoff / hbar
                filter_cutoff_Hz = filter_cutoff_omega * 10**15 / (2 * np.pi)
                sos = signal.butter(filter_order, filter_cutoff_Hz, 'lp', fs=dt * 10**15, output='sos')
                for i in range(n_state):
                    filtered_noise = signal.sosfilt(sos, chunk_Hfull[:, :, i, i] - torch.mean(chunk_Hfull[:, :, i, i]))
                    chunk_Hfull[:, :, i, i] = torch.tensor(filtered_noise) + torch.mean(chunk_Hfull[:, :, i, i])

        return chunk_Hfull

    psi0 = initiallyExcitedState
    if reals > maxreps:
        num_chunks = (reals + maxreps - 1) // maxreps  # This ensures rounding up
        print("splitting calculation into ", num_chunks, " chunks")
    else:
        num_chunks=1
    chunk_size = (reals + num_chunks - 1) // num_chunks  # This ensures each chunk is not greater than reals

    if mode=="population" and T_correction in ["ML","Jansen"]:
        all_coherence = torch.zeros(num_chunks,save_steps,n_state,n_state)
        all_lifetimes = torch.zeros(num_chunks,n_state)
    elif mode =="absorption":
        #all_meancoherence=[]
        all_absorb_time=[]
    saveU = mode =="absorption"#True #

    weights = []
    all_output = torch.zeros(num_chunks,save_steps,n_state)
    for i in range(0, reals, chunk_size):
        chunk_reps = min(chunk_size, reals - i)
        weights.append(chunk_reps)
        chunk_Hfull = generate_Hfull_chunk(chunk_reps, start_index=i,window=window)

        print(chunk_Hfull.shape)
        print("running calculation")
        result,MSE, meancoherence,coherence_ave,U, old_res, matrix_ave, lifetimes = model.simulate(0, 0, T, Er, tau, total_time, dt, chunk_reps, psi0, chunk_Hfull, device=device, T_correction=T_correction,saveU= saveU,memory_mapped=memory_mapped,save_Interval=save_Interval,saveCoherence=True)
        if saveU:
            torch.save(U,"U.pt")
        if mode=="population" and T_correction in ["ML","Jansen"]:
            all_coherence[i//chunk_size,:,:,:]= coherence_ave
            all_lifetimes[i//chunk_size,:] = lifetimes
        elif mode =="absorption":
            absorb_time=Absorption_time_domain(U,mu)
            #all_meancoherence.append(meancoherence)
            all_absorb_time.append(absorb_time)
        all_output[i//chunk_size,:,:]=old_res
    if mode=="population" and T_correction in ["ML","Jansen"]:
        lifetimes=torch.mean(all_lifetimes,dim=0)
        print(f"lifetimes are {lifetimes}")
        avg_boltzmann,_ = averaging(all_output,"boltzmann",lifetimes=lifetimes,step=dt,coherence=all_coherence,weight=torch.tensor(weights,dtype=torch.float))#np.average(all_oldave, axis=0, weights=weights)
        avg_blend,_ = averaging(all_output,"blend",lifetimes=lifetimes,step=dt,coherence=all_coherence,weight=torch.tensor(weights,dtype=torch.float))
    else:
        avg_blend=None
        avg_boltzmann=None
        lifetimes=None
    
    if mode =="absorption":
        #avg_meancoherence = np.average(all_meancoherence, axis=0, weights=weights)
        avg_absorb_time = np.average(all_absorb_time, axis=0, weights=weights)
    avg_output = np.average(all_output, axis=0, weights=weights)
    


    if mode=="absorption":
        pad=int(10000/dt)
        absorb_f, x_axis = absorb_time_to_freq(avg_absorb_time,pad,total_time,dt)


    return avg_output, avg_blend, avg_boltzmann,avg_absorb_time,x_axis, absorb_f

# Original x and y data
def PadJ(J,pad):
    # Calculate the step size of the original x data
    step = np.mean(J[1:,0] - J[0:-1,0])

    # Create new x-axis
    x_padded = np.arange(0, J[-1,0]*(1+pad/J.shape[0]), step)
    x_padded_flip =np.flip( -x_padded[1:])
    x_padded_total =np.concatenate((x_padded_flip, x_padded))
    # Create zero-padded y data
    y_padded = np.zeros(len(x_padded))
    y_padded[:J.shape[0]] = J[:,1] # Insert the original data in the middle
    y_padded_flip =np.flip( -y_padded[1:])
    y_padded_total =np.concatenate((y_padded_flip, y_padded))
    # Optionally, save the new data or plot it
    J_padded= np.column_stack((x_padded_total, y_padded_total))
    return J_padded
def load_pigments(file_path):
    pigments = []

    with open(file_path, 'r') as file:
        # Skip the header line
        next(file)

        for line in file:
            # Split the line by whitespace
            parts = line.strip().split()

            # Extract complex, pigment, and number
            #complex_pigment_number = parts[1].split('_')
            LH_complex = parts[0]
            pigment = parts[1]
            number = parts[2]

            # Create a dictionary and append to the list
            pigment_dict = {
                "Complex": LH_complex,
                "Type": pigment,
                "Number": int(number)
            }
            pigments.append(pigment_dict)

    return pigments
def replace_nan_with_neighbor_mean(arr, axis):
    """
    Replaces NaN values with the mean of their neighboring values along the specified axis.

    Parameters:
    arr (numpy.ndarray): Input array with NaN values.
    axis (int): Axis along which to replace NaN values.

    Returns:
    numpy.ndarray: Array with NaN values replaced.
    """
    arr = np.asarray(arr)
    nan_indices = np.where(np.isnan(arr))

    # Iterate through all NaN indices
    for idx in zip(*nan_indices):
        idx = list(idx)  # Convert to list to allow modification
        if idx[axis] > 0 and idx[axis] < arr.shape[axis] - 1:
            idx_left = idx.copy()
            idx_left[axis] -= 1
            idx_right = idx.copy()
            idx_right[axis] += 1

            left_neighbor = arr[tuple(idx_left)]
            right_neighbor = arr[tuple(idx_right)]

            if not np.isnan(left_neighbor) and not np.isnan(right_neighbor):
                arr[tuple(idx)] = (left_neighbor + right_neighbor) / 2
            elif np.isnan(left_neighbor) and not np.isnan(right_neighbor):
                arr[tuple(idx)] = right_neighbor
            elif not np.isnan(left_neighbor) and np.isnan(right_neighbor):
                arr[tuple(idx)] = left_neighbor
        elif idx[axis] == 0:
            idx_right = idx.copy()
            idx_right[axis] += 1
            arr[tuple(idx)] = arr[tuple(idx_right)]
        elif idx[axis] == arr.shape[axis] - 1:
            idx_left = idx.copy()
            idx_left[axis] -= 1
            arr[tuple(idx)] = arr[tuple(idx_left)]

    return arr
def make_positive_spectral_func(spectral_func):
    def positive_spectral_func(x):
        return np.clip(spectral_func(x), a_min=0, a_max=None)
    return positive_spectral_func

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MLNise')

    parser.add_argument('--reals', type=int, default=1000, metavar='N',
                        help='number of realizations for testing')
    parser.add_argument('--total_time', type=float, default=1000, metavar='N',
                        help='length of population dynamics in fs')
    parser.add_argument('--Hamiltonian', type=str, default="H.npy", metavar='N',
                        help='Hamiltonian')
    parser.add_argument('--dt', type=float, default=1, metavar='N',
                        help='timestep in fs')
    parser.add_argument('--T', type=float, default=300, metavar='N',
                        help='Temperature in K')
    parser.add_argument('--tau', type=float, default=1, metavar='N',
                    help='correlation time in fs')
    parser.add_argument('--Er', type=float, default=1, metavar='N',
                    help='reorganization energy in cm-1')
    parser.add_argument('--initial_state', type=int, default=0, metavar='N',
                        help='initially excited state (counting starts with 0)')
    parser.add_argument('--Net', type=str, default="None", metavar='N',
                        help='name of the state dict file to load')
    parser.add_argument('--Correction', type=str, default="None", metavar='N',
                        help='T-correction to use Jansen, ML or None')
    parser.add_argument('--J', type=str, default="J.npy", metavar='N',
                        help='Spectral Density file')
    parser.add_argument('--pad', type=int, default=10000, metavar='N',
                        help='padding the spectral density')

    args = parser.parse_args()
    H=np.array([[100,100],[100,0]],dtype=np.float32)#   np.load(args.Hamiltonian)#np.load("Pytorch HEOM/HEOM/hamil.npy")#
    #H=np.loadtxt("data/superComplex/Hamiltonian_sup_new_E.dat")
    #H=H/cm_to_eV
    n_state=H.shape[0]
    AveE=np.mean(H[np.eye(n_state,dtype=bool)])
    H[np.eye(n_state,dtype=bool)]-=AveE

    T=args.T
    dt=args.dt
    tau=args.tau
    Er=args.Er
    initiallyExcitedState=args.initial_state
    total_time=args.total_time
    trajectory_time=600000
    reals=args.reals
    myNet=args.Net
    correction=args.Correction


    gamma=1/100
    reals=1000
    maxreps=1000
    E_reorg=100
    total_time=1000
    """
    pigments=load_pigments("data/superComplex/Pigment_naming.dat")


    chlb_ind=[1,5,6,7,8]
    chla_ind=[2,3,4,9,10,11]
    J_b=0
    for i in chlb_ind:
        J_b+=np.loadtxt(f"data/superComplex/SPD/J_av_{i}.dat")
    J_b=J_b/len(chlb_ind)
    plt.plot(J_b[:,0],J_b[:,1])
    plt.show()
    plt.close()
    pad=J_b.shape[0]*10
    J_padded_b=PadJ(J_b,pad)
    S_padded_b=J_padded_b
    #J=S/(2*np.pi*k*T)*ww
    S_padded_b[:,0]=S_padded_b[:,0]/hbar
    #S_padded_b[:,0][S_padded_b[:,0]==0]=1e-6
    S_padded_b[:,1]=S_padded_b[:,1]/cm_to_eV*2*np.pi*k*T/S_padded_b[:,0]
    replace_nan_with_neighbor_mean(S_padded_b,axis=0)
    #S_padded_b[np.isnan(S_padded_b)]=0
    plt.plot(S_padded_b[:,0],S_padded_b[:,1])
    plt.show()
    plt.close()
    spectral_func_b=interp1d(S_padded_b[:, 0], S_padded_b[:, 1], kind="cubic", assume_sorted=False,fill_value=0)
    spectral_func_b=make_positive_spectral_func(spectral_func_b)

    J_a=0
    for i in chla_ind:
        J_a+=np.loadtxt(f"data/superComplex/SPD/J_av_{i}.dat")
    J_a=J_a/len(chla_ind)
    plt.plot(J_a[:,0],J_a[:,1])
    plt.show()
    plt.close()
    pad=J_a.shape[0]*10
    J_padded_a=PadJ(J_a,pad)
    S_padded_a=J_padded_a
    #J=S/(2*np.pi*k*T)*ww
    S_padded_a[:,0]=S_padded_a[:,0]/hbar
    #S_padded_a[:,0][S_padded_a[:,0]==0]=1e-6
    S_padded_a[:,1]=S_padded_a[:,1]/cm_to_eV*2*np.pi*k*T/S_padded_a[:,0]
    replace_nan_with_neighbor_mean(S_padded_a,axis=0)
    #S_padded_a[np.isnan(S_padded_a)]=0
    plt.plot(S_padded_a[:,0],S_padded_a[:,1])
    plt.show()
    plt.close()
    spectral_func_a=interp1d(S_padded_a[:, 0], S_padded_a[:, 1], kind="cubic", assume_sorted=False,fill_value=0)
    spectral_func_a=make_positive_spectral_func(spectral_func_a)
    midpoint=len(S_padded_a[:,0])//2
    print(midpoint)
    print(S_padded_a[midpoint-5:midpoint+5, 1])
    print(spectral_func_a(S_padded_a[midpoint-5:midpoint+5, 0]))
    spectral_funcs=[]
    for i,pigment in enumerate(pigments):
        if pigment["Type"]=="CLA":
            spectral_funcs.append(spectral_func_a)
        elif pigment["Type"] =="CHL":
            spectral_funcs.append(spectral_func_b)
        else:
            raise NotImplementedError(f"Type {pigment['Type']} not available")
    """

    spectral_func=functools.partial(spectral_Drude,gamma=gamma,strength=E_reorg,k=k,T=T)
    device="cpu"
    memory_mapped=True

    model = MLNISE()
    if correction=="ML":
        model.load_state_dict(torch.load(myNet))
    #model.to("cuda")
    correction="None"
    H_0=torch.tensor(H)


    spectral_funcs=[spectral_func]


    ww=np.linspace(-1,1,500)/hbar
    """S=spectral_func(S_padded_a[:,0])
    #SD=S/(2*np.pi*k*T)*S_padded_a[:,0]
    #plt.plot(SD_heom[:,0],SD_heom[:,1],label="heom code")

    plt.plot(ww,spectral_func(ww),label="S_Drude")
    plt.plot(S_padded_a[:,0],S_padded_a[:,1],label="S_a_file")
    plt.plot(S_padded_a[:,0],spectral_func_a(S_padded_a[:,0]),label="S_a")
    plt.xlim(-0.4,0.4)
    plt.legend()
    plt.show()
    plt.close()


    ww=np.linspace(-0.2,0.2,500)/hbar
    S=spectral_func_a(ww)
    SD=S/(2*np.pi*k*T)*ww

    #plt.plot(SD_heom[:,0],SD_heom[:,1],label="heom code")

    plt.plot(ww*hbar,SD/cm_to_eV,label="SD_a_func")
    plt.plot(J_a[:,0]*cm_to_eV,J_a[:,1],label="SD_a_file")
    plt.xlim(-0.2,0.2)
    plt.legend()
    plt.show()
    plt.close()"""
    save_Interval=1
    avg_output, avg_oldave, avg_newave,avg_absorb_time,x_axis, absorb_f = RunNiseOptions(model, reals, H_0,tau,Er, T, dt, initiallyExcitedState, total_time, spectral_funcs, trajectory_time=None, T_correction=correction, maxreps=maxreps, use_filter=False, filter_order=10, filter_cutoff=0.1, mode="population",mu=None,device=device,memory_mapped=memory_mapped,save_Interval=save_Interval)
    #print(avg_output[0,:])
    Complexes={}


    """for i in range(n_state):
        if pigments[i]["Complex"] not in Complexes:
            Complexes[pigments[i]["Complex"]] = [avg_output[:,i]]
        else:
            Complexes[pigments[i]["Complex"]].append(avg_output[:,i])
    #print (Complexes)
    for lh_complex_name,pops in Complexes.items():
        pop_lh=0
        for pop in pops:
            pop_lh+= pop
        plt.plot(np.linspace(0,total_time/1000,len(pop_lh)),pop_lh,label=lh_complex_name)
    plt.xlabel("time [ps]")
    plt.ylabel("population")
    plt.legend()
    plt.show()
    plt.close()"""


    plt.plot(avg_output[:,0])
    plt.show()
    #print(avg_output[0:100,0])
