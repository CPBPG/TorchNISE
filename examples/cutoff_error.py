import mlnise
import numpy as np
import matplotlib.pyplot as plt
from mlnise.example_spectral_functions import spectral_Log_Normal_Lorentz,spectral_Lorentz,spectral_Drude
import functools
from mlnise.fft_noise_gen import noise_algorithm
import tqdm
from mlnise.sd_from_noise import SD_Reconstruct_FFT, get_auto, sd_reconstruct_superresolution
import torch

k = 8.6173303E-5 # in eV/K. 
T = 300 #Temperature in K
hbar = 0.658211951 #in eV fs
cm_to_eV=1.23984E-4
#k=k/cm_to_eV
save_name="data/noise.npy"
Sk = np.array([0.011, 0.011, 0.009, 0.009, 0.010, 0.011, 0.011, 0.012, 0.003, 0.008,
 0.008, 0.003, 0.006,  0.002, 0.002, 0.002,  0.001, 0.002, 0.004,  0.007, 
 0.004, 0.004, 0.003,  0.006, 0.004, 0.003, 0.007, 0.010, 0.005, 0.004, 
 0.009, 0.018,  0.007, 0.006, 0.007, 0.003, 0.004, 0.001,  0.001,  0.002, 
 0.002,  0.003, 0.001, 0.002, 0.002, 0.001, 0.001, 0.003, 0.003, 0.009, 0.007,
 0.010, 0.003, 0.005, 0.002, 0.004, 0.007, 0.002, 0.004, 0.002, 0.003, 0.003])
Wk = np.array([46, 68, 117, 167, 180, 191, 202, 243, 263, 284, 291, 327, 366, 385, 404, 423, 440, 
 481, 541, 568, 582, 597, 630, 638, 665, 684, 713, 726, 731, 750, 761, 770, 795, 821,
856, 891, 900, 924, 929, 946, 966, 984, 1004, 1037, 1058, 1094, 1104, 1123, 1130, 1162,
 1175, 1181, 1201, 1220, 1283, 1292, 1348, 1367, 1386, 1431, 1503, 1545])
Wk=Wk*cm_to_eV/hbar


total_time=100000
step=10
#number of realizations of the noise
reals = 1
#Temperature of the noise in K
T=300

S_HR=0.3
sigma=0.7
w_c=38*cm_to_eV/hbar
Gammak= 0.0009419458262008981
spectralfunc=functools.partial(spectral_Log_Normal_Lorentz,Wk=Wk,Sk=Sk,hbar=hbar,k=k,T=T,Gammak=Gammak,S_HR=S_HR,sigma=sigma,w_c=w_c)


N_cut=400_000//step #N/2
dt=step
dw_t = 2*np.pi/(2*N_cut*dt)
full_w_t = np.arange(0,N_cut*dw_t,dw_t)
ww=full_w_t #np.linspace(0,0.2,1000)/hbar
S=spectralfunc(ww)
SD=S/(2*np.pi*k*T)*ww
SD_FFT=SD/step*hbar*2*np.pi*k*T/ww
#SD_FFT=SD/4*np.pi*k*T/ww
SD_FFT[0]=0
reverse_SD_FFT=np.flip(SD_FFT[1:-1])
concat_SD_FFT=np.concatenate((SD_FFT,reverse_SD_FFT))
auto_theoretical=np.fft.ifft(concat_SD_FFT).real/hbar
auto_theoretical=auto_theoretical[0:N_cut]
plt.plot(auto_theoretical)
plt.show()
plt.close()
autos=[auto_theoretical]
for total_time in (100_000,1_000_000):
    Generated_Noise=noise_algorithm((reals,total_time//step), step,spectralfunc,save=True,save_name=f"noise_{total_time}.npy")
    auto=get_auto(Generated_Noise)
    autos.append(auto)


cutoffs=200*np.arange(10,100)#[5000]#

errors_auto={}
errors_J={}

for auto in autos[1:]:
    print(len(auto))
    for cutoff in cutoffs:
        print(cutoff)
        for damping in ["exp","gauss","step"]:       
            J_new, x_axis ,auto_damp = SD_Reconstruct_FFT(auto,step,T,hbar,k,damping_type=damping,cutoff=cutoff,rescale=False)
            S=spectralfunc(x_axis/hbar)
            SD=S/(2*np.pi*k*T)*x_axis/hbar
            errors_auto[f"{damping}_{cutoff}_{len(auto)}"]=np.mean(np.abs(auto_damp[0:2000]-auto_theoretical[0:2000]))/cm_to_eV/cm_to_eV
            errors_J[f"{damping}_{cutoff}_{len(auto)}"]=np.mean(np.abs(SD-J_new))/cm_to_eV
            plt.plot(x_axis,J_new,label=f"{damping}")
            print(f"{damping}: errors_auto {np.mean(np.abs(auto_damp[0:2000]-auto_theoretical[0:2000]))/cm_to_eV/cm_to_eV} error_J {np.mean(np.abs(SD-J_new))/cm_to_eV}")
        sample_frequencies=x_axis/hbar
        torch.cuda.empty_cache()
        J_new, x_axis ,auto_damp,J_new_debias,auto_debias=sd_reconstruct_superresolution(auto, dt, T, hbar, k, sparcity_penalty=0, l1_norm_penalty=0 ,
                                           solution_penalty=1e4,negative_penalty=1, ljnorm_penalty=1,j=1, lr=10, max_iter=1000, eta=1e-7, 
                                           tol=1e-7, device='cuda', cutoff=cutoff, 
                                           sample_frequencies=x_axis/hbar, top_n=False,top_tresh=1e-6, second_optimization=True,chunk_memory=5e8, auto_length=400_000)
        
        errors_auto[f"super_{cutoff}_{len(auto)}"]=np.mean(np.abs(auto_damp[0:2000]-auto_theoretical[0:2000]))/cm_to_eV/cm_to_eV
        errors_J[f"super_{cutoff}_{len(auto)}"]=np.mean(np.abs(SD-J_new))/cm_to_eV
        errors_auto[f"super_debias_{cutoff}_{len(auto)}"]=np.mean(np.abs(auto_debias[0:2000]-auto_theoretical[0:2000]))/cm_to_eV/cm_to_eV
        errors_J[f"super_debias_{cutoff}_{len(auto)}"]=np.mean(np.abs(SD-J_new_debias))/cm_to_eV
        print(f"super: errors_auto {np.mean(np.abs(auto_damp[0:2000]-auto_theoretical[0:2000]))/cm_to_eV/cm_to_eV} error_J {np.mean(np.abs(SD-J_new))/cm_to_eV}")
        print(f"super_debias: errors_auto {np.mean(np.abs(auto_debias[0:2000]-auto_theoretical[0:2000]))/cm_to_eV/cm_to_eV} error_J {np.mean(np.abs(SD-J_new_debias))/cm_to_eV}")
        plt.plot(x_axis,J_new,label="super")
        plt.plot(x_axis,SD,label="original")
        plt.legend()
        plt.show()
        plt.close()
    for damping in ["exp","gauss","step","super","super_debias"]:
        plot_error=[]
        plot_error_auto=[]
        for cutoff in cutoffs:
            plot_error.append(errors_J[f"{damping}_{cutoff}_{len(auto)}"])
            plot_error_auto.append(errors_auto[f"{damping}_{cutoff}_{len(auto)}"])
        np.save(f"{damping}_{len(auto)}.npy",np.array(plot_error))
        np.save(f"{damping}_{len(auto)}.npy",np.array(plot_error_auto))
        plt.plot(cutoffs,plot_error,label=damping)
    plt.legend()
    plt.show()
    plt.close()
    for damping in ["exp","gauss","step","super","super_debias"]:
        plot_error_auto=[]
        for cutoff in cutoffs:
            plot_error_auto.append(errors_auto[f"{damping}_{cutoff}_{len(auto)}"])
        plt.plot(cutoffs,plot_error_auto,label=damping)
    plt.legend()
    plt.show()
    plt.close()
    

       
for auto in autos:
    for damping in ["exp","gauss","step","super","super_debias"]:
        plot_error=[]
        for cutoff in cutoffs:
            plot_error.append(errors_J[f"{damping}_{cutoff}_{len(auto)}"])
        plt.plot(cutoffs,plot_error,label=damping)
    plt.legend()
    plt.show()
    plt.close()
        
        

            
    
    
    