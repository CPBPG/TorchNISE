import numpy as np
import matplotlib.pyplot as plt
import torch
import functools
from torchnise.spectral_functions import spectral_drude_lorentz_heom

import torchnise


energy_unit="cm-1"
time_unit="fs"
torchnise.units.set_units(e_unit=energy_unit,t_unit=time_unit)
dt=1
total_time=1000
realizations=100
device="cpu" #"cuda" for GPU "cpu" for CPU
excited_site=49
if device=="cuda":
    torch.backends.cuda.preferred_linalg_library(backend="magma") #bottleneck for gpu calculations is torch.linalg.eigh which we found to be slightly faster with this backend


n_sites=100

H=torch.zeros((n_sites,n_sites))

for i in range(n_sites-1):
    H[i,i+1]=100
    H[i+1,i]=100

positions=torch.zeros((n_sites,3))

positions[:,0]=torch.linspace(0,100,n_sites)

T=300 #K

Omega_k=torch.tensor([0,725,1200])/torchnise.units.HBAR
lambda_k=torch.tensor([100,100,100])
v_k=torch.tensor([1/100,1/100,1/100])

spectralfunc=functools.partial(spectral_drude_lorentz_heom,omega_k=Omega_k,
                               lambda_k=lambda_k,vk=v_k,temperature=T)
spectral_funcs=[spectralfunc]*n_sites #u can specify different spectral functions for different sites. for numerical spectral densites use interpolate1d from scipy

initialState = torch.zeros(n_sites)

initialState[excited_site]=1

Mode1="Population"
T_correction1='None' #TNISE or NONE MLNISE will come soon
Averaging1='standard' #boltzmann standard or interpolated

population, xaxis_p = torchnise.nise.run_nise(H,realizations, total_time,dt, initialState,T, spectral_funcs,
               t_correction=T_correction1,mode=Mode1,averaging_method=Averaging1, device=device,max_reps=10000,use_h5=True)


distance_squared=torch.sum((positions[excited_site,:].reshape((1,3))-positions)**2,dim=1)

diffusion=distance_squared.reshape((1,n_sites))*population
diffusion=torch.sum(diffusion,dim=1)

plt.plot(xaxis_p/1000,population[:,excited_site],label=f"Population")
plt.xlabel("time [ps]")
plt.ylabel("Population")
plt.xlim([torch.min(xaxis_p/1000),torch.max(xaxis_p/1000)])
#plt.ylim([0,1])
plt.legend()
plt.show()
plt.close()

plt.plot(xaxis_p/1000,diffusion,label=f"Diffusion")
plt.xlabel("time [ps]")
plt.ylabel("Diffusion")
plt.xlim([torch.min(xaxis_p/1000),torch.max(xaxis_p/1000)])
#plt.ylim([0,1])
plt.legend()
plt.show()
plt.close()


plt.plot(xaxis_p[:-1]/1000,(diffusion[1:]-diffusion[:-1])/dt,label=f"Diffusion Constant")
plt.xlabel("time [ps]")
plt.ylabel("Diffusion Constant")
plt.xlim([torch.min(xaxis_p/1000),torch.max(xaxis_p/1000)])
#plt.ylim([0,1])
plt.legend()
plt.show()
plt.close()