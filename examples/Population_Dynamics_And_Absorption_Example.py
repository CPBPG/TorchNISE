import numpy as np
import matplotlib.pyplot as plt
import torch
import functools
from torchnise.example_spectral_functions import spectral_drude_lorentz_heom

import torchnise


energy_unit="cm-1"
time_unit="fs"
torchnise.units.set_units(e_unit=energy_unit,t_unit=time_unit)
dt=1
total_time=1000
realizations=10000
device="cpu" #"cuda" for GPU "cpu" for CPU
if device=="cuda":
    torch.backends.cuda.preferred_linalg_library(backend="magma") #bottleneck for gpu calculations is torch.linalg.eigh which we found to be slightly faster with this backend

H=torch.tensor([[100,100],
                [100,0  ]])

n_sites=H.shape[0]
T=300 #K

Omega_k=torch.tensor([0,725,1200])/torchnise.units.HBAR
lambda_k=torch.tensor([100,100,100])
v_k=torch.tensor([1/100,1/100,1/100])

spectralfunc=functools.partial(spectral_drude_lorentz_heom,omega_k=Omega_k,
                               lambda_k=lambda_k,vk=v_k,temperature=T)
spectral_funcs=[spectralfunc,spectralfunc] #u can specify different spectral functions for different sites. for numerical spectral densites use interpolate1d from scipy

initialState = torch.tensor([1,0])

Mode1="Population"
T_correction1='TNISE' #TNISE or NONE MLNISE will come soon
Averaging1='interpolated' #boltzmann standard or interpolated

Mode2="Absorption"
T_correction2='None'
Averaging2="interpolated"
dipole_moments=torch.tensor([[1,0,0],
                             [1,0,0]])

population, xaxis_p = torchnise.nise.run_nise(H,realizations, total_time,dt, initialState,T, spectral_funcs,
               t_correction=T_correction1,mode=Mode1,averaging_method=Averaging1, device=device,max_reps=10000)

for i in range(n_sites):
    plt.plot(xaxis_p,population[:,i],label=f"site {i+1}")
plt.xlabel("time [ps]")
plt.ylabel("population")
plt.xlim([torch.min(xaxis_p),torch.max(xaxis_p)])
plt.ylim([0,1])
plt.legend()
plt.show()
plt.close()

"""absorption, xaxis_f = torchnise.nise.run_nise(H,realizations, total_time,dt, initialState,T, spectral_funcs,
               t_correction=T_correction2,mode=Mode2,averaging_method=Averaging2,mu=dipole_moments, device=device)



plt.plot(xaxis_f,absorption)
plt.ylabel("absorption [a.u.]")
plt.xlabel("frequency [cm$^-1$]")
plt.show()
plt.close()
"""