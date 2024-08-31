import numpy as np
import matplotlib.pyplot as plt
import torch
import functools
from mlnise.example_spectral_functions import spectral_Drude_Lorentz_Heom

import mlnise


energy_unit="cm-1"
time_unit="fs"
mlnise.units.set_units(e_unit=energy_unit,t_unit=time_unit)
dt=1
total_time=1000
realizations=1000
device="cpu" #"cuda" for GPU "cpu" for CPU
if device=="cuda":
    torch.backends.cuda.preferred_linalg_library(backend="magma") #bottleneck for gpu calculations is torch.linalg.eigh which we found to be slightly faster with this backend

H=torch.tensor([[100,100],
                [100,0  ]])

n_sites=H.shape[0]
T=300 #K

Omega_k=torch.tensor([0,725,1200])
lambda_k=torch.tensor([100,100,100])
v_k=torch.tensor([1/100,1/100,1/100])

spectralfunc=functools.partial(spectral_Drude_Lorentz_Heom,Omega_k=Omega_k,
                               lambda_k=lambda_k,v_k=v_k,T=T)
spectral_funcs=[spectralfunc,spectralfunc] 

initialState = torch.tensor([1,0])

Mode1="Population"
T_correction1='TNISE'
Averaging1='Interpolated'

Mode2="Absorption"
T_correction2='None'
Averaging2="Standard"
dipole_moments=torch.tensor([[1,0,0],
                             [1,0,0]])

population, xaxis_p = mlnise.nise.run_NISE(H,realizations, total_time,dt, initialState,T, spectral_funcs,
               T_correction=T_correction1,mode=Mode1,averaging_method=Averaging1, device=device)

absorption, xaxis_f = mlnise.nise.run_NISE(H,realizations, total_time,dt, initialState,T, spectral_funcs,
               T_correction=T_correction2,mode=Mode2,averaging_method=Averaging2,mu=dipole_moments, device=device)

for i in range(n_sites):
    plt.plot(xaxis_p,population[:,i],label=f"site {i+1}")
plt.xlabel("time [ps]")
plt.ylabel("population")
plt.xlim([torch.min(xaxis_p),torch.max(xaxis_p)])
plt.ylim([0,1])
plt.legend()
plt.show()
plt.close()

plt.plot(xaxis_f,absorption)
plt.ylabel("absorption [a.u.]")
plt.xlabel("frequency [cm$^-1$]")
plt.show()
plt.close()
