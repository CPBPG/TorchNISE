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
realizations=100000
device="cuda" #"cuda" for GPU "cpu" for CPU
if device=="cuda":
    #torch.backends.cuda.preferred_linalg_library(backend="magma")
    torch.backends.cuda.preferred_linalg_library(backend="cusolver")
    #we find for some Hamiltonians that the magma backend is faster than cusolver.
    #while for other Hamiltonians the cusolver backend is faster.
    #we could not find a way to predict which backend is faster for a given Hamiltonian.
    #sometimes botha re only abput as fast or even slower than cpu
    #other times we get a speedup of rouhgly 20x
n_sites=2
H=torch.zeros((n_sites,n_sites),dtype=torch.float32)
for i in range(n_sites):
    H[i,i]=0 #torch.rand(1)*200
    if i<n_sites-1:
        V=100# torch.rand(1)*100
        H[i,i+1]=V
        H[i+1,i]=V

#H=torch.tensor([[300,100,0,0],
#                [100,200,100,0],
#                [0,100,100,100],
#                [0,0,100,0]],dtype=torch.float32)

#n_sites=H.shape[0]
T=300 #K

Omega_k=torch.tensor([0,725,1200])/torchnise.units.HBAR
lambda_k=torch.tensor([20,20,20])
v_k=torch.tensor([1/100,1/100,1/100])

spectralfunc=functools.partial(spectral_drude_lorentz_heom,omega_k=Omega_k,
                               lambda_k=lambda_k,vk=v_k,temperature=T)
spectral_funcs=[spectralfunc]*n_sites #u can specify different spectral functions for different sites. for numerical spectral densites use interpolate1d from scipy

initialState = torch.zeros(n_sites)
initialState[0]=1

Mode1="Population"
T_correction1='TNISE' #TNISE, NONE or MLNISE (need to specify model for MLNISE)
Averaging1='standard' #boltzmann standard or interpolated

Mode2="Absorption"
T_correction2='None'
Averaging2="interpolated"
dipole_moments=torch.tensor([[1,0,0]]*n_sites) #dipole moments for each site.

population, xaxis_p = torchnise.nise.run_nise(H,realizations, total_time,dt, initialState,T, spectral_funcs,
               t_correction=T_correction1,mode=Mode1,averaging_method=Averaging1, device=device,max_reps=500000
               ,save_interval=1)



for i in range(n_sites):
    plt.plot(xaxis_p,population[:,i],label=f"site {i+1}")
plt.xlabel("time [ps]")
plt.ylabel("population")
plt.xlim([torch.min(xaxis_p),torch.max(xaxis_p)])
plt.ylim([0,1])
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