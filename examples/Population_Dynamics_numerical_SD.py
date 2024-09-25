import numpy as np
import matplotlib.pyplot as plt
import torch
import functools
from torchnise.spectral_functions import spectral_drude_lorentz_heom, spectral_numerical

import torchnise


energy_unit="cm-1"
time_unit="fs"
torchnise.units.set_units(e_unit=energy_unit,t_unit=time_unit)
dt=1
total_time=100000
realizations=10
device="cpu" #"cuda" for GPU "cpu" for CPU
if device=="cuda":
    torch.backends.cuda.preferred_linalg_library(backend="magma") #bottleneck for gpu calculations is torch.linalg.eigh which we found to be slightly faster with this backend

E=np.loadtxt("examples/data/Excitation_energy_average.dat")
E=E-np.mean(E)
H=np.loadtxt("examples/data/Hamiltonian_sup.dat")
H=torch.tensor(H)
for i in range(0,H.shape[0]):
    H[i,i]=E[i]
H=H/torchnise.units.CM_TO_EV

n_sites=H.shape[0]
T=300 #K

J_a=np.loadtxt("examples/data/SPD/J_chl_a.dat")
J_a[:,0]=J_a[:,0]/torchnise.units.HBAR
J_b=np.loadtxt("examples/data/SPD/J_chl_b.dat")
J_b[:,0]=J_b[:,0]/torchnise.units.HBAR
spectralfunc_a = spectral_numerical(J_a,T)
spectralfunc_b = spectral_numerical(J_b,T)
spectral_funcs=[] #u can specify different spectral functions for different sites. for numerical spectral densites use interpolate1d from scipy


lh_complexes=[]
lh_complex_of_pigment=[]


with open('examples/data/Pigment_naming.dat', 'r') as file:
    # Read each line in the file
    for line in file:
        # Split the line into elements
        complex, type, number = line.split()
        if type=="CLA":
            spectral_funcs.append(spectralfunc_a)
            if not complex in lh_complexes:
                lh_complexes.append(complex)
            lh_complex_of_pigment.append(complex)
        elif type=="CHL":
            spectral_funcs.append(spectralfunc_b)
            if not complex in lh_complexes:
                lh_complexes.append(complex)
            lh_complex_of_pigment.append(complex)




initialState = torch.zeros(n_sites)
initialState[0] = 1

Mode1="Population"
T_correction1='NISE' #TNISE or NONE MLNISE will come soon
Averaging1='standard' #boltzmann standard or interpolated


population, xaxis_p = torchnise.nise.run_nise(H,realizations, total_time,dt, initialState,T, spectral_funcs,
            t_correction=T_correction1,mode=Mode1,averaging_method=Averaging1, device=device,max_reps=100,save_interval=100,
            save_u= True, save_u_file=f"examples/data/u_test.pt")

lh_complex_pop=[0]*len(lh_complexes)
for i in range(n_sites):
    pop_index=lh_complexes.index(lh_complex_of_pigment[i])
    lh_complex_pop[pop_index]+=population[:,i]
for i in range(len(lh_complexes)):
    plt.plot(xaxis_p,lh_complex_pop[i],label=lh_complexes[i])
plt.xlabel("time [ps]")
plt.ylabel("population")
plt.xlim([torch.min(xaxis_p),torch.max(xaxis_p)])
plt.ylim([0,1])

plt.legend()
plt.savefig(f"examples/data/fig{i}.pdf")
#plt.show()
plt.close()
