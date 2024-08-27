import mlnise
import numpy as np
import functools
from mlnise.example_spectral_functions import spectral_Drude_Lorentz_Heom, spectral_Drude
import torch
import matplotlib.pyplot as plt

k = 8.6173303E-5 # in eV/K. 
T = 300 #Temperature in K
hbar = 0.658211951 #in eV fs
cm_to_eV=1.23984E-4
k=k/cm_to_eV

H=np.load("data/H_8site_time_independent_time_dependent_E.npy")
n_state=H.shape[0]
Omegak=np.array([0,725,1200])*cm_to_eV/hbar
Omegak=Omegak#[:2]
HEOM_ER=200
HEOM_ER_peak=HEOM_ER/len(Omegak)
print("HEOM Er Peak",HEOM_ER_peak)
lambdak=np.array([HEOM_ER_peak,HEOM_ER_peak,HEOM_ER_peak])
lambdak=lambdak#[:2]
vk=np.array([1/100,1/100,1/100])
vk=vk#[:2]
spectralfunc=functools.partial(spectral_Drude_Lorentz_Heom,Omegak=Omegak,lambdak=lambdak,hbar=hbar,k=k,T=T,vk=vk)
tau=100
Er=100
H_0=torch.tensor(H,dtype=torch.float)
model=mlnise.mlnise_model.MLNISE()
reals=100
dt=1
initiallyExcitedState=0
total_time=250
spectral_funcs=[spectralfunc]

#trajectory_time=None, T_correction='None', maxreps=1000000, use_filter=False, filter_order=10, filter_cutoff=0.1, mode="population",mu=None,device="cpu",memory_mapped=False,save_Interval=10
device="cpu"
T_correction='None'
save_Interval=10
memory_mapped=True
maxreps=1000
avg_output, avg_oldave, avg_newave,avg_absorb_time,x_axis, absorb_f = mlnise.run_mlnise.RunNiseOptions(model, reals, H_0, tau, Er, T, dt, initiallyExcitedState, total_time, spectral_funcs,T_correction=T_correction,save_Interval=save_Interval,memory_mapped=memory_mapped,device=device,maxreps=maxreps)

np.save("avg_old.npy",avg_oldave)
np.save("avg_output.npy",avg_output)
np.save("avg_newave.npy",avg_newave)


for i in range(0,n_state):
    plt.plot(avg_output[:,i])
plt.show()