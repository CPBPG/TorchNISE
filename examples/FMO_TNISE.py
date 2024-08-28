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
Omegak=Omegak[:2]
HEOM_ER=200
HEOM_ER_peak=HEOM_ER/len(Omegak)
print("HEOM Er Peak",HEOM_ER_peak)
lambdak=np.array([HEOM_ER_peak,HEOM_ER_peak,HEOM_ER_peak])
lambdak=lambdak[:2]
vk=np.array([1/100,1/100,1/100])
vk=vk[:2]
spectralfunc=functools.partial(spectral_Drude_Lorentz_Heom,Omegak=Omegak,lambdak=lambdak,hbar=hbar,k=k,T=T,vk=vk)
tau=100
Er=100
H_0=torch.tensor(H,dtype=torch.float)
model=mlnise.mlnise_model.MLNISE()
reals=10000
dt=1
initiallyExcitedState=0
total_time=60000
spectral_funcs=[spectralfunc]

#trajectory_time=None, T_correction='None', maxreps=1000000, use_filter=False, filter_order=10, filter_cutoff=0.1, mode="population",mu=None,device="cpu",memory_mapped=False,save_Interval=10
device="cpu"
T_correction='Jansen'
save_Interval=10
memory_mapped=False
maxreps=200
avg_output, avg_blend, avg_boltzmann,avg_absorb_time,x_axis, absorb_f = mlnise.run_mlnise.RunNiseOptions(model, reals, H_0, tau, Er, T, dt, initiallyExcitedState, total_time, spectral_funcs,T_correction=T_correction,save_Interval=save_Interval,memory_mapped=memory_mapped,device=device,maxreps=maxreps)

np.save("avg_blend_60ps_2peak.npy",avg_blend)
np.save("avg_output_60ps_2peak.npy",avg_output)
np.save("avg_boltzmann_60ps_2peak.npy",avg_boltzmann)

colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
 '#7f7f7f', '#bcbd22', '#17becf']
boltzman_correct=torch.diag(torch.matrix_exp(torch.tensor(-H/(k*T))))/torch.sum(torch.diag(torch.matrix_exp(torch.tensor(-H/(k*T)))))
H_TD=torch.zeros((1000, 1000, n_state, n_state))
H_TD[:]=torch.tensor(H)
mynoise=mlnise.run_mlnise.gen_noise(spectral_funcs, dt, (1000,1000,n_state))
for i in range(n_state):
    H_TD[:, :, i, i] += mynoise[:, :, i]

H_eq=torch.diagonal(torch.matrix_exp(-H_TD/(k*T)),dim1=-2,dim2=-1)
boltzman_incorrect=torch.mean(H_eq/torch.sum(H_eq,dim=2,keepdim=True),dim=[0,1])

xaxis=torch.linspace(0, total_time/1000,avg_blend.shape[0] )

for i in range(0,n_state):
    plt.plot(xaxis,avg_blend[:,i],color=colors[i],label=f"site {i+1}")
    plt.plot([torch.min(xaxis),torch.max(xaxis)],[boltzman_correct[i],boltzman_correct[i]],linestyle="dashed",color=colors[i])
    plt.plot([torch.min(xaxis),torch.max(xaxis)],[boltzman_incorrect[i],boltzman_incorrect[i]],linestyle="dotted",color=colors[i])
plt.xlabel("time [ps]")
plt.ylabel("population")
plt.xlim([torch.min(xaxis),torch.max(xaxis)])
plt.ylim([0,1])
plt.legend()
plt.show()
plt.close()
for i in range(0,n_state):
    plt.plot(xaxis,avg_boltzmann[:,i],color=colors[i],label=f"site {i+1}")
    plt.plot([torch.min(xaxis),torch.max(xaxis)],[boltzman_correct[i],boltzman_correct[i]],color=colors[i],linestyle="dashed")
    plt.plot([torch.min(xaxis),torch.max(xaxis)],[boltzman_incorrect[i],boltzman_incorrect[i]],linestyle="dotted",color=colors[i])
plt.xlabel("time [ps]")
plt.ylabel("population")
plt.xlim([torch.min(xaxis),torch.max(xaxis)])
plt.ylim([0,1])
plt.legend()
plt.show()
plt.close()
for i in range(0,n_state):
    plt.plot(xaxis,avg_output[:,i],color=colors[i],label=f"site {i+1}")
    plt.plot([torch.min(xaxis),torch.max(xaxis)],[boltzman_correct[i],boltzman_correct[i]],color=colors[i],linestyle="dashed")
    plt.plot([torch.min(xaxis),torch.max(xaxis)],[boltzman_incorrect[i],boltzman_incorrect[i]],linestyle="dotted",color=colors[i])
plt.xlabel("time [ps]")
plt.ylabel("population")
plt.xlim([torch.min(xaxis),torch.max(xaxis)])
plt.ylim([0,1])
plt.legend()
plt.show()
plt.close()