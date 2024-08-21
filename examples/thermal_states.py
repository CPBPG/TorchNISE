import mlnise
import numpy as np
import functools
from mlnise.example_spectral_functions import spectral_Drude_Lorentz_Heom_thermal, spectral_Drude,spectral_Drude_Lorentz_Heom
import torch
import matplotlib.pyplot as plt

k = 8.6173303E-5 # in eV/K. 
T = 300 #Temperature in K
hbar = 0.658211951 #in eV fs
cm_to_eV=1.23984E-4
k=k/cm_to_eV
plots=[]
plots_old_ave=[]
spectral_func_start = spectral_Drude_Lorentz_Heom
for tau in [100,800,200,400]:

    for i in range (2,4):
        H=np.array([[200,100],[100,0]]) #(f"data/FMO/H_{8}site_time_independent_time_{"dependent"}_E.npy")
        n_state=H.shape[0]
        Omegak=np.array([0,725,1200])*cm_to_eV/hbar
        Omegak=Omegak[:i]
        HEOM_ER=100
        HEOM_ER_peak=HEOM_ER/len(Omegak)
        print("HEOM Er Peak",HEOM_ER_peak)
        lambdak=np.array([20,20,20])
        lambdak=lambdak[:i]
        vk=np.array([1/100,1/100,1/tau])
        vk=vk[:i]
        spectralfunc=functools.partial(spectral_func_start,Omegak=Omegak,lambdak=lambdak,hbar=hbar,k=k,T=T,vk=vk)
        
        ww=np.linspace(0,0.2,500)/hbar
        S=spectralfunc(ww)
        SD=S/(2*np.pi*k*T)*ww
        plt.plot(ww*hbar,SD,label="original")
        #plt.show()
        #plt.close()
        tau=100
        Er=100
        H_0=torch.tensor(H,dtype=torch.float)
        model=mlnise.mlnise_model.MLNISE()
        reals=10000
        dt=1
        initiallyExcitedState=0
        total_time=1000
        spectral_funcs=[spectralfunc]
        
        #trajectory_time=None, T_correction='None', maxreps=1000000, use_filter=False, filter_order=10, filter_cutoff=0.1, mode="population",mu=None,device="cpu",memory_mapped=False,save_Interval=10
        device="cpu"
        T_correction='Jansen'
        save_Interval=1
        memory_mapped=False
        maxreps=10000
        avg_output, avg_oldave, avg_newave,avg_absorb_time,x_axis, absorb_f = mlnise.run_mlnise.RunNiseOptions(model, reals, H_0, tau, Er, T, dt, initiallyExcitedState, total_time, spectral_funcs,T_correction=T_correction,save_Interval=save_Interval,memory_mapped=memory_mapped,device=device,maxreps=maxreps)
        
        #for i in range(0,n_state):
        plots.append(avg_output[:,0])
        plots_old_ave.append(avg_oldave[:,0])
    plt.show()
    plt.close()
    for plot in plots:
        plt.plot(plot)
    plt.show()
    plt.close()
    for plot in plots_old_ave:
        plt.plot(plot)
    plt.show()
    plt.close()
    
