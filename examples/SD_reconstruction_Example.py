import torch
import numpy as np
import matplotlib.pyplot as plt

from torchnise.fft_noise_gen import noise_algorithm
from torchnise.sd_from_noise import SD_Reconstruct_FFT, get_auto, sd_reconstruct_superresolution
from torchnise.example_spectral_functions import spectral_drude_lorentz_heom
import functools
import torchnise.units as units

energy_unit="cm-1"
time_unit="fs"
units.set_units(e_unit=energy_unit,t_unit=time_unit)
T=300
Omega_k=torch.tensor([0,725,1200])/units.hbar
lambda_k=torch.tensor([100,100,100])
v_k=torch.tensor([1/100,1/100,1/100])

spectralfunc=functools.partial(spectral_drude_lorentz_heom,omega_k=Omega_k,
                               lambda_k=lambda_k,vk=v_k,temperature=T)

dt=10 #fs
cutoff=5000 #fs

N_cut=400_000//dt
dw_t = 2*np.pi/(2*N_cut*dt)
full_w_t = np.arange(0,N_cut*dw_t,dw_t)
ww=full_w_t 
S=np.array(spectralfunc(ww))
SD=S/(2*np.pi*units.k*T)*ww
SD_FFT=SD/dt*units.hbar*2*np.pi*units.k*T/ww
SD_FFT[0]=0
reverse_SD_FFT=np.flip(SD_FFT[1:-1])
concat_SD_FFT=np.concatenate((SD_FFT,reverse_SD_FFT))
auto_theoretical=np.fft.ifft(concat_SD_FFT).real/units.hbar
auto_theoretical=auto_theoretical[0:N_cut]
plt.plot(np.linspace(0,100*dt,100),auto_theoretical[0:100])
plt.xlabel("time [fs]")
plt.ylabel("autocorrelation [cm$^{-2}$]")
plt.show()
plt.close()

autos=[auto_theoretical]
titles=["theoretical"]
for total_time in (100_000,1_000_000):
    Generated_Noise=noise_algorithm((1,total_time//dt), dt,spectralfunc,save=False)
    auto=get_auto(Generated_Noise)
    autos.append(auto)
    titles.append(f"{total_time} fs noise")

for i in range (len(autos)):
    auto=autos[i]
    for damping in ["exp","gauss","step"]:
        J_new, x_axis ,auto_damp = SD_Reconstruct_FFT(auto,dt,T,damping_type=damping,cutoff=cutoff,rescale=False)
        S=spectralfunc(x_axis/units.hbar)
        SD=S/(2*np.pi*units.k*T)*x_axis/units.hbar
        plt.plot(x_axis*units.cm_to_eV,J_new,label=f"{damping}")
    plt.plot(x_axis*units.cm_to_eV,SD,label="original")
    plt.xlim([np.min(x_axis*units.cm_to_eV),np.max(x_axis*units.cm_to_eV)])
    plt.ylim(0)
    plt.xlabel("$\omega$ [eV]")
    plt.ylabel("J($\omega$) [cm$^{-1}$]")
    plt.title(titles[i])
    plt.legend()
    plt.show()
    plt.close()
