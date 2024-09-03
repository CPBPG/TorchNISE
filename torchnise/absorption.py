import numpy as np
from torchnise.pytorch_utility import smooth_damp_to_zero
import torchnise.units as units


def absorption_time_domain(U,mu):
    pigments=U.shape[-1]
    realizations=U.shape[0]
    timesteps=U.shape[1]

    if len(mu.shape)==2: ## time dependence is not supplied
        mu=np.tile(mu,(realizations,timesteps,1,1)) ##copy the same mu vector along the time domain

    absorption_time_domain=0
    damp=1#np.exp(-np.arange(0,timesteps)/1000)
    for xyz in range(0,3):
        for real in range(0,realizations):
            for m in range (0,pigments):
                for n in range(0,pigments):

                    absorption_time_domain+=U[real,:,m,n]*mu[real,:,m,xyz]*mu[real,0,n,xyz]/realizations*damp
    return absorption_time_domain

def absorb_time_to_freq(absorb_time,pad,total_time,dt):
    absorb=np.pad(absorb_time,(0,pad))
    absorb=smooth_damp_to_zero(absorb,int(total_time//dt-total_time//(dt*10)),int(total_time/dt)-1)
    absorb_f=np.fft.fftshift(np.fft.fft(absorb))
    hbar = units.hbar
    x_axis=-hbar*2*np.pi*np.fft.fftshift(np.fft.fftfreq(int((total_time+dt)/dt)+pad, d=dt))
    absorb_f=(absorb_f.real-absorb_f.real[0])/np.max(absorb_f.real-absorb_f.real[0])
    return absorb_f,x_axis

