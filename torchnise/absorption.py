import numpy as np
from torchnise.pytorch_utility import smooth_damp_to_zero
import torchnise.units as units


def absorption_time_domain(U,mu,use_damping=False,lifetime=1000,dt=1):
    """
    Calculate the time domain absorption based on the time evolution operator.
    
    Args:
        U (numpy.ndarray): Time evolution operator with dimensions (realizations, timesteps, n_sites, n_sites).
        mu (numpy.ndarray): Dipole moments with either shape (realizations, timesteps, n_sites, 3) for time-dependent cases,
                            or shape (n_sites, 3) for time-independent cases.
        use_damping (bool, optional): Whether to apply exponential damping to account for the lifetime of the state. Default is False.
        lifetime (float, optional): lifetime for the damping. Default is 1000. Units are not important as long as dt and lifetime have
                                    the same unit.
        dt (float, optional): Time step size. Default is 1.
    
    Returns:
        numpy.ndarray: Time domain absorption.
    
    Notes:
        This function calculates the time domain absorption by summing over the contributions of different realizations,
        timesteps, and sites. An optional exponential damping factor can be applied to simulate the decay of the signal
        over time.
    """
    n_sites=U.shape[-1]
    realizations=U.shape[0]
    timesteps=U.shape[1]

    if len(mu.shape)==2: ## time dependence is not supplied
        mu=np.tile(mu,(realizations,timesteps,1,1)) ##copy the same mu vector along the time domain

    absorption_time_domain=0
    if use_damping:
        damp=np.exp(-np.arange(0,timesteps)*dt/lifetime)
    else:
        damp=1
    for xyz in range(0,3):
        for real in range(0,realizations):
            for m in range (0,n_sites):
                for n in range(0,n_sites):

                    absorption_time_domain+=U[real,:,m,n]*mu[real,:,m,xyz]*mu[real,0,n,xyz]/realizations*damp
    return absorption_time_domain

def absorb_time_to_freq(absorb_time,total_time,dt,pad=0,smoothdamp=True,smoothdamp_start_percent=10):
    """
    Convert time domain absorption to an absorption spectrum.

    Args:
        absorb_time (numpy.ndarray): Time domain absorption.
        total_time (float): Total time duration of the absorption signal.
        dt (float): Time step size.
        pad (int, optional): Number of zero padding points for higher frequency resolution. Default is 0.
        smoothdamp (bool, optional): Whether to smooth the transition to the padded region with an exponential damping. Default is True.
        smoothdamp_start_percent (int, optional): Percentage of the time domain absorption affected by smoothing. Default is 10.

    Returns:
        tuple: (numpy.ndarray, numpy.ndarray)
            - Absorption spectrum in the frequency domain.
            - Corresponding frequency axis.

    Notes:
        This function performs a Fast Fourier Transform (FFT) on the time domain absorption data to obtain the absorption spectrum
        in the frequency domain. Optional smoothing can be applied to the transition to the padded region to reduce artifacts.
    """
    absorb=np.pad(absorb_time,(0,pad)) #zero_padding for higher frequency resolution
    
    if smoothdamp:  #smooth the transition to padded region with an exponential damp, 
                    #smoothdamp_start_percent % of the time domain absorbtion is affected.
        absorb_steps=int(total_time//dt)-1
        damp_start=int((100-smoothdamp_start_percent)/100* absorb_steps)
        absorb=smooth_damp_to_zero(absorb,damp_start,absorb_steps)
    #fft to frequency domain
    absorb_f=np.fft.fftshift(np.fft.fft(absorb))
    x_axis=-units.hbar*2*np.pi*np.fft.fftshift(np.fft.fftfreq(int((total_time+dt)/dt)+pad, d=dt*units.t_unit))
    absorb_f=(absorb_f.real-absorb_f.real[0])/np.max(absorb_f.real-absorb_f.real[0])
    return absorb_f,x_axis

