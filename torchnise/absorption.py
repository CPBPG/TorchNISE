"""
This module provides functions to compute time domain absorption and convert it 
to an absorption spectrum using FFT. It allows for optional damping to simulate 
signal decay over time.
"""

import numpy as np
from torchnise.pytorch_utility import smooth_damp_to_zero
from torchnise import units



def absorption_time_domain(time_evolution_operator, dipole_moments,
                           use_damping=False, lifetime=1000, dt=1):
    """
    Calculate the time domain absorption based on the time evolution operator.
    
    Args:
        time_evolution_operator (numpy.ndarray): Time evolution operator with 
            dimensions (realizations, timesteps, n_sites, n_sites).
        dipole_moments (numpy.ndarray): Dipole moments with either shape 
            (realizations, timesteps, n_sites, 3) for time-dependent cases, or 
            shape (n_sites, 3) for time-independent cases.
        use_damping (bool, optional): Whether to apply exponential damping to 
            account for the lifetime of the state. Default is False.
        lifetime (float, optional): Lifetime for the damping. Default is 1000. 
            Units are not important as long as dt and lifetime have the same 
            unit. dt (float, optional): Time step size. Default is 1.
    
    Returns:
        numpy.ndarray: Time domain absorption.
    
    Notes:
        This function calculates the time domain absorption by summing over the 
        contributions of different realizations, timesteps, and sites. An 
        optional exponential damping factor can be applied to simulate the 
        decay of the signal over time.
    """
    n_sites = time_evolution_operator.shape[-1]
    realizations = time_evolution_operator.shape[0]
    timesteps = time_evolution_operator.shape[1]

    if len(dipole_moments.shape) == 2:  # Time dependence is not supplied
        dipole_moments = np.tile(dipole_moments,
                                 (realizations, timesteps, 1, 1))

    absorption_td = 0
    if use_damping:
        damp = np.exp(-np.arange(0, timesteps) * dt / lifetime)
    else:
        damp = 1

    for xyz in range(3):
        for real in range(realizations):
            for m in range(n_sites):
                for n in range(n_sites):
                    absorption_td += (time_evolution_operator[real, :, m, n] *
                                      dipole_moments[real, :, m, xyz] *
                                      dipole_moments[real, 0, n, xyz] /
                                      realizations * damp)
    return absorption_td


def absorb_time_to_freq(absorb_time, config):
    """
    Convert time domain absorption to an absorption spectrum.

    Args:
        absorb_time (numpy.ndarray): Time domain absorption.
        config (dict): Configuration dictionary containing parameters:
            - total_time (float): Total time duration of the absorption signal.
            - dt (float): Time step size.
            - pad (int): Number of zero padding points for higher frequency 
            resolution.
            - smoothdamp (bool): Whether to smooth the transition to the padded 
            region with an exponential damping.
            - smoothdamp_start_percent (int): Percentage of the time domain 
            absorption affected by smoothing.

    Returns:
        tuple: (numpy.ndarray, numpy.ndarray)
            - Absorption spectrum in the frequency domain.
            - Corresponding frequency axis.
    """
    total_time = config.get("total_time")
    dt = config.get("dt", 1)
    pad = config.get("pad", 0)
    smoothdamp = config.get("smoothdamp", True)
    smoothdamp_start_percent = config.get("smoothdamp_start_percent", 10)
    # Zero padding for higher frequency resolution
    absorb = np.pad(absorb_time, (0, pad))

    if smoothdamp:
        # Smooth the transition to the padded region with an exponential damp
        absorb_steps = int(total_time // dt) - 1
        damp_start = int((100 - smoothdamp_start_percent) / 100 * absorb_steps)
        absorb = smooth_damp_to_zero(absorb, damp_start, absorb_steps)

    # FFT to frequency domain
    absorb_f = np.fft.fftshift(np.fft.fft(absorb))
    freq = np.fft.fftfreq(int((total_time + dt) / dt) + pad,
                          d=dt * units.T_UNIT)
    x_axis = -units.HBAR * 2 * np.pi * np.fft.fftshift(freq)
    absorb_f_max = np.max(absorb_f.real - absorb_f.real[0])
    absorb_f = (absorb_f.real - absorb_f.real[0]) / absorb_f_max
    return absorb_f, x_axis
