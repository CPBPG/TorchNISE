"""
This file implements various spectral functions returning the power spectrum.
"""
import numpy as np
from torchnise import units
from scipy.interpolate import interp1d


def spectral_numerical(data,temperature):
    """
    Creates the poser spectrum from a numerical spectral density.

    Args:
        data (numpy.ndarray): numerical spectral density with data[:,0] frequencies
                              and [:,0] the spectral density in the selected unit
        temperature (float): Temperature.

    Returns:
        numpy.ndarray: Spectral density.
    """
    ww=np.concatenate([np.flip(-1*data[:,0]),data[1:,0]])
    ww[ww==0]=1e-15
    spectral_density=np.concatenate([np.flip(-1*data[:,1]),data[1:,1]])
    power_spectrum=np.abs(spectral_density*(2*np.pi*units.K*temperature)/(ww))
    spectralfunc = interp1d(ww,power_spectrum,kind="linear"
                            ,bounds_error=False,fill_value=0)
    return spectralfunc

def spectral_drude(w, gamma, strength, temperature):
    """
    Drude spectral density function.

    Args:
        w (numpy.ndarray): Frequency array.
        gamma (float): Drude relaxation rate.
        strength (float): Strength of the spectral density.
        temperature (float): Temperature.

    Returns:
        numpy.ndarray: Spectral density.
    """
    power_spectrum = (4 * gamma * strength * units.K * temperature /
                        (w**2 + gamma**2))
    return power_spectrum


def spectral_lorentz(w, wk, sk, temperature, gammak):
    """
    Lorentz spectral density function.

    Args:
        w (numpy.ndarray): Frequency array.
        wk (list): Frequencies of the Lorentz peaks.
        sk (list): Strengths of the Lorentz peaks.
        temperature (float): Temperature.
        gammak (float): Damping factor.

    Returns:
        numpy.ndarray: Spectral density.
    """
    power_spectrum = 0
    for i, wk_i in enumerate(wk):
        power_spectrum += (units.HBAR * 4 * units.K * temperature * sk[i]
                             * wk_i**3 * gammak /
                             ((wk_i**2 - w**2)**2 + (w**2 * gammak**2)))
    return power_spectrum


def spectral_drude_lorentz(w, gamma, strength, wk, sk, temperature, gammak):
    """
    Combined Drude and Lorentz spectral density function.

    Args:
        w (numpy.ndarray): Frequency array.
        gamma (float): Drude relaxation rate.
        strength (float): Strength of the spectral density.
        wk (list): Frequencies of the Lorentz peaks.
        sk (list): Strengths of the Lorentz peaks.
        temperature (float): Temperature.
        gammak (float): Damping factor.

    Returns:
        numpy.ndarray: Spectral density.
    """
    power_spectrum = (4 * gamma * strength * units.K * temperature /
                        (w**2 + gamma**2))
    for i, wk_i in enumerate(wk):
        power_spectrum += (units.HBAR * 4 * units.K * temperature * sk[i] *
                             wk_i**3 * gammak /
                             ((wk_i**2 - w**2)**2 + (w**2 * gammak**2)))
    return power_spectrum


def spectral_drude_lorentz_heom(w, omega_k, lambda_k, temperature, vk):
    """
    Drude-Lorentz spectral density function for HEOM.

    Args:
        w (numpy.ndarray): Frequency array.
        omega_k (list): Frequencies of the peaks.
        lambda_k (list): Strengths of the peaks.
        temperature (float): Temperature.
        vk (float): Damping factor.

    Returns:
        numpy.ndarray: Spectral density.
    """
    power_spectrum = 0
    for i, omega_k_i in enumerate(omega_k):
        power_spectrum += (2 * units.K * temperature * vk[i] * lambda_k[i] /
                             ((omega_k_i - w)**2 + vk[i]**2))
        power_spectrum += (2 * units.K * temperature * vk[i] * lambda_k[i] /
                             ((omega_k_i + w)**2 + vk[i]**2))
    return power_spectrum


def spectral_log_normal(w, s_hr, sigma, wc, temperature):
    """
    Log-normal spectral density function.

    Args:
        w (numpy.ndarray): Frequency array.
        s_hr (float): Huang-Rhys factor.
        sigma (float): Width of the log-normal distribution.
        wc (float): Central frequency of the log-normal distribution.
        temperature (float): Temperature.

    Returns:
        numpy.ndarray: Spectral density.
    """
    power_spectrum = (np.sqrt(2 * np.pi) * units.K * temperature * s_hr *
                        units.HBAR / sigma *
                        np.exp(-(np.log(w / wc))**2 / (2 * sigma**2)))
    power_spectrum[np.isnan(power_spectrum)] = 0
    return power_spectrum


def spectral_log_normal_lorentz(w, wk, sk, temperature, gammak, s_hr, sigma,
                                wc):
    """
    Combined Log-Normal and Lorentz spectral density function.

    Args:
        w (numpy.ndarray): Frequency array.
        wk (list): Frequencies of the Lorentz peaks.
        sk (list): Strengths of the Lorentz peaks.
        temperature (float): Temperature.
        gammak (float): Damping factor.
        s_hr (float): Huang-Rhys factor.
        sigma (float): Width of the log-normal distribution.
        wc (float): Central frequency of the log-normal distribution.

    Returns:
        numpy.ndarray: Spectral density.
    """
    power_spectrum = spectral_lorentz(w, wk, sk, temperature, gammak)
    power_spectrum += spectral_log_normal(w, s_hr, sigma, wc, temperature)
    return power_spectrum
