"""
This file implements various spectral functions returning the power spectrum.
"""
import numpy as np
from torchnise import units


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
    spectral_density = (4 * gamma * strength * units.k * temperature /
                        (w**2 + gamma**2))
    return spectral_density


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
    spectral_density = 0
    for i, wk_i in enumerate(wk):
        spectral_density += (units.hbar * 4 * units.k * temperature * sk[i]
                             * wk_i**3 * gammak /
                             ((wk_i**2 - w**2)**2 + (w**2 * gammak**2)))
    return spectral_density


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
    spectral_density = (4 * gamma * strength * units.k * temperature /
                        (w**2 + gamma**2))
    for i, wk_i in enumerate(wk):
        spectral_density += (units.hbar * 4 * units.k * temperature * sk[i] *
                             wk_i**3 * gammak /
                             ((wk_i**2 - w**2)**2 + (w**2 * gammak**2)))
    return spectral_density


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
    spectral_density = 0
    for i, omega_k_i in enumerate(omega_k):
        spectral_density += (2 * units.k * temperature * vk[i] * lambda_k[i] /
                             ((omega_k_i - w)**2 + vk[i]**2))
        spectral_density += (2 * units.k * temperature * vk[i] * lambda_k[i] /
                             ((omega_k_i + w)**2 + vk[i]**2))
    return spectral_density


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
    spectral_density = (np.sqrt(2 * np.pi) * units.k * temperature * s_hr *
                        units.hbar / sigma *
                        np.exp(-(np.log(w / wc))**2 / (2 * sigma**2)))
    spectral_density[np.isnan(spectral_density)] = 0
    return spectral_density


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
    spectral_density = spectral_lorentz(w, wk, sk, temperature, gammak)
    spectral_density += spectral_log_normal(w, s_hr, sigma, wc, temperature)
    return spectral_density
