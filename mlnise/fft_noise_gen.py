"""
This file implements the fftNoiseGEN algorithm for time correlated Noise.
"""
import numpy as np
from scipy.interpolate import interp1d


#inspired by https://stackoverflow.com/a/64288861
def inverse_sample(dist, shape, x_min=-100, x_max=100, n=1e5, **kwargs):
    """
    Generates samples from a given distribution using the inverse transform sampling method.
    
    Parameters:
    - dist (callable): Probability density function (PDF) of the desired distribution.
    - pnts (int): Number of points/samples to generate.
    - x_min (float): Minimum x value for the range of the distribution.
    - x_max (float): Maximum x value for the range of the distribution.
    - n (int): Number of points used to approximate the cumulative distribution function (CDF).
    - **kwargs: Additional arguments to pass to the PDF function.
    
    Returns:
    - np.ndarray: Samples drawn from the specified distribution.
    """
    x = np.linspace(x_min, x_max, int(n))
    cumulative = np.cumsum(dist(x, **kwargs))
    cumulative -= cumulative.min()
    f = interp1d(cumulative / cumulative.max(), x)
    return f(np.random.random(shape))



def noise_algorithm(shape, dt, spectral_func,axis=-1, sample_dist=None, discard_half=True, save=False, save_name=None):
    """
     Generates time-correlated noise following the power spectrum provided in spectral_func.
    
     Parameters:
     - shape (tuple): Shape of the output noise array. The first dimension is the number of realizations,
                      the second dimension is the number of steps, and the remaining dimensions can be arbitrary
                      for example, number of sites.
     - dt (float): Time step size.
     - spectral_func (callable): Function that defines the power spectrum of the noise.
     - axis (int, optional): The axis along which the noise should be correlated. Default is -1 (last axis).
     - sample_dist (callable, optional): A function that generates an array of random numbers. Can be used
                                         if a non-normal distribution in the time domain is desired.
     - discard_half (bool, optional): If True, generates noise for twice the number of steps and discards the second half
                                      to avoid circular correlation. Default is True.
     - save (bool, optional): If True, the generated noise array will be saved to a file.
     - save_name (str, optional): The name of the file to save the noise array. Required if save is True.
    
     Returns:
     - np.ndarray: Time-correlated noise with the specified shape.
     """
    steps = shape[axis]
    if discard_half:
        extended_shape = list(shape)
        extended_shape[axis] = 2 * steps
        shape = tuple(extended_shape)
    
    # Generate white noise with the correct dimensions
    if sample_dist:
        noise_samples = sample_dist(shape)
        # Rescale to zero mean and unit variance
        white_noise = (noise_samples - np.mean(noise_samples)) / np.std(noise_samples)
    else:
        white_noise = np.random.normal(0, 1, size=shape)

    # Fourier transform of the white noise along the steps axis
    freq = np.fft.fft(white_noise, axis=axis) * (1 / np.sqrt(dt))

    # Frequencies associated with the FFT of white noise
    freq_bins = np.fft.fftfreq(shape[axis], dt) * 2 * np.pi

    # Envelope the frequencies with the spectral function
    spectral_density = np.sqrt(spectral_func(freq_bins))  
    reshaped_spectral_density = np.reshape(spectral_density, [1 if dim != axis else len(spectral_density) for dim in range(len(shape))])

    
    freq_enveloped = reshaped_spectral_density * freq

    # Inverse Fourier transform to obtain the time-domain noise
    time_domain_noise = np.real(np.fft.ifft(freq_enveloped, axis=axis))
    
    if discard_half:
        slices = [slice(None)] * len(shape)
        slices[axis] = slice(0, steps)
        time_domain_noise = time_domain_noise[tuple(slices)]
    
    # Save the noise array if required
    if save and save_name:
        np.save(save_name, time_domain_noise)

    return time_domain_noise
        