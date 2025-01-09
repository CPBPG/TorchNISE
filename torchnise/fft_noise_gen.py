"""
This file implements the fftNoiseGEN algorithm for time correlated Noise.
"""
import os
import time
import numpy as np
from scipy.interpolate import interp1d
import torch
import tqdm
from torchnise.pytorch_utility import H5Tensor
from torchnise import units
#inspired by https://stackoverflow.com/a/64288861
def inverse_sample(dist, shape, x_min=-100, x_max=100, n=1e5, **kwargs):
    """
    Generates samples from a given distribution using the inverse transform
    sampling method.
    
    Args:
        dist (callable): Probability density function (PDF) of the desired
            distribution.
        shape (tuple): Shape of the output samples.
        x_min (float): Minimum x value for the range of the distribution.
        x_max (float): Maximum x value for the range of the distribution.
        n (int): Number of points used to approximate the cumulative distribution
            function (CDF).
        **kwargs: Additional arguments to pass to the PDF function.
    
    Returns:
        np.ndarray: Samples drawn from the specified distribution.
    """
    x = np.linspace(x_min, x_max, int(n))
    cumulative = np.cumsum(dist(x, **kwargs))
    cumulative -= cumulative.min()
    f = interp1d(cumulative / cumulative.max(), x)
    return f(np.random.random(shape))

def gen_noise(spectral_funcs, dt, shape, use_h5,dtype=torch.float32):
    """
    Generates time-correlated noise following the power spectrums provided in
    spectral_funcs.
    
    Args:
        shape (tuple): Shape of the output noise array. The first dimension is
            the number of realizations, the second dimension is the number of
            steps, and the remaining dimension is the number of sites.
        dt (float): Time step size.
        spectral_funcs (list(callable)): Must have either len 1 if all sites
            follow the same power spectrum, or len n_sites=shape[-1] to provide a
            separate power spectrum for each site.
        use_h5 (bool): If True, uses h5 to save memory.
        dtype (torch.dtype): Data type of the output noise array.
    
    Returns:
        torch.Tensor: Time-correlated noise with the specified shape.
    """
    if len(shape) != 3:
        raise ValueError(f""""
                         gen_noise requires a shape tuple with
                          (reals,steps,n_sites)
                          but a tuple of size {len(shape)} was given""")

    steps, reals, n_sites = shape
    if use_h5:
        filepath1=f"noise_{torch.randint(0,1000000,(1,)).item()}.h5"
        while os.path.exists(filepath1):
            filepath1=f"noise_{torch.randint(0,1000000,(1,)).item()}.h5"
        filepath2=f"noise_{torch.randint(0,1000000,(1,)).item()}.h5"
        while os.path.exists(filepath2):
            filepath2=f"noise_{torch.randint(0,1000000,(1,)).item()}.h5"
        noise2 = H5Tensor(shape=(n_sites,reals,steps),h5_filepath=filepath2,dtype=dtype)
        noise = H5Tensor(shape=shape,h5_filepath=filepath1,dtype=dtype)
    else:
        noise = torch.zeros(shape, dtype=dtype)

    if len(spectral_funcs) == 1:
        for i in range(n_sites):
            noise[:, :, i] = noise_algorithm_torch((steps, reals), dt,
                                                   spectral_funcs[0], axis=0)
        return noise

    if len(spectral_funcs) == n_sites:
        if use_h5:

            for i in tqdm.tqdm(range(n_sites)):
                noise2[i, :,:] = noise_algorithm_torch((reals,steps), dt,
                                                spectral_funcs[i], axis=1).squeeze()
            # Iterate over timesteps in chunks
            for s in tqdm.tqdm(range(0, steps, 10000)):
                s_end = min(s + 10000, steps)
                # Read a chunk of data (shape: n_sites, reals, chunk_size)
                start_time = time.time()
                data_chunk = noise2[:, :, s:s_end]
                print(f"Read time: {time.time() - start_time}")
                # Rearrange axes to get shape (chunk_size, reals, n_sites)
                data_chunk = data_chunk.swapaxes(0, 2)
                start_time = time.time()
                noise[s:s_end, :, :] = data_chunk
                print(f"Write time: {time.time() - start_time}")
                # Process your data_chunk here without loading the full dataset
                # For example, pass it to your model or perform computations


            os.remove(filepath2)
        else:
            for i in tqdm.tqdm(range(n_sites),desc="Noise gen site"):
                noise[:, :, i] = noise_algorithm_torch((steps, reals), dt,
                                                   spectral_funcs[i], axis=0)
        return noise



def noise_algorithm(shape, dt, spectral_func, axis=-1, sample_dist=None,
                    discard_half=True, save=False, save_name=None):
    """
    Generates time-correlated noise following the power spectrum provided in
    spectral_func.
    
    Args:
        shape (tuple): Shape of the output noise array.
        dt (float): Time step size.
        spectral_func (callable): Function that defines the power spectrum of
            the noise.
        axis (int, optional): The axis along which the noise should be
            correlated. Default is -1 (last axis).
        sample_dist (callable, optional): Function to generate an array of
            random numbers for non-normal distribution.
        discard_half (bool, optional): If True, generates noise for twice the
            number of steps and discards the second half. Default is True.
        save (bool, optional): If True, saves the generated noise array to a
            file.
        save_name (str, optional): Name of the file to save the noise array.
            Required if save is True.
    
    Returns:
        np.ndarray: Time-correlated noise with the specified shape.
    """
    # Get positive axis
    axis = axis % len(shape)
    steps = shape[axis]

    if discard_half:
        extended_shape = list(shape)
        extended_shape[axis] = 2 * steps
        shape = tuple(extended_shape)

    # Generate white noise with the correct dimensions
    if sample_dist:
        noise_samples = sample_dist(shape)
        # Rescale to zero mean and unit variance
        white_noise = ((noise_samples - np.mean(noise_samples)) /
                       np.std(noise_samples))
    else:
        white_noise = np.random.normal(0, 1, size=shape)

    # Fourier transform of the white noise along the steps axis
    freq = (np.fft.fft(white_noise, axis=axis) *
            (1 / np.sqrt(dt * units.T_UNIT)))

    # Frequencies associated with the FFT of white noise
    freq_bins = np.fft.fftfreq(shape[axis], dt * units.T_UNIT) * 2 * np.pi

    # Envelope the frequencies with the spectral function
    spectral_density = np.sqrt(spectral_func(freq_bins))
    reshaped_spectral_density = np.reshape(spectral_density,
                                           [1 if dim != axis else
                                            len(spectral_density)
                                            for dim in range(len(shape))])

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

def noise_algorithm_torch(shape, dt, spectral_func, axis=-1, sample_dist=None,
                    discard_half=True, save=False, save_name=None):
    """
    Generates time-correlated noise following the power spectrum provided in
    spectral_func.
    
    Args:
        shape (tuple): Shape of the output noise array.
        dt (float): Time step size.
        spectral_func (callable): Function that defines the power spectrum of
            the noise.
        axis (int, optional): The axis along which the noise should be
            correlated. Default is -1 (last axis).
        sample_dist (callable, optional): Function to generate an array of
            random numbers for non-normal distribution.
        discard_half (bool, optional): If True, generates noise for twice the
            number of steps and discards the second half. Default is True.
        save (bool, optional): If True, saves the generated noise array to a
            file.
        save_name (str, optional): Name of the file to save the noise array.
            Required if save is True.
    
    Returns:
        torch.tensor: Time-correlated noise with the specified shape.
    """
    # Get positive axis
    axis = axis % len(shape)
    steps = shape[axis]

    if discard_half:
        extended_shape = list(shape)
        extended_shape[axis] = 2 * steps
        shape = tuple(extended_shape)

    # Generate white noise with the correct dimensions
    if sample_dist:
        noise_samples = sample_dist(shape)
        # Rescale to zero mean and unit variance
        white_noise = ((noise_samples - torch.mean(noise_samples)) /
                       torch.std(noise_samples))
    else:
        white_noise = torch.normal(0, 1, size=shape)

    # Fourier transform of the white noise along the steps axis
    freq = (torch.fft.fft(white_noise, axis=axis) *
            (1 / np.sqrt(dt * units.T_UNIT)))

    # Frequencies associated with the FFT of white noise
    freq_bins = torch.fft.fftfreq(shape[axis], dt * units.T_UNIT) * 2 * np.pi

    # Envelope the frequencies with the spectral function
    spectral_density = torch.sqrt(torch.tensor(spectral_func(freq_bins)))
    reshaped_spectral_density = spectral_density.reshape(
                                           [1 if dim != axis else
                                            len(spectral_density)
                                            for dim in range(len(shape))])

    freq_enveloped = reshaped_spectral_density * freq

    # Inverse Fourier transform to obtain the time-domain noise
    time_domain_noise = (torch.fft.ifft(freq_enveloped, axis=axis)).real

    if discard_half:
        slices = [slice(None)] * len(shape)
        slices[axis] = slice(0, steps)
        time_domain_noise = time_domain_noise[tuple(slices)]

    # Save the noise array if required
    if save and save_name:
        torch.save(save_name, time_domain_noise)

    return time_domain_noise
