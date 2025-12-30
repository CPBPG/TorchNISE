"""
This file contains techniques for getting spectral densities from noise
"""

import warnings
import numpy as np
import torch
from scipy.optimize import nnls
from torchnise import units


def nnls_pyytorch_scipy(A, b):
    """
    Solve the non-negative least squares problem for PyTorch Tensors using scipy.

    Args:
        A (torch.Tensor): The input matrix A.
        b (torch.Tensor): The input vector b.

    Returns:
        torch.Tensor: Solution vector x_nnls as a PyTorch tensor.
    """
    A_np = A.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    # Solve the NNLS problem
    x_nnls, _ = nnls(A_np, b_np)

    # Convert the result back to a PyTorch tensor
    x_nnls_torch = torch.from_numpy(x_nnls).to(A.device).to(A.dtype)
    return x_nnls_torch


def autocorrelation(noise, i, n):
    """
    Calculate the autocorrelation function for a given noise matrix.

    Args:
        noise (np.ndarray): The input noise matrix.
        i (int): The current index for autocorrelation calculation.
        n (int): The total number of timesteps.

    Returns:
        np.ndarray: The autocorrelation value for the given index i.
    """
    cor1 = noise[:, : n - i]
    cor2 = noise[:, i:]
    res = cor1 * cor2
    c = np.mean(res, axis=1)
    return c


def ccalc(noise, n, reals):
    """
    Calculate the autocorrelation matrix for the entire noise dataset using FFT.
    """
    # Pad to next power of 2 for efficiency
    pad_n = 2**((2 * n - 1).bit_length())
    noise_fft = np.fft.rfft(noise, n=pad_n, axis=1)
    res_fft = noise_fft * np.conj(noise_fft)
    res = np.fft.irfft(res_fft, n=pad_n, axis=1)[:, :n]
    
    # Biased normalization by n (standard for spectral density reconstruction)
    # but to match common practice for unbiased, we could divide by (n - arr)
    # however, sd_reconstruct_fft expects normalized auto.
    return res / n  # returns reals x n


def expval_auto(noise, n, reals):
    """
    Calculate the expectation value of the autocorrelation function.

    Args:
        noise (np.ndarray): The input noise matrix.
        n (int): The total number of timesteps.
        reals (int): The number of realizations.

    Returns:
        np.ndarray: The expectation value of the autocorrelation function.
    """
    summation = ccalc(noise, n, reals)
    return np.mean(summation, axis=0)


def get_auto(noise):
    """
    Get the autocorrelation function for a given noise dataset.

    Args:
        noise (np.ndarray): The input noise matrix.

    Returns:
        np.ndarray: The autocorrelation function.
    """
    n = len(noise[0, :])
    reals = len(noise[:, 0])
    c = expval_auto(noise, n, reals)
    auto = c[: int(n // 2)]  # only trusting half the values employed
    return auto


def cross_correlation(noise1, noise2, i, n):
    """
    Calculate the cross-correlation function for two noise matrices.
    """
    cor1 = noise1[:, : n - i]
    cor2 = noise2[:, i:]
    res = cor1 * cor2
    c = np.mean(res, axis=1)
    return c


def ccalc_cross(noise1, noise2, n, reals):
    """
    Calculate the cross-correlation matrix for two noise datasets using FFT.
    """
    pad_n = 2**((2 * n - 1).bit_length())
    fft1 = np.fft.rfft(noise1, n=pad_n, axis=1)
    fft2 = np.fft.rfft(noise2, n=pad_n, axis=1)
    res_fft = fft1 * np.conj(fft2)
    res = np.fft.irfft(res_fft, n=pad_n, axis=1)[:, :n]
    return res / n


def expval_cross(noise1, noise2, n, reals):
    """
    Calculate the expectation value of the cross-correlation function.
    """
    summation = ccalc_cross(noise1, noise2, n, reals)
    return np.mean(summation, axis=0)


def get_cross(noise1, noise2):
    """
    Get the cross-correlation function for two noise datasets.
    """
    n = len(noise1[0, :])
    reals = len(noise1[:, 0])
    c = expval_cross(noise1, noise2, n, reals)
    cross = c[: int(n // 2)]
    return cross


def sd_reconstruct_fft(
    auto,
    dt,
    temperature,
    min_w=None,
    max_w=None,
    damping_type=None,
    cutoff=None,
    rescale=False,
):
    """
    Reconstruct the spectral density using FFT from the autocorrelation function.

    Args:
        auto (np.ndarray): Autocorrelation function.
        dt (float): Time step between autocorrelation points.
        temperature (float): Temperature.
        min_w (float, optional): Minimum frequency to consider. Defaults to None.
        max_w (float, optional): Maximum frequency to consider. Defaults to None.
        damping_type (str, optional): Type of damping to apply ('step', 'gauss',
            'exp'). Defaults to None.
        cutoff (float, optional): Cutoff for damping. Defaults to None.
        rescale (bool, optional): If True, rescale the autocorrelation
            function. Defaults to False.

    Returns:
        tuple: (np.ndarray, np.ndarray, np.ndarray) - Reconstructed spectral
            density, frequency axis, and damped autocorrelation.
    """
    n_cut = len(auto)
    dw_t = 2 * np.pi / (2 * n_cut * dt * units.T_UNIT)
    max_w = max_w if max_w is not None else n_cut * dw_t
    min_w = min_w if min_w is not None else 0

    if max_w > n_cut * dw_t:
        print(
            f"Warning maxW {max_w} bigger than maximum computable value"
            f"{n_cut * dw_t}"
        )

    t_axis = dt * np.arange(0, n_cut)
    damping = np.ones_like(auto)

    if damping_type:
        if damping_type.lower() == "step":
            damping[int(cutoff // dt) :] = 0
        elif damping_type.lower() == "gauss":
            damping = np.exp(-((t_axis / cutoff) ** 2))
        elif damping_type.lower() == "exp":
            damping = np.exp(-t_axis / cutoff)
        else:
            raise NotImplementedError(
                f"Damping {damping_type} not available. "
                "Must be one of 'step', 'gauss', or 'exp'"
            )

    auto_damp = auto * damping

    if rescale:
        auto_damp = auto_damp * np.mean(np.abs(auto)) / np.mean(np.abs(auto_damp))

    # Calculation of spectral density
    # creating dw in units of 1/fs. denominator: 2*N_cut = N
    dw_t = 2 * np.pi / (2 * n_cut * dt)
    full_w_t = np.arange(0, n_cut * dw_t, dw_t)
    # find the closest value to minW and maxW from the fourier axis
    min_w_index = np.argmin(np.abs(full_w_t - min_w))
    max_w_index = np.argmin(np.abs(full_w_t - max_w))
    # array of frequencies to use in 1/fs. Max frequency is not necessarily this.
    # Only ought to be less than nyquist frequency: dw*N/2
    w_t = full_w_t[min_w_index : max_w_index + 1]
    x_axis = units.HBAR * w_t
    # calculate DCT with FFT
    reverse_auto = np.flip(auto_damp[1:-1])
    concat_auto = np.concatenate((auto_damp, reverse_auto))
    j_new = (
        x_axis
        * np.fft.fft(concat_auto)[: len(x_axis)].real
        * dt
        / (units.HBAR * 2 * np.pi * units.K * temperature)
    )
    return j_new, x_axis, auto_damp


def tv_norm_2d(lambda_ij):
    """
    Calculate the total variation norm across both dimensions.

    Args:
        lambda_ij (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Total variation norm.
    """
    tv = torch.sum(torch.abs(lambda_ij[:, :-1] - lambda_ij[:, 1:])) + torch.sum(
        torch.abs(lambda_ij[:-1, :] - lambda_ij[1:, :])
    )
    return tv


def objective_function(
    lambda_ij,
    A,
    b,
    sparcity_penalty,
    l1_norm_penalty,
    solution_penalty,
    negative_penalty,
    ljnorm_penalty,
    j,
):
    """
    Objective function with TV norm, L1 norm, and penalties for constraints.
        Used for Superresolution

    Args:
        lambda_ij (torch.Tensor): Current solution tensor.
        A (torch.Tensor): Matrix for the linear system.
        b (torch.Tensor): Target vector.
        sparcity_penalty (float): Penalty term for sparsity in the solution.
        l1_norm_penalty (float): L1 norm penalty for regularization.
        solution_penalty (float): Penalty for the solution norm.
        negative_penalty (float): Penalty for negative peaks.
        ljnorm_penalty (float): L_j norm penalty for regularization.
        j (float): Exponent for the L_j norm.

    Returns:
        torch.Tensor: Value of the objective function.
    """
    penalty = solution_penalty * 0.5 * torch.norm(A @ lambda_ij.flatten() - b) ** 2
    result = (
        objective_function_no_penalty(
            lambda_ij,
            sparcity_penalty,
            l1_norm_penalty,
            negative_penalty,
            ljnorm_penalty,
            j,
        )
        + penalty
    )
    return result


def objective_function_no_penalty(
    lambda_ij, sparcity_penalty, l1_norm_penalty, negative_penalty, ljnorm_penalty, j
):
    """
    Objective function without the solution penalty.

    Args:
        lambda_ij (torch.Tensor): Current solution tensor.
        sparcity_penalty (float): Penalty term for sparsity in the solution.
        l1_norm_penalty (float): L1 norm penalty for regularization.
        negative_penalty (float): Penalty for negative peaks.
        ljnorm_penalty (float): L_j norm penalty for regularization.
        j (float): Exponent for the L_j norm.

    Returns:
        torch.Tensor: Value of the objective function without the solution penalty.
    """
    tv = tv_norm_2d(lambda_ij)
    l1_norm = torch.sum(torch.abs(lambda_ij))
    lj_norm = torch.sum(torch.abs(lambda_ij) ** j)
    negative_penalty_value = -torch.sum(lambda_ij) + torch.sum(torch.abs(lambda_ij))
    result = (
        sparcity_penalty * tv
        + l1_norm_penalty * l1_norm
        + ljnorm_penalty * lj_norm
        + negative_penalty * negative_penalty_value
    )
    return result


def optimize_lambda(
    A,
    b,
    sparcity_penalty,
    l1_norm_penalty,
    solution_penalty,
    negative_penalty,
    ljnorm_penalty,
    j,
    eta,
    max_iter=1000,
    tol=1e-6,
    lr=0.01,
    device="cuda",
    initial_guess=None,
    verbose=False,
):
    """
    Optimization loop using PyTorch.

    Args:
        A (torch.Tensor): Matrix for the linear system.
        b (torch.Tensor): Target vector.
        sparcity_penalty (float): Penalty term for sparsity in the solution.
        l1_norm_penalty (float): L1 norm penalty for regularization.
        solution_penalty (float): Penalty for the solution norm.
        negative_penalty (float): Penalty for negative peaks.
        ljnorm_penalty (float): L_j norm penalty for regularization.
        j (float): Exponent for the L_j norm.
        eta (float): Regularization term for optimization.
        max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-6.
        lr (float, optional): Learning rate for the optimization algorithm.
            Defaults to 0.01.
        device (str, optional): Device for computation ('cuda' or 'cpu').
            Defaults to 'cuda'.
        initial_guess (torch.Tensor, optional): Initial guess for the
            optimization. Defaults to None.
        verbose (bool, optional): decide if information should be printed

    Returns:
        torch.Tensor: Optimized solution tensor.
    """
    num_k, num_i, num_j = A.shape
    A = A.reshape(num_k, num_i * num_j)

    # Initialize lambda_ij as a PyTorch tensor with gradients
    def A_transpose(y):
        return A.T @ y

    # Initialize lambda_ij as a PyTorch tensor with gradients
    lambda_ij = A_transpose(b) if initial_guess is None else initial_guess
    lambda_ij = lambda_ij.reshape(num_i, num_j)
    lambda_ij.requires_grad = True

    optimizer = torch.optim.LBFGS([lambda_ij], lr=lr, line_search_fn="strong_wolfe")

    min_objective_value = torch.inf  # initialize best objective

    return_lambda = lambda_ij
    for _ in range(max_iter):
        optimizer.zero_grad()  # Clear previous gradients

        # Compute the objective function
        obj_value = objective_function(
            lambda_ij,
            A,
            b,
            sparcity_penalty,
            l1_norm_penalty,
            solution_penalty,
            negative_penalty,
            ljnorm_penalty,
            j,
        )
        obj_value_no_penalty = objective_function_no_penalty(
            lambda_ij,
            sparcity_penalty,
            l1_norm_penalty,
            negative_penalty,
            ljnorm_penalty,
            j,
        )
        obj_value.backward()  # Perform a backward pass to compute gradients

        def closure():
            optimizer.zero_grad()
            # Compute the objective function
            obj_value = objective_function(
                lambda_ij,
                A,
                b,
                sparcity_penalty,
                l1_norm_penalty,
                solution_penalty,
                negative_penalty,
                ljnorm_penalty,
                j,
            )
            obj_value.backward()  # Perform a backward pass to compute gradients
            return obj_value

        optimizer.step(closure)  # LBFGS step

        # Calculate the constraint violation
        with torch.no_grad():
            constraint_violation = torch.norm(A @ lambda_ij.flatten() - b).item()

        if iter % 10 == 0 and verbose:
            print(
                f"Iteration {iter}: Objective value = with pen"
                f" {obj_value.item():.6f} without pen "
                f"{obj_value_no_penalty.item():.6f}, Constraint violation = "
                f"{constraint_violation> eta}, {constraint_violation:.6f}"
            )
        if obj_value.item() < min_objective_value:
            min_objective_value = obj_value.item()
            no_improvement_iters = 0
            return_lambda = lambda_ij
        else:
            no_improvement_iters += 1
        # Convergence check
        if obj_value_no_penalty.item() < tol and constraint_violation < eta:
            if verbose:
                print("Converged")
            break
        if no_improvement_iters > 20:
            if verbose:
                print("no improvement for 20 iters")
            break

    return return_lambda


def optimize_lambda_nnls(
    A, b, initial_guess=None, max_iter=1000, lr=0.01, verbose=False
):
    """
    Perform non-negative least squares optimization using PyTorch.

    Args:
        A (torch.Tensor): Matrix for the linear system.
        b (torch.Tensor): Target vector.
        initial_guess (torch.Tensor, optional): Initial guess for the
            optimization. Defaults to None.
        max_iter (int, optional): Maximum number of iterations.
            Defaults to 1000.
        lr (float, optional): Learning rate for the optimization algorithm.
            Defaults to 0.01.
        erbose (bool, optional): decide if information should be printed

    Returns:
        torch.Tensor: Optimized solution tensor.
    """

    def A_transpose(y):
        return A.T @ y

    lambda_ij = A_transpose(b) if initial_guess is None else initial_guess
    lambda_ij = lambda_ij.detach().clone()
    lambda_ij.requires_grad = True
    return_lambda = lambda_ij

    optimizer = torch.optim.LBFGS([lambda_ij], lr=lr, line_search_fn="strong_wolfe")
    min_loss = torch.inf
    no_improvement_iters = 0

    def closure():
        optimizer.zero_grad()
        loss = torch.norm(A @ torch.abs(lambda_ij) - b) ** 2
        loss.backward()
        return loss

    for i in range(max_iter):
        optimizer.step(closure)
        loss = closure().item()

        if i % 10 == 0 and verbose:
            print(f"Step {i}, Loss: {loss}")

        if loss < min_loss:
            min_loss = loss
            no_improvement_iters = 0
            return_lambda = lambda_ij
        else:
            no_improvement_iters += 1

        if no_improvement_iters > 20:
            if verbose:
                print("No improvement for 20 iterations")
            break

    return torch.abs(return_lambda).detach()


def adjust_tensor_length(a, l):
    """
    Adjust the length of a tensor by trimming or padding with zeros.

    Args:
        a (torch.Tensor): Input tensor.
        l (int): Desired length of the output tensor.

    Returns:
        torch.Tensor: Adjusted tensor of length l.
    """
    if len(a) == l:
        return a
    if len(a) > l:
        # Trimming the tensor if it's longer than the desired length
        a = a[:l]
    else:
        # Padding the tensor with zeros if it's shorter than the desired length
        a = torch.cat([a, torch.zeros(l - len(a), device=a.device)])

    return a


def ensure_tensor_on_device(array, device="cuda", dtype=torch.float):
    """
    Ensure the input is a PyTorch tensor on the specified device.
    And move it if its not.

    Args:
        array (np.ndarray or torch.Tensor): Input array or tensor.
        device (str, optional): Desired device ('cuda' or 'cpu').
            Defaults to 'cuda'.
        dtype (torch.dtype, optional): Desired data type.
            Defaults to torch.float.

    Returns:
        torch.Tensor: Tensor on the specified device with the desired data type.
    """
    if isinstance(array, np.ndarray):
        # Convert numpy array to PyTorch tensor and move to the specified device
        return torch.from_numpy(array).to(device).to(dtype)
    if isinstance(array, torch.Tensor):
        # Ensure the tensor is on the correct device
        return array.to(device).to(dtype)
    raise TypeError("Input must be a numpy array or a PyTorch tensor.")


def sd_reconstruct_superresolution(
    auto,
    dt,
    temperature,
    sparcity_penalty=1,
    l1_norm_penalty=1,
    solution_penalty=10000,
    negative_penalty=1,
    ljnorm_penalty=0,
    j=0.5,
    lr=0.01,
    max_iter=1000,
    eta=1e-7,
    tol=1e-7,
    device="cuda",
    cutoff=None,
    frequencies=None,
    linewidths=None,
    sample_frequencies=None,
    top_n=False,
    top_tresh=False,
    second_optimization=False,
    chunk_memory=1e9,
    auto_length_debias=None,
    auto_length_return=None,
):
    """
    Reconstruct the super-resolution spectral density from the autocorrelation
    function.

    Parameters:
        auto (torch.Tensor): Autocorrelation function.
        dt (float): Time step between autocorrelation points.
        temperature (float): Temperature.
        sparcity_penalty (float): Penalty term for sparsity in the solution.
        l1_norm_penalty (float): L1 norm penalty for regularization.
        solution_penalty (float): Penalty for the solution norm.
        negative_penalty (float): Penalty for negative peaks.
        ljnorm_penalty (float): L_j norm penalty for regularization.
        j (float): j determining the L_j norm.
        lr (float): Learning rate for the optimization algorithm.
        max_iter (int): Maximum number of iterations for the optimization.
        eta (float): Regularization term for optimization.
        tol (float): Tolerance for convergence in the optimization.
        device (str): Device for computation ('cuda' or 'cpu').
        cutoff (float, optional): Cutoff for damping. Defaults to None.
        frequencies (torch.tensor, optional): Frequencies for peaks that should
            be included in the optimization, otherwise default values are used
        linewidths (torch.tensor, optional): Linewidths for peaks that should be
            included in the optimization, otherwise default values are used
        sample_frequencies (torch.Tensor, optional): Frequencies for sampling
            the spectral density. Defaults to None.
        top_n (int): If not False, only the top n coefficients are used.
            Defaults to False.
        top_tresh (float): Alternative to top_n chooses all coefficients above
            a treshold
        second_optimization (bool): If True, a second optimization step is
            performed using top n coefficients. Defaults to False.
        chunk_memory (float): The maximum amount of memory (in bytes) to use
            for each chunk. Defaults to 1GB.
        auto_length_debias (float): if not False: length of autocorrelation in
            fs for second optimizazion fitting, will be zero padded or cut
        auto_length_return (float): if not False: length of the returned
            autocorrelation inn fs

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Reconstructed spectral
            density, sampled frequencies, and super-resolved autocorrelation
            function.
    """
    auto = ensure_tensor_on_device(auto, device=device)
    auto_very_orig = auto.clone()
    if auto_length_return:
        auto_very_orig = adjust_tensor_length(auto, auto_length_return // dt)
    if auto_length_debias:
        auto = adjust_tensor_length(auto, auto_length_debias // dt)

    n = len(auto)
    n_cut = int(cutoff / dt) if cutoff else n // 2

    t_axis = dt * torch.arange(0, n_cut, device=device)
    t_axis_full = dt * torch.arange(0, n, device=device)
    t_axis_very_orig = dt * torch.arange(0, len(auto_very_orig), device=device)
    if frequencies is None:
        frequencies = torch.arange(0, 0.2, 0.00005, device=device) / units.HBAR
    if linewidths is None:
        linewidths = torch.arange(1, 200, 0.5, device=device) / units.HBAR

    linewidths_term = torch.exp(-linewidths[None, :] * t_axis[:, None])
    frequency_term = torch.cos(frequencies[None, :] * t_axis[:, None])

    A = linewidths_term[:, :, None] * frequency_term[:, None, :]

    auto_orig = auto.clone()
    auto = auto[:n_cut]

    lambda_ij = optimize_lambda(
        A,
        auto,
        sparcity_penalty,
        l1_norm_penalty,
        solution_penalty,
        negative_penalty,
        ljnorm_penalty,
        j,
        eta,
        max_iter=max_iter,
        tol=tol,
        lr=lr,
        device=device,
    )
    if top_n:
        if top_n > len(linewidths) * len(frequencies):
            warnings.warn(
                "top_n larger than total elements," " using all available coefficients."
            )
            top_n = len(linewidths) * len(frequencies)
            # removed abs since we don't want negative peaks anyway
        _, indices = torch.topk(lambda_ij.flatten(), top_n)

        if second_optimization:
            original_indices = torch.unravel_index(indices, lambda_ij.shape)
            A_new = torch.zeros((len(t_axis_full), top_n), device=device)
            for k in range(top_n):
                i = original_indices[0][k]
                j = original_indices[1][k]
                A_new[:, k] = torch.exp(-linewidths[i] * t_axis_full) * torch.cos(
                    frequencies[j] * t_axis_full
                )

            lambda_ij_new = optimize_lambda_nnls(
                A_new, auto_orig, initial_guess=lambda_ij.flatten()[indices]
            )
            lambda_ij_debias = torch.zeros_like(lambda_ij, device=device)
            lambda_ij_debias.flatten()[indices] = lambda_ij_new
        else:
            lambda_zero = torch.zeros_like(lambda_ij, device=device)
            lambda_zero.flatten()[indices] = lambda_ij.flatten()[indices]
            lambda_ij = lambda_zero
    elif top_tresh:
        # removed abs since we don't want negative peaks anyway
        indices = torch.where(lambda_ij.flatten() > top_tresh)[0]
        top_n = len(lambda_ij.flatten()[indices])
        print(torch.max(lambda_ij), torch.min(lambda_ij))
        print(f"selected {top_n} values")
        if second_optimization:
            original_indices = torch.unravel_index(indices, lambda_ij.shape)
            A_new = torch.zeros((len(t_axis_full), top_n), device=device)
            for k in range(top_n):
                i = original_indices[0][k]
                j = original_indices[1][k]
                A_new[:, k] = torch.exp(-linewidths[i] * t_axis_full) * torch.cos(
                    frequencies[j] * t_axis_full
                )

            lambda_ij_new = optimize_lambda_nnls(
                A_new, auto_orig, initial_guess=lambda_ij.flatten()[indices]
            )
            lambda_ij_debias = torch.zeros_like(lambda_ij, device=device)
            lambda_ij_debias.flatten()[indices] = lambda_ij_new
        else:
            lambda_zero = torch.zeros_like(lambda_ij, device=device)
            lambda_zero.flatten()[indices] = lambda_ij.flatten()[indices]
            lambda_ij = lambda_zero

    if sample_frequencies is None:
        sample_frequencies = torch.arange(0, 0.2, 0.00001, device=device) / units.HBAR
    else:
        sample_frequencies = ensure_tensor_on_device(sample_frequencies, device=device)

    with torch.no_grad():
        j_new = torch.zeros_like(sample_frequencies)

        # Calculate the chunk size that would use less than the specified chunk_memory
        chunk_size = int(
            chunk_memory
            / (3 * len(frequencies) * len(linewidths) * auto.element_size())
        )
        chunk_size = max(chunk_size, 1)
        for i in range(0, sample_frequencies.shape[0], chunk_size):
            max_idx = min(i + chunk_size, sample_frequencies.shape[0])
            sample_freq_chunk = sample_frequencies[i:max_idx]
            temperature_frequencies_gamma = (
                sample_freq_chunk[:, None, None] * linewidths[None, :, None] * np.pi * 2
            )

            term1 = temperature_frequencies_gamma / (
                linewidths[None, :, None] ** 2
                + (sample_freq_chunk[:, None, None] - frequencies[None, None, :]) ** 2
            )
            term2 = temperature_frequencies_gamma / (
                linewidths[None, :, None] ** 2
                + (sample_freq_chunk[:, None, None] + frequencies[None, None, :]) ** 2
            )

            j_new[i:max_idx] = (lambda_ij[None, :, :] * (term1 + term2)).sum(dim=[1, 2])
        j_new = j_new.detach().cpu().numpy()

        auto_super = torch.zeros_like(auto_very_orig)
        for i in range(0, len(t_axis_full), chunk_size):
            max_idx = min(i + chunk_size, t_axis_full.shape[0])
            linewidths_term_chunk = torch.exp(
                -linewidths[None, :] * t_axis_full[i:max_idx, None]
            )
            frequency_term_chunk = torch.cos(
                frequencies[None, :] * t_axis_full[i:max_idx, None]
            )
            A_chunk = (
                linewidths_term_chunk[:, :, None] * frequency_term_chunk[:, None, :]
            ).reshape(max_idx - i, len(linewidths) * len(frequencies))
            auto_super[i:max_idx] = A_chunk @ lambda_ij.flatten()
        auto_super = auto_super.detach().cpu().numpy()

        if second_optimization:
            j_new_debias = torch.zeros_like(sample_frequencies)
            # auto_super_debias = torch.zeros_like(auto_very_orig)
            A_new = torch.zeros((len(t_axis_very_orig), top_n), device=device)
            for k in range(top_n):
                i = original_indices[0][k]
                j = original_indices[1][k]
                A_new[:, k] = torch.exp(-linewidths[i] * t_axis_very_orig) * torch.cos(
                    frequencies[j] * t_axis_very_orig
                )
            auto_super_debias = A_new @ lambda_ij_new.flatten()
            for k in range(top_n):
                i = original_indices[0][k]
                j = original_indices[1][k]
                temperature_frequencies_gamma = (
                    sample_frequencies * linewidths[i] * np.pi * 2
                )
                term1 = temperature_frequencies_gamma / (
                    linewidths[i] ** 2 + (sample_frequencies - frequencies[j]) ** 2
                )
                term2 = temperature_frequencies_gamma / (
                    linewidths[i] ** 2 + (sample_frequencies + frequencies[j]) ** 2
                )
                j_new_debias += lambda_ij_new[k] * (term1 + term2)
            j_new_debias = j_new_debias.detach().cpu().numpy()
            auto_super_debias = auto_super_debias.detach().cpu().numpy()
        else:
            j_new_debias = None
            auto_super_debias = None

        xaxis = (sample_frequencies * units.HBAR).detach().cpu().numpy()

        return j_new, xaxis, auto_super, j_new_debias, auto_super_debias
