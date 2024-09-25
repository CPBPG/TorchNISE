"""
This module provides functions for averaging populations and estimating lifetimes
of quantum states using various methods.
"""
import torch
from torch.linalg import matrix_exp
import numpy as np
from scipy.optimize import dual_annealing
from torchnise.pytorch_utility import (
    golden_section_search,
    matrix_logh,
    batch_trace
    )

def averaging(population, averaging_type, lifetimes=None, step=None,
              coherence=None, weight=None):
    """
    Average populations using various methods.

    Args:
        population (torch.Tensor): Population tensor.
        averaging_type (str):   Type of averaging to perform. Options are 
            "standard", "boltzmann", "interpolated".
        lifetimes (torch.Tensor, optional): Lifetimes of the states, required 
            for "interpolated" averaging.
        step (float, optional): Time step size, required for "interpolated" 
            averaging.
        coherence (torch.Tensor, optional): Coherence matrix, needed for 
            "boltzmann" and "interpolated".
        weight (torch.Tensor, optional): Weights for averaging.

    Returns:
        tuple: (torch.Tensor, torch.Tensor) - Averaged population and 
            coherence.
    """
    averaging_types = ["standard", "boltzmann", "interpolated"]

    if averaging_type.lower() not in averaging_types:
        raise NotImplementedError(f"""{averaging_type} not implemented; only
                                  {averaging_types} are available""")

    if averaging_type.lower() in ["interpolated", "boltzmann"]:
        assert coherence is not None, (
            """Coherence matrix is required for 'interpolated' and 
               'boltzmann' averaging types."""
        )
        if averaging_type.lower() == "interpolated":
            assert lifetimes is not None, """"Lifetimes are required for
                                              'interpolated' averaging."""
            assert step is not None, """Time step size is required for
                                         'interpolated' averaging."""
    if weight is None:
        weight = 1
        weight_coherence = 1
    else:
        weight, weight_coherence = reshape_weights(weight, population,
                                                   coherence)

    meanres_orig = population_return = torch.mean(population * weight, dim=0)

    if coherence is None:
        meanres_coherence = None
    else:
        meanres_coherence = torch.mean(coherence * weight_coherence, dim=0)

    if averaging_type.lower() == "standard":
        population_return = meanres_orig
        coherence_return = meanres_coherence
    else:
        logres_matrix = torch.mean(matrix_logh(coherence) * weight_coherence,
                                   dim=0)
        meanexp_matrix = matrix_exp(logres_matrix)
        normalization= batch_trace(meanexp_matrix).unsqueeze(1).unsqueeze(1)
        meanres_matrix = meanexp_matrix / normalization
        meanres_matrix[0] = meanres_coherence[0]
        coherence_return = meanres_matrix

        if averaging_type.lower() == "boltzmann":
            population_return = torch.diagonal(meanres_matrix, dim1=-2,
                                               dim2=-1)
        else:
            population_return = blend_and_normalize_populations(
                meanres_orig, torch.diagonal(meanres_matrix, dim1=-2, dim2=-1),
                lifetimes, step)

    return population_return, coherence_return


def estimate_lifetime(u_tensor, delta_t, method="oscillatory_fit_mae"):
    """
    Estimate lifetimes of quantum states using various fitting methods.

    Args:
        u_tensor (torch.Tensor): Time evolution operator with dimensions 
            (realizations, timesteps, n_sites, n_sites).
        delta_t (float): Time step size.
        method (str, optional): Method to use for lifetime estimation. 
            Options are "oscillatory_fit_mae", "oscillatory_fit_mse", 
            "simple_fit", "reverse_cummax", "simple_fit_mae". Default is 
            "oscillatory_fit_mae".

    Returns:
        torch.Tensor: Estimated lifetimes of each state.
    """
    n = u_tensor.shape[2]
    timesteps = u_tensor.shape[1]
    
    population = torch.zeros((timesteps,n), device=population.device, dtype=torch.float)
    for i in range(n):
        population[:,i] = torch.mean(torch.abs(u_tensor[:, :, i, i]) ** 2,
                                dim=0).real
        
    lifetimes=estimate_lifetime_population(population,delta_t,method=method)
    return lifetimes


def estimate_lifetime_population(full_population, delta_t, method="oscillatory_fit_mae",equilib=None):
    """
    Estimate lifetimes of quantum states using various fitting methods.

    Args:
        u_tensor (torch.Tensor): Time evolution operator with dimensions 
            (realizations, timesteps, n_sites, n_sites).
        delta_t (float): Time step size.
        method (str, optional): Method to use for lifetime estimation. 
            Options are "oscillatory_fit_mae", "oscillatory_fit_mse", 
            "simple_fit", "reverse_cummax", "simple_fit_mae". Default is 
            "oscillatory_fit_mae".

    Returns:
        torch.Tensor: Estimated lifetimes of each state.
    """
    n = full_population.shape[-1]
    timesteps = full_population.shape[0]
    time_array = torch.arange(timesteps, device=full_population.device,
                              dtype=torch.float) * delta_t
    lifetimes = torch.zeros(n, device=full_population.device, dtype=torch.float)
    if equilib==None:
        equilib=torch.ones(n)
        equilib= equilib*1/n

    # Helper function outside the loop
    def get_func(method, equilib_i, time_array, population):
        if method.lower() == "simple_fit":
            return lambda tau: objective(tau, equilib_i, time_array, population)
        if method.lower() == "reverse_cummax":
            return lambda tau: objective_reverse_cummax(tau, equilib_i, time_array,
                                                        population)
        if method.lower() == "simple_fit_mae":
            return lambda tau: objective_mae(tau, equilib_i, time_array, population)
        if method.lower() == "oscillatory_fit_mae":
            return lambda tau: objective_oscil_mae(tau, equilib_i, time_array,
                                                   population)
        if method.lower() == "oscillatory_fit_mse":
            return lambda tau: objective_oscil_mse(tau, equilib_i, time_array,
                                                   population)
        raise ValueError(f""""
                         Unknown method for litetime interpolation:
                          {method}, must be one of "simple_fit",
                          "reverse_cummax", "simple_fit_mae",
                          "oscillatory_fit_mae", "oscillatory_fit_mse"
                         """)

    for i in range(n):
        population=full_population[:,i]
        tolerance = 0.1 * delta_t
        
        # Get the appropriate function
        func = get_func(method, equilib[i], time_array, population)

        if method.lower() in ["simple_fit", "reverse_cummax",
                              "simple_fit_mae"]:
            tau_opt = golden_section_search(func, delta_t,
                                            100 * timesteps * delta_t,
                                            tolerance)
        elif method.lower() in ["oscillatory_fit_mae", "oscillatory_fit_mse"]:
            func_init = get_func("reverse_cummax", n, time_array, population)
            tau_0 = golden_section_search(func_init, delta_t,
                                          100 * timesteps * delta_t, tolerance)
            x0 = [tau_0, tau_0, 0.5]
            lw = [delta_t, delta_t, 0]
            up = [100 * timesteps * delta_t, 0.5 * timesteps * delta_t, 1]

            if method == "oscillatory_fit_mae":
                func = objective_oscil_mae
            else:
                func = objective_oscil_mse
            res = dual_annealing(
                func, bounds=list(zip(lw, up)), args=(n, time_array,
                                                      population),
                minimizer_kwargs={"method": "BFGS"}, x0=x0, maxiter=500
            )
            tau_opt, _, _ = res.x
        else:
            raise ValueError(f""""
                             Unknown method for litetime interpolation:
                              {method}, must be one of "simple_fit",
                              "reverse_cummax", "simple_fit_mae",
                              "oscillatory_fit_mae", "oscillatory_fit_mse"
                             """)

        lifetimes[i] = torch.tensor(tau_opt)

    return lifetimes


def blend_and_normalize_populations(pop1, pop2, lifetimes, delta_t):
    """
    Blend and normalize populations based on lifetimes.

    Args:
        pop1 (torch.Tensor): Initial population tensor.
        pop2 (torch.Tensor): Final population tensor to blend towards.
        lifetimes (torch.Tensor): Lifetimes of the states.
        delta_t (float): Time step size.

    Returns:
        torch.Tensor: Normalized blended population tensor.
    """
    timesteps, sites = pop1.shape
    time_array = torch.arange(timesteps, dtype=torch.float) * delta_t
    blended_population = torch.zeros_like(pop1)

    for site in range(sites):
        lifetime = lifetimes[site]
        blending_factor = torch.exp(-time_array / lifetime)
        blended_population[:, site] = (
            blending_factor * pop1[:, site] +
            (1 - blending_factor) * pop2[:, site]
        )

    total_population = torch.sum(blended_population, dim=1, keepdim=True)
    normalized_population = blended_population / total_population

    return normalized_population


def reshape_weights(weight, population, coherence):
    """
    Reshape weights for averaging.

    Args:
        weight (torch.Tensor): Weight tensor.
        population (torch.Tensor): Population tensor.
        coherence (torch.Tensor, optional): Coherence matrix.

    Returns:
        tuple: (torch.Tensor, torch.Tensor) - Reshaped weights for population
            and coherence.
    """
    weight = weight / torch.sum(weight) * len(weight)

    if coherence is not None:
        weight_shape_coherence = list(coherence.shape)
        for i in range(1, len(weight_shape_coherence)):
            weight_shape_coherence[i] = 1
        weight_coherence = weight.reshape(weight_shape_coherence)
    else:
        weight_coherence = 1

    weight_shape = list(population.shape)
    for i in range(1, len(weight_shape)):
        weight_shape[i] = 1

    weight_reshaped = weight.reshape(weight_shape)
    return weight_reshaped, weight_coherence


def objective_oscil_mae(tau_oscscale_oscstrength, equilib_i, time_array,
                        population):
    """
    Objective function using Mean Absolute Error (MAE) for fitting
    oscillatory decays.

    Args:
        tau_oscscale_oscstrength (tuple): Tuple containing tau, osc_scale,
            and osc_strength parameters.
        n (int): Number of states.
        time_array (torch.Tensor): Array of time steps.
        population (torch.Tensor): Population tensor for the state.

    Returns:
        float: Mean absolute error (MAE) between the population and the fit.
    """
    tau, osc_scale, osc_strength = tau_oscscale_oscstrength
    amplitude = osc_strength
    vert_shift = 1 - osc_strength
    fit = (
        (1 - equilib_i) * torch.exp(-time_array / tau) *
        (amplitude * torch.cos(2 * np.pi * time_array / osc_scale) +
         vert_shift) + equilib_i
    )
    mae = torch.mean(torch.abs((population[1:] - fit[1:])))
    return mae.item()


def objective_oscil_mse(tau_oscscale_oscstrength, equilib_i, time_array,
                        population):
    """
    Objective function using Mean Squared Error (MSE) for fitting oscillatory
    decays.

    Args:
        tau_oscscale_oscstrength (tuple): Tuple containing tau, osc_scale,
            and osc_strength parameters.
        n (int): Number of states.
        time_array (torch.Tensor): Array of time steps.
        population (torch.Tensor): Population tensor for the state.

    Returns:
        float: Mean squared error (MSE) between the population and the fit.
    """
    tau, osc_scale, osc_strength = tau_oscscale_oscstrength
    amplitude = osc_strength
    vert_shift = 1 - osc_strength
    fit = (
        (1 - equilib_i) * torch.exp(-time_array / tau) *
        (amplitude * torch.cos(2 * np.pi * time_array / osc_scale) +
         vert_shift) + equilib_i
    )
    mse = torch.mean((population[1:] - fit[1:]) ** 2)
    return mse.item()


def objective_mae(tau, equilib_i, time_array, population):
    """
    Objective function using Mean Absolute Error (MAE) for fitting decays.

    Args:
        tau (float): Decay constant.
        n (int): Number of states.
        time_array (torch.Tensor): Array of time steps.
        population (torch.Tensor): Population tensor for the state.

    Returns:
        float: Mean absolute error (MAE) between the population and the fit.
    """
    fit = (1 - equilib_i) * torch.exp(-time_array / tau) + equilib_i
    mae = torch.mean(torch.abs((population[1:] - fit[1:])))
    return mae.item()


def objective(tau, equilib_i, time_array, population):
    """
    Objective function using Mean Squared Error (MSE) for fitting decays.

    Args:
        tau (float): Decay constant.
        n (int): Number of states.
        time_array (torch.Tensor): Array of time steps.
        population (torch.Tensor): Population tensor for the state.

    Returns:
        float: Mean squared error (MSE) between the population and the fit.
    """
    fit = (1 - equilib_i) * torch.exp(-time_array / tau) + equilib_i
    mse = torch.mean((population[1:] - fit[1:]) ** 2)
    return mse.item()


def objective_reverse_cummax(tau, equilib_i, time_array, population):
    """
    Objective function using reverse cumulative maximum for fitting decays.

    Args:
        tau (float): Decay constant.
        n (int): Number of states.
        time_array (torch.Tensor): Array of time steps.
        population (torch.Tensor): Population tensor for the state.

    Returns:
        float: Weighted mean squared error (MSE) between the population and
            the fit.
    """
    fit = (1 - equilib_i) * torch.exp(-time_array / tau) + equilib_i

    reverse_cummax, _ = torch.flip(population, [0]).cummax(dim=0)
    reverse_cummax = torch.flip(reverse_cummax, [0])
    selected_values = population == reverse_cummax

    times = time_array[selected_values]
    weights = times[1:] - times[:-1]

    mse = torch.mean(((population[selected_values][1:] -
                       fit[selected_values][1:]) * weights) ** 2)
    return mse.item()
