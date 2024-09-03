import torch
import numpy as np
from scipy.optimize import dual_annealing
from torchnise.pytorch_utility import (
    golden_section_search,
    matrix_logh,
    batch_trace
    )

def averaging(population, averiging_type, lifetimes=None, step=None, coherence=None, weight=None):
    """
    Average populations using various methods.

    Args:
        population (torch.Tensor): Population tensor.
        averiging_type (str): Type of averaging to perform. Options are "standard", "boltzmann", "interpolated".
        lifetimes (torch.Tensor, optional): Lifetimes of the states, required for "interpolated" averaging.
        step (float, optional): Time step size, required for "interpolated" averaging.
        coherence (torch.Tensor, optional): Coherence matrix.
        weight (torch.Tensor, optional): Weights for averaging.

    Returns:
        tuple: (torch.Tensor, torch.Tensor) - Averaged population and coherence.
    """
    averaging_types = ["standard", "boltzmann", "interpolated"]
    
    if averiging_type.lower() not in averaging_types:
        raise NotImplementedError(f"{averiging_type} not implemented; only {averaging_types} are available")
    
    if averiging_type.lower() in ["interpolated", "boltzmann"]:
        assert coherence is not None, "Coherence matrix is required for 'interpolated' and 'boltzmann' averaging types."
        if averiging_type.lower() == "interpolated":
            assert lifetimes is not None, "Lifetimes are required for 'interpolated' averaging."
            assert step is not None, "Time step size is required for 'interpolated' averaging."
    
    if weight is None:
        weight = 1
        weight_coherence = 1
    else:
        weight, weight_coherence = reshape_weights(weight, population, coherence)
    
    meanres_orig = population_return = torch.mean(population * weight, dim=0)
    
    if coherence is None:
        meanres_coherence = None
    else:
        meanres_coherence = torch.mean(coherence * weight_coherence, dim=0)
    
    if averiging_type.lower() == "standard":
        population_return = meanres_orig
        coherence_return = meanres_coherence
    else:
        logres_matrix = torch.mean(matrix_logh(coherence) * weight_coherence, dim=0)
        meanExp_matrix = torch.linalg.matrix_exp(logres_matrix)
        meanres_matrix = meanExp_matrix / batch_trace(meanExp_matrix).unsqueeze(1).unsqueeze(1)
        meanres_matrix[0] = meanres_coherence[0]
        coherence_return = meanres_matrix
        
        if averiging_type.lower() == "boltzmann":
            population_return = torch.diagonal(meanres_matrix, dim1=-2, dim2=-1)
        else:
            population_return = blend_and_normalize_populations(meanres_orig, torch.diagonal(meanres_matrix, dim1=-2, dim2=-1), lifetimes, step)
    
    return population_return, coherence_return


def estimate_lifetime(U, delta_t, method="oscillatory_fit_mae"):
    """
    Estimate lifetimes of quantum states using various fitting methods.

    Args:
        U (torch.Tensor): Time evolution operator with dimensions (realizations, timesteps, n_sites, n_sites).
        delta_t (float): Time step size.
        method (str, optional): Method to use for lifetime estimation. Options are "oscillatory_fit_mae", 
                                "oscillatory_fit_mse", "simple_fit", "reverse_cummax", "simple_fit_mae". 
                                Default is "oscillatory_fit_mae".

    Returns:
        torch.Tensor: Estimated lifetimes of each state.

    """
    n = U.shape[2]
    timesteps = U.shape[1]
    time_array = torch.arange(timesteps, device=U.device, dtype=torch.float) * delta_t
    lifetimes = torch.zeros(n, device=U.device, dtype=torch.float)
    
    for i in range(n):
        population = torch.mean(torch.abs(U[:, :, i, i]) ** 2, dim=0).real
        tolerance = 0.1 * delta_t

        if method.lower() in ["simple_fit", "reverse_cummax", "simple_fit_mae"]:
            if method.lower() == "simple_fit":
                func = lambda tau: objective(tau, delta_t, n, time_array, population)
            elif method.lower() == "reverse_cummax":
                func = lambda tau: objective_reverse_cummax(tau, delta_t, n, time_array, population)
            elif method.lower() == "simple_fit_mae":
                func = lambda tau: objective_mae(tau,  delta_t, n, time_array, population)
            tau_opt = golden_section_search(func, delta_t, 100 * timesteps * delta_t, tolerance)

        elif method.lower() in ["oscillatory_fit_mae","oscillatory_fit_mse"]:
            func = lambda tau: objective_reverse_cummax(tau, delta_t, n, time_array, population)
            tau_0 = golden_section_search(func, delta_t, 100 * timesteps * delta_t, tolerance)
            x0 = [tau_0, tau_0, 0.5]
            lw = [delta_t, delta_t, 0]
            up = [100 * timesteps * delta_t, 0.5 * timesteps * delta_t, 1]

            if method == "oscillatory_fit_mae":
                func=objective_oscil_mae
            else:
                func=objective_oscil_mse
            res = dual_annealing(func, bounds=list(zip(lw, up)), args=(delta_t, n, time_array, population),
                                     minimizer_kwargs={"method": "BFGS"}, x0=x0, maxiter=500)
            tau_opt, osc_scale_opt, osc_strength_opt = res.x                

        lifetimes[i] = torch.tensor(tau_opt)
        #print(f"lifetime state {i+1}: {tau_opt} fs")

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
        blended_population[:, site] = blending_factor * pop1[:, site] + (1 - blending_factor) * pop2[:, site]

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
        tuple: (torch.Tensor, torch.Tensor) - Reshaped weights for population and coherence.
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

def objective_oscil_mae(tau_oscscale_oscstrength, delta_t, n, time_array,population):
    """
    Objective function using Mean Absolute Error (MAE) for fitting oscillatory decays.

    Args:
        tau_oscscale_oscstrength (tuple): Tuple containing tau, osc_scale, and osc_strength parameters.
        i (int): Index of the state to estimate the lifetime.
        delta_t (float): Time step size.
        n (int): Number of states.
        time_array (torch.Tensor): Array of time steps.
        U (torch.Tensor): Time evolution operator with dimensions (realizations, timesteps, n_sites, n_sites).

    Returns:
        float: Mean squared error (MSE) between the population and the fit.
    """
    tau, osc_scale, osc_strength = tau_oscscale_oscstrength
    amplitude = osc_strength
    vert_shift = (1 - osc_strength)
    fit = (1 - 1 / n) * torch.exp(-time_array / tau) * (amplitude * torch.cos(2 * np.pi * time_array / osc_scale) + vert_shift) + 1 / n
    mse = torch.mean(torch.abs((population[1:] - fit[1:])))
    return mse.item()

def objective_oscil_mse(tau_oscscale_oscstrength,  delta_t, n, time_array,population):
    """
    Objective function using Mean Squared Error (MAE) for fitting oscillatory decays.

    Args:
        tau_oscscale_oscstrength (tuple): Tuple containing tau, osc_scale, and osc_strength parameters.
        i (int): Index of the state to estimate the lifetime.
        delta_t (float): Time step size.
        n (int): Number of states.
        time_array (torch.Tensor): Array of time steps.
        U (torch.Tensor): Time evolution operator with dimensions (realizations, timesteps, n_sites, n_sites).

    Returns:
        float: Mean squared error (MSE) between the population and the fit.
    """
    tau, osc_scale, osc_strength = tau_oscscale_oscstrength
    amplitude = osc_strength
    vert_shift = (1 - osc_strength)
    fit = (1 - 1 / n) * torch.exp(-time_array / tau) * (amplitude * torch.cos(2 * np.pi * time_array / osc_scale) + vert_shift) + 1 / n
    mse = torch.mean((population[1:] - fit[1:])**2)
    return mse.item()

def objective_mae(tau,  delta_t, n, time_array,population):
    """
    Objective function using Mean Absolute Error (MAE) for fitting decays.

    Args:
        tau (float): Decay constant.
        i (int): Index of the state to estimate the lifetime.
        delta_t (float): Time step size.
        n (int): Number of states.
        time_array (torch.Tensor): Array of time steps.
        U (torch.Tensor): Time evolution operator with dimensions (realizations, timesteps, n_sites, n_sites).

    Returns:
        float: Mean absolute error (MAE) between the population and the fit.
    """
    fit = (1 - 1 / n) * torch.exp(-time_array / tau) + 1 / n
    mae = torch.mean(torch.abs((population[1:] - fit[1:])))
    return mae.item()

def objective(tau, delta_t, n, time_array,population):
    """
    Objective function using Mean Squared Error (MSE) for fitting decays.

    Args:
        tau (float): Decay constant.
        i (int): Index of the state to estimate the lifetime.
        delta_t (float): Time step size.
        n (int): Number of states.
        time_array (torch.Tensor): Array of time steps.
        U (torch.Tensor): Time evolution operator with dimensions (realizations, timesteps, n_sites, n_sites).

    Returns:
        float: Mean squared error (MSE) between the population and the fit.
    """
    fit = (1 - 1 / n) * torch.exp(-time_array / tau) + 1 / n
    mse = torch.mean(((population[1:] - fit[1:])) ** 2)
    return mse.item()

def objective_reverse_cummax(tau, delta_t, n, time_array, population):
    """
    Objective function using reverse cumulative maximum for fitting decays.

    Args:
        tau (float): Decay constant.
        i (int): Index of the state to estimate the lifetime.
        delta_t (float): Time step size.
        n (int): Number of states.
        time_array (torch.Tensor): Array of time steps.
        U (torch.Tensor): Time evolution operator with dimensions (realizations, timesteps, n_sites, n_sites).

    Returns:
        float: Weighted mean squared error (MSE) between the population and the fit.
    """
    fit = (1 - 1 / n) * torch.exp(-time_array / tau) + 1 / n

    
    reverse_cummax, _ = torch.flip(population, [0]).cummax(dim=0)
    reverse_cummax = torch.flip(reverse_cummax, [0])
    selected_values = population == reverse_cummax
    
    times = time_array[selected_values]
    weights = times[1:] - times[:-1]

    mse = torch.mean(((population[selected_values][1:] - fit[selected_values][1:]) * weights) ** 2)
    return mse.item()