"""
This file contains the main module Implementing the NISE calculations
"""

import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, replace
from typing import Optional, Any, Union
import tqdm
from torchnise.pytorch_utility import renorm, H5Tensor, weighted_mean, CupyEigh, HAS_CUPY
from torchnise import units
from torchnise.averaging_and_lifetimes import estimate_lifetime, averaging

from torchnise.absorption import absorption_time_domain, absorb_time_to_freq, absorption_time_domain_vector

from torchnise.fft_noise_gen import gen_noise


@dataclass
class NISEParameters:
    """Configuration parameters for NISE simulation."""
    dt: float
    total_time: float
    temperature: float
    save_interval: int = 1
    t_correction: str = "None"
    device: str = "cpu"
    save_u: bool = False
    save_coherence: bool = False
    mlnise_inputs: Any = None
    mlnise_model: Any = None
    use_h5: bool = False
    constant_v: bool = False
    v_dt: Optional[float] = None
    # Additional params for run_nise / averaging
    save_multi_pop: bool = False
    save_multi_slice: Any = None
    save_pop_file: Any = None
    save_u_file: Any = None
    mode: str = "Population"
    mu: Any = None
    absorption_padding: int = 10000
    averaging_method: str = "standard"
    lifetime_factor: int = 5
    max_reps: int = 100000
    save_wavefunction: bool = False
    track_grads: bool = False
    keep_on_cuda: bool = False


def nise_propagate(
    hfull,
    realizations,
    psi0,
    params: NISEParameters,
    site_noise=None,
    v_time_dependent=None,
):
    """

    Propagate the quantum state using the NISE algorithm with optional thermal
        corrections.

    Args:
        hfull (torch.Tensor): Hamiltonian of the system over time for different
            realizations. Shape should be (time_steps,realizations,n_sites,n_sites)
            except for constant_V or v_time_dependent mode where it is
            (n_sites,n_sites)
        realizations (int): Number of noise realizations to simulate.
        psi0 (torch.Tensor): Initial state of the system.
        params (NISEParameters): Configuration parameters for the simulation.
        site_noise (torch.tensor): site noise used for constant coupling mode
            or v_time_dependent mode. Should have shape (steps,realizations,n_sites)
            Defaults to None.
        v_time_dependent (torch.tensor): if not None these couplings will be
            used with the specified timestep v_dt. Should only be used if v_dt
            is larger than dt, otherwise use the Full Hamiltonian in hfull.
            Defaults to None.

    Returns:
        tuple: (torch.Tensor, torch.Tensor, torch.Tensor) - Populations,
        coherences, and time evolution operators.
    """

    n_sites = hfull.shape[-1]
    
    # Delegate logic to StandardPropagator
    from torchnise.propagator import StandardPropagator
    propagator = StandardPropagator(n_sites, realizations, params)
    population, coherence, u, psi_loc = propagator.propagate(hfull, psi0, site_noise=site_noise, v_time_dependent=v_time_dependent, keep_on_cuda=params.keep_on_cuda)
    return population, coherence, u, psi_loc


def apply_t_correction(
    s,
    n_sites,
    realizations,
    e,
    eold,
    phi_b,
    aranged_realizations,
    params: NISEParameters,
):
    # s: (R, J, I)    R = realizations, J = n_sites, I = n_sites
    R, J, I = s.shape

    # 1) clone once
    s_new = s.clone()

    # 2) find the "winning" channel kk for each (r,i)
    #    mag2 = |s|^2  →  (R,J,I)
    mag2 = s_new.abs().pow(2)
    #    kk[r, i] = argmax_j  mag2[r, j, i]   → (R, I)
    kk = mag2.argmax(dim=1)

    # 3) build a one-hot mask for those maxima → (R,J,I)
    mask_max = F.one_hot(kk, num_classes=J).permute(0, 2, 1).to(torch.bool)

    # 4) construct the full delta-energy tensor de[r,j,i] = e[r,j] - eold[r,i]
    #    then zero out the "winning" channels
    de = e.unsqueeze(2) - eold.unsqueeze(1)  # (R,J,I)
    de = de.masked_fill(mask_max, 0.0)

    # 5) compute the correction factor in one shot
    kbt = params.temperature * units.K
    if params.t_correction == "TNISE":
        correction = torch.exp(-de / (kbt * 4))
    else:
        # if you have a model that supports batching over the full (R,J,I) tensor you can do:
        correction = params.mlnise_model(params.mlnise_inputs, de, kbt, phi_b, s_new)

    # 6) apply it in one broadcasted multiply
    s_new = s_new * correction  # (R,J,I)

    # 7) rebuild the cross‐term cd[r,i] = sum_{j≠kk} |s_new[r,j,i]|^2
    squared = s_new.abs().pow(2)
    cd = (squared * (~mask_max)).sum(dim=1)  # (R,I)

    # 8) compute how much room we have left: norm[r,i] = sqrt(max(1 - cd,0))
    slack = (1.0 - cd).clamp(min=0.0)  # (R,I)
    norm_phase = s_new / s_new.abs().clamp(min=1e-12)  # (R,J,I)
    norm = torch.sqrt(slack)  # (R,I)

    # 9) build the new maxima values and scatter them back
    new_max = norm.unsqueeze(1) * torch.gather(  # (R,1,I)
        norm_phase, 1, kk.unsqueeze(1)
    )  # (R,1,I)
    # only update those entries where slack > 0
    update_mask = mask_max & (slack.unsqueeze(1) > 0)

    s_new = torch.where(
        update_mask, new_max, s_new  # broadcast (R,1,I) → (R,J,I) under mask
    )

    return s_new


def nise_averaging(
    hfull,
    realizations,
    psi0,
    params: NISEParameters,
    site_noise=None,
    v_time_dependent=None,
):
    """
    Run NISE propagation with different averaging methods to calculate averaged
    population dynamics.

    Args:
        hfull (torch.Tensor): Hamiltonian of the system over time for different
            realizations.
        realizations (int): Number of noise realizations to simulate.
        psi0 (torch.Tensor): Initial state of the system.
        params (NISEParameters): Configuration parameters.
        site_noise (torch.tensor): site noise used for constant coupling mode
            or v_time_dependent mode. Should have shape (steps,realizations,n_sites)
            Defaults to None.
        v_time_dependent (torch.tensor): if not None these couplings will be
            used with the specified timestep v_dt. Should only be used if v_dt
            is larger than dt, otherwise use the Full Hamiltonian in hfull.
            Defaults to None.


    Returns:
        tuple: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -
            Averaged populations, coherences, time evolution operators,
            and lifetimes.
    """
    # with torch.no_grad():
    # with torch.autograd.set_detect_anomaly(True):
    # get the number of sites from the size of the hamiltonian
    lifetimes = None
    
    # Extract needed params
    averaging_method = params.averaging_method
    lifetime_factor = params.lifetime_factor
    dt = params.dt
    save_interval = params.save_interval
    device = params.device
    save_u = params.save_u
    save_coherence = params.save_coherence
    save_multi_pop = params.save_multi_pop
    save_multi_slice = params.save_multi_slice

    # run NISE without T correction
    # Run NISE without T correction if necessary
    if averaging_method.lower() in ["boltzmann", "interpolated"]:
        params_no_t = replace(params, t_correction="None", save_u=True, save_coherence=True)
        population, coherence, u, _ = nise_propagate(
            hfull.cpu(),
            realizations,
            psi0.to(device),
            params_no_t,
            site_noise=site_noise,
            v_time_dependent=v_time_dependent,
        )
        lifetimes = estimate_lifetime(u, dt * save_interval) * lifetime_factor

    should_save_u = save_u or (save_multi_slice is not None)
    params_run = replace(params, save_u=should_save_u, save_coherence=True)
    
    population, coherence, u, psi_loc = nise_propagate(
        hfull.cpu(),
        realizations,
        psi0.to(device),
        params_run,
        site_noise=site_noise,
        v_time_dependent=v_time_dependent,
    )

    population_averaged, coherence_averaged = averaging(
        population,
        averaging_method,
        lifetimes=lifetimes,
        step=dt * save_interval,
        coherence=coherence,
    )
    multi_pop = None
    if save_multi_pop:
        if save_multi_slice is None:
            multi_pop = torch.mean((torch.abs(u) ** 2).real, dim=0)
        else:
            multi_pop = torch.mean(
                (torch.abs(u[:, :, :, save_multi_slice]) ** 2).real, dim=0
            )
        # TODO add other averaging methods
    if not save_u:
        u = None
    if not save_coherence:
        coherence_averaged = None
    return population_averaged, coherence_averaged, u, lifetimes, multi_pop, psi_loc


def run_nise(
    h,
    realizations,
    initial_state,
    spectral_funcs,
    params: NISEParameters,
    v_time_dependent=None,
):
    """
    Main function to run NISE simulations for population dynamics or
    absorption spectra.

    Args:
        h (torch.Tensor): Hamiltonian of the system over time or single
            Hamiltonian with noise.
        realizations (int): Number of noise realizations to simulate.
        initial_state (torch.Tensor): Initial state of the system.
        spectral_funcs (list(callable)): Spectral density functions for noise
            generation.
        params (NISEParameters): Configuration parameters.
        v_time_dependent (torch.tensor): if not None these couplings will be
            used with the specified timestep v_dt. Should only be used if v_dt
            is larger than dt, otherwise use the Full Hamiltonian in hfull.
            Defaults to None.

    Returns:
        tuple: Depending on mode, returns either (np.ndarray, np.ndarray) for
            absorption spectrum and frequency axis, or
            (torch.Tensor, torch.Tensor) for averaged populations and time
            axis.
    """
    track_grads = params.track_grads
    context = torch.enable_grad() if track_grads else torch.no_grad()
    with context:
        dt = params.dt
        total_time = params.total_time
        save_interval = params.save_interval
        mode = params.mode
        absorption_padding = params.absorption_padding
        use_h5 = params.use_h5
        max_reps = params.max_reps
        device = params.device
        t_correction = params.t_correction
        averaging_method = params.averaging_method
        save_u = params.save_u
        mu = params.mu
        save_u_file = params.save_u_file
        save_multi_pop = params.save_multi_pop
        save_pop_file = params.save_pop_file
        
        total_steps = int((total_time + dt) / dt)
        save_steps = int((total_time + dt * save_interval) / (dt * save_interval))
        n_states = h.shape[-1]
        time_dependent_h = len(h.shape) >= 3
        
        # Determine constant_v
        if time_dependent_h:
            constant_v = False
        else:
            constant_v = True
            
        # Update constant_v in params
        params = replace(params, constant_v=constant_v)

        window = 1
        shuffled_indices = None
        
        if time_dependent_h:
            trajectory_steps = h.shape[0]
            if realizations > 1:
                window = int((trajectory_steps - total_steps) / (realizations - 1))
                print(f"window is {window * dt} {units.CURRENT_T_UNIT}")
            total_slices = trajectory_steps - total_steps + 1
            available_slices = list(range(0, total_slices, window))
            num_available_slices = len(available_slices)
            print(f"Number of available slices: {num_available_slices}")

            # Initialize shuffled_indices with shape (realizations, n_states)
            shuffled_indices = torch.zeros(
                (realizations, n_states), dtype=torch.int, device="cpu"
            )

            # For each site, shuffle the available slices and assign to realizations
            for i in range(n_states):
                slices_for_site = available_slices.copy()
                np.random.shuffle(slices_for_site)
                repeats = (
                    realizations + num_available_slices - 1
                ) // num_available_slices
                slices_for_site_extended = (slices_for_site * repeats)[:realizations]
                shuffled_indices[:, i] = torch.tensor(
                    slices_for_site_extended, dtype=torch.int, device="cpu"
                )

        else:
            constant_v = True
            shuffled_indices = None

        def generate_hfull_chunk(
            chunk_size, start_index=0, window=1, shuffled_indices=None
        ):

            if time_dependent_h:
                chunk_shape = (total_steps, chunk_size, n_states, n_states)
                if use_h5:
                    chunk_hfull = H5Tensor(
                        shape=chunk_shape, h5_filepath=f"H_{start_index}.h5"
                    )
                else:
                    chunk_hfull = torch.zeros(chunk_shape, device="cpu")
                for j in range(chunk_size):
                    h_index = start_index + j
                    chunk_hfull[:, j, :, :] = torch.tensor(
                        h[window * h_index : window * h_index + total_steps, :, :],
                        device="cpu",
                    )
                if shuffled_indices is not None:
                    for i in range(n_states):
                        shuffled_start = shuffled_indices[h_index, i]
                        chunk_hfull[:, j, i, i] = torch.tensor(
                            h[shuffled_start : shuffled_start + total_steps, i, i],
                            device="cpu",
                        )
                return chunk_hfull, None

            print("Generating noise")

            noise = gen_noise(
                spectral_funcs,
                dt,
                (total_steps, chunk_size, n_states),
                dtype=torch.float32,
                device=device,
                use_h5=use_h5,
            ).cpu()
            # if use_h5:
            # noise=H5Tensor(noise,"noise.h5")
            return h, noise
            # print("Building H")
            # chunk_hfull[:] = h
            # if use_h5:
            #    for step in tqdm.tqdm(range(total_steps)
            #                          ,desc="timesteps of noise added to Hamiltonian"):
            #        #print(step)
            #        chunk_hfull_step=chunk_hfull[step, :, :, :]

            #        for i in range (n_states):
            #            chunk_hfull_step[ :, i, i]=chunk_hfull_step[ :, i, i] + noise[step, :, i]
            #        chunk_hfull[step, :, :, :] = chunk_hfull_step
            # else:
            # chunk_hfull[:] = h
            #    for i in range (n_states):
            #        chunk_hfull[:, :, i, i] += noise[:, :, i]
            # return chunk_hfull, None

        num_chunks = (
            (realizations + max_reps - 1) // max_reps if realizations > max_reps else 1
        )
        print(f"Splitting calculation into {num_chunks} chunks")
        chunk_size = (realizations + num_chunks - 1) // num_chunks

        # Logic for save_u update based on mode
        save_u_loop = mode.lower() == "absorption" or save_u
        weights = []
        all_output = torch.zeros(num_chunks, save_steps, n_states, device="cpu")

        if mode.lower() == (
            "population"
            and t_correction.lower() in ["mlnise", "tnise"]
            and averaging_method in ["interpolated", "boltzmann"]
        ):
            all_coherence = torch.zeros(
                num_chunks, save_steps, n_states, n_states, device="cpu"
            )
            all_lifetimes = torch.zeros(num_chunks, n_states, device="cpu")
        elif mode.lower() == "absorption":
            all_absorb_time = []

        for i in range(0, realizations, chunk_size):
            chunk_reps = min(chunk_size, realizations - i)
            weights.append(chunk_reps)
            chunk_hfull, site_noise = generate_hfull_chunk(
                chunk_reps,
                start_index=i,
                window=window,
                shuffled_indices=shuffled_indices,
            )
            print("Running calculation")
            save_coherence_loop = False
            if mode.lower() == (
                "population"
                and t_correction.lower() in ["mlnise", "tnise"]
                and averaging_method in ["interpolated", "boltzmann"]
            ):
                print("Saving coherence")
                save_coherence_loop = True

            # Update params for this chunk
            # For Absorption, we want to save wavefunction but NOT u
            # Also override psi0 if absorption mode to be the dipole moments
            chunk_psi0 = initial_state
            
            if mode.lower() == "absorption":
                save_wavefunction_loop = True
                save_u_loop = False # Disable saving U for absorption efficiency
                if mu is None:
                    raise ValueError("Dipole moments (mu) must be provided in parameters for Absorption mode.")
                
                # Use mu as initial state. mu might be (N, 3).
                # If mu is time dependent, use mu[0]? Or handle outside?
                # Assuming mu is (N, 3) or (R, T, N, 3)
                # If dynamic, we need mu at t=0 for psi0.
                if isinstance(mu, (np.ndarray, list)):
                     mu_tensor = torch.tensor(mu, device=device, dtype=torch.float32) # real dipoles
                else:
                     mu_tensor = mu.to(device=device, dtype=torch.float32)
                
                if len(mu_tensor.shape) == 2: # (N, 3)
                     chunk_psi0 = mu_tensor
                elif len(mu_tensor.shape) >= 3:
                     # e.g. (T, N, 3) or (R, T, N, 3)
                     # Take time 0
                     if len(mu_tensor.shape) == 3: # (T, N, 3)
                         chunk_psi0 = mu_tensor[0]
                     else: # (R, T, N, 3)
                         chunk_psi0 = mu_tensor[i:i+chunk_reps, 0, :, :]
            else:
                save_wavefunction_loop = params.save_wavefunction

            chunk_params = replace(params, save_u=save_u_loop, save_coherence=save_coherence_loop, save_wavefunction=save_wavefunction_loop)
            
            if mode.lower() == "absorption":
                 # Call propagate directly to avoid forced coherence calculation in nise_averaging
                 pop_avg, _, u, psi_loc = nise_propagate(
                    chunk_hfull,
                    chunk_reps,
                    chunk_psi0,
                    chunk_params,
                    site_noise=site_noise,
                    v_time_dependent=v_time_dependent,
                )
                 pop_avg = pop_avg.mean(dim=0) # Average over realizations for placeholder
                 # Dummy placeholders for variables used below if needed (though absorption block handles it)
                 lifetimes = None
                 multi_pop = None
            else:
                pop_avg, coherence_avg, u, lifetimes, multi_pop, psi_loc = nise_averaging(
                    chunk_hfull,
                    chunk_reps,
                    chunk_psi0,
                    chunk_params,
                    site_noise=site_noise,
                    v_time_dependent=v_time_dependent,
                )

            if mode.lower() == (
                "population"
                and t_correction.lower() in ["mlnise", "tnise"]
                and averaging_method in ["interpolated", "boltzmann"]
            ):
                all_coherence[i // chunk_size, :, :, :] = coherence_avg.cpu()
                all_lifetimes[i // chunk_size, :] = lifetimes.cpu()
                all_lifetimes[i // chunk_size, :] = lifetimes.cpu()
            elif mode.lower() == "absorption":
                # Use vector propagation result
                if psi_loc is not None:
                    # Only pass the relevant slice of mu if it's chunked by realization
                    mu_chunk = mu
                    if isinstance(mu, torch.Tensor) and len(mu.shape) == 4: # (R, T, N, 3)
                        mu_chunk = mu[i:i+chunk_reps]
                        
                    absorb_time = absorption_time_domain_vector(psi_loc, mu_chunk, dt=dt)  # Note: absorption_time_domain_vector currently handles damping inside? No, it has use_damping arg.
                    # Wait, absorption.py has optional args. Let's assume defaults or pass them if params has them.
                    # params has no explicit damping params for absorption_time_domain call in original code, 
                    # but absorption_time_domain definition has defaults.
                    # Original code used absorption_time_domain(u, mu).
                    # We should probably match behavior. 
                    # However, absorption_time_domain_vector returns numpy array already.
                    absorb_time = torch.tensor(absorb_time, device="cpu") # convert back to tensor for stacking
                    all_absorb_time.append(absorb_time)
                else:
                    # Fallback if somehow psi_loc is None (should not happen with new logic)
                    absorb_time = absorption_time_domain(u, mu)
                    all_absorb_time.append(absorb_time)
            if save_u:
                if save_u and save_u_file is not None:
                    if "." in save_u_file:
                        name, ending = save_u_file.split(".")
                    else:
                        name = save_u_file
                        ending = "pt"
                    # Ensure the directory exists
                    dir_path = os.path.dirname(name)
                    if dir_path:  # Only try to create it if it's not an empty string
                        os.makedirs(dir_path, exist_ok=True)
                    torch.save(u, f"{name}_{i}.{ending}")
            if save_multi_pop:
                if save_multi_pop and save_pop_file is not None:
                    if "." in save_pop_file:
                        name, ending = save_pop_file.split(".")
                    else:
                        name = save_pop_file
                        ending = "pt"
                    # Ensure the directory exists
                    dir_path = os.path.dirname(name)
                    if dir_path:  # Only try to create it if it's not an empty string
                        os.makedirs(dir_path, exist_ok=True)
                    torch.save(multi_pop, f"{name}_{i}.{ending}")
            all_output[i // chunk_size, :, :] = pop_avg.cpu()

        if mode.lower() == (
            "population"
            and t_correction.lower() in ["mlnise", "tnise"]
            and averaging_method in ["interpolated", "boltzmann"]
        ):
            lifetimes = torch.mean(all_lifetimes, dim=0)
            print(f"lifetimes multiplied by lifetime factor are {lifetimes}")
            avg_output, _ = averaging(
                all_output,
                averaging_method,
                lifetimes=lifetimes,
                step=dt * save_interval,
                coherence=all_coherence,
                weight=torch.tensor(weights, dtype=torch.float),
                device=device,
            )
        else:
            lifetimes = None
            avg_output = weighted_mean(all_output, weights=weights)

        if mode.lower() == "absorption":
            # Stack the list of tensors into a single tensor
            all_absorb_time = torch.stack(all_absorb_time)
            avg_absorb_time = weighted_mean(all_absorb_time, weights=weights, dim=0)
            pad = int(absorption_padding / (dt * units.T_UNIT))
            absorb_config = {
                "total_time": total_time,
                "dt": dt,
                "pad": pad,
                "smoothdamp": True,
                "smoothdamp_start_percent": 10,
            }
            absorb_f, x_axis = absorb_time_to_freq(avg_absorb_time, absorb_config)
            return absorb_f, x_axis

    return avg_output.cpu(), torch.linspace(0, total_time, avg_output.shape[0])


class MLNISEModel(nn.Module):
    """
    Neural network model to predict correction factors for non-adiabatic
    coupling based on input features.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.
        fc4 (nn.Linear): Output fully connected layer.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 25)
        self.fc2 = nn.Linear(25, 25)
        self.fc3 = nn.Linear(25, 25)
        self.fc4 = nn.Linear(25, 1)

    def forward(
        self, mlnise_inputs, de, kbt, phi_b, s, jj, ii, realizations, device="cpu"
    ):
        """
        Forward pass through the MLNISE model to calculate correction factors.

        Args:
            mlnise_inputs (tuple): Inputs for the MLNISE model, containing
                reorganization energy and correlation time.
            de (torch.Tensor): Energy differences between states.
            kbt (float): Thermal energy (k_B * T).
            phi_b (torch.Tensor): Wavefunction in the eigenbasis.
            s (torch.Tensor): Non-adiabatic coupling matrix.
            jj (int): Index of the target state.
            ii (int): Index of the current state.
            realizations (int): Number of noise realizations.
            device (str, optional): Device for computation ("cpu" or "cuda").
                Defaults to "cpu".

        Returns:
            torch.Tensor: Correction factor for the non-adiabatic coupling
                matrix.
        """
        # Initialize the input vector for the neural network
        input_vec = torch.zeros(realizations, 8, device=device)

        # Feature 0: Energy difference
        input_vec[:, 0] = de

        # Feature 1: Current population ratio between eigenstates i and j
        self._calculate_population_ratio(input_vec, phi_b.squeeze(-1), ii, jj)

        # Feature 2: Original value of the non-adiabatic coupling
        input_vec[:, 2] = s[:, jj, ii]

        # Feature 3: Temperature
        input_vec[:, 3] = kbt

        # Feature 4: Reorganization energy
        input_vec[:, 4] = mlnise_inputs[0]

        # Feature 5: Correlation time
        input_vec[:, 5] = mlnise_inputs[1]

        # Feature 6: Placeholder for future use (currently inactive)
        # input_vec[:, 6] = t * dt

        # Feature 7: Exponential of the energy difference divided by
        # thermal energy
        input_vec[:, 7] = torch.exp(de / kbt)

        # Apply the neural network layers
        res1 = F.elu(self.fc1(input_vec))
        res2 = F.elu(self.fc2(res1))
        res3 = F.elu(self.fc3(res2))
        correction = F.elu(self.fc4(res3)) + 1

        return correction.squeeze()

    @staticmethod
    def _calculate_population_ratio(input_vec, phi_b, ii, jj):
        """
        Calculate the population ratio feature for the input vector.

        Args:
            input_vec (torch.Tensor): Input vector for the neural network.
            phi_b (torch.Tensor): Wavefunction in the eigenbasis.
            ii (int): Index of the current state.
            jj (int): Index of the target state.
        """
        # Calculate the population ratio and avoid exploding gradients
        # by capping the ratio at 100
        if (
            (phi_b.real[:, ii] ** 2 + phi_b.imag[:, ii] ** 2)
            < 0.01 * (phi_b.real[:, jj] ** 2 + phi_b.imag[:, jj] ** 2)
        ).any():

            mask = (phi_b.real[:, ii] ** 2 + phi_b.imag[:, ii] ** 2) > 0.01 * (
                phi_b.real[:, jj] ** 2 + phi_b.imag[:, jj] ** 2
            )
            input_vec[:, 1] = 100
            input_vec[mask, 1] = (
                (phi_b.real[mask, jj] ** 2 + phi_b.imag[mask, jj] ** 2)
                / (phi_b.real[mask, ii] ** 2 + phi_b.imag[mask, ii] ** 2)
            ).squeeze()
        else:
            input_vec[:, 1] = (
                (phi_b.real[:, jj] ** 2 + phi_b.imag[:, jj] ** 2)
                / (phi_b.real[:, ii] ** 2 + phi_b.imag[:, ii] ** 2)
            ).squeeze()
