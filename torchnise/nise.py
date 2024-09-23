"""
This file contains the main module Implementing the NISE calculations
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import tqdm
from torchnise.pytorch_utility import renorm
from torchnise import units
from torchnise.averaging_and_lifetimes import (
    estimate_lifetime,
    averaging
    )

from torchnise.absorption import (
    absorption_time_domain,
    absorb_time_to_freq
    )

from torchnise.fft_noise_gen import gen_noise

def nise_propagate(hfull, realizations, psi0, total_time, dt, temperature,
                   save_interval=1, t_correction="None", device="cpu",
                   save_u=False, save_coherence=False, mlnise_inputs=None,
                   mlnise_model=None):
    """
    Propagate the quantum state using the NISE algorithm with optional thermal
        corrections.

    Args:
        hfull (torch.Tensor): Hamiltonian of the system over time for different
            realizations.
        realizations (int): Number of noise realizations to simulate.
        psi0 (torch.Tensor): Initial state of the system.
        total_time (float): Total time for the simulation.
        dt (float): Time step size.
        temperature (float): Temperature for thermal corrections.
        save_interval (int, optional): Interval for saving results. Defaults to
            1.
        t_correction (str, optional): Method for thermal correction
            ("None", "TNISE", "MLNISE"). Defaults to "None".
        device (str, optional): Device for computation ("cpu" or "cuda").
            Defaults to "cpu".
        save_u (bool, optional): If True, save time evolution operators.
            Defaults to False.
        save_coherence (bool, optional): If True, save coherences.
            Defaults to False.
        mlnise_inputs (tuple, optional): Inputs for MLNISE model.
            Defaults to None.
        mlnise_model (nn.Module, optional): Machine learning model for MLNISE
            corrections. Defaults to None.

    Returns:
        tuple: (torch.Tensor, torch.Tensor, torch.Tensor) - Populations,
        coherences, and time evolution operators.
    """
    n_sites = hfull.shape[-1]
    factor = 1j * 1 / units.HBAR * dt * units.T_UNIT
    kbt = temperature * units.K

    total_steps = int(total_time / dt) + 1
    total_steps_saved = int(total_time / dt / save_interval) + 1
    psloc = torch.zeros((realizations, total_steps_saved, n_sites),
                        device=device)

    if save_coherence:
        coh_loc = torch.zeros((realizations, total_steps_saved, n_sites,
                               n_sites), device=device, dtype=torch.complex64)
    #grab the 0th timestep
    #[all realizations : 0th timestep  : all sites : all sites]
    h = hfull[:, 0, :, :].clone().to(device=device)
    #get initial eigenenergies and transition matrix from eigen to site basis.
    e, c = torch.linalg.eigh(h)
    # Since the Hamiltonian is hermitian we van use eigh
    # H contains the hamiltonians of all realizations.
    # To our advantage The eigh torch function (like almost all torch
    # functions) is setup so that it can efficently calculate the results for a
    # whole batch of inputs
    cold = c
    eold = e
    psi0 = psi0.repeat(realizations, 1)
    psi0 = psi0.unsqueeze(-1)
    pop0 = (psi0[:, :, 0] ** 2)
    psloc[:, 0, :] = pop0 #Save the population of the first timestep
    #Use the transition Matrix to transfer to the eigenbasis.
    #Bmm is a batched matrix multiplication, so it does the matrix
    #multiplication for all batches at once
    phi_b = cold.transpose(1, 2).to(dtype=torch.complex64).bmm(
                                                psi0.to(dtype=torch.complex64))
    if save_coherence:
        for i in range(n_sites):
            coh_loc[:, 0, i, i] = pop0[:, i]
    if save_u:
        uloc = torch.zeros((realizations, total_steps_saved, n_sites, n_sites),
                           device=device, dtype=torch.complex64)
        identity = torch.eye(n_sites, dtype=torch.complex64)
        identity = identity.reshape(1, n_sites, n_sites)
        uloc[:, 0, :, :] = identity.repeat(realizations, 1, 1)
        ub = cold.transpose(1, 2).to(dtype=torch.complex64).bmm(
                                                            uloc[:, 0, :, :])
        ub = ub.to(dtype=torch.complex64).to(device=device)

    #Now we get to the step by step timepropagation.
    #We start at 1 and not 0 since we have already filled the first slot of
    #our population dynamics
    for t in tqdm.tqdm(range(1,total_steps)):
        #grab the t'th timestep
        h = hfull[:, t, :, :].clone().to(device=device)
        #[all realizations : t'th timestep  : all sites : all sites]
        e, v_eps = torch.linalg.eigh(h)
        c = v_eps
        #multiply the old with the new transition matrix to get the
        #non-adiabatic coupling
        s = torch.matmul(c.transpose(1, 2), cold)
        if t_correction.lower() in ["mlnise", "tnise"]:
            s = apply_t_correction(s, n_sites, realizations, device, e, eold,
                                   t_correction, kbt, mlnise_model,
                                   mlnise_inputs, phi_b)
        #Make the Time Evolution operator
        u = torch.diag_embed(torch.exp(-e[:, :] * factor)).bmm(
                                                s.to(dtype=torch.complex64))
        phi_b = u.bmm(phi_b) #Apply the time evolution operator
        if save_u:
            ub = u.bmm(ub)

        if t_correction.lower() in ["mlnise", "tnise"]:
            phi_b = renorm(phi_b, dim=1)
        cold = c
        eold = e
        c = c.to(dtype=torch.complex64)

        phi_bin_loc_base = c.bmm(phi_b) #Transition to the site basis

        if t % save_interval == 0:
            psloc[:, t // save_interval, :] = (
                        (phi_bin_loc_base.abs() ** 2)[:, :, 0] )
            if save_u:
                if t_correction.lower() in ["mlnise", "tnise"]:
                    for i in range(n_sites):
                        ub_norm_row = renorm(ub[:, :, i], dim=1)
                        ub[:, :, i] = ub_norm_row[:, :]

                uloc[:, t // save_interval, :, :] = c.bmm(ub)

            if save_coherence:
                coh_loc[:, t // save_interval, :, :] = (
                    phi_bin_loc_base.squeeze(-1)[:, :, None] *
                    phi_bin_loc_base.squeeze(-1)[:, None, :].conj()
                )
    coh_loc = coh_loc.cpu() if save_coherence else None
    uloc = uloc.cpu() if save_u else None

    return psloc.cpu(), coh_loc, uloc

def apply_t_correction(s, n_sites, realizations, device, e, eold,
                       t_correction, kbt, mlnise_model, mlnise_inputs, phi_b):
    """
    Apply thermal corrections to the non-adiabatic coupling matrix.

    Args:
        s (torch.Tensor): Non-adiabatic coupling matrix.
        n_sites (int): Number of sites in the system.
        realizations (int): Number of noise realizations.
        device (str): Device for computation ("cpu" or "cuda").
        e (torch.Tensor): Eigenvalues of the Hamiltonian at the current time
            step.
        eold (torch.Tensor): Eigenvalues of the Hamiltonian at the previous
            time step.
        t_correction (str): Method for thermal correction ("TNISE", "MLNISE").
        kbt (float): Thermal energy (k_B * T).
        mlnise_model (nn.Module): Machine learning model for MLNISE
            corrections.
        mlnise_inputs (tuple, optional): Inputs for MLNISE model.
        phi_b (torch.Tensor): Wavefunction in the eigenbasis.
 
    Returns:
        torch.Tensor: Corrected non-adiabatic coupling matrix.
    """
    for ii in range(0,n_sites):
        max_c = torch.max(s[:, :, ii].real ** 2, 1)
        #The order of the eigenstates is not well defined and might be flipped
        #from one transition matrix to the transition matrix
        #to find the eigenstate that matches the previous eigenstate we find
        #the eigenvectors that overlapp the most and we use the index with
        #the highest overlap (kk) instead of the original index ii to index
        #the non adiabatic coupling correctly
        kk = max_c.indices
        cd = torch.zeros(realizations, device=device)
        for jj in range(n_sites):
            de = e[:, jj] - eold[:, ii]
            de[jj == kk] = 0 #if they are the same state the energy difference
            #is 0
            if t_correction == "TNISE":
                correction = torch.exp(-de / kbt / 4)
            else:
                correction = mlnise_model(mlnise_inputs, de, kbt, phi_b, s, jj,
                                          ii, realizations, device=device)

            s[:, jj, ii] = s[:, jj, ii].clone() * correction
            add_cd = s[:, jj, ii].clone() ** 2
            add_cd[jj == kk] = 0
            cd = cd.clone() + add_cd
        #The renormalization procedure broken into smaller steps,
        #because previously some errors showed
        #should probably be simplified
        norm = torch.abs(s[torch.arange(realizations), kk, ii].clone())
        dummy1 = torch.ones(realizations, device=device)
        dummy1[1 - cd > 0] = torch.sqrt(1 - cd[1 - cd > 0])
        dummy2 = torch.zeros(realizations, device=device)
        s_clone = s[torch.arange(realizations), kk, ii].clone()
        dummy2[1 - cd > 0] = s_clone.clone()[1 - cd > 0] / norm[1 - cd > 0]
        dummy3 = dummy1 * dummy2
        s[1 - cd > 0, kk[1 - cd > 0], ii] = dummy3[1 - cd > 0]
        return s

def nise_averaging(hfull, realizations, psi0, total_time, dt, temperature,
                   save_interval=1, t_correction="None",
                   averaging_method="standard", lifetime_factor=5,
                   device="cpu", save_coherence=True, save_u=False,
                   mlnise_inputs=None):
    """
    Run NISE propagation with different averaging methods to calculate averaged
    population dynamics.

    Args:
        hfull (torch.Tensor): Hamiltonian of the system over time for different
            realizations.
        realizations (int): Number of noise realizations to simulate.
        psi0 (torch.Tensor): Initial state of the system.
        total_time (float): Total time for the simulation.
        dt (float): Time step size.
        temperature (float): Temperature for thermal corrections.
        save_interval (int, optional): Interval for saving results.
            Defaults to 1.
        t_correction (str, optional): Method for thermal correction
            ("None", "TNISE", "MLNISE"). Defaults to "None".
        averaging_method (str, optional): Method for averaging results
            ("standard", "boltzmann", "interpolated"). Defaults to "standard".
        lifetime_factor (int, optional): Factor to scale estimated lifetimes.
            Defaults to 5.
        device (str, optional): Device for computation ("cpu" or "cuda").
            Defaults to "cpu".
        save_coherence (bool, optional): If True, save coherences.
            Defaults to False.
        save_u (bool, optional): If True, save time evolution operators.
            Defaults to False.
        mlnise_inputs (tuple, optional): Inputs for MLNISE model.
            Defaults to None.

    Returns:
        tuple: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -
            Averaged populations, coherences, time evolution operators,
            and lifetimes.
    """
    with torch.no_grad():
        # get the number of sites from the size of the hamiltonian
        lifetimes = None

        # run NISE without T correction
        # Run NISE without T correction if necessary
        if averaging_method.lower() in ["boltzmann", "interpolated"]:
            population, coherence, u = nise_propagate(
                hfull.to(device), realizations, psi0.to(device), total_time,
                dt, temperature, save_interval=save_interval,
                t_correction="None", device=device, save_u=True,
                save_coherence=True
            )
            lifetimes = (estimate_lifetime(u, dt * save_interval) *
                         lifetime_factor)

        population, coherence, u = nise_propagate(
            hfull.to(device), realizations, psi0.to(device), total_time, dt,
            temperature, save_interval=save_interval,
            t_correction=t_correction, device=device, save_u=save_u,
            save_coherence=True, mlnise_inputs=mlnise_inputs
        )

        population_averaged, coherence_averaged = averaging(
            population, averaging_method, lifetimes=lifetimes, step=dt*save_interval,
            coherence=coherence
        )

        if not save_coherence:
            coherence_averaged = None
        return population_averaged, coherence_averaged, u, lifetimes


def run_nise(h, realizations, total_time, dt, initial_state, temperature,
             spectral_funcs, save_interval=1,save_u=False,save_u_file=None, 
             t_correction="None", mode="Population", mu=None, 
             absorption_padding=10000, averaging_method="standard", 
             lifetime_factor=5, max_reps=100000, mlnise_inputs=None, device="cpu"):
    """
    Main function to run NISE simulations for population dynamics or
    absorption spectra.

    Args:
        h (torch.Tensor): Hamiltonian of the system over time or single
            Hamiltonian with noise.
        realizations (int): Number of noise realizations to simulate.
        total_time (float): Total time for the simulation.
        dt (float): Time step size.
        initial_state (torch.Tensor): Initial state of the system.
        temperature (float): Temperature for thermal corrections.
        spectral_funcs (list(callable)): Spectral density functions for noise
            generation.
        save_interval (int, optional): Interval for saving results.
            Defaults to 1.
        save_u (bool, optional): if the time_evolution operator should be saved
            Defaults to False
        save_u_file (str, optional): if not none, u will be saved to this file
        t_correction (str, optional): Method for thermal correction
            ("None", "TNISE", "MLNISE"). Defaults to "None".
        mode (str, optional): Simulation mode ("Population" or "Absorption").
            Defaults to "Population".
        mu (torch.Tensor, optional): Dipole moments for absorption
            calculations. Defaults to None.
        absorption_padding (int, optional): Padding for absorption spectra
            calculation. Defaults to 10000.
        averaging_method (str, optional): Method for averaging results
            ("standard", "boltzmann", "interpolated"). Defaults to "standard".
        lifetime_factor (int, optional): Factor to scale estimated lifetimes.
            Defaults to 5.
        max_reps (int, optional): Maximum number of realizations per chunk.
            Defaults to 100000.
        mlnise_inputs (tuple, optional): Inputs for MLNISE model. Defaults to
            None.
        device (str, optional): Device for computation ("cpu" or "cuda").
            Defaults to "cpu".

    Returns:
        tuple: Depending on mode, returns either (np.ndarray, np.ndarray) for
            absorption spectrum and frequency axis, or
            (torch.Tensor, torch.Tensor) for averaged populations and time
            axis.
    """
    total_steps = int((total_time + dt) / dt)
    save_steps = int((total_time + dt * save_interval) / (dt * save_interval))
    n_states = h.shape[-1]
    time_dependent_h = len(h.shape) >= 3
    window=1

    if time_dependent_h:
        trajectory_steps = h.shape[0]
        if realizations > 1:
            window = int((trajectory_steps - total_steps) / (realizations - 1))
            print(f"window is {window * dt} {units.CURRENT_T_UNIT}")

    def generate_hfull_chunk(chunk_size, start_index=0, window=1):
        if time_dependent_h:
            chunk_hfull = torch.zeros((chunk_size, total_steps, n_states, n_states))
            for j in range(chunk_size):
                h_index = start_index + j
                chunk_hfull[j, :, :, :] = torch.tensor(
                    h[window * h_index:window * h_index + total_steps, :, :])
            return chunk_hfull
        chunk_hfull = torch.zeros((chunk_size, total_steps, n_states,
                                   n_states))
        print("Generating noise")
        noise = gen_noise(spectral_funcs, dt, (chunk_size, total_steps,
                                               n_states))
        print("Building H")
        chunk_hfull[:] = h
        for i in range(n_states):
            chunk_hfull[:, :, i, i] += noise[:, :, i]
        return chunk_hfull

    num_chunks = ((realizations + max_reps - 1) // max_reps
                  if realizations > max_reps else 1)
    print(f"Splitting calculation into {num_chunks} chunks")
    chunk_size = (realizations + num_chunks - 1) // num_chunks

    save_u = mode.lower() == "absorption" or save_u
    weights = []
    all_output = torch.zeros(num_chunks, save_steps, n_states)
    if save_u:
        all_u = torch.zeros((num_chunks, save_steps, n_states,n_states),dtype=torch.complex64)

    if mode.lower() == (
            "population" and t_correction.lower() in ["mlnise", "tnise"] and
            averaging_method in ["interpolated", "boltzmann"]):
        all_coherence = torch.zeros(num_chunks, save_steps, n_states, n_states)
        all_lifetimes = torch.zeros(num_chunks, n_states)
    elif mode.lower() == "absorption":
        all_absorb_time = []

    for i in range(0, realizations, chunk_size):
        chunk_reps = min(chunk_size, realizations - i)
        weights.append(chunk_reps)
        chunk_hfull = generate_hfull_chunk(chunk_reps, start_index=i,
                                           window=window)
        print("Running calculation")
        pop_avg, coherence_avg, u, lifetimes = nise_averaging(
            chunk_hfull, chunk_reps, initial_state, total_time, dt,
            temperature, save_interval=save_interval,
            t_correction=t_correction, averaging_method=averaging_method,
            lifetime_factor=lifetime_factor, device=device, save_u=save_u,
            save_coherence=True, mlnise_inputs=mlnise_inputs
        )

        if mode.lower() == (
                "population" and t_correction.lower() in ["mlnise", "tnise"]
                and averaging_method in ["interpolated", "boltzmann"]):
            all_coherence[i // chunk_size, :, :, :] = coherence_avg
            all_lifetimes[i // chunk_size, :] = lifetimes
        elif mode.lower() == "absorption":
            absorb_time = absorption_time_domain(u, mu)
            all_absorb_time.append(absorb_time)
        if save_u:
            all_u [i // chunk_size, :, :, :] = torch.mean(u, dim=0)
        all_output[i // chunk_size, :, :] = pop_avg


    if mode.lower() == (
            "population" and t_correction.lower() in ["mlnise", "tnise"] and
            averaging_method in ["interpolated", "boltzmann"]):
        lifetimes = torch.mean(all_lifetimes, dim=0)
        print(f"lifetimes multiplied by lifetime factor are {lifetimes}")
        avg_output, _ = averaging(all_output, averaging_method,
                                  lifetimes=lifetimes, step=dt*save_interval,
                                  coherence=all_coherence,
                                  weight=torch.tensor(weights,
                                                      dtype=torch.float))
    else:
        lifetimes = None
        avg_output = np.average(all_output, axis=0, weights=weights)
    if save_u and save_u_file!=None:
        torch.save(torch.mean(all_u,dim=0),save_u_file)

    if mode.lower() == "absorption":
        avg_absorb_time = np.average(all_absorb_time, axis=0, weights=weights)
        pad = int(absorption_padding / (dt * units.T_UNIT))
        absorb_config = {
            "total_time": total_time,
            "dt": dt,
            "pad": pad,
            "smoothdamp": True,
            "smoothdamp_start_percent": 10
        }
        absorb_f, x_axis = absorb_time_to_freq(avg_absorb_time, absorb_config)
        return absorb_f, x_axis

    return avg_output, torch.linspace(0, total_time, avg_output.shape[0])


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
        self.fc1 = nn.Linear(8, 75)
        self.fc2 = nn.Linear(75, 75)
        self.fc3 = nn.Linear(75, 75)
        self.fc4 = nn.Linear(75, 1)

    def forward(self, mlnise_inputs, de, kbt, phi_b, s, jj, ii, realizations,
                device="cpu"):
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
        self._calculate_population_ratio(input_vec, phi_b, ii, jj)

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
        #thermal energy
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
        if ((phi_b.real[:, ii] ** 2 + phi_b.imag[:, ii] ** 2) <
            0.01 * (phi_b.real[:, jj] ** 2 + phi_b.imag[:, jj] ** 2)).any():

            mask = ((phi_b.real[:, ii] ** 2 + phi_b.imag[:, ii] ** 2) >
                0.01 * (phi_b.real[:, jj] ** 2 + phi_b.imag[:, jj] ** 2))
            input_vec[:, 1] = 100
            input_vec[mask, 1] = ((phi_b.real[mask, jj] ** 2 +
                                   phi_b.imag[mask, jj] ** 2) /
                                  (phi_b.real[mask, ii] ** 2 +
                                   phi_b.imag[mask, ii] ** 2)).squeeze()
        else:
            input_vec[:, 1] = ((phi_b.real[:, jj] ** 2 +
                                phi_b.imag[:, jj] ** 2) /
                               (phi_b.real[:, ii] ** 2 +
                                phi_b.imag[:, ii] ** 2)).squeeze()
