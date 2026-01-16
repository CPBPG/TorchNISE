"""
This module provides functions to compute time domain absorption and convert it
to an absorption spectrum using FFT. It allows for optional damping to simulate
signal decay over time.
"""

import numpy as np
import torch

from torchnise.pytorch_utility import smooth_damp_to_zero
from torchnise import units


def absorption_time_domain(
    time_evolution_operator, dipole_moments, use_damping=False, lifetime=1000, dt=1
):
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
            unit.
        dt (float, optional): Time step size. Default is 1.

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
        dipole_moments = np.tile(dipole_moments, (realizations, timesteps, 1, 1))

    absorption_td = 0
    if use_damping:
        damp = np.exp(-np.arange(0, timesteps) * dt / lifetime)
    else:
        damp = 1

    for xyz in range(3):
        for real in range(realizations):
            for m in range(n_sites):
                for n in range(n_sites):
                    absorption_td += (
                        time_evolution_operator[real, :, m, n]
                        * dipole_moments[real, :, m, xyz]
                        * dipole_moments[real, 0, n, xyz]
                        / realizations
                        * damp
                    )
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
    freq = np.fft.fftfreq(int((total_time + dt) / dt) + pad, d=dt * units.T_UNIT)
    x_axis = -units.HBAR * 2 * np.pi * np.fft.fftshift(freq)
    absorb_f_max = np.max(absorb_f.real - absorb_f.real[0])
    absorb_f = (absorb_f.real - absorb_f.real[0]) / absorb_f_max
    return absorb_f, x_axis


def absorption_time_domain_vector(
    time_evolution_operator, dipole_moments, use_damping=False, lifetime=1000, dt=1
):
    """
    Calculate the time domain absorption efficiently using propagated response vectors (psi_t).
    
    Absorption(t) ~ < mu(t) . psi(t) >
    
    Args:
        time_evolution_operator (torch.Tensor): Propagated wavefunction trajectories 
             initialized with dipole moments (psi_t).
             Shape: (Realizations, Timesteps, N_sites, 3)
        dipole_moments (torch.Tensor): Dipole moments (mu).
             Shape: (Realizations, Timesteps, N_sites, 3) or (N_sites, 3) etc.
        use_damping (bool): Apply lifetime damping.
    """
    psi_t = time_evolution_operator
    
    if not isinstance(psi_t, torch.Tensor):
        psi_t = torch.tensor(psi_t)
        
    device = psi_t.device
    target_dtype = psi_t.dtype
    real_dtype = target_dtype if not psi_t.is_complex() else (torch.float32 if target_dtype==torch.complex64 else torch.float64)
    
    mu = dipole_moments
    if not isinstance(mu, torch.Tensor):
        mu = torch.tensor(mu, device=device, dtype=real_dtype)
    else:
        mu = mu.to(device=device, dtype=real_dtype)
    
    realizations, timesteps, n_sites, _ = psi_t.shape
    
    # Normalize mu shape to match psi_t for elementwise mul
    if mu.ndim == 2: # (N, 3)
        mu = mu.unsqueeze(0).unsqueeze(0) # (1, 1, N, 3)
    elif mu.ndim == 3:
         # Could be (R, N, 3) or (T, N, 3)?
         # Usually (R, N, 3) for static dipoles varying per realization
         if mu.shape[0] == realizations:
              mu = mu.unsqueeze(1) # (R, 1, N, 3)
         else:
              mu = mu.unsqueeze(0) # (1, T, N, 3)

    # Calculate Overlap < mu | psi(t) >
    # Dot product over N and 3 dims.
    # psi_t is (R, T, N, 3). mu is broadcastable.
    
    # (R, T, N, 3) * (R, T, N, 3) -> sum(N, 3) -> (R, T)
    overlap = (psi_t * mu.to(psi_t.dtype)).sum(dim=(-1, -2))
    
    avg_overlap = overlap.mean(dim=0) # Average over realizations
    
    if use_damping:
        t_axis = torch.arange(timesteps, device=psi_t.device) * dt
        damp = torch.exp(-t_axis / lifetime)
        avg_overlap = avg_overlap * damp
        
    return avg_overlap.cpu().numpy()


def cd_time_domain_vector(
    psi_t, dipole_moments, positions, use_damping=False, lifetime=1000, dt=1
):
    """
    Calculate the time domain Circular Dichroism (CD) signal efficiently using 
    propagated response vectors.
    
    CD(t) ~ Sum <m|U|n> * (R_m - R_n) . (mu_m x mu_n)
          ~ 2 * Sum <psi_mu_m(t) | M_m >  (Assuming real symmetric Hamiltonian)
          where M_m = R_m x mu_m.
    
    Args:
        psi_t (torch.Tensor): Propagated wavefunction trajectories initialized with dipoles.
             Shape: (Realizations, Timesteps, N_sites, 3)
        dipole_moments (torch.Tensor): Dipole moments (mu).
             Shape: (Realizations, Timesteps, N_sites, 3) or shorter.
        positions (torch.Tensor): Positions (R).
             Shape: (Realizations, Timesteps, N_sites, 3) or shorter.
    """
    if not isinstance(psi_t, torch.Tensor):
        psi_t = torch.tensor(psi_t)
        
    device = psi_t.device
    target_dtype = psi_t.dtype
    real_dtype = target_dtype if not psi_t.is_complex() else (torch.float32 if target_dtype==torch.complex64 else torch.float64)
    
    def ensure_tensor(t):
        if not isinstance(t, torch.Tensor):
            return torch.tensor(t, device=device, dtype=real_dtype)
        return t.to(device=device, dtype=real_dtype)
        
    mu = ensure_tensor(dipole_moments)
    pos = ensure_tensor(positions)
    
    realizations, timesteps, n_sites, _ = psi_t.shape
    
    # Broadcasting normalization
    def normalize_shape(tensor):
         # Expected target (R, T, N, 3)
         if tensor.ndim == 2: # (N, 3)
             return tensor.unsqueeze(0).unsqueeze(0)
         if tensor.ndim == 3: 
             if tensor.shape[0] == realizations: # (R, N, 3)
                  return tensor.unsqueeze(1)
             else: # (T, N, 3)
                  return tensor.unsqueeze(0)
         return tensor
         
    mu = normalize_shape(mu)
    pos = normalize_shape(pos)
    
    # Calculate Magnetic Moment proxy M = R x mu
    # Cross product dims: (..., 3) x (..., 3) -> (..., 3)
    M_vec = torch.cross(pos, mu, dim=-1)
    
    # Contract psi_t with M_vec
    # psi_t is complex, M_vec is real.
    # We want sum_m M_m . psi_m(t)
    overlap = (psi_t * M_vec.to(psi_t.dtype)).sum(dim=(-1, -2)) # Sum over Pol(3) and Sites
    
    # NISE derivation suggests factor of 2?
    # Original code: Sum U_mn (R_m - R_n) . (mu_m x mu_n)
    # Vector code: Sum psi_m . M_m = Sum (U_mn mu_n) . (R_m x mu_m)
    # = Sum U_mn mu_n . (R_m x mu_m) = Sum U_mn R_m . (mu_m x mu_n)
    # This is Term 1. Term 2 involves R_n.
    # If symmetric, Term 1 == - Term 2? Wait.
    # (R_m - R_n) = Term 1 + (-Term 2).
    # If Term 2 == -Term 1, then Total = 2 * Term 1.
    # So yes, multiply by 2.
    
    avg_signal = 2.0 * overlap.mean(dim=0)
    
    # Scaling: 1/hbar? NISE absorption.c does not scale. CD might.
    # Original cd_time_domain has * (1/units.HBAR).
    # We should match that.
    
    if use_damping:
        t_axis = torch.arange(timesteps, device=device) * dt
        damp = torch.exp(-t_axis / lifetime)
        avg_signal = avg_signal * damp
        
    return avg_signal.cpu().numpy() * (1.0 / units.HBAR)


def ld_time_domain_vector(
    psi_t, dipole_moments, axis=2, use_damping=False, lifetime=1000, dt=1
):
    """
    Calculate Linear Dichroism (LD) using vector propagation.
    
    LD = 3 * <Z-pol> - <Isotropic>
    """
    if not isinstance(psi_t, torch.Tensor):
        psi_t = torch.tensor(psi_t)
    
    device = psi_t.device
    target_dtype = psi_t.dtype
    real_dtype = target_dtype if not psi_t.is_complex() else (torch.float32 if target_dtype==torch.complex64 else torch.float64)
    
    mu = dipole_moments
    if not isinstance(mu, torch.Tensor):
        mu = torch.tensor(mu, device=device, dtype=real_dtype)
    else:
        mu = mu.to(device=device, dtype=real_dtype)
        
    realizations, timesteps, n_sites, _ = psi_t.shape
    
    # Normalize mu shape
    if mu.ndim == 2: mu = mu.unsqueeze(0).unsqueeze(0)
    elif mu.ndim == 3:
         if mu.shape[0] == realizations: mu = mu.unsqueeze(1)
         else: mu = mu.unsqueeze(0)
         
    # 1. Isotropic Term: sum(mu . psi)
    # Matches absorption_time_domain_vector logic
    iso_overlap = (psi_t * mu.to(psi_t.dtype)).sum(dim=(-1, -2)) # (R, T)
    
    # 2. Z-Polarized Term (or specified axis)
    # Extract component
    psi_z = psi_t[..., axis] # (R, T, N)
    mu_z = mu[..., axis]     # (R, T, N)
    
    z_overlap = (psi_z * mu_z.to(psi_t.dtype)).sum(dim=-1) # (R, T)
    
    # LD = 3 * Z - Iso (Mean over realizations)
    ld_traj = 3.0 * z_overlap - iso_overlap
    avg_ld = ld_traj.mean(dim=0)
    
    if use_damping:
        t_axis = torch.arange(timesteps, device=device) * dt
        damp = torch.exp(-t_axis / lifetime)
        avg_ld = avg_ld * damp
        
    return avg_ld.cpu().numpy()


def luminescence_time_domain_vector(
    psi_t, dipole_moments, dt=1
):
    """
    Calculate Luminescence (Fluorescence) signal efficiently.
    
    L(t) ~ < mu(t) . psi(t) >
    where psi(0) = rho_eq * mu(0).
    
    This is mathematically identical to absorption_time_domain_vector
    after the initial state preparation.
    """
    # Just reuse absorption logic
    return absorption_time_domain_vector(psi_t, dipole_moments, dt=dt)


def raman_time_domain_vector(
    psi_t, alpha_t, dt=1
):
    """
    Calculate Raman scattering signals (VV and VH).
    
    Args:
        psi_t (torch.Tensor): Propagated alpha trajectories. 
             Expected Shape: (Realizations, Timesteps, N_sites, 6)
             Ordering: xx, xy, xz, yy, yz, zz
        alpha_t (torch.Tensor): Polarizability tensor trajectory.
             Expected Shape: (Realizations, Timesteps, N_sites, 6)
             
    Returns:
        vv_signal (numpy.array): Parallel scattering signal (xx, yy, zz)
        vh_signal (numpy.array): Perpendicular scattering signal (xy, xz, yz)
    """
    if not isinstance(psi_t, torch.Tensor):
        psi_t = torch.tensor(psi_t)
        
    device = psi_t.device
    target_dtype = psi_t.dtype
    real_dtype = target_dtype if not psi_t.is_complex() else (torch.float32 if target_dtype==torch.complex64 else torch.float64)
    
    def ensure_tensor(t):
        if not isinstance(t, torch.Tensor):
            return torch.tensor(t, device=device, dtype=real_dtype)
        return t.to(device=device, dtype=real_dtype)
        
    alpha = ensure_tensor(alpha_t)
    
    realizations, timesteps, n_sites, _ = psi_t.shape
    
    # PSI_T assumed to be initialized with ALPHA(0) components.
    # We calculate Overlap O_k = < alpha_k(t) | psi_k(t) > for k in [0..5]
    # Actually, NISE does:
    # VV: sum_{k in {xx, yy, zz}} < alpha_k(t) | psi_k(t) >
    # VH: sum_{k in {xy, xz, yz}} 2 * < alpha_k(t) | psi_k(t) >
    
    # Normalize alpha shape if needed (e.g. static)
    if alpha.ndim == 2: alpha = alpha.unsqueeze(0).unsqueeze(0)
    elif alpha.ndim == 3:
         if alpha.shape[0] == realizations: alpha = alpha.unsqueeze(1)
         else: alpha = alpha.unsqueeze(0)
         
    # Calculate overlap for each component k
    # (R, T, N, 6) * (R, T, N, 6) -> sum over N -> (R, T, 6)
    overlaps = (psi_t * alpha.to(psi_t.dtype)).sum(dim=-2) 
    
    # Indices: 0:xx, 1:xy, 2:xz, 3:yy, 4:yz, 5:zz
    
    # VV: 0, 3, 5
    vv_comp = overlaps[..., 0] + overlaps[..., 3] + overlaps[..., 5]
    
    # VH: 1, 2, 4 (times 2)
    vh_comp = 2.0 * (overlaps[..., 1] + overlaps[..., 2] + overlaps[..., 4])
    
    vv_avg = vv_comp.mean(dim=0).cpu().numpy()
    vh_avg = vh_comp.mean(dim=0).cpu().numpy()
    
    return vv_avg, vh_avg

