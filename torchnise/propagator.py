
import torch
import torch.nn.functional as F
import tqdm
from abc import ABC, abstractmethod
from typing import Optional, Any
from dataclasses import dataclass
from torchnise.pytorch_utility import renorm, H5Tensor, CupyEigh, HAS_CUPY
from torchnise import units

@dataclass
class PropagatorState:
    """Holds the current state of the propagation."""
    h: torch.Tensor
    e: torch.Tensor
    c: torch.Tensor
    phi_b: torch.Tensor
    ub: Optional[torch.Tensor] = None
    eold: Optional[torch.Tensor] = None
    cold: Optional[torch.Tensor] = None

class NISEPropagator(ABC):
    """
    Abstract base class for NISE propagation logic.
    """
    def __init__(self, n_sites, realizations, params):
        self.n_sites = n_sites
        self.realizations = realizations
        self.params = params
        self.device = params.device
        
        # Determine strict unit handling
        self.factor = 1j * 1 / units.HBAR * params.dt * units.T_UNIT
        
    @abstractmethod
    def propagate(self, hfull, psi0, site_noise=None, v_time_dependent=None):
        pass

class StandardPropagator(NISEPropagator):
    """
    Standard implementation of NISE propagation.
    """
    def propagate(self, hfull, psi0, site_noise=None, v_time_dependent=None, keep_on_cuda=False, verbose=True, backward: bool = False):
        params = self.params
        dt = params.dt
        total_time = params.total_time
        save_interval = params.save_interval
        t_correction = params.t_correction
        save_u = params.save_u
        save_coherence = params.save_coherence
        use_h5 = params.use_h5
        constant_v = params.constant_v
        v_dt = params.v_dt
        save_wavefunction = params.save_wavefunction
        device = self.device
        n_sites = self.n_sites
        realizations = self.realizations

        # --- Prepare initial state (Normalize Shape) ---
        if psi0.ndim == 1:
            psi0 = psi0.unsqueeze(-1)  # (N, 1)
        
        # psi0 can be (N, K), (R, N), or (R, N, K)
        if psi0.ndim == 3 and psi0.shape[0] == realizations:
             psi0 = psi0.clone()
        elif psi0.ndim == 2 and psi0.shape[0] == realizations and psi0.shape[1] == n_sites:
             # (R, N) -> (R, N, 1)
             psi0 = psi0.unsqueeze(-1).clone()
        else:
             # Assume (N, K) -> Expand to (R, N, K)
             psi0 = psi0.unsqueeze(0).expand(realizations, -1, -1).clone()

        # Determine K (number of vectors being propagated)
        psi_cols = psi0.shape[-1]

        if constant_v and len(hfull.shape) == 2:
             hfull = hfull.unsqueeze(0) # (1, n_sites, n_sites)
             
        total_steps = round(total_time / dt + 1e-6) + 1
        total_steps_saved = round(total_time / dt / save_interval + 1e-6) + 1
        
        # Initialize output tensors
        if use_h5:
            psloc = H5Tensor(shape=(realizations, total_steps_saved, n_sites), h5_filepath="psloc.h5")
        else:
            if keep_on_cuda:
                psloc = torch.zeros((realizations, total_steps_saved, n_sites), device=device)
            else:
                psloc = torch.zeros((realizations, total_steps_saved, n_sites), device="cpu")

        if save_coherence:
            if use_h5:
                coh_loc = H5Tensor(shape=(realizations, total_steps_saved, n_sites, n_sites), h5_filepath="cohloc.h5")
            else:
                coh_loc = torch.zeros((realizations, total_steps_saved, n_sites, n_sites), device="cpu", dtype=torch.complex64)
        else:
            coh_loc = None

        if save_u:
             if use_h5:
                  uloc = H5Tensor(shape=(realizations, total_steps_saved, n_sites, n_sites), h5_filepath="uloc.h5", dtype=torch.complex64)
             else:
                  uloc = torch.zeros((realizations, total_steps_saved, n_sites, n_sites), device="cpu", dtype=torch.complex64)
        else:
             uloc = None

        if save_wavefunction:
             if use_h5:
                  # Note: H5Tensor might need complex support update if not present, assuming it works or fallback
                  psi_loc = H5Tensor(shape=(realizations, total_steps_saved, n_sites, psi_cols), h5_filepath="psiloc.h5", dtype=torch.complex64)
             else:
                if keep_on_cuda:
                  psi_loc = torch.zeros((realizations, total_steps_saved, n_sites, psi_cols), device=device, dtype=torch.complex64)
                else:
                  psi_loc = torch.zeros((realizations, total_steps_saved, n_sites, psi_cols), device="cpu", dtype=torch.complex64)
        else:
             psi_loc = None


        aranged_realizations = torch.arange(realizations, device=device)
        
        # --- Initialization Step ---
        if v_time_dependent is not None:
             efull = torch.diagonal(hfull, dim1=-2, dim2=-1)
             v_current = v_time_dependent[0, :, :, :].clone()
             current_v_index = 0
             v_next = v_time_dependent[1, :, :, :].clone()
             h = (v_current + torch.diag_embed(site_noise[0, :, :].clone() + efull)).to(device=device)
        elif constant_v:
             # hfull is (1, N, N) or (N, N). We unsqueezed it to (1, N, N).
             # site_noise[0] is (Reals, N)
             # hfull[0] is (N, N). We need (Reals, N, N).
             # h = hfull[0] + diag(noise)
             h = (hfull[0].unsqueeze(0) + torch.diag_embed(site_noise[0, :, :].clone())).to(device=device)
        else:
             h = hfull[0, :, :, :].clone().to(device=device)
        
        using_cuda = (h.device.type == "cuda")
        if using_cuda:
            torch.cuda.empty_cache()
            
        h = h.to(dtype=torch.float32)
        if using_cuda and n_sites > 32 and n_sites <= 512 and HAS_CUPY:
             e, c = CupyEigh.apply(h)
        else:
             e, c = torch.linalg.eigh(h)
        e = e.to(dtype=torch.float32)
        c = c.to(dtype=torch.float32)
        
        cold = c
        eold = e
        
        # Calculate initial population (using first vector for compatibility if K>1)
        pop0 = (psi0.abs() ** 2)[:, :, 0]
        psloc[:, 0, :] = pop0
        
        phi_b = cold.transpose(1, 2).to(dtype=torch.complex64).bmm(psi0.to(dtype=torch.complex64))
        
        if save_wavefunction:
             # Transform back to site basis for saving
             # psi(0) = c @ phi_b(0) = psi0 (since we just projected it)
             # But for consistency with loop:
             psi_loc[:, 0, :, :] = psi0.cpu()

        if save_coherence:
             for i in range(n_sites):
                  coh_loc[:, 0, i, i] = pop0[:, i]
                  
        ub = None
        if save_u:
             identity = torch.eye(n_sites, dtype=torch.complex64, device="cpu").reshape(1, n_sites, n_sites)
             uloc[:, 0, :, :] = identity.repeat(realizations, 1, 1)
             ub = cold.transpose(1, 2).to(dtype=torch.complex64).bmm(uloc[:, 0, :, :].to(device=device))

        # --- Time Loop ---
        iterator = range(1, total_steps)
        if verbose:
             iterator = tqdm.tqdm(iterator, desc="Propagation")
             
        for t in iterator:
             # Update Hamiltonian
            if v_time_dependent is not None:
                if (t * dt) // v_dt != current_v_index:
                    current_v_index = (t * dt) // v_dt
                    v_current = v_time_dependent[current_v_index, :, :, :].clone()
                    v_next = v_time_dependent[current_v_index + 1, :, :, :].clone()
                remainder = ((t * dt) % v_dt) / v_dt
                h = (v_current * (1 - remainder) + v_next * remainder) + torch.diag_embed(
                    site_noise[t, :, :].clone() + efull
                ).to(device=device)
            elif constant_v:
                # hfull[0] is (N, N), site_noise[t] is (Reals, N)
                h = (hfull[0].unsqueeze(0) + torch.diag_embed(site_noise[t, :, :].clone())).to(device=device)
            else:
                h = hfull[t, :, :, :].clone().to(device=device)

            h = h.to(dtype=torch.float32)
            if using_cuda and n_sites > 32 and n_sites <= 512 and HAS_CUPY:
                e, c = CupyEigh.apply(h)
            else:
                e, c = torch.linalg.eigh(h)
            e = e.to(dtype=torch.float32)
            c = c.to(dtype=torch.float32)
            
            # Non-adiabatic coupling
            s = torch.matmul(c.transpose(1, 2), cold)
            
            # T-Correction (Delegated check)
            if t_correction.lower() in ["mlnise", "tnise"]:
                 # We need to import apply_t_correction - circular import risk?
                 # It's better if nise.py passes it or we import it inside
                 from torchnise.nise import apply_t_correction
                 s = apply_t_correction(s, n_sites, realizations, e, eold, phi_b, aranged_realizations, params)

            # Time Evolution
            # Forward: exp(-i H t) -> factor = i * dt / hbar
            # Backward: exp(+i H t) -> factor = -i * dt / hbar
            
            effective_factor = self.factor
            if backward:
                 effective_factor = -self.factor
                 
            u = torch.diag_embed(torch.exp(-e[:, :] * effective_factor).to(dtype=torch.complex64)).bmm(s.to(dtype=torch.complex64))
            phi_b = u.bmm(phi_b)
            
            if save_u:
                 ub = u.bmm(ub)
            
            # phi_b = renorm(phi_b, dim=1) # REMOVED: Breaks non-normalized initial states (e.g. after dipole interaction)
            cold = c
            eold = e
            c = c.to(dtype=torch.complex64)
            
            phi_bin_loc_base = c.bmm(phi_b)
            
            # Save results
            if t % save_interval == 0:
                idx = t // save_interval
                # psloc saves population of FIRST vector for compatibility
                if keep_on_cuda:
                    psloc[:, idx, :] = ((phi_bin_loc_base.abs() ** 2)[:, :, 0]).real
                else:
                    psloc[:, idx, :] = ((phi_bin_loc_base.abs() ** 2)[:, :, 0]).real.cpu()
                
                if save_wavefunction:
                    if keep_on_cuda:
                        psi_loc[:, idx, :, :] = phi_bin_loc_base
                    else:
                        psi_loc[:, idx, :, :] = phi_bin_loc_base.cpu()

                if save_u:
                    for i in range(n_sites):
                        ub_norm_row = renorm(ub[:, :, i], dim=1)
                        ub[:, :, i] = ub_norm_row[:, :]
                    uloc[:, idx, :, :] = c.bmm(ub).cpu()
                if save_coherence:
                     # Coherence of first vector
                     coh_loc[:, idx, :, :] = (
                        phi_bin_loc_base.squeeze(-1)[:, :, None]
                        * phi_bin_loc_base.squeeze(-1)[:, None, :].conj()
                    ).real.cpu()

        if using_cuda:
            torch.cuda.empty_cache()
        if keep_on_cuda:
            return psloc, (coh_loc if save_coherence else None), (uloc if save_u else None), (psi_loc if save_wavefunction else None)
        return psloc.cpu(), (coh_loc.cpu() if save_coherence else None), (uloc.cpu() if save_u else None), (psi_loc.cpu() if save_wavefunction else None)

class RK4Propagator(NISEPropagator):
    """
    Runge-Kutta 4th Order Propagator.
    Supports matrix-vector multiplication or custom Hamiltonian application function.
    """
    def __init__(self, n_sites, realizations, params, apply_h_fn=None):
        super().__init__(n_sites, realizations, params)
        self.apply_h_fn = apply_h_fn

    def _apply_h(self, psi, h):
        """Helper to apply H to psi."""
        if self.apply_h_fn is not None:
             # h might be a tuple of args? Or just h matrix/tensor?
             # For 2DIR vector mode, h is h_singles matrix but applied to Doubles psi via function.
             return self.apply_h_fn(psi, h)
        else:
             # Standard Matrix Multiplication
             # psi: (R, N) or (R, N, K)
             # h: (R, N, N) or (N, N)
             h = h.to(psi.dtype)
             if psi.ndim == 2:
                 return torch.bmm(h, psi.unsqueeze(-1)).squeeze(-1)
             elif psi.ndim == 3:
                 return torch.bmm(h, psi)
             else:
                 raise ValueError(f"Unexpected psi shape {psi.shape}")

    def propagate(self, hfull, psi0, site_noise=None, v_time_dependent=None, verbose=True):
        params = self.params
        dt = params.dt
        total_time = params.total_time
        save_interval = params.save_interval
        
        realizations = self.realizations
        # If psi0 has K cols
        if psi0.ndim == 1:
            psi0 = psi0.unsqueeze(0).expand(realizations, -1)
        if psi0.ndim == 2 and psi0.shape[0] != realizations:
             # Assume (N, K) -> (R, N, K)
             psi0 = psi0.unsqueeze(0).expand(realizations, -1, -1).clone()
        
        # determine dims
        psi_shape = list(psi0.shape) # (R, N, K?)
        
        total_steps = round(total_time / dt + 1e-6) + 1
        save_steps = round(total_time / dt / save_interval + 1e-6) + 1
        
        # We always save wavefunction in this specialized propagator for now
        save_wavefunction = True 
        
        # Buffers
        psi_loc = torch.zeros((realizations, save_steps, *psi_shape[1:]), dtype=torch.complex64, device="cpu")
        
        current_psi = psi0.to(self.device).to(torch.complex64)
        
        # Save t=0
        psi_loc[:, 0] = current_psi.cpu()

        time_dependent_h = (hfull.ndim >= 3) and (not params.constant_v)
        
        # Unit scaling for derivative (-i/hbar * H * psi)
        # StandardPropagator uses self.factor which includes dt.
        # Here we need derivative k, so we divide by dt or just use 1/hbar.
        scale_factor = (1.0 / units.HBAR) * units.T_UNIT
        
        # Time Loop
        iterator = range(1, total_steps)
        if verbose:
             iterator = tqdm.tqdm(iterator, desc="RK4 Propagation", leave=False)
             
        for t in iterator:
             # Get H(t)
             if time_dependent_h:
                  h_t = hfull[t].to(self.device)
             else:
                  h_t = hfull[0].to(self.device) # Constant
                  
             # Add Noise if present
             # Warning: For vector mode, noise must be inside hfull or handled by apply_h_fn 
             # (not implemented here generically, assuming hfull is sufficient)
             
             # RK4 Steps
             # k1 = -i/hbar * H(t) * psi(t)
             
             tensor_dt = torch.tensor(dt, device=self.device, dtype=torch.complex64)
             
             # k1
             h_psi = self._apply_h(current_psi, h_t)
             k1 = -1j * scale_factor * h_psi 
             
             # k2
             psi_k1 = current_psi + 0.5 * tensor_dt * k1
             k2 = -1j * scale_factor * self._apply_h(psi_k1, h_t)
             
             # k3
             psi_k2 = current_psi + 0.5 * tensor_dt * k2
             k3 = -1j * scale_factor * self._apply_h(psi_k2, h_t)
             
             # k4
             psi_k3 = current_psi + tensor_dt * k3
             k4 = -1j * scale_factor * self._apply_h(psi_k3, h_t)
             
             # Update
             current_psi = current_psi + (tensor_dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
             
             # Save
             if t % save_interval == 0:
                  idx = t // save_interval
                  psi_loc[:, idx] = current_psi.cpu()
                  
        return None, None, None, psi_loc
