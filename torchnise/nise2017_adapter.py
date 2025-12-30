import os
import struct
import numpy as np
import torch
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Type
from torchnise.nise import NISEParameters, run_nise
from torchnise import units

@dataclass
class NISE2017Config:
    technique: str = ""
    propagation: str = "Sparse"
    coupling_cut: float = 0.0
    threshold: float = 0.0
    hamiltonian_file: str = ""
    dipole_file: str = ""
    anharmonic_file: str = ""
    overtone_dipole_file: str = ""
    position_file: str = ""
    coupling_file: str = ""
    length: int = 0
    sample_rate: int = 1
    lifetime: float = 0.0
    fft: int = 0
    tmax1: int = 0
    tmax2: int = 0
    tmax3: int = 0
    begin_point: int = 0
    end_point: int = 1
    singles: int = 0
    doubles: int = 0
    basis: str = "Local"
    temperature: float = 300.0
    min1: float = 0.0
    min2: float = 0.0
    min3: float = 0.0
    max1: float = 0.0
    max2: float = 0.0
    max3: float = 0.0
    homogen: float = 0.0
    inhomogen: float = 0.0
    initial_state_site: int = 0 # 0-indexed internally, 1-indexed in NISE
    projection_sites: List[int] = field(default_factory=list)

class NISEInputParser:
    """Parses NISE 2017 input files."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.config = NISE2017Config()
        self.base_dir = os.path.dirname(file_path)

    def parse(self) -> NISE2017Config:
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('#'):
                i += 1
                continue
            
            parts = line.split()
            if not parts:
                i += 1
                continue
                
            keyword = parts[0]
            
            if keyword == "Projection":
                 i = self._handle_projection(lines, i, self.config)
            else:
                 self._handle_simple_keyword(keyword, parts, self.config)
                 i += 1
                 
        return self.config

    def _handle_simple_keyword(self, keyword, parts, config):
        if keyword == "Technique":
            config.technique = parts[1]
        elif keyword == "Propagation":
            config.propagation = parts[1]
        elif keyword == "Couplingcut":
            config.coupling_cut = float(parts[1])
        elif keyword == "Threshold":
            config.threshold = float(parts[1])
        elif keyword == "Hamiltonianfile":
            config.hamiltonian_file = os.path.join(self.base_dir, parts[1])
        elif keyword == "Dipolefile":
            config.dipole_file = os.path.join(self.base_dir, parts[1])
        elif keyword == "Anharmonicfile":
            config.anharmonic_file = os.path.join(self.base_dir, parts[1])
        elif keyword == "Overtonedipolefile":
            config.overtone_dipole_file = os.path.join(self.base_dir, parts[1])
        elif keyword == "Positionfile":
            config.position_file = os.path.join(self.base_dir, parts[1])
        elif keyword == "Couplingfile":
            config.coupling_file = os.path.join(self.base_dir, parts[1])
        elif keyword == "Length":
            config.length = int(parts[1])
        elif keyword == "Samplerate":
            config.sample_rate = int(parts[1])
        elif keyword == "Lifetime":
            config.lifetime = float(parts[1])
        elif keyword == "Homogen":
            config.homogen = float(parts[1])
        elif keyword == "Inhomogen":
            config.inhomogen = float(parts[1])
        elif keyword == "Timestep":
            config.timestep = float(parts[1])
        elif keyword == "Anharmonicity":
            config.anharmonicity = float(parts[1])
        elif keyword == "FFT":
            config.fft = int(parts[1])
        elif keyword == "RunTimes":
            config.tmax1 = int(parts[1])
            if len(parts) > 2: config.tmax2 = int(parts[2])
            if len(parts) > 3: config.tmax3 = int(parts[3])
        elif keyword == "MinFrequencies":
            config.min1 = float(parts[1])
            if len(parts) > 2: config.min2 = float(parts[2])
            if len(parts) > 3: config.min3 = float(parts[3])
        elif keyword == "MaxFrequencies":
            config.max1 = float(parts[1])
            if len(parts) > 2: config.max2 = float(parts[2])
            if len(parts) > 3: config.max3 = float(parts[3])
        elif keyword in ["BeginPoint", "BeginSample"]:
            config.begin_point = int(parts[1])
        elif keyword == "EndPoint":
            config.end_point = int(parts[1])
        elif keyword == "Singles":
            config.singles = int(parts[1])
        elif keyword == "Doubles":
            config.doubles = int(parts[1])
        elif keyword == "Basis":
            config.basis = parts[1]
        elif keyword == "Temperature":
            config.temperature = float(parts[1])
        elif keyword == "InitialState":
            config.initial_state_site = int(parts[1]) - 1

    def _handle_projection(self, lines, i, config):
        # Current logic from procedural function:
        # i is at "Projection" line
        # Check next line
        i += 1 
        if i < len(lines):
            proj_line = lines[i].strip()
            if proj_line.startswith("Sites"):
                num_proj = int(proj_line.split()[1])
                proj_sites = []
                while len(proj_sites) < num_proj and i + 1 < len(lines):
                    i += 1
                    proj_sites.extend([int(x) for x in lines[i].split()])
                config.projection_sites = proj_sites
        # i is now at the last line consumed.
        # Outer loop does NOT increment i if we return it? 
        # Actually outer loop continues. so we should return new i + 1 usually.
        # Let's say we return the index of the last line processed.
        # So next iteration starts at i+1?
        # My outer loop logic was: i += 1 at end if not special.
        # If special, we return the NEW i.
        return i + 1

class NISEBinaryLoader:
    """Handles reading of NISE 2017 binary files."""
    
    def __init__(self, config: NISE2017Config):
        self.config = config
        
    def load_hamiltonian(self) -> torch.Tensor:
        """
        Loads Hamiltonian. 
        TODO: Future extension to return (singles, doubles) tuple if needed.
        Currently returns Singles block reshaped to full matrix.
        """
        file_path = self.config.hamiltonian_file
        n_sites = self.config.singles
        n_doubles = self.config.doubles
        begin = self.config.begin_point
        end_point = self.config.end_point
        sample_rate = self.config.sample_rate
        tmax = self.config.tmax1
        
        num_realizations = end_point - begin
        h_full = torch.zeros((tmax, num_realizations, n_sites, n_sites), dtype=torch.float32)
        
        n_tri_singles = n_sites * (n_sites + 1) // 2
        n_tri_doubles = n_doubles * (n_doubles + 1) // 2
        
        # Stride includes both single and double blocks
        stride = 4 + 4 * (n_tri_singles + n_tri_doubles) 
        
        with open(file_path, 'rb') as f:
            for r in range(num_realizations):
                base_pos = (begin + r) * sample_rate
                for t in range(tmax):
                    pos = (base_pos + t) * stride
                    f.seek(pos)
                    
                    f.read(4) # Time index
                    
                    # Read singles
                    data = f.read(4 * n_tri_singles)
                    if not data:
                        raise EOFError(f"Unexpected EOF in {file_path}")
                    h_tri = struct.unpack(f"{n_tri_singles}f", data)
                    
                    # Map to full matrix
                    for i in range(n_sites):
                        for j in range(i, n_sites):
                            idx = j + n_sites * i - (i * (i + 1) // 2)
                            val = h_tri[idx]
                            h_full[t, r, i, j] = val
                            h_full[t, r, j, i] = val
                            
                    # Note: We skip Doubles data here, but it's physically present in the file
                    # if n_doubles > 0. The stride accounts for it.
                    # If we wanted to read it, we would read 4 * n_tri_doubles next.
        
        return h_full

    def load_dipole(self) -> torch.Tensor:
        file_path = self.config.dipole_file
        n_sites = self.config.singles
        begin = self.config.begin_point
        end_point = self.config.end_point
        sample_rate = self.config.sample_rate
        tmax = self.config.tmax1

        num_realizations = end_point - begin
        mu = torch.zeros((tmax, num_realizations, n_sites, 3), dtype=torch.float32)
        
        stride = 4 + 4 * (3 * n_sites)
        
        with open(file_path, 'rb') as f:
            for r in range(num_realizations):
                base_pos = (begin + r) * sample_rate
                for t in range(tmax):
                    pos = (base_pos + t) * stride
                    f.seek(pos)
                    f.read(4) 
                    data = f.read(4 * 3 * n_sites)
                    if not data:
                        break
                    vals = struct.unpack(f"{3*n_sites}f", data)
                    mu[t, r, :, 0] = torch.tensor(vals[0:n_sites])
                    mu[t, r, :, 1] = torch.tensor(vals[n_sites:2*n_sites])
                    mu[t, r, :, 2] = torch.tensor(vals[2*n_sites:3*n_sites])
        return mu

    def load_positions(self) -> torch.Tensor:
        file_path = self.config.position_file
        n_sites = self.config.singles
        length = self.config.length
        
        pos = torch.zeros((length, n_sites, 3), dtype=torch.float32)
        with open(file_path, 'rb') as f:
            box_size_data = f.read(4)
            if not box_size_data:
                 return pos, 0.0
            box_size = struct.unpack('f', box_size_data)[0]
            data = f.read(4 * length * n_sites * 3)
            if data:
                vals = struct.unpack(f"{length * n_sites * 3}f", data)
                for t in range(length):
                    base = t * n_sites * 3
                    pos[t, :, 0] = torch.tensor(vals[base : base + n_sites])
                    pos[t, :, 1] = torch.tensor(vals[base + n_sites : base + 2 * n_sites])
                    pos[t, :, 2] = torch.tensor(vals[base + 2 * n_sites : base + 3 * n_sites])
        return pos, box_size

    def load_hamiltonian_diagonal(self) -> np.ndarray:
        """Loads only the diagonal of the Hamiltonian for the full trajectory."""
        file_path = self.config.hamiltonian_file
        n_sites = self.config.singles
        n_doubles = self.config.doubles
        length = self.config.length
        
        n_tri_singles = n_sites * (n_sites + 1) // 2
        n_tri_doubles = n_doubles * (n_doubles + 1) // 2
        stride = 4 + 4 * (n_tri_singles + n_tri_doubles)
        
        freq_traj = np.zeros((length, n_sites))
        with open(file_path, 'rb') as f:
            for t in range(length):
                f.seek(t * stride + 4) # Skip time
                data = f.read(4 * n_sites) # Read first N floats (diagonal)
                if not data:
                    break
                freq_traj[t] = struct.unpack(f"{n_sites}f", data)
        return freq_traj


def save_nise2017_output(u, total_time, dt):
    """
    Saves output in NISE 2017 format (Pop.dat and PopF.dat).
    u: Time evolution operator tensor of shape (realizations, steps, n_sites, n_sites)
    """
    # u[r, t, b, a] is the amplitude <b|U(t)|a>
    # Probabilities: |u|^2
    probs = (torch.abs(u)**2).real # (realizations, steps, n_sites, n_sites)
    
    # Average over realizations
    probs_avg = torch.mean(probs, dim=0) # (steps, n_sites, n_sites)
    
    n_steps = probs_avg.shape[0]
    n_sites = probs_avg.shape[1]
    
    time_axis = np.linspace(0, total_time, n_steps)
    
    # Pop.dat: Average population over all sites
    # NISE 2017 calculates Pop[t1] = sum_a |<a|U|a>|^2 / N
    # Actually, looking at population.c:
    # Pop[t1]+=vecr[a+a*non->singles]*vecr[a+a*non->singles];
    # fprintf(outone,"%f %e\n",t1*non->deltat,Pop[t1]/samples/non->singles);
    # This is the average of diagonal elements of |U|^2
    pop_total = torch.zeros(n_steps)
    for i in range(n_sites):
        pop_total += probs_avg[:, i, i]
    pop_total /= n_sites
    
    with open("Pop.dat", 'w') as f:
        for t_val, p in zip(time_axis, pop_total):
            f.write(f"{t_val:.6f} {p:.6e}\n")
            
    # PopF.dat: All site-to-site populations
    # NISE 2017 PopF[t1+(non->singles*b+a)*non->tmax] = |<a|U|b>|^2
    # Wait, in NISE 2017: PopF[t1+(non->singles*b+a)*non->tmax]+=vecr[a+b*non->singles]...
    # vecr[a+b*N] is <a|U|b>
    
    with open("PopF.dat", 'w') as f:
        for t in range(n_steps):
            f.write(f"{time_axis[t]:.6f} ")
            # Follow NISE 2017 order: for a in singles: for b in singles:
            # Note: TorchNISE u is [r, t, b, a] where b is final, a is initial
            # NISE 2017 vecr[a+b*N] where a is final, b is initial?
            # Let's check propagate_matrix.
            for a in range(n_sites):
                for b in range(n_sites):
                    # We want P(b -> a) if NISE index is a+b*N
                    f.write(f"{probs_avg[t, a, b]:.6e} ")
            f.write("\n")


def save_nise2017_absorption(avg_s1, config: NISE2017Config):
    """Saves Absorption.dat and TD_Absorption.dat in NISE 2017 format."""
    total_time = (config.tmax1 - 1) * config.timestep
    n_steps = config.tmax1
    time_axis = np.linspace(0, total_time, n_steps)
    
    # 1. Save TD_Absorption.dat
    # format: time Re(S1) Im(S1)
    # NISE 2017 avg s1 is normalized by realizations.
    with open("TD_Absorption.dat", 'w') as f:
        for t in range(n_steps):
            f.write(f"{time_axis[t]:.6f} {avg_s1[t].real:.6e} {avg_s1[t].imag:.6e}\n")
            
    # 2. Fourier Transform for Absorption.dat
    # Apply damping if specified
    damp = torch.ones(n_steps, dtype=torch.complex64)
    # NISE 2017 1DFFT.c: exp(-i*non->deltat/(2*non->lifetime))
    # Note: 2*lifetime? Wait. Yes, it seems they use it for E-field damping which is half of population damping.
    # Actually TorchNISE absorption damping is also configurable.
    for i in range(n_steps):
        t_val = i * config.timestep
        if config.lifetime > 0:
            damp[i] *= np.exp(-t_val / (2 * config.lifetime))
        if config.homogen > 0:
             damp[i] *= np.exp(-t_val / (2 * config.homogen))
        if config.inhomogen > 0:
             damp[i] *= np.exp(- (t_val**2) / (2 * config.inhomogen**2))
             
    s1_damped = avg_s1 * damp
    
    # FFT Padding
    fft_len = config.fft if config.fft >= n_steps else n_steps
    # NISE 2017 Scale first point (trapezoidal rule)
    s1_padded = torch.zeros(fft_len, dtype=torch.complex64)
    s1_padded[:n_steps] = s1_damped
    s1_padded[0] *= 0.5
    
    # NISE 2017 does: fftIn[i][0] = Im, fftIn[i][1] = Re.
    # Spectrum = fftOut[i][1].
    # This corresponds to Re(FT(s1)).
    # We can just use np.fft.fft
    spec_f = np.fft.fft(s1_padded.numpy())
    
    # Frequency axis
    # NISE 2017: -((fft-i)/non->deltat/c_v/fft-shift1)
    # c_v = 2.99792458e-5 cm/fs
    c_v_fs = 2.99792458e-5 # cm/fs
    freq_indices = np.arange(fft_len)
    # NISE 2017 1DFFT logic for freq mapping:
    shift1 = (config.min1 + config.max1) / 2.0
    
    # Replicate the loop from lines 201-220 in 1DFFT.c
    freqs = []
    vals_re = []
    vals_im = []
    
    for i in range(fft_len):
        if i >= fft_len // 2:
            f_val = -((fft_len - i) / (config.timestep * c_v_fs * fft_len) - shift1)
        else:
            f_val = -((-i) / (config.timestep * c_v_fs * fft_len) - shift1)
            
        if config.min1 <= f_val <= config.max1:
            # We want Re(FT(s1)) logic.
            # fftOut[i][1] in NISE corresponds to Re(FFT(Im + iRe)) = Im * sin + Re * cos = Re(s1 * e^-iwt)
            # Standard FFT in numpy is sum s1 * e^-i2pi kn/N
            # If we want Re(s1 * e^-iwt), we take the real part of the numpy FFT output?
            # Wait, numpy FFT(s) = sum s(n) exp(-i 2pi k n / N)
            # Re(sum s exp(-iwt)) = sum (Re s cos + Im s sin). 
            # Yes, it's just the real part of numpy FFT.
            freqs.append(f_val)
            vals_re.append(spec_f[i].real)
            vals_im.append(spec_f[i].imag)
            
    # Sort by frequency
    idx = np.argsort(freqs)
    with open("Absorption.dat", 'w') as f:
        for i in idx:
            f.write(f"{freqs[i]:.6f} {vals_re[i]:.6e} {vals_im[i]:.6e}\n")

class NISECalculation:
    """Base class for all NISE 2017 calculation modes."""
    def __init__(self, config: NISE2017Config):
        self.config = config
        self.loader = NISEBinaryLoader(config)
        
    def run(self):
        raise NotImplementedError("run() method must be implemented by subclasses")

    def validate_propagation(self):
        if self.config.propagation not in ["Sparse", "RK4"]:
            warnings.warn(f"NISE 2017 Propagation method '{self.config.propagation}' is not directly supported. Falling back to default TorchNISE propagation.")

class PropagationBasedCalculation(NISECalculation):
    """Base class for calculations involving time propagation (Population, Diffusion, Absorption, etc.)."""
    
    def get_common_params(self, mode="Population"):
        total_time = (self.config.tmax1 - 1) * self.config.timestep
        return NISEParameters(
            dt=self.config.timestep,
            total_time=total_time,
            temperature=self.config.temperature,
            mode=mode,
            save_interval=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            save_u=True 
        )

    def load_and_prep_hamiltonian(self):
        print(f"Loading Hamiltonian from {self.config.hamiltonian_file}")
        if not os.path.exists(self.config.hamiltonian_file):
            raise FileNotFoundError(f"Hamiltonian file not found: {self.config.hamiltonian_file}")
            
        h_nise = self.loader.load_hamiltonian()
        
        # SHIFT: NISE 2017 uses H(t) to propagate from t to t+1.
        # TorchNISE uses hfull[t] to propagate from t-1 to t.
        h_torch = torch.zeros_like(h_nise)
        if self.config.tmax1 > 1:
            h_torch[1:] = h_nise[:-1]
        h_torch[0] = h_nise[0] 
        return h_torch, h_nise # Return original too if needed for modifications

    def run_propagation(self, h_torch, params, initial_state=None):
        if initial_state is None:
            initial_state = torch.zeros(self.config.singles)
            initial_state[0] = 1.0
            
        num_realizations = self.config.end_point - self.config.begin_point
        from torchnise.nise import nise_propagate
        print(f"Running NISE calculation on {params.device}...")
        return nise_propagate(h_torch, num_realizations, initial_state, params)

class PopulationCalculation(PropagationBasedCalculation):
    def run(self):
        self.validate_propagation()
        params = self.get_common_params(mode="Population")
        h_torch, _ = self.load_and_prep_hamiltonian()
        
        # Determine initial state
        initial_state = torch.zeros(self.config.singles)
        # NISE 2017 initial state site is already 0-indexed in config
        site_idx = self.config.initial_state_site
        if site_idx < 0 or site_idx >= self.config.singles:
             # Fallback or default
             site_idx = 0
        initial_state[site_idx] = 1.0

        population, coherence, u = self.run_propagation(h_torch, params, initial_state)
        
        print("Saving output to Pop.dat and PopF.dat")
        total_time = (self.config.tmax1 - 1) * self.config.timestep
        save_nise2017_output(u, total_time, self.config.timestep)
        print("Done.")

class ResponseFunctionCalculation(PropagationBasedCalculation):
    """Base class for Response Function calculations (Absorption, 2DES, etc.)."""
    pass

class AbsorptionCalculation(ResponseFunctionCalculation):
    def run(self):
        self.validate_propagation()
        params = self.get_common_params(mode="Absorption")
        h_torch, h_nise = self.load_and_prep_hamiltonian()
        
        # Load Dipole
        print(f"Loading Dipole from {self.config.dipole_file}")
        mu_nise = self.loader.load_dipole()
        mu_torch = torch.zeros_like(mu_nise)
        if self.config.tmax1 > 1:
            mu_torch[1:] = mu_nise[:-1]
        mu_torch[0] = mu_nise[0]
        params.mu = mu_torch.cpu().numpy()

        # Rotating Frame Shift
        shifte = (self.config.min1 + self.config.max1) / 2.0
        print(f"Applying rotating frame shift: {shifte} cm-1")
        # Apply shift to h_torch
        # Original code applied shift to h_nise then copied. We can apply to h_torch directly.
        for i in range(self.config.singles):
            h_torch[:, :, i, i] -= shifte
            
        initial_state = torch.zeros(self.config.singles)
        initial_state[0] = 1.0 # Standard start? usually arbitrary for absorption if dipoles handled correctly
        
        population, coherence, u = self.run_propagation(h_torch, params, initial_state)
        
        print("Calculating time-domain absorption...")
        from torchnise.nise import absorption_time_domain
        # mu_torch is (steps, realizations, n, 3), need (realizations, steps, n, 3)
        mu_for_absorb = mu_torch.permute(1, 0, 2, 3).cpu().numpy()
        avg_s1 = absorption_time_domain(u.cpu().numpy(), mu_for_absorb, dt=self.config.timestep)
        
        print("Saving output to Absorption.dat and TD_Absorption.dat")
        save_nise2017_absorption(torch.tensor(avg_s1), self.config)
        print("Done.")

class TwoDESCalculation(ResponseFunctionCalculation):
    """
    Placeholder for 2D Electronic Spectroscopy (2DES) calculation.
    """
    def run(self):
        raise NotImplementedError("2D Spectroscopy is not yet implemented.")

class DiffusionCalculation(PropagationBasedCalculation):
    def run(self):
        # We need custom logic because it runs propagation multiple times (per site)
        params = self.get_common_params(mode="Population")
        h_torch, _ = self.load_and_prep_hamiltonian()
        
        pos_traj, box_size = self.loader.load_positions()
        
        num_realizations = self.config.end_point - self.config.begin_point
        msd_pop = torch.zeros(self.config.tmax1)
        
        print(f"Running Diffusion calculation on {params.device}...")
        
        for start_site in range(self.config.singles):
            initial_state = torch.zeros(self.config.singles)
            initial_state[start_site] = 1.0
            
            _, _, u = self.run_propagation(h_torch, params, initial_state)
            probs = (torch.abs(u)**2).real.cpu() # (real, t, b, a)
            
            for r in range(num_realizations):
                ti = (self.config.begin_point + r) * self.config.sample_rate
                p0 = pos_traj[ti] 
                for t in range(self.config.tmax1):
                    tj = ti + t
                    pt = pos_traj[tj]
                    for b in range(self.config.singles):
                        dx = pt[b, 0] - p0[start_site, 0]
                        dy = pt[b, 1] - p0[start_site, 1]
                        dz = pt[b, 2] - p0[start_site, 2]
                        if box_size > 0:
                            dx -= box_size * torch.round(dx / box_size)
                            dy -= box_size * torch.round(dy / box_size)
                            dz -= box_size * torch.round(dz / box_size)
                        d2 = dx*dx + dy*dy + dz*dz
                        msd_pop[t] += (probs[r, t, b, 0] * d2)
                        
        norm = num_realizations * self.config.singles
        msd_pop /= norm
        
        with open("RMSD.dat", 'w') as f:
            for t in range(self.config.tmax1):
                f.write(f"{t*self.config.timestep:.6f} {msd_pop[t]:.6e} 0.000000e+00\n")
        print("Done.")

class StaticPropertyCalculation(NISECalculation):
    """Base class for calculations that analyse trajectories without propagation."""
    pass

class CorrelationCalculation(StaticPropertyCalculation):
    def run(self):
        from torchnise.spectral_density_generation import get_auto, get_cross, sd_reconstruct_fft
        
        freq_traj = self.loader.load_hamiltonian_diagonal()
        sites = self.config.singles
        length = self.config.length
                
        T = self.config.tmax1
        dt = self.config.timestep
        auto_matrix = np.zeros((sites, sites, length))
        
        print("Calculating Correlation...")
        for i in range(sites):
            for j in range(i, sites):
                data_i = freq_traj[:, i] - np.mean(freq_traj[:, i])
                data_j = freq_traj[:, j] - np.mean(freq_traj[:, j])
                
                noise_i = data_i.reshape(1, -1)
                noise_j = data_j.reshape(1, -1)
                
                if i == j:
                    auto = get_auto(noise_i)
                    auto_matrix[i, i, :len(auto)] = auto
                else:
                    if self.config.technique == "Correlation":
                        cross = get_cross(noise_i, noise_j)
                        auto_matrix[i, j, :len(cross)] = cross
                        auto_matrix[j, i, :len(cross)] = cross
                        
        w_axis = None
        sd_results = []
        if sites > 0:
            for i in range(sites):
                 auto = auto_matrix[i, i, :length//2]
                 J_new, curr_w_axis, _ = sd_reconstruct_fft(
                      auto, dt, self.config.temperature, damping_type="gauss", cutoff=T*dt
                 )
                 sd_results.append(J_new)
                 w_axis = curr_w_axis
            
        if w_axis is not None:
            with open("SpectralDensity.dat", 'w') as f:
                for i in range(len(w_axis)):
                     line = f"{w_axis[i]:.6f} " + " ".join([f"{sd_results[s][i]:.6e}" for s in range(sites)]) + "\n"
                     f.write(line)

        with open("CorrelationMatrix.dat", 'w') as f:
            for t in range(T):
                line = f"{t*dt:.6f} "
                for i in range(sites):
                    for j in range(i, sites):
                        line += f"{auto_matrix[i, j, t]:.6e} "
                        if self.config.technique == "Autocorrelation":
                            break 
                f.write(line + "\n")
        print("Done.")

# Registry of available calculation modes
CALCULATION_MODES: Dict[str, Type[NISECalculation]] = {
    "Pop": PopulationCalculation,
    "Population": PopulationCalculation,
    "Absorption": AbsorptionCalculation,
    "Correlation": CorrelationCalculation,
    "Autocorrelation": CorrelationCalculation,
    "Dif": DiffusionCalculation,
    "Diffusion": DiffusionCalculation,
    "2DES": TwoDESCalculation, # Placeholder
}

# List of all known techniques in NISE 2017 for validation
ALL_NISE_TECHNIQUES = [
    "Pop", "Population", "Analyse", "Analyze", "AnalyseFull", "AnalyzeFull",
    "Correlation", "Autocorrelation", "Dif", "Diffusion", "Ani", "Anisotropy",
    "Absorption", "DOS", "MCFRET", "Redfield", "Luminescence", "PL", "Fluorescence",
    "LD", "CD", "Raman", "SFG", "2DES"
]


def run_nise2017(input_file_path: str):
    print(f"Loading NISE 2017 input: {input_file_path}")
    parser = NISEInputParser(input_file_path)
    config = parser.parse()
    
    technique = config.technique
    if technique not in CALCULATION_MODES:
        if technique in ALL_NISE_TECHNIQUES:
            raise NotImplementedError(f"Technique '{technique}' is a valid NISE 2017 technique but is not yet implemented in the TorchNISE adapter.")
        else:
            raise ValueError(f"Unknown NISE 2017 technique: '{technique}'")
            
    calc_class = CALCULATION_MODES[technique]
    calculation = calc_class(config)
    calculation.run()

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python nise2017_adapter.py <input_file>")
        return
        
    input_file = sys.argv[1]
    run_nise2017(input_file)

if __name__ == "__main__":
    main()
