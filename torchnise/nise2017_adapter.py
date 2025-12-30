import os
import struct
import numpy as np
import torch
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
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

def parse_nise2017_input(file_path: str) -> NISE2017Config:
    config = NISE2017Config()
    base_dir = os.path.dirname(file_path)
    with open(file_path, 'r') as f:
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
        
        if keyword == "Technique":
            config.technique = parts[1]
        elif keyword == "Propagation":
            config.propagation = parts[1]
        elif keyword == "Couplingcut":
            config.coupling_cut = float(parts[1])
        elif keyword == "Threshold":
            config.threshold = float(parts[1])
        elif keyword == "Hamiltonianfile":
            config.hamiltonian_file = os.path.join(base_dir, parts[1])
        elif keyword == "Dipolefile":
            config.dipole_file = os.path.join(base_dir, parts[1])
        elif keyword == "Anharmonicfile":
            config.anharmonic_file = os.path.join(base_dir, parts[1])
        elif keyword == "Overtonedipolefile":
            config.overtone_dipole_file = os.path.join(base_dir, parts[1])
        elif keyword == "Positionfile":
            config.position_file = os.path.join(base_dir, parts[1])
        elif keyword == "Couplingfile":
            config.coupling_file = os.path.join(base_dir, parts[1])
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
            # NISE uses 1-based indexing for sites? 
            # Let's check NISE source for InitialState parsing.
            # Actually NISE doesn't seem to have InitialState keyword in readinput.c
            # but it was in the benchmark script.
            config.initial_state_site = int(parts[1]) - 1
        elif keyword == "Sites":
            # This is often used inside Projection, but some inputs might have it.
            pass
        elif keyword == "Projection":
            # Complex keyword that might span multiple lines
            i += 1
            if i < len(lines):
                proj_line = lines[i].strip()
                if proj_line.startswith("Sites"):
                    num_proj = int(proj_line.split()[1])
                    # Read the next num_proj numbers
                    proj_sites = []
                    while len(proj_sites) < num_proj and i + 1 < len(lines):
                        i += 1
                        proj_sites.extend([int(x) for x in lines[i].split()])
                    config.projection_sites = proj_sites
        # Ignore unknown keywords for now
        i += 1
    
    return config

def load_nise2017_hamiltonian(file_path: str, n_sites: int, n_doubles: int, begin: int, end_point: int, sample_rate: int, tmax: int) -> torch.Tensor:
    """
    Loads Hamiltonian from NISE 2017 binary format.
    Returns a tensor of shape (tmax, realizations, n_sites, n_sites).
    """
    num_realizations = end_point - begin
    h_full = torch.zeros((tmax, num_realizations, n_sites, n_sites), dtype=torch.float32)
    
    n_tri_singles = n_sites * (n_sites + 1) // 2
    n_tri_doubles = n_doubles * (n_doubles + 1) // 2
    
    stride = 4 + 4 * (n_tri_singles + n_tri_doubles) # int + floats
    
    with open(file_path, 'rb') as f:
        for r in range(num_realizations):
            base_pos = (begin + r) * sample_rate
            for t in range(tmax):
                pos = (base_pos + t) * stride
                f.seek(pos)
                
                # Read time index
                f.read(4)
                
                # Read singles Hamiltonian (upper triangular)
                data = f.read(4 * n_tri_singles)
                if not data:
                    raise EOFError(f"Unexpected EOF in {file_path}")
                h_tri = struct.unpack(f"{n_tri_singles}f", data)
                
                # Map upper triangular to full matrix
                for i in range(n_sites):
                    for j in range(i, n_sites):
                        idx = j + n_sites * i - (i * (i + 1) // 2)
                        val = h_tri[idx]
                        h_full[t, r, i, j] = val
                        h_full[t, r, j, i] = val
    
    return h_full

def load_nise2017_dipole(file_path: str, n_sites: int, begin: int, end_point: int, sample_rate: int, tmax: int) -> torch.Tensor:
    """Reads dipole moments from NISE 2017 binary Dipole.bin format."""
    num_realizations = end_point - begin
    mu = torch.zeros((tmax, num_realizations, n_sites, 3), dtype=torch.float32)
    
    # Format: time_index (int) + 3 * n_sites (floats, mux, muy, muz)
    stride = 4 + 4 * (3 * n_sites)
    
    with open(file_path, 'rb') as f:
        for r in range(num_realizations):
            base_pos = (begin + r) * sample_rate
            for t in range(tmax):
                pos = (base_pos + t) * stride
                f.seek(pos)
                f.read(4) # skip time index
                data = f.read(4 * 3 * n_sites)
                if not data:
                    break
                vals = struct.unpack(f"{3*n_sites}f", data)
                # vals is mux1..muxN, muy1..muyN, muz1..muzN
                mu[t, r, :, 0] = torch.tensor(vals[0:n_sites])
                mu[t, r, :, 1] = torch.tensor(vals[n_sites:2*n_sites])
                mu[t, r, :, 2] = torch.tensor(vals[2*n_sites:3*n_sites])
    return mu

def load_nise2017_positions(file_path: str, n_sites: int, length: int) -> torch.Tensor:
    """Reads positions from NISE 2017 binary Position.bin format."""
    # Format: box_size (float) then length * n_sites * 3 (floats)
    pos = torch.zeros((length, n_sites, 3), dtype=torch.float32)
    with open(file_path, 'rb') as f:
        box_size = struct.unpack('f', f.read(4))[0]
        data = f.read(4 * length * n_sites * 3)
        if data:
            vals = struct.unpack(f"{length * n_sites * 3}f", data)
            # vals is x1..xN, y1..yN, z1..zN for t0, then t1...
            for t in range(length):
                base = t * n_sites * 3
                pos[t, :, 0] = torch.tensor(vals[base : base + n_sites])
                pos[t, :, 1] = torch.tensor(vals[base + n_sites : base + 2 * n_sites])
                pos[t, :, 2] = torch.tensor(vals[base + 2 * n_sites : base + 3 * n_sites])
    return pos, box_size

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

def run_nise2017(input_file_path: str):
    print(f"Loading NISE 2017 input: {input_file_path}")
    config = parse_nise2017_input(input_file_path)
    
    implemented_techniques = ["Pop", "Population"]
    all_nise_techniques = [
        "Pop", "Population", "Analyse", "Analyze", "AnalyseFull", "AnalyzeFull",
        "Correlation", "Autocorrelation", "Dif", "Diffusion", "Ani", "Anisotropy",
        "Absorption", "DOS", "MCFRET", "Redfield", "Luminescence", "PL", "Fluorescence",
        "LD", "CD", "Raman", "SFG"
    ]
    
    if config.technique not in implemented_techniques:
        if config.technique in all_nise_techniques:
            raise NotImplementedError(f"Technique '{config.technique}' is a valid NISE 2017 technique but is not yet implemented in the TorchNISE adapter.")
        else:
            raise ValueError(f"Unknown NISE 2017 technique: '{config.technique}'")
        
    if config.propagation not in ["Sparse", "RK4"]:
        # TorchNISE handles propagation differently, but "Sparse" and "RK4" in NISE 
        # are closest to what TorchNISE does (integrating the Schrödinger equation).
        warnings.warn(f"NISE 2017 Propagation method '{config.propagation}' is not directly supported. Falling back to default TorchNISE propagation.")

    # NISE 2017 tmax1 corresponds to the number of snapshots/steps.
    # The output has tmax1 lines (0 to tmax1-1).
    # Total time for the last recorded step is (tmax1 - 1) * dt.
    total_time = (config.tmax1 - 1) * config.timestep
    
    params = NISEParameters(
        dt=config.timestep,
        total_time=total_time,
        temperature=config.temperature,
        mode="Population",
        save_interval=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_u=True 
    )
    
    # Load Hamiltonian
    print(f"Loading Hamiltonian from {config.hamiltonian_file}")
    if not os.path.exists(config.hamiltonian_file):
        raise FileNotFoundError(f"Hamiltonian file not found: {config.hamiltonian_file}")
        
    # We need tmax1 snapshots from NISE.
    h_nise = load_nise2017_hamiltonian(
        config.hamiltonian_file,
        config.singles,
        config.doubles,
        config.begin_point,
        config.end_point,
        config.sample_rate,
        config.tmax1
    )
    
    # SHIFT: NISE 2017 uses H(t) to propagate from t to t+1.
    # TorchNISE uses hfull[t] to propagate from t-1 to t.
    # To match, we need h_torch[t] = h_nise[t-1] for t >= 1.
    num_realizations = config.end_point - config.begin_point
    h_torch = torch.zeros_like(h_nise)
    if config.tmax1 > 1:
        h_torch[1:] = h_nise[:-1]
    h_torch[0] = h_nise[0] # Used for initial state if needed
    
    # Initial state
    initial_state = torch.zeros(config.singles)
    initial_state[0] = 1.0 
    
    from torchnise.nise import nise_propagate
    
    print(f"Running NISE calculation on {params.device}...")
    
    population, coherence, u = nise_propagate(
        h_torch,
        num_realizations,
        initial_state,
        params
    )
    
    print("Saving output to Pop.dat and PopF.dat")
    save_nise2017_output(u, total_time, config.timestep)
    print("Done.")

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

def run_nise2017(input_file_path: str):
    print(f"Loading NISE 2017 input: {input_file_path}")
    config = parse_nise2017_input(input_file_path)
    
    implemented_techniques = ["Pop", "Population", "Absorption"]
    all_nise_techniques = [
        "Pop", "Population", "Analyse", "Analyze", "AnalyseFull", "AnalyzeFull",
        "Correlation", "Autocorrelation", "Dif", "Diffusion", "Ani", "Anisotropy",
        "Absorption", "DOS", "MCFRET", "Redfield", "Luminescence", "PL", "Fluorescence",
        "LD", "CD", "Raman", "SFG"
    ]
    
    if config.technique not in implemented_techniques:
        if config.technique in all_nise_techniques:
            raise NotImplementedError(f"Technique '{config.technique}' is a valid NISE 2017 technique but is not yet implemented in the TorchNISE adapter.")
        else:
            raise ValueError(f"Unknown NISE 2017 technique: '{config.technique}'")
        
    if config.propagation not in ["Sparse", "RK4"]:
        warnings.warn(f"NISE 2017 Propagation method '{config.propagation}' is not directly supported. Falling back to default TorchNISE propagation.")

    # NISE 2017 tmax1 corresponds to the number of snapshots/steps.
    # The output has tmax1 lines (0 to tmax1-1).
    total_time = (config.tmax1 - 1) * config.timestep
    
    params = NISEParameters(
        dt=config.timestep,
        total_time=total_time,
        temperature=config.temperature,
        mode="Population" if config.technique in ["Pop", "Population"] else "Absorption",
        save_interval=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_u=True 
    )
    
    # Load Hamiltonian
    print(f"Loading Hamiltonian from {config.hamiltonian_file}")
    if not os.path.exists(config.hamiltonian_file):
        raise FileNotFoundError(f"Hamiltonian file not found: {config.hamiltonian_file}")
        
    h_nise = load_nise2017_hamiltonian(
        config.hamiltonian_file,
        config.singles,
        config.doubles,
        config.begin_point,
        config.end_point,
        config.sample_rate,
        config.tmax1
    )
    
    # Load Dipole if Absorption
    mu_torch = None
    if config.technique == "Absorption":
        print(f"Loading Dipole from {config.dipole_file}")
        mu_nise = load_nise2017_dipole(
            config.dipole_file,
            config.singles,
            config.begin_point,
            config.end_point,
            config.sample_rate,
            config.tmax1
        )
        # Shift mu the same way as h
        mu_torch = torch.zeros_like(mu_nise)
        if config.tmax1 > 1:
            mu_torch[1:] = mu_nise[:-1]
        mu_torch[0] = mu_nise[0]
        params.mu = mu_torch.cpu().numpy() # TorchNISE expects (steps, realizations, n, 3)? 
        # Actually absorption_time_domain expects (realizations, timesteps, n, 3) 
        # but run_nise loop uses chunks. 
        # Let's check absorption_time_domain shape expectations again.
        # it says (realizations, timesteps, n, sites). Wait.
        # line 20: (realizations, timesteps, n_sites, n_sites) for U.
        # line 21: (realizations, timesteps, n_sites, 3) for mu.
        
    # SHIFT h for Rotating Frame if Absorption
    # NISE 2017 subtracts (min + max) / 2 from the diagonal
    shifte = 0.0
    if config.technique == "Absorption":
        shifte = (config.min1 + config.max1) / 2.0
        print(f"Applying rotating frame shift: {shifte} cm-1")
        for i in range(config.singles):
            h_nise[:, :, i, i] -= shifte

    # Apply Hamiltonian one-step shift
    num_realizations = config.end_point - config.begin_point
    h_torch = torch.zeros_like(h_nise)
    if config.tmax1 > 1:
        h_torch[1:] = h_nise[:-1]
    h_torch[0] = h_nise[0] 
    
    # Initial state
    initial_state = torch.zeros(config.singles)
    initial_state[0] = 1.0 
    
    from torchnise.nise import nise_propagate, absorption_time_domain
    
    print(f"Running NISE calculation on {params.device}...")
    
    population, coherence, u = nise_propagate(
        h_torch,
        num_realizations,
        initial_state,
        params
    )
    
    if config.technique in ["Pop", "Population"]:
        print("Saving output to Pop.dat and PopF.dat")
        save_nise2017_output(u, total_time, config.timestep)
    else:
        # Technique Absorption
        print("Calculating time-domain absorption...")
        # TorchNISE absorption_time_domain expects mu as (realizations, steps, n, 3)
        # Our mu_torch is (steps, realizations, n, 3)
        mu_for_absorb = mu_torch.permute(1, 0, 2, 3).cpu().numpy()
        avg_s1 = absorption_time_domain(u.cpu().numpy(), mu_for_absorb, dt=config.timestep)
        print("Saving output to Absorption.dat and TD_Absorption.dat")
        save_nise2017_absorption(torch.tensor(avg_s1), config)
    print("Done.")

def run_correlation(config: NISE2017Config):
    from torchnise.spectral_density_generation import get_auto, get_cross, sd_reconstruct_fft
    
    # Load Hamiltonian diagonal
    stride = 4 + 4 * (config.singles * (config.singles + 1) // 2 + config.doubles * (config.doubles + 1) // 2)
    sites = config.singles
    length = config.length
    
    # We load the entire trajectory for correlation
    freq_traj = np.zeros((length, sites))
    with open(config.hamiltonian_file, 'rb') as f:
        for t in range(length):
            f.seek(t * stride + 4) # Skip time index
            data = f.read(4 * sites) # Diagonal is the first 'singles' elements of the triangular matrix
            if not data:
                break
            freq_traj[t] = struct.unpack(f"{sites}f", data)
            
    # Autocorrelation
    # NISE 2017: T = config.tmax1. Total length = config.length
    T = config.tmax1
    TT = config.length
    dt = config.timestep
    
    auto_matrix = np.zeros((sites, sites, length))
    
    print("Calculating Correlation...")
    for i in range(sites):
        for j in range(i, sites):
            # Autocorrelation or Cross-correlation
            data_i = freq_traj[:, i] - np.mean(freq_traj[:, i])
            data_j = freq_traj[:, j] - np.mean(freq_traj[:, j])
            
            # get_auto expects (realizations, timesteps)
            # We treat the single trajectory as one realization if it's long enough,
            # but get_auto uses FFT-based autocorrelation.
            # Let's ensure it's (1, length) 
            noise_i = data_i.reshape(1, -1)
            noise_j = data_j.reshape(1, -1)
            
            # For cross-correlation, get_auto might not be directly applicable if it
            # only does autocorrelation. Checking get_auto source... 
            # It calls expval_auto which does result[i] = np.mean(noise[:, i:] * noise[:, :-i])
            # We can implement a simple cross-correlation here.
            
            if i == j:
                auto = get_auto(noise_i)
                # auto is length/2 by default in get_auto
                auto_matrix[i, i, :len(auto)] = auto
            else:
                if config.technique == "Correlation":
                    # Cross correlation: <di(0)dj(t)>
                    cross = get_cross(noise_i, noise_j)
                    auto_matrix[i, j, :len(cross)] = cross
                    auto_matrix[j, i, :len(cross)] = cross # Symmetry
                else:
                    # Autocorrelation mode, skip cross
                    pass
            
    # Spectral Density Reconstruction for each site
    w_axis = None
    sd_results = []
    if sites > 0:
        for i in range(sites):
             auto = auto_matrix[i, i, :length//2]
             J_new, curr_w_axis, _ = sd_reconstruct_fft(
                  auto, dt, config.temperature, damping_type="gauss", cutoff=T*dt
             )
             sd_results.append(J_new)
             w_axis = curr_w_axis
        
    # Save SpectralDensity.dat
    if w_axis is not None:
        with open("SpectralDensity.dat", 'w') as f:
            for i in range(len(w_axis)):
                 line = f"{w_axis[i]:.6f} " + " ".join([f"{sd_results[s][i]:.6e}" for s in range(sites)]) + "\n"
                 f.write(line)

    # CorrelationMatrix.dat
    with open("CorrelationMatrix.dat", 'w') as f:
        for t in range(T):
            line = f"{t*dt:.6f} "
            for i in range(sites):
                for j in range(i, sites):
                    line += f"{auto_matrix[i, j, t]:.6e} "
                    if config.technique == "Autocorrelation":
                        break # Only diagonal
            f.write(line + "\n")
            
    print("Done.")

def run_diffusion(config: NISE2017Config):
    # Load Hamiltonian and Positions
    # For diffusion, we need propagation
    total_time = (config.tmax1 - 1) * config.timestep
    params = NISEParameters(
        dt=config.timestep,
        total_time=total_time,
        temperature=config.temperature,
        mode="Population",
        save_u=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    h_nise = load_nise2017_hamiltonian(
        config.hamiltonian_file, config.singles, config.doubles,
        config.begin_point, config.end_point, config.sample_rate, config.tmax1
    )
    # Applying shift logic as in run_nise2017
    h_torch = torch.zeros_like(h_nise)
    if config.tmax1 > 1:
        h_torch[1:] = h_nise[:-1]
    h_torch[0] = h_nise[0]

    # Load Positions
    pos_traj, box_size = load_nise2017_positions(config.position_file, config.singles, config.length)
    
    # We need to run for each realization and average
    num_realizations = config.end_point - config.begin_point
    msd_pop = torch.zeros(config.tmax1)
    msd_ori = torch.zeros(config.tmax1)
    
    from torchnise.nise import nise_propagate
    
    print(f"Running Diffusion calculation on {params.device}...")
    # For diffusion, NISE 2017 seems to start multiple realizations from ALL sites?
    # actually vecr[a+a*N]=1.0 in calc_Diffusion.c loop (line 164)
    # and it does it for each 'a' in singles.
    # So it actually calculates the diffusion starting from EACH site.
    
    for start_site in range(config.singles):
        initial_state = torch.zeros(config.singles)
        initial_state[start_site] = 1.0
        
        _, _, u = nise_propagate(h_torch, num_realizations, initial_state, params)
        # u is (real, t, current_site, start_site)
        probs = (torch.abs(u)**2).real.cpu() # (real, t, b, a)
        
        # Calculate MSD
        for r in range(num_realizations):
            ti = (config.begin_point + r) * config.sample_rate
            p0 = pos_traj[ti] # shape (n_sites, 3)
            for t in range(config.tmax1):
                tj = ti + t
                pt = pos_traj[tj] # shape (n_sites, 3)
                
                # PopP: sum_b P(b,t) * dist(pos_b(t), pos_a(start))
                # Dist function in NISE correct for PBC
                # dist(pt[b], p0[a])
                for b in range(config.singles):
                    dx = pt[b, 0] - p0[start_site, 0]
                    dy = pt[b, 1] - p0[start_site, 1]
                    dz = pt[b, 2] - p0[start_site, 2]
                    # PBC
                    if box_size > 0:
                        dx -= box_size * torch.round(dx / box_size)
                        dy -= box_size * torch.round(dy / box_size)
                        dz -= box_size * torch.round(dz / box_size)
                    d2 = dx*dx + dy*dy + dz*dz
                    msd_pop[t] += (probs[r, t, b, 0] * d2) # initial site is 0 in 'initial_state' slice
                    
    # Normalization: realized samples * singles
    norm = num_realizations * config.singles
    msd_pop /= norm
    
    # Save RMSD.dat
    with open("RMSD.dat", 'w') as f:
        for t in range(config.tmax1):
            f.write(f"{t*config.timestep:.6f} {msd_pop[t]:.6e} 0.000000e+00\n")
    print("Done.")

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python nise2017_adapter.py <input_file>")
        return
        
    input_file = sys.argv[1]
    config = parse_nise2017_input(input_file)
    
    if config.technique in ["Correlation", "Autocorrelation"]:
        run_correlation(config)
    elif config.technique in ["Dif", "Diffusion"]:
        run_diffusion(config)
    else:
        run_nise2017(input_file)

if __name__ == "__main__":
    main()
