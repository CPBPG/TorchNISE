
"""
Benchmark script comparing TorchNISE (CPU/GPU) vs NISE_2017 scaling performance.
Extends to N=1000.
"""

import functools
import multiprocessing
import os
import shutil
import struct
import subprocess
import time
from typing import List, Tuple, Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch

import torchnise.nise
import torchnise.units as units
from torchnise.fft_noise_gen import gen_noise
from torchnise.nise2017_adapter import (
    NISE2017Config,
    NISEInputParser,
    PopulationCalculation,
)
from torchnise.pytorch_utility import H5Tensor
from torchnise.spectral_functions import spectral_drude_lorentz_heom



def read_popf(
    filepath: str, n_sites: int, initial_site: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Reads PopF.dat and extracts population dynamics P(initial -> final).

    Args:
        filepath: Path to the PopF.dat file.
        n_sites: Number of sites in the system.
        initial_site: The initial excitation site index (0-based).

    Returns:
        A tuple containing:
        - time_axis: Array of time points.
        - pops: Population matrix of shape (steps, n_sites).
    """
    data = np.loadtxt(filepath)
    time_axis = data[:, 0]
    raw_probs = data[:, 1:]

    # Format is for a in sites: for b in sites: P(b->a)
    # We want indices where b == initial_site
    # storage index = a * n_sites + b

    indices = np.arange(n_sites) * n_sites + initial_site
    valid_mask = indices < raw_probs.shape[1]

    pops = np.zeros((len(time_axis), n_sites))
    if np.any(valid_mask):
        pops[:, valid_mask] = raw_probs[:, indices[valid_mask]]

    return time_axis, pops

def aggregate_nise_jobs_mode(
    n_jobs: int,
    n_sites: int,
    reals_per_job: List[int],
    propagation_mode: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Aggregates results from multiple NISE jobs reading Pop.dat."""
    total_reals = sum(reals_per_job)
    weighted_pops = None
    common_time = None

    for i in range(n_jobs):
        job_dir = os.path.join(WORK_DIR, f"job_{propagation_mode}_{i}")
        t, p = read_popf(os.path.join(job_dir, "PopF.dat"), n_sites, initial_site=0)

        if t is None or p is None:
            print(f"Job {i} ({propagation_mode}) failed to produce PopF.dat output.")
            continue

        if common_time is None:
            common_time = t
            weighted_pops = np.zeros_like(p)

        # Ensure shapes match (handle potential single-step output mismatch)
        if p.shape != weighted_pops.shape:
             # Truncate to min len
             min_len = min(p.shape[0], weighted_pops.shape[0])
             p = p[:min_len]
             if common_time is not None:
                  common_time = common_time[:min_len]
                  weighted_pops = weighted_pops[:min_len]

        # Weighted average based on realizations in this job
        weight = reals_per_job[i] / total_reals
        weighted_pops += p * weight

    return common_time, weighted_pops

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
NISE_BIN = os.path.join(EXAMPLES_DIR, "NISE_2017/bin/NISE")
WORK_DIR = os.path.abspath("benchmark_workdir")

def write_binary_energy_file(
    filepath: str,
    n_sites: int,
    steps: int,
    realizations: int,
    dt: float,
    h_static: np.ndarray,
    noise: Optional[torch.Tensor] = None,
):
    """Writes Hamiltonian to NISE 2017 Binary Format.

    Format: Sequence of records.
    Each record: [Time(float)] [H_tri (N*(N+1)/2 floats)]
    H_tri is Row-Major Upper Triangular.

    Args:
        filepath: Output file path.
        n_sites: Number of sites.
        steps: Time steps per realization.
        realizations: Number of realizations.
        dt: Time step size.
        h_static: Static part of Hamiltonian (n_sites, n_sites).
        noise: Optional noise tensor (steps, realizations, n_sites).
    """
    n_tri = n_sites * (n_sites + 1) // 2
    total_records = realizations * steps

    # Initialize data array: Time + n_tri
    data = np.zeros((total_records, 1 + n_tri), dtype=np.float32)

    # Fill Time: Global index (t + r*steps)
    data[:, 0] = np.arange(total_records, dtype=np.float32)

    # Get upper triangular indices in row-major order
    iu = np.triu_indices(n_sites)

    # Static Hamiltonian triangular part
    h_tri_static = h_static[iu].astype(np.float32)  # (n_tri,)

    # Broadcast static H to all records
    data[:, 1:] = h_tri_static

    # Add noise to diagonal
    if noise is not None:
        # Diagonal indices in the triangular vector: where row == col
        diag_mask = iu[0] == iu[1]
        # noise shape: (steps, realizations, n_sites)
        # We need (realizations * steps, n_sites) to match data layout
        # permute(1, 0, 2) makes it (realizations, steps, n_sites)
        # reshape(-1, n_sites) makes it (realizations * steps, n_sites)
        noise_np = (
            noise.permute(1, 0, 2)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1, n_sites)
            .astype(np.float32)
        )
        data[:, 1:][:, diag_mask] += noise_np

    # Write to file in bulk
    data.tofile(filepath)

def read_pop(filepath: str, n_sites: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Reads Pop.dat file (Time + N site populations)."""
    if not os.path.exists(filepath):
        return None, None
    try:
        data = np.loadtxt(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None
        
    if data.ndim == 1:
        data = data.reshape(1, -1)
        
    # Expect 1 time col + n_sites cols
    expected_cols = n_sites + 1
    if data.shape[1] < expected_cols:
         print(f"Warning: {filepath} has {data.shape[1]} columns, expected {expected_cols}. Returning available columns.")
         return data[:, 0], data[:, 1:] 

    return data[:, 0], data[:, 1 : expected_cols]

def write_binary_site_energies(
    filepath: str,
    n_sites: int,
    steps: int,
    realizations: int,
    dt: float,
    noise: Optional[torch.Tensor] = None,
):
    """Writes Time-Dependent Site Energies to NISE 2017 Binary Format.

    Format: Sequence of records.
    Each record: [Time(float)] [E1 E2 ... EN (float)]
    
    Args:
        filepath: Output file path.
        n_sites: Number of sites.
        steps: Time steps per realization.
        realizations: Number of realizations.
        dt: Time step size.
        noise: Optional noise tensor (steps, realizations, n_sites).
               If None, energies are 0.0.
    """
    total_records = realizations * steps
    
    # Initialize data array: Time + n_sites
    data = np.zeros((total_records, 1 + n_sites), dtype=np.float32)
    
    # Fill Time
    data[:, 0] = np.arange(total_records, dtype=np.float32)
    
    # If using constant H diagonal 0, initially 0.
    # Add noise if present
    if noise is not None:
        # noise shape: (steps, realizations, n_sites)
        # Target: (realizations * steps, n_sites)
        noise_np = (
            noise.permute(1, 0, 2)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1, n_sites)
            .astype(np.float32)
        )
        data[:, 1:] = noise_np
        
    data.tofile(filepath)

def write_binary_static_coupling(
    filepath: str,
    h_static: np.ndarray,
):
    """Writes Static Coupling Matrix to NISE 2017 Binary Format.
    
    Format: [H_tri] (N*(N+1)/2 floats)
    Row-Major Upper Triangular.
    """
    n_sites = h_static.shape[0]
    iu = np.triu_indices(n_sites)
    h_tri = h_static[iu].astype(np.float32)
    h_tri.tofile(filepath)

def write_binary_dipole_file(
    filepath: str, n_sites: int, total_steps: int, dt: float
):
    """Writes constant dipole 1.0 for x-axis to NISE 2017 Format.

    Args:
        filepath: Output file path.
        n_sites: Number of sites.
        total_steps: Total number of time steps (all realizations).
        dt: Time step size.
    """
    # Dipole: [Time] [Mu_0x Mu_0y Mu_0z ...]
    # 1 + 3*N floats per record

    data = np.zeros((total_steps, 1 + 3 * n_sites), dtype=np.float32)
    data[:, 0] = np.arange(total_steps, dtype=np.float32)

    # x=1.0, y=0.0, z=0.0 for all sites
    # Pattern: [T, s0x, s0y, s0z, s1x, s1y, s1z, ...]
    # s_ix is at index 1 + 3*i
    data[:, 1::3] = 1.0

    data.tofile(filepath)

def generate_nise_inputs(
    n_sites: int,
    steps: int,
    dt: float,
    realizations: int,
    omega_k: List[float],
    lambda_k: List[float],
    v_k: List[float],
    temperature: float,
    n_neighbours: int = 4,
) -> int:
    """Generates NISE 2017 binary input files (Energy.bin, Dipole.bin).

    Args:
        n_sites: Number of sites.
        steps: Time steps per realization.
        dt: Time step.
        realizations: Number of realizations.
        omega_k: Drude-Lorentz frequencies.
        lambda_k: Reorganization energies.
        v_k: Inverse timescales.
        temperature: Temperature in Kelvin.

    Returns:
        Total length (steps * realizations).
    """
    os.makedirs(WORK_DIR, exist_ok=True)

    total_len = steps * realizations

    # 1. Generate Noise
    print(
        f"Generating noise for N={n_sites}, Steps={steps}, Reals={realizations}..."
    )

    # Params
    spectral_func = functools.partial(
        spectral_drude_lorentz_heom,
        omega_k=torch.tensor(omega_k) / units.HBAR,  # Convert cm-1 if needed
        lambda_k=torch.tensor(lambda_k),
        vk=torch.tensor(v_k),
        temperature=temperature,
    )

    spectral_funcs = [spectral_func] * n_sites

    # Shape: (steps, reals, n_sites)
    noise = gen_noise(
        spectral_funcs,
        dt,
        shape=(steps, realizations, n_sites),
        use_h5=False,
        device="cpu",
    )

    # 2. Static Hamiltonian
    h_step = np.zeros((n_sites, n_sites))
    # Simple nearest neighbor coupling with halving scheme
    base_coupling = 100.0
    for k in range(1, n_neighbours + 1):
        coupling_val = base_coupling / (2 ** (k - 1))
        for i in range(n_sites - k):
            h_step[i, i + k] = coupling_val
            h_step[i + k, i] = coupling_val
    # 3. Write Energy.bin (Site Energies only for Coupling mode) and Coupling.bin
    print(f"Writing Energy.bin (Site Energies) and Coupling.bin...")
    # Write only site energies (diagonal noise) to Energy.bin
    write_binary_site_energies(
        os.path.join(WORK_DIR, "Energy.bin"),
        n_sites,
        steps,
        realizations,
        dt,
        noise,
    )
    
    # Write static matrix to Coupling.bin
    # Note: write_binary_static_coupling handles full H but extracts tri.
    write_binary_static_coupling(
         os.path.join(WORK_DIR, "Coupling.bin"),
         h_step
    )

    # 4. Write Dipole.bin
    print("Writing Dipole.bin...")
    write_binary_dipole_file(
        os.path.join(WORK_DIR, "Dipole.bin"), n_sites, total_len, dt
    )

    # 5. Dummy Anharmonic and Overtone
    # Create empty files as placeholders
    with open(os.path.join(WORK_DIR, "Anharmonicity.bin"), "wb") as f:
        pass

    return total_len

def create_job_input(
    job_dir: str,
    n_sites: int,
    steps: int,
    dt: float,
    begin_realization: int,
    end_realization: int,
    total_len: int,
    sample_rate_is_steps: bool = True,
    propagation_mode: str = "Sparse",
    save_files: bool = True,
):
    """Creates the 'input' file for a specific NISE job."""

    sample_rate = steps if sample_rate_is_steps else 1
    save_files_flag = 1 if save_files else 0

    input_content = f"""Propagation {propagation_mode}
Couplingcut 1
Threshold 0.0
SaveFiles {save_files_flag}
HamiltonianType Coupling
Hamiltonianfile Energy.bin
Couplingfile Coupling.bin
Dipolefile Dipole.bin
Length {total_len}
Samplerate {sample_rate}
Lifetime {steps*dt}
Timestep {dt}
Format Dislin
Anharmonicity 0
MinFrequencies -2000
MaxFrequencies 2000
Technique Pop
FFT 2048
RunTimes {steps} 0 0
BeginPoint {begin_realization}
EndPoint {end_realization - 1}
Singles {n_sites}
Doubles 0
Skip Doubles
Sites {n_sites}
InitialState 1
"""

    with open(os.path.join(job_dir, "input"), "w") as f:
        f.write(input_content)

def run_nise_job(args: Tuple) -> None:
    """Worker function for running a NISE job."""
    os.environ["OMP_NUM_THREADS"] = "1"
    job_id, n_sites, steps, dt, begin, end, traj_steps, propagation_mode, save_files = args
    if n_sites >= 256:
        os.environ['OMP_NUM_THREADS'] = '2'
    if n_sites >= 512:
        os.environ['OMP_NUM_THREADS'] = '4'
    if n_sites >= 1024:
        os.environ['OMP_NUM_THREADS'] = '16'
    job_dir = os.path.join(WORK_DIR, f"job_{propagation_mode}_{job_id}")
    os.makedirs(job_dir, exist_ok=True)

    # Symlink binaries
    files_to_link = ["Energy.bin", "Dipole.bin", "Coupling.bin"]
    for f in files_to_link:
        src = os.path.join(WORK_DIR, f)
        dst = os.path.join(job_dir, f)
        if os.path.exists(dst):
            os.remove(dst)
        if os.path.exists(src):
            os.symlink(src, dst)

    # Remove stale output files to avoid reading old results if this run fails
    output_files = ["PopF.dat", "Pop.dat", "nise.out"]
    for f in output_files:
        p = os.path.join(job_dir, f)
        if os.path.exists(p):
            os.remove(p)

    # Create input
    create_job_input(
        job_dir,
        n_sites,
        steps,
        dt,
        begin,
        end,
        traj_steps,
        sample_rate_is_steps=True,
        propagation_mode=propagation_mode,
        save_files=save_files,
    )

    # Use mpirun to restrict OpenMP threads per process as requested by user
    cmd_nise = f"{NISE_BIN} input"

    with open(os.path.join(job_dir, "nise.out"), "w") as outfile:
        # stdin=subprocess.DEVNULL is good practice to avoid hangs on background proc
        subprocess.run(
            cmd_nise,
            shell=True,
            cwd=job_dir,
            stdout=outfile,
            stderr=subprocess.STDOUT,
            check=True,
            stdin=subprocess.DEVNULL,
        )

def benchmark_nise_2017(
    n_sites: int,
    steps: int = 1000,
    dt: float = 1.0,
    realizations: int = 1,
    n_jobs: int = 1,
    propagation_mode: str = "Sparse",
    save_files: bool = True,
    noise_args: Tuple = (),
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """Runs standard NISE 2017 benchmark.

    Args:
        n_sites: Number of sites.
        steps: Time steps.
        dt: Time step.
        realizations: Number of realizations.
        n_jobs: Number of parallel jobs.
        noise_args: Noise parameters (unused here, used in generation).

    Returns:
        (elapsed_time, time_axis, population_matrix)
    """
    # Input already generated by main run logic
    traj_steps = (
        steps * realizations
    )  # Known total length but we pass what's needed

    if n_jobs > realizations:
        n_jobs = realizations

    job_args = []
    chunk_size = realizations // n_jobs
    remainder = realizations % n_jobs

    current_begin = 0
    for i in range(n_jobs):
        count = chunk_size + (1 if i < remainder else 0)
        current_end = current_begin + count
        if count > 0:
            job_args.append(
                (i, n_sites, steps, dt, current_begin, current_end, traj_steps, propagation_mode, save_files)
            )
        current_begin = current_end

    start = time.time()

    if n_jobs > 1:
        # Use spawn context to be safe, though OMP fix might mitigate the need.
        # But stick to spawn as it's more robust for libraries like this.
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(n_jobs) as pool:
            pool.map(run_nise_job, job_args)
    else:
        # Serial fallback
        run_nise_job(job_args[0])

    # Aggregate results for comparison
    reals_per_job = []
    for i in range(n_jobs):
        count = chunk_size + (1 if i < remainder else 0)
        reals_per_job.append(count)

    # To aggregate, we need to know the mode to find the directories
    if save_files:
        time_axis, pops = aggregate_nise_jobs_mode(n_jobs, n_sites, reals_per_job, propagation_mode)
    else:
        time_axis, pops = None, None

    end = time.time()
    return end - start, time_axis, pops


class BenchmarkPopulationCalculation(PopulationCalculation):
    def __init__(self, config: NISE2017Config, device: str = "cpu"):
        super().__init__(config, device=device)
        self.device = device

    def get_common_params(self, mode="Population"):
        params = super().get_common_params(mode)
        params.device = self.device
        params.save_interval = 9999999999999999
        return params


def benchmark_torchnise(
    n_sites: int,
    steps: int = 1000,
    dt: float = 1.0,
    device: str = "cpu",
    realizations: int = 1,
    save_files: bool = True,
) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
    """Runs TorchNISE benchmark using the adapter.

    Args:
        n_sites: Number of sites.
        steps: Time steps.
        dt: Time step.
        device: 'cpu' or 'cuda'.
        realizations: Number of realizations.

    Returns:
        (elapsed_time, time_axis, population_matrix)
    """
    # Use adapter to run from generated files
    start = time.time()

    # Generate temporary config for adapter
    work_dir = WORK_DIR

    conf = NISE2017Config()
    conf.hamiltonian_file = os.path.join(work_dir, "Energy.bin")
    conf.hamiltonian_type = "Coupling"
    conf.coupling_file = os.path.join(work_dir, "Coupling.bin")
    conf.dipole_file = os.path.join(
        work_dir, "Dipole.bin"
    )  # Optional for pop but safer
    conf.length = steps * realizations
    conf.sample_rate = steps
    conf.t_max_1 = steps
    conf.timestep = dt
    conf.begin_point = 0
    conf.end_point = realizations
    conf.singles = n_sites
    conf.doubles = 0
    conf.initial_state_site = 0  # 0-indexed
    conf.technique = "Population"
    conf.temperature = 300
    conf.save_popf = save_files

    calc = BenchmarkPopulationCalculation(conf, device=device)

    try:
        population, _, _ = calc.run()
    except Exception as e:
        print(f"Error TorchNISE {device} N={n_sites}: {e}")
        return None, None, None

    end = time.time()

    t, p = read_popf("PopF.dat", n_sites, initial_site=0)
    return end - start, t, p


def run_comparison(
    sizes: List[int],
    realizations: List[int],
    steps: int = 1000,
    n_jobs: int = 1,
    save_nise_output: bool = True,
    noise_args: Tuple = (),
    n_neighbours: int = 4,
) -> Tuple[List[int], List[float], List[float], List[float], List[float], Dict[int, Any]]:
    """Runs comparison between NISE 2017 and TorchNISE.

    Args:
        sizes: List of system sizes.
        realizations: List of realizations per size.
        steps: Time steps.
        n_jobs: Parallel jobs for NISE 2017.
        noise_args: Noise parameters.

    Returns:
        Tuple containing lists of timings and population data.
    """
    t_nise_sparse = []
    t_nise_coupling = []
    t_torch_cpu = []
    t_torch_gpu = []

    # Store population data for the largest run
    pop_data_store = (
        {}
    )  # Key: N, Value: {'nise': (t, p), 'cpu': (t, p), 'gpu': (t, p)}

    print(f"{'N':<5} | {'NISE Sparse':<15} | {'NISE Coupling':<15} | {'Torch CPU':<10} | {'Torch GPU':<10}")
    print("-" * 75)

    # Unpack noise args
    omega_k, lambda_k, v_k, temp = noise_args

    for N, realizations_N in zip(sizes, realizations):
        # Generate inputs ONCE per N
        print(
            f"\nGeneraring shared inputs for N={N}, Reals={realizations_N}, Steps={steps}"
        )
        generate_nise_inputs(
            N, steps, 1.0, realizations_N, omega_k, lambda_k, v_k, temp, n_neighbours
        )

        pop_data_store[N] = {}

        # Run NISE 2017 -> Limit to N=100 as N=200 takes too long (>3 min)
        # Run NISE 2017 -> Limit to N=100 as N=200 takes too long (>3 min)
        if N <= 100:
            print("Running NISE_2017 Sparse calculations on cpu...")
            tn_s, t_axis_s, p_nise_s = benchmark_nise_2017(
                N,
                steps=steps,
                dt=1.0,
                realizations=realizations_N,
                n_jobs=n_jobs,
                propagation_mode="Sparse",
                save_files=save_nise_output,
            )
            pop_data_store[N]["nise_sparse"] = (t_axis_s, p_nise_s)
        else:
            tn_s = None
        if N <= 1024:    
            print("Running NISE_2017 Coupling calculations on cpu...")
            tn_c, t_axis_c, p_nise_c = benchmark_nise_2017(
                N,
                steps=steps,
                dt=1.0,
                realizations=realizations_N,
                n_jobs=n_jobs,
                propagation_mode="Coupling",
                save_files=save_nise_output,
            )
            pop_data_store[N]["nise_coupling"] = (t_axis_c, p_nise_c)
        else:
            tn_c = None

        # Run TorchNISE CPU
        if N <= 1024:
            tt_cpu, t_axis_c, p_cpu = benchmark_torchnise(
                N,
                steps=steps,
                dt=1.0,
                device="cpu",
                realizations=realizations_N,
                save_files=save_nise_output,
            )
            pop_data_store[N]["cpu"] = (t_axis_c, p_cpu)
        else:
            tt_cpu = None

        # Run TorchNISE GPU
        if torch.cuda.is_available():
            tt_gpu, t_axis_g, p_gpu = benchmark_torchnise(
                N,
                steps=steps,
                dt=1.0,
                device="cuda",
                realizations=realizations_N,
                save_files=save_nise_output,
            )
            pop_data_store[N]["gpu"] = (t_axis_g, p_gpu)
        else:
            print("No GPU available, skipping TorchNISE GPU")
            tt_gpu = None

        t_nise_sparse.append(tn_s)
        t_nise_coupling.append(tn_c)
        t_torch_cpu.append(tt_cpu)
        t_torch_gpu.append(tt_gpu)

        tns_str = f"{tn_s:.4f}" if tn_s is not None else "N/A"
        tnc_str = f"{tn_c:.4f}" if tn_c is not None else "N/A"
        tc_str = f"{tt_cpu:.4f}" if tt_cpu is not None else "N/A"
        tg_str = f"{tt_gpu:.4f}" if tt_gpu is not None else "N/A"

        print(f"{N:<5} | {tns_str:<15} | {tnc_str:<15} | {tc_str:<10} | {tg_str:<10}")

    return sizes, t_nise_sparse, t_nise_coupling, t_torch_cpu, t_torch_gpu, pop_data_store



def plot(
    sizes: List[int],
    t_nise_sparse: List[Optional[float]],
    t_nise_coupling: List[Optional[float]],
    t_torch_cpu: List[Optional[float]],
    t_torch_gpu: List[Optional[float]],
    realizations: List[int],
    steps: int = 1000,
    n_jobs: int = 1,
    n_neighbours: int = 4,
):
    """Plots the performance comparison."""
    realizations = realizations[:len(sizes)]
    plt.figure(figsize=(10, 6))

    # Filter Nones
    mask_nise_s = [x is not None for x in t_nise_sparse]
    if any(mask_nise_s):
        plt.plot(
            np.array(sizes)[mask_nise_s],
            np.array(t_nise_sparse)[mask_nise_s]
            / np.array(realizations)[mask_nise_s]
            * max(realizations),
            "o--",
            label="NISE (Sparse)",
        )

    mask_nise_c = [x is not None for x in t_nise_coupling]
    if any(mask_nise_c):
        plt.plot(
            np.array(sizes)[mask_nise_c],
            np.array(t_nise_coupling)[mask_nise_c]
            / np.array(realizations)[mask_nise_c]
            * max(realizations),
            "x--",
            label="NISE (Coupling)",
        )

    mask_cpu = [x is not None for x in t_torch_cpu]
    if any(mask_cpu):
        plt.plot(
            np.array(sizes)[mask_cpu],
            np.array(t_torch_cpu)[mask_cpu]
            / np.array(realizations)[mask_cpu]
            * max(realizations),
            "s-",
            label="TorchNISE (CPU)",
        )

    mask_gpu = [x is not None for x in t_torch_gpu]
    if any(mask_gpu):
        plt.plot(
            np.array(sizes)[mask_gpu],
            np.array(t_torch_gpu)[mask_gpu]
            / np.array(realizations)[mask_gpu]
            * max(realizations),
            "^-",
            label="TorchNISE (GPU)",
        )

    plt.xlabel("System Size N")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.title(f"Performance Comparison: Population Dynamics 100 steps ({n_neighbours} Neighbours)")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="-")
    filename = f"comparison_benchmark_{realizations}_{steps}_{max(sizes)}_j{n_jobs}_{n_neighbours}_neighbours.png"
    plt.savefig(filename)
    print(f"Saved {filename}")


def plot_population_dynamics(
    pop_data_store: Dict[int, Any], steps: int, n_jobs: int
):
    """Plots population dynamics for the largest system size."""
    # Select the largest N for which we have data
    valid_ns = [N for N in pop_data_store if any(pop_data_store[N].values())]
    if not valid_ns:
        return

    # Prefer an N that has both NISE and Torch if possible
    # Otherwise just largest
    target_n = valid_ns[-1]

    data = pop_data_store[target_n]

    # Determine which source to use for selecting "highest average population"
    # Prefer Torch CPU or GPU as baseline
    # Determine which source to use for selecting "highest average population"
    # Prefer Torch CPU or GPU as baseline
    ref_key = "cpu" if "cpu" in data else "gpu" if "gpu" in data else "nise_coupling" if "nise_coupling" in data else "nise_sparse"
    if ref_key not in data:
        return

    t_ref, p_ref = data[ref_key]

    # Calculate average population for each site
    if p_ref is None or p_ref.size == 0 or len(p_ref.shape) < 2:
         print(f"Skipping population plot for N={target_n}: Invalid data shape {p_ref.shape if p_ref is not None else 'None'}")
         return

    avg_pops = np.mean(p_ref, axis=0)  # (n_sites,)

    # Get top 5 sites
    top_indices = np.argsort(avg_pops)[::-1][:5]

    plt.figure(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, site_idx in enumerate(top_indices):
        color = colors[i]

        if "nise_sparse" in data:
            t, p = data["nise_sparse"]
            if p is not None:
                if site_idx < p.shape[1]:
                    plt.plot(
                        t,
                        p[:, site_idx],
                        "o--",
                        label=f"NISE Sparse Site {site_idx}" if i == 0 else None,
                        color=color,
                        markersize=4,
                        alpha=0.7,
                    )
        if "nise_coupling" in data:
            t, p = data["nise_coupling"]
            if p is not None:
                if site_idx < p.shape[1]:
                    plt.plot(
                        t,
                        p[:, site_idx],
                        "x--",
                        label=f"NISE Coupling Site {site_idx}" if i == 0 else None,
                        color=color,
                        markersize=4,
                        alpha=0.7,
                    )

        # CPU
        if "cpu" in data:
            t, p = data["cpu"]
            plt.plot(
                t,
                p[:, site_idx],
                "-",
                label=f"Torch CPU Site {site_idx}" if i == 0 else None,
                color=color,
                linewidth=2,
                alpha=0.8,
            )

        # GPU
        if "gpu" in data:
            t, p = data["gpu"]
            plt.plot(
                t,
                p[:, site_idx],
                ":",
                label=f"Torch GPU Site {site_idx}" if i == 0 else None,
                color=color,
                linewidth=2,
            )

    plt.xlabel("Time (fs)")
    plt.ylabel("Population")
    plt.title(f"Population Dynamics (Top 5 Sites) N={target_n}")

    # Custom legend
    from matplotlib.lines import Line2D

    custom_lines = [
        Line2D([0], [0], color="gray", linestyle="--", marker="o"),
        Line2D([0], [0], color="gray", linestyle="-"),
        Line2D([0], [0], color="gray", linestyle=":"),
    ]
    plt.legend(custom_lines, ["NISE 2017", "Torch CPU", "Torch GPU"])

    plt.grid(True)
    plt.savefig(f"population_dynamics_comparison_N{target_n}.png")
    print(f"Saved population_dynamics_comparison_N{target_n}.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark TorchNISE vs NISE_2017")
    parser.add_argument("--realizations", type=int, nargs="+", default=[10000, 10000, 10000,1000,500,100,50,8,4,1,1,1,1], help="Number of realizations per size (default: ...)")
    parser.add_argument("--steps", type=int, default=100, help="Number of time steps (default: 1000)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2,4,8,16,32,64,128,256,512,1024,2048,4096,8192], help="List of system sizes N to benchmark")
    parser.add_argument("--jobs", type=int, default=16, help="Number of parallel jobs for NISE 2017")
    parser.add_argument("--save-nise-output", action="store_true", help="Save NISE 2017 output files (for faster benchmarking)")
    parser.add_argument("--neighbours", type=int, default=1, help="Number of nearest neighbours for coupling (default: 4)")
    parser.add_argument("--omega_k", type=float, nargs="+", default=[0, 725, 1200], help="Omega_k for Drude-Lorentz (cm-1)")
    parser.add_argument("--lambda_k", type=float, nargs="+", default=[20, 20, 20], help="Lambda_k (cm-1)")
    parser.add_argument("--v_k", type=float, nargs="+", default=[1/100, 1/100, 1/100], help="v_k inverse timescale (1/fs?) - check units")
    parser.add_argument("--temp", type=float, default=300, help="Temperature (K)")

    args = parser.parse_args()
    
    units.set_units(e_unit="cm-1", t_unit="fs")
    
    print(f"Running benchmark with R={args.realizations}, Steps={args.steps}, Jobs={args.jobs}")
    print(f"System sizes: {args.sizes}")
    
    noise_args = (args.omega_k, args.lambda_k, args.v_k, args.temp)
    
    with torch.no_grad():
        s, ns, nc, tc, tg, pop_data = run_comparison(args.sizes, realizations=args.realizations, steps=args.steps, n_jobs=args.jobs, save_nise_output=args.save_nise_output, noise_args=noise_args, n_neighbours=args.neighbours)
    plot(s, ns, nc, tc, tg, realizations=args.realizations, steps=args.steps, n_jobs=args.jobs, n_neighbours=args.neighbours)
    plot_population_dynamics(pop_data, args.steps, args.jobs)
