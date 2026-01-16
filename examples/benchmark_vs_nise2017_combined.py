"""
Benchmark script comparing TorchNISE (CPU/GPU) vs NISE_2017 scaling performance.

NOTE: This script benchmarks the underlying propagation algorithms: Matrix vs Vector.
- "Matrix" propagation is triggered by the "Population" technique in NISE 2017.
- "Vector" propagation is triggered by the "Absorption" (Linear Response) technique in NISE 2017.
We use these mappings to force NISE 2017 into the desired propagation mode for performance comparison,
rather than for the specific observables themselves.
"""

import functools
import multiprocessing
import os
import shutil
import struct
import subprocess
import time
import argparse
from typing import List, Tuple, Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch

import torchnise.nise
import torchnise.units as units
from torchnise.fft_noise_gen import gen_noise
from torchnise.nise2017_adapter import (
    NISE2017Config,
    PopulationCalculation,
    AbsorptionCalculation,
)
from torchnise.spectral_functions import spectral_drude_lorentz_heom


EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
NISE_BIN = os.path.join(EXAMPLES_DIR, "NISE_2017/bin/NISE")
WORK_DIR = os.path.abspath("benchmark_workdir")


# --- Helper Classes for Benchmarking ---

class BenchmarkPopulationCalculation(PopulationCalculation):
    def __init__(self, config: NISE2017Config, device: str = "cpu"):
        super().__init__(config, device=device)
        self.device = device

    def get_common_params(self, mode="Population"):
        params = super().get_common_params(mode)
        params.device = self.device
        # Disable frequent saving to isolate computation time
        params.save_interval = 999999999
        return params

class BenchmarkAbsorptionCalculation(AbsorptionCalculation):
    def __init__(self, config: NISE2017Config, device: str = "cpu"):
        super().__init__(config, device=device)
        self.device = device

    def get_common_params(self, mode="Absorption"):
        params = super().get_common_params(mode)
        params.device = self.device
        # Disable frequent saving/logging
        params.save_interval = 999999999
        return params


# --- Output Reading Helpers ---

def read_popf(
    filepath: str, n_sites: int, initial_site: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Reads PopF.dat and extracts population dynamics P(initial -> final)."""
    if not os.path.exists(filepath):
        return None, None
    try:
        data = np.loadtxt(filepath)
    except Exception:
        return None, None
        
    if data.size == 0:
         return None, None

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
    """Aggregates results from multiple NISE jobs reading PopF.dat."""
    total_reals = sum(reals_per_job)
    weighted_pops = None
    common_time = None

    for i in range(n_jobs):
        job_dir = os.path.join(WORK_DIR, f"job_{propagation_mode}_{i}")
        t, p = read_popf(os.path.join(job_dir, "PopF.dat"), n_sites, initial_site=0)

        if t is None or p is None:
            # Maybe Absorption mode?
            # Absorption calculation output is Absorption.dat (Freq) and TD_Absorption.dat (Time)
            # For benchmarking we might not parse the full complex output, just timing.
            # But let's check TD_Absorption.dat if PopF failed.
            p_abs = read_td_absorption(os.path.join(job_dir, "TD_Absorption.dat"))
            if p_abs is not None:
                 # Absorption mode returns (Time, Re, Im). 
                 # We can use that for verification if needed.
                 # For now, if PopF fails, we might just return None or handle differently based on mode?
                 # But valid for Population mode logic.
                 pass
            continue

        if common_time is None:
            common_time = t
            weighted_pops = np.zeros_like(p)

        if p.shape != weighted_pops.shape:
             min_len = min(p.shape[0], weighted_pops.shape[0])
             p = p[:min_len]
             if common_time is not None:
                  common_time = common_time[:min_len]
                  weighted_pops = weighted_pops[:min_len]

        weight = reals_per_job[i] / total_reals
        weighted_pops += p * weight

    return common_time, weighted_pops

def read_td_absorption(filepath: str):
    if not os.path.exists(filepath):
        return None
    try:
        return np.loadtxt(filepath)
    except:
        return None

# --- Binary Input Generation ---

def write_binary_site_energies(filepath, n_sites, steps, realizations, dt, noise=None):
    """Writes Time-Dependent Site Energies to NISE 2017 Binary Format."""
    total_records = realizations * steps
    data = np.zeros((total_records, 1 + n_sites), dtype=np.float32)
    data[:, 0] = np.arange(total_records, dtype=np.float32)
    
    if noise is not None:
        noise_np = (
            noise.permute(1, 0, 2)
            .detach().cpu().numpy()
            .reshape(-1, n_sites)
            .astype(np.float32)
        )
        data[:, 1:] = noise_np
        
    data.tofile(filepath)

def write_binary_static_coupling(filepath, h_static):
    """Writes Static Coupling Matrix to NISE 2017 Binary Format."""
    n_sites = h_static.shape[0]
    iu = np.triu_indices(n_sites)
    h_tri = h_static[iu].astype(np.float32)
    h_tri.tofile(filepath)

def write_binary_dipole_file(filepath, n_sites, total_steps, dt):
    """Writes constant dipole 1.0 for x-axis to NISE 2017 Format."""
    data = np.zeros((total_steps, 1 + 3 * n_sites), dtype=np.float32)
    data[:, 0] = np.arange(total_steps, dtype=np.float32)
    data[:, 1::3] = 1.0 # x=1.0 for all sites
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
    n_neighbours: int,
) -> int:
    os.makedirs(WORK_DIR, exist_ok=True)
    total_len = steps * realizations

    print(f"Generating noise for N={n_sites}, Steps={steps}, Reals={realizations}...")
    spectral_func = functools.partial(
        spectral_drude_lorentz_heom,
        omega_k=torch.tensor(omega_k) / units.HBAR,
        lambda_k=torch.tensor(lambda_k),
        vk=torch.tensor(v_k),
        temperature=temperature,
    )
    spectral_funcs = [spectral_func] * n_sites
    noise = gen_noise(
        spectral_funcs, dt, shape=(steps, realizations, n_sites), use_h5=False, device="cpu"
    )

    h_step = np.zeros((n_sites, n_sites))
    base_coupling = 100.0
    for k in range(1, n_neighbours + 1):
        coupling_val = base_coupling / (2 ** (k - 1))
        for i in range(n_sites - k):
            h_step[i, i + k] = coupling_val
            h_step[i + k, i] = coupling_val

    print(f"Writing Energy.bin and Coupling.bin...")
    write_binary_site_energies(
        os.path.join(WORK_DIR, "Energy.bin"), n_sites, steps, realizations, dt, noise
    )
    write_binary_static_coupling(
        os.path.join(WORK_DIR, "Coupling.bin"), h_step
    )

    print("Writing Dipole.bin...")
    write_binary_dipole_file(
        os.path.join(WORK_DIR, "Dipole.bin"), n_sites, total_len, dt
    )

    with open(os.path.join(WORK_DIR, "Anharmonicity.bin"), "wb") as f:
        pass

    return total_len


# --- Job Execution Helper ---

def create_job_input(
    job_dir: str,
    n_sites: int,
    steps: int,
    dt: float,
    begin_realization: int,
    end_realization: int,
    total_len: int,
    propagation_mode: str = "Sparse",
    technique: str = "Population", # "Population" or "Absorption"
    save_files: bool = True,
):
    save_files_flag = 1 if save_files else 0
    # For population typically sample_rate=steps (snapshot at end)
    # For absorption, usually sample_rate=1 (full trajectory)
    # But benchmarks often use steps as samplerate for speed if just checking end population?
    # Absorption needs full trajectory usually.
    # NISE code: If Technique==Absorption, it typically propagates full time.
    
    # We will stick to Samplerate = Steps for Population to match Matrix benchmark behavior (only final state).
    # For Absorption, we likely need Samplerate = 1 if we want meaningful physics, 
    # but for pure Computational Benchmarking, maybe samplerate doesn't affect speed much? 
    # Actually, high samplerate (sparse sampling) implies less I/O but propagation is same.
    # Let's set Samplerate=1 for Absorption to be realistic, or Steps if we just want raw speed?
    # Vector benchmark had Samplerate=1 in `write_nise_inputs_base`? No wait.
    # Vector benchmark had `Samplerate {steps}` line 156. So it was skipping frames?
    # If NISE skips frames, it does less work? No, propagation is step by step. Output is skipped.
    
    sample_rate = steps # Default high sample rate to minimize I/O overhead for pure compute benchmark

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
Technique {technique}
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
    os.environ["OMP_NUM_THREADS"] = "1"
    job_id, n_sites, steps, dt, begin, end, traj_steps, propagation_mode, technique, save_files = args
    if n_sites >= 256: os.environ['OMP_NUM_THREADS'] = '2'
    if n_sites >= 512: os.environ['OMP_NUM_THREADS'] = '4'
    if n_sites >= 1024: os.environ['OMP_NUM_THREADS'] = '16'

    job_dir = os.path.join(WORK_DIR, f"job_{propagation_mode}_{job_id}")
    os.makedirs(job_dir, exist_ok=True)

    files_to_link = ["Energy.bin", "Dipole.bin", "Coupling.bin"]
    for f in files_to_link:
        src = os.path.join(WORK_DIR, f)
        dst = os.path.join(job_dir, f)
        if os.path.exists(dst): os.remove(dst)
        if os.path.exists(src): os.symlink(src, dst)

    # Clean old outputs
    for f in ["PopF.dat", "Pop.dat", "TD_Absorption.dat", "Absorption.dat", "nise.out"]:
        p = os.path.join(job_dir, f)
        if os.path.exists(p): os.remove(p)

    create_job_input(
        job_dir, n_sites, steps, dt, begin, end, traj_steps, 
        propagation_mode=propagation_mode, technique=technique, save_files=save_files
    )

    cmd_nise = f"{NISE_BIN} input"
    with open(os.path.join(job_dir, "nise.out"), "w") as outfile:
        subprocess.run(
            cmd_nise, shell=True, cwd=job_dir, stdout=outfile, 
            stderr=subprocess.STDOUT, check=True, stdin=subprocess.DEVNULL
        )

def benchmark_nise_2017(
    n_sites: int,
    steps: int,
    dt: float,
    realizations: int,
    n_jobs: int,
    propagation_mode: str,
    technique: str,
    save_files: bool,
) -> float:
    traj_steps = steps * realizations
    if n_jobs > realizations: n_jobs = realizations

    job_args = []
    chunk_size = realizations // n_jobs
    remainder = realizations % n_jobs
    current_begin = 0
    for i in range(n_jobs):
        count = chunk_size + (1 if i < remainder else 0)
        current_end = current_begin + count
        if count > 0:
            job_args.append(
                (i, n_sites, steps, dt, current_begin, current_end, traj_steps, propagation_mode, technique, save_files)
            )
        current_begin = current_end

    start = time.time()
    if n_jobs > 1:
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(n_jobs) as pool:
            pool.map(run_nise_job, job_args)
    else:
        run_nise_job(job_args[0])
    end = time.time()
    return end - start


# --- TorchNISE Benchmark ---

def benchmark_torchnise(
    n_sites: int,
    steps: int,
    dt: float,
    device: str,
    realizations: int,
    propagation_type: str, # "matrix" or "vector"
    save_files: bool,
) -> Optional[float]:
    start = time.time()
    
    work_dir = WORK_DIR
    conf = NISE2017Config()
    conf.hamiltonian_file = os.path.join(work_dir, "Energy.bin")
    conf.hamiltonian_type = "Coupling"
    conf.coupling_file = os.path.join(work_dir, "Coupling.bin")
    conf.dipole_file = os.path.join(work_dir, "Dipole.bin")
    conf.length = steps * realizations
    conf.sample_rate = steps # To match NISE benchmark inputs generally
    conf.t_max_1 = steps
    conf.timestep = dt
    conf.begin_point = 0
    conf.end_point = realizations
    conf.singles = n_sites
    conf.doubles = 0
    conf.initial_state_site = 0
    # conf.technique set by class choice below implicitly
    conf.temperature = 300
    conf.save_popf = save_files

    try:
        if propagation_type == "matrix":
            calc = BenchmarkPopulationCalculation(conf, device=device)
            calc.run()
        elif propagation_type == "vector":
            calc = BenchmarkAbsorptionCalculation(conf, device=device)
            calc.run()
        else:
            raise ValueError(f"Unknown propagation_type: {propagation_type}")
            
    except Exception as e:
        print(f"Error TorchNISE {device} N={n_sites}: {e}")
        return None

    end = time.time()
    return end - start


# --- Main Loop & Plotting ---

def run_comparison(
    sizes: List[int],
    realizations: List[int],
    steps: int,
    n_jobs: int,
    save_nise_output: bool,
    noise_args: Tuple,
    n_neighbours: int,
    propagation_type: str,
) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    
    t_nise_sparse = []
    t_nise_coupling = []
    t_torch_cpu = []
    t_torch_gpu = []
    
    # Map propagation_type to NISE Technique
    nise_technique = "Population" if propagation_type == "matrix" else "Absorption"

    print(f"\nMode: {propagation_type.upper()} Propagation (NISE Technique: {nise_technique})")
    print(f"{'N':<5} | {'NISE Sparse':<15} | {'NISE Coupling':<15} | {'Torch CPU':<10} | {'Torch GPU':<10}")
    print("-" * 75)

    omega_k, lambda_k, v_k, temp = noise_args

    for N, realizations_N in zip(sizes, realizations):
        generate_nise_inputs(
            N, steps, 1.0, realizations_N, omega_k, lambda_k, v_k, temp, n_neighbours
        )
        
        # Upper limits for NISE to keep benchmark reasonable
        # Sparse often scales poorly with N
        if N <= 100: 
            tn_s = benchmark_nise_2017(
                N, steps, 1.0, realizations_N, n_jobs, "Sparse", nise_technique, save_nise_output
            )
        else:
            tn_s = None

        # Coupling often scales better
        if N <= 1024:
            tn_c = benchmark_nise_2017(
                N, steps, 1.0, realizations_N, n_jobs, "Coupling", nise_technique, save_nise_output
            )
        else:
            tn_c = None
            
        # TorchNISE CPU
        tt_cpu = benchmark_torchnise(
            N, steps, 1.0, "cpu", realizations_N, propagation_type, save_nise_output
        )
        
        # TorchNISE GPU
        if torch.cuda.is_available():
            tt_gpu = benchmark_torchnise(
                N, steps, 1.0, "cuda", realizations_N, propagation_type, save_nise_output
            )
        else:
            tt_gpu = None

        t_nise_sparse.append(tn_s)
        t_nise_coupling.append(tn_c)
        t_torch_cpu.append(tt_cpu)
        t_torch_gpu.append(tt_gpu)

        def fmt(x): return f"{x:.4f}" if x is not None else "N/A"
        print(f"{N:<5} | {fmt(tn_s):<15} | {fmt(tn_c):<15} | {fmt(tt_cpu):<10} | {fmt(tt_gpu):<10}")

    return sizes, t_nise_sparse, t_nise_coupling, t_torch_cpu, t_torch_gpu


def plot(
    sizes, t_nise_sparse, t_nise_coupling, t_torch_cpu, t_torch_gpu, 
    realizations, steps, n_jobs, n_neighbours, propagation_type
):
    plt.figure(figsize=(10, 6))
    realizations = realizations[:len(sizes)]
    max_reals = max(realizations)
    
    def plot_line(data, label, marker):
        mask = [x is not None for x in data]
        if any(mask):
            x = np.array(sizes)[mask]
            y = np.array(data)[mask]
            # Normalize to per max_reals equivalent
            # y_norm = time / reals * max_reals
            reals = np.array(realizations)[mask]
            y_norm = y / reals * max_reals
            plt.plot(x, y_norm, marker, label=label)

    plot_line(t_nise_sparse, "NISE (Sparse)", "o--")
    plot_line(t_nise_coupling, "NISE (Coupling)", "x--")
    plot_line(t_torch_cpu, "TorchNISE (CPU)", "s-")
    plot_line(t_torch_gpu, "TorchNISE (GPU)", "^-")

    plt.xlabel('System Size N')
    plt.ylabel(f'Time (s) normalized to {max_reals} realizations')
    plt.legend()
    plt.title(f'{propagation_type.capitalize()} Propagation Benchmark\nSteps={steps}, Neighbours={n_neighbours}')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-")
    
    filename = f'comparison_benchmark_{propagation_type}_{steps}_{n_jobs}j_{n_neighbours}neighbors.png'
    plt.savefig(filename)
    print(f"Saved plot to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TorchNISE vs NISE_2017")
    parser.add_argument("--propagation-type", choices=["matrix", "vector"], default="matrix", help="Propagation mode")
    parser.add_argument("--realizations", type=int, nargs="+", default=[10000, 10000, 10000, 1000])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2,4,8,16])
    parser.add_argument("--jobs", type=int, default=16, help="NISE 2017 Parallel Jobs")
    parser.add_argument("--no-save-nise-output", action="store_true")
    parser.add_argument("--neighbours", type=int, default=4)

    args = parser.parse_args()
    units.set_units(e_unit="cm-1", t_unit="fs")
    
    # Noise params (Dummy/Basic)
    noise_args = ([0.0], [0.0], [1.0], 300.0)

    with torch.no_grad():
        res = run_comparison(
            args.sizes, args.realizations, args.steps, args.jobs, 
            not args.no_save_nise_output, noise_args, args.neighbours, 
            args.propagation_type
        )
    
    plot(*res, args.realizations, args.steps, args.jobs, args.neighbours, args.propagation_type)
