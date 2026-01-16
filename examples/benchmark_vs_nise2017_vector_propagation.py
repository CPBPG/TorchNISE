
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
from torchnise.spectral_functions import spectral_drude_lorentz_heom


EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
NISE_BIN = os.path.join(EXAMPLES_DIR, "NISE_2017/bin/NISE")
WORK_DIR = os.path.abspath("benchmark_workdir")

def write_nise_inputs_base(n_sites: int, steps: int, dt: float, realizations: int, n_neighbours: int = 4):
    # Prepare directory
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.makedirs(WORK_DIR)
    
    # Generate Noise using TorchNISE
    # n_sites, steps, realizations
    # Same parameters as in benchmark_torchnise
    
    # Standard Drude-Lorentz
    # These match the benchmark_torchnise logic (0 noise for now in benchmark_torchnise? No, see below)
    # The benchmark_torchnise below sets vk=1, omega=0, lambda=0, T=300.
    # Effectively NO noise or constant?
    
    # Actually, let's use the parameters from the NISE 2017 input generation logic if we want to be fair.
    # But here we just want to create the basic files to make NISE run.
    
    # Create Energy.bin
    # Format: [Time] [H_tri (N*(N+1)/2)]
    
    # We will generate a constant Hamiltonian
    n_tri = n_sites * (n_sites + 1) // 2
    total_len = steps * realizations
    
    # Create a large binary file
    # We can probably chunk this if memory is an issue, but for benchmark N=1000 it might be large.
    # N=1000 -> n_tri ~ 500,000.
    # 500,000 floats * 4 bytes = 2MB per step.
    # 1000 steps -> 2GB. 
    # This is doable.
    
    # BUT wait, the NISE part of the benchmark is often limited to smaller N (<= 100).
    # N=100 -> n_tri ~ 5000 -> 20KB/step -> 20MB total.
    
    # Limit removed for larger benchmarks
    # if n_sites > 200:
    #    # Avoid generating huge files for large N if we are skipping NISE anyway
    #    return total_len
        
    print(f"Generating NISE input for N={n_sites}...")
    
    # H static
    H = np.zeros((n_sites, n_sites))
    base_coupling = 100.0
    for k in range(1, n_neighbours + 1):
        coupling_val = base_coupling / (2 ** (k - 1))
        for i in range(n_sites - k):
            H[i, i + k] = coupling_val
            H[i + k, i] = coupling_val
        
    iu = np.triu_indices(n_sites)
    h_tri = H[iu]
    
    # Write Energy.bin
    energy_file = os.path.join(WORK_DIR, "Energy.bin")
    
    # We can write row by row to save memory
    # Structure: Time(float), h_tri(floats...)
    
    # For speed, let's just create a dummy file of correct size if we just want to benchmark compute time?
    # NO, NISE reads the file.
    
    with open(energy_file, "wb") as f:
        # Create a single record bytes
        # time + h_tri
        record_fmt = f"f{n_tri}f"
        record_len = 4 * (1 + n_tri)
        
        # We can implement a loop or vectorized
        # Vectorized is better
        # Data: (total_len, 1 + n_tri)
        # However, for N=100 and steps=1000*reals, it fits in memory
        
        # Time axis
        t_axis = np.arange(total_len, dtype=np.float32)
        
        # We need to interleave time and H
        # Create a structured array or just a unified float32 array
        
        # data = np.zeros((total_len, 1 + n_tri), dtype=np.float32)
        # data[:, 0] = t_axis
        # data[:, 1:] = h_tri # Broadcast
        
        # This is memory intensive.
        # Write in chunks.
        chunk_size = 1000
        for i in range(0, total_len, chunk_size):
            current_chunk_size = min(chunk_size, total_len - i)
            data_chunk = np.zeros((current_chunk_size, 1 + n_tri), dtype=np.float32)
            data_chunk[:, 0] = t_axis[i:i+current_chunk_size]
            data_chunk[:, 1:] = h_tri
            data_chunk.tofile(f)
            
    # Dipole.bin
    # [Time] [Mu (3*N)]
    dipole_file = os.path.join(WORK_DIR, "Dipole.bin")
    with open(dipole_file, "wb") as f:
        # All dipoles 1.0 in x
        mu_vec = np.zeros(3 * n_sites, dtype=np.float32)
        mu_vec[0::3] = 1.0
        
        for i in range(0, total_len, chunk_size):
            current_chunk_size = min(chunk_size, total_len - i)
            data_chunk = np.zeros((current_chunk_size, 1 + 3*n_sites), dtype=np.float32)
            data_chunk[:, 0] = t_axis[i:i+current_chunk_size]
            data_chunk[:, 1:] = mu_vec
            data_chunk.tofile(f)
            
    # Dummy Anharmonicity and OvertoneDipole (some NISE versions need them)
    with open(os.path.join(WORK_DIR, "Anharmonicity.bin"), "wb") as f:
        pass
    with open(os.path.join(WORK_DIR, "OvertoneDipole.bin"), "wb") as f:
        pass
        
    return total_len

def create_job_input(job_dir, n_sites, steps, dt, begin_realization, end_realization, total_len, propagation_mode="Sparse", save_files=True):
    save_files_flag = 1 if save_files else 0
    input_content = f"""Propagation {propagation_mode}
Couplingcut 0
Threshold 0.0
SaveFiles {save_files_flag}
Hamiltonianfile Energy.bin
Dipolefile Dipole.bin
Length {total_len}
Samplerate {steps}
Lifetime {steps*dt}
Timestep {dt}
Format Dislin
Anharmonicity 0
MinFrequencies -2000
MaxFrequencies 2000
Technique Absorption
FFT 2048
RunTimes {steps} 0 0
BeginPoint {begin_realization}
EndPoint {end_realization-1}
Singles {n_sites}
Doubles 0
Skip Doubles
Sites {n_sites}
InitialState 1
"""
    with open(os.path.join(job_dir, "input"), "w") as f:
        f.write(input_content)

def run_nise_job(args):
    os.environ['OMP_NUM_THREADS'] = '1'
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
    files_to_link = ["Energy.bin", "Dipole.bin", "Anharmonicity.bin", "OvertoneDipole.bin"]
    for f in files_to_link:
        src = os.path.join(WORK_DIR, f)
        dst = os.path.join(job_dir, f)
        if os.path.exists(dst):
            os.remove(dst)
        if os.path.exists(src):
             os.symlink(src, dst)
             
    # Create input
    create_job_input(job_dir, n_sites, steps, dt, begin, end, traj_steps, propagation_mode, save_files)
    
    # Use mpirun to restrict OpenMP threads per process as requested by user
    cmd_nise = f"{NISE_BIN} input"
    
    with open(os.path.join(job_dir, "nise.out"), "w") as outfile:
        # stdin=subprocess.DEVNULL is good practice to avoid hangs on background proc
        subprocess.run(cmd_nise, shell=True, cwd=job_dir, stdout=outfile, stderr=subprocess.STDOUT, check=True, stdin=subprocess.DEVNULL)

def benchmark_nise_2017(n_sites, steps=1000, dt=1.0, realizations=1, n_jobs=1, propagation_mode="Sparse", save_files=True, n_neighbours=4):
    traj_steps = write_nise_inputs_base(n_sites, steps, dt, realizations, n_neighbours)

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
            job_args.append((i, n_sites, steps, dt, current_begin, current_end, traj_steps, propagation_mode, save_files))
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
        
    end = time.time()
    
    return end - start

def benchmark_torchnise(n_sites, steps=1000, dt=1.0, device="cpu", realizations=1, n_neighbours=4):
    # Setup same system: 0-diag, 100-coupling
    start = time.time()
    
    # Create Constant Hamiltonian
    # H shape: (n_sites, n_sites) if constant_v used, or (steps, n_sites, n_sites)
    
    H = torch.zeros((n_sites, n_sites)).to(device)
    base_coupling = 100.0
    for k in range(1, n_neighbours + 1):
        if n_sites - k <= 0:
            continue
        coupling_val = base_coupling / (2 ** (k - 1))
        # Vectorized coupling setting
        indices = torch.arange(n_sites - k).to(device)
        H[indices, indices + k] = coupling_val
        H[indices + k, indices] = coupling_val
    
    # If we pass H to run_nise, it usually expects (realizations, n_sites, n_sites) or (steps, ...)?
    # nise_propagate docstring: (steps, realizations, n, n)
    # But run_nise handles single H as "constant H + noise"?
    # If we pass H with shape (n,n), run_nise treats it as constant + noise.
    # Here noise is 0.
    
    spectral_funcs = [
         functools.partial(spectral_drude_lorentz_heom, omega_k=torch.tensor([0.]), lambda_k=torch.tensor([0.]), vk=torch.tensor([1.]), temperature=300)
    ] * n_sites
    
    initialState = torch.zeros(n_sites)
    initialState[0] = 1.0 # Site 1 (index 0)
    
    from torchnise.nise import NISEParameters

    # Run
    try:
        params = NISEParameters(
            dt=dt,
            total_time=steps * dt,
            temperature=300,
            t_correction="None",
            mode="Population",
            device=device,
            save_interval=steps,
        )
        torchnise.nise.run_nise(
            H.cpu(), # run_nise handles device internally
            realizations, 
            initialState, 
            spectral_funcs,
            params,
        )
    except Exception as e:
        print(f"Error TorchNISE {device} N={n_sites}: {e}")
        return None
        
    end = time.time()
    return end - start

def run_comparison(sizes, realizations=1, steps=1000, n_jobs=1, save_nise_output=True, n_neighbours=4):
    t_nise_sparse = []
    t_nise_coupling = []
    t_torch_cpu = []
    t_torch_gpu = []
    
    print(f"{'N':<5} | {'NISE Sparse':<15} | {'NISE Coupling':<15} | {'Torch CPU':<10} | {'Torch GPU':<10}")
    print("-" * 75)
    
    for N,realizations_N in zip(sizes,realizations):
        # Benchmark steps = 1000
        # Benchmark steps from args
        # steps = steps

        # Run NISE 2017 -> Limit to N=100 as N=200 takes too long (>3 min)
        # Run NISE 2017 -> Limit to N=100 as N=200 takes too long (>3 min)
        if N<=256: # Removed N limit
             tn_s = benchmark_nise_2017(N, steps=steps, dt=1.0, realizations=realizations_N, n_jobs=n_jobs, propagation_mode="Sparse", save_files=save_nise_output)
            
        else:
             tn_s = None
        if N<=8192: # Removed N limit
             tn_c =  benchmark_nise_2017(N, steps=steps, dt=1.0, realizations=realizations_N, n_jobs=n_jobs, propagation_mode="Coupling", save_files=save_nise_output)
        else:
             tn_c = None
        
        # Run TorchNISE CPU
        if N <= 2000:
            tt_cpu = benchmark_torchnise(N, steps=steps, dt=1.0, device="cpu", realizations=realizations_N)
        else:
            tt_cpu = None
        
        # Run TorchNISE GPU
        if torch.cuda.is_available():
            tt_gpu = benchmark_torchnise(N, steps=steps, dt=1.0, device="cuda", realizations=realizations_N)
        else:
            print("No GPU available, skipping TorchNISE GPU")
            print("Tip: On WSL, ensure you have exported the library path:")
            print("export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH")
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
        
    return sizes, t_nise_sparse, t_nise_coupling, t_torch_cpu, t_torch_gpu

def plot(sizes, t_nise_sparse, t_nise_coupling, t_torch_cpu, t_torch_gpu, realizations=1, steps=1000, n_jobs=1, n_neighbours=4):
    plt.figure(figsize=(10,6))
    
    # Filter Nones
    mask_nise_s = [x is not None for x in t_nise_sparse]
    if any(mask_nise_s):
        plt.plot(np.array(sizes)[mask_nise_s], np.array(t_nise_sparse)[mask_nise_s]
            / np.array(realizations)[mask_nise_s]
            * max(realizations), 'o--', label='NISE (Sparse)')

    mask_nise_c = [x is not None for x in t_nise_coupling]
    if any(mask_nise_c):
        plt.plot(np.array(sizes)[mask_nise_c], np.array(t_nise_coupling)[mask_nise_c]
            / np.array(realizations)[mask_nise_c]
            * max(realizations), 'x--', label='NISE (Coupling)')
        
    mask_cpu = [x is not None for x in t_torch_cpu]
    if any(mask_cpu):
        plt.plot(np.array(sizes)[mask_cpu],np.array(t_torch_cpu)[mask_cpu]
            / np.array(realizations)[mask_cpu]
            * max(realizations), 's-', label='TorchNISE (CPU)')

    mask_gpu = [x is not None for x in t_torch_gpu]
    if any(mask_gpu):
         plt.plot(np.array(sizes)[mask_gpu], np.array(t_torch_gpu)[mask_gpu]
            / np.array(realizations)[mask_gpu]
            * max(realizations), '^-', label='TorchNISE (GPU)')

    plt.xlabel('System Size N')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.title(f'Performance Comparison: Population Dynamics 100 steps ({n_neighbours} Neighbours)')
    plt.xscale('log') 
    plt.yscale('log') 
    plt.grid(True, which="both", ls="-")
    filename = f'comparison_benchmark_{realizations}_{steps}_{max(sizes)}_j{n_jobs}_{n_neighbours}_neighbours_vector_propagation.png'
    plt.savefig(filename)
    print(f"Saved {filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark TorchNISE vs NISE_2017")
    parser.add_argument("--realizations", type=int, nargs="+", default=[10000, 10000, 10000,1000,500,500,100,50,8,4,1,1,1,1], help="Number of realizations for TorchNISE (default: 1)")
    parser.add_argument("--steps", type=int, default=100, help="Number of time steps (default: 1000)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2,4,8,16,32,33,64,128,256,512,1024,2048,4096,8192], help="List of system sizes N to benchmark")
    parser.add_argument("--jobs", type=int, default=16, help="Number of parallel jobs for NISE 2017")
    parser.add_argument("--no-save-nise-output", action="store_true", help="Don't save NISE 2017 output files")
    parser.add_argument("--neighbours", type=int, default=4, help="Number of nearest neighbours for coupling (default: 4)")
    
    args = parser.parse_args()
    
    units.set_units(e_unit="cm-1", t_unit="fs")
    
    print(f"Running benchmark with R={args.realizations}, Steps={args.steps}, Jobs={args.jobs}")
    print(f"System sizes: {args.sizes}")
    with torch.no_grad():
        s, ns, nc, tc, tg = run_comparison(args.sizes, realizations=args.realizations, steps=args.steps, n_jobs=args.jobs, save_nise_output=not args.no_save_nise_output, n_neighbours=args.neighbours)
    plot(s, ns, nc, tc, tg, realizations=args.realizations, steps=args.steps, n_jobs=args.jobs, n_neighbours=args.neighbours)
