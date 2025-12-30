
"""
Benchmark script comparing TorchNISE (CPU/GPU) vs NISE_2017 scaling performance.
Extends to N=1000.
"""

import time
import subprocess
import os
import shutil
import struct
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchnise.nise
import torchnise.units as units
from torchnise.spectral_functions import spectral_drude_lorentz_heom
import functools

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
NISE_BIN = os.path.join(EXAMPLES_DIR, "NISE_2017/bin/NISE")
WORK_DIR = os.path.abspath("benchmark_workdir")

def write_nise_inputs_base(n_sites, steps=1000, dt=1.0, total_realizations=1):
    """Generates base input files and binaries for NISE_2017."""
    os.makedirs(WORK_DIR, exist_ok=True)
    
    # NISE needs data > simulation time
    # Trajectory length
    # To run "steps" simulation, we need at least steps+1 data points.
    # We generate steps+2 to get ~1 sample.
    traj_steps = steps + total_realizations
    
    # 1. Energy - 0 diag, 100 coupling
    # Upper triangular row major
    energy_data = []
    
    # Define H for one step
    # n_sites = 2
    # H = torch.zeros((n_sites, n_sites), dtype=torch.float32)
    # for i in range(n_sites):
    #     H[i, i] = 0  
    #     if i < n_sites - 1:
    #         V = 100  
    #         H[i, i + 1] = V
    #         H[i + 1, i] = V

    H_step = np.zeros((n_sites, n_sites))
    for i in range(n_sites-1):
        H_step[i, i+1] = 100.0
        H_step[i+1, i] = 100.0
    for i in range(n_sites-2):
        H_step[i, i+2] = 50.0
        H_step[i+2, i] = 50.0
    for i in range(n_sites-3):
        H_step[i, i+3] = 25.0
        H_step[i+3, i] = 25.0
    for i in range(n_sites-4):
        H_step[i, i+4] = 12.5
        H_step[i+4, i] = 12.5
        
    row = [0]
    for k in range(n_sites):
        for l in range(k, n_sites):
            row.append(H_step[k,l])
            
    # Replicate for all steps
    for t in range(traj_steps):
        r = list(row)
        r[0] = t
        energy_data.append(r)

    # 2. Dipole - Dummy
    dipole_data = []
    d_row = [0] + [1.0]*n_sites + [0.0]*n_sites + [0.0]*n_sites
    for t in range(traj_steps):
        r = list(d_row)
        r[0] = t
        dipole_data.append(r)
    
    # Save TXT for translate
    np.savetxt(os.path.join(WORK_DIR, "Energy.txt"), energy_data, fmt="%g")
    np.savetxt(os.path.join(WORK_DIR, "Dipole.txt"), dipole_data, fmt="%g")

    # inpTra file
    inp_tra = f"""
InputEnergy Energy.txt
InputDipole Dipole.txt
OutputEnergy Energy.bin
OutputDipole Dipole.bin
OutputAnharm Anharmonicity.bin
OutputOverto OvertoneDipole.bin
Singles {n_sites}
Doubles 0
Skip Doubles
Length {traj_steps}
InputFormat GROASC
OutputFormat GROBIN
"""
    with open(os.path.join(WORK_DIR, "inpTra"), "w") as f:
        f.write(inp_tra)
        
    # Run Translate once
    cmd_translate = f"{os.path.join(EXAMPLES_DIR, 'NISE_2017/bin/translate')} inpTra"
    with open(os.path.join(WORK_DIR, "translate.out"), "w") as outfile:
        subprocess.run(cmd_translate, shell=True, cwd=WORK_DIR, stdout=outfile, stderr=subprocess.STDOUT, check=True)

    return traj_steps

def create_job_input(job_dir, n_sites, steps, dt, begin_point, end_point, traj_steps):
    """Creates input file for a specific job."""
    # 5. input file
    
    # RunTimes {steps} 0 0: Duration {steps}, Start 0, max T3 0 (doesn't matter for Pop)
    # BeginPoint 0, EndPoint 1: 1 realization.
    
    input_content = f"""Propagation Sparse
Couplingcut 0
Threshold 0.0
Hamiltonianfile Energy.bin
Dipolefile Dipole.bin
Anharmonicfile Anharmonicity.bin
Overtonedipolefile OvertoneDipole.bin
Length {traj_steps}
Samplerate 1
Lifetime {steps*dt}
Timestep {dt}
Format Dislin
Anharmonicity 0
MinFrequencies -2000
MaxFrequencies 2000
Technique Pop
FFT 2048
RunTimes {steps} 0 0
BeginPoint {begin_point}
EndPoint {end_point}
Singles {n_sites}
Doubles 0
Skip Doubles
Sites {n_sites}
InitialState 1
"""

    with open(os.path.join(job_dir, "input"), "w") as f:
        f.write(input_content)

def run_nise_job(args):
    """Worker function for running a NISE job."""
    os.environ['OMP_NUM_THREADS'] = '1'
    job_id, n_sites, steps, dt, begin, end, traj_steps = args
    
    job_dir = os.path.join(WORK_DIR, f"job_{job_id}")
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
    create_job_input(job_dir, n_sites, steps, dt, begin, end, traj_steps)
    
    # Use mpirun to restrict OpenMP threads per process as requested by user
    cmd_nise = f"{NISE_BIN} input"
    
    with open(os.path.join(job_dir, "nise.out"), "w") as outfile:
        # stdin=subprocess.DEVNULL is good practice to avoid hangs on background proc
        subprocess.run(cmd_nise, shell=True, cwd=job_dir, stdout=outfile, stderr=subprocess.STDOUT, check=True, stdin=subprocess.DEVNULL)

def benchmark_nise_2017(n_sites, steps=1000, dt=1.0, realizations=1, n_jobs=1):
    traj_steps = write_nise_inputs_base(n_sites, steps, dt, realizations)

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
            job_args.append((i, n_sites, steps, dt, current_begin, current_end, traj_steps))
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

def benchmark_torchnise(n_sites, steps=1000, dt=1.0, device="cpu", realizations=1):
    # Setup same system: 0-diag, 100-coupling
    start = time.time()
    
    # Create Constant Hamiltonian
    # H shape: (n_sites, n_sites) if constant_v used, or (steps, n_sites, n_sites)
    
    H = torch.zeros((n_sites, n_sites)).to(device)
    # Diagonal 0
    # Coupling 100
    for i in range(n_sites-1):
        H[i, i+1] = 100.0
        H[i+1, i] = 100.0
    for i in range(n_sites-2):
        H[i, i+2] = 50
        H[i+2, i] = 50
    for i in range(n_sites-3):
        H[i, i+3] = 25
        H[i+3, i] = 25
    for i in range(n_sites-4):
        H[i, i+4] = 12.5
        H[i+4, i] = 12.5
    
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

def run_comparison(sizes, realizations=1, steps=1000, n_jobs=1):
    t_nise = []
    t_torch_cpu = []
    t_torch_gpu = []
    
    print(f"{'N':<5} | {'NISE_2017':<10} | {'Torch CPU':<10} | {'Torch GPU':<10}")
    print("-" * 50)
    
    for N,realizations_N in zip(sizes,realizations):
        # Benchmark steps = 1000
        # Benchmark steps from args
        # steps = steps

        # Run NISE 2017 -> Limit to N=100 as N=200 takes too long (>3 min)
        if N <= 150:
             tn = benchmark_nise_2017(N, steps=steps, dt=1.0, realizations=realizations_N, n_jobs=n_jobs)

            
               
        else:
             tn = None
        
        # Run TorchNISE CPU
        if N <= 5000:
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
            
        t_nise.append(tn)
        t_torch_cpu.append(tt_cpu)
        t_torch_gpu.append(tt_gpu)
        
        tn_str = f"{tn:.4f}" if tn is not None else "N/A"
        tc_str = f"{tt_cpu:.4f}" if tt_cpu is not None else "N/A"
        tg_str = f"{tt_gpu:.4f}" if tt_gpu is not None else "N/A"
        
        print(f"{N:<5} | {tn_str:<10} | {tc_str:<10} | {tg_str:<10}")
        
    return sizes, t_nise, t_torch_cpu, t_torch_gpu

def plot(sizes, t_nise, t_torch_cpu, t_torch_gpu, realizations=1, steps=1000, n_jobs=1):
    plt.figure(figsize=(10,6))
    
    # Filter Nones
    mask_nise = [x is not None for x in t_nise]
    if any(mask_nise):
        plt.plot(np.array(sizes)[mask_nise], np.array(t_nise)[mask_nise], 'o--', label='NISE_2017 (Verified)')
        
    mask_cpu = [x is not None for x in t_torch_cpu]
    if any(mask_cpu):
        plt.plot(np.array(sizes)[mask_cpu], np.array(t_torch_cpu)[mask_cpu], 's-', label='TorchNISE (CPU)')

    mask_gpu = [x is not None for x in t_torch_gpu]
    if any(mask_gpu):
         plt.plot(np.array(sizes)[mask_gpu], np.array(t_torch_gpu)[mask_gpu], '^-', label='TorchNISE (GPU)')

    plt.xlabel('System Size N')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.title(f'Performance Comparison: Population Dynamics 100 steps')
    plt.xscale('log') 
    plt.yscale('log') 
    plt.grid(True, which="both", ls="-")
    plt.savefig(f'comparison_benchmark_{realizations}_{steps}_{max(sizes)}_j{n_jobs}_4_neighbours.png')
    print(f"Saved comparison_benchmark_{realizations}_{steps}_{max(sizes)}_j{n_jobs}_4_neighbours.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark TorchNISE vs NISE_2017")
    parser.add_argument("--realizations", type=int, default=[10000, 10000, 10000,1000,100,100,10,10,1,1,1,1], help="Number of realizations for TorchNISE (default: 1)")
    parser.add_argument("--steps", type=int, default=100, help="Number of time steps (default: 1000)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000], help="List of system sizes N to benchmark")
    parser.add_argument("--jobs", type=int, default=16, help="Number of parallel jobs for NISE 2017")
    
    args = parser.parse_args()
    
    units.set_units(e_unit="cm-1", t_unit="fs")
    
    print(f"Running benchmark with R={args.realizations}, Steps={args.steps}, Jobs={args.jobs}")
    print(f"System sizes: {args.sizes}")
    with torch.no_grad():
        s, n, tc, tg = run_comparison(args.sizes, realizations=args.realizations, steps=args.steps, n_jobs=args.jobs)
    plot(s, n, tc, tg, realizations=args.realizations, steps=args.steps, n_jobs=args.jobs)
