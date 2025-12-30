
"""
Benchmark script for TorchNISE scaling over realizations.
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchnise.nise
import torchnise.units as units
from torchnise.spectral_functions import spectral_drude_lorentz_heom
from torchnise.nise import NISEParameters
import functools

def run_benchmark_realizations(sizes=[10, 50, 100], realizations_list=[1, 10, 100, 1000, 10000], total_time=1000, dt=1.0, time_limit=60.0):
    
    results = {} # structure: { size: { 'cpu': {r: time, ...}, 'cuda': {r: time, ...} } }

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    else:
        print("Warning: CUDA not available, skipping GPU benchmark.")

    print(f"Benchmarking Realization Scaling...")
    print(f"Sizes: {sizes}")
    print(f"Realizations: {realizations_list}")
    print(f"Time Limit: {time_limit}s per config")
    print("-" * 60)

    for n_sites in sizes:
        results[n_sites] = {}
        
        # Setup System (invariant for device/realizations)
        H = torch.zeros((n_sites, n_sites), dtype=torch.float32)
        for i in range(n_sites):
            H[i, i] = 0 
            if i < n_sites - 1:
                H[i, i + 1] = 100
                H[i + 1, i] = 100
        
        T = 300
        Omega_k = torch.tensor([0, 725]) / units.HBAR
        lambda_k = torch.tensor([20, 20])
        v_k = torch.tensor([1/100, 1/100])
        spectralfunc = functools.partial(spectral_drude_lorentz_heom, omega_k=Omega_k, lambda_k=lambda_k, vk=v_k, temperature=T)
        spectral_funcs = [spectralfunc] * n_sites
        initialState = torch.zeros(n_sites); initialState[0] = 1

        for device in devices:
            results[n_sites][device] = {}
            print(f"Testing N={n_sites} Device={device}")
            
            for r in realizations_list:
                # Time Check: If previous run for this N/device exceeded limit, stopping early is better than checking after?
                # User requirement: "if it is exceeded exclude that one from the larger system sizes" -> Ambiguous.
                # My interp: Check previous run. But R increases. So if R=100 was > limit, don't run R=1000.
                pass 
                # Actually, I'll run and check AFTER. 
                # But if R=1000 took 70s, R=10000 will take 700s. I should skip!
                
                # Check if previous R for same N/device was already too slow or skipped?
                # Efficient check: if we already have data for this N/device, check last time.
                prev_rs = sorted(results[n_sites][device].keys())
                if prev_rs:
                    last_r = prev_rs[-1]
                    last_time = results[n_sites][device][last_r]
                    if last_time > time_limit:
                        print(f"  Skipping R={r} (Time limit exceeded at R={last_r}: {last_time:.2f}s)")
                        continue
                
                start_time = time.time()
                try:
                    params = NISEParameters(
                        dt=dt,
                        total_time=total_time,
                        temperature=T,
                        t_correction="None",
                        mode="Population",
                        device=device,
                        save_interval=100 # reduce I/O overhead
                    )
                    
                    # Move H to device? run_nise handles it if we pass CPU, but passing params.device
                    # H.cpu() is safe.
                    
                    torchnise.nise.run_nise(
                        H,
                        r,
                        initialState,
                        spectral_funcs,
                        params,
                    )
                    duration = time.time() - start_time
                    results[n_sites][device][r] = duration
                    print(f"  R={r:<6} | {duration:.4f}s")
                    
                    if duration > time_limit:
                         print(f"  -> Time limit reached ({duration:.2f}s > {time_limit}s). Stopping scaling for N={n_sites}, Device={device}.")
                         # Break R loop for this device/N
                         # The next iterations of R loop will be handled by the 'continue' check above or break here.
                         pass
                         
                except Exception as e:
                    print(f"  Error R={r}: {e}")
                    break # Stop R loop on error

    return results

def plot_benchmark(results, filename='realization_scaling.png'):
    plt.figure(figsize=(10, 7))
    
    # Colors for sites
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(results)))
    
    for idx, (n, device_dict) in enumerate(results.items()):
        color = colors[idx]
        
        # CPU
        if 'cpu' in device_dict and device_dict['cpu']:
            rs = sorted(device_dict['cpu'].keys())
            ts = [device_dict['cpu'][r] for r in rs]
            plt.plot(rs, ts, 'o-', color=color, label=f'N={n} (CPU)')
            
        # GPU
        if 'cuda' in device_dict and device_dict['cuda']:
            rs = sorted(device_dict['cuda'].keys())
            ts = [device_dict['cuda'][r] for r in rs]
            plt.plot(rs, ts, '^--', color=color, label=f'N={n} (GPU)')

    plt.xlabel('Realizations')
    plt.ylabel('Time (s)')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Benchmark: Wall Time vs Realizations')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig(filename)
    print(f"Saved {filename}")

if __name__ == "__main__":
    units.set_units(e_unit="cm-1", t_unit="fs")
    # Reduced sizes list to keep runtime reasonable for interactive check
    # But user asked for "1 to 10000".
    # I'll enable 10000 but time limit will catch it.
    res = run_benchmark_realizations(
        sizes=[2,10, 50, 100,500,1000], 
        realizations_list=[1,2,5, 10,20,50, 100,200,500, 1000,2000,5000, 10000], 
        time_limit=120.0 # Strict limit to return quickly
    )
    plot_benchmark(res)
