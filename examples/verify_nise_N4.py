
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt

WORK_DIR = "nise_N4_verification"
NISE_BIN = os.path.abspath("examples/NISE_2017/bin/NISE")
TRANSLATE_BIN = os.path.abspath("examples/NISE_2017/bin/translate")

def run_verification(N=4, steps=1000,realizations=10000):
    os.makedirs(WORK_DIR, exist_ok=True)
    
    # Generate Data
    # GROASC Format
    # Energy: Step He_00 He_01 He_02 ... (Row-major upper triangular)
    
    energy_data = []
    
    # 1D Chain with nearest neighbor coupling
    # Diagonal 0, Coupling 100
    H = np.zeros((N, N))
    for i in range(N):
        H[i, i] = 0.0
        if i < N - 1:
            H[i, i+1] = 10
            H[i+1, i] = 10
            
    # Flatten to list for each step
    row = [0] # Time 0 placeholder
    for k in range(N):
        for l in range(k, N):
            row.append(H[k,l])
            
    # Replicate for all steps + buffer
    traj_steps = steps + realizations
    for t in range(traj_steps):
        r = list(row)
        r[0] = t
        energy_data.append(r)
        
    # Dummy Dipole (Required by NISE even if unused?)
    dipole_data = []
    d_row = [0] + [1.0]*N + [0.0]*N + [0.0]*N
    for t in range(traj_steps):
        r = list(d_row)
        r[0] = t
        dipole_data.append(r)
        
    np.savetxt(os.path.join(WORK_DIR, "Energy.txt"), energy_data, fmt="%g")
    np.savetxt(os.path.join(WORK_DIR, "Dipole.txt"), dipole_data, fmt="%g")
    
    # Create inpTra
    inp_tra = f"""InputEnergy Energy.txt
InputDipole Dipole.txt
OutputEnergy Energy.bin
OutputDipole Dipole.bin
OutputAnharm Anharmonicity.bin
OutputOverto OvertoneDipole.bin
Singles {N}
Doubles 0
Skip Doubles
Length {traj_steps}
InputFormat GROASC
OutputFormat GROBIN
"""
    with open(os.path.join(WORK_DIR, "inpTra"), "w") as f:
        f.write(inp_tra)
        
    # Run Translate
    print("Running translate...")
    subprocess.run(f"{TRANSLATE_BIN} inpTra", shell=True, cwd=WORK_DIR, check=True)
    
    # Create NISE input
    input_content = f"""Propagation Sparse
Couplingcut 0
Threshold 0.0
Hamiltonianfile Energy.bin
Dipolefile Dipole.bin
Anharmonicfile Anharmonicity.bin
Overtonedipolefile OvertoneDipole.bin
Length {traj_steps}
Samplerate 1
Lifetime {steps}
Timestep 1.0
Trotter 1
Format Dislin
Anharmonicity 0
MinFrequencies -2000
MaxFrequencies 2000
Technique Pop
FFT 1024
RunTimes {steps} 0 0
BeginPoint 0
EndPoint {realizations}
Singles {N}
Sites {N}
InitialState 1
"""
    with open(os.path.join(WORK_DIR, "input"), "w") as f:
        f.write(input_content)
        
    # Run NISE
    print("Running NISE...")
    os.environ['OMP_NUM_THREADS'] = '1'
    #subprocess.run(f"mpirun -np 1 -x OMP_NUM_THREADS=16 --bind-to socket {NISE_BIN} input", shell=True, cwd=WORK_DIR, check=True)
    subprocess.run(f"{NISE_BIN} input", shell=True, cwd=WORK_DIR, check=True)
    print("NISE Run Complete. Listing output directory:")
    subprocess.run(f"ls -l {WORK_DIR}", shell=True)
    
    # Check output
    # Population output is typically Pop_site<N>.dat or similar? 
    # Or Population.dat?
    # Let's try to find any .dat file
    
    dat_files = [f for f in os.listdir(WORK_DIR) if f.endswith("TD_Absorption.dat")]
    if dat_files:
        print(f"Population files found: {dat_files}")
        plt.figure()
        for df in dat_files:
            data = np.loadtxt(os.path.join(WORK_DIR, df))
            # Usually format: Time Pop
            # If multiple columns, maybe Time Pop1 Pop2...
            if data.ndim == 2:
                if data.shape[1] > 1:
                    plt.plot(data[:,0], data[:,1:], label=df)
                else:
                    plt.plot(data, label=df)
        
        plt.title(f"NISE N={N} Population Dynamics")
        plt.xlabel("Time (fs)")
        plt.ylabel("Population")
        plt.legend()
        plt.savefig("NISE_N4_Population.png")
        print("Plot saved to NISE_N4_Population.png")
    else:
        print("Population file NOT found. Check logs.")

if __name__ == "__main__":
    run_verification()
