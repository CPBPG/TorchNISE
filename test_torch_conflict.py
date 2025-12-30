import torch
from torchnise.mlnise_dataset import runHeomDrude
import numpy as np
from multiprocessing import Process
import time

def run_one():
    pop = runHeomDrude(
        2,
        np.array([[100, 100], [100, 0]]),
        tau=100,
        Temp=300,
        E_reorg=20,
        dt__unit=1,
        initiallyExcitedState=0,
        totalTime=1000,
        tier=7,
        matrix_type="dense",
    )

if __name__ == "__main__":
    print("Starting multiprocessing test loop WITH TORCH IMPORTED")
    try:
        import multiprocessing
        multiprocessing.set_start_method("spawn", force=True)
    except:
        pass
    
    # Loop run in separate processes
    for i in range(10):
        try:
            p = Process(target=run_one)
            p.start()
            p.join()
            if p.exitcode != 0:
                print(f"Run {i+1} failed with exit code {p.exitcode}")
                break
            print(f"Run {i+1} success")
        except Exception as e:
            print(f"Run {i+1} exception: {e}")
