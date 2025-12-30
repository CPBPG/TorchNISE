import numpy as np
import torch
from torchnise.mlnise_dataset import runHeomDrude
import matplotlib.pyplot as plt
import multiprocessing

def run_manual():
    print("Running manual example...")
    try:
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
        print("Success")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    p = multiprocessing.Process(target=run_manual)
    p.start()
    p.join()
    print(f"Exit code: {p.exitcode}")
