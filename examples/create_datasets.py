# examples/create_datasets.py

"""
Example: Creating two datasets (train + test) for MLNISE usage.
"""
import numpy as np
import torch
from torchnise.mlnise_dataset import MLNiseDrudeDataset,runHeomDrude
import matplotlib.pyplot as plt

def main():

    pop = runHeomDrude(
    2,
    np.array([[100,100],[100,0]]),
    tau=100,
    Temp=300,
    E_reorg=20,
    dt__unit=1,
    initiallyExcitedState=0,
    totalTime=1000,
    tier=7,
    matrix_type = "dense"
    )
    print(pop[0,:])
    print(pop[1,:])
    plt.plot(pop[:,0])
    plt.plot(pop[:,1])
    plt.show()
    plt.close()
    # 1) Create a training dataset
    train_ds = MLNiseDrudeDataset(
        length=50,              # number of random Hamiltonians
        total_time=100.0,       # total simulation time (fs)
        dt_fs=1.0,              # timestep (fs)
        n_sites=2,
        seed=1234,              # random seed for reproducibility
        depth=7,
        dataset_folder="GeneratedHeom",
        dataset_name="mlnise_train_data_smalltest_short",  # will create a subfolder
        generate_if_missing=True
    )
    print(f"Training dataset created with length = {len(train_ds)}")

    # 2) Create a testing dataset (shorter or different seed, etc.)
    test_ds = MLNiseDrudeDataset(
        length=10,
        total_time=100.0,
        dt_fs=1.0,
        n_sites=2,
        seed=9999,
        depth=7,
        dataset_folder="GeneratedHeom",
        dataset_name="mlnise_test_data_smalltest_short",
        generate_if_missing=True
    )
    print(f"Testing dataset created with length = {len(test_ds)}")

    # Datasets are automatically saved to 'GeneratedHeom/mlnise_train_data' and
    # 'GeneratedHeom/mlnise_test_data' in .npy files. Future runs will load
    # from disk, unless you change "generate_if_missing=False".

if __name__ == "__main__":
    main()
