# examples/compare_mlnise_vs_tnise.py

"""
Example: Compare MLNISE-corrected vs TNISE vs None (no correction)
on a small test dataset. Plots final population dynamics.
"""

import torch
import matplotlib.pyplot as plt
import functools
from torchnise.nise import run_nise, MLNISEModel
from torchnise.mlnise_dataset import MLNiseDrudeDataset
from torchnise.units import set_units
from torchnise import units
from torchnise.spectral_functions import spectral_drude
import tqdm


def compute_mse(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean().item()


# Set consistent units
set_units(e_unit="cm-1", t_unit="fs")

# 1) Load the test dataset
test_ds = MLNiseDrudeDataset(
    dataset_folder="GeneratedHeom",
    dataset_name="mlnise_test_data",
    generate_if_missing=False,
)
print(f"Test dataset loaded: length = {len(test_ds)}")

# 2) Load the trained MLNISE model
ml_model = MLNISEModel()
ml_model.load_state_dict(torch.load("mlnise_model_final.pt"))
ml_model.eval()

device = torch.device("cpu")

# Initialize cumulative MSEs
mse_totals = {"None": 0.0, "TNISE": 0.0, "MLNISE": 0.0}
for idx in tqdm.tqdm(range(len(test_ds))):
    inputs, pop_target = test_ds[idx]
    pop_target = pop_target.to(device).squeeze(0)
    (
        h_torch,
        T_torch,
        reorg_torch,
        tau_torch,
        total_time_torch,
        dt_torch,
        psi0_torch,
        n_sites_torch,
    ) = [x.to(device).squeeze(0) for x in inputs]

    h_torch = h_torch.to(device)
    T = T_torch.item()
    reorg = reorg_torch.item()
    tau = tau_torch.item()
    total_time = total_time_torch.item()
    dt = dt_torch.item()
    psi0_torch = psi0_torch.to(device)
    n_sites = n_sites_torch.item()

    spectral_func = functools.partial(
        spectral_drude, temperature=T, strength=reorg, gamma=1 / tau
    )
    spectral_funcs = [spectral_func] * n_sites

    from torchnise.nise import NISEParameters

    for correction in mse_totals.keys():
        if correction == "MLNISE":
            mlnise_inputs = (
                torch.tensor([reorg], device=device),
                torch.tensor([tau], device=device),
            )
            params = NISEParameters(
                dt=dt,
                total_time=total_time,
                temperature=T,
                t_correction=correction,
                mode="Population",
                device=device,
                mlnise_model=ml_model,
                mlnise_inputs=mlnise_inputs,
            )
            pop_pred, _ = run_nise(
                h=h_torch,
                realizations=1000,
                initial_state=psi0_torch,
                spectral_funcs=spectral_funcs,
                params=params,
            )
        else:
            params = NISEParameters(
                dt=dt,
                total_time=total_time,
                temperature=T,
                t_correction=correction,
                mode="Population",
                device=device,
            )
            pop_pred, _ = run_nise(
                h=h_torch,
                realizations=1000,
                initial_state=psi0_torch,
                spectral_funcs=spectral_funcs,
                params=params,
            )

        # Make sure shapes match
        min_len = min(pop_pred.shape[0], pop_target.shape[0])
        pop_pred = pop_pred[:min_len]
        pop_ref = pop_target[:min_len]

        mse = compute_mse(pop_pred, pop_ref)
        print(correction, mse)
        mse_totals[correction] += mse

# Normalize by dataset size
for key in mse_totals:
    mse_totals[key] /= len(test_ds)

print("\nAverage MSE over test set:")
for method, mse in mse_totals.items():
    print(f"{method}: {mse:.6f}")
