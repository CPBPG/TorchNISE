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

def main():
    # Set consistent units
    set_units(e_unit="cm-1", t_unit="fs")

    # 1) Load the test dataset
    test_ds = MLNiseDrudeDataset(
        dataset_folder="GeneratedHeom",
        dataset_name="mlnise_test_data",
        generate_if_missing=False
    )
    print(f"Test dataset loaded: length = {len(test_ds)}")

    # 2) Load the trained MLNISE model
    ml_model = MLNISEModel()
    ml_model.load_state_dict(torch.load("mlnise_model_final.pt"))
    ml_model.eval()

    device = torch.device("cpu")

    # Let's pick a single item from the test set to compare
    # For a more robust comparison, you can do a loop over multiple items
    idx_to_check = 0
    (h_torch, T_torch, reorg_torch, tau_torch, total_torch, dt_torch, psi0_torch, n_sites_torch), pop_target = test_ds[idx_to_check]

    # Convert to device
    h_torch = h_torch.to(device)
    T = T_torch.item()
    reorg = reorg_torch.item()
    tau = tau_torch.item()
    total_time = total_torch.item()
    dt = dt_torch.item()
    psi0_torch = psi0_torch.to(device)
    n_sites = n_sites_torch.item()

    # Build spectral funcs if needed
    # But if run_nise is set up to generate noise from e.g. spectral_drude,
    # we can do that. For demonstration, we'll just pass an empty list 
    # and rely on static + noise approach from nise. Or define partial:
    # e.g. spectral_funcs = [functools.partial(...), ...]
    spectral_funcs = []

    # 3) Compare three corrections: "None", "TNISE", and "MLNISE"
    #    We'll store the results in a dict
    corrections = ["None", "TNISE", "MLNISE"]
    results = {}

    for correction in corrections:
        if correction == "MLNISE":
            # We must pass the ML model plus inputs
            # Typically reorg + tau or whichever your code uses
            mlnise_inputs = (
                torch.tensor([reorg], device=device),
                torch.tensor([tau], device=device)
            )
            pop_pred, time_axis = run_nise(
                h=h_torch,
                realizations=1,
                total_time=total_time,
                dt=dt,
                initial_state=psi0_torch,
                temperature=T,
                spectral_funcs=spectral_funcs,
                t_correction=correction,
                mode="Population",
                device=device,
                mlnise_model=ml_model,
                mlnise_inputs=mlnise_inputs
            )
        else:
            pop_pred, time_axis = run_nise(
                h=h_torch,
                realizations=1,
                total_time=total_time,
                dt=dt,
                initial_state=psi0_torch,
                temperature=T,
                spectral_funcs=spectral_funcs,
                t_correction=correction,
                mode="Population",
                device=device
            )
        results[correction] = (pop_pred, time_axis)

    # 4) Plot
    plt.figure(figsize=(6,4))
    # Plot the "None" vs "TNISE" vs "MLNISE"
    for cor in corrections:
        pop_pred, t_axis = results[cor]
        # pop_pred: shape (time_steps, n_sites)
        for site_idx in range(n_sites):
            plt.plot(
                t_axis, pop_pred[:, site_idx],
                label=f"{cor} - site {site_idx+1}"
            )

    # Optionally, plot the "target" from the dataset if you want:
    # (That is the PyHEOM-based population.)
    # For a direct comparison, you may need to unify time steps if there's any difference.
    # Usually the shape is (time_steps, n_sites). We can plot it if we like:
    time_axis_heom = torch.linspace(0, total_time, pop_target.shape[0])
    for site_idx in range(n_sites):
        plt.plot(
            time_axis_heom, pop_target[:, site_idx],
            "--", label=f"HEOM ref - site {site_idx+1}"
        )

    plt.xlabel("time [fs]")
    plt.ylabel("Population")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
