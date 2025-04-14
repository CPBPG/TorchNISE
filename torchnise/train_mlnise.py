# torchnise/train_mlnise.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import your nise.py stuff exactly as is:
# nise.py must have:  from nise import MLNISEModel, run_nise
# If that file is in the same package, do:
from torchnise.nise import MLNISEModel, run_nise
from torchnise.mlnise_dataset import MLNiseDrudeDataset

def main():
    # 1) Load dataset
    train_ds = MLNiseDrudeDataset(
        length=20,
        total_time=100.0,
        dt_fs=1.0,
        dataset_name="mlnise_demo_data",
        generate_if_missing=False
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

    # 2) Instantiate the MLNISE model from nise.py (unchanged)
    mlnise_model = MLNISEModel()
    # Optionally move to GPU if you like
    device = "cpu"
    mlnise_model.to(device)

    optimizer = optim.Adam(mlnise_model.parameters(), lr=1e-3)
    n_epochs = 5

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_idx, (inputs, pop_target) in enumerate(train_loader):
            """
            inputs = (H, T, E_reorg, tau, total_time, dt, psi0, n_sites)
              - H: (time_steps, n_sites, n_sites)
              - T, E_reorg, tau, total_time, dt, shape: (1,)
              - psi0: (n_sites,)
              - n_sites: (1,) integer
            pop_target: (time_steps, n_sites)
            """
            # Move them to device
            (H, T, E_reorg, tau, total_time, dt_fs, psi0, n_sites) = [
                x.to(device) for x in inputs
            ]
            pop_target = pop_target.to(device)

            # Reshape H to the shape expected by run_nise for 1 realization:
            # run_nise wants H of shape (time_steps, realizations, n_sites, n_sites).
            # We'll do realizations=1 -> (time_steps, 1, n_sites, n_sites).
            H = H.unsqueeze(1)  # (time_steps, 1, n_sites, n_sites)

            # We also define t_correction="MLNISE"
            # For the correction logic, nise.py calls:
            #    s = apply_t_correction(..., mlnise_model=..., mlnise_inputs=...)
            # so we must pass `mlnise_model=mlnise_model` and
            # `mlnise_inputs=(reorg, tau)` or whatever your code uses.
            reorg_float = E_reorg.item()
            tau_float = tau.item()
            # In your code, the forward pass references: "mlnise_inputs[0]" as reorg
            # and "mlnise_inputs[1]" as correlation time. So do:
            mlnise_inputs = (torch.tensor([reorg_float], device=device),
                             torch.tensor([tau_float], device=device))

            # run_nise returns (avg_output, time_axis) for mode="Population" 
            # By default, it might not have a "mlnise_model" arg in the function signature.
            # If your nise.py doesn't have that, you can pass it as a leftover kwarg:
            #   run_nise(..., mlnise_model=mlnise_model)
            # or if that isn't recognized, you'd need to confirm that `nise_averaging` 
            # passes it along to `nise_propagate`.  
            # For demonstration, we assume we can do:
            pop_pred, t_axis = run_nise(
                h=H,
                realizations=1,
                total_time=total_time.item(),
                dt=dt_fs.item(),
                initial_state=psi0,
                temperature=T.item(),
                spectral_funcs=[],  # Not generating noise here
                t_correction="MLNISE",
                mode="Population",
                device=device,
                mlnise_model=mlnise_model,       # crucial
                mlnise_inputs=mlnise_inputs      # crucial
            )
            # pop_pred shape -> (time_steps, n_sites)

            # Make sure shape matches target
            # Some versions might yield an off-by-one in time steps, so verify:
            min_len = min(pop_pred.shape[0], pop_target.shape[0])
            loss = F.mse_loss(pop_pred[:min_len], pop_target[:min_len])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.6f}")

    # Save final
    torch.save(mlnise_model.state_dict(), "mlnise_model_final.pt")
    print("Training complete. Model saved as mlnise_model_final.pt")

if __name__ == "__main__":
    main()
