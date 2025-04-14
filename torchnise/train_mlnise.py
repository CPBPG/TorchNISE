# torchnise/train_mlnise.py

"""
This module demonstrates Hogwild-style training of an MLNISE model 
using a PyTorch Dataset. It is intended to be imported and called 
from another script (not from the command line directly).

Example usage in a separate script:

    import torch
    from torchnise.train_mlnise import train_mlnise_hogwild
    from torchnise.nise import MLNISEModel
    from torchnise.mlnise_dataset import MLNiseDrudeDataset

    # Create a dataset
    dataset = MLNiseDrudeDataset(length=50, total_time=500.0, dt_fs=1.0, n_sites=2)

    # Instantiate the model
    model = MLNISEModel()
    model.share_memory()  # for Hogwild

    # Train
    trained_model = train_mlnise_hogwild(
        model=model,
        dataset=dataset,
        num_epochs=20,
        batch_size=10,
        num_processes=4,
        learning_rate=0.1,
        runname="testrun",
        realizations=1  # or however many you want
    )
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, Callable
import functools

from torchnise.nise import run_nise
from torchnise.spectral_functions import spectral_drude

def _train_one_epoch(
    model: torch.nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: Optimizer,
    epoch: int,
    realizations: int,
    log_fn: Optional[Callable[[str], None]] = None
) -> None:
    """
    Performs one epoch of training on a single worker (process) in Hogwild mode.

    Args:
        model (torch.nn.Module): The shared-memory model to be trained (MLNISE).
        device (torch.device): CPU or GPU device to run on.
        train_loader (DataLoader): DataLoader subset for this worker.
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates.
        epoch (int): Current epoch index (for logging).
        realizations (int): Number of noise realizations to pass to run_nise(...).
        log_fn (Callable, optional): Logging function. Defaults to print() if None.
    """
    # Force single-threaded execution in each process
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    model.train()
    if log_fn is None:
        def log_fn(msg: str) -> None:
            print(msg)

    for batch_idx, (inputs, pop_target) in enumerate(train_loader):
        """
        inputs = (H, T, E_reorg, tau, total_time, dt, psi0, n_sites)
            - H: (n_sites, n_sites), for a static Hamiltonian,
                                noise is generated inside run_nise.
            - T, E_reorg, tau, total_time, dt: shape (1,)
            - psi0: (n_sites,)
            - n_sites: (1,) integer
        pop_target: shape (time_steps, n_sites)
        """
        # Move them to the device
        (h, temperature, E_reorg, tau, total_time, dt_fs, psi0, n_sites) = [
            x.to(device) for x in inputs
        ]
        pop_target = pop_target.to(device)

        # Prepare MLNISE model inputs:
        reorg_float = E_reorg.item()
        tau_float = tau.item()
        mlnise_inputs = (
            torch.tensor([reorg_float], device=device),
            torch.tensor([tau_float], device=device)
        )
        gamma = 1 / tau
        spectral_func = functools.partial(spectral_drude,temperature=temperature,
                                           stenght=reorg_float, gamma=gamma)
        spectral_funcs = [spectral_func]*n_sites
        # Use run_nise to get predicted populations
        pop_pred, t_axis = run_nise(
            h=h,
            realizations=realizations,
            total_time=total_time.item(),
            dt=dt_fs.item(),
            initial_state=psi0,
            temperature=temperature.item(),
            spectral_funcs=spectral_funcs,
            t_correction="MLNISE",
            mode="Population",
            device=device,
            mlnise_model=model,       # important for the correction
            mlnise_inputs=mlnise_inputs
        )
        # pop_pred has shape (time_steps, n_sites).

        # Match length if there's any off-by-one
        min_len = min(pop_pred.shape[0], pop_target.shape[0])
        loss = F.mse_loss(pop_pred[:min_len], pop_target[:min_len])

        optimizer.zero_grad()
        loss.backward()

        # Check for NaNs
        skip_update = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                if torch.isnan(param.grad).any() or torch.isnan(param.data).any():
                    skip_update = True
                    log_fn(f"[WARN] NaN encountered in {name} -> skipping update.")
                    break

        if not skip_update:
            optimizer.step()

        log_fn(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")


def train_mlnise_hogwild(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    num_epochs: int = 10,
    batch_size: int = 100,
    num_processes: int = 4,
    learning_rate: float = 0.1,
    runname: str = "testrun",
    device: Optional[torch.device] = None,
    scheduler_class: Optional[type] = None,
    scheduler_kwargs: Optional[dict] = None,
    save_interval: int = 5,
    save_checkpoint_fn: Optional[Callable[[int, torch.nn.Module], None]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    realizations: int = 1
) -> torch.nn.Module:
    """
    Trains a given MLNISE model in a Hogwild (multi-process) fashion.
    Each process operates on a partition of the dataset with shared parameters.

    Args:
        model (torch.nn.Module): The MLNISE model to train. Must have 
            share_memory() called on it if using Hogwild.
        dataset (Dataset): The dataset from which samples are drawn 
            (e.g. MLNiseDrudeDataset).
        num_epochs (int, optional): Number of epochs to train. Defaults to 10.
        batch_size (int, optional): Batch size per worker. Defaults to 100.
        num_processes (int, optional): Number of parallel processes for Hogwild. Defaults to 4.
        learning_rate (float, optional): Initial learning rate. Defaults to 0.1.
        runname (str, optional): Name of the training run (used in logs). Defaults to "testrun".
        device (torch.device, optional): Device to run on. If None, uses CPU.
        scheduler_class (type, optional): A scheduler class like StepLR or ReduceLROnPlateau.
            If None, no scheduler is used.
        scheduler_kwargs (dict, optional): Keyword arguments for the scheduler.
            If None, uses default arguments.
        save_interval (int, optional): Save checkpoint every N epochs. Defaults to 5.
        save_checkpoint_fn (Callable, optional): Function to handle model checkpointing,
            e.g. `lambda ep, m: torch.save(m.state_dict(), f"checkpoint_{ep}.pt")`.
            If None, no checkpoints are saved except final.
        log_fn (Callable, optional): Logging function. If None, logs go to stdout.
        realizations (int, optional): Number of noise realizations to pass to run_nise(...).
            If you want multi-realization runs, ensure your `H` shape is 
            (time_steps, realizations, n_sites, n_sites). Defaults to 1.

    Returns:
        torch.nn.Module: The trained model (same object).
    """
    if device is None:
        device = torch.device("cpu")

    if log_fn is None:
        def log_fn(msg: str) -> None:
            print(msg)

    # Prepare an optimizer
    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)

    # Optional scheduler (e.g. StepLR)
    scheduler = None
    if scheduler_class is not None:
        if scheduler_kwargs is None:
            scheduler_kwargs = {}
        scheduler = scheduler_class(optimizer, **scheduler_kwargs)

    # Main epoch loop
    for epoch in range(1, num_epochs + 1):
        log_fn(f"===== Starting epoch {epoch}/{num_epochs} =====")
        log_fn(f"Runname: {runname}")

        # Hogwild: spawn multiple processes, each with a DistributedSampler
        processes = []
        for rank in range(num_processes):
            sampler = DistributedSampler(
                dataset=dataset,
                num_replicas=num_processes,
                rank=rank,
                shuffle=True,
                seed=123  # or configure as you like
            )
            train_loader = DataLoader(
                dataset=dataset,
                sampler=sampler,
                batch_size=batch_size
            )

            p = mp.Process(
                target=_train_one_epoch,
                args=(model, device, train_loader, optimizer, epoch, realizations, log_fn)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Scheduler step
        if scheduler is not None:
            # If it's ReduceLROnPlateau, pass a metric if needed
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # e.g. scheduler.step(metric)
                scheduler.step()
            else:
                scheduler.step()

        # Save checkpoint at intervals
        if save_checkpoint_fn is not None and (epoch % save_interval == 0):
            save_checkpoint_fn(epoch, model)

    # Final checkpoint
    if save_checkpoint_fn is not None:
        save_checkpoint_fn(num_epochs, model)

    log_fn("Training complete.")
    return model
