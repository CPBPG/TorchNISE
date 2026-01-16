# torchnise/train_mlnise.py

"""
This module demonstrates Hogwild-style training of an MLNISE model
using a PyTorch Dataset, refactored into a `NISETrainer` class.
"""
import contextlib
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
import torch.multiprocessing as mp
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, Callable
import functools
import tqdm

from torchnise.nise import run_nise
from torchnise.spectral_functions import spectral_drude


@contextlib.contextmanager
def suppress_stdout():
    old_stdout = sys.stdout
    tqdm_backup = tqdm.tqdm

    def silent_tqdm(*args, **kwargs):
        return tqdm_backup(*args, disable=True, **kwargs)

    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        tqdm.tqdm = silent_tqdm
        try:
            yield
        finally:
            sys.stdout = old_stdout
            tqdm.tqdm = tqdm_backup

class NISETrainer:
    """
    Class to handle training of MLNISE models, encapsulating Hogwild logic.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        num_epochs: int = 10,
        learning_rate: float = 0.1,
        device: torch.device = None,
        realizations: int = 1,
        log_fn: Optional[Callable[[str], None]] = None,
        save_checkpoint_fn: Optional[Callable[[int, torch.nn.Module], None]] = None,
        save_interval: int = 5,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[dict] = None
    ):
        self.model = model
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device if device is not None else torch.device("cpu")
        self.realizations = realizations
        self.log_fn = log_fn if log_fn is not None else print
        self.save_checkpoint_fn = save_checkpoint_fn
        self.save_interval = save_interval
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs or {}
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-5
        )
        self.scheduler = None
        if self.scheduler_class:
             self.scheduler = self.scheduler_class(self.optimizer, **self.scheduler_kwargs)

    def _train_one_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        process_num: int = 0,
    ) -> None:
        """
        Performs one epoch of training on a single worker (process).
        """
        # Force single-threaded execution in each process
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        
        self.model.train()
        batch_losses = []
        
        iterable = (
            tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
            if process_num == 0
            else enumerate(train_loader)
        )

        for batch_idx, (inputs, pop_target) in iterable:
             # Move to device
            (h, temperature, E_reorg, tau, total_time, dt_fs, psi0, n_sites) = [
                x.to(self.device).squeeze(0) for x in inputs
            ]
            pop_target = pop_target.to(self.device).squeeze(0).requires_grad_()

            # Prepare MLNISE inputs
            reorg_float = E_reorg.item()
            tau_float = tau.item()
            mlnise_inputs = (
                torch.tensor([reorg_float], device=self.device),
                torch.tensor([tau_float], device=self.device),
            )
            gamma = 1 / tau
            spectral_func = functools.partial(
                spectral_drude, temperature=temperature, strength=reorg_float, gamma=gamma
            )
            spectral_funcs = [spectral_func] * n_sites

            # Run NISE
            with suppress_stdout():
                pop_pred, t_axis = run_nise(
                    h=h,
                    realizations=self.realizations,
                    total_time=total_time.item(),
                    dt=dt_fs.item(),
                    initial_state=psi0,
                    temperature=temperature.item(),
                    spectral_funcs=spectral_funcs,
                    t_correction="MLNISE",
                    mode="Population",
                    device=self.device,
                    mlnise_model=self.model,
                    mlnise_inputs=mlnise_inputs,
                    track_grads=True,
                )
            
            # Loss calculation
            min_len = min(pop_pred.shape[0], pop_target.shape[0])
            loss = F.mse_loss(pop_pred[:min_len], pop_target[:min_len])

            self.optimizer.zero_grad()
            loss.backward()

            # Check for NaNs
            skip_update = False
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                     if torch.isnan(param.grad).any() or torch.isnan(param.data).any():
                          skip_update = True
                          self.log_fn(f"[WARN] NaN in {name} -> skipping update.")
                          break
            
            if not skip_update:
                 clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                 self.optimizer.step()
            
            batch_losses.append(loss.item())

            if batch_idx % 10 == 0 and len(batch_losses) >= 10:
                avg_loss_last_10 = sum(batch_losses[-10:]) / len(batch_losses[-10:])
                if process_num == 0: # Only log from main process usually
                     # self.log_fn(f"Epoch {epoch} | Batch {batch_idx} | Avg Loss: {avg_loss_last_10:.4f}")
                     pass

        avg_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
        self.log_fn(f"Process {process_num} | Epoch {epoch} complete | Avg Loss: {avg_loss:.4f}")

    def train(self, num_processes: int = 4, runname: str = "testrun"):
        """
        Runs the training loop, handling multiprocessing if num_processes > 1.
        """
        self.model.share_memory()
        
        for epoch in range(1, self.num_epochs + 1):
             self.log_fn(f"===== Starting epoch {epoch}/{self.num_epochs} =====")
             self.log_fn(f"Runname: {runname}")
             
             if num_processes > 1:
                  processes = []
                  for rank in range(num_processes):
                       sampler = DistributedSampler(
                            dataset=self.dataset,
                            num_replicas=num_processes,
                            rank=rank,
                            shuffle=True,
                            seed=123 + epoch
                       )
                       train_loader = DataLoader(self.dataset, sampler=sampler, batch_size=1)
                       p = mp.Process(
                            target=self._train_wrapper, 
                            args=(train_loader, epoch, rank)
                       )
                       p.start()
                       processes.append(p)
                  for p in processes:
                       p.join()
             else:
                  sampler = DistributedSampler(
                       self.dataset, num_replicas=1, rank=0, shuffle=True, seed=123
                  )
                  train_loader = DataLoader(self.dataset, sampler=sampler, batch_size=1)
                  self._train_one_epoch(train_loader, epoch, 0)
            
             if self.scheduler:
                  self.scheduler.step()
                  
             if self.save_checkpoint_fn and (epoch % self.save_interval == 0):
                  self.save_checkpoint_fn(epoch, self.model)
                  
        if self.save_checkpoint_fn:
             self.save_checkpoint_fn(self.num_epochs, self.model)
             
        self.log_fn("Training complete.")
        return self.model

    def _train_wrapper(self, train_loader, epoch, rank):
         """Wrapper to be pickled for multiprocessing."""
         self._train_one_epoch(train_loader, epoch, rank)

# Wrapper function for backward compatibility
def train_mlnise_hogwild(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    num_epochs: int = 10,
    num_processes: int = 4,
    learning_rate: float = 0.1,
    runname: str = "testrun",
    device: Optional[torch.device] = None,
    scheduler_class: Optional[type] = None,
    scheduler_kwargs: Optional[dict] = None,
    save_interval: int = 5,
    save_checkpoint_fn: Optional[Callable[[int, torch.nn.Module], None]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    realizations: int = 1,
) -> torch.nn.Module:
    
    trainer = NISETrainer(
        model=model,
        dataset=dataset,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        realizations=realizations,
        log_fn=log_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        save_interval=save_interval,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs
    )
    
    return trainer.train(num_processes=num_processes, runname=runname)
