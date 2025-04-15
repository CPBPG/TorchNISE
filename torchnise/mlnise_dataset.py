"""
mlnise_dataset.py

This module provides a dataset class (MLNiseDrudeDataset) for training
ML-based NISE models using HEOM calculations from PyHEOM. It includes
a runHeomDrude function that demonstrates how to generate time-dependent
populations for random Hamiltonians with Drude spectral density noise.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tqdm
# Attempt to import PyHEOM; if unavailable, dataset generation will fail unless pre-generated data is present.
try:
    import pyheom
    _HAS_PYHEOM = True
except ImportError:
    _HAS_PYHEOM = False


def runHeomDrude(
    n_state: int,
    H: np.ndarray,
    tau: float,
    Temp: float,
    E_reorg: float,
    dt__unit: float,
    initiallyExcitedState: int,
    totalTime: float,
    tier: int,
    matrix_type: str = "sparse"
) -> np.ndarray:
    """
    Generate time-dependent populations via PyHEOM for a system with Drude 
    spectral density. Each site is coupled to a bath with reorganization 
    energy E_reorg and correlation time derived from tau.

    Args:
        n_state (int): Number of states (sites).
        H (np.ndarray): Hamiltonian of shape (..., n_state, n_state).
            Typically just (n_state, n_state) for static usage, or
            (time_steps, n_state, n_state) if time-dependent.
        tau (float): Bath correlation time (fs).
        Temp (float): Temperature (K).
        E_reorg (float): Reorganization energy (cm^-1).
        dt__unit (float): Timestep in fs.
        initiallyExcitedState (int): Index of the initially excited site.
        totalTime (float): Total simulation time in fs.
        tier (int): HEOM hierarchy depth.
        matrix_type (str): PyHEOM matrix type ("sparse", "dense"), etc.

    Returns:
        np.ndarray: 2D array of populations, shape (time_steps, n_state).
    """

    pyheom.units["energy"] = pyheom.unit.wavenumber
    pyheom.units["time"]   = pyheom.unit.femtosecond

    # Approx. relation to Drude damping
    gamma = 53.08 / tau * 100

    # Build site projectors
    Vs = []
    for i in range(n_state):
        V_i = np.zeros((n_state, n_state), dtype=complex)
        V_i[i, i] = 1.0
        Vs.append(V_i)

    # Drude decomposition
    corr_dict = pyheom.noise_decomposition(
        pyheom.drude(2 * E_reorg / gamma, gamma),
        T=Temp,
        type_ltc="PSD",
        n_msd=1,
        n_fsd_rec=2,
        chi_fsd=100.0,
        n_psd=1,
        type_psd="n-1/n"
    )

    # Combine for each site
    noises = []
    for i in range(n_state):
        noises.append(dict(V=Vs[i], **corr_dict))

    # If H is time-dependent (shape > 2 dims), we typically pass H[0] here.
    if H.ndim == 3 and H.shape[0] > 1:
        static_ham = H[0]
    else:
        static_ham = H if H.ndim == 2 else H[0]
    dtype           = np.complex128
    static_ham = static_ham.astype(dtype)
    space           = 'liouville'
    format          = 'dense'
    engine          = 'eigen'
    solver          = 'lsrk4'
    order_liouville = 'row_major'
    # Setup PyHEOM
    h_solver = pyheom.heom_solver(
        static_ham,noises,
        space=space, format=format, engine=engine,
        order_liouville=order_liouville,
        solver=solver,
        engine_args=dict(),
        depth = tier,
        n_inner_threads = 4,
        n_outer_threads = 1
    )
    n_storage = h_solver.storage_size()
    rho = np.zeros((n_storage,n_state,n_state), dtype=dtype)
    rho_0 = rho[0,:,:]
    rho_0[0,0] = 1
    callback_interval = 2
    dt__unit = float(dt__unit)/callback_interval
    
    count = int(totalTime / dt__unit) + 1
    t_list= np.arange(0, count, callback_interval)*dt__unit
    population = np.zeros((len(t_list), n_state), dtype=np.float32)
    solver_params    = dict(
    dt = dt__unit,
    # atol=1e-6, rtol=1e-3
    )   
    
    def callback(t):
        idx = round(t / (dt__unit*pyheom.calc_unit() )/ callback_interval)
        #idx_non_int=t / (dt__unit*pyheom.calc_unit() )/ callback_interval
        #print(idx,idx_non_int,t, rho_0[0,0].real, rho_0[1,1].real)
        
        pop=np.zeros(n_state)
        if idx<len(t_list):
            for i in range(n_state):
                population[idx, i] = rho_0[i,i].real
                pop[i]= rho_0[i,i].real
        
    

    h_solver.solve(rho, t_list, callback, **solver_params)
    h_solver = None
    return population


class MLNiseDrudeDataset(Dataset):
    """
    A Dataset class that uses PyHEOM to generate or load population 
    dynamics for training ML-based NISE. Each sample's Hamiltonian 
    and population are stored on disk or generated on the fly.

    Each item yields:
      inputs: (H, T, E_reorg, tau, total_time, dt, psi0, n_sites)
      target: population array of shape (time_steps, n_sites)

    The shape of H is (time_steps, n_sites, n_sites), 
    and population is (time_steps, n_sites).
    """
    def __init__(
        self,
        length: int = 1000,
        total_time: float = 1000.0,
        dt_fs: float = 1.0,
        n_sites: int = 2,
        seed: int = 42,
        min_energy: float = -500.0,
        max_energy: float = 500.0,
        min_coupling: float = -200.0,
        max_coupling: float = 200.0,
        min_tau: float = 10.0,
        max_tau: float = 200.0,
        min_temp: float = 300.0,
        max_temp: float = 300.0,
        min_reorg: float = 0.0,
        max_reorg: float = 500.0,
        depth: int = 9,
        dataset_folder: str = "GeneratedHeom",
        dataset_name: str = "mlnise_example_dataset",
        nn_coupling: str = "ring",
        generate_if_missing: bool = True,
    ):
        """
        Args:
            length (int): Number of samples (random Hamiltonians) to generate/load.
            total_time (float): HEOM simulation length (fs).
            dt_fs (float): Time step (fs).
            n_sites (int): Number of sites.
            seed (int): RNG seed for reproducible Hamiltonians.
            min_energy, max_energy (float): Range of diagonal energies.
            min_coupling, max_coupling (float): Range of off-diagonal couplings.
            min_tau, max_tau (float): Range of bath correlation times.
            min_temp, max_temp (float): Range of temperature (K).
            min_reorg, max_reorg (float): Range of reorganization energies.
            depth (int): HEOM hierarchy depth.
            dataset_folder (str): Folder for storing/loading .npy data.
            dataset_name (str): Subfolder name for the dataset.
            nn_coupling (str): Not used here, but can be "ring", "star", etc.
            generate_if_missing (bool): If True, generate data if not found.
        """
        super().__init__()
        self.length = length
        self.total_time = total_time
        self.dt_fs = dt_fs
        self.n_sites = n_sites
        self.seed = seed
        self.min_energy = min_energy
        self.max_energy = max_energy
        self.min_coupling = min_coupling
        self.max_coupling = max_coupling
        self.min_tau = min_tau
        self.max_tau = max_tau
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.min_reorg = min_reorg
        self.max_reorg = max_reorg
        self.depth = depth
        self.nn_coupling = nn_coupling

        self.save_path = os.path.join(dataset_folder, dataset_name)
        os.makedirs(self.save_path, exist_ok=True)

        # Data file paths
        self.h_file = os.path.join(self.save_path, "H.npy")
        self.pop_file = os.path.join(self.save_path, "Pop.npy")
        self.temp_file = os.path.join(self.save_path, "T.npy")
        self.reorg_file = os.path.join(self.save_path, "Ereorg.npy")
        self.tau_file = os.path.join(self.save_path, "tau.npy")

        # Load or generate
        if self._check_files():
            self._load_data()
        else:
            if not generate_if_missing:
                raise RuntimeError(
                    f"No dataset found in {self.save_path}, and generate_if_missing=False."
                )
            if not _HAS_PYHEOM:
                raise ImportError(
                    "PyHEOM is not installed. Cannot generate dataset. "
                    "Install pyheom or provide pre-generated data."
                )
            self._generate_data()
            self._save_data()

    def _check_files(self) -> bool:
        """Check if all required .npy files exist."""
        return all(os.path.exists(f) for f in [
            self.h_file, self.pop_file, self.temp_file,
            self.reorg_file, self.tau_file
        ])

    def _load_data(self) -> None:
        """Load dataset from existing .npy files."""
        self.resultH = np.load(self.h_file)
        self.resultPop = np.load(self.pop_file)
        self.resultTemp = np.load(self.temp_file)
        self.resultEReorg = np.load(self.reorg_file)
        self.resultTau = np.load(self.tau_file)
        self.length = self.resultH.shape[0]
        self.total_time = self.resultPop.shape[1]

    def _save_data(self) -> None:
        """Save dataset to .npy files."""
        np.save(self.h_file, self.resultH)
        np.save(self.pop_file, self.resultPop)
        np.save(self.temp_file, self.resultTemp)
        np.save(self.reorg_file, self.resultEReorg)
        np.save(self.tau_file, self.resultTau)

    def _generate_data(self) -> None:
        """
        Generate random Hamiltonians, run PyHEOM to obtain population dynamics,
        and store them in memory.
        """
        import random
        random.seed(self.seed)
        np.random.seed(self.seed)

        time_steps = int(self.total_time / self.dt_fs) + 1
        self.resultH = np.zeros(
            (self.length,self.n_sites, self.n_sites),
            dtype=np.float32
        )
        self.resultPop = np.zeros(
            (self.length, time_steps, self.n_sites),
            dtype=np.float32
        )
        self.resultTemp = np.zeros((self.length,), dtype=np.float32)
        self.resultEReorg = np.zeros((self.length,), dtype=np.float32)
        self.resultTau = np.zeros((self.length,), dtype=np.float32)

        for idx in tqdm.tqdm(range(self.length)):
            tau_sample = np.random.uniform(self.min_tau, self.max_tau)
            temp_sample = np.random.uniform(self.min_temp, self.max_temp)
            reorg_sample = np.random.uniform(self.min_reorg, self.max_reorg)

            # Build a random Hamiltonian repeated across time steps
            ham = self._build_random_hamiltonian()

            # Run PyHEOM -> populations
            population = runHeomDrude(
                n_state=self.n_sites,
                H=ham,
                tau=tau_sample,
                Temp=temp_sample,
                E_reorg=reorg_sample,
                dt__unit=self.dt_fs,
                initiallyExcitedState=0,
                totalTime=self.total_time,
                tier=self.depth
            )

            self.resultH[idx] = ham
            self.resultPop[idx] = population
            self.resultTemp[idx] = temp_sample
            self.resultEReorg[idx] = reorg_sample
            self.resultTau[idx] = tau_sample

    def _build_random_hamiltonian(self) -> np.ndarray:
        """
        Create a symmetric random Hamiltonian with diagonal energies in 
        [min_energy, max_energy] and off-diagonal couplings in 
        [min_coupling, max_coupling]. Replicate for each time step.

        Returns:
            np.ndarray of shape (n_sites, n_sites).
        """
        time_steps = int(self.total_time // self.dt_fs) + 1

        # Single random Hamiltonian
        static_ham = np.zeros((self.n_sites, self.n_sites), dtype=np.float32)
        for i in range(self.n_sites):
            static_ham[i, i] = np.random.uniform(self.min_energy, self.max_energy)

        for i in range(self.n_sites):
            for j in range(i + 1, self.n_sites):
                coupling = np.random.uniform(self.min_coupling, self.max_coupling)
                static_ham[i, j] = coupling
                static_ham[j, i] = coupling



        return static_ham

    def __len__(self) -> int:
        """Number of Hamiltonians/samples in the dataset."""
        return self.length

    def __getitem__(self, idx: int):
        """
        Returns:
            inputs: (H, T, E_reorg, tau, total_time, dt, psi0, n_sites)
            pop: shape (time_steps, n_sites)
        """
        psi0 = np.zeros((self.n_sites,), dtype=np.float32)
        psi0[0] = 1.0

        h_torch = torch.tensor(self.resultH[idx], dtype=torch.float32)
        pop_torch = torch.tensor(self.resultPop[idx], dtype=torch.float32)
        T_torch = torch.tensor([self.resultTemp[idx]], dtype=torch.float32)
        reorg_torch = torch.tensor([self.resultEReorg[idx]], dtype=torch.float32)
        tau_torch = torch.tensor([self.resultTau[idx]], dtype=torch.float32)
        dt_torch = torch.tensor([self.dt_fs], dtype=torch.float32)
        ttime_torch = torch.tensor([self.total_time], dtype=torch.float32)
        psi0_torch = torch.tensor(psi0, dtype=torch.float32)
        n_sites_torch = torch.tensor([self.n_sites], dtype=torch.int32)

        inputs = (
            h_torch,      # (n_sites, n_sites)
            T_torch,      # (1,)
            reorg_torch,  # (1,)
            tau_torch,    # (1,)
            ttime_torch,  # (1,)
            dt_torch,     # (1,)
            psi0_torch,   # (n_sites,)
            n_sites_torch
        )
        return inputs, pop_torch
