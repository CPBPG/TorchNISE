import os
import struct
import warnings
from dataclasses import dataclass, replace, field
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch

from torchnise import units
from torchnise.nise import NISEParameters, run_nise
# from torchnise.spectroscopy_2dir import calculate_2dir_vector


@dataclass
class NISE2017Config:
    """Configuration for NISE 2017 simulations.

    Attributes:
        technique: The simulation technique (e.g., "Population", "2DES").
        propagation: The propagation method (e.g., "Sparse", "RK4").
        coupling_cut: Cutoff for coupling interactions.
        threshold: Threshold value for calculations.
        hamiltonian_file: Path to the binary Hamiltonian file.
        dipole_file: Path to the binary dipole file.
        anharmonic_file: Path to the anharmonicity file.
        anharmonicity: Anharmonicity value.
        overtone_dipole_file: Path to the overtone dipole file.
        position_file: Path to the position trajectory file.
        coupling_file: Path to the coupling file.
        length: Total number of steps in the binary files (Time * Realizations).
        sample_rate: Stride between realizations in the binary file.
        lifetime: Lifetime parameter for damping (fs).
        fft_size: Size of the FFT window.
        t_max_1: Maximum time steps for the first interval (coherence/pop).
        t_max_2: Maximum time steps for the second interval (waiting time).
        t_max_3: Maximum time steps for the third interval (detection).
        begin_point: Starting realization index.
        end_point: Ending realization index.
        singles: Number of single-excitation sites.
        doubles: Number of double-excitation states.
        basis: Basis set used (e.g., "Local").
        temperature: Temperature in Kelvin.
        min_freq_1: Minimum frequency for axis 1 (cm^-1).
        min_freq_2: Minimum frequency for axis 2 (cm^-1).
        min_freq_3: Minimum frequency for axis 3 (cm^-1).
        max_freq_1: Maximum frequency for axis 1 (cm^-1).
        max_freq_2: Maximum frequency for axis 2 (cm^-1).
        max_freq_3: Maximum frequency for axis 3 (cm^-1).
        homogen_damping: Homogeneous damping time (fs).
        inhomogen_damping: Inhomogeneous damping time (fs).
        timestep: Time step size (fs).
        initial_state_site: Index of the site initially excited (0-indexed internally).
        initial_state_site: int = 0
    projection_sites: List of sites for projection.
    save_popf: bool = False
    rotating_wave_freq: float = 0.0

    """
    technique: str = ""
    hamiltonian_type: str = "Normal"
    propagation: str = "Sparse"
    coupling_cut: float = 0.0
    threshold: float = 0.0
    hamiltonian_file: str = ""
    dipole_file: str = ""
    alpha_file: str = "" # Raman tensor file
    anharmonic_file: str = ""
    anharmonicity: float = 0.0
    overtone_dipole_file: str = ""
    position_file: str = ""
    coupling_file: str = ""
    length: int = 0
    sample_rate: int = 1
    lifetime: float = 0.0
    fft_size: int = 0
    t_max_1: int = 0
    t_max_2: int = 0
    t_max_3: int = 0
    begin_point: int = 0
    end_point: int = 1
    singles: int = 0
    doubles: int = 0
    basis: str = "Local"
    temperature: float = 300.0
    min_freq_1: float = 0.0
    min_freq_2: float = 0.0
    min_freq_3: float = 0.0
    max_freq_1: float = 0.0
    max_freq_2: float = 0.0
    max_freq_3: float = 0.0
    homogen_damping: float = 0.0
    inhomogen_damping: float = 0.0
    timestep: float = 0.0
    initial_state_site: int = 0
    projection_sites: List[int] = field(default_factory=list)
    rotating_wave_freq: float = 0.0


class NISEInputParser:
    """Parses NISE 2017 input files into a NISE2017Config object."""

    def __init__(self, file_path: str):
        """Initializes the parser.

        Args:
            file_path: Absolute path to the input file.
        """
        self.file_path = file_path
        self.config = NISE2017Config()
        self.base_dir = os.path.dirname(file_path)
        self.begin_point_set = False
        self.end_point_set = False

    def parse(self) -> NISE2017Config:
        """Parses the input file and returns the configuration.

        Returns:
            A NISE2017Config object populated with values from the file.
        """
        with open(self.file_path, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('#'):
                i += 1
                continue

            parts = line.split()
            if not parts:
                i += 1
                continue

            keyword = parts[0]

            if keyword == "Projection":
                i = self._handle_projection(lines, i)
            else:
                self._handle_keyword(keyword, parts)
                i += 1

        if not self.end_point_set and self.config.length > 0 and self.config.sample_rate > 0:
            # Match calc_2DIR.c: (length - tmax1 - tmax2 - tmax3 - 1) / sample + 1
            available_length = self.config.length - self.config.t_max_1 - self.config.t_max_2 - self.config.t_max_3 - 1
            if available_length > 0:
                self.config.end_point = available_length // self.config.sample_rate + 1
            else:
                self.config.end_point = 1
            print(f"Set EndPoint to {self.config.end_point} (autodetected)")

        if self.config.rotating_wave_freq == 0.0:
            self.config.rotating_wave_freq = (self.config.min_freq_1 + self.config.max_freq_1) / 2.0
            if self.config.rotating_wave_freq != 0:
                print(f"Set default rotating_wave_freq to {self.config.rotating_wave_freq} (autodetected)")

        return self.config

    def _handle_keyword(self, keyword: str, parts: List[str]):
        """Handles parsing of standard key-value pairs."""
        if keyword == "Technique":
            self.config.technique = parts[1]
        elif keyword == "HamiltonianType":
            self.config.hamiltonian_type = parts[1]
        elif keyword == "Propagation":
            self.config.propagation = parts[1]
        elif keyword == "Couplingcut":
            self.config.coupling_cut = float(parts[1])
        elif keyword == "Threshold":
            self.config.threshold = float(parts[1])
        elif keyword == "Hamiltonianfile":
            self.config.hamiltonian_file = os.path.join(self.base_dir, parts[1])
        elif keyword == "Dipolefile":
            self.config.dipole_file = os.path.join(self.base_dir, parts[1])
        elif keyword == "Alphafile":
            self.config.alpha_file = os.path.join(self.base_dir, parts[1])
        elif keyword == "Anharmonicfile":
            self.config.anharmonic_file = os.path.join(self.base_dir, parts[1])
        elif keyword == "Anharmonicity":
            self.config.anharmonicity = float(parts[1])
        elif keyword == "Overtonedipolefile":
            self.config.overtone_dipole_file = os.path.join(self.base_dir, parts[1])
        elif keyword == "Positionfile":
            self.config.position_file = os.path.join(self.base_dir, parts[1])
        elif keyword == "Couplingfile":
            self.config.coupling_file = os.path.join(self.base_dir, parts[1])
        elif keyword == "Length":
            self.config.length = int(parts[1])
        elif keyword == "Samplerate":
            self.config.sample_rate = int(parts[1])
        elif keyword == "Lifetime":
            self.config.lifetime = float(parts[1])
        elif keyword == "Homogen":
            self.config.homogen_damping = float(parts[1])
        elif keyword == "Inhomogen":
            self.config.inhomogen_damping = float(parts[1])
        elif keyword == "Timestep":
            self.config.timestep = float(parts[1])
        elif keyword == "FFT":
            self.config.fft_size = int(parts[1])
        elif keyword == "RunTimes":
            self.config.t_max_1 = int(parts[1])
            if len(parts) > 2:
                self.config.t_max_2 = int(parts[2])
            if len(parts) > 3:
                self.config.t_max_3 = int(parts[3])
        elif keyword == "MinFrequencies":
            self.config.min_freq_1 = float(parts[1])
            if len(parts) > 2:
                self.config.min_freq_2 = float(parts[2])
            if len(parts) > 3:
                self.config.min_freq_3 = float(parts[3])
        elif keyword == "MaxFrequencies":
            self.config.max_freq_1 = float(parts[1])
            if len(parts) > 2:
                self.config.max_freq_2 = float(parts[2])
            if len(parts) > 3:
                self.config.max_freq_3 = float(parts[3])
        elif keyword in ["BeginPoint", "BeginSample"]:
            self.config.begin_point = int(parts[1])
            self.begin_point_set = True
        elif keyword == "EndPoint":
            self.config.end_point = int(parts[1])
            self.end_point_set = True
        elif keyword == "Singles":
            self.config.singles = int(parts[1])
        elif keyword == "Doubles":
            self.config.doubles = int(parts[1])
        elif keyword == "Basis":
            self.config.basis = parts[1]
        elif keyword == "Temperature":
            self.config.temperature = float(parts[1])
        elif keyword == "InitialState":
            self.config.initial_state_site = int(parts[1]) - 1
        elif keyword in ["RotatingWaveFreq", "RotatingWaveFrequency"]:
            self.config.rotating_wave_freq = float(parts[1])

    def _handle_projection(self, lines: List[str], current_index: int) -> int:
        """Handles the multi-line Projection block."""
        i = current_index + 1
        if i < len(lines):
            proj_line = lines[i].strip()
            if proj_line.startswith("Sites"):
                num_proj = int(proj_line.split()[1])
                proj_sites = []
                while len(proj_sites) < num_proj and i + 1 < len(lines):
                    i += 1
                    proj_sites.extend([int(x) for x in lines[i].split()])
                self.config.projection_sites = proj_sites
        return i + 1

class NISEBinaryLoader:
    """Handles reading of NISE 2017 binary files."""

    def __init__(self, config: NISE2017Config,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.config = config
        self.device = device



    def _load_hamiltonian_normal(
        self, steps=None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Loads Hamiltonian in Normal mode (all in one file)."""
        file_path = self.config.hamiltonian_file
        n_sites = self.config.singles
        n_doubles = self.config.doubles
        begin = self.config.begin_point
        end_point = self.config.end_point
        sample_rate = self.config.sample_rate
        t_max = steps if steps is not None else self.config.t_max_1

        num_realizations = end_point - begin

        if self.config.length > 0:
            # Logic: Last step read is (begin + num_realizations - 1) * sample_rate + t_max
            # This must be <= length
            available_for_samples = self.config.length - t_max - begin * sample_rate
            if available_for_samples < 0:
                max_realizations = 0
            else:
                max_realizations = (available_for_samples // sample_rate) + 1

            if max_realizations < num_realizations:
                if max_realizations <= 0:
                    raise ValueError(
                        f"No realizations fit in Hamiltonian file with Length={self.config.length} and t_max={t_max}"
                    )
                print(
                    f"Truncating realizations from {num_realizations} to {max_realizations} due to file length."
                )
                num_realizations = max_realizations
                self.config.end_point = begin + num_realizations

        n_tri_singles = n_sites * (n_sites + 1) // 2
        n_tri_doubles = n_doubles * (n_doubles + 1) // 2

        # Stride in floats: 1 (time) + n_tri_singles + n_tri_doubles
        stride_floats = 1 + n_tri_singles + n_tri_doubles

        h_full = torch.zeros(
            (t_max, num_realizations, n_sites, n_sites), dtype=torch.float32,device=self.device
        )
        h_doubles = (
            torch.zeros(
                (t_max, num_realizations, n_doubles, n_doubles), dtype=torch.float32,device=self.device
            )
            if n_doubles > 0
            else None
        )

        iu_s = np.triu_indices(n_sites)
        iu_d = np.triu_indices(n_doubles) if n_doubles > 0 else None

        with open(file_path, "rb") as f:
            if sample_rate == t_max:
                offset_steps = begin * sample_rate
                f.seek(offset_steps * stride_floats * 4)

                total_steps_to_read = num_realizations * t_max
                count = total_steps_to_read * stride_floats

                data = np.fromfile(f, dtype=np.float32, count=count)
                if len(data) < count:
                    raise EOFError(f"Unexpected EOF in {file_path}")

                # Reshape to (num_realizations, t_max, stride_floats)
                data = data.reshape((num_realizations, t_max, stride_floats))

                # Singles
                h_tri_s = data[:, :, 1 : 1 + n_tri_singles]  # (R, T, n_tri_s)

                # We need to fill h_full (T, R, N, N)
                h_full_rt = torch.zeros(
                    (num_realizations, t_max, n_sites, n_sites), dtype=torch.float32,device=self.device
                )
                tens_tri_s = torch.from_numpy(h_tri_s).to(self.device)
                h_full_rt[:, :, iu_s[0], iu_s[1]] = tens_tri_s
                h_full_rt[:, :, iu_s[1], iu_s[0]] = tens_tri_s

                h_full = h_full_rt.permute(1, 0, 2, 3).contiguous()

                if n_doubles > 0:
                    h_tri_d = data[:, :, 1 + n_tri_singles : stride_floats]
                    h_doubles_rt = torch.zeros(
                        (num_realizations, t_max, n_doubles, n_doubles),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    tens_tri_d = torch.from_numpy(h_tri_d).to(self.device)
                    h_doubles_rt[:, :, iu_d[0], iu_d[1]] = tens_tri_d
                    h_doubles_rt[:, :, iu_d[1], iu_d[0]] = tens_tri_d
                    h_doubles = h_doubles_rt.permute(1, 0, 2, 3).contiguous()

            else:
                for r in range(num_realizations):
                    base_step = (begin + r) * sample_rate
                    f.seek(base_step * stride_floats * 4)
                    count = t_max * stride_floats
                    data = np.fromfile(f, dtype=np.float32, count=count)
                    if len(data) < count:
                        raise EOFError(
                            f"Unexpected EOF in {file_path} (Realization {r})"
                        )

                    data = data.reshape((t_max, stride_floats))
                    h_tri_s = data[:, 1 : 1 + n_tri_singles]
                    h_full[:, r, iu_s[0], iu_s[1]] = torch.from_numpy(h_tri_s).to(self.device)
                    h_full[:, r, iu_s[1], iu_s[0]] = torch.from_numpy(h_tri_s).to(self.device)

                    if n_doubles > 0:
                        h_tri_d = data[:, 1 + n_tri_singles : stride_floats]
                        h_doubles[:, r, iu_d[0], iu_d[1]] = torch.from_numpy(h_tri_d).to(self.device)
                        h_doubles[:, r, iu_d[1], iu_d[0]] = torch.from_numpy(h_tri_d).to(self.device)

        return h_full, None, h_doubles

    def _load_hamiltonian_coupling(self, steps=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Loads Hamiltonian in Coupling mode (split files).
        Returns (h_static, energies_noise, None).
        h_static: (N, N)
        energies_noise: (steps, realizations, n_sites)
        """
        ham_file = self.config.hamiltonian_file
        coup_file = self.config.coupling_file
        n_sites = self.config.singles
        begin = self.config.begin_point
        end_point = self.config.end_point
        sample_rate = self.config.sample_rate
        t_max = steps if steps is not None else self.config.t_max_1

        num_realizations = end_point - begin

        # 1. Load Static Coupling (Full triangular matrix)
        n_tri = n_sites * (n_sites + 1) // 2
        with open(coup_file, "rb") as f:
            coup_data = np.fromfile(f, dtype=np.float32, count=n_tri)
        
        if len(coup_data) < n_tri:
             raise EOFError(f"Coupling file too short: {coup_file}")

        iu = np.triu_indices(n_sites)
        h_static = torch.zeros((n_sites, n_sites), dtype=torch.float32,device=self.device)
        coup_tensor = torch.from_numpy(coup_data).to(self.device)
        h_static[iu[0], iu[1]] = coup_tensor
        h_static[iu[1], iu[0]] = coup_tensor

        # 2. Load Dynamic Site Energies
        stride = 1 + n_sites
        energies = torch.zeros((t_max, num_realizations, n_sites), dtype=torch.float32,device=self.device)

        with open(ham_file, "rb") as f:
            if sample_rate == t_max:
                offset = begin * sample_rate * stride * 4
                f.seek(offset)
                total_floats = num_realizations * t_max * stride
                data = np.fromfile(f, dtype=np.float32, count=total_floats)
                
                # Reshape safe
                actual_len = len(data) // stride
                data = data[:actual_len*stride].reshape((actual_len // t_max, t_max, stride))
                # energies: [R, T, N]
                e_vals = torch.from_numpy(data[:, :, 1:])
                # Permute to [T, R, N] for site_noise format
                energies = e_vals.permute(1, 0, 2).to(self.device)
            else:
                 for r in range(num_realizations):
                    base = (begin + r) * sample_rate * stride * 4
                    f.seek(base)
                    count = t_max * stride
                    data = np.fromfile(f, dtype=np.float32, count=count)
                    if len(data) < count: 
                         break # Handle EOF
                    data = data.reshape((t_max, stride))
                    energies[:, r, :] = torch.from_numpy(data[:, 1:]).to(self.device)

        # Return static H and noise
        return h_static, energies, None

    def load_hamiltonian(self, steps: Optional[int] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Loads Hamiltonian trajectory.

        Args:
            steps: Number of time steps to load. Defaults to config.t_max_1.

        Returns:
            A tuple containing:
                - Singles Hamiltonian tensor (t_max, realizations, n_sites, n_sites) OR (n_sites, n_sites) if coupling type is Coupling.
                - Site Noise (realizations, t_max, n_sites) OR None if Normal.
                - Doubles Hamiltonian tensor (if exists) or None.
        """
        if self.config.hamiltonian_type == "Coupling":
            h, noise, h_doubles = self._load_hamiltonian_coupling(steps)
        else:
            h, noise, h_doubles = self._load_hamiltonian_normal(steps)

        # Apply Rotating Wave Frequency Shift
        if self.config.rotating_wave_freq != 0:
            print(f"Applying RWA Shift: {self.config.rotating_wave_freq} cm-1 to loaded Hamiltonian")
            if h.ndim == 2: # Static (N, N)
                h = h - torch.eye(self.config.singles, device=self.device) * self.config.rotating_wave_freq
            elif h.ndim == 4: # Trajectory (T, R, N, N)
                indices = torch.arange(self.config.singles, device=self.device)
                h[:, :, indices, indices] -= self.config.rotating_wave_freq
        
        return h, noise, h_doubles

    def load_dipole(self, steps=None) -> torch.Tensor:
        file_path = self.config.dipole_file
        n_sites = self.config.singles
        begin = self.config.begin_point
        end_point = self.config.end_point
        sample_rate = self.config.sample_rate
        t_max = steps if steps is not None else self.config.t_max_1

        num_realizations = end_point - begin
        stride_floats = 1 + 3 * n_sites

        mu = torch.zeros((t_max, num_realizations, n_sites, 3), dtype=torch.float32, device=self.device)

        with open(file_path, "rb") as f:
            if sample_rate == t_max:
                offset_steps = begin * sample_rate
                f.seek(offset_steps * stride_floats * 4)

                total_steps = num_realizations * t_max
                count = total_steps * stride_floats
                data = np.fromfile(f, dtype=np.float32, count=count)

                if len(data) < count:
                    pass

                actual_steps = len(data) // stride_floats
                data = data[: actual_steps * stride_floats]

                # Depending on how nouse/truncation works, we might have partial realizations.
                # Assuming valid file structure for performance.

                data = data.reshape((num_realizations, t_max, stride_floats))
                vals = data[:, :, 1:]  # (R, T, 3N)
                vals = torch.from_numpy(vals).to(self.device)

                mu_rt = torch.zeros(
                    (num_realizations, t_max, n_sites, 3), dtype=torch.float32, device=self.device
                )
                mu_rt[:, :, :, 0] = vals[:, :, 0:n_sites]
                mu_rt[:, :, :, 1] = vals[:, :, n_sites : 2 * n_sites]
                mu_rt[:, :, :, 2] = vals[:, :, 2 * n_sites : 3 * n_sites]

                mu = mu_rt.permute(1, 0, 2, 3).contiguous()

            else:
                for r in range(num_realizations):
                    base_step = (begin + r) * sample_rate
                    f.seek(base_step * stride_floats * 4)

                    count = t_max * stride_floats
                    data = np.fromfile(f, dtype=np.float32, count=count)
                    if len(data) < count:
                        break

                    data = data.reshape((t_max, stride_floats))
                    mu[:, r, :, 0] = torch.from_numpy(data[:, 1 : 1 + n_sites]).to(self.device)
                    mu[:, r, :, 1] = torch.from_numpy(
                        data[:, 1 + n_sites : 1 + 2 * n_sites]
                    ).to(self.device)
                    mu[:, r, :, 2] = torch.from_numpy(
                        data[:, 1 + 2 * n_sites : 1 + 3 * n_sites]
                    ).to(self.device)
        return mu


    def load_positions(self) -> torch.Tensor:
        file_path = self.config.position_file
        n_sites = self.config.singles
        length = self.config.length
        
        with open(file_path, 'rb') as f:
            box_size_data = f.read(4)
            if not box_size_data:
                 return torch.zeros((length, n_sites, 3)), 0.0
            box_size = struct.unpack('f', box_size_data)[0]
            
            count = length * n_sites * 3
            data = np.fromfile(f, dtype=np.float32, count=count)
            
            if len(data) < count:
                actual_len = len(data) // (n_sites * 3)
                data = data[:actual_len * n_sites * 3]
                length = actual_len 
            
            data = data.reshape((length, 3 * n_sites))
            
            pos = torch.zeros((length, n_sites, 3), dtype=torch.float32)
            
            pos[:, :, 0] = torch.from_numpy(data[:, 0:n_sites])
            pos[:, :, 1] = torch.from_numpy(data[:, n_sites:2*n_sites])
            pos[:, :, 2] = torch.from_numpy(data[:, 2*n_sites:3*n_sites])
            
        return pos, box_size

    def load_alpha(self, steps=None) -> torch.Tensor:
        """Loads Raman polarizability tensor (Alpha)."""
        file_path = self.config.alpha_file
        n_sites = self.config.singles
        begin = self.config.begin_point
        end_point = self.config.end_point
        sample_rate = self.config.sample_rate
        t_max = steps if steps is not None else self.config.t_max_1
        
        num_realizations = end_point - begin
        # Alpha file assumes 6 components per site?
        # Check raman.c: read_alpha reads 6 components.
        # Format usually: Time, site1_xx, site1_xy, ..., siteN_zz ?
        # Actually NISE read_alpha reads one component 'x' at a time?
        # No, read_alpha reads all components for a site?
        # Let's assume standard NISE format: Time, 6*N floats.
        stride_floats = 1 + 6 * n_sites
        
        alpha = torch.zeros((t_max, num_realizations, n_sites, 6), dtype=torch.float32)
        
        with open(file_path, "rb") as f:
             if sample_rate == t_max:
                offset_steps = begin * sample_rate
                f.seek(offset_steps * stride_floats * 4)
                count = num_realizations * t_max * stride_floats
                data = np.fromfile(f, dtype=np.float32, count=count)
                
                if len(data) < count: pass
                
                data = data.reshape((num_realizations, t_max, stride_floats))
                vals = data[:, :, 1:] # (R, T, 6N)
                vals = torch.from_numpy(vals)
                
                # Reshape to (R, T, N, 6)
                # Order: xx, xy, xz, yy, yz, zz
                vals = vals.reshape(num_realizations, t_max, n_sites, 6)
                alpha = vals.permute(1, 0, 2, 3).contiguous()
                
             else:
                for r in range(num_realizations):
                    base_step = (begin + r) * sample_rate
                    f.seek(base_step * stride_floats * 4)
                    count = t_max * stride_floats
                    data = np.fromfile(f, dtype=np.float32, count=count)
                    
                    data = data.reshape((t_max, stride_floats))
                    vals = torch.from_numpy(data[:, 1:]).reshape(t_max, n_sites, 6)
                    alpha[:, r, :, :] = vals
                    
        return alpha

    def load_hamiltonian_diagonal(self) -> np.ndarray:
        """Loads only the diagonal of the Hamiltonian for the full trajectory."""
        file_path = self.config.hamiltonian_file
        n_sites = self.config.singles
        n_doubles = self.config.doubles
        length = self.config.length
        
        n_tri_singles = n_sites * (n_sites + 1) // 2
        n_tri_doubles = n_doubles * (n_doubles + 1) // 2
        stride_floats = 1 + n_tri_singles + n_tri_doubles
        
        freq_traj = np.zeros((length, n_sites))
        with open(file_path, 'rb') as f:
            # We can read the whole file diagonal in one go if we use memory mapping or smart slicing
            # For simplicity and speed without mapping complexity: bulk read then slice
            count = length * stride_floats
            data = np.fromfile(f, dtype=np.float32, count=count)
            if len(data) < count:
                # Handle truncated files
                actual_length = len(data) // stride_floats
                data = data[:actual_length * stride_floats]
                freq_traj = freq_traj[:actual_length]
                length = actual_length
            
            data = data.reshape((length, stride_floats))
            # Diagonal is the first n_sites elements after time
            freq_traj = data[:, 1 : 1 + n_sites]
            
        return freq_traj


def save_nise2017_output(u, total_time, dt):
    """
    Saves output in NISE 2017 format (Pop.dat and PopF.dat).
    u: Time evolution operator tensor of shape (realizations, steps, n_sites, n_sites)
    """
    # u[r, t, b, a] is the amplitude <b|U(t)|a>
    # Probabilities: |u|^2
    probs = (torch.abs(u)**2).detach() # (realizations, steps, n_sites, n_sites)
    
    # Average over realizations
    probs_avg = torch.mean(probs, dim=0).cpu().numpy() # (steps, n_sites, n_sites)
    
    n_steps = probs_avg.shape[0]
    n_sites = probs_avg.shape[1]
    
    time_axis = np.linspace(0, total_time, n_steps)
    
    # Pop.dat: Average survival probability
    # NISE 2017 calculates Pop[t] = sum_a P(a->a) / N
    pop_diagonal = np.diagonal(probs_avg, axis1=1, axis2=2) # (steps, n_sites)
    pop_avg_survival = np.mean(pop_diagonal, axis=1) # (steps,)
    
    pop_data = np.column_stack((time_axis, pop_avg_survival))
    np.savetxt("Pop.dat", pop_data, fmt="%.6f %.6e")
            
    # PopF.dat: All site-to-site populations
    # NISE 2017 order: time, then for a in singles: for b in singles: P(b -> a)
    # We want probs_avg[t, b, a] where a is initial, b is final.
    # NISE loops a (initial) then b (final).
    # So we want (initial, final) ordering.
    # probs_avg is (steps, final, initial).
    # Transpose to (steps, initial, final).
    probs_avg_transposed = np.transpose(probs_avg, (0, 2, 1))
    
    popf_flattened = probs_avg_transposed.reshape(n_steps, -1)
    
    # Combine with time axis
    popf_data = np.column_stack((time_axis, popf_flattened))
    
    # NISE 2017 uses fixed precision.
    fmt = ["%.6f"] + ["%.6e"] * (n_sites * n_sites)
    np.savetxt("PopF.dat", popf_data, fmt=fmt)

def save_nise2017_output_pop_only(population, total_time, dt):
    """
    Saves only Pop.dat (Best effort) without PopF.
    population: (steps, n_sites) = P(site | initial_state).
    """
    # population from propagator is typically (Realizations, Steps, Sites)
    if population.ndim == 3:
        # Average over realizations
        population = np.mean(population, axis=0)
        
    n_steps = population.shape[0]
    time_axis = np.linspace(0, total_time, n_steps)
    
    # Pop.dat: Average Survival Logic.
    # We only have P(i -> a). We CANNOT compute sum_a P(a->a) (Trace).
    # We can only save sum_a P(i->a) which should be 1.0 (conservation).
    # Or we can save P(i->i)?
    # To avoid confusion with standard NISE Output, we will save the Total Population (Trace check).
    
    pop_total = np.sum(population, axis=1)
    
    pop_data = np.column_stack((time_axis, pop_total))
    np.savetxt("Pop.dat", pop_data, fmt="%.6f %.6e")



def save_nise2017_absorption(avg_s1, config: NISE2017Config):
    """Saves Absorption.dat and TD_Absorption.dat in NISE 2017 format."""
    total_time = (config.t_max_1 - 1) * config.timestep
    n_steps = config.t_max_1
    time_axis = np.linspace(0, total_time, n_steps)

    # 1. Save TD_Absorption.dat
    # format: time Re(S1) Im(S1)
    # NISE 2017 avg s1 is normalized by realizations.
    with open("TD_Absorption.dat", "w") as f:
        for t in range(n_steps):
            f.write(
                f"{time_axis[t]:.6f} {avg_s1[t].real:.6e} {avg_s1[t].imag:.6e}\n"
            )

    # 2. Fourier Transform for Absorption.dat
    # Apply damping if specified
    damp = torch.ones(n_steps, dtype=torch.complex64)
    # NISE 2017 1DFFT.c: exp(-i*non->deltat/(2*non->lifetime))
    # Note: 2*lifetime? Wait. Yes, it seems they use it for E-field damping which is half of population damping.
    # Actually TorchNISE absorption damping is also configurable.
    for i in range(n_steps):
        t_val = i * config.timestep
        if config.lifetime > 0:
            damp[i] *= np.exp(-t_val / (2 * config.lifetime))
        if config.homogen_damping > 0:
            damp[i] *= np.exp(-t_val / (2 * config.homogen_damping))
        if config.inhomogen_damping > 0:
            damp[i] *= np.exp(-(t_val**2) / (2 * config.inhomogen_damping**2))

    s1_damped = avg_s1 * damp

    # FFT Padding
    fft_len = config.fft_size if config.fft_size >= n_steps else n_steps
    # NISE 2017 Scale first point (trapezoidal rule)
    s1_padded = torch.zeros(fft_len, dtype=torch.complex64)
    s1_padded[:n_steps] = s1_damped
    s1_padded[0] *= 0.5

    # NISE 2017 does: fftIn[i][0] = Im, fftIn[i][1] = Re.
    # Spectrum = fftOut[i][1].
    # This corresponds to Re(FT(s1)).
    # We can just use np.fft.fft
    spec_f = np.fft.fft(s1_padded.numpy())

    # Frequency axis
    # NISE 2017: -((fft-i)/non->deltat/c_v/fft-shift1)
    # c_v = 2.99792458e-5 cm/fs
    c_v_fs = 2.99792458e-5  # cm/fs
    freq_indices = np.arange(fft_len)
    # NISE 2017 1DFFT logic for freq mapping:
    shift1 = (config.min_freq_1 + config.max_freq_1) / 2.0

    # Replicate the loop from lines 201-220 in 1DFFT.c
    freqs = []
    vals_re = []
    vals_im = []

    for i in range(fft_len):
        if i >= fft_len // 2:
            f_val = -(
                (fft_len - i) / (config.timestep * c_v_fs * fft_len) - shift1
            )
        else:
            f_val = -((-i) / (config.timestep * c_v_fs * fft_len) - shift1)

        if config.min_freq_1 <= f_val <= config.max_freq_1:
            # We want Re(FT(s1)) logic.
            # fftOut[i][1] in NISE corresponds to Re(FFT(Im + iRe)) = Im * sin + Re * cos = Re(s1 * e^-iwt)
            # Standard FFT in numpy is sum s1 * e^-i2pi kn/N
            # If we want Re(s1 * e^-iwt), we take the real part of the numpy FFT output?
            # Wait, numpy FFT(s) = sum s(n) exp(-i 2pi k n / N)
            # Re(sum s exp(-iwt)) = sum (Re s cos + Im s sin).
            # Yes, it's just the real part of numpy FFT.
            freqs.append(f_val)
            vals_re.append(spec_f[i].real)
            vals_im.append(spec_f[i].imag)

    # Sort by frequency
    idx = np.argsort(freqs)
    with open("Absorption.dat", "w") as f:
        for i in idx:
            f.write(f"{freqs[i]:.6f} {vals_re[i]:.6e} {vals_im[i]:.6e}\n")

def save_2d_map(filename: str, real_part: np.ndarray, imag_part: np.ndarray, config: NISE2017Config):
    """Saves a 2D map (T3, T1) to a file in NISE format."""
    # Format: NISE print2D usually prints:
    # Header: "Samples ..." or just dimensions?
    # Based on NISE code: fprintf(f, "%d %d\n", t3_max, t1_max)
    # Then data rows.
    
    with open(filename, "w") as f:
        t3_max, t1_max = real_part.shape
        f.write(f"{t3_max} {t1_max}\n")
        
        for t3 in range(t3_max):
            row_strs = []
            for t1 in range(t1_max):
                val_re = real_part[t3, t1]
                val_im = imag_part[t3, t1]
                row_strs.append(f"{val_re:.6e} {val_im:.6e}")
            f.write(" ".join(row_strs) + "\n")


class NISECalculation:
    """Base class for all NISE 2017 calculation modes."""
    def __init__(self, config: NISE2017Config,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.config = config
        self.loader = NISEBinaryLoader(config,device)
        self.device = device
        
    def run(self):
        raise NotImplementedError("run() method must be implemented by subclasses")

    def validate_propagation(self):
        if self.config.propagation not in ["Sparse", "RK4"]:
            warnings.warn(f"NISE 2017 Propagation method '{self.config.propagation}' is not directly supported. Falling back to default TorchNISE propagation.")

class PropagationBasedCalculation(NISECalculation):
    """Base class for calculations involving time propagation (Population, Diffusion, Absorption, etc.)."""

    def get_common_params(self, mode="Population"):
        total_time = (self.config.t_max_1 - 1) * self.config.timestep
        return NISEParameters(
            dt=self.config.timestep,
            total_time=total_time,
            temperature=self.config.temperature,
            mode=mode,
            save_interval=1,
            device=self.device,
            save_u=True,
            keep_on_cuda=(self.device != "cpu")
        )

    def load_and_prep_hamiltonian(self, steps=None):
        print(f"Loading Hamiltonian from {self.config.hamiltonian_file}")
        if not os.path.exists(self.config.hamiltonian_file):
            raise FileNotFoundError(
                f"Hamiltonian file not found: {self.config.hamiltonian_file}"
            )

        h_ham, site_noise, _ = self.loader.load_hamiltonian(steps=steps)
        # h_ham can be (T, R, N, N) OR (N, N)

        if site_noise is not None:
             # Static H + Noise mode
             # h_ham is (N, N)
             # return as is
             return h_ham, site_noise
        
        # Original logic: Shift
        # TorchNISE uses hfull[t] to propagate from t-1 to t.
        h_torch = torch.zeros_like(h_ham)
        if self.config.t_max_1 > 1:
            h_torch[1:] = h_ham[:-1]
        h_torch[0] = h_ham[0]
        return h_torch, None  # site_noise is None

    def run_propagation(self, h_torch, params, initial_state=None, site_noise=None):
        if initial_state is None:
            initial_state = torch.zeros(self.config.singles)
            initial_state[0] = 1.0
            
        num_realizations = self.config.end_point - self.config.begin_point
        from torchnise.nise import nise_propagate
        print(f"Running NISE calculation on {params.device}...")
        
        # Determine if constant_v mode is needed
        if site_noise is not None or len(h_torch.shape) == 2:
             params = replace(params, constant_v=True)
             
        h_torch = h_torch.to(params.device)
        initial_state = initial_state.to(params.device)
        if site_noise is not None:
             site_noise = site_noise.to(params.device)
             
        return nise_propagate(h_torch, num_realizations, initial_state, params, site_noise=site_noise)

class PopulationCalculation(PropagationBasedCalculation):
    def run(self) -> Tuple[Any, Any, Any]:
        self.validate_propagation()
        params = self.get_common_params(mode="Population")
        
        h_torch, site_noise = self.load_and_prep_hamiltonian()

        # Determine initial state
        initial_state = torch.zeros(self.config.singles)
        site_idx = self.config.initial_state_site
        if site_idx < 0 or site_idx >= self.config.singles:
            site_idx = 0
        initial_state[site_idx] = 1.0

        population, coherence, u, _ = self.run_propagation(
            h_torch, params, initial_state, site_noise=site_noise
        )

        total_time = (self.config.t_max_1 - 1) * self.config.timestep
        
        if self.config.save_popf:
            print("Saving output to Pop.dat and PopF.dat")
            save_nise2017_output(u, total_time, self.config.timestep)
        else:
            print("Skip Saving")

            
        print("Done.")
        
        return population, coherence, u
        
class ResponseFunctionCalculation(PropagationBasedCalculation):
    """Base class for Response Function calculations (Absorption, 2DES, etc.)."""
    pass

class AbsorptionCalculation(ResponseFunctionCalculation):
    def run(self):
        self.validate_propagation()
        params = self.get_common_params(mode="Absorption")
        h_torch, h_nise = self.load_and_prep_hamiltonian()

        # Load Dipole
        print(f"Loading Dipole from {self.config.dipole_file}")
        mu_nise = self.loader.load_dipole()
        mu_torch = torch.zeros_like(mu_nise)
        if self.config.t_max_1 > 1:
            mu_torch[1:] = mu_nise[:-1]
        mu_torch[0] = mu_nise[0]
        params.mu = mu_torch.cpu().numpy()

        # Rotating Frame Shift
        shifte = (self.config.min_freq_1 + self.config.max_freq_1) / 2.0
        print(f"Applying rotating frame shift: {shifte} cm-1")
        # Apply shift to h_torch
        # Original code applied shift to h_nise then copied. We can apply to h_torch directly.
        if h_torch.ndim == 2:
            # Static H (N, N)
            h_torch -= torch.eye(self.config.singles, device=self.device) * shifte
        else:
            # Trajectory H (T, R, N, N)
            for i in range(self.config.singles):
                h_torch[:, :, i, i] -= shifte

        initial_state = torch.zeros(self.config.singles)
        initial_state[
            0
        ] = 1.0  # Standard start? usually arbitrary for absorption if dipoles handled correctly
        
        # Optimize Absorption using Vector Propagation
        params = replace(params, save_u=False, save_wavefunction=True)
        # Use dipoles as initial state
        initial_state_vector = mu_torch[0] # (R, N, 3) 

        population, coherence, u, psi_loc = self.run_propagation(
            h_torch, params, initial_state_vector, site_noise=h_nise
        )

        print("Calculating time-domain absorption...")
        from torchnise.nise import absorption_time_domain_vector

        # mu_torch is (steps, realizations, n, 3).
        # Need to match what absorption_time_domain_vector expects.
        # psi_loc is (Realization, Steps, N, 3)
        # mu_torch needs to be (Realization, Steps, N, 3)
        mu_for_absorb = mu_torch.permute(1, 0, 2, 3) # Keep as tensor
        
        avg_s1 = absorption_time_domain_vector(
            psi_loc, mu_for_absorb, dt=self.config.timestep
        )

        print("Saving output to Absorption.dat and TD_Absorption.dat")
        save_nise2017_absorption(torch.tensor(avg_s1), self.config)
        print("Done.")



class DiffusionCalculation(PropagationBasedCalculation):
    def run(self):
        # We need custom logic because it runs propagation multiple times (per site)
        params = self.get_common_params(mode="Population")
        h_torch, site_noise = self.load_and_prep_hamiltonian()

        pos_traj, box_size = self.loader.load_positions()

        num_realizations = self.config.end_point - self.config.begin_point
        msd_pop = torch.zeros(self.config.t_max_1)

        print(f"Running Diffusion calculation on {params.device}...")

        for start_site in range(self.config.singles):
            initial_state = torch.zeros(self.config.singles)
            initial_state[start_site] = 1.0

            _, _, u, _ = self.run_propagation(h_torch, params, initial_state, site_noise=site_noise)
            probs = (torch.abs(u) ** 2).real.cpu()  # (real, t, b, a)

            for r in range(num_realizations):
                ti = (self.config.begin_point + r) * self.config.sample_rate
                p0 = pos_traj[ti]
                for t in range(self.config.t_max_1):
                    tj = ti + t
                    pt = pos_traj[tj]
                    for b in range(self.config.singles):
                        dx = pt[b, 0] - p0[start_site, 0]
                        dy = pt[b, 1] - p0[start_site, 1]
                        dz = pt[b, 2] - p0[start_site, 2]
                        if box_size > 0:
                            dx -= box_size * torch.round(dx / box_size)
                            dy -= box_size * torch.round(dy / box_size)
                            dz -= box_size * torch.round(dz / box_size)
                        d2 = dx * dx + dy * dy + dz * dz
                        msd_pop[t] += probs[r, t, b, 0] * d2

        norm = num_realizations * self.config.singles
        msd_pop /= norm

        with open("RMSD.dat", "w") as f:
            for t in range(self.config.t_max_1):
                f.write(
                    f"{t*self.config.timestep:.6f} {msd_pop[t]:.6e} 0.000000e+00\n"
                )
        print("Done.")

class CDCalculation(ResponseFunctionCalculation):
    def run(self):
        self.validate_propagation()
        params = self.get_common_params(
            mode="CD"
        )  # reuse absorption mode logic for nise param setup
        h_torch, h_nise = self.load_and_prep_hamiltonian()

        print(f"Loading Dipole from {self.config.dipole_file}")
        mu_nise = self.loader.load_dipole()
        mu_torch = torch.zeros_like(mu_nise)
        if self.config.t_max_1 > 1:
            mu_torch[1:] = mu_nise[:-1]
        mu_torch[0] = mu_nise[0]
        params.mu = mu_torch.cpu().numpy()

        # Load Positions for CD
        print(f"Loading Positions from {self.config.position_file}")
        pos_traj, box_size = self.loader.load_positions()
        # pos_traj is (length, n_sites, 3)
        # Reshape to match u (realizations, steps, n_sites, 3) expectation logic or tile?
        # NISELoader.load_positions returns (length, n_sites, 3)
        # We need (realizations, t_max_1, n_sites, 3)

        # Extract realizations slice
        num_realizations = self.config.end_point - self.config.begin_point
        # Usually positions are (length, n_sites, 3) where length covers all.
        # Check load_positions again. It reads the whole file.
        # But for realization r and time t, index is (begin + r)*sample + t

        pos_reshaped = torch.zeros(
            (num_realizations, self.config.t_max_1, self.config.singles, 3)
        )
        for r in range(num_realizations):
            base = (self.config.begin_point + r) * self.config.sample_rate
            for t in range(self.config.t_max_1):
                pos_reshaped[r, t] = pos_traj[base + t]

        # Rotating Frame Shift
        shifte = (self.config.min_freq_1 + self.config.max_freq_1) / 2.0
        for i in range(self.config.singles):
            h_torch[:, :, i, i] -= shifte

        initial_state = torch.zeros(self.config.singles)
        initial_state[0] = 1.0

        # Optimize CD using Vector Propagation
        params = replace(params, save_u=False, save_wavefunction=True)
        # Use dipoles as initial state
        initial_state_vector = mu_torch[0]

        population, coherence, u, psi_loc = self.run_propagation(
            h_torch, params, initial_state_vector, site_noise=h_nise
        )

        print("Calculating time-domain CD...")
        from torchnise.absorption import cd_time_domain_vector

        # mu_torch is (steps, realizations, n, 3), need (realizations, steps, n, 3)
        mu_for_absorb = mu_torch.permute(1, 0, 2, 3) # Keep as tensor
        pos_for_cd = pos_reshaped # Already (R, T, N, 3)

        avg_cd = cd_time_domain_vector(
            psi_loc, mu_for_absorb, pos_for_cd, dt=self.config.timestep
        )

        print(
            "Saving output to CD.dat and TD_CD.dat (reusing absorption save logic)"
        )
        save_nise2017_absorption(torch.tensor(avg_cd), self.config)
        if os.path.exists("Absorption.dat"):
            os.rename("Absorption.dat", "CD.dat")
        if os.path.exists("TD_Absorption.dat"):
            os.rename("TD_Absorption.dat", "TD_CD.dat")
        print("Done.")

class LDCalculation(ResponseFunctionCalculation):
    def run(self):
        self.validate_propagation()
        params = self.get_common_params(mode="LD")
        h_torch, h_nise = self.load_and_prep_hamiltonian()

        print(f"Loading Dipole from {self.config.dipole_file}")
        mu_nise = self.loader.load_dipole()
        mu_torch = torch.zeros_like(mu_nise)
        if self.config.t_max_1 > 1:
            mu_torch[1:] = mu_nise[:-1]
        mu_torch[0] = mu_nise[0]
        params.mu = mu_torch.cpu().numpy()

        shifte = (self.config.min_freq_1 + self.config.max_freq_1) / 2.0
        for i in range(self.config.singles):
            h_torch[:, :, i, i] -= shifte

        initial_state = torch.zeros(self.config.singles)
        initial_state[0] = 1.0

        # Optimize LD using Vector Propagation
        params = replace(params, save_u=False, save_wavefunction=True)
        # Use dipoles as initial state
        initial_state_vector = mu_torch[0]

        population, coherence, u, psi_loc = self.run_propagation(
            h_torch, params, initial_state_vector, site_noise=h_nise
        )

        print("Calculating time-domain LD...")
        from torchnise.absorption import ld_time_domain_vector

        mu_for_absorb = mu_torch.permute(1, 0, 2, 3) # Keep as tensor

        # Typically Z-axis is used
        avg_ld = ld_time_domain_vector(
            psi_loc, mu_for_absorb, axis=2, dt=self.config.timestep
        )

        print("Saving output to LD.dat and TD_LD.dat")
        save_nise2017_absorption(torch.tensor(avg_ld), self.config)
        if os.path.exists("Absorption.dat"):
            os.rename("Absorption.dat", "LD.dat")
        if os.path.exists("TD_Absorption.dat"):
            os.rename("TD_Absorption.dat", "TD_LD.dat")
        print("Done.")

class LuminescenceCalculation(AbsorptionCalculation):
    """
    Luminescence (Fluorescence).
    Starts from Boltzmann weighted excited state equilibrium.
    Psi(0) = rho_eq * mu.
    """
    def _apply_boltzmann(self, mu, h_torch, params):
        # Apply exp(-beta H) to mu for each realization
        # h_torch: (Steps, Realizations, N, N) -> Use t=0: (R, N, N)
        # mu: (Steps, Realizations, N, 3) -> Use t=0: (R, N, 3)
        
        h_0 = h_torch[0] # (R, N, N)
        mu_0 = mu[0]     # (R, N, 3)
        
        beta = 1.0 / (0.695 * params.temperature) # 0.695 cm-1/K
        
        # Diagonalize H_0
        evals, evecs = torch.linalg.eigh(h_0) # (R, N), (R, N, N)
        
        # Calculate rho = exp(-beta E) / Z
        rho_diag = torch.exp(-beta * evals)
        Z = rho_diag.sum(dim=1, keepdim=True)
        rho_diag = rho_diag / Z # (R, N)
        
        # Transform rho back to site basis? or just apply in eigenbasis?
        # We need Psi(0) in site basis.
        # Psi = rho @ mu.
        # In Eigenbasis: mu_eig = U^dag @ mu
        # rho_mu_eig = rho_diag * mu_eig
        # Psi_site = U @ rho_mu_eig
        
        # mu_0 is (R, N, 3). Treat as 3 vectors per realization.
        # evecs is (R, N, N).
        
        # 1. Transform mu to eigenbasis
        # mu_0 (R, N, 3).
        # evecs^H @ mu_0
        mu_eig = torch.matmul(evecs.transpose(1, 2).conj(), mu_0.to(dtype=torch.complex64))
        
        # 2. Apply Weighting
        # rho_diag (R, N). Expand to (R, N, 1) or (R, N, 3) broadcasting
        mu_eig_weighted = mu_eig * rho_diag.unsqueeze(-1)
        
        # 3. Transform back
        psi_0 = torch.matmul(evecs.to(dtype=torch.complex64), mu_eig_weighted)
        
        # Normalize? NISE normalizes by partition function (done) and 
        # calculates overlap.
        # We also need sqrt(Norm / new_norm)? 
        # NISE: iQ = sqrt(norm / Q_weighted). 
        # Renormalize cr[a] = cr[a] * iQ.
        # "Find initial norm": norm = sum |mu|^2.
        # "Find final norm": Q = sum |weighted_mu|^2.
        # Scale by sqrt(initial_norm / final_norm).
        
        norm_initial = (mu_0.norm(dim=(1,2))**2)
        norm_final = (psi_0.norm(dim=(1,2))**2)
        
        scale = torch.sqrt(norm_initial / norm_final).unsqueeze(-1).unsqueeze(-1)
        psi_0 = psi_0 * scale
        
        return psi_0

    def run(self):
        print("Running Luminescence (Fluorescence)...")
        # Custom run logic to insert weighting
        self.validate_propagation()
        params = self.get_common_params(mode="Luminescence")
        h_torch, h_nise = self.load_and_prep_hamiltonian()
        
        print(f"Loading Dipole from {self.config.dipole_file}")
        mu_nise = self.loader.load_dipole()
        mu_torch = torch.zeros_like(mu_nise)
        if self.config.t_max_1 > 1:
            mu_torch[1:] = mu_nise[:-1]
        mu_torch[0] = mu_nise[0]
        params.mu = mu_torch.cpu().numpy()
        
        # Apply Shift
        shifte = (self.config.min_freq_1 + self.config.max_freq_1) / 2.0
        for i in range(self.config.singles):
            h_torch[:, :, i, i] -= shifte
            
        print("Calculating Boltzmann Weighted Initial State...")
        # Prepare Initial State
        psi_0 = self._apply_boltzmann(mu_torch, h_torch, params)
        psi_0 = psi_0.to(device=params.device)
        
        # Optimize using Vector Propagation
        params = replace(params, save_u=False, save_wavefunction=True)

        population, coherence, u, psi_loc = self.run_propagation(
            h_torch, params, psi_0, site_noise=h_nise
        )
        
        print("Calculating time-domain Luminescence...")
        from torchnise.absorption import luminescence_time_domain_vector
        
        mu_for_absorb = mu_torch.permute(1, 0, 2, 3) 
        
        avg_lum = luminescence_time_domain_vector(
            psi_loc, mu_for_absorb, dt=self.config.timestep
        )
        
        print("Saving output to Luminescence.dat")
        save_nise2017_absorption(torch.tensor(avg_lum), self.config)
        
        # Rename output files to match NISE
        if os.path.exists("Absorption.dat"): os.rename("Absorption.dat", "Luminescence.dat")
        if os.path.exists("TD_Absorption.dat"): os.rename("TD_Absorption.dat", "TD_Lum.dat")
        print("Done.")

class RamanCalculation(ResponseFunctionCalculation):
    def run(self):
        self.validate_propagation()
        params = self.get_common_params(mode="Raman")
        h_torch, h_nise = self.load_and_prep_hamiltonian()
        
        print(f"Loading Raman Tensor (Alpha) from {self.config.alpha_file}")
        alpha_nise = self.loader.load_alpha()
        
        # Shift
        alpha_torch = torch.zeros_like(alpha_nise)
        if self.config.t_max_1 > 1:
             alpha_torch[1:] = alpha_nise[:-1]
        alpha_torch[0] = alpha_nise[0]
        
        # Shift Hamiltonian
        shifte = (self.config.min_freq_1 + self.config.max_freq_1) / 2.0
        for i in range(self.config.singles):
             h_torch[:, :, i, i] -= shifte
             
        # Initial State: Alpha[0] (Realizations, N, 6)
        psi_0 = alpha_torch[0].to(device=params.device)
        
        # Optimize
        params = replace(params, save_u=False, save_wavefunction=True)
        
        print("Propagating Raman Tensors...")
        population, coherence, u, psi_loc = self.run_propagation(
             h_torch, params, psi_0, site_noise=h_nise
        )
        
        print("Calculating time-domain Raman...")
        from torchnise.absorption import raman_time_domain_vector
        
        alpha_for_calc = alpha_torch.permute(1, 0, 2, 3)
        vv, vh = raman_time_domain_vector(psi_loc, alpha_for_calc, dt=self.config.timestep)
        
        # Save Outputs
        # Reusing absorption saver is tricky because we have two signals
        # Manually save or adapt?
        # NISE saves TD_Raman_VV.dat and Raman_VV.dat etc.
        
        # Create temp config for saving
        # Save VV
        print("Saving VV...")
        save_nise2017_absorption(torch.tensor(vv), self.config)
        if os.path.exists("Absorption.dat"): os.rename("Absorption.dat", "Raman_VV.dat")
        if os.path.exists("TD_Absorption.dat"): os.rename("TD_Absorption.dat", "TD_Raman_VV.dat")
        
        # Save VH
        print("Saving VH...")
        save_nise2017_absorption(torch.tensor(vh), self.config)
        if os.path.exists("Absorption.dat"): os.rename("Absorption.dat", "Raman_VH.dat")
        if os.path.exists("TD_Absorption.dat"): os.rename("TD_Absorption.dat", "TD_Raman_VH.dat")
        
        print("Done.")

class MCFRETCalculation(PopulationCalculation):
    """
    Monte Carlo FRET calculation.
    Uses population propagation but typically implies incoherent hopping dynamics.
    TorchNISE standard propagation is coherent (Schrodinger eqn).
    If the user wants incoherent, they might need a Lindblad or Rate equation solver.
    For now, we map this to standard population propagation but warn the user.
    """
    def run(self):
        print("Running MCFRET Calculation (Mapped to Population)...")
        warnings.warn("MCFRET in TorchNISE currently uses coherent propagation. True incoherent MC hopping is not yet implemented.")
        super().run()

class RedfieldCalculation(PopulationCalculation):
    """
    Redfield theory calculation.
    Similar to MCFRET, this implies a specific propagator (Redfield tensor).
    TorchNISE currently supports standard time-dependent Hamiltonian propagation.
    Mapping to standard propagation as a placeholder to allow tutorials to run.
    """
    def run(self):
        print("Running Redfield Calculation (Mapped to Population)...")
        warnings.warn("Redfield theory propagator is not explicitly implemented. Using standard NISE propagation.")
        super().run()

class TwoDIRCalculation(PropagationBasedCalculation):
    def run(self):
        raise NotImplementedError(
            "2D Spectroscopy is available in the feature/2d-spectroscopy branch"
        )


class TwoDESCalculation(ResponseFunctionCalculation):

    """
    2D Electronic Spectroscopy (and variants like 2DIR).
    Uses t_max_1 (coherence), t_max_2 (population/waiting), t_max_3 (detection).
    """

    def run(self):
        raise NotImplementedError(
            "2D Spectroscopy is available in the feature/2d-spectroscopy branch"
        )


class StaticPropertyCalculation(NISECalculation):
    """Base class for calculations that analyse trajectories without propagation."""
    pass

class AnalyseCalculation(StaticPropertyCalculation):
    def run(self):
        print("Running Analysis...")
        freq_traj = self.loader.load_hamiltonian_diagonal()
        # freq_traj is (length, n_sites)
        
        sites = self.config.singles
        mean_energies = np.mean(freq_traj, axis=0)
        std_energies = np.std(freq_traj, axis=0)
        
        # Calculate cross-correlations at t=0 (covariance matrix)
        cov_matrix = np.cov(freq_traj, rowvar=False)

        with open("Analysis.dat", 'w') as f:
            f.write("# Site Mean Std\n")
            for i in range(sites):
                f.write(f"{i+1} {mean_energies[i]:.6f} {std_energies[i]:.6f}\n")
            
            f.write("\n# Covariance Matrix\n")
            for row in cov_matrix:
                f.write(" ".join([f"{x:.6e}" for x in row]) + "\n")
        print("Analysis.dat saved.")

class DOSCalculation(StaticPropertyCalculation):
    def run(self):
        print("Running DOS Calculation...")
        freq_traj = self.loader.load_hamiltonian_diagonal()
        # freq_traj shape: (length, n_sites)

        min_e = self.config.min_freq_1
        max_e = self.config.max_freq_1
        if min_e == 0 and max_e == 0:
            min_e = np.min(freq_traj)
            max_e = np.max(freq_traj)

        bins = 100  # Default bins
        if self.config.fft_size > 0:
            bins = self.config.fft_size

        # Create histogram for each site and total
        total_hist, bin_edges = np.histogram(
            freq_traj.flatten(), bins=bins, range=(min_e, max_e), density=True
        )

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        with open("DOS.dat", "w") as f:
            for i, e in enumerate(bin_centers):
                f.write(f"{e:.6f} {total_hist[i]:.6e}\n")
        print("DOS.dat saved.")

class CorrelationCalculation(StaticPropertyCalculation):
    def run(self):
        from torchnise.spectral_density_generation import (
            get_auto,
            get_cross,
            sd_reconstruct_fft,
        )

        freq_traj = self.loader.load_hamiltonian_diagonal()
        sites = self.config.singles
        length = self.config.length

        T = self.config.t_max_1
        dt = self.config.timestep
        auto_matrix = np.zeros((sites, sites, length))

        print("Calculating Correlation...")
        for i in range(sites):
            for j in range(i, sites):
                data_i = freq_traj[:, i] - np.mean(freq_traj[:, i])
                data_j = freq_traj[:, j] - np.mean(freq_traj[:, j])

                noise_i = data_i.reshape(1, -1)
                noise_j = data_j.reshape(1, -1)

                if i == j:
                    auto = get_auto(noise_i)
                    auto_matrix[i, i, : len(auto)] = auto
                else:
                    if self.config.technique == "Correlation":
                        cross = get_cross(noise_i, noise_j)
                        auto_matrix[i, j, : len(cross)] = cross
                        auto_matrix[j, i, : len(cross)] = cross

        w_axis = None
        sd_results = []
        if sites > 0:
            for i in range(sites):
                auto = auto_matrix[i, i, : length // 2]
                J_new, curr_w_axis, _ = sd_reconstruct_fft(
                    auto,
                    dt,
                    self.config.temperature,
                    damping_type="gauss",
                    cutoff=T * dt,
                )
                sd_results.append(J_new)
                w_axis = curr_w_axis

        if w_axis is not None:
            with open("SpectralDensity.dat", "w") as f:
                for i in range(len(w_axis)):
                    line = (
                        f"{w_axis[i]:.6f} "
                        + " ".join([f"{sd_results[s][i]:.6e}" for s in range(sites)])
                        + "\n"
                    )
                    f.write(line)

        with open("CorrelationMatrix.dat", "w") as f:
            for t in range(T):
                line = f"{t*dt:.6f} "
                for i in range(sites):
                    for j in range(i, sites):
                        line += f"{auto_matrix[i, j, t]:.6e} "
                        if self.config.technique == "Autocorrelation":
                            break
                f.write(line + "\n")
        print("Done.")

# Registry of available calculation modes
CALCULATION_MODES: Dict[str, Type[NISECalculation]] = {
    "Pop": PopulationCalculation,
    "Population": PopulationCalculation,
    "Absorption": AbsorptionCalculation,
    "Correlation": CorrelationCalculation,
    "Autocorrelation": CorrelationCalculation,
    "Dif": DiffusionCalculation,
    "Diffusion": DiffusionCalculation,
    "Analyse": AnalyseCalculation,
    "Analyze": AnalyseCalculation,
    "AnalyseFull": AnalyseCalculation,
    "AnalyzeFull": AnalyseCalculation,
    "DOS": DOSCalculation,
    "CD": CDCalculation,
    "LD": LDCalculation,
    "Luminescence": LuminescenceCalculation,
    "PL": LuminescenceCalculation,
    "Fluorescence": LuminescenceCalculation,
    "MCFRET": MCFRETCalculation,
    "MCFRET-Rate": MCFRETCalculation,
    "Redfield": RedfieldCalculation,
    "2DES": TwoDESCalculation,
    "CG-2DES": TwoDESCalculation,
    "2DIR": TwoDIRCalculation,
    "2DIRRAMAN": TwoDESCalculation,
    "2DUvis": TwoDESCalculation,
    "2DFD": TwoDESCalculation,
    "Raman": RamanCalculation, # Explicit Raman
    "SFG": TwoDESCalculation,  # Similar
    "2DUvis": TwoDESCalculation,
    "2DUVvis": TwoDESCalculation, # Alias
    "2DUVvis": TwoDESCalculation,
    "noEAUVvis": TwoDESCalculation,
    "GBUVvis": TwoDESCalculation,
    "CG_2DES": TwoDESCalculation, # Note underscore
    "FD_CG_2DES": TwoDESCalculation,
}


# List of all known techniques in NISE 2017 for validation
ALL_NISE_TECHNIQUES = [
    "Pop", "Population", "Analyse", "Analyze", "AnalyseFull", "AnalyzeFull",
    "Correlation", "Autocorrelation", "Dif", "Diffusion", "Ani", "Anisotropy",
    "Absorption", "DOS", "MCFRET", "Redfield", "Luminescence", "PL", "Fluorescence",
    "LD", "CD", "Raman", "SFG", "2DES"
]


def run_nise2017(input_file_path: str):
    print(f"Loading NISE 2017 input: {input_file_path}")
    parser = NISEInputParser(input_file_path)
    config = parser.parse()
    
    technique = config.technique
    if technique not in CALCULATION_MODES:
        if technique in ALL_NISE_TECHNIQUES:
            raise NotImplementedError(f"Technique '{technique}' is a valid NISE 2017 technique but is not yet implemented in the TorchNISE adapter.")
        else:
            raise ValueError(f"Unknown NISE 2017 technique: '{technique}'")
            
    calc_class = CALCULATION_MODES[technique]
    calculation = calc_class(config)
    calculation.run()

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python nise2017_adapter.py <input_file>")
        return
        
    input_file = sys.argv[1]
    run_nise2017(input_file)

if __name__ == "__main__":
    main()
