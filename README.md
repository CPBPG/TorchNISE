
#  TorchNISE

**TorchNISE** is a Python package offering a fully differentiable, GPU-accelerated implementation of the NISE (Numerical Integration of Schrödinger Equation) algorithm in PyTorch. It includes enhancements introduced in:

- 📄 [Spectral Densities, Structured Noise and Ensemble Averaging with Open Quantum Dynamics](https://doi.org/10.1063/5.0224807)  
- 🤖 [Machine-learned Correction to Ensemble-Averaged Wave Packet Dynamics](https://doi.org/10.1063/5.0166694)

---

## ✨ Features

-  **Population & Coherence Dynamics**
-  **Exciton Diffusion**
-  **Absorption Spectra Calculation**
-  **Structured Noise Generation** following arbitrary spectral densities
-  **Thermal Corrections**: TNISE and MLNISE
-  **MLNISE Training**: Train neural networks to correct ensemble dynamics as described in [here](https://doi.org/10.1063/5.0166694)
-  **Training Data Generation** using [pyheom](https://github.com/tatsushi-ikeda/pyheom) (tested with `v1.0.0a2`)
-  **Averaging Strategies**: Standard, Boltzmann, and Interpolated as described in [here](https://doi.org/10.1063/5.0224807) 
-  **Fully Differentiable**: Backpropagation-compatible with PyTorch autograd.

Gradients can be automatically calculated throughout all calculations. This can be used to train MLNISE models, but many other applications are thinkable. One could use this e.g. to get the gradient of the error between the calculated and an experimental spectral density with respect to the average Hamiltonian. One could then optimize the Hamiltonian so that it matches an experimental absorption spectrum.
-  **GPU Acceleration**: Especially effective for large systems or small systems with many realizations

As most of the code is build on top of the pytorch package, which can delegate calculations to the GPU, TorchNISE can also be run on the GPU. This gives a considerable speedup for many realizations or large Hamiltonians.
. 
-  **Efficient Realizations** via PyTorch batched ops instead of loops

In TorchNISE the realizations are calculated as batched PyTorch operationsl rather than parallel for loops. This improves speed as the calculations are otherwise bottlenecked by cache and memory access. By saving on memory and cache operations, the calculation time actually increases less than linearly with the number of realizations in this way.
-  **Spectral Density Reconstruction** via FFT or super-resolution methods

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/CPBPG/TorchNISE.git
cd TorchNISE
```

Then install it:

```bash
pip install .
```

We recommend using a virtual environment via `venv` or `conda`.

---

## 📈 Example Usage

See the `examples/` folder for hands-on demonstrations:

-  **Population and Absorption**:  
  `Population_Dynamics_And_Absorption_Example.py`

-  **Numerical Spectral Densities**:  
  `Population_Dynamics_numerical_SD.py`

-  **Diffusion Simulations**:  
  `Diffusion_example.py`

- **Spectral Density Reconstruction**
  `SD_reconstruction_Example.py`

-  **MLNISE Training**:  
  - Generate datasets: `create_datasets_example.py`  ([pyheom](https://github.com/tatsushi-ikeda/pyheom) required)
  - Train a model: `training_example.py`  
  - Compare TNISE vs MLNISE: `compare_TNISE_MLNISE_example.py`

---

## 📄 Documentation

A **PDF version** of the documentation is included and generated via Sphinx from inline docstrings (some AI-assisted). Please open an issue if you spot inconsistencies or missing info.

> _Note: The codebase is still evolving. Expect active updates and improvements._

---

## 📜 License

Licensed under [**CC BY 4.0**](https://creativecommons.org/licenses/by/4.0/):  
Feel free to use and adapt, but please cite the original papers:

- [Machine-learned correction to ensemble-averaged wave packet dynamics](https://doi.org/10.1063/5.0166694)
- [Spectral Densities, Structured Noise and Ensemble Averaging with Open Quantum Dynamics](https://doi.org/10.1063/5.0224807)

---

## 📄 Used In

- [Machine-learned correction to ensemble-averaged wave packet dynamics](https://doi.org/10.1063/5.0166694)
- [Spectral Densities, Structured Noise and Ensemble Averaging with Open Quantum Dynamics](https://doi.org/10.1063/5.0224807)
- [Excitation Energy Transfer on Clay Surfaces via Multifidelity ML](https://arxiv.org/abs/2410.20551)

---

## 🛠️ Changelog
### v0.3.3
- GPU optimizations for intermediate size (32-512) Hamiltonains and bug fixes
### v0.3.2
- Avoid GPU memory overflow during noise gen, further gpu otimizations
### v0.3.1
- GPU now better supported, it also runs faster for small systems with many realizations.
### v0.3.0
- MLNISE training support
- PyTorch-based weighted averaging
- Fully differentiable (no in-place ops)
- PyHEOM training dataset support

### v0.2.0
- HDF5 tensor storage support
- Diffusion calculation examples
- Numerical spectral density support
- PyTorch-based noise generation
- Coupling-only or partial time-dependence
- Batch population simulations
- Performance improvements & bugfixes

### v0.1.0
- Initial release

---

## 🧭 Roadmap

### Planned
-  Tutorials and training documentation
-  Decentralized computation (OpenMP, etc.)
-  Code cleanup & function splitting
-  Complete migration from NumPy to PyTorch

### Completed
- Reduced memory usage via h5py (v0.2.0)
- PyTorch-based noise generation (v0.2.0)
- PyTorch-based weighted averaging (v0.3.0)
- MLNISE training support (v0.3.0)





