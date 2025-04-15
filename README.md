# TorchNISE

`TorchNISE` is a python package with a fully differentiable PyTorch implementation of the Numerical Integration of Schrödinger Equation (NISE) algorithm and the extensions presented in the papers "[Spectral Densities, Structured Noise and Ensemble Averaging with Open Quantum Dynamics](https://doi.org/10.1063/5.0224807)" and "[Machine-learned correction to ensemble-averaged wave packet dynamics](https://doi.org/10.1063/5.0166694)" .

## Features

- Population (and Coherence) Dynamics 

- Exciton Diffusion

- Absorption Spectra

- Noise generation following spectral densities, to generate time dependent Hamiltonians.

- Thermal corrections - TNISE and MLNISE: 

- Training for MLNISE corrections. 

- Generation of training data sets with pyheom (pyheom installation required, tested with https://github.com/tatsushi-ikeda/pyheom, version 1.0.0a2)

- Averaging - Different Averaging procedures for the thermalized variants as described in "Spectral Densities, Structured Noise and Ensemble Averaging within Open
Quantum Dynamics".

- Fully differentiable - As a pytorch implementation, the implementation is fully differentiable via automatic differentiation: i.E. gradients can be automatically calculated throughout all calculations. This can be used to train MLNISE models, but many other applications are thinkable. One could use this e.g. to get the gradient of the error between the calculated and an experimental spectral density with respect to the average Hamiltonian. One could then optimize the Hamiltonian so that it matches an experimental spectral density.

- GPU support - As most of the code is build on top of the pytorch package, which can delegate calculations to the GPU, TorchNISE can also be run on the GPU. The GPU calculations are currently mostly bottlenecked by the PyTorch implementation of eigh. In our testing - with our hardware - we only found GPU acceleration to be useful for very large Hamiltonians for NISE and TNISE. 

- Efficient realizations - In TorchNISE the realizations are calculated as batched PyTorch operationsl rather than parallel for loops. This improves speed as the calculations are otherwise bottlenecked by cache and memory access. By saving on memory and cache operations, the calculation time actually increases less than linearly with the number of realizations in this way.

- Spectral density reconstruction - with fft-based or super resolution methods.

## Installation

The first step to install `torchnise` is to clone this repository.

```bash
git clone https://github.com/CPBPG/TorchNISE.git
```

We recommend creating a virtual environment using either `venv` or `miniconda`. Once the environment is active, installing `mlnise` is straightforward using the package manager [pip](https://pip.pypa.io/en/stable/).

```bash
pip install .
```

## Usage

2 Example Scripts can be found in the example folder and serve as a starting point.

Documentation is available in HTML format and a print-to-pdf version of the HTML.

It is generated with sphinx based on the doc strings which are partially AI generated. Please raise an Issue for any inconsistencies in the documentation.

The current state of the code is not stable and will be continued to be updated

## Licence

Licenced under Creative Commons Attribution 4.0 International Public License. 
That is, you can use this work in any way you want as long as you give Attribution to the original authors.
We Request the Attributions to be in the form of citing the respective papers.

## Utilized in 

- [Machine-learned correction to ensemble-averaged wave packet dynamics](https://doi.org/10.1063/5.0166694)"

- [Spectral Densities, Structured Noise and Ensemble Averaging with Open Quantum Dynamics](https://doi.org/10.1063/5.0224807)

- [Excitation Energy Transfer between Porphyrin Dyes on a Clay Surface: A study employing Multifidelity Machine Learning](https://arxiv.org/abs/2410.20551)

## Changelog

### 0.1.0

- Initial Release

### 0.2.0

- support for keeping large Tensors on the disk via h5py
- added example for Diffusion Calculation
- added example showing usage of numerical spectral density
- moved the noise Algorithm to use pytorch
- added options to have constant coupling or couplings with a different timestep than the energies so that the full time dependent Hamiltonian does not have to be put in memory
- added option to automatically save the population dynamics starting in multiple sites
- multiple bugfixes and optimizations

### 0.3.0

- added support for MLNISE training
- weighted averages migrated to Pytorch
- removed in-place operations to allow for backpropagation
- added support for generating training sets with pyheom (https://github.com/tatsushi-ikeda/pyheom, version 1.0.0a2)

## Future plans

- Include detailed documentation for MLNISE training

- Add Tutorials

- Add Support for Decentralized compute options with open MP or similar.

- Make the code more readable by splitting long functions into smaller ones and reduce the number of inputs.

- Migrate remaining NumPy based functions to PyTorch
(0.2.0 noise generation has been migrated to PyTorch)
(0.3.0 weighted averages migrated to Pytorch)

## Completed

- Optimize memory usage: right now too many intermediate results are kept in memory unnecessarily.
Now tensors can be kept on disk by using h5py



