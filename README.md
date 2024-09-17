# TorchNISE

`TorchNISE` is a python package with a fully differentiable PyTorch implementation of the Numerical Integration of Schrödinger Equation (NISE) algorithm and the extensions presented in the papers "Spectral Densities, Structured Noise and Ensemble Averaging within Open
Quantum Dynamics" (add a link here) and "[Machine-learned correction to ensemble-averaged wave packet dynamics]([https://link-url-here.org](https://doi.org/10.1063/5.0166694))" .

## Features

- Population (and Coherence) Dynamics 

- Absorption Spectra

- Noise generation following spectral densities, to generate time dependent Hamiltonians.

- Thermal corrections - TNISE and MLNISE: Documentation and Code on how to train and use MLNISE will follow soon. 

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
We Request the Attributions to be in the form of citing the respective papers.

## Future plans

- Include documented support for MLNISE training

- Optimize memory usage: right now too many intermediate results are kept in memory unnecessarily.

- Add Tutorials

- Add Support for Decentralized compute options with open MP or similar.

- Make the code more readable by splitting long functions into smaller ones and reduce the number of inputs.

- Migrate remaining NumPy based functions to PyTorch


