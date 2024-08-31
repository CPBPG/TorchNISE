# TorchNISE

`TorchNISE` is a python package with a fully differentiable pytorch implementation of the Numerical Integration of Schrödinger Equation (NISE) algorithm and the extensions presented in the paper "Spectral Densities, Structured Noise and Ensemble Averaging within Open
Quantum Dynamics" (add a link here) and "[Machine-learned correction to ensemble-averaged wave packet dynamics]([https://link-url-here.org](https://doi.org/10.1063/5.0166694))" .


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

```python
import torchnise

# include code here
```

or run the examples from the example folder
they include one example for population dynamics and absorption and one example for noise generation and spectral density reconstruction

The current state of the code is not stable and will be continued to be updated


