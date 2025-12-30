
import warnings
import numpy as np
import torch
import sys
import os

# Filter warnings to ensure we see them
warnings.simplefilter('always', DeprecationWarning)
warnings.simplefilter('always', FutureWarning)

print(f"Numpy version: {np.__version__}")

try:
    from torchnise import nise
    from torchnise import pytorch_utility
    from torchnise import units
    from torchnise import averaging_and_lifetimes
    from torchnise import absorption
    from torchnise import fft_noise_gen
    
    print("Successfully imported modules.")
    
    # Try to trigger some code paths
    # Create dummy NISEParameters
    params = nise.NISEParameters(
        dt=0.1,
        total_time=1.0,
        temperature=300
    )
    print("Created NISEParameters.")
    
except Exception as e:
    print(f"Error during import or execution: {e}")
    sys.exit(1)
