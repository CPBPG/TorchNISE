"""
This file contains some utility functions
"""
import os
import uuid
import weakref
import tempfile
import warnings
import glob
import numpy as np
import torch

# Utility Functions
def renorm(phi: torch.Tensor, eps: float = 1e-8, dim: int = -1) -> torch.Tensor:
    """
    Renormalize a batch of wavefunctions.

    Args:
        phi (torch.Tensor): Batch of wavefunctions to be renormalized.
        eps (float): Small threshold to avoid division by zero.
        dim (int): Dimension along which the wavefunctions are stored. Default is -1.

    Returns:
        torch.Tensor: Renormalized wavefunctions.
    """
    # Calculate the inner product along the specified dimension
    inner_product = torch.sum(phi.conj() * phi, dim=dim, keepdim=True)

    # Create a mask where the real part of the inner product is greater than
    # the threshold
    mask = inner_product.real > eps

    # Calculate the square root of the inner product for renormalization
    sqrt_inner_product = torch.sqrt(inner_product.real)

    # Avoid division by zero by setting values where the mask is False to 1
    sqrt_inner_product[~mask] = 1.0

    # Renormalize phi
    phi = phi / sqrt_inner_product
    return phi


def matrix_logh(A: torch.Tensor, dim1: int = -1, dim2: int = -2,
                epsilon: float = 1e-5) -> torch.Tensor:
    """
    Compute the Hermitian matrix logarithm of a square matrix or a batch of 
    square matrices.It is the unique hermitian matrix logarithm see
    math.stackexchange.com/questions/4474139/logarithm-of-a-positive-definite-matrix

    Args:
        A (torch.Tensor): Input tensor with square matrices in the last two
            dimensions.
        dim1 (int): First dimension of the square matrices. Default is -1.
        dim2 (int): Second dimension of the square matrices. Default is -2.
        epsilon (float): Small value to add to the diagonal to avoid numerical
            issues.

    Returns:
        torch.Tensor: Matrix logarithm of the input tensor.
    """
    dim1 = dim1 % len(A.shape)  # Convert negative to positive indices
    dim2 = dim2 % len(A.shape)  # Convert negative to positive indices

    if dim1 == dim2:
        raise ValueError("dim1 and dim2 cannot be the same for batch trace")
    if A.shape[dim1] != A.shape[dim2]:
        raise ValueError(f"The input tensor must have square "
                         "matrices in the specified dimensions. "
                         f"Dimension {dim1} has size {A.shape[dim1]} and "
                         f"dimension {dim2} has size {A.shape[dim2]}.")

    if dim1 != -1 or dim2 != -2:
        A = A.transpose(dim1, -1).transpose(dim2, -2)

    n = A.shape[-1]
    identity = torch.eye(n, dtype=A.dtype,
                         device=A.device).view(*([1] * (A.dim() - 2)), n, n)
    A = A + epsilon * identity

    e, v = torch.linalg.eigh(A)
    e = e.to(dtype=v.dtype)
    log_A = torch.matmul(torch.matmul(v,
                torch.diag_embed(torch.log(e))), v.conj().transpose(-2, -1))

    return log_A

#thx https://discuss.pytorch.org/t/how-to-calculate-matrix-trace-in-3d-tensor/132435
def batch_trace(A: torch.Tensor, dim1: int = -1, dim2: int = -2) -> torch.Tensor:
    """
    Compute the batch trace of a tensor along specified dimensions.

    Args:
        A (torch.Tensor): Input tensor.
        dim1 (int): First dimension to compute trace along.
        dim2 (int): Second dimension to compute trace along.

    Returns:
        torch.Tensor: Trace of the input tensor along the specified dimensions.
    """
    dim1 = dim1 % len(A.shape)  # Convert negative to positive indices
    dim2 = dim2 % len(A.shape)  # Convert negative to positive indices

    if dim1 == dim2:
        raise ValueError("dim1 and dim2 cannot be the same for batch trace")
    if A.shape[dim1] != A.shape[dim2]:
        raise ValueError(f"The tensor does not have the same dimension on the "
                         "trace dimensions. "
                         f"Dimension {dim1} has size {A.shape[dim1]} and "
                         f"dimension {dim2} has size {A.shape[dim2]}.")

    return torch.diagonal(A, offset=0, dim1=dim1, dim2=dim2).sum(dim=-1)

def tensor_to_mmap(tensor) -> torch.Tensor:
    """
    A custom function to create memory-mapped tensors.

    This function handles the creation of memory-mapped tensors, ensuring that the
    data is efficiently managed and temporary files are cleaned up properly.
    The mmaped tensors are always on cpu.
    
    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Memory-mapped tensor.
    """
    if is_memory_mapped(tensor):
        return tensor


    # Create a tensor using the storage
    mmap_tensor = create_empty_mmap_tensor(shape=tensor.shape, dtype=tensor.dtype)
    mmap_tensor.copy_(tensor)


    return mmap_tensor

def create_empty_mmap_tensor(shape, dtype=torch.float32) -> torch.Tensor:
    """
    A custom function to create memory-mapped tensors.

    This function handles the creation of memory-mapped tensors, ensuring that the
    data is efficiently managed and temporary files are cleaned up properly.
    The mmaped tensors are always on cpu.
    
    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Memory-mapped tensor.
    """


    # Generate a unique filename in a temporary directory
    temp_dir = tempfile.gettempdir()
    filename = os.path.join(temp_dir, f'{uuid.uuid4().hex}.bin')

    # Calculate the number of bytes needed
    nbytes= torch.Size(shape).numel() * torch.tensor([], dtype=dtype).element_size()

    # Create untyped storage from the file
    storage = torch.storage.UntypedStorage.from_file(filename, shared=True,
                                                     nbytes=nbytes)

    # Create a tensor using the storage
    mmap_tensor = torch.tensor(storage,dtype=dtype, device="cpu").view(shape)

    # Setup to automatically cleanup the file when the tensor is garbage collected
    weakref.finalize(mmap_tensor, delete_file, filename)

    return mmap_tensor

def delete_file(filename: str) -> None:
    """
    Delete the temporary file.

    Args:
        filename (str): Path to the file to be deleted.
    """
    if os.path.exists(filename):
        try:
            os.remove(filename)
        except OSError as e:
            print(f"Error deleting file {filename}: {e}")


def is_memory_mapped(tensor) -> bool:
    """
    Check if a given PyTorch tensor is memory-mapped from a file.

    A memory-mapped tensor is created by mapping a file into the tensor's storage.
    This function inspects the storage of the given tensor to determine if it
    was created from a memory-mapped file.

    Parameters:
        tensor (torch.Tensor): The PyTorch tensor to check.

    Returns:
        bool: True if the tensor is memory-mapped, False otherwise.

    Raises:
        Warning: If the tensor's storage does not have a filename attribute, 
            (usually because pytorch version is less than 2.2) it can not be 
            determined if the tensor is memory mapped. It is assumed that it 
            is not.
    """
    storage = tensor.untyped_storage()
    if not hasattr(storage, 'filename'):
        warnings.warn("The tensor's storage does not have a filename attribute."
                      " Can't determine if the tensor is Memory mapped assuming"
                      " it is not.")
        return False
    return storage.filename is not None

def clean_temp_files():
    """
    Remove all temporary .bin files.
    """
    temp_files = glob.glob(os.path.join(tempfile.gettempdir(), '*.bin'))
    print("Deleting temporary files:", temp_files)
    for file in temp_files:
        os.remove(file
    )

def golden_section_search(func, a, b, tol):
    """
    Perform a golden section search to find the minimum of a unimodal function
    on a closed interval [a, b].

    Args:
        func (callable): The unimodal function to minimize.
        a (float): The lower bound of the search interval.
        b (float): The upper bound of the search interval.
        tol (float): The tolerance for stopping the search. The search stops when
                     the interval length is less than this value.

    Returns:
        float: The point at which the function has its minimum within the
        interval [a, b].
    """
    golden_ratio = (1 + 5 ** 0.5) / 2

    c = b - (b - a) / golden_ratio
    d = a + (b - a) / golden_ratio

    while abs(c - d) > tol:
        if func(c) < func(d):
            b = d
        else:
            a = c

        c = b - (b - a) / golden_ratio
        d = a + (b - a) / golden_ratio

    return (b + a) / 2

def smooth_damp_to_zero(f_init, start, end):
    """
    Smoothly damp a segment of an array to zero using an exponential damping function.

    Args:
        f_init (numpy.ndarray): Initial array to be damped.
        start (int): Starting index of the segment to damp.
        end (int): Ending index of the segment to damp.

    Returns:
        numpy.ndarray: Array after applying the damping.
    """
    f = f_init.copy()
    f[end:] = 0

    def expdamp_helper(a):
        x = a.copy()
        x[x <= 0] = 0
        x[x > 0] = np.exp(-1 / x[x > 0])
        return x

    damprange = np.arange(end - start, dtype=float)[::-1] / (end - start)
    f[start:end] = (f[start:end] * expdamp_helper(damprange) /
                    (expdamp_helper(damprange) + expdamp_helper(1 - damprange)))
    return f

"""
# Check if torch.Tensor already has the to_mmap method
if hasattr(torch.Tensor, 'to_mmap'):
    print("torch.Tensor.to_mmap already exists.")
else:
    def to_mmap(self: torch.Tensor) -> torch.Tensor:
        """""""
        Extends the torch.Tensor class: torch.Tensor.to_mmap = to_mmap
        Convert the tensor to a memory-mapped tensor.

        Returns:
            torch.Tensor: Memory-mapped tensor.
        """""""
        return tensor_to_mmap(self)

    # Adding the method to the torch.Tensor class
    torch.Tensor.to_mmap = to_mmap
    #print("torch.Tensor.to_mmap has been added.")"""
