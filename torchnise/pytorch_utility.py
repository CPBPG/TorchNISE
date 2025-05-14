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
import h5py

numpy_to_torch_dtype_dict = {
        np.bool_       : torch.bool,
        np.uint8      : torch.uint8,
        np.int8       : torch.int8,
        np.int16      : torch.int16,
        np.int32      : torch.int32,
        np.int64      : torch.int64,
        np.float16    : torch.float16,
        np.float32    : torch.float32,
        np.float64    : torch.float64,
        np.complex64  : torch.complex64,
        np.complex128 : torch.complex128
    }

    # Dict of torch dtype -> NumPy dtype
torch_to_numpy_dtype_dict = {value : key for (key, value) in numpy_to_torch_dtype_dict.items()}



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

    # Create a mask where the real part of the inner product is greater than eps
    mask = inner_product.real > eps

    # Safe sqrt with masking
    sqrt_inner_product = torch.sqrt(inner_product.real)
    sqrt_inner_product = torch.where(mask, sqrt_inner_product, torch.ones_like(sqrt_inner_product))

    # Renormalize phi
    phi_new = phi / sqrt_inner_product
    return phi_new

def free_vram(device=0):
    """
    Returns the free memory in bytes of the specified GPU device.
    Args:
        device (int or string): The index of the GPU device or 
                                "cuda" to get the Default "cuda device".
                                Default is 0.
    Returns:
        int: Free memory in bytes.
    """
    total    = torch.cuda.get_device_properties(device).total_memory
    reserved = torch.cuda.memory_reserved(device)
    allocated= torch.cuda.memory_allocated(device)
    return total - (reserved + allocated)

def tensor_bytes(shape, dtype):
    """
    Calculate the number of bytes required to store a tensor with 
    the given shape and dtype.
    Args:
        shape (tuple): Shape of the tensor.
        dtype (torch.dtype): Data type of the tensor.
    Returns:
        int: Number of bytes required to store the tensor.
    """
    # number of elements × bytes per element
    return np.prod(shape) * torch.tensor([], dtype=dtype).element_size()


def weighted_mean(tensor: torch.Tensor, weights: torch.Tensor, dim=0) -> torch.Tensor:
    """
    Compute the weighted mean of 'tensor' along dimension 'dim' using
    'weights' as the weights for each slice along 'dim'.

    Args:
        tensor (torch.Tensor): Input data of shape (..., N, ...)
        weights (torch.Tensor): 1D weights of shape (N,).
        dim (int): The dimension along which to apply the weights.

    Returns:
        torch.Tensor: Weighted average along dimension 'dim'.
    """
    # Ensure weights is float and on the same device
    weights = torch.tensor(weights,dtype=tensor.dtype, device=tensor.device)

    # Sum of all weights
    w_sum = weights.sum()

    # Reshape weights to broadcast along the specified dimension
    # For instance, if dim=0 and weights.size= (N,),
    # we reshape weights to (N, 1, 1, ...)
    shape = [1]*tensor.ndim
    shape[dim] = -1  # place weights along 'dim'
    w_view = weights.reshape(*shape)

    # Weighted sum along 'dim'
    weighted_sum = (tensor * w_view).sum(dim=dim)
    return weighted_sum / w_sum


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


class H5Tensor:
    """
    A class for handling PyTorch tensors stored in HDF5 files.

    This class provides a way to work with large tensors that do not fit into memory.
    
    """
    def __init__(self, data=None, h5_filepath=None, requires_grad=False,
                 dtype=torch.float,shape=None):
        """
        Flexible constructor for H5Tensor.
        
        Args:
            data: Can be one of the following:
                - A list, NumPy array, or PyTorch tensor for direct initialization.
                - An existing H5Tensor for copying.
            dataset_name: If loading from an HDF5 file, the name of the dataset.
            h5_filepath: If loading from an HDF5 file, the file path.
            device: The PyTorch device for the tensor ('cpu' or 'cuda').
            requires_grad: If the tensor should track gradients.
            shape: shape of the tensor if no data is provided
        """
        self.device = "cpu"
        self.requires_grad = requires_grad
        self.dtype=dtype
        self.h5_filepath = h5_filepath
        self.shape=shape


        if isinstance(data, H5Tensor):
            # Copy constructor for H5Tensor
            data = data.to_tensor()
            self._save_to_hdf5(data)

        elif isinstance(data, (list, np.ndarray, torch.Tensor)):
            # Initialize from tensor-like data (list, NumPy array, PyTorch tensor)        
            # Save data to HDF5
            if isinstance(data, (list, np.ndarray)):
                data=torch.tensor(data,requires_grad=self.requires_grad,dtype=self.dtype)
            else:
                self.requires_grad=data.requires_grad
                self.dtype=data.dtype
            self.shape=data.shape  
            self._save_to_hdf5(data)
        elif shape is not None:
            self._save_to_hdf5(data=None)
        elif os.path.exists(h5_filepath):
            # Load data from HDF5 file
            with h5py.File(self.h5_filepath, 'r') as f:
                if "data" in f:
                    self.shape = f["data"].shape
                if "grad" in f:
                    self.requires_grad=True
        else:
            raise ValueError("""Invalid constructor arguments. Must provide either an HDF5
                              file with dataset, an H5Tensor, or tensor-like data.""")

    def _save_to_hdf5(self, data):
        """Helper method to save data to HDF5 file.""" 
        if not self.h5_filepath:
            raise ValueError("No HDF5 file path provided to save data.")

        with h5py.File(self.h5_filepath, 'w') as f:
            if data is not None:
                # Save provided data to HDF5
                f.create_dataset("data", data=data.cpu().detach().numpy())
            else:
                if self.dtype.is_complex:
                    f.create_dataset("data", shape=self.shape,
                                     dtype=torch_to_numpy_dtype_dict[self.dtype],
                                     fillvalue=0+0j)
                else:
                    f.create_dataset("data", shape=self.shape,
                                     dtype=torch_to_numpy_dtype_dict[self.dtype],
                                     fillvalue=0)
            if self.requires_grad:
                if data.grad is not None:
                    f.create_dataset("grad", data=data.grad.cpu().detach().numpy())
                else:
                    if self.dtype.is_complex:
                        f.create_dataset("grad", shape=self.shape,
                                         dtype=torch_to_numpy_dtype_dict[self.dtype],
                                         fillvalue=0+0j)
                    else:
                        f.create_dataset("grad", shape=self.shape,
                                         dtype=torch_to_numpy_dtype_dict[self.dtype],
                                         fillvalue=0)
    def __setitem__(self, index, value):
        """Set a slice of the data in HDF5."""

        with h5py.File(self.h5_filepath, 'a') as f: # Open HDF5 file in append mode
                                                    # to allow modifications
            if isinstance(value, H5Tensor):
                value=value.to_tensor()
            if isinstance(value, torch.Tensor):
                # Write the new data to the HDF5 file
                f["data"][index] = value.to(self.dtype).cpu().detach().numpy()
                if self.requires_grad and value.grad is not None:
                    f["grad"][index] = value.grad.to(self.dtype).cpu().detach().numpy()
            else:
                # Write the new data to the HDF5 file
                f["data"][index] = np.array(value,
                                            dtype=torch_to_numpy_dtype_dict[self.dtype])


    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # Convert H5Tensor to a regular torch.Tensor for the operation
        def convert(x):
            return x.to_tensor() if isinstance(x, H5Tensor) else x

        converted_args = tuple(map(convert, args))
        converted_kwargs = {k: convert(v) for k, v in kwargs.items()}

        # Call the original function with the converted arguments
        return func(*converted_args, **converted_kwargs)

    def __getattr__(self, name):
        """
        Catch attribute access and forward it to the PyTorch tensor.
        This method is invoked when an attribute (like .reshape or .transpose)
        is accessed.
        """
        # If the method is called (e.g., h5tensor.reshape), it will load the
        # full tensor and apply the corresponding method.
        if hasattr(torch.Tensor, name):
            tensor = self.to_tensor()  # Load data from HDF5 and convert to a PyTorch tensor
            attr = getattr(tensor, name)

            # If it's callable (a method like .reshape()), return a function that applies it
            if callable(attr):
                def method_wrapper(*args, **kwargs):
                    return attr(*args, **kwargs)
                return method_wrapper
            else:
                # Otherwise, return the attribute directly (like .shape)
                return attr
        else:
            raise AttributeError(f"'H5Tensor' object has no attribute '{name}'")

    def __getitem__(self, index):
        with h5py.File(self.h5_filepath, 'r') as f:
            data_slice = torch.tensor(f["data"][index], device=self.device,dtype=self.dtype
                                      ,requires_grad=self.requires_grad)
            if self.requires_grad:
                data_slice.grad= torch.tensor(f["grad"][index],dtype=self.dtype)
        return data_slice


    def to_tensor(self):
        """
        Convert the H5Tensor to a PyTorch tensor.
        If the data is stored in HDF5, it will load it into memory.
        """

        # Load entire dataset from HDF5 file
        with h5py.File(self.h5_filepath, 'r') as f:
            data = torch.tensor(f["data"][:], device=self.device,dtype=self.dtype,
                                requires_grad=self.requires_grad)
            if self.requires_grad:
                data.grad= torch.tensor(f["grad"][:],dtype=self.dtype)
        return data


    def __len__(self):
        return self.shape[0]

    def to(self, device):
        if not device=="cpu":
            warnings.warn("H5Tensor can only be on CPU. Using .to_tensor().to{device} " \
                          "to load data into memory of specified device as a regular"
                          "torch.tensor.")
            return self.to_tensor().to(device)
        else:    
            return self

    def __repr__(self):
        if self.h5_filepath:
            return f"""H5Tensor(HDF5 file: {self.h5_filepath}, dataset: {self.dataset_name},
            shape={self.shape}, device={self.device}, requires_grad={self.requires_grad})"""
        else:
            return f"""H5Tensor(shape={self.shape}, device={self.device},
            requires_grad={self.requires_grad})"""

    # Implementing operators
    def _apply_op(self, torch_op, other):
        if isinstance(other, H5Tensor):
            other = other.to_tensor()
        return torch_op(self.to_tensor(), other)

    def _apply_op_reverse(self, torch_op, other):
        if isinstance(other, H5Tensor):
            other = other.to_tensor()
        return torch_op(other, self.to_tensor())

    def _in_place_op(self, torch_op, other):
        if isinstance(other, H5Tensor):
            other = other.to_tensor()
        tensor = self.to_tensor()
        tensor = torch_op(tensor, other)
        with h5py.File(self.h5_filepath, 'a') as f:
            # Write the modified data back to the HDF5 file
            f["data"][:] = tensor.cpu().detach().numpy()

            # If requires_grad is enabled, handle gradients as well
            if self.requires_grad and tensor.grad is not None:
                f["grad"][:] = tensor.grad.cpu().detach().numpy()
        return self

    # Overriding binary operators
# Binary Arithmetic Operations
    def __add__(self, other):
        return self._apply_op(torch.add, other)

    def __radd__(self, other):
        return self._apply_op(torch.add, other)

    def __sub__(self, other):
        return self._apply_op(torch.sub, other)

    def __rsub__(self, other):
        return self._apply_op_reverse(torch.sub, other)

    def __mul__(self, other):
        return self._apply_op(torch.mul, other)

    def __rmul__(self, other):
        return self._apply_op(torch.mul, other)

    def __truediv__(self, other):
        return self._apply_op(torch.div, other)

    def __rtruediv__(self, other):
        return self._apply_op_reverse(torch.div, other)

    def __floordiv__(self, other):
        return self._apply_op(torch.floor_divide, other)

    def __rfloordiv__(self, other):
        return self._apply_op_reverse(torch.floor_divide, other)

    def __mod__(self, other):
        return self._apply_op(torch.remainder, other)

    def __rmod__(self, other):
        return self._apply_op_reverse(torch.remainder, other)

    def __pow__(self, other):
        return self._apply_op(torch.pow, other)

    def __rpow__(self, other):
        return self._apply_op_reverse(torch.pow, other)

    def __matmul__(self, other):
        return self._apply_op(torch.matmul, other)

    def __rmatmul__(self, other):
        return self._apply_op_reverse(torch.matmul, other)

    # In-place Arithmetic Operations
    def __iadd__(self, other):
        return self._in_place_op(torch.add, other)

    def __isub__(self, other):
        return self._in_place_op(torch.sub, other)

    def __imul__(self, other):
        return self._in_place_op(torch.mul, other)

    def __itruediv__(self, other):
        return self._in_place_op(torch.div, other)

    def __ifloordiv__(self, other):
        return self._in_place_op(torch.floor_divide, other)

    def __imod__(self, other):
        return self._in_place_op(torch.remainder, other)

    def __ipow__(self, other):
        return self._in_place_op(torch.pow, other)

    def __imatmul__(self, other):
        return self._in_place_op(torch.matmul, other)

    # Comparison Operations
    def __eq__(self, other):
        return self._apply_op(torch.eq, other)

    def __ne__(self, other):
        return self._apply_op(torch.ne, other)

    def __lt__(self, other):
        return self._apply_op(torch.lt, other)

    def __le__(self, other):
        return self._apply_op(torch.le, other)

    def __gt__(self, other):
        return self._apply_op(torch.gt, other)

    def __ge__(self, other):
        return self._apply_op(torch.ge, other)

    # Unary Arithmetic Operations
    def __neg__(self):
        return -self.to_tensor()

    def __pos__(self):
        return +self.to_tensor()

    def __abs__(self):
        return torch.abs(self.to_tensor())

    # Bitwise Operators
    def __and__(self, other):
        return self._apply_op(torch.bitwise_and, other)

    def __rand__(self, other):
        return self._apply_op_reverse(torch.bitwise_and, other)

    def __or__(self, other):
        return self._apply_op(torch.bitwise_or, other)

    def __ror__(self, other):
        return self._apply_op_reverse(torch.bitwise_or, other)

    def __xor__(self, other):
        return self._apply_op(torch.bitwise_xor, other)

    def __rxor__(self, other):
        return self._apply_op_reverse(torch.bitwise_xor, other)

    def __invert__(self):
        return torch.bitwise_not(self.to_tensor())

    def __lshift__(self, other):
        return self._apply_op(torch.bitwise_left_shift, other)

    def __rlshift__(self, other):
        return self._apply_op_reverse(torch.bitwise_left_shift, other)

    def __rshift__(self, other):
        return self._apply_op(torch.bitwise_right_shift, other)

    def __rrshift__(self, other):
        return self._apply_op_reverse(torch.bitwise_right_shift, other)

    # In-place Bitwise Operations
    def __iand__(self, other):
        return self._in_place_op(torch.bitwise_and, other)

    def __ior__(self, other):
        return self._in_place_op(torch.bitwise_or, other)

    def __ixor__(self, other):
        return self._in_place_op(torch.bitwise_xor, other)

    def __ilshift__(self, other):
        return self._in_place_op(torch.bitwise_left_shift, other)

    def __irshift__(self, other):
        return self._in_place_op(torch.bitwise_right_shift, other)
