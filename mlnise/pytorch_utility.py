import numpy as np
import torch
import os
import uuid
import weakref
import tempfile
import warnings
import glob
from scipy.optimize import dual_annealing
import mlnise.units as units
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

    # Create a mask where the real part of the inner product is greater than the threshold
    mask = inner_product.real > eps

    # Calculate the square root of the inner product for renormalization
    sqrt_inner_product = torch.sqrt(inner_product.real)

    # Avoid division by zero by setting values where the mask is False to 1
    sqrt_inner_product[~mask] = 1.0

    # Renormalize phi
    phi = phi / sqrt_inner_product
    return phi


def matrix_logh(A: torch.Tensor, dim1: int = -1, dim2: int = -2, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Compute the Hermitian matrix logarithm of a square matrix or a batch of square matrices.
    It is the unique hermitian matrix logarithm see
    https://math.stackexchange.com/questions/4474139/logarithm-of-a-positive-definite-matrix

    Args:
        A (torch.Tensor): Input tensor with square matrices in the last two dimensions.
        dim1 (int): First dimension of the square matrices. Default is -1.
        dim2 (int): Second dimension of the square matrices. Default is -2.
        epsilon (float): Small value to add to the diagonal to avoid numerical issues.

    Returns:
        torch.Tensor: Matrix logarithm of the input tensor.
    """
    dim1 = dim1 % len(A.shape)  # Convert negative to positive indices
    dim2 = dim2 % len(A.shape)  # Convert negative to positive indices

    if dim1 == dim2:
        raise ValueError("dim1 and dim2 cannot be the same for batch trace")
    if A.shape[dim1] != A.shape[dim2]:
        raise ValueError(f"The input tensor must have square matrices in the specified dimensions. "
                         f"Dimension {dim1} has size {A.shape[dim1]} and dimension {dim2} has size {A.shape[dim2]}.")

    if dim1 != -1 or dim2 != -2:
        A = A.transpose(dim1, -1).transpose(dim2, -2)

    n = A.shape[-1]
    identity = torch.eye(n, dtype=A.dtype, device=A.device).view(*([1] * (A.dim() - 2)), n, n)
    A = A + epsilon * identity

    e, v = torch.linalg.eigh(A)
    e = e.to(dtype=v.dtype)
    log_A = torch.matmul(torch.matmul(v, torch.diag_embed(torch.log(e))), v.conj().transpose(-2, -1))

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
        raise ValueError(f"The tensor does not have the same dimension on the trace dimensions. "
                         f"Dimension {dim1} has size {A.shape[dim1]} and dimension {dim2} has size {A.shape[dim2]}.")

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
    storage = torch.storage.UntypedStorage.from_file(filename, shared=True, nbytes=nbytes)

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
    Warning: If the tensor's storage does not have a filename attribute, (usually
             because pytorch version is less than 2.2) it can not be determined 
             if the tensor is memory mapped. It is assumed that it is not.
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


def objective_oscil(tau_oscscale_oscstrength, i,delta_t,n,time_array,U):
     tau,osc_scale,osc_strength=tau_oscscale_oscstrength
     osc_scale=osc_scale
     osc_strength=osc_strength
     amplitude=osc_strength
     vert_shift= (1-osc_strength)
     fit = (1 - 1/n) * torch.exp(-time_array/tau)*(amplitude*torch.cos(2*np.pi*time_array/osc_scale)+vert_shift) + 1/n
     population = torch.mean(torch.abs(U[:,:,i,i])**2, dim=0).real  
     mse = torch.mean(torch.abs((population[1:] - fit[1:])))
     return mse.item()
def objective_oscil_freq(tau,osc_scale, i,delta_t,n,time_array,U):
     
     osc_strength=1
     amplitude=osc_strength
     osc_scale=osc_scale.unsqueeze(1)
     osc_scale=osc_scale/(2*np.pi)
     time_array=time_array.clone().unsqueeze(0)
     vert_shift= (1-osc_strength)
     fit = (1 - 1/n) * torch.exp(-time_array/tau)*(amplitude*torch.cos(time_array/osc_scale)+vert_shift) + 1/n
     population = torch.mean(torch.abs(U[:,:,i,i])**2, dim=0).real  
     population =population.unsqueeze(1)
     mse = torch.mean(torch.abs((population[:,1:] - fit[:,1:])),dim=1)
     return mse
    
 
def objective_mae(tau, i,delta_t,n,time_array,U):
    fit = (1 - 1/n) * torch.exp(-time_array/tau) + 1/n
    population = torch.mean(torch.abs(U[:,:,i,i])**2, dim=0).real
    mse = torch.mean(torch.abs((population[1:] - fit[1:])))
    return mse.item() 
def objective(tau, i,delta_t,n,time_array,U):
    fit = (1 - 1/n) * torch.exp(-time_array/tau) + 1/n
    population = torch.mean(torch.abs(U[:,:,i,i])**2, dim=0).real
    mse = torch.mean(((population[1:] - fit[1:]))**2)
    return mse.item() 
def objective_reverse_cummax(tau, i,delta_t,n,time_array,U):
    fit = (1 - 1/n) * torch.exp(-time_array/tau) + 1/n
    population = torch.mean(torch.abs(U[:,:,i,i])**2, dim=0).real
    
    reverse_cummax, _ = torch.flip(population, [0]).cummax(dim=0)

    # Reverse the cumulative maximum back to the original order
    reverse_cummax = torch.flip(reverse_cummax, [0])

    # Select values where the element is equal to its reverse cumulative maximum
    selected_values = population == reverse_cummax
    
    times=time_array[selected_values]
    weights=times[1:]-times[:-1]

    mse = torch.mean(((population[selected_values][1:] - fit[selected_values][1:])*weights) ** 2)
    return mse.item()

def estimate_lifetime(U, delta_t,method="oscillatory_fit"):
    # Number of states and time steps
    n = U.shape[2]
    timesteps = U.shape[1]
    time_array = torch.arange(timesteps, device=U.device, dtype=torch.float) * delta_t

    # Lifetime estimates
    lifetimes = torch.zeros(n, device=U.device, dtype=torch.float)

    # Estimate lifetime for each state
    for i in range(n): 
        func = lambda tau: objective(tau, i,delta_t,n,time_array,U)
        tolerance = 0.1 * delta_t
        if method in ["simple_fit", "reverse_cummax","simple_fit_mae"]:
            if method == "simple_fit":
                func = lambda tau: objective(tau, i,delta_t,n,time_array,U)
            elif method == "reverse_cummax":
                func = lambda tau: objective_reverse_cummax(tau, i,delta_t,n,time_array,U)
            elif method == "simple_fit_mae":
                func = lambda tau: objective_mae(tau, i,delta_t,n,time_array,U)
            #tau_opt = golden_section_search(func, delta_t, 100 * timesteps * delta_t, tolerance)
            lw=[delta_t]
            up=[100 * timesteps * delta_t]
            tau_opt= dual_annealing(func,bounds=list(zip(lw,up)),minimizer_kwargs={"method":"BFGS"}).x
            print(f"lifetime state {i+1} {tau_opt} fs")
            
        elif method in ["oscillatory_fit"]:
            func = lambda tau: objective_reverse_cummax(tau, i,delta_t,n,time_array,U)
            tau_0 = golden_section_search(func, delta_t, 100 * timesteps * delta_t, tolerance)                  
            x0=[tau_0,tau_0,0.5]
            #tau_opt,osc_scale_opt,osc_strength_opt = minimize(func,torch.tensor([100*delta_t,100*delta_t,0]),tol=tolerance).x
            lw=[delta_t,delta_t,0]
            up=[100 * timesteps * delta_t,0.5* timesteps * delta_t,1]
            #minres=1e99
            
            #for x0 in [[tau_0,tau_0,0.5],[tau_0,tau_1,0.5],[tau_1,tau_0,0.5],[tau_1,tau_1,0.5]]:
            #    print(x0)
            res = dual_annealing(objective_oscil,bounds=list(zip(lw,up)),args=(i,delta_t,n,time_array,U,),minimizer_kwargs={"method":"BFGS"},x0=x0,maxiter=500)
                #if res.fun<minres:
            tau_opt,osc_scale_opt,osc_strength_opt = res.x
                #print(res.x)
            
            #tau_opt,osc_scale_opt,osc_strength_opt = basinhopping(func,x0,minimizer_kwargs={"method":"BFGS"},stepsize=50*delta_t).x
            #tau_opt,osc_scale_opt,osc_strength_opt = differential_evolution(objective,bounds=list(zip(lw,up)),args=(i,delta_t,n,time_array,U,),tol=tolerance,popsize=100).x
            #print(f"lifetime state {i+1} {tau_opt} fs, with oscillation strength {osc_strength_opt} and oscillation length  {osc_scale_opt} fs")
            
            

              
        
        lifetimes[i] = torch.tensor(tau_opt)
        #print(f"lifetime state {i+1} {tau_opt} fs")
    return lifetimes

def blend_and_normalize_populations(pop1, pop2, lifetimes, delta_t):
    timesteps, sites = pop1.shape
    time_array = torch.arange(timesteps, dtype=torch.float) * delta_t

    blended_population = torch.zeros_like(pop1)

    for site in range(sites):
        # Exponential blending factor
        lifetime = lifetimes[site]
        blending_factor = torch.exp(-time_array / lifetime)

        # Blend populations
        blended_population[:, site] = blending_factor * pop1[:, site] + (1 - blending_factor) * pop2[:, site]

    # Renormalize the population at each timestep
    total_population = torch.sum(blended_population, dim=1, keepdim=True)
    normalized_population = blended_population / total_population

    return normalized_population       

def averaging(population,averiging_type,lifetimes=None,step=None,coherence=None,weight=None):
    averiging_types= ["standard", "boltzmann", "interpolated"]
    
    if averiging_type.lower() not in averiging_types:
        raise NotImplementedError(f"{averiging_type} not implemented only {averiging_types} are available")
    if averiging_type.lower() in ["interpolated","boltzmann"]:
        assert not coherence == None
        if averiging_type.lower()=="interpolated":
            assert not lifetimes==None
            assert not step==None
    if weight==None:
        weight=1
        weight_coherence=1
    else:
        weight,weight_coherence = reshape_weights(weight,population,coherence)
    
    meanres_orig=population_return=torch.mean(population*weight,dim=0)
    if coherence == None:
        meanres_coherence = None
    else:
        meanres_coherence = torch.mean(coherence*weight_coherence,dim=0)
    if averiging_type.lower()=="standard":
        population_return=meanres_orig
        coherence_return=meanres_coherence
    else:
        logres_Matrix=torch.mean(matrix_logh(coherence)*weight_coherence,dim=0)      
        meanExp_Matrix=torch.linalg.matrix_exp(logres_Matrix)
        #renormalize
        meanres_Matrix=meanExp_Matrix/batch_trace(meanExp_Matrix).unsqueeze(1).unsqueeze(1)  
        meanres_Matrix[0]=meanres_coherence[0]
        coherence_return=meanres_Matrix
        if averiging_type.lower()=="boltzmann":          
            population_return=torch.diagonal(meanres_Matrix, dim1=-2, dim2=-1)
        else:
            population_return=blend_and_normalize_populations(meanres_orig,torch.diagonal(meanres_Matrix, dim1=-2, dim2=-1),lifetimes,step)               

    
    return population_return, coherence_return

def reshape_weights(weight,population,coherence):
    weight=weight/torch.sum(weight)*len(weight) #normalize      
    
    if not coherence == None:
        weight_shape_coherence=list(coherence.shape)
        for i in range(1,len(weight_shape_coherence)): 
            weight_shape_coherence[i]=1
        weight_coherence=weight.reshape(weight_shape_coherence)
    else:
        weight_coherence=1
    weight_shape=list(population.shape)
    for i in range(1,len(weight_shape)): 
        weight_shape[i]=1        
    weight_reshaped=weight.reshape(weight_shape) #broadcasting weights to multiply with all elements
    return weight_reshaped, weight_coherence

def absorption_time_domain(U,mu):
    pigments=U.shape[-1]
    realizations=U.shape[0]
    timesteps=U.shape[1]

    if len(mu.shape)==2: ## time dependence is not supplied
        mu=np.tile(mu,(realizations,timesteps,1,1)) ##copy the same mu vector along the time domain

    absorption_time_domain=0
    damp=1#np.exp(-np.arange(0,timesteps)/1000)
    for xyz in range(0,3):
        for real in range(0,realizations):
            for m in range (0,pigments):
                for n in range(0,pigments):

                    absorption_time_domain+=U[real,:,m,n]*mu[real,:,m,xyz]*mu[real,0,n,xyz]/realizations*damp
    return absorption_time_domain

def smooth_damp_to_zero(f_init,start,end):
    f=f_init.copy()
    f[end:]=0
    def expdamp_helper(a):
        x=a.copy()
        x[x<=0]=0
        x[x>0]=np.exp(-1/x[x>0])
        return x
    damprange=np.arange(end-start,dtype=float)[::-1]/(end-start)
    f[start:end]=f[start:end]*expdamp_helper(damprange)/(expdamp_helper(damprange)+expdamp_helper(1-damprange))
    return f

def absorb_time_to_freq(absorb_time,pad,total_time,dt):
    absorb=np.pad(absorb_time,(0,pad))
    absorb=smooth_damp_to_zero(absorb,int(total_time//dt-total_time//(dt*10)),int(total_time/dt)-1)
    absorb_f=np.fft.fftshift(np.fft.fft(absorb))
    hbar = units.hbar
    x_axis=-hbar*2*np.pi*np.fft.fftshift(np.fft.fftfreq(int((total_time+dt)/dt)+pad, d=dt))
    absorb_f=(absorb_f.real-absorb_f.real[0])/np.max(absorb_f.real-absorb_f.real[0])
    return absorb_f,x_axis

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
