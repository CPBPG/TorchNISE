import numpy as np
import matplotlib.pyplot as plt
from mlnise.example_spectral_functions import spectral_Log_Normal_Lorentz,spectral_Lorentz,spectral_Drude
import functools
from mlnise.fft_noise_gen import noise_algorithm
import torch
import tqdm
import warnings
from scipy.optimize import nnls

cm_to_eV=1.23984E-4
def nnls_pyytorch(A,b):
    A_np = A.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    # Solve the NNLS problem
    x_nnls, _ = nnls(A_np, b_np)
    
    # Convert the result back to a PyTorch tensor
    x_nnls_torch = torch.from_numpy(x_nnls).to(A.device).to(A.dtype)
    return x_nnls_torch
    
#Paper Method Autocorrelation function 
def autocorrelation(noise,i,N): #matrix implementation of the autocorrelation calculation. 
    cor1=noise[:,:N-i]
    cor2=noise[:,i:]
    res=cor1*cor2
    C=np.mean(res,axis=1)
    return C #returns the Calculation for a certain noise matrix, and index i, i.e. C(t_i). Since it is a matrix, this is the ith-column of the total autocorrelation matrix

def Ccalc(noise,N,reals):
    C = np.zeros((reals,N))
    for i in range(int(N//2)): #tqdm(range(N//2)):#Calculating the autocorrelation for the whole matrix. Rows: realizations, Columns: different i's
        C[:,i] = autocorrelation(noise,i,N)
    return C # matrix with size reals x N

#Calculate expectation value
def expval(noise,N,reals):
    summation = Ccalc(noise,N,reals) #Matrix with size reals x N. each row contains C(t)_i for eachh realization i 
    return np.mean(summation,axis=0) #calculating the mean over the different realizations


def get_auto(noise):
    N = len(noise[0,:])
    reals = len(noise[:,0])
    C = expval(noise,N,reals) #autocorrelation array, len = N, with dt, gamma and strength
    auto = C[:int(N//2)] #only trusting half the values employed
    return auto


def SD_Reconstruct_FFT(auto,dt,T,hbar,k,minW=None,maxW=None,damping_type=None,cutoff=None,rescale=False):
    
    N_cut=len(auto)
    dw_t = 2*np.pi/(2*N_cut*dt)
    if maxW==None:
        maxW=N_cut*dw_t
    if minW==None:
        minW=0
        
    if maxW>N_cut*dw_t:
        print("Warning maxW bigger than maximum computable value")
    #    return

    t_axis=dt*np.arange(0, N_cut)        
    damping =np.ones_like(auto)
    
    if damping_type:
        if damping_type.lower()=="step":
            damping[int(cutoff)//dt:]=0
        elif damping_type.lower()=="gauss":
            damping=np.exp(-(t_axis/cutoff)**2)
        elif damping_type.lower()=="exp":
            damping=np.exp(-(t_axis/cutoff))
        else:
            raise NotImplementedError(f"Damping {damping_type} not available must be one of 'step', 'gauss' or 'exp'")
    
    auto_damp=auto*damping
    
    if rescale:
        auto_damp=auto_damp*np.mean(np.abs(auto))/np.mean(np.abs(auto_damp))       

    #Calculation of spectral density
    dw_t = 2*np.pi/(2*N_cut*dt) #creating dw in units of 1/fs. denominator: 2*N_cut = N
    full_w_t = np.arange(0,N_cut*dw_t,dw_t)
    minWindex = np.argmin(np.abs(full_w_t-minW)) #find the closest value to minW and maxW from the fourier axis
    maxWindex = np.argmin(np.abs(full_w_t-maxW))
    w_t = full_w_t[minWindex:maxWindex+1]  #array of frequencies to use in 1/fs. Max frequency is not necessarily this. Only ought to be less than nyquist frequency: dw*N/2
    x_axis = hbar*w_t # x-axis on the graph from paper  jcp12    
    reverse_auto=np.flip(auto_damp[1:-1])
    concat_auto=np.concatenate((auto_damp,reverse_auto))
    
    J_new=x_axis*np.fft.fft(concat_auto)[0:len(x_axis)].real*dt/(hbar*2*np.pi*k*T)
    return J_new, x_axis ,auto_damp

def tv_norm_2d(lambda_ij):
    """Calculate the total variation norm across both dimensions."""
    tv = torch.sum(torch.abs(lambda_ij[:, :-1] - lambda_ij[:, 1:])) + \
         torch.sum(torch.abs(lambda_ij[:-1, :] - lambda_ij[1:, :]))
    return tv

def objective_function(lambda_ij, A, C, sparcity_penalty, l1_norm_penalty, solution_penalty,negative_penalty,ljnorm_penalty, j):
    """Objective function with TV norm, L1 norm, and penalty for constraint violation."""
    tv = tv_norm_2d(lambda_ij)
    l1_norm = torch.sum(torch.abs(lambda_ij)) #torch.sum(torch.sqrt(torch.abs(lambda_ij)))
    lj_norm = torch.sum(torch.abs(lambda_ij)**j)
    negative_penalty= -torch.sum(lambda_ij)+torch.sum(torch.abs(lambda_ij))
    penalty = solution_penalty * 0.5 * torch.norm(A @ lambda_ij.flatten() - C) ** 2
    result= sparcity_penalty*tv + l1_norm_penalty*l1_norm +ljnorm_penalty*lj_norm + negative_penalty*negative_penalty+ penalty
    #print(f"penalty {100*penalty/result:.3f}% Variation {100*1/mu*tv/result:.3f}%  l1_norm {100*l1_norm/result:.3f}%")
    return result

def objective_function_no_penalty(lambda_ij, A, C, sparcity_penalty, l1_norm_penalty,negative_penalty,ljnorm_penalty, j):
    """Objective function with TV norm, L1 norm for constraint violation."""
    tv = tv_norm_2d(lambda_ij)
    l1_norm = torch.sum(torch.abs(lambda_ij)) #torch.sum(torch.sqrt(torch.abs(lambda_ij)))
    lj_norm = torch.sum(torch.abs(lambda_ij)**j)
    negative_penalty= -torch.sum(lambda_ij)+torch.sum(torch.abs(lambda_ij))
    result= sparcity_penalty*tv + l1_norm_penalty*l1_norm  +ljnorm_penalty*lj_norm + negative_penalty*negative_penalty
    return result

def optimize_lambda(A, C, sparcity_penalty, l1_norm_penalty, solution_penalty,negative_penalty, ljnorm_penalty, j, eta, max_iter=1000, tol=1e-6, lr=0.01, device='cuda',initial_guess=None):
    """Optimization loop using PyTorch."""
    
    num_k, num_i, num_j = A.shape
    A = A.reshape(num_k, num_i * num_j)
    # Initialize lambda_ij as a PyTorch tensor with gradients
    def A_transpose(y):
        return A.T @ y
    # Initialize lambda_ij as a PyTorch tensor with gradients
    lambda_ij = A_transpose(C) if initial_guess is None else initial_guess #torch.zeros((num_i, num_j)
    lambda_ij=lambda_ij.reshape(num_i,num_j)
    lambda_ij.requires_grad=True
    #else:
    #    lambda_ij = initial_guess.to(device)
    #    lambda_ij.requires_grad=True
        
    #rho_initial=torch.tensor(rho)
    # Use an optimizer, e.g., Adam
    #optimizer = torch.optim.Adam([lambda_ij], lr=lr)
    optimizer = torch.optim.LBFGS([lambda_ij],lr=lr,line_search_fn="strong_wolfe") #
    min_objective_value=torch.inf
    
    return_lambda=lambda_ij
    for iter in range(max_iter):
        optimizer.zero_grad()  # Clear previous gradients

        # Compute the objective function
        obj_value = objective_function(lambda_ij, A, C, sparcity_penalty, l1_norm_penalty, solution_penalty,negative_penalty,ljnorm_penalty, j)
        obj_value_no_penalty =  objective_function_no_penalty(lambda_ij, A, C,sparcity_penalty, l1_norm_penalty,negative_penalty,ljnorm_penalty, j)
        # Perform a backward pass to compute gradients
        obj_value.backward()
        def closure():
            optimizer.zero_grad()  # Clear previous gradients
    
            # Compute the objective function
            obj_value = objective_function(lambda_ij, A, C, sparcity_penalty, l1_norm_penalty, solution_penalty,negative_penalty,ljnorm_penalty, j)
            obj_value.backward()  # Perform a backward pass to compute gradients
            
            return obj_value
        # Gradient descent step
        optimizer.step(closure)
        #optimizer.step(closure)

        # Calculate the constraint violation
        with torch.no_grad():
            constraint_violation = torch.norm(A @ lambda_ij.flatten() - C).item()

        # If the constraint is violated, increase the penalty, else decrease
        #if constraint_violation > eta:  
        #    if rho<rho_initial:
        #        rho *= 1.001  # Increase the penalty parameter
        #    else:
        #        rho=rho_initial.item()
        #else:
        #    rho /= 1.001  # Decrease the penalty parameter

        # Print or log the objective value and constraint violation
        if iter%10==0:
            print(f"Iteration {iter}: Objective value = with pen {obj_value.item():.6f} without pen {obj_value_no_penalty.item():.6f}, Constraint violation = {constraint_violation> eta}, {constraint_violation:.6f}")
        if obj_value.item()<min_objective_value:
            min_objective_value=obj_value.item()
            no_improvement_iters=0
            return_lambda=lambda_ij
        else:
            no_improvement_iters +=1
        # Convergence check
        if obj_value_no_penalty.item() < tol and constraint_violation < eta:
            print("Converged")
            break
        if no_improvement_iters >20:
            print("no improvement for 20 iters")
            break

    return return_lambda

def optimize_lambda_nnls(A, b,initial_guess=None, max_iter=1000,lr=0.01):
    def A_transpose(y):
        return A.T @ y
    lambda_ij = A_transpose(b) if initial_guess is None else initial_guess
    
    lambda_ij=lambda_ij.detach().clone()
    lambda_ij.requires_grad=True
    return_lambda = lambda_ij 
    optimizer = torch.optim.LBFGS([lambda_ij],lr=lr,line_search_fn="strong_wolfe") 
    A @ torch.abs(lambda_ij)
    def closure():
        optimizer.zero_grad()
        loss = torch.norm(A @ torch.abs(lambda_ij) - b) ** 2
        loss.backward()
        return loss
    min_loss=torch.inf
    no_improvement_iters=0
    for i in range(max_iter):
        
        optimizer.step(closure)
        loss=closure().item()
        if i % 10 == 0:
            print(f'Step {i}, Loss: {loss}')
        if loss<min_loss:
            min_loss=loss
            no_improvement_iters=0
            return_lambda=lambda_ij
        else:
            no_improvement_iters +=1
        if no_improvement_iters >20:
            print("no improvement for 20 iters")
            break
    return torch.abs(return_lambda).detach()
    

def adjust_tensor_length(a, l):
    if len(a) == l:
        return a
    elif len(a) > l:
        # Trimming the tensor if it's longer than the desired length
        a = a[:l]
    else:
        # Padding the tensor with zeros if it's shorter than the desired length
        a = torch.cat([a, torch.zeros(l - len(a),device=a.device)])

    return a

def ensure_tensor_on_device(array, device='cuda',dtype=torch.float):
    if isinstance(array, np.ndarray):
        # Convert numpy array to PyTorch tensor and move to the specified device
        return torch.from_numpy(array).to(device).to(dtype)
    elif isinstance(array, torch.Tensor):
        # Ensure the tensor is on the correct device
        return array.to(device).to(dtype)
    else:
        raise TypeError("Input must be a numpy array or a PyTorch tensor.")

def sd_reconstruct_superresolution(auto, dt, T, hbar, k, sparcity_penalty=1, l1_norm_penalty=1, 
                                   solution_penalty=10000,negative_penalty=1,ljnorm_penalty=0 ,j=0.5, lr=0.01, max_iter=1000, eta=1e-7, 
                                   tol=1e-7, device='cuda', cutoff=None, 
                                   sample_frequencies=None, top_n=False,top_tresh=False, second_optimization=False,chunk_memory=1e9,auto_length_debias=None,auto_length_return=None):
    """
    Reconstruct the super-resolution spectral density from the autocorrelation function.

    Parameters:
        auto (torch.Tensor): Autocorrelation function.
        dt (float): Time step between autocorrelation points.
        T (float): Temperature.
        hbar (float): Reduced Planck constant.
        k (float): Boltzmann constant.
        sparcity_penalty (float): Penalty term for sparsity in the solution.
        l1_norm_penalty (float): L1 norm penalty for regularization.
        solution_penalty (float): Penalty for the solution norm.
        negative_penalty (float): Penalty for negative peaks.
        ljnorm_penalty (float): L_j norm penalty for regularization.
        j (float): j determining the L_j norm.
        lr (float): Learning rate for the optimization algorithm.
        max_iter (int): Maximum number of iterations for the optimization.
        eta (float): Regularization term for optimization.
        tol (float): Tolerance for convergence in the optimization.
        device (str): Device for computation ('cuda' or 'cpu').
        cutoff (float, optional): Cutoff for damping. Defaults to None.
        sample_frequencies (torch.Tensor, optional): Frequencies for sampling the spectral density. Defaults to None.
        top_n (int): If not False, only the top n coefficients are used. Defaults to False.
        top_tresh (float): Alternative to top_n chooses all coefficients above a treshold
        second_optimization (bool): If True, a second optimization step is performed using top n coefficients. Defaults to False.
        chunk_memory (float): The maximum amount of memory (in bytes) to use for each chunk. Defaults to 1GB.
        auto_length_debias (float): if not False: length of autocorrelation in fs for second optimizazion fitting, will be zero padded or cut
        auto_length_return (float): if not False: length of the returned autocorrelation inn fs

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Reconstructed spectral density, sampled frequencies, and super-resolved autocorrelation function.
    """
    auto = ensure_tensor_on_device(auto, device=device)
    auto_very_orig = auto.clone()
    if auto_length_return:
        auto_very_orig = adjust_tensor_length(auto, auto_length_return//dt)
    if auto_length_debias:
        auto = adjust_tensor_length(auto, auto_length_debias//dt)
    
    N = len(auto)
    N_cut = int(cutoff / dt) if cutoff else N // 2
    

    t_axis = dt * torch.arange(0, N_cut, device=device)
    t_axis_full = dt * torch.arange(0, N, device=device)
    t_axis_very_orig = dt *torch.arange(0, len(auto_very_orig), device=device)
    frequencies = torch.arange(0, 0.2, 0.00005, device=device) / hbar
    dampings = torch.arange(1, 200, 0.5, device=device) * cm_to_eV / hbar

    damping_term = torch.exp(-dampings[None, :] * t_axis[:, None])
    frequency_term = torch.cos(frequencies[None, :] * t_axis[:, None])

    A = damping_term[:, :, None] * frequency_term[:, None, :]

    auto_orig = auto.clone()
    auto = auto[:N_cut]

    lambda_ij = optimize_lambda(A, auto, sparcity_penalty, l1_norm_penalty, solution_penalty,
                                negative_penalty, ljnorm_penalty, j, eta, max_iter=max_iter, 
                                tol=tol, lr=lr, device=device)
    if top_n:
        if top_n>len(dampings) * len(frequencies):
            warnings.warn("top_n larger than total elements, using all available coefficients.")
            top_n=len(dampings) * len(frequencies)
        _, indices = torch.topk(lambda_ij.flatten(), top_n) #removed abs since we don't want negative peaks anyway
        
        if second_optimization:         
            original_indices = torch.unravel_index(indices, lambda_ij.shape)
            A_new = torch.zeros((len(t_axis_full), top_n), device=device)
            for k in range(top_n):
                i = original_indices[0][k]
                j = original_indices[1][k]
                A_new[:, k] = torch.exp(-dampings[i] * t_axis_full) * torch.cos(frequencies[j] * t_axis_full) 
            
            lambda_ij_new = optimize_lambda_nnls(A_new,auto_orig,initial_guess=lambda_ij.flatten()[indices]) #torch.linalg.lstsq(A_new, auto_orig)[0] #
            lambda_ij_debias = torch.zeros_like(lambda_ij, device=device)
            lambda_ij_debias.flatten()[indices] = lambda_ij_new
        else:
            lambda_zero = torch.zeros_like(lambda_ij, device=device)
            lambda_zero.flatten()[indices] = lambda_ij.flatten()[indices]
            lambda_ij = lambda_zero
    elif top_tresh:

        indices = torch.where(lambda_ij.flatten()>top_tresh)[0] #removed abs since we don't want negative peaks anyway
        top_n= len(lambda_ij.flatten()[indices])
        print(torch.max(lambda_ij),torch.min(lambda_ij))
        print(f'selected {top_n} values')
        if second_optimization:         
            original_indices = torch.unravel_index(indices, lambda_ij.shape)
            A_new = torch.zeros((len(t_axis_full), top_n), device=device)
            for k in range(top_n):
                i = original_indices[0][k]
                j = original_indices[1][k]
                A_new[:, k] = torch.exp(-dampings[i] * t_axis_full) * torch.cos(frequencies[j] * t_axis_full) 
            
            lambda_ij_new = optimize_lambda_nnls(A_new,auto_orig,initial_guess=lambda_ij.flatten()[indices]) #torch.linalg.lstsq(A_new, auto_orig)[0] #
            lambda_ij_debias = torch.zeros_like(lambda_ij, device=device)
            lambda_ij_debias.flatten()[indices] = lambda_ij_new
        else:
            lambda_zero = torch.zeros_like(lambda_ij, device=device)
            lambda_zero.flatten()[indices] = lambda_ij.flatten()[indices]
            lambda_ij = lambda_zero
    

    if sample_frequencies is None:
        sample_frequencies = torch.arange(0, 0.2, 0.00001, device=device) / hbar
    else:
        sample_frequencies = ensure_tensor_on_device(sample_frequencies, device=device)

    

    with torch.no_grad():
        
        J_new = torch.zeros_like(sample_frequencies)
       
        # Calculate the chunk size that would use less than the specified chunk_memory
        chunk_size = int(chunk_memory / (3*len(frequencies) * len(dampings) * auto.element_size()))
        chunk_size = max (chunk_size,1)
        for i in range(0, sample_frequencies.shape[0], chunk_size):
            max_idx = min(i + chunk_size, sample_frequencies.shape[0])
            sample_freq_chunk = sample_frequencies[i:max_idx]
            T_frequencies_gamma = sample_freq_chunk[:, None, None] * dampings[None, :, None] * np.pi * 2

            term1 = T_frequencies_gamma / (dampings[None, :, None]**2 + (sample_freq_chunk[:, None, None] - frequencies[None, None, :])**2)
            term2 = T_frequencies_gamma / (dampings[None, :, None]**2 + (sample_freq_chunk[:, None, None] + frequencies[None, None, :])**2)

            J_new[i:max_idx] = (lambda_ij[None, :, :] * (term1 + term2)).sum(dim=[1, 2]) 
        J_new=J_new.detach().cpu().numpy()
        
        auto_super = torch.zeros_like(auto_very_orig)
        for i in range(0, len(t_axis_full), chunk_size):
            max_idx = min(i + chunk_size, t_axis_full.shape[0])
            damping_term_chunk = torch.exp(-dampings[None, :] * t_axis_full[i:max_idx, None])
            frequency_term_chunk = torch.cos(frequencies[None, :] * t_axis_full[i:max_idx, None])
            A_chunk = (damping_term_chunk[:, :, None] * frequency_term_chunk[:, None, :]).reshape(max_idx - i, len(dampings) * len(frequencies))
            auto_super[i:max_idx] = A_chunk @ lambda_ij.flatten()
        auto_super = auto_super.detach().cpu().numpy()

        if second_optimization:
            J_new_debias = torch.zeros_like(sample_frequencies)
            #auto_super_debias = torch.zeros_like(auto_very_orig) # 
            A_new = torch.zeros((len(t_axis_very_orig), top_n), device=device)
            for k in range(top_n):
                i = original_indices[0][k]
                j = original_indices[1][k]
                A_new[:, k] = torch.exp(-dampings[i] * t_axis_very_orig) * torch.cos(frequencies[j] * t_axis_very_orig) 
            auto_super_debias = A_new @ lambda_ij_new.flatten()
            for k in range(top_n):
                i = original_indices[0][k]
                j = original_indices[1][k]
                T_frequencies_gamma= sample_frequencies*dampings[i] * np.pi * 2
                term1 = T_frequencies_gamma / (dampings[i]**2 + (sample_frequencies - frequencies[j])**2)
                term2 = T_frequencies_gamma / (dampings[i]**2 + (sample_frequencies + frequencies[j])**2)
                J_new_debias += lambda_ij_new[k]*(term1 + term2)
            J_new_debias=J_new_debias.detach().cpu().numpy()
            auto_super_debias=auto_super_debias.detach().cpu().numpy()
        else:
            J_new_debias = None
            auto_super_debias=None
            
        xaxis =    (sample_frequencies * hbar).detach().cpu().numpy()    
    
        return J_new, xaxis , auto_super , J_new_debias,auto_super_debias
        
        

