import numpy as np
import matplotlib.pyplot as plt
from mlnise.example_spectral_functions import spectral_Log_Normal_Lorentz,spectral_Lorentz,spectral_Drude
import functools
from mlnise.fft_noise_gen import noise_algorithm
import torch
import tqdm

cm_to_eV=1.23984E-4

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

def objective_function(lambda_ij, A, C, sparcity_penalty, l1_norm_penalty, solution_penalty):
    """Objective function with TV norm, L1 norm, and penalty for constraint violation."""
    tv = tv_norm_2d(lambda_ij)
    l1_norm = torch.sum(torch.abs(lambda_ij)) #torch.sum(torch.sqrt(torch.abs(lambda_ij)))
    negative_penalty= -torch.sum(lambda_ij)+torch.sum(torch.abs(lambda_ij))
    penalty = solution_penalty * 0.5 * torch.norm(A @ lambda_ij.flatten() - C) ** 2
    result= sparcity_penalty*tv + l1_norm_penalty*l1_norm + 1000*negative_penalty+ penalty
    #print(f"penalty {100*penalty/result:.3f}% Variation {100*1/mu*tv/result:.3f}%  l1_norm {100*l1_norm/result:.3f}%")
    return result

def objective_function_no_penalty(lambda_ij, A, C, sparcity_penalty, l1_norm_penalty):
    """Objective function with TV norm, L1 norm for constraint violation."""
    tv = tv_norm_2d(lambda_ij)
    l1_norm = torch.sum(torch.abs(lambda_ij)) #torch.sum(torch.sqrt(torch.abs(lambda_ij)))
    negative_penalty= -torch.sum(lambda_ij)+torch.sum(torch.abs(lambda_ij))
    result= sparcity_penalty*tv + l1_norm_penalty*l1_norm  + 1000*negative_penalty
    return result

def optimize_lambda(A, C, sparcity_penalty, l1_norm_penalty, solution_penalty, eta, max_iter=1000, tol=1e-6, lr=0.01, device='cuda',initial_guess=None):
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
    no_improvement_iters=0
    return_lambda=lambda_ij
    for iter in range(max_iter):
        optimizer.zero_grad()  # Clear previous gradients

        # Compute the objective function
        obj_value = objective_function(lambda_ij, A, C, sparcity_penalty, l1_norm_penalty, solution_penalty)
        obj_value_no_penalty =  objective_function_no_penalty(lambda_ij, A, C,sparcity_penalty, l1_norm_penalty)
        # Perform a backward pass to compute gradients
        obj_value.backward()
        def closure():
            optimizer.zero_grad()  # Clear previous gradients
    
            # Compute the objective function
            obj_value = objective_function(lambda_ij, A, C, sparcity_penalty, l1_norm_penalty, solution_penalty)
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


def ensure_tensor_on_device(array, device='cuda',dtype=torch.float):
    if isinstance(array, np.ndarray):
        # Convert numpy array to PyTorch tensor and move to the specified device
        return torch.from_numpy(array).to(device).to(dtype)
    elif isinstance(array, torch.Tensor):
        # Ensure the tensor is on the correct device
        return array.to(device).to(dtype)
    else:
        raise TypeError("Input must be a numpy array or a PyTorch tensor.")
        
        
def SD_Reconstruct_SuperResolution(auto, dt, T,hbar,k, sparcity_penalty=1, l1_norm_penalty=1, solution_penalty=10000, lr=0.01, max_iter=1000, eta=1e-7, tol=1e-7, minW=None, maxW=None, device='cuda',cutoff=None,sample_frequencies=None):
    auto = ensure_tensor_on_device(auto, device=device)
    
    
    N = len(auto)
    N_cut = int(cutoff/dt)# N//2 #
    dw_t = 2 * np.pi / (2 * N_cut * dt)
    
    if maxW is None:
        maxW = N_cut * dw_t
    if minW is None:
        minW = 0
    
    # Define time axis
    t_axis = dt * torch.arange(0, N_cut, device=device)

    # Define the grid of possible frequencies and damping coefficients
    frequencies = torch.arange(0, 0.2, 0.0001, device=device)/ hbar
    dampings = (torch.arange(1, 160, 5, device=device))*cm_to_eV/hbar
    #print(t_axis.shape)
    #print(dampings.shape)
    #print(frequencies.shape)
    dampings = dampings.unsqueeze(1)  # Shape: (len(dampings), 1)
    frequencies = frequencies.unsqueeze(1)  # Shape: (1, len(frequencies)) 
    t_axis = t_axis.unsqueeze(0)  # Shape: (1, N_cut)
    
    # Calculate the damping term: (len(dampings), N_cut)
    damping_term = torch.exp(-dampings * t_axis)
    
    # Calculate the frequency term: (len(frequencies), N_cut)
    frequency_term = torch.cos(frequencies * t_axis)
    
    # Perform an outer product and reshape to form the matrix A
    # Now combine them using broadcasting, resulting in a tensor of shape (len(dampings), len(frequencies), N_cut)
    A = damping_term.unsqueeze(1) * frequency_term.unsqueeze(0)

    # Finally, reshape A to have the shape (N_cut, len(dampings) , len(frequencies))
    A = A.permute(2, 0, 1)
    # Average the autocorrelation function    
    auto_orig=torch.clone(auto)
    auto = auto[:N_cut]
    #print(auto.shape)
    #if cutoff:
    #    print("applying damping")
    #    auto=auto*torch.exp(-(t_axis.squeeze()/cutoff)**2)
    #print(auto.shape)
    lambda_ij=optimize_lambda(A, auto, sparcity_penalty, l1_norm_penalty, solution_penalty, eta,  max_iter=max_iter, tol=tol, lr=lr, device=device)
    A = A.reshape(N_cut, len(dampings) * len(frequencies) )
    
    # Solve the sparse recovery problem using FISTA or TWIST
    
    
    # Calculate the spectral density using the recovered lambda coefficients
    # Reshape dampings and lambda_ij to allow broadcasting
    # Reshape dampings and lambda_ij to allow broadcasting
    frequencies = frequencies.squeeze()
    #frequencies = frequencies.unsqueeze(0)
    #plt.plot((auto).cpu().numpy(),label="auto")
    #plt.plot((A@lambda_ij.flatten()).detach().cpu().numpy(),label="superresolution")
   
    top_n=True
    
    if top_n:
        n = 500  # Replace with the desired number of largest values
        _, indices = torch.topk(torch.abs(lambda_ij.flatten()), n)
        
        #lambda_ij_new = lambda_ij[indices] 
        
        
        second_optimization=True
        if second_optimization:  
            t_axis = dt * torch.arange(0, N, device=device)
            frequencies = torch.arange(0, 0.2, 0.0001, device=device)/ hbar
            dampings = (torch.arange(1, 160, 5, device=device))*cm_to_eV/hbar
            
            original_indices = torch.unravel_index(indices, lambda_ij.shape)
            A_new=torch.zeros((len(t_axis),n),device=device)
            for k in range(n):
                i=original_indices[0][k]
                j = original_indices[1][k]
                A_new[:,k]=torch.exp(-dampings[i] * t_axis)*torch.cos(frequencies[j] * t_axis)
            #A_new = A[:, indices]
            print(A_new.shape)
            print(auto.shape)
            lambda_ij_new = torch.linalg.lstsq(A_new, auto_orig)[0]
            lambda_ij=torch.zeros_like(lambda_ij,device=device)
            lambda_ij.flatten()[indices]=lambda_ij_new
            #dampings = dampings.unsqueeze(1)  # Shape: (len(dampings), 1)
            #frequencies = frequencies.unsqueeze(0)  # Shape: (1, len(frequencies)) 
            #t_axis = t_axis.unsqueeze(0)  # Shape: (1, N_cut)
            #plt.plot((A@lambda_ij.flatten()).detach().cpu().numpy(),label="lstsqs")
        else:
            lambda_zero=torch.zeros_like(lambda_ij,device=device)
            lambda_zero.flatten()[indices]=lambda_ij.flatten()[indices]
            lambda_ij=lambda_zero
            #plt.plot((A@lambda_ij.flatten()).detach().cpu().numpy(),label=f"top {n}")
    #plt.legend()
    #plt.show()
    #plt.close()
    
    lambda_ij = lambda_ij.view(len(dampings), len(frequencies))  # Ensure lambda_ij matches the reshaped sizes
    lambda_ij=lambda_ij.unsqueeze(0)
    frequencies = frequencies.unsqueeze(0)
    dampings=dampings.unsqueeze(0)
    if sample_frequencies is None:
        sample_frequencies=torch.arange(0, 0.2, 0.00001, device=device)/ torch.tensor(hbar, device=device)
    else:
        sample_frequencies=ensure_tensor_on_device(sample_frequencies, device=device)
    
    J_new=torch.zeros_like(sample_frequencies)
    n=1
    with torch.no_grad():
        frequencies = torch.arange(0, 0.2, 0.0001, device=device)/ hbar
        dampings = (torch.arange(1, 160, 5, device=device))*cm_to_eV/hbar
        
        frequencies = frequencies.unsqueeze(0).unsqueeze(0)
        dampings = dampings.unsqueeze(1).unsqueeze(0)
        
        
        for i in range(0, sample_frequencies.shape[0], n):
        # Extract the current batch of sample frequencies
            min_i=i
            max_i= min(i+n,sample_frequencies.shape[0])
            sample_frequencies_batch = sample_frequencies[min_i:max_i]
        
            # Expand dimensions to match the dampings and frequencies shapes
            sample_frequencies_batch = sample_frequencies_batch.unsqueeze(1).unsqueeze(1)

            # Calculate the numerator for both terms in the sum
            T_frequencies_gamma = sample_frequencies_batch * dampings * np.sqrt(np.pi) * 2
            
            # Compute the two parts of the summation for this batch
            term1 = T_frequencies_gamma / (dampings**2 + (sample_frequencies_batch - frequencies)**2)
            term2 = T_frequencies_gamma / (dampings**2 + (sample_frequencies_batch + frequencies)**2)
            
            # Sum both terms and multiply by the corresponding lambda_ij
            J_new_batch = (lambda_ij * (term1 + term2)).sum(dim=[1, 2]) * torch.sqrt(torch.tensor(torch.pi, device=lambda_ij.device))
            
            # Accumulate the results
            J_new[min_i:max_i] = J_new_batch    
        
    
    """sample_frequencies=sample_frequencies.unsqueeze(1)
    sample_frequencies=sample_frequencies.unsqueeze(1)
    # Calculate the numerator for both terms in the sum
    print(sample_frequencies.shape)
    print(dampings.shape)
    print(frequencies.shape)
    T_frequencies_gamma =  sample_frequencies * dampings *np.sqrt(np.pi)*2
    
    # Compute the two parts of the summation
    term1 = T_frequencies_gamma / (dampings**2 + (sample_frequencies - frequencies)**2)
    term2 = T_frequencies_gamma / (dampings**2 + (sample_frequencies + frequencies)**2)
    
    # Sum both terms and multiply by the corresponding lambda_ij
    J_new = (lambda_ij * (term1 + term2)).sum(dim=[1,2]) * torch.sqrt(torch.tensor(torch.pi, device=lambda_ij.device))"""
    t_axis = dt * torch.arange(0, N, device=device)
    frequencies = torch.arange(0, 0.2, 0.0001, device=device)/ hbar
    dampings = (torch.arange(1, 160, 5, device=device))*cm_to_eV/hbar
    dampings = dampings.unsqueeze(1)  # Shape: (len(dampings), 1)
    frequencies = frequencies.unsqueeze(1)  # Shape: (1, len(frequencies)) 
    t_axis = t_axis.unsqueeze(0)  # Shape: (1, N_cut)
    # Calculate the damping term: (len(dampings), N_cut)
    damping_term = torch.exp(-dampings * t_axis)
    
    # Calculate the frequency term: (len(frequencies), N_cut)
    frequency_term = torch.cos(frequencies * t_axis)
    
    # Perform an outer product and reshape to form the matrix A
    # Now combine them using broadcasting, resulting in a tensor of shape (len(dampings), len(frequencies), N_cut)
    if second_optimization:  
        auto_super=A_new@lambda_ij_new.flatten()
    else:
        A = damping_term.unsqueeze(1) * frequency_term.unsqueeze(0)
    
        # Finally, reshape A to have the shape (N_cut, len(dampings) , len(frequencies))
        A = A.permute(2, 0, 1).reshape(N, len(dampings) * len(frequencies))
        auto_super=A@lambda_ij.flatten()
    
    return J_new.detach().cpu().numpy(), (sample_frequencies.squeeze().squeeze() * hbar).detach().cpu().numpy(),auto_super.detach().cpu().numpy()
