import mlnise
import numpy as np
import matplotlib.pyplot as plt
from mlnise.example_spectral_functions import spectral_Log_Normal_Lorentz,spectral_Lorentz,spectral_Drude
import functools
from mlnise.fft_noise_gen import noise_algorithm
import tqdm
import torch


k = 8.6173303E-5 # in eV/K. 
T = 300 #Temperature in K
hbar = 0.658211951 #in eV fs
cm_to_eV=1.23984E-4
#k=k/cm_to_eV
save_name="data/noise.npy"
Sk = np.array([0.011, 0.011, 0.009, 0.009, 0.010, 0.011, 0.011, 0.012, 0.003, 0.008,
 0.008, 0.003, 0.006,  0.002, 0.002, 0.002,  0.001, 0.002, 0.004,  0.007, 
 0.004, 0.004, 0.003,  0.006, 0.004, 0.003, 0.007, 0.010, 0.005, 0.004, 
 0.009, 0.018,  0.007, 0.006, 0.007, 0.003, 0.004, 0.001,  0.001,  0.002, 
 0.002,  0.003, 0.001, 0.002, 0.002, 0.001, 0.001, 0.003, 0.003, 0.009, 0.007,
 0.010, 0.003, 0.005, 0.002, 0.004, 0.007, 0.002, 0.004, 0.002, 0.003, 0.003])
Wk = np.array([46, 68, 117, 167, 180, 191, 202, 243, 263, 284, 291, 327, 366, 385, 404, 423, 440, 
 481, 541, 568, 582, 597, 630, 638, 665, 684, 713, 726, 731, 750, 761, 770, 795, 821,
856, 891, 900, 924, 929, 946, 966, 984, 1004, 1037, 1058, 1094, 1104, 1123, 1130, 1162,
 1175, 1181, 1201, 1220, 1283, 1292, 1348, 1367, 1386, 1431, 1503, 1545])
Wk=Wk*cm_to_eV/hbar


total_time=100000
step=10
#number of realizations of the noise
reals = 1
#Temperature of the noise in K
T=300

S_HR=0.3
sigma=0.7
w_c=38*cm_to_eV/hbar
Gammak= 0.0009419458262008981
spectralfunc=functools.partial(spectral_Log_Normal_Lorentz,Wk=Wk,Sk=Sk,hbar=hbar,k=k,T=T,Gammak=Gammak,S_HR=S_HR,sigma=sigma,w_c=w_c)
#spectralfunc=functools.partial(spectral_Lorentz,Wk=Wk,Sk=Sk,hbar=hbar,k=k,T=T,Gammak=Gammak)
#spectralfunc=functools.partial(spectral_Drude, gamma=1/100, strength=100*cm_to_eV, k=k, T=T)
Generated_Noise=noise_algorithm((reals,total_time//step), step,spectralfunc,save=True,save_name=save_name)
from scipy.optimize import minimize


def SD_Reconstruct_FFT(noise,dt,T,minW=None,maxW=None,cutoff=None):
    
    N = len(noise[0,:])
    if cutoff==None:
        cutoff=N//5*dt
        print(cutoff)
    reals = len(noise[:,0])
    N_cut=N/2
    dw_t = 2*np.pi/(2*N_cut*dt)
    if maxW==None:
        maxW=N_cut*dw_t
    if minW==None:
        minW=0
        
    if maxW>N_cut*dw_t:
        print("Warning maxW bigger than maximum computable value")
    #    return

    
    #Paper Method Autocorrelation function 
    def autocorrelation(noise,i): #matrix implementation of the autocorrelation calculation. 
        cor1=noise[:,:N-i]
        cor2=noise[:,i:]
        res=cor1*cor2
        C=np.mean(res,axis=1)
        return C #returns the Calculation for a certain noise matrix, and index i, i.e. C(t_i). Since it is a matrix, this is the ith-column of the total autocorrelation matrix

    def Ccalc(noise):
        C = np.zeros((reals,N))
        for i in tqdm.tqdm(range(int(N//2))): #tqdm(range(N//2)):#Calculating the autocorrelation for the whole matrix. Rows: realizations, Columns: different i's
            C[:,i] = autocorrelation(noise,i)
        return C # matrix with size reals x N

    #Calculate expectation value
    def expval(noise):
        summation = Ccalc(noise) #Matrix with size reals x N. each row contains C(t)_i for eachh realization i 
        return np.mean(summation,axis=0) #calculating the mean over the different realizations
    C = expval(noise) #autocorrelation array, len = N, with dt, gamma and strength
    auto = C[:int(N//2)] #only trusting half the values employed
    
    N_cut = len(auto) #N/2
    t_axis=dt*np.arange(0, N_cut)    
    
    auto_step=np.copy(auto)
    auto_step[int(cutoff)//dt:]=0
    auto_gauss=auto*np.exp(-(t_axis/cutoff)**2)
    auto_exp=auto*np.exp(-(t_axis/cutoff))
    
    #rescale
    #auto_step=auto_step*np.mean(np.abs(auto))/np.mean(np.abs(auto_step))    
    #auto_gauss=auto_gauss*np.mean(np.abs(auto))/np.mean(np.abs(auto_gauss))    
    #auto_exp=auto_exp*np.mean(np.abs(auto))/np.mean(np.abs(auto_exp))
    
    np.save("data/auto.npy",auto)
    np.save("data/auto_gauss.npy",auto_gauss)
    np.save("data/auto_exp.npy",auto_exp)
    np.save("data/auto_step.npy",auto_step)
    #Calculation of spectral density
    dw_t = 2*np.pi/(2*N_cut*dt) #creating dw in units of 1/fs. denominator: 2*N_cut = N
    full_w_t = np.arange(0,N_cut*dw_t,dw_t)
    minWindex = np.argmin(np.abs(full_w_t-minW)) #find the closest value to minW and maxW from the fourier axis
    maxWindex = np.argmin(np.abs(full_w_t-maxW))
    w_t = full_w_t[minWindex:maxWindex+1]  #array of frequencies to use in 1/fs. Max frequency is not necessarily this. Only ought to be less than nyquist frequency: dw*N/2
    x_axis = hbar*w_t # x-axis on the graph from paper  jcp12    
    reverse_auto=np.flip(auto[1:-1])
    reverse_auto_step=np.flip(auto_step[1:-1])
    reverse_auto_gauss=np.flip(auto_gauss[1:-1])
    reverse_auto_exp=np.flip(auto_exp[1:-1])

    concat_auto=np.concatenate((auto,reverse_auto))
    concat_auto_step=np.concatenate((auto_step,reverse_auto_step))
    concat_auto_gauss=np.concatenate((auto_gauss,reverse_auto_gauss))
    concat_auto_exp=np.concatenate((auto_exp,reverse_auto_exp))
    
    J_new=x_axis*np.fft.fft(concat_auto)[0:len(x_axis)].real*dt/(hbar*2*np.pi*k*T)
    J_new_step=x_axis*np.fft.fft(concat_auto_step)[0:len(x_axis)].real*dt/(hbar*2*np.pi*k*T)
    J_new_gauss=x_axis*np.fft.fft(concat_auto_gauss)[0:len(x_axis)].real*dt/(hbar*2*np.pi*k*T)
    J_new_exp=x_axis*np.fft.fft(concat_auto_exp)[0:len(x_axis)].real*dt/(hbar*2*np.pi*k*T)
    return J_new, x_axis ,J_new_step,J_new_gauss,J_new_exp



def tv_norm_2d(lambda_ij):
    """Calculate the total variation norm across both dimensions."""
    tv = torch.sum(torch.abs(lambda_ij[:, :-1] - lambda_ij[:, 1:])) + \
         torch.sum(torch.abs(lambda_ij[:-1, :] - lambda_ij[1:, :]))
    return tv

def objective_function(lambda_ij, A, C, sparcity_penalty, l1_norm_penalty, solution_penalty):
    """Objective function with TV norm, L1 norm, and penalty for constraint violation."""
    tv = tv_norm_2d(lambda_ij)
    l1_norm = torch.sum(torch.abs(lambda_ij))
    negative_penalty= -torch.sum(lambda_ij)+torch.sum(torch.abs(lambda_ij))
    penalty = solution_penalty * 0.5 * torch.norm(A @ lambda_ij.flatten() - C) ** 2
    result= sparcity_penalty*tv + l1_norm_penalty*l1_norm + 1000*negative_penalty+ penalty
    #print(f"penalty {100*penalty/result:.3f}% Variation {100*1/mu*tv/result:.3f}%  l1_norm {100*l1_norm/result:.3f}%")
    return result

def objective_function_no_penalty(lambda_ij, A, C, sparcity_penalty, l1_norm_penalty):
    """Objective function with TV norm, L1 norm for constraint violation."""
    tv = tv_norm_2d(lambda_ij)
    l1_norm = torch.sum(torch.abs(lambda_ij))
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
        if no_improvement_iters >100:
            print("no improvement for 100 iters")
            break

    return return_lambda


def ensure_tensor_on_device(array, device='cuda'):
    if isinstance(array, np.ndarray):
        # Convert numpy array to PyTorch tensor and move to the specified device
        return torch.from_numpy(array).to(device)
    elif isinstance(array, torch.Tensor):
        # Ensure the tensor is on the correct device
        return array.to(device)
    else:
        raise TypeError("Input must be a numpy array or a PyTorch tensor.")
        
        
def SD_Reconstruct_SuperResolution(noise, dt, T, sparcity_penalty=1, l1_norm_penalty=1, solution_penalty=10000, lr=0.01, max_iter=1000, eta=1e-7, tol=1e-7, minW=None, maxW=None, device='cuda',cutoff=None,sample_frequencies=None):
    noise = ensure_tensor_on_device(noise, device=device)
    
    N = noise.shape[1]
    reals = noise.shape[0]
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
    dampings = (torch.arange(1, 160, 2, device=device))*cm_to_eV/hbar
    print(t_axis.shape)
    print(dampings.shape)
    print(frequencies.shape)
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
    def autocorrelation(noise, i):
        cor1 = noise[:, :N-i]
        cor2 = noise[:, i:]
        res = cor1 * cor2
        return torch.mean(res, axis=1)
    
    def Ccalc(noise):
        C = torch.zeros((reals, N), device=device)
        for i in tqdm.tqdm(range(int(N // 2))):
            C[:, i] = autocorrelation(noise, i)
        return C
    
    C = torch.mean(Ccalc(noise), axis=0)
    auto = C[:N_cut]
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
    frequencies = frequencies.unsqueeze(0)
    plt.plot((auto).cpu().numpy(),label="auto")
    plt.plot((A@lambda_ij.flatten()).detach().cpu().numpy(),label="superresolution")
   
    top_n=False
    
    if top_n:
        n = 200  # Replace with the desired number of largest values
        _, indices = torch.topk(torch.abs(lambda_ij.flatten()), n)
        A_new = A[:, indices]
        #lambda_ij_new = lambda_ij[indices] 
        print(A_new.shape)
        print(auto.shape)
        lambda_ij_new = torch.linalg.lstsq(A_new, auto)[0]
        second_optimization=False
        if second_optimization:
            lambda_ij=torch.zeros_like(lambda_ij,device=device)
            lambda_ij.flatten()[indices]=lambda_ij_new
            plt.plot((A@lambda_ij.flatten()).detach().cpu().numpy(),label="lstsqs")
        else:
            lambda_zero=torch.zeros_like(lambda_ij,device=device)
            lambda_zero.flatten()[indices]=lambda_ij.flatten()[indices]
            lambda_ij=lambda_zero
            plt.plot((A@lambda_ij.flatten()).detach().cpu().numpy(),label=f"top {n}")
    plt.legend()
    plt.show()
    plt.close()
    
    lambda_ij = lambda_ij.view(len(dampings), len(frequencies[0]))  # Ensure lambda_ij matches the reshaped sizes
    lambda_ij=lambda_ij.unsqueeze(0)
    frequencies = frequencies.unsqueeze(0)
    dampings=dampings.unsqueeze(0)
    if sample_frequencies==None:
        sample_frequencies=torch.arange(0, 0.2, 0.00001, device=device)/ torch.tensor(hbar, device=device)
    J_new=torch.zeros_like(sample_frequencies)
    n=1
    for i in tqdm.tqdm(range(0, sample_frequencies.shape[0], n)):
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
                
    
    return J_new.detach().cpu().numpy(), (sample_frequencies.squeeze().squeeze() * hbar).detach().cpu().numpy(),lambda_ij.detach().cpu().numpy()

# Example of setting the device to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cutoff=np.sqrt(total_time)/np.sqrt(100000)*5000
print(cutoff)
Jw, x ,Jw_step,Jw_gauss,Jw_exp= SD_Reconstruct_FFT(Generated_Noise,step,T,cutoff=cutoff)
ww = x/hbar #

S=spectralfunc(ww)
SD=S/(2*np.pi*k*T)*ww


#plt.xlim(0,0.2)
#plt.legend()
#plt.show()
#plt.close()
sample_frequencies=None#torch.tensor(x/hbar,device="cuda")
J_new, x_axis,lambda_ij = SD_Reconstruct_SuperResolution(Generated_Noise,step,T, sparcity_penalty=10000, l1_norm_penalty=0, solution_penalty=50000000 ,lr=25, max_iter=10000, eta=1e-5, tol=1e-7, minW=None, maxW=None, device='cuda',cutoff=cutoff,sample_frequencies =sample_frequencies)
plt.plot(ww*hbar,SD/cm_to_eV,label="original")
plt.plot(x,Jw_gauss/cm_to_eV,label="from noise gauss")
plt.plot(x_axis,J_new/cm_to_eV,label="from noise super Resolution")

plt.legend()
plt.xlim(0.14,0.15)
plt.show()
plt.close()