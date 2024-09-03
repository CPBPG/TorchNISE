import torch
import numpy as np
from scipy.optimize import dual_annealing
import torchnise.units as units
from torchnise.pytorch_utility import (
    golden_section_search,
    matrix_logh,
    batch_trace
    )

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
