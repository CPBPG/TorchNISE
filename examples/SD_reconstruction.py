import mlnise
import numpy as np
import matplotlib.pyplot as plt
from mlnise.example_spectral_functions import spectral_Log_Normal_Lorentz,spectral_Lorentz
import functools
from mlnise.fft_noise_gen import noise_algorithm
import tqdm

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
step=1
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
    auto_step[cutoff//dt:]=0
    auto_step=auto_step*np.mean(np.abs(auto))/np.mean(np.abs(auto_step))
    auto_gauss=auto*np.exp(-(t_axis/cutoff)**2)
    auto_gauss=auto_gauss*np.mean(np.abs(auto))/np.mean(np.abs(auto_gauss))
    auto_exp=auto*np.exp(-(t_axis/cutoff))
    auto_exp=auto_exp*np.mean(np.abs(auto))/np.mean(np.abs(auto_exp))
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

def SD_Reconstruct_Super_Resolution(noise, dt, T, freqs=None, gammas=None, eta=1e-9, max_iter=1000):
    N = len(noise[0,:])
    reals = len(noise[:,0])
    
    if freqs is None:
        freqs = np.arange(0, 2000, 2)  # Frequency grid in cm^-1
    if gammas is None:
        gammas = np.arange(0, 160, 6)  # Damping coefficients in cm^-1
    
    dw_t = 2*np.pi/(2*N*dt)
    w_t = freqs * dw_t
    x_axis = w_t  # Frequency axis in 1/fs
    
    def autocorrelation(noise, i):
        cor1 = noise[:, :N-i]
        cor2 = noise[:, i:]
        res = cor1 * cor2
        C = np.mean(res, axis=1)
        return C
    
    def Ccalc(noise):
        C = np.zeros((reals, N))
        for i in tqdm.tqdm(range(int(N//2))):
            C[:, i] = autocorrelation(noise, i)
        return C
    
    def expval(noise):
        summation = Ccalc(noise)
        return np.mean(summation, axis=0)
    
    C = expval(noise)
    auto = C[:int(N//2)]
    t_axis = dt * np.arange(0, len(auto))
    
    # Construct the measurement matrix A
    A = np.zeros((len(auto), len(gammas) * len(freqs)))
    
    for k, t in enumerate(t_axis):
        for i, gamma in enumerate(gammas):
            for j, freq in enumerate(freqs):
                A[k, i*len(freqs) + j] = np.exp(-gamma * t) * np.cos(freq * t)
    
    # Define the objective function for super-resolution
    def objective(lmbda):
        return np.sum(np.abs(np.gradient(lmbda))) + np.sum(np.abs(lmbda))
    
    # Perform the super-resolution optimization
    result = minimize(lambda lmbda: objective(lmbda), np.zeros(A.shape[1]), method='L-BFGS-B',
                      constraints={'type': 'eq', 'fun': lambda lmbda: np.linalg.norm(A @ lmbda - auto) - eta},
                      options={'maxiter': max_iter})
    
    lmbda = result.x
    
    # Construct the spectral density J(ω)
    J_new = np.zeros(len(x_axis))
    for i, gamma in enumerate(gammas):
        for j, freq in enumerate(freqs):
            J_new += lmbda[i*len(freqs) + j] * np.sqrt(np.pi) * (T * freq * gamma) / (gamma**2 + (x_axis - freq)**2)
            J_new += lmbda[i*len(freqs) + j] * np.sqrt(np.pi) * (T * freq * gamma) / (gamma**2 + (x_axis + freq)**2)
    
    # Save the reconstructed autocorrelation function and spectral density
    np.save("data/auto_super_resolution.npy", auto)
    np.save("data/J_super_resolution.npy", J_new)
    
    return J_new, x_axis
Jw, x ,Jw_step,Jw_gauss,Jw_exp= SD_Reconstruct_FFT(Generated_Noise,step,T,cutoff=1500)
ww = x/hbar #

S=spectralfunc(ww)
SD=S/(2*np.pi*k*T)*ww

plt.plot(ww*hbar,SD/cm_to_eV,label="original")
plt.plot(x,Jw_gauss/cm_to_eV,label="from noise gauss")
#J_new, x_axis = SD_Reconstruct_Super_Resolution(Generated_Noise, step, T)
#plt.plot(x_axis,J_new/cm_to_eV,label="from noise super Resolution")
plt.legend()
plt.show()
plt.close()