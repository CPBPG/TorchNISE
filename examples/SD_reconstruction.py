import mlnise
import numpy as np
import matplotlib.pyplot as plt
from mlnise.example_spectral_functions import spectral_Log_Normal_Lorentz,spectral_Lorentz,spectral_Drude
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
    auto_step[cutoff//dt:]=0
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

# Helper functions for TWIST/FISTA
def soft_thresholding(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def ista(A, b, lambda_, L, max_iter=1000, tol=1e-7):
    x = np.zeros(A.shape[1])
    for i in range(max_iter):
        print(i,np.linalg.norm(A @ x - b),x)
        x = soft_thresholding(x - (1/L) * A.T @ (A @ x - b), lambda_/L)
        if np.linalg.norm(A @ x - b) < tol:
            break
    return x

def fista(A, b, lambda_, L, max_iter=1000, tol=1e-7):
    plt.show()
    x = np.zeros(A.shape[1])
    y = x.copy()
    t = 1
    for i in range(max_iter):
        print(i,np.linalg.norm(A @ x - b),x)
        x_old = x.copy()
        x = soft_thresholding(y - (1/L) * A.T @ (A @ y - b), lambda_/L)
        t_old = t
        t = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t_old - 1) / t) * (x - x_old)
        if np.linalg.norm(A @ x - b) < tol:
            break
    return x

def twist(A, b, lambda_, L, max_iter=1000, tol=1e-7):
    x = np.zeros(A.shape[1])
    x_old = x.copy()
    for i in range(max_iter):
        print(i,np.linalg.norm(A @ x - b),x)
        x_new = soft_thresholding(x - (1/L) * A.T @ (A @ x - b), lambda_/L)
        if i > 0:
            beta = np.dot(x_new - x, x - x_old) / np.linalg.norm(x - x_old)**2
            x = x_new + beta * (x_new - x)
        else:
            x = x_new
        x_old = x.copy()
        if np.linalg.norm(A @ x - b) < tol:
            break
    return x

def SD_Reconstruct_SuperResolution(noise, dt, T, method='fista', lambda_=1e-3, L=1.0, max_iter=1000, tol=1e-7, minW=None, maxW=None):
    N = len(noise[0, :])
    reals = len(noise[:, 0])
    N_cut = N//2
    dw_t = 2 * np.pi / (2 * N_cut * dt)
    
    if maxW is None:
        maxW = N_cut * dw_t
    if minW is None:
        minW = 0
    
    # Define time axis
    t_axis = dt * np.arange(0, N_cut)

    
    # Define the grid of possible frequencies and damping coefficients
    frequencies = np.arange(0,0.2,0.002)/hbar
    dampings = 1/np.arange(0.1, 160, 0.2)
    A = np.zeros((N_cut, len(frequencies) * len(dampings)))
    
    for i, gamma in enumerate(dampings):
        for j, omega in enumerate(frequencies):
            A[:, i * len(frequencies) + j] = np.exp(-gamma * t_axis) * np.cos(omega * t_axis)
            #plt.plot(t_axis[0:50],(np.exp(-gamma * t_axis) * np.cos(omega * t_axis))[0:50])
            #plt.title(f"gamma {gamma}, omega,{omega}")
            #plt.show()
            #plt.close()
    
    # Average the autocorrelation function
    def autocorrelation(noise, i):
        cor1 = noise[:, :N-i]
        cor2 = noise[:, i:]
        res = cor1 * cor2
        return np.mean(res, axis=1)
    
    def Ccalc(noise):
        C = np.zeros((reals, N))
        for i in range(int(N//2)):
            C[:, i] = autocorrelation(noise, i)
        return C
    
    C = np.mean(Ccalc(noise), axis=0)
    auto = C[:N_cut]
    
    
    # Solve the sparse recovery problem using FISTA or TWIST
    if method.lower() == 'fista':
        lambda_ij = fista(A, auto, lambda_, L, max_iter, tol)
    elif method.lower() == 'twist':
        lambda_ij = twist(A, auto, lambda_, L, max_iter, tol)
    elif method.lower() == 'ista':
        lambda_ij = ista(A, auto, lambda_, L, max_iter, tol)
    else:
        raise ValueError("Method must be 'fista' or 'twist'")
    #lambda_ij=lambda_ij*0
    
    #lambda_ij[0]=1
    # Calculate the spectral density using the recovered lambda coefficients
    J_new = np.zeros_like(frequencies)
    for i, gamma in enumerate(dampings):
        for j, omega in enumerate(frequencies):
            J_new += lambda_ij[i * len(frequencies) + j] * np.sqrt(np.pi) * (
                (T * frequencies * gamma) / (gamma**2 + (frequencies - omega)**2) +
                (T * frequencies* gamma) / (gamma**2 + (frequencies + omega)**2)
            )
    
    return J_new, frequencies*hbar

Jw, x ,Jw_step,Jw_gauss,Jw_exp= SD_Reconstruct_FFT(Generated_Noise,step,T,cutoff=5000)
ww = x/hbar #

S=spectralfunc(ww)
SD=S/(2*np.pi*k*T)*ww
plt.plot(x,Jw_gauss/cm_to_eV,label="from noise")
plt.plot(ww*hbar,SD/cm_to_eV,label="original")

J_new, x_axis = SD_Reconstruct_SuperResolution(Generated_Noise,step,T, method='fista', lambda_=1e-9, L=100000, max_iter=100)
plt.plot(x_axis,J_new/cm_to_eV,label="from noise super Resolution")
#plt.xlim(0,0.2)
plt.legend()

plt.show()
plt.close()