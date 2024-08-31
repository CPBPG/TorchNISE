"""
This file implements various Spectral functions returning the power spectrum.
"""
import numpy as np
import torchnise.units as units

def spectral_Drude(w,gamma,strength,T): #drude spectral density
    #converting the frequencies to angular frequencies. Update: done already on the definition as requested by professor        
    S = 4*gamma*strength*units.k*T/((w**2+gamma**2)) #calculating the spectral density. Recall S(w) =/ J(w)
    return S
def spectral_Lorentz(w,Wk,Sk,T,Gammak): #drude spectral density
        #converting the frequencies to angular frequencies. Update: done already on the definition as requested by professor        
        S = 0
        for i in range(len(Wk)):
            S += units.hbar*4*units.k*T*Sk[i]*Wk[i]**3*Gammak/((Wk[i]**2-w**2)**2+(w**2*Gammak**2))
        return S
def spectral_Drude_Lorentz(w,gamma,strength,Wk,Sk,T,Gammak): #drude spectral density
        #converting the frequencies to angular frequencies. Update: done already on the definition as requested by professor        
        S = 4*gamma*strength*units.k*T/((w**2+gamma**2))
        for i in range(len(Wk)):
            S += units.hbar*4*units.k*T*Sk[i]*Wk[i]**3*Gammak/((Wk[i]**2-w**2)**2+(w**2*Gammak**2))
        return S

def spectral_Drude_Lorentz_Heom(w,Omega_k,lambda_k,T,v_k): #drude spectral density
        #converting the frequencies to angular frequencies. Update: done already on the definition as requested by professor        
        S = 0
        for i in range(len(Omega_k)):
            S += 2*units.k*T*v_k[i]*lambda_k[i]/((Omega_k[i]-w)**2+v_k[i]**2)
            S += 2*units.k*T*v_k[i]*lambda_k[i]/((Omega_k[i]+w)**2+v_k[i]**2)
        

        return S

def spectral_Log_Normal(w,S_HR,sigma,w_c,T):
    S=np.sqrt(2*np.pi)*units.k*T*S_HR*units.hbar/sigma*np.exp(-(np.log(w/w_c))**2/(2*sigma**2))    
    S[np.isnan(S)]=0
    return S
def spectral_Log_Normal_Lorentz(w,Wk,Sk,T,Gammak,S_HR,sigma,w_c):
    S=0
    S += spectral_Lorentz(w,Wk,Sk,T,Gammak)
    S += spectral_Log_Normal(w,S_HR,sigma,w_c,T)
    return S