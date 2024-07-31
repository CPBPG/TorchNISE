"""
This file implements various Spectral functions returning the power spectrum.
"""
import numpy as np

def spectral_Drude(w,gamma,strength,k,T): #drude spectral density
    #converting the frequencies to angular frequencies. Update: done already on the definition as requested by professor        
    S = 4*gamma*strength*k*T/((w**2+gamma**2)) #calculating the spectral density. Recall S(w) =/ J(w)
    return S
def spectral_Lorentz(w,Wk,Sk,hbar,k,T,Gammak): #drude spectral density
        #converting the frequencies to angular frequencies. Update: done already on the definition as requested by professor        
        S = 0
        for i in range(len(Wk)):
            S += hbar*4*k*T*Sk[i]*Wk[i]**3*Gammak/((Wk[i]**2-w**2)**2+(w**2*Gammak**2))
        return S
def spectral_Drude_Lorentz(w,gamma,strength,Wk,Sk,hbar,k,T,Gammak): #drude spectral density
        #converting the frequencies to angular frequencies. Update: done already on the definition as requested by professor        
        S = 4*gamma*strength*k*T/((w**2+gamma**2))
        for i in range(len(Wk)):
            S += hbar*4*k*T*Sk[i]*Wk[i]**3*Gammak/((Wk[i]**2-w**2)**2+(w**2*Gammak**2))
        return S

def spectral_Drude_Lorentz_Heom(w,Omegak,lambdak,hbar,k,T,vk): #drude spectral density
        #converting the frequencies to angular frequencies. Update: done already on the definition as requested by professor        
        S = 0
        for i in range(len(Omegak)):
            S += 2*k*T*vk[i]*lambdak[i]/((Omegak[i]-w)**2+vk[i]**2)
            S += 2*k*T*vk[i]*lambdak[i]/((Omegak[i]+w)**2+vk[i]**2)
        return S
def spectral_Log_Normal(w,S_HR,sigma,w_c,k,T,hbar):
    S=np.sqrt(2*np.pi)*k*T*S_HR*hbar/sigma*np.exp(-(np.log(w/w_c))**2/(2*sigma**2))    
    S[np.isnan(S)]=0
    return S
def spectral_Log_Normal_Lorentz(w,Wk,Sk,hbar,k,T,Gammak,S_HR,sigma,w_c):
    S=0
    S += spectral_Lorentz(w,Wk,Sk,hbar,k,T,Gammak)
    S += spectral_Log_Normal(w,S_HR,sigma,w_c,k,T,hbar)
    return S