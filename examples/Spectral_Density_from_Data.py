import numpy as np
import matplotlib.pyplot as plt
from torchnise.spectral_density_generation import sd_reconstruct_fft, get_auto
import torchnise.units as units

energy_unit="cm-1"
time_unit="fs"
units.set_units(e_unit=energy_unit,t_unit=time_unit)
data=np.loadtxt("E.txt")/units.CM_TO_EV
data=data-np.mean(data)
auto = get_auto(data)
damping="gauss" # "gauss" "exp" or "step"
cutoff=3000 # fs
dt=2 # fs
T=300 #K


plt.plot(np.linspace(0,len(auto)*dt,len(auto)),auto)
plt.xlabel("time [fs]")
plt.ylabel("autocorrelation [cm$^{-2}$]")
plt.show()
plt.close()


J_new, x_axis ,auto_damp = sd_reconstruct_fft(auto,dt,T,damping_type=damping,cutoff=cutoff,rescale=False)
plt.plot(x_axis*units.CM_TO_EV,J_new,label=f"{damping}")
plt.xlim([np.min(x_axis*units.CM_TO_EV),np.max(x_axis*units.CM_TO_EV)])
plt.ylim(0)
plt.xlabel("$\omega$ [eV]")
plt.ylabel("J($\omega$) [cm$^{-1}$]")
plt.title("Spectral density")
plt.legend()
plt.show()
plt.close()