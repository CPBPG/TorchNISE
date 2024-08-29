import mlnise
import numpy as np
import functools
from mlnise.example_spectral_functions import spectral_Drude_Lorentz_Heom, spectral_Drude
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


k = 8.6173303E-5 # in eV/K. 
T = 300 #Temperature in K
hbar = 0.658211951 #in eV fs
cm_to_eV=1.23984E-4
k=k/cm_to_eV

H=np.load("data/H_8site_time_independent_time_dependent_E.npy")
n_state=H.shape[0]
Omegak=np.array([0,725,1200])*cm_to_eV/hbar
Omegak=Omegak[:2]
HEOM_ER=200
HEOM_ER_peak=HEOM_ER/len(Omegak)
print("HEOM Er Peak",HEOM_ER_peak)
lambdak=np.array([HEOM_ER_peak,HEOM_ER_peak,HEOM_ER_peak])
lambdak=lambdak[:2]
vk=np.array([1/100,1/100,1/100])
vk=vk[:2]
spectralfunc=functools.partial(spectral_Drude_Lorentz_Heom,Omegak=Omegak,lambdak=lambdak,hbar=hbar,k=k,T=T,vk=vk)
tau=100
Er=100
H_0=torch.tensor(H,dtype=torch.float)
model=mlnise.mlnise_model.MLNISE()
reals=10000
dt=1
initiallyExcitedState=0
total_time=60000
spectral_funcs=[spectralfunc]

#trajectory_time=None, T_correction='None', maxreps=1000000, use_filter=False, filter_order=10, filter_cutoff=0.1, mode="population",mu=None,device="cpu",memory_mapped=False,save_Interval=10
device="cpu"
T_correction='Jansen'
save_Interval=10
memory_mapped=False
maxreps=200
avg_output, avg_blend, avg_boltzmann,avg_absorb_time,x_axis, absorb_f = mlnise.run_mlnise.RunNiseOptions(model, reals, H_0, tau, Er, T, dt, initiallyExcitedState, total_time, spectral_funcs,T_correction=T_correction,save_Interval=save_Interval,memory_mapped=memory_mapped,device=device,maxreps=maxreps)

np.save("avg_blend_60ps_2peak.npy",avg_blend)
np.save("avg_output_60ps_2peak.npy",avg_output)
np.save("avg_boltzmann_60ps_2peak.npy",avg_boltzmann)

colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
 '#7f7f7f', '#bcbd22', '#17becf']
boltzman_correct=torch.diag(torch.matrix_exp(torch.tensor(-H/(k*T))))/torch.sum(torch.diag(torch.matrix_exp(torch.tensor(-H/(k*T)))))
H_TD=torch.zeros((1000, 1000, n_state, n_state))
H_TD[:]=torch.tensor(H)
mynoise=mlnise.run_mlnise.gen_noise(spectral_funcs, dt, (1000,1000,n_state))
for i in range(n_state):
    H_TD[:, :, i, i] += mynoise[:, :, i]

H_eq=torch.diagonal(torch.matrix_exp(-H_TD/(k*T)),dim1=-2,dim2=-1)
boltzman_incorrect=torch.mean(H_eq/torch.sum(H_eq,dim=2,keepdim=True),dim=[0,1])

xaxis=torch.linspace(0, total_time/1000,avg_blend.shape[0] )

for i in range(0,2):
    plt.plot(xaxis,avg_blend[:,i],color=colors[i],label=f"site {i+1}")
    plt.plot([torch.min(xaxis),torch.max(xaxis)],[boltzman_correct[i],boltzman_correct[i]],linestyle="dashed",color=colors[i])
    plt.plot([torch.min(xaxis),torch.max(xaxis)],[boltzman_incorrect[i],boltzman_incorrect[i]],linestyle="dotted",color=colors[i])
boltzmann_correct_remaining=boltzman_correct[2].item()
boltzman_incorrect_remaining=boltzman_incorrect[2].item()
pop_remaining=avg_blend[:,2].clone()
for i in range(3,8):
    boltzmann_correct_remaining+=boltzman_correct[i]
    boltzman_incorrect_remaining+=boltzman_incorrect[i]
    pop_remaining+=avg_blend[:,i]
plt.plot(xaxis,pop_remaining,label="sites 3-8",color=colors[2])
plt.plot([torch.min(xaxis),torch.max(xaxis)],[boltzmann_correct_remaining,boltzmann_correct_remaining],linestyle="dashed",color=colors[2])
plt.plot([torch.min(xaxis),torch.max(xaxis)],[boltzman_incorrect_remaining,boltzman_incorrect_remaining],linestyle="dotted",color=colors[2])

plt.xlabel("time [ps]")
plt.ylabel("population")
plt.xlim([torch.min(xaxis),torch.max(xaxis)])
plt.ylim([0,1])
plt.legend()
plt.show()
plt.close()
for i in range(0,2):
    plt.plot(xaxis,avg_boltzmann[:,i],color=colors[i],label=f"site {i+1}")
    plt.plot([torch.min(xaxis),torch.max(xaxis)],[boltzman_correct[i],boltzman_correct[i]],color=colors[i],linestyle="dashed")
    plt.plot([torch.min(xaxis),torch.max(xaxis)],[boltzman_incorrect[i],boltzman_incorrect[i]],linestyle="dotted",color=colors[i])
pop_remaining=avg_boltzmann[:,2].clone()
for i in range(3,8):
    pop_remaining+=avg_boltzmann[:,i]
plt.plot(xaxis,pop_remaining,label="sites 3-8",color=colors[2])
plt.plot([torch.min(xaxis),torch.max(xaxis)],[boltzmann_correct_remaining,boltzmann_correct_remaining],linestyle="dashed",color=colors[2])
plt.plot([torch.min(xaxis),torch.max(xaxis)],[boltzman_incorrect_remaining,boltzman_incorrect_remaining],linestyle="dotted",color=colors[2])


plt.xlabel("time [ps]")
plt.ylabel("population")
plt.xlim([torch.min(xaxis),torch.max(xaxis)])
plt.ylim([0,1])
plt.legend()
plt.show()
plt.close()
for i in range(0,2):
    plt.plot(xaxis,avg_output[:,i],color=colors[i],label=f"site {i+1}")
    plt.plot([torch.min(xaxis),torch.max(xaxis)],[boltzman_correct[i],boltzman_correct[i]],color=colors[i],linestyle="dashed")
    plt.plot([torch.min(xaxis),torch.max(xaxis)],[boltzman_incorrect[i],boltzman_incorrect[i]],linestyle="dotted",color=colors[i])
pop_remaining=avg_output[:,2].clone()
for i in range(3,8):
    pop_remaining+=avg_output[:,i]
plt.plot(xaxis,pop_remaining,label="sites 3-8",color=colors[2])
plt.plot([torch.min(xaxis),torch.max(xaxis)],[boltzmann_correct_remaining,boltzmann_correct_remaining],linestyle="dashed",color=colors[2])
plt.plot([torch.min(xaxis),torch.max(xaxis)],[boltzman_incorrect_remaining,boltzman_incorrect_remaining],linestyle="dotted",color=colors[2])

plt.xlabel("time [ps]")
plt.ylabel("population")
plt.xlim([torch.min(xaxis),torch.max(xaxis)])
plt.ylim([0,1])
plt.legend()
plt.show()
plt.close()

def plot_pop(pop,method_name,save_name,xaxis):
    # Create the main figure and axis
    fig, ax1 = plt.subplots()
    
    # Main plot
    for i in range(0, 2):
        ax1.plot(xaxis, pop[:, i], color=colors[i], label=f"site {i+1}")
        ax1.plot([torch.min(xaxis), torch.max(xaxis)], [boltzman_correct[i], boltzman_correct[i]], color=colors[i], linestyle="dashed")
        ax1.plot([torch.min(xaxis), torch.max(xaxis)], [boltzman_incorrect[i], boltzman_incorrect[i]], linestyle="dotted", color=colors[i])
    
    pop_remaining = pop[:, 2].clone()
    for i in range(3, 8):
        pop_remaining += pop[:, i]
    ax1.plot(xaxis, pop_remaining, label="sites 3-8", color=colors[2])
    ax1.plot([torch.min(xaxis), torch.max(xaxis)], [boltzmann_correct_remaining, boltzmann_correct_remaining], linestyle="dashed", color=colors[2])
    ax1.plot([torch.min(xaxis), torch.max(xaxis)], [boltzman_incorrect_remaining, boltzman_incorrect_remaining], linestyle="dotted", color=colors[2])
    
    # Main plot settings
    ax1.set_xlabel("time [ps]")
    ax1.set_ylabel("population")
    ax1.set_xlim([torch.min(xaxis), torch.max(xaxis)])
    ax1.set_ylim([0, 1])
    
    # First legend for the main plot data
    first_legend = ax1.legend(loc='lower right', prop={'size': 10})
    ax1.add_artist(first_legend)
    
    # Adding the secondary legend for the black lines
    black_lines = [
        plt.Line2D([0], [0], color='black', linestyle='solid', label=method_name),
        plt.Line2D([0], [0], color='black', linestyle='dashed', label=r'$\hat{\rho}_{i,i}^{eq} \left ( \left\langle\overline{\hat{H}_S^{\rm  {eff},\alpha}(t)} \right\rangle\right )$'),
        plt.Line2D([0], [0], color='black', linestyle='dotted', label=r'$\left\langle \overline{\hat{\rho}_{i,i}^{eq}\left(\hat{H}_S^{\rm  {eff},\alpha}(t)\right)} \right\rangle$')
    ]
    secondary_legend = ax1.legend(handles=black_lines, loc='upper right', prop={'size': 10})
    ax1.add_artist(secondary_legend)
    
    # Creating the inset plot for zoomed-in section
    ax2 = inset_axes(ax1, width="45%", height="30%", loc='upper left', bbox_to_anchor=(0.05, 0, 1, 0.99), bbox_transform=ax1.transAxes)
    
    # Plotting the same data in the inset
    for i in range(0, 2):
        ax2.plot(xaxis, pop[:, i], color=colors[i])
    ax2.plot(xaxis, pop_remaining, color=colors[2])
    
    # Inset plot settings
    ax2.set_xlim(0, 0.2)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', labelsize='small')
    ax2.tick_params(axis='x', labelsize='small') #.set_visible(False)
    ax2.set_yticks(ax2.get_yticks()[[0,-1]])

    
    # Mark the inset on the main plot
    #mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none", ec="0.5")
    
    # Display the plot
    plt.savefig(save_name,bbox_inches="tight",pad_inches=0)
    plt.show()
    plt.close()
    print(f"{method_name}: Mean squared error right-hand side {torch.mean((boltzman_correct-torch.mean(pop[int(0.1*pop.shape[0]):, :],dim=0))**2)} ")
    print(f"{method_name}: Mean squared error left-hand side {torch.mean((boltzman_incorrect-torch.mean(pop[int(0.1*pop.shape[0]):, :],dim=0))**2)} ")

    
xaxis=torch.linspace(0, total_time/1000,avg_blend.shape[0] )
plot_pop(avg_blend,"TNISE int","plots/FMO_60ps_TNISE_interpolated.pdf",xaxis)
plot_pop(avg_output,"TNISE orig ave","plots/FMO_60ps_TNISE_orig.pdf",xaxis)
plot_pop(avg_boltzmann,"TNISE new ave","plots/FMO_60ps_TNISE_new.pdf",xaxis)

def loadHEOM(file):
    # Initialize lists to store time and population data
    time_data = []
    population_data = []
    
    # Open the file and read its contents
    with open(file, 'r') as file:
        lines = file.readlines()
    
    # Initialize a flag to start capturing data and a site index
    capture_data = False
    site_index = -1
    
    # Iterate over the lines in the file
    for line in lines:
        line = line.strip()  # Remove leading and trailing whitespace
    
        # Check if the line indicates a new site's data
        if 'Re[pop(site' in line:
            # Start capturing data for a new site
            capture_data = True
            site_index += 1
            
            # Initialize a new list for the current site's population data
            population_data.append([])  # Ensure there's a list to append data to
            
            continue
    
        # If we are capturing data and the line is not empty
        if capture_data and line:
            try:
                # Split the line by comma and convert to floats
                time_value, pop_value = map(float, line.split(','))
                
                # Append time data if this is the first site
                if site_index == 0:
                    time_data.append(time_value*10**12)
                
                # Append population data to the current site
                population_data[site_index].append(pop_value)
            except ValueError:
                # Handle lines that can't be converted (headers or invalid data)
                continue

    # Convert lists to torch tensors
    time_data.append(60)
    for i in range(len(population_data)):
        population_data[i].append(population_data[i][-1])
    time_array = torch.tensor(time_data)
    population_array = torch.tensor(population_data).swapaxes(0, 1)
    return  time_array,population_array
xaxis,pop_HEOM = loadHEOM("data/population (12).txt")
plot_pop(pop_HEOM,"HEOM","plots/FMO_60ps_HEOM.pdf",xaxis)