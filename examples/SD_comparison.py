import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

x_original = np.load("x_original_vib.npy")
x_axis_100_000 = np.load("x_axis_100_000_vib.npy")
J_new_step_100_000 = np.load("J_new_step_100_000_vib.npy") 
J_new_gauss_100_000 = np.load("J_new_gauss_100_000_vib.npy") 
J_new_exp_100_000 = np.load("J_new_exp_100_000_vib.npy") 
SD_100_000 = np.load("SD_100_000_original_vib.npy") 
J_new_step = np.load("J_new_step_original_vib.npy") 
J_new_gauss = np.load("J_new_gauss_original_vib.npy") 
J_new_exp = np.load("J_new_exp_original_vib.npy")
SD = np.load("SD_original_vib.npy")

fig, ax1 = plt.subplots()
ax1.plot(x_axis_100_000,SD_100_000,'r',label="Original")
ax1.plot(x_axis_100_000,J_new_step_100_000,"--",label="Step")
ax1.plot(x_axis_100_000,J_new_gauss_100_000,"--",label="Gaussian")
ax1.plot(x_axis_100_000,J_new_exp_100_000,"--",label="Exponential")
ax1.legend(loc='lower left', prop={'size': 8})
ax1.set_xlabel("ω [eV]")
ax1.set_ylabel("J(ω) [cm$^{-1}$]")
ax1.set_xlim(x_axis_100_000[0], x_axis_100_000[-1])
ax1.set_ylim(0)

# Create inset of the desired range on x-axis, placing it on the top right
# The parameters [0.5, 0.5, 0.45, 0.45] define the inset axes' position and size relative to the parent axes
# These parameters are [left, bottom, width, height] in figure fraction coordinates
ax2 = inset_axes(ax1, width="65%", height="60%", loc='upper left')

# Define the range for the inset x-axis
x_start, x_end = 0.14,0.15
inset_x_range = np.arange(x_start, x_end+10, 10)

# Interpolate or slice your data to match the new x-axis range for the inset
# Assuming your data can be directly indexed for simplicity
ax2.plot(x_axis_100_000,SD_100_000,'r',label="Original")
ax2.plot(x_axis_100_000,J_new_step_100_000,"--",label="Step")
ax2.plot(x_axis_100_000,J_new_gauss_100_000,"--",label="Gaussian")
ax2.plot(x_axis_100_000,J_new_exp_100_000,"--",label="Exponential")
ax2.set_xlim(x_start, x_end )
ax2.set_ylim(30,2100)
ax2.yaxis.set_visible(False)



# Optional: Customize the inset's appearance
ax2.tick_params(axis='x', labelsize='small')
ax2.tick_params(axis='y', labelsize='small')

current_xticks = ax2.get_xticks()
current_xticklabels = [f'{tick:.3f}' for tick in current_xticks]

# Remove the first entry from the x-ticks and x-tick labels
new_xticks = current_xticks[1:-1]  # Skip the first tick
new_xticklabels = current_xticklabels[1:-1]  # Skip the first tick label

# Set the new x-ticks and x-tick labels
ax2.set_xticks(new_xticks)
ax2.set_xticklabels(new_xticklabels)
#ax2.legend(loc='upper right', prop={'size': 6})
ax2.tick_params(axis='x', labelsize='small')
mark_inset(ax1, ax2, loc1=1, loc2=3, fc="none", ec="0.5")
plt.savefig("SD_comparison_100k.pdf")
plt.show()