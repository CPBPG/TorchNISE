import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

cutoffs=0.2*np.arange(10,100)
errors_exp_theoretical=np.load("data/exp_40000_0.npy")
errors_exp_100_000=np.load("data/exp_5000_0.npy")
errors_exp_1_000_000=np.load("data/exp_50000_0.npy")
errors_gauss_theoretical=np.load("data/gauss_40000_0.npy")
errors_gauss_100_000=np.load("data/gauss_5000_0.npy")
errors_gauss_1_000_000=np.load("data/gauss_50000_0.npy")
errors_step_theoretical=np.load("data/step_40000_0.npy")
errors_step_100_000=np.load("data/step_5000_0.npy")
errors_step_1_000_000=np.load("data/step_50000_0.npy")
errors_super_debias_theoretical=np.load("data/super_debias_40000_0.npy")
errors_super_debias_100_000=np.load("data/super_debias_5000_0.npy")
errors_super_debias_1_000_000=np.load("data/super_debias_50000_0.npy")

errors_exp_theoretical_autocorr=np.load("data/exp_40000_0_auto.npy")
errors_exp_100_000_autocorr=np.load("data/exp_5000_0_auto.npy")
errors_exp_1_000_000_autocorr=np.load("data/exp_50000_0_auto.npy")
errors_gauss_theoretical_autocorr=np.load("data/gauss_40000_0_auto.npy")
errors_gauss_100_000_autocorr=np.load("data/gauss_5000_0_auto.npy")
errors_gauss_1_000_000_autocorr=np.load("data/gauss_50000_0_auto.npy")
errors_step_theoretical_autocorr=np.load("data/step_40000_0_auto.npy")
errors_step_100_000_autocorr=np.load("data/step_5000_0_auto.npy")
errors_step_1_000_000_autocorr=np.load("data/step_50000_0_auto.npy")
errors_super_debias_theoretical_autocorr=np.load("data/super_debias_40000_0_auto.npy")
errors_super_debias_100_000_autocorr=np.load("data/super_debias_5000_0_auto.npy")
errors_super_debias_1_000_000_autocorr=np.load("data/super_debias_50000_0_auto.npy")

plt.plot(cutoffs, errors_exp_theoretical, 'r', label="Exponential Theoretical")
#plt.plot(cutoffs, errors_gauss_theoretical, 'g', label="Gaussian Theoretical")
#plt.plot(cutoffs, errors_step_theoretical, 'b', label="Step Theoretical")
plt.plot(cutoffs, errors_super_debias_theoretical, 'orange', label="Super-Resolution Theoretical")
plt.plot(cutoffs, errors_exp_100_000, '--r', label="Exponential Noisy 100 ps")
#plt.plot(cutoffs, errors_gauss_100_000, '--g', label="Gaussian Noisy 100 ps")
#plt.plot(cutoffs, errors_step_100_000, '--b', label="Step Noisy 100 ps")
plt.plot(cutoffs, errors_super_debias_100_000, '--', color="orange",label="Super-Resolution Noisy 100 ps")
plt.plot(cutoffs, errors_exp_1_000_000, '-.r', label="Exponential Noisy 1 ns")
#plt.plot(cutoffs, errors_gauss_1_000_000, '-.g', label="Gaussian Noisy 1 ns")
#plt.plot(cutoffs, errors_step_1_000_000, '-.b', label="Step Noisy 1 ns")
plt.plot(cutoffs, errors_super_debias_1_000_000, '-.', color="orange", label="Super-Resolution Noisy 1 ns")

plt.xlabel("scale factor [ps]")
plt.ylabel("MAE [cm$^{-1}$]")

# Define legend for line types
legend_elements_type = [
    #Line2D([0], [0], color='r', lw=2, label='Exponential'),
    Line2D([0], [0], color='g', lw=2, label='Gaussian'),
    #Line2D([0], [0], color='b', lw=2, label='Step'),
    Line2D([0], [0], color='orange', lw=2, label='Super-Resolution')
]

# Define legend for line styles
legend_elements_style = [
    Line2D([0], [0], color='black', lw=1, linestyle='-', label='Theoretical'),
    Line2D([0], [0], color='black', lw=1, linestyle='--', label='100 ps Noise'),
    Line2D([0], [0], color='black', lw=1, linestyle='-.', label='1 ns Noise')
]

# Get the current axes
ax = plt.gca()

# First legend
first_legend = ax.legend(handles=legend_elements_type, loc='upper left')
ax.add_artist(first_legend)  # Manually add the first legend back

# Second legend
ax.legend(handles=legend_elements_style, loc='upper right')
ax.set_xlim(cutoffs[0], cutoffs[-1])
ax.set_ylim(0)
# Show plot
plt.savefig("plots/error_spectral_density_with_super.pdf",bbox_inches="tight",pad_inches=0)
plt.show()
plt.close()

#plt.plot(cutoffs, errors_exp_theoretical_autocorr, 'r', label="Exponential Theoretical")
plt.plot(cutoffs, errors_gauss_theoretical_autocorr, 'g', label="Gaussian Theoretical")
#plt.plot(cutoffs, errors_step_theoretical_autocorr, 'b', label="Step Theoretical")
plt.plot(cutoffs, errors_super_debias_theoretical_autocorr, 'orange', label="Super-Resolution Theoretical")
#plt.plot(cutoffs, errors_exp_100_000_autocorr, '--r', label="Exponential Noisy 100 ps")
plt.plot(cutoffs, errors_gauss_100_000_autocorr, '--g', label="Gaussian Noisy 100 ps")
#plt.plot(cutoffs, errors_step_100_000_autocorr, '--b', label="Step Noisy 100 ps")
plt.plot(cutoffs, errors_super_debias_100_000_autocorr, '--', color="orange", label="Super-Resolution Noisy 100 ps")
#plt.plot(cutoffs, errors_exp_1_000_000_autocorr, '-.r', label="Exponential Noisy 1 ns")
plt.plot(cutoffs, errors_gauss_1_000_000_autocorr, '-.g', label="Gaussian Noisy 1 ns")
#plt.plot(cutoffs, errors_step_1_000_000_autocorr, '-.b', label="Step Noisy 1 ns")
plt.plot(cutoffs, errors_super_debias_1_000_000_autocorr, '-.', color="orange", label="Super-Resolution Noisy 1 ns")

plt.xlabel("t$_c$ [ps]")
plt.ylabel("MAE [cm$^{-2}$]")

# Define legend for line types
legend_elements_type = [
    #Line2D([0], [0], color='r', lw=2, label='Exponential'),
    Line2D([0], [0], color='g', lw=2, label='Gaussian'),
    #Line2D([0], [0], color='b', lw=2, label='Step'),
    Line2D([0], [0], color='orange', lw=2, label='Super-Resolution')
]

# Define legend for line styles
legend_elements_style = [
    Line2D([0], [0], color='black', lw=1, linestyle='-', label='Theoretical'),
    Line2D([0], [0], color='black', lw=1, linestyle='--', label='100 ps Noise'),
    Line2D([0], [0], color='black', lw=1, linestyle='-.', label='1 ns Noise')
]

# Get the current axes
ax = plt.gca()

# First legend
first_legend = ax.legend(handles=legend_elements_type, loc='upper left')
ax.add_artist(first_legend)  # Manually add the first legend back

# Second legend
ax.legend(handles=legend_elements_style, loc='upper right')
ax.set_xlim(cutoffs[0], cutoffs[-1])
ax.set_ylim(0)
# Show plot
plt.savefig("plots/error_autocorrelation_with_super.pdf",bbox_inches="tight",pad_inches=0)
plt.show()
plt.close()
