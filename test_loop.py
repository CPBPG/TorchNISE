from torchnise.mlnise_dataset import runHeomDrude
import numpy as np

print("Starting test loop")

# First run (known good parameters)
pop = runHeomDrude(
    2,
    np.array([[100, 100], [100, 0]]),
    tau=100,
    Temp=300,
    E_reorg=20,
    dt__unit=1,
    initiallyExcitedState=0,
    totalTime=1000,
    tier=7,
    matrix_type="dense",
)
print("Run 1 success")

# Loop run
for i in range(10):
    pop = runHeomDrude(
        2,
        np.array([[100, 100], [100, 0]]),
        tau=100,
        Temp=300,
        E_reorg=20,
        dt__unit=1,
        initiallyExcitedState=0,
        totalTime=1000,
        tier=7,
        matrix_type="dense",
    )
    print(f"Run {i+2} success")
