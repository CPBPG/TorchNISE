"""
THIS IMPLEMENTS THE UNITS AND CONSTANTS FOR THE ENTIRE PROJECT
"""
# Constants
K_CM = 0.6950389
HBAR_CM = 5308.8459
CM_TO_EV = 1.23984E-4
EV_TO_J = 1.60218E-19  # Conversion factor from electron volts to Joules
FS_TO_S = 1e-15  # Conversion factor from femtoseconds to seconds
FS_TO_PS = 1e-3

# Allowed units
E_UNITS = ["cm-1", "ev", "j"]
T_UNITS = ["fs", "ps", "s"]

# Default units
CURRENT_E_UNIT = "cm-1"
CURRENT_T_UNIT = "fs"

# Initialize global variables
K = K_CM
HBAR = HBAR_CM
T_UNIT = 1

def set_units(e_unit="cm-1", t_unit="fs"):
    """
    set_the units for time and energy for the entire module

    Parameters:
        e_unit (string): Energy unit to be used. Must be one of "cm-1", "ev", "j"
        t_unit (float):  Time unit to be used. Must be one of "fs", "ps", "s"
    """
    # Declare globals to modify them
    global K, HBAR, CURRENT_E_UNIT, CURRENT_T_UNIT, T_UNIT

    if e_unit.lower() not in E_UNITS:
        raise NotImplementedError(f"{e_unit} not implemented. Must be one of "
                                  f"{E_UNITS}")
    if t_unit.lower() not in T_UNITS:
        raise NotImplementedError(f"{t_unit} not implemented. Must be one of "
                                  f"{T_UNITS}")

    CURRENT_E_UNIT  = e_unit.lower()
    CURRENT_T_UNIT = t_unit.lower()

    # Set energy unit constants
    if e_unit.lower() == "cm-1":
        HBAR = HBAR_CM
        K = K_CM
    elif e_unit.lower() == "ev":
        HBAR = HBAR_CM * CM_TO_EV
        K = K_CM * CM_TO_EV
    elif e_unit.lower() == "j":
        HBAR = HBAR_CM * CM_TO_EV * EV_TO_J
        K = K_CM *CM_TO_EV * EV_TO_J

    # Set time unit constants
    if t_unit.lower() == "fs":
        T_UNIT = 1
    elif t_unit.lower() == "ps":
        T_UNIT = 1/FS_TO_PS
    elif t_unit.lower() == "s":
        T_UNIT = 1/FS_TO_S
