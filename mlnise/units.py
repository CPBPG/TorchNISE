# Constants
k_cm = 0.6950389
hbar_cm = 5308.8459
cm_to_eV = 1.23984E-4
eV_to_J = 1.60218E-19  # Conversion factor from electron volts to Joules
fs_to_s = 1e-15  # Conversion factor from femtoseconds to seconds
fs_to_ps = 1e-3

# Allowed units
e_units = ["cm-1", "ev", "j"]
t_units = ["fs", "ps", "s"]

# Default units
current_e_unit = "cm-1"
current_t_unit = "fs"

# Initialize global variables
k = k_cm
hbar = hbar_cm
t_unit = 1

def set_units(e_unit="cm-1", t_unit="fs"):
    global k, hbar, current_e_unit, current_t_unit  # Declare globals to modify them
    
    if e_unit.lower() not in e_units:
        raise NotImplementedError(f"{e_unit} not implemented. Must be one of {e_units}")
    if t_unit.lower() not in t_units:
        raise NotImplementedError(f"{t_unit} not implemented. Must be one of {t_units}")
    
    current_e_unit = e_unit.lower()
    current_t_unit = t_unit.lower()
    
    # Set energy unit constants
    if e_unit.lower() == "cm-1":
        hbar = hbar_cm
        k = k_cm
    elif e_unit.lower() == "ev":
        hbar = hbar_cm * cm_to_eV
        k = k_cm * cm_to_eV
    elif e_unit.lower() == "j":
        hbar = hbar_cm * cm_to_eV * eV_to_J
        k = k_cm * cm_to_eV * eV_to_J
    
    # Set time unit constants
    if t_unit.lower() == "fs":
        t_unit = 1
    elif t_unit.lower() == "ps":
        t_unit = 1/fs_to_ps
    elif t_unit.lower() == "s":
        t_unit = 1/fs_to_s
        