"""
THIS IMPLEMENTS THE UNITS AND CONSTANTS FOR THE ENTIRE PROJECT
"""

# Constants
K_CM = 0.6950389
HBAR_CM = 5308.8459
CM_TO_EV = 1.23984e-4
EV_TO_J = 1.60218e-19  # Conversion factor from electron volts to Joules
FS_TO_S = 1e-15  # Conversion factor from femtoseconds to seconds
FS_TO_PS = 1e-3

# Allowed units
E_UNITS = ["cm-1", "ev", "j"]
T_UNITS = ["fs", "ps", "s"]

# Default units
CURRENT_E_UNIT = "cm-1"
CURRENT_T_UNIT = "fs"

class UnitSystem:
    def __init__(self, e_unit="cm-1", t_unit="fs"):
        self.e_unit = e_unit.lower()
        self.t_unit = t_unit.lower()
        
        if self.e_unit not in E_UNITS:
            raise NotImplementedError(f"{e_unit} not implemented. Must be one of {E_UNITS}")
        if self.t_unit not in T_UNITS:
            raise NotImplementedError(f"{t_unit} not implemented. Must be one of {T_UNITS}")
            
        # Set energy unit constants
        if self.e_unit == "cm-1":
            self.HBAR = HBAR_CM
            self.K = K_CM
        elif self.e_unit == "ev":
            self.HBAR = HBAR_CM * CM_TO_EV
            self.K = K_CM * CM_TO_EV
        elif self.e_unit == "j":
            self.HBAR = HBAR_CM * CM_TO_EV * EV_TO_J
            self.K = K_CM * CM_TO_EV * EV_TO_J

        # Set time unit constants
        if self.t_unit == "fs":
            self.T_UNIT = 1
        elif self.t_unit == "ps":
            self.T_UNIT = 1 / FS_TO_PS
        elif self.t_unit == "s":
            self.T_UNIT = 1 / FS_TO_S

# Initialize global default system
_default_system = UnitSystem()

# Initialize global variables for backward compatibility
K = _default_system.K
HBAR = _default_system.HBAR
T_UNIT = _default_system.T_UNIT

def set_units(e_unit="cm-1", t_unit="fs"):
    """
    set_the units for time and energy for the entire module

    Parameters:
        e_unit (string): Energy unit to be used. Must be one of "cm-1", "ev", "j"
        t_unit (float):  Time unit to be used. Must be one of "fs", "ps", "s"
    """
    # Declare globals to modify them
    global K, HBAR, CURRENT_E_UNIT, CURRENT_T_UNIT, T_UNIT, _default_system

    _default_system = UnitSystem(e_unit, t_unit)

    CURRENT_E_UNIT = _default_system.e_unit
    CURRENT_T_UNIT = _default_system.t_unit
    K = _default_system.K
    HBAR = _default_system.HBAR
    T_UNIT = _default_system.T_UNIT
