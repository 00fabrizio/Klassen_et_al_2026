# kernel_config.py
import numpy as np

R_KERNEL = 1.0    # cm
L_KERNEL = 45.0   # cm

# dimensionless interpolation grids
P_HAT_GRID     = np.linspace(0.0, 1.0, 30)
KAPPA_HAT_GRID = np.linspace(0.0, 6.0, 50)
XI_GRID        = np.linspace(0.0, 1.0, 50)
RHO_GRID       = np.linspace(0.0, 5.5, 55)

NP = len(P_HAT_GRID)
NK = len(KAPPA_HAT_GRID)
NZ = len(XI_GRID)
NR = len(RHO_GRID)