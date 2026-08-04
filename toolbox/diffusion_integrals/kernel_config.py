# kernel_config.py
import numpy as np

R_KERNEL = 1.0    # cm
L_KERNEL = 45.0   # cm

# dimensionless interpolation grids
P_HAT_GRID     = np.linspace(0.0, 1.0, 30)
XI_GRID        = np.linspace(0.0, 1.0, 50)
RHO_GRID       = np.linspace(0.0, 5.5, 55)

# Evaporation. The fitted kappa_ev spans 0.07 to 1.04, so 0-1.5 covers every
# value with headroom at 0.052 resolution -- four times finer than the old
# 0-6 grid, which spent 80 % of its points above any value in use.
KAPPA_HAT_GRID = np.linspace(0.0, 1.5, 30)

# The two-group thermal kernel only needs the thermal group's kappa, which is
# held fixed at 0.128 cm^-1, so a coarse narrow grid suffices.
KAPPA_HAT_GRID_TH = np.linspace(0.0, 0.5, 20)

NP = len(P_HAT_GRID)
NK = len(KAPPA_HAT_GRID)
NK_TH = len(KAPPA_HAT_GRID_TH)
NZ = len(XI_GRID)
NR = len(RHO_GRID)