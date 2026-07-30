import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .kernel_config import (
    R_KERNEL, L_KERNEL,
    P_HAT_GRID, KAPPA_HAT_GRID, XI_GRID, RHO_GRID,
    NP, NK, NZ, NR
)

_here = os.path.dirname(os.path.abspath(__file__))
evap_path = os.path.join(_here, "evap_dim.dat")
therm_path = os.path.join(_here, "therm_dim.dat")

evap_dim = np.memmap(
    evap_path,
    dtype="float32",
    mode="r",
    shape=(NP, NK, NZ, NR),
)
therm_dim = np.memmap(
    therm_path,
    dtype="float32",
    mode="r",
    shape=(NP, NK, NZ, NR),
)

_evap_interp = RegularGridInterpolator(
    (P_HAT_GRID, KAPPA_HAT_GRID, XI_GRID, RHO_GRID),
    evap_dim,
    bounds_error=False,
    fill_value=0.0,
)
_therm_interp = RegularGridInterpolator(
    (P_HAT_GRID, KAPPA_HAT_GRID, XI_GRID, RHO_GRID),
    therm_dim,
    bounds_error=False,
    fill_value=0.0,
)


def _to_dimless(P, kappa, z, r):
    P = np.asarray(P, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)

    P_hat = P / L_KERNEL
    kappa_hat = kappa * R_KERNEL
    xi = z / L_KERNEL
    rho = r / R_KERNEL

    return P_hat, kappa_hat, xi, rho


def evaporation_diff(P, kappa, z, r):
    P_hat, kappa_hat, xi, rho = _to_dimless(P, kappa, z, r)
    P_arr, K_arr, X_arr, R_arr = np.broadcast_arrays(P_hat, kappa_hat, xi, rho)
    pts = np.stack([P_arr, K_arr, X_arr, R_arr], axis=-1)
    return _evap_interp(pts)


def thermal_diff(P, kappa, z, r):
    P_hat, kappa_hat, xi, rho = _to_dimless(P, kappa, z, r)
    P_arr, K_arr, X_arr, R_arr = np.broadcast_arrays(P_hat, kappa_hat, xi, rho)
    pts = np.stack([P_arr, K_arr, X_arr, R_arr], axis=-1)
    return _therm_interp(pts)