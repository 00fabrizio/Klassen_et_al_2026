"""Interpolators over the precomputed kernel tables."""
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

evap_dim = np.memmap(
    evap_path,
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

def _to_dimless(P, kappa, z, r):
    P = np.asarray(P, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)

    P_hat = np.clip(P / L_KERNEL, 0.0, 1.0)
    kappa_hat = kappa * R_KERNEL
    xi = z / L_KERNEL
    rho = r / R_KERNEL

    return P_hat, kappa_hat, xi, rho


def evaporation_diff(P, kappa, z, r):
    P_hat, kappa_hat, xi, rho = _to_dimless(P, kappa, z, r)
    P_arr, K_arr, X_arr, R_arr = np.broadcast_arrays(P_hat, kappa_hat, xi, rho)
    pts = np.stack([P_arr, K_arr, X_arr, R_arr], axis=-1)
    return _evap_interp(pts)


from .precompute_cascade import (
    P_HAT_GRID_C, SIGMA_H_GRID, THETA_GRID_C, N_ANG_GRID, XI_GRID_C, RHO_GRID_C,
    NP_C, NS_C, NN_C, NZ_C, NR_C, cascade_path, theta_to_n, n_to_theta,
)

cascade_dim = np.memmap(cascade_path, dtype="float32", mode="r",
                        shape=(NP_C, NS_C, NN_C, NZ_C, NR_C))
_cascade_interp = RegularGridInterpolator(
    (P_HAT_GRID_C, SIGMA_H_GRID, THETA_GRID_C, XI_GRID_C, RHO_GRID_C),
    cascade_dim, bounds_error=False, fill_value=None,
)


def cascade_coupled_theta(P, Sigma_h, theta_bar, z, r):
    P = np.asarray(P, dtype=float)
    Sigma_h = np.asarray(Sigma_h, dtype=float)
    theta_bar = np.asarray(theta_bar, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)

    P_hat = np.clip(P / L_KERNEL, 0.0, 1.0)
    xi = np.clip(z / L_KERNEL, 0.0, 1.0)
    rho = np.clip(r / R_KERNEL, RHO_GRID_C[0], RHO_GRID_C[-1])
    s_c = np.clip(Sigma_h, SIGMA_H_GRID[0], SIGMA_H_GRID[-1])
    t_c = np.clip(theta_bar, THETA_GRID_C[0], THETA_GRID_C[-1])

    A, B, C, D, E = np.broadcast_arrays(P_hat, s_c, t_c, xi, rho)
    return np.maximum(_cascade_interp(np.stack([A, B, C, D, E], axis=-1)), 0.0)


def cascade_coupled(P, Sigma_h, n_ang, z, r):
    return cascade_coupled_theta(P, Sigma_h, n_to_theta(n_ang), z, r)


from .precompute_slow import (
    KAPPA_SLOW_GRID, NK_SLOW, sq_path,
)

slow_sq_dim = np.memmap(sq_path, dtype="float32", mode="r",
                        shape=(NP, NK_SLOW, NZ, NR))
_slow_sq_interp = RegularGridInterpolator(
    (P_HAT_GRID, KAPPA_SLOW_GRID, XI_GRID, RHO_GRID),
    slow_sq_dim, bounds_error=False, fill_value=None,
)


def slow_diff_single(P, kappa, z, r):
    P_hat = np.clip(np.asarray(P, dtype=float) / L_KERNEL, 0.0, 1.0)
    k_hat = np.clip(np.asarray(kappa, dtype=float) * R_KERNEL,
                    KAPPA_SLOW_GRID[0], KAPPA_SLOW_GRID[-1])
    xi = np.clip(np.asarray(z, dtype=float) / L_KERNEL, 0.0, 1.0)
    rho = np.asarray(r, dtype=float) / R_KERNEL
    A, B, C, D = np.broadcast_arrays(P_hat, k_hat, xi, rho)
    return _slow_sq_interp(np.stack([A, B, C, D], axis=-1))
