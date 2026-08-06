import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .kernel_config import (
    R_KERNEL, L_KERNEL,
    P_HAT_GRID, KAPPA_HAT_GRID, KAPPA_HAT_GRID_TH, XI_GRID, RHO_GRID,
    NP, NK, NK_TH, NZ, NR
)

_here = os.path.dirname(os.path.abspath(__file__))
evap_path = os.path.join(_here, "evap_dim.dat")
_therm_path = lambda sp: os.path.join(_here, f"therm_dim_{sp}.dat")

evap_dim = np.memmap(
    evap_path,
    dtype="float32",
    mode="r",
    shape=(NP, NK, NZ, NR),
)
# one thermal table per species: the source is the cascade, so the table is
# built at that species' Sigma_t (proton 0.346, carbon 0.125)
therm_dim = {sp: np.memmap(_therm_path(sp), dtype="float32", mode="r",
                           shape=(NP, NK_TH, NZ, NR))
             for sp in ("proton", "carbon")}

_evap_interp = RegularGridInterpolator(
    (P_HAT_GRID, KAPPA_HAT_GRID, XI_GRID, RHO_GRID),
    evap_dim,
    bounds_error=False,
    fill_value=0.0,
)
_therm_interp = {
    sp: RegularGridInterpolator(
        (P_HAT_GRID, KAPPA_HAT_GRID_TH, XI_GRID, RHO_GRID),
        tab, bounds_error=False, fill_value=0.0)
    for sp, tab in therm_dim.items()
}


def _to_dimless(P, kappa, z, r):
    P = np.asarray(P, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)

    # P is the production range and the source is Theta(z)Theta(P-z) inside a
    # phantom of length L, so P > L means the source fills the phantom and the
    # kernel saturates at P = L. Without this clamp P_hat leaves the [0,1]
    # interpolation grid and the kernel silently returns fill_value = 0,
    # switching the regime off instead of saturating it.
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


def _therm_one_group(P, kappa, z, r, species):
    """One-group cascade-source kernel for one species, straight from the table."""
    P_hat, kappa_hat, xi, rho = _to_dimless(P, kappa, z, r)
    P_arr, K_arr, X_arr, R_arr = np.broadcast_arrays(P_hat, kappa_hat, xi, rho)
    return _therm_interp[species](np.stack([P_arr, K_arr, X_arr, R_arr], axis=-1))


def thermal_diff_onegroup(P, kappa, z, r, species='proton'):
    """Single-group cascade-source kernel -- the quantity actually tabulated.

    Exposed for diagnostics; the model itself uses the two-group thermal_diff.
    """
    return _therm_one_group(P, kappa, z, r, species)


def thermal_diff(P, kappa_f, kappa_s, z, r, species='proton'):
    """Two-group slow-neutron kernel (epithermal and thermal share it).

        F_2 = [F_1(kappa_f) - F_1(kappa_s)] / (kappa_s^2 - kappa_f^2)

    Both kappas index the same tabulated one-group kernel, so making kappa_f
    a free parameter costs nothing. As kappa_s -> kappa_f the expression is
    0/0; the limit is -(1/2 kappa) dF_1/dkappa, evaluated by central
    difference.
    """
    kf = float(kappa_f)
    ks = float(kappa_s)
    d = ks ** 2 - kf ** 2

    if abs(d) < 1e-3:                       # removable singularity
        h = 1e-3
        km = 0.5 * (kf + ks)
        dF = (_therm_one_group(P, km + h, z, r, species)
              - _therm_one_group(P, max(km - h, 0.0), z, r, species)) / (2 * h)
        return -dF / (2 * max(km, 1e-6))

    return (_therm_one_group(P, kf, z, r, species)
            - _therm_one_group(P, ks, z, r, species)) / d

# ----------------------------------------------------------------------
# coupled ballistic cascade kernel (replaces cascade_lateral x cascade)
# ----------------------------------------------------------------------
from .precompute_cascade import (                                  # noqa: E402
    P_HAT_GRID_C, SIGMA_H_GRID, N_ANG_GRID, XI_GRID_C, RHO_GRID_C,
    NP_C, NS_C, NN_C, NZ_C, NR_C, cascade_path,
)

cascade_dim = np.memmap(cascade_path, dtype="float32", mode="r",
                        shape=(NP_C, NS_C, NN_C, NZ_C, NR_C))
_cascade_interp = RegularGridInterpolator(
    (P_HAT_GRID_C, SIGMA_H_GRID, N_ANG_GRID, XI_GRID_C, RHO_GRID_C),
    cascade_dim, bounds_error=False, fill_value=None,   # clamp, never zero out
)


def cascade_coupled(P, Sigma_h, n_ang, z, r):
    """Coupled first-flight cascade kernel: spatial factor for (P, Sigma_h, n_ang).

    Supersedes cascade_lateral(r) * cascade(z), which could not reproduce the
    rho-dependent shift of the axial peak.
    """
    P = np.asarray(P, dtype=float)
    Sigma_h = np.asarray(Sigma_h, dtype=float)
    n_ang = np.asarray(n_ang, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)

    P_hat = np.clip(P / L_KERNEL, 0.0, 1.0)
    xi = np.clip(z / L_KERNEL, 0.0, 1.0)
    rho = np.clip(r / R_KERNEL, RHO_GRID_C[0], RHO_GRID_C[-1])
    s_c = np.clip(Sigma_h, SIGMA_H_GRID[0], SIGMA_H_GRID[-1])
    n_c = np.clip(n_ang, N_ANG_GRID[0], N_ANG_GRID[-1])

    A, B, C, D, E = np.broadcast_arrays(P_hat, s_c, n_c, xi, rho)
    return np.maximum(_cascade_interp(np.stack([A, B, C, D, E], axis=-1)), 0.0)


# ----------------------------------------------------------------------
# slow-neutron kernel over the UNIFORM cylindrical source
# ----------------------------------------------------------------------
from .precompute_slow import (                                     # noqa: E402
    KAPPA_SLOW_GRID, NK_SLOW, slow_path,
)

slow_dim = np.memmap(slow_path, dtype="float32", mode="r",
                     shape=(NP, NK_SLOW, NZ, NR))
_slow_interp = RegularGridInterpolator(
    (P_HAT_GRID, KAPPA_SLOW_GRID, XI_GRID, RHO_GRID),
    slow_dim, bounds_error=False, fill_value=None,
)


def _slow_one_group(P, kappa, z, r):
    P_hat = np.clip(np.asarray(P, dtype=float) / L_KERNEL, 0.0, 1.0)
    k_hat = np.clip(np.asarray(kappa, dtype=float) * R_KERNEL,
                    KAPPA_SLOW_GRID[0], KAPPA_SLOW_GRID[-1])
    xi = np.clip(np.asarray(z, dtype=float) / L_KERNEL, 0.0, 1.0)
    rho = np.asarray(r, dtype=float) / R_KERNEL
    A, B, C, D = np.broadcast_arrays(P_hat, k_hat, xi, rho)
    return _slow_interp(np.stack([A, B, C, D], axis=-1))


def slow_diff(P, kappa_f, kappa_s, z, r):
    """Two-group slow-neutron kernel over the uniform cylindrical source.

        F_2 = [F_1(kappa_f) - F_1(kappa_s)] / (kappa_s^2 - kappa_f^2)

    Species-independent: the source is the same cylinder used by the cascade
    and evaporation regimes, so unlike thermal_diff this carries no Sigma_t and
    needs no per-species table. Symmetric under kappa_f <-> kappa_s.
    """
    kf, ks = float(kappa_f), float(kappa_s)
    d = ks ** 2 - kf ** 2
    if abs(d) < 1e-3:                       # removable singularity
        h = 1e-3
        km = 0.5 * (kf + ks)
        dF = (_slow_one_group(P, km + h, z, r)
              - _slow_one_group(P, max(km - h, 0.0), z, r)) / (2 * h)
        return -dF / (2 * max(km, 1e-6))
    return (_slow_one_group(P, kf, z, r) - _slow_one_group(P, ks, z, r)) / d


# ----------------------------------------------------------------------
# single-diffusion-length slow kernel (squared propagator)
# ----------------------------------------------------------------------
from .precompute_slow import sq_path                               # noqa: E402

slow_sq_dim = np.memmap(sq_path, dtype="float32", mode="r",
                        shape=(NP, NK_SLOW, NZ, NR))
_slow_sq_interp = RegularGridInterpolator(
    (P_HAT_GRID, KAPPA_SLOW_GRID, XI_GRID, RHO_GRID),
    slow_sq_dim, bounds_error=False, fill_value=None,
)


def slow_diff_single(P, kappa, z, r):
    """Two-group slow kernel with a single diffusion length.

    Collapsing kappa_f = kappa_s = kappa gives the squared propagator
    1/(k^2+kappa^2)^2, tabulated directly, so this is one lookup rather than a
    difference of two -- no removable singularity and no cancellation. The
    migration length is M = sqrt(2)/kappa.
    """
    P_hat = np.clip(np.asarray(P, dtype=float) / L_KERNEL, 0.0, 1.0)
    k_hat = np.clip(np.asarray(kappa, dtype=float) * R_KERNEL,
                    KAPPA_SLOW_GRID[0], KAPPA_SLOW_GRID[-1])
    xi = np.clip(np.asarray(z, dtype=float) / L_KERNEL, 0.0, 1.0)
    rho = np.asarray(r, dtype=float) / R_KERNEL
    A, B, C, D = np.broadcast_arrays(P_hat, k_hat, xi, rho)
    return _slow_sq_interp(np.stack([A, B, C, D], axis=-1))
