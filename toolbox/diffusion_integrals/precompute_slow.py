"""Builds the slow-neutron kernel table."""
import os

import numpy as np

try:
    from . import precompute_kernels as PK
    from .kernel_config import P_HAT_GRID, XI_GRID, RHO_GRID, NP, NZ, NR, R_KERNEL, L_KERNEL
except ImportError:
    import precompute_kernels as PK
    from kernel_config import P_HAT_GRID, XI_GRID, RHO_GRID, NP, NZ, NR, R_KERNEL, L_KERNEL


KAPPA_SLOW_GRID = np.linspace(0.0, 0.5, 60)
NK_SLOW = len(KAPPA_SLOW_GRID)

_here = os.path.dirname(os.path.abspath(__file__))


def axial_factor_squared_bc(z_vals, P, L, alpha, rel_h=1e-6):
    alpha = np.asarray(alpha, dtype=float)
    h = np.maximum(rel_h * np.maximum(alpha, 1e-6), 1e-12)

    def D(step):
        return (PK.axial_factor_thermal_bc(z_vals, P, L, alpha + step)
                - PK.axial_factor_thermal_bc(z_vals, P, L, np.maximum(alpha - step, 1e-12))
                ) / (2.0 * step[None, :])

    d = (4.0 * D(h / 2.0) - D(h)) / 3.0
    return -d / (2.0 * alpha[None, :])


sq_path = os.path.join(_here, "slow_sq_dim.dat")


def slow_kernel_sq(P, kappa):
    B = axial_factor_squared_bc(XI_GRID * L_KERNEL, P, L_KERNEL,
                                np.hypot(PK.k, kappa))
    return PK._hankel_transform(RHO_GRID * R_KERNEL, R_KERNEL, B).T


def main_sq():
    tab = np.memmap(sq_path, dtype="float32", mode="w+",
                    shape=(NP, NK_SLOW, NZ, NR))
    print(f"single-kappa slow table: {NP} x {NK_SLOW} x {NZ} x {NR}")
    for iP, P_hat in enumerate(P_HAT_GRID):
        P = P_hat * L_KERNEL
        for ik, kh in enumerate(KAPPA_SLOW_GRID):
            tab[iP, ik] = slow_kernel_sq(P, kh / R_KERNEL).astype(np.float32)
        print(f"  P_hat = {P_hat:.3f}", flush=True)
    tab.flush()
    print(f"wrote {sq_path}  ({os.path.getsize(sq_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main_sq()
