"""Axial factors, Hankel transform and the evaporation kernel table."""
import numpy as np
from scipy.special import j0, j1
import os

out_dir = os.path.dirname(os.path.abspath(__file__))

evap_path = os.path.join(out_dir, "evap_dim.dat")


try:
    from .kernel_config import (
        R_KERNEL, L_KERNEL,
        P_HAT_GRID, KAPPA_HAT_GRID, XI_GRID, RHO_GRID,
        NP, NK, NZ, NR
    )
except ImportError:
    from kernel_config import (
        R_KERNEL, L_KERNEL,
        P_HAT_GRID, KAPPA_HAT_GRID, XI_GRID, RHO_GRID,
        NP, NK, NZ, NR
    )


NK_INT = 801
K_MAX = 30.0

k = np.linspace(0.0, K_MAX, NK_INT)
dk = k[1] - k[0]


w = np.ones(NK_INT)
w[1:-1:2] = 4.0
w[2:-1:2] = 2.0
w *= dk / 3.0

ALPHA_MIN = 1e-12


def top_hat_factor(R: float) -> np.ndarray:
    th = np.empty_like(k)
    th[0] = 0.0
    th[1:] = j1(k[1:] * R)
    return th


def axial_factor_evap(z_vals: np.ndarray, P: float, alpha: np.ndarray) -> np.ndarray:
    z_vals = np.asarray(z_vals, dtype=float)
    a = np.maximum(alpha[None, :], ALPHA_MIN)

    A = np.zeros((z_vals.size, alpha.size), dtype=np.float64)

    mask_in = (z_vals >= 0.0) & (z_vals <= P)
    mask_above = z_vals > P

    if np.any(mask_in):
        z_in = z_vals[mask_in][:, None]
        A[mask_in] = (
            2.0
            - np.exp(-a * z_in)
            - np.exp(-a * (P - z_in))
        ) / (2.0 * a**2)

    if np.any(mask_above):
        z_ab = z_vals[mask_above][:, None]
        A[mask_above] = (
            np.exp(-a * (z_ab - P))
            * (1.0 - np.exp(-a * P))
        ) / (2.0 * a**2)

    return A


def axial_factor_thermal_bc(z_vals: np.ndarray, P: float, L: float, alpha: np.ndarray) -> np.ndarray:
    z_vals = np.asarray(z_vals, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    Nz = z_vals.size
    Nk = alpha.size
    A = np.zeros((Nz, Nk), dtype=np.float64)

    small = alpha < 1e-8
    large = ~small
    large_idx = np.where(large)[0]

    if np.any(small):
        for iz, z in enumerate(z_vals):
            if 0.0 < z < P:
                A[iz, small] = (z * (P - z) / 2.0) + (z * (L - P) / L) * (P / 2.0)
            elif P <= z < L:
                A[iz, small] = ((L - z) / L) * (P**2 / 2.0)
            else:
                A[iz, small] = 0.0

    if np.any(large):
        a = alpha[large]
        denom = 0.5 * (1.0 - np.exp(-2.0 * a * L))

        idx_in = np.where((z_vals > 0.0) & (z_vals < P))[0]
        idx_ab = np.where((z_vals >= P) & (z_vals < L))[0]

        if idx_in.size:
            z = z_vals[idx_in][:, None]
            aa = a[None, :]

            term1 = 0.5 * (1.0 - np.exp(-2.0 * aa * L))
            term2 = 0.5 * (
                np.exp(-aa * z)
                - np.exp(-aa * (2.0 * L - z))
            )
            term3 = 0.25 * (
                np.exp(-aa * (P - z))
                + np.exp(-aa * (2.0 * L - P - z))
                - np.exp(-aa * (P + z))
                - np.exp(-aa * (2.0 * L - P + z))
            )

            num_scaled = term1 - term2 - term3
            A[np.ix_(idx_in, large_idx)] = num_scaled / (aa**2 * denom)

        if idx_ab.size:
            z = z_vals[idx_ab][:, None]
            aa = a[None, :]

            num_scaled = (
                0.25
                * np.exp(-aa * (z - P))
                * (1.0 - np.exp(-aa * P))**2
                * (1.0 - np.exp(-2.0 * aa * (L - z)))
            )

            A[np.ix_(idx_ab, large_idx)] = num_scaled / (aa**2 * denom)

    return A


def _hankel_transform(r_vals: np.ndarray, R: float, A_ax: np.ndarray) -> np.ndarray:
    top_hat = top_hat_factor(R)
    WT = w * top_hat
    J = j0(r_vals[:, None] * k[None, :])

    return (J * WT[None, :]) @ A_ax.T


def evap_kernel(r_vals, z_vals, R: float, kappa: float, P: float) -> np.ndarray:
    r_vals = np.asarray(r_vals, dtype=float)
    z_vals = np.asarray(z_vals, dtype=float)

    alpha = np.hypot(k, kappa)
    A_ax = axial_factor_evap(z_vals, P=P, alpha=alpha)

    return _hankel_transform(r_vals, R, A_ax)


def main():
    R, L = R_KERNEL, L_KERNEL
    evap_dim = np.memmap(
        evap_path, dtype="float32", mode="w+", shape=(NP, NK, NZ, NR)
    )
    r_phys = RHO_GRID * R
    z_phys = XI_GRID * L
    print(f"Precomputing evaporation kernel: {NP}x{NK} "
          f"(kappa 0-{KAPPA_HAT_GRID[-1]:g})")
    for iP, P_hat in enumerate(P_HAT_GRID):
        P_phys = P_hat * L
        for ik, kappa_hat in enumerate(KAPPA_HAT_GRID):
            evap_dim[iP, ik, :, :] = evap_kernel(
                r_phys, z_phys, R=R, kappa=kappa_hat / R, P=P_phys,
            ).T.astype("float32")
        print(f"  P_hat={P_hat:5.2f}  (P={P_phys:6.2f} cm)", flush=True)
    evap_dim.flush()
    print("Done.")


if __name__ == "__main__":
    main()
