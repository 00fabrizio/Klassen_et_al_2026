#!/usr/bin/env python3
import numpy as np
from scipy.special import j0, j1
import os

out_dir = os.path.dirname(os.path.abspath(__file__))

evap_path = os.path.join(out_dir, "evap_dim.dat")
def therm_path_for(species):
    return os.path.join(out_dir, f"therm_dim_{species}.dat")

try:                                    # imported as part of the package
    from .kernel_config import (
        R_KERNEL, L_KERNEL,
        P_HAT_GRID, KAPPA_HAT_GRID, KAPPA_HAT_GRID_TH, XI_GRID, RHO_GRID,
        NP, NK, NK_TH, NZ, NR
    )
except ImportError:                     # run directly as a script
    from kernel_config import (
        R_KERNEL, L_KERNEL,
        P_HAT_GRID, KAPPA_HAT_GRID, KAPPA_HAT_GRID_TH, XI_GRID, RHO_GRID,
        NP, NK, NK_TH, NZ, NR
    )

# -----------------------
# Global integration setup
# -----------------------
NK_INT = 801
K_MAX = 30.0  # 1/cm

k = np.linspace(0.0, K_MAX, NK_INT)
dk = k[1] - k[0]

# Simpson weights
w = np.ones(NK_INT)
w[1:-1:2] = 4.0
w[2:-1:2] = 2.0
w *= dk / 3.0

ALPHA_MIN = 1e-12


# -----------------------
# Radial factor
# -----------------------
def top_hat_factor(R: float) -> np.ndarray:
    """
    Manuscript-consistent radial factor after radial source integration:
        J1(kR)
    The factor R is absorbed in the fitted amplitude A_ev / A_ep/th.
    """
    th = np.empty_like(k)
    th[0] = 0.0  # J1(0) = 0
    th[1:] = j1(k[1:] * R)
    return th


# -----------------------
# Axial factors from manuscript
# -----------------------
def axial_factor_evap(z_vals: np.ndarray, P: float, alpha: np.ndarray) -> np.ndarray:
    """
    Eq. (24):
        A_free(z,P,alpha) = {...} / (2 alpha^2)
    """
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
    """
    Stable thermal / bounded axial factor with overall 1/alpha^2 scaling.
    Numerically safe for large alpha*L.
    """
    z_vals = np.asarray(z_vals, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    Nz = z_vals.size
    Nk = alpha.size
    A = np.zeros((Nz, Nk), dtype=np.float64)

    small = alpha < 1e-8
    large = ~small
    large_idx = np.where(large)[0]

    # ---- small-alpha limit
    if np.any(small):
        for iz, z in enumerate(z_vals):
            if 0.0 < z < P:
                A[iz, small] = (z * (P - z) / 2.0) + (z * (L - P) / L) * (P / 2.0)
            elif P <= z < L:
                A[iz, small] = ((L - z) / L) * (P**2 / 2.0)
            else:
                A[iz, small] = 0.0

    # ---- stable evaluation for alpha > 0
    if np.any(large):
        a = alpha[large]                      # (Nk_large,)
        denom = 0.5 * (1.0 - np.exp(-2.0 * a * L))   # scaled sinh(aL)

        idx_in = np.where((z_vals > 0.0) & (z_vals < P))[0]
        idx_ab = np.where((z_vals >= P) & (z_vals < L))[0]

        # z in (0, P)
        if idx_in.size:
            z = z_vals[idx_in][:, None]      # (n_in, 1)
            aa = a[None, :]                  # (1, Nk_large)

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

        # z in [P, L)
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

# -----------------------
# Two-group thermal source and axial factor
# -----------------------
# Cascade attenuation of the thermal source. The thermal source IS the cascade,
# so this must equal the cascade Sigma_t of the species being modelled. Giving
# it a table axis would cost ~90 MB, so instead one table is built per species
# and SIGMA_T_SRC selects which. Rebuild when the cascade fit changes.
SIGMA_T_BY_SPECIES = {
    "proton": 0.346286,
    "carbon": 0.125166,
}
SIGMA_T_SRC = SIGMA_T_BY_SPECIES["proton"]   # default; main() overrides per species


def axial_factor_cascade_bc(z_vals, P, L, alpha, Sigma_t=None):
    """int_0^L G_Dirichlet(z, z'; alpha) f_cas(z') dz'   in closed form.

    Source is the cascade axial profile,
        f_cas = 1 - exp(-Sigma_t z')                         0 <= z' <= P
              = (1 - exp(-Sigma_t P)) exp(-Sigma_t (z'-P))   z' > P

    so only two antiderivatives are needed (note the sign difference):
        int e^{-S z} sinh(a z)     dz = e^{-S z} [S sinh(a z)     + a cosh(a z)]     / (a^2 - S^2)
        int e^{-S z} sinh(a(L-z))  dz = e^{-S z} [S sinh(a(L-z))  - a cosh(a(L-z))]  / (a^2 - S^2)

    Written with e^{aL} factored out so every exponent is <= 0:
        A = { [1-e^{-2a(L-z)}] Uh + [1-e^{-2az}] Vh } / ( a [1-e^{-2aL}] )
    with Uh = e^{-az} int_0^z sinh(az')f dz' and Vh = e^{-a(L-z)} int_z^L sinh(a(L-z'))f dz',
    both O(1). Returns (Nz, Nalpha).
    """
    if Sigma_t is None:                 # resolved at call time, not def time
        Sigma_t = SIGMA_T_SRC
    z = np.asarray(z_vals, dtype=float)[:, None]
    a = np.maximum(np.asarray(alpha, dtype=float)[None, :], ALPHA_MIN)
    S = Sigma_t
    den = a**2 - S**2
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)      # alpha == Sigma_t is removable

    def e(x):                                            # exp of a non-positive argument
        return np.exp(np.minimum(x, 0.0))

    def Is_hat(b, ref):        # e^{-a ref} int_0^b sinh(a z') dz'
        return (e(a * (b - ref)) + e(-a * (b + ref))) / (2 * a) - e(-a * ref) / a

    def Js_hat(lo, hi, ref):   # e^{-a ref} int_lo^hi e^{-S z'} sinh(a z') dz'
        def term(x):
            sh = (e(a * (x - ref)) - e(-a * (x + ref))) / 2
            ch = (e(a * (x - ref)) + e(-a * (x + ref))) / 2
            return np.exp(-S * x) * (S * sh + a * ch) / den
        return term(hi) - term(lo)

    def Ic_hat(lo, hi, ref):   # e^{-a ref} int_lo^hi sinh(a(L - z')) dz'
        def ch(x):
            return (e(a * (L - x - ref)) + e(-a * (L - x + ref))) / 2
        return (ch(lo) - ch(hi)) / a

    def Jc_hat(lo, hi, ref):   # e^{-a ref} int_lo^hi e^{-S z'} sinh(a(L - z')) dz'
        def term(x):
            sh = (e(a * (L - x - ref)) - e(-a * (L - x + ref))) / 2
            ch = (e(a * (L - x - ref)) + e(-a * (L - x + ref))) / 2
            return np.exp(-S * x) * (S * sh - a * ch) / den
        return term(hi) - term(lo)

    in_P = (z <= P)
    tail = (1.0 - np.exp(-S * P)) * np.exp(S * P)        # = e^{S P} - 1

    Uh = np.where(in_P,
                  Is_hat(z, z) - Js_hat(0.0, z, z),
                  Is_hat(P, z) - Js_hat(0.0, P, z) + tail * Js_hat(P, z, z))
    Vh = np.where(in_P,
                  Ic_hat(z, P, L - z) - Jc_hat(z, P, L - z) + tail * Jc_hat(P, L, L - z),
                  tail * Jc_hat(z, L, L - z))

    return ((1 - e(-2 * a * (L - z))) * Uh
            + (1 - e(-2 * a * z)) * Vh) / (a * (1 - e(-2 * a * L)))


# -----------------------
# Hankel kernel evaluator
# -----------------------
def _hankel_transform(r_vals: np.ndarray, R: float, A_ax: np.ndarray) -> np.ndarray:
    """
    Radial Hankel transform shared by both kernels:
        integral dk J0(k r) J1(kR) A_axial(...)
    Returns I(r,z) with shape (Nr, Nz).
    """
    top_hat = top_hat_factor(R)               # J1(kR)
    WT = w * top_hat                          # (Nk,)
    J = j0(r_vals[:, None] * k[None, :])      # (Nr, Nk)

    return (J * WT[None, :]) @ A_ax.T         # (Nr, Nz)


def evap_kernel(r_vals, z_vals, R: float, kappa: float, P: float) -> np.ndarray:
    """Evaporation kernel: free-space axial factor (Eqs. 23-24)."""
    r_vals = np.asarray(r_vals, dtype=float)
    z_vals = np.asarray(z_vals, dtype=float)

    alpha = np.hypot(k, kappa)
    A_ax = axial_factor_evap(z_vals, P=P, alpha=alpha)

    return _hankel_transform(r_vals, R, A_ax)


def thermal_kernel(r_vals, z_vals, R: float, kappa: float, P: float,
                   L: float) -> np.ndarray:
    """ONE-group kernel over a cascade-shaped source.

    This is what gets tabulated. The two-group result for any pair
    (kappa_f, kappa_s) follows from the difference identity

        F_2(kappa_f, kappa_s) = [F_1(kappa_f) - F_1(kappa_s)] / (kappa_s^2 - kappa_f^2)

    which holds because kappa_s^2 - kappa_f^2 carries no k dependence and the
    Hankel integral is linear. Both kappas are therefore lookups on the SAME
    axis -- giving kappa_f its own table dimension would cost 126 MB for
    nothing. The differencing is done at call time in diffusion_kernels.py.
    """
    r_vals = np.asarray(r_vals, dtype=float)
    z_vals = np.asarray(z_vals, dtype=float)

    A_ax = axial_factor_cascade_bc(z_vals, P=P, L=L, alpha=np.hypot(k, kappa))

    return _hankel_transform(r_vals, R, A_ax)


# -----------------------
# Precomputation
# -----------------------
def main(species="proton"):
    global SIGMA_T_SRC
    SIGMA_T_SRC = SIGMA_T_BY_SPECIES[species]
    therm_path = therm_path_for(species)
    R = R_KERNEL
    L = L_KERNEL

    evap_dim = np.memmap(
        evap_path, dtype="float32", mode="w+", shape=(NP, NK, NZ, NR)
    )
    therm_dim = np.memmap(
        therm_path, dtype="float32", mode="w+", shape=(NP, NK_TH, NZ, NR)
    )

    r_phys = RHO_GRID * R
    z_phys = XI_GRID * L

    print(f"Precomputing kernels: evaporation {NP}x{NK} "
          f"(kappa 0-{KAPPA_HAT_GRID[-1]:g}), "
          f"thermal {NP}x{NK_TH} (kappa 0-{KAPPA_HAT_GRID_TH[-1]:g}, "
          f"one-group over a cascade source, Sigma_t={SIGMA_T_SRC})")

    for iP, P_hat in enumerate(P_HAT_GRID):
        P_phys = P_hat * L

        for ik, kappa_hat in enumerate(KAPPA_HAT_GRID):
            evap_dim[iP, ik, :, :] = evap_kernel(
                r_phys, z_phys, R=R, kappa=kappa_hat / R, P=P_phys,
            ).T.astype("float32")

        for ik, kappa_hat in enumerate(KAPPA_HAT_GRID_TH):
            therm_dim[iP, ik, :, :] = thermal_kernel(
                r_phys, z_phys, R=R, kappa=kappa_hat / R, P=P_phys, L=L,
            ).T.astype("float32")

        print(f"  P_hat={P_hat:5.2f}  (P={P_phys:6.2f} cm)", flush=True)

    evap_dim.flush()
    therm_dim.flush()
    print("Done.")


if __name__ == "__main__":
    import sys
    for sp in (sys.argv[1:] or list(SIGMA_T_BY_SPECIES)):
        print(f"=== {sp} ===")
        main(sp)