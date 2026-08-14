"""Slow-neutron (epithermal + thermal) one-group table over a UNIFORM source.

Two changes relative to the tables in precompute_kernels.py:

1. The source is the same uniform cylinder used by the cascade and evaporation
   regimes, Theta(R-rho')Theta(z')Theta(P-z'), rather than the cascade axial
   profile. The regimes then differ only in how neutrons are transported out of
   a common source volume. This also makes the table species-independent: it no
   longer references Sigma_t, so one file serves both species. Fitting gamma
   absorbs the difference -- a uniform column has to be longer to cover the same
   axial extent as a cascade profile with its exponential tail, so the fitted
   gamma comes out larger, but the resulting kernel shape agrees to ~3-5 % for
   protons (~10-20 % for carbon, whose tail is a larger share of the source).

2. A denser kappa axis. The two-group kernel is assembled at call time from

       F_2(kappa_f, kappa_s) = [F_1(kappa_f) - F_1(kappa_s)] / (kappa_s^2 - kappa_f^2)

   which is a DIFFERENCE of two interpolated values divided by a small number.
   On the old 20-node grid (spacing 0.026) that amplified linear-interpolation
   error badly when the two kappas fell close together: ~7 % at a separation of
   0.2 grid spacings, against 0.5 % at 0.84 spacings. Since freeing kappa_f and
   kappa_s lets an optimizer wander into that region and be rewarded for
   numerical artifacts, the axis is refined here.

The identity itself is exact, including the Dirichlet boundaries: the Helmholtz
operator is diagonal in the sin(m pi z / L) basis, so the partial fraction runs
in a k-independent denominator mode by mode.
"""
import os

import numpy as np

try:
    from . import precompute_kernels as PK
    from .kernel_config import P_HAT_GRID, XI_GRID, RHO_GRID, NP, NZ, NR, R_KERNEL, L_KERNEL
except ImportError:                                  # running as a script
    import precompute_kernels as PK
    from kernel_config import P_HAT_GRID, XI_GRID, RHO_GRID, NP, NZ, NR, R_KERNEL, L_KERNEL

# refined kappa axis: 60 nodes over the same range the 20-node grid covered
KAPPA_SLOW_GRID = np.linspace(0.0, 0.5, 60)
NK_SLOW = len(KAPPA_SLOW_GRID)

_here = os.path.dirname(os.path.abspath(__file__))
# ----------------------------------------------------------------------
# single-diffusion-length (squared-propagator) variant
# ----------------------------------------------------------------------
def axial_factor_squared_bc(z_vals, P, L, alpha, rel_h=1e-6):
    """Axial factor for the SQUARED Helmholtz operator, B = -(1/2a) dA/da.

    Collapsing kappa_f = kappa_s = kappa turns the two-group system into

        (grad^2 - kappa^2)^2 phi = const * S

    whose propagator is 1/(k^2+kappa^2)^2 = -(1/2 kappa) d/dkappa [1/(k^2+kappa^2)].
    Since kappa enters only through alpha = sqrt(k^2+kappa^2), the same relation
    holds in alpha, so the bounded axial factor is just -(1/2a) dA_BC/da.

    In free space this is e^{-kappa r}/(8 pi kappa): the r from differentiating
    the exponential cancels the 1/r of the Yukawa, which is why the two-group
    kernel is flat near the source and broad enough to match the MC.

    A closed form exists (differentiate N/(a^2 sinh aL) branch by branch), but it
    has to be rewritten with e^{-aL} factored out to stay finite at aL ~ 20, and
    the sinh/cosh product terms are easy to get wrong by a sign. Since the table
    is built once, this differentiates the ALREADY overflow-safe A_BC numerically
    with Richardson extrapolation instead -- accurate to ~1e-10 relative, and the
    key property is preserved: the derivative is taken at build time on the exact
    axial factor, not at call time on an interpolated table.
    """
    alpha = np.asarray(alpha, dtype=float)
    h = np.maximum(rel_h * np.maximum(alpha, 1e-6), 1e-12)

    def D(step):
        return (PK.axial_factor_thermal_bc(z_vals, P, L, alpha + step)
                - PK.axial_factor_thermal_bc(z_vals, P, L, np.maximum(alpha - step, 1e-12))
                ) / (2.0 * step[None, :])

    d = (4.0 * D(h / 2.0) - D(h)) / 3.0            # Richardson
    return -d / (2.0 * alpha[None, :])


sq_path = os.path.join(_here, "slow_sq_dim.dat")


def slow_kernel_sq(P, kappa):
    """Single-kappa slow-neutron kernel on (XI_GRID, RHO_GRID)."""
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
