"""Coupled ballistic cascade kernel.

The factorized cascade -- a Gaussian-smeared disk in rho times an axial
buildup/attenuation profile in z -- cannot reproduce the MC field beyond
rho ~ 3 cm, because a neutron reaching large rho was emitted obliquely and
therefore arrives DEEPER than its emission point. A rho-independent axial
factor has no way to express that, so the z-profile peak stays put while the
MC peak migrates (15 -> 18 cm between rho = 1 and 5.5 cm at 170 MeV).

This module tabulates the coupled first-flight integral instead,

    phi(rho, z; P, n) = int_0^P dz' S(z') g(theta) exp(-Sigma_h s) / (4 pi s^2)

with s = sqrt(rho^2 + (z-z')^2), cos(theta) = (z-z')/s and a forward lobe
g(theta) = cos^n(theta) restricted to the forward hemisphere. The source
column is uniform over [0, P]; because it factorizes as disk(r') S(z'), the
finite source radius enters exactly as a lateral convolution with the disk,
done here in Hankel space (multiply by J1(kR)/(kR)).

Setting g = delta(forward) recovers the axial law used previously, so the old
cascade profile is the forward limit of this kernel rather than a separate
ingredient.

Sigma_h and n are degenerate to a degree -- both steepen the lateral falloff --
but the residual surface has a clear minimum in Sigma_h, so it is tabulated
rather than assumed.

Axes: (P_hat, Sigma_h, n, xi, rho_hat) = (P/L, removal constant, angular
exponent, z/L, r/R).
"""
import os

import numpy as np
from scipy.special import j0, j1

try:
    from .kernel_config import R_KERNEL, L_KERNEL
except ImportError:                                  # running as a script
    from kernel_config import R_KERNEL, L_KERNEL

# ---------------------------------------------------------------- grids
P_HAT_GRID_C = np.linspace(0.0, 1.0, 31)
N_ANG_GRID   = np.linspace(0.0, 8.0, 17)
XI_GRID_C    = np.linspace(0.0, 1.0, 46)
RHO_GRID_C   = np.linspace(0.0, 6.0, 61)

NP_C, NN_C, NZ_C, NR_C = (len(P_HAT_GRID_C), len(N_ANG_GRID),
                          len(XI_GRID_C), len(RHO_GRID_C))

# Sigma_h is a REMOVAL constant, not sigma_tot: a fast neutron scattering
# elastically off oxygen stays in the band, so removal is largely cancelled by
# in-scattering and the effective attenuation length is far longer than the
# total-cross-section mfp (~15 cm). It is identifiable from the data -- holding
# it at sigma_tot costs a factor two in residual -- so it gets its own axis.
SIGMA_H_GRID = np.concatenate([[0.002], np.geomspace(0.004, 0.40, 11)])
NS_C = len(SIGMA_H_GRID)

_here = os.path.dirname(os.path.abspath(__file__))
cascade_path = os.path.join(_here, "cascade_dim.dat")

# integration nodes
_NZP = 160                       # source column
_NRE = 220                       # extended radial grid for the Hankel step
_NK = 1400
_R_EXT = np.linspace(1e-4, 40.0, _NRE)
_K = np.linspace(1e-6, 60.0, _NK)
_DK = _K[1] - _K[0]
_DISK = j1(_K * R_KERNEL) / (_K * R_KERNEL)
_J0_FWD = j0(np.outer(_R_EXT, _K))        # (_NRE, _NK)
_J0_BWD = j0(np.outer(RHO_GRID_C * R_KERNEL, _K))   # (NR_C, _NK)
_DRE = _R_EXT[1] - _R_EXT[0]


def _disk_convolve(prof_ext):
    """Lateral convolution with the source disk, exact for a factorized source.

    prof_ext : (NZ_C, _NRE) line-source profile -> (NZ_C, NR_C) on the output grid
    """
    Fk = (prof_ext * _R_EXT[None, :]) @ _J0_FWD * _DRE        # (NZ_C, _NK)
    return (Fk * (_K * _DISK)[None, :]) @ _J0_BWD.T * _DK     # (NZ_C, NR_C)


def cascade_kernel(P, n, Sigma_h):
    """Coupled kernel on (XI_GRID_C, RHO_GRID_C) for one (P, n, Sigma_h)."""
    z = XI_GRID_C * L_KERNEL
    if P <= 0:
        return np.zeros((NZ_C, NR_C))
    zp = np.linspace(0.0, P, _NZP)
    dz = z[:, None, None] - zp[None, None, :]                 # (NZ_C, 1, _NZP)
    s = np.sqrt(_R_EXT[None, :, None] ** 2 + dz ** 2)         # (NZ_C, _NRE, _NZP)
    base = np.exp(-Sigma_h * s) / (4.0 * np.pi * s ** 2)
    ct = np.clip(dz / s, 0.0, None)                           # forward hemisphere
    line = np.trapezoid(base * ct ** n, zp, axis=2)
    return _disk_convolve(line)


def main():
    tab = np.memmap(cascade_path, dtype="float32", mode="w+",
                    shape=(NP_C, NS_C, NN_C, NZ_C, NR_C))
    z = XI_GRID_C * L_KERNEL
    zp_unit = np.linspace(0.0, 1.0, _NZP)
    for iP, P_hat in enumerate(P_HAT_GRID_C):
        P = P_hat * L_KERNEL
        if P <= 0:
            tab[iP] = 0.0
            continue
        # s and cos(theta) depend only on P, so build them once per P and
        # reuse across the whole n axis -- the n loop is then just a power
        zp = zp_unit * P
        dz = z[:, None, None] - zp[None, None, :]
        s = np.sqrt(_R_EXT[None, :, None] ** 2 + dz ** 2)
        geo = 1.0 / (4.0 * np.pi * s ** 2)
        ct = np.clip(dz / s, 0.0, None)
        for iS, Sh in enumerate(SIGMA_H_GRID):
            base = np.exp(-Sh * s) * geo
            for iN, n in enumerate(N_ANG_GRID):
                line = np.trapezoid(base * ct ** n, zp, axis=2)
                tab[iP, iS, iN] = _disk_convolve(line).astype(np.float32)
        print(f"  P_hat = {P_hat:.3f}  (P = {P:5.1f} cm)  done", flush=True)
    tab.flush()
    print(f"wrote {cascade_path}  shape {(NP_C, NS_C, NN_C, NZ_C, NR_C)}  "
          f"({os.path.getsize(cascade_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
