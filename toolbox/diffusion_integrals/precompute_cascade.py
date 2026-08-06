"""Coupled ballistic cascade kernel, evaluated in field-point spherical coordinates.

The factorized cascade -- a Gaussian-smeared disk in rho times an axial
build-up/attenuation profile -- cannot reproduce the MC field beyond rho ~ 3 cm,
because a neutron reaching large rho was emitted obliquely and therefore arrives
DEEPER than its emission point. A rho-independent axial factor has no way to
express that, so the modelled z-profile peak stays put while the MC peak migrates
(14 -> 18 cm between rho = 1 and 5.5 cm at 170 MeV).

The coupled first-flight integral over the source cylinder V is

    F(rho, z) = int_V d3x' g(theta) exp(-Sigma_h s) / (4 pi s^2)

with s = |x - x'| and cos(theta) = (z - z')/s, and a forward lobe
g(theta) = cos^n(theta) restricted to the forward hemisphere. Setting
n -> infinity recovers the axial law of the separable model, so that profile is
the forward limit of this kernel rather than a separate ingredient.

WHY SPHERICAL COORDINATES
-------------------------
Written in cylindrical coordinates about the beam axis the integrand carries a
1/s^2 singularity wherever the field point lies inside the source. It is
integrable, but only awkwardly: the previous implementation evaluated a line
source on a radial grid and then convolved with the disk in Hankel space, which
is exact in principle yet inherits that singularity in the intermediate profile.
On a 0.18 cm radial grid it came out ~12 % low inside the source region.

Putting the origin at the FIELD point removes the singularity outright. With
x' = x - s*u, the volume element is d3x' = s^2 ds dOmega and the s^2 cancels the
1/s^2 exactly:

    F = (1/4pi) int dOmega g(theta) int ds exp(-Sigma_h s)
      = (1/4pi Sigma_h) int dOmega cos^n(theta) [e^{-Sigma_h s1} - e^{-Sigma_h s2}]

Every direction contributes the attenuated CHORD the ray cuts through the source,
between entry s1 and exit s2. The s integral is closed form; only a 2D angular
quadrature remains, and nothing diverges -- a nearby source element subtends a
correspondingly small solid angle.

Chord limits for the cylinder (rho' <= R, 0 <= z' <= P), with mu = cos(theta):

    axial   0 <= z - s mu <= P            ->  s in [ (z-P)/mu , z/mu ]
    radial  |rho_vec - s u_perp| <= R     ->  sin^2(theta) s^2
                                              - 2 rho sin(theta) cos(phi) s
                                              + (rho^2 - R^2) <= 0
            roots  s_pm = [ rho cos(phi) +- sqrt(R^2 - rho^2 sin^2(phi)) ] / sin(theta)

intersected, and empty when R^2 < rho^2 sin^2(phi) (the ray misses the cylinder).
No special case is needed as sin(theta) -> 0: for rho < R the radial roots run to
-/+ infinity and the axial window takes over, while for rho > R the entry root
runs to +infinity and the chord closes to zero.

Axes: (P_hat, Sigma_h, n, xi, rho_hat) = (P/L, removal constant, angular
exponent, z/L, r/R).

Sigma_h is a REMOVAL constant, not sigma_tot: a fast neutron scattering
elastically off oxygen stays in the band, so removal is largely cancelled by
in-scattering and the effective attenuation length is far longer than the
total-cross-section mfp (~15 cm). Sigma_h and n are partly degenerate -- both
steepen the lateral falloff -- but the residual surface has a clear minimum, so
it is tabulated rather than assumed.
"""
import os

import numpy as np

try:
    from .kernel_config import R_KERNEL, L_KERNEL
except ImportError:                                  # running as a script
    from kernel_config import R_KERNEL, L_KERNEL

# ---------------------------------------------------------------- grids
P_HAT_GRID_C = np.linspace(0.0, 1.0, 31)
N_ANG_GRID   = np.linspace(0.0, 8.0, 17)
XI_GRID_C    = np.linspace(0.0, 1.0, 46)
RHO_GRID_C   = np.linspace(0.0, 6.0, 61)
SIGMA_H_GRID = np.concatenate([[0.002], np.geomspace(0.004, 0.40, 11)])

NP_C, NS_C, NN_C, NZ_C, NR_C = (len(P_HAT_GRID_C), len(SIGMA_H_GRID),
                                len(N_ANG_GRID), len(XI_GRID_C), len(RHO_GRID_C))

_here = os.path.dirname(os.path.abspath(__file__))
cascade_path = os.path.join(_here, "cascade_dim.dat")

# Angular quadrature. The node counts are modest because the grids are ADAPTED
# to each field point: the source subtends only a narrow cone once the field
# point is far downstream, and only a narrow azimuthal wedge once rho > R, so a
# fixed grid would put nearly all its nodes where the integrand vanishes.
_NMU = 96
_NPH = 96


def _angular_grid(z, rho, P, R=R_KERNEL):
    """mu and phi nodes covering exactly the directions that can see the source.

    Forward emission needs z' < z, so only the source below the field point is
    visible. For z > P the whole source lies below and the half-cone opening is
    arctan((rho+R)/(z-P)); for z <= P the visible part reaches theta -> 90 deg.
    Azimuthally the ray must pass within R of the axis, giving |sin(phi)| < R/rho
    once rho > R. The integrand is even in phi, so only [0, phi_max] is built and
    the result is doubled.
    """
    if z > P:
        d = z - P
        mu_min = d / np.hypot(d, rho + R)
    else:
        mu_min = 0.0
    mu = np.linspace(max(mu_min, 1e-9), 1.0 - 1e-12, _NMU)
    ph_max = np.pi if rho <= R else np.arcsin(min(R / rho, 1.0))
    ph = np.linspace(0.0, ph_max, _NPH)
    return mu, ph


def _chords(z, rho, P, R=R_KERNEL, grid=None):
    """Entry/exit distances of each ray through the source cylinder.

    Returns (s1, s2, valid, mu, phi); the chord is empty where ``valid`` is False.
    """
    mu_1d, ph_1d = _angular_grid(z, rho, P, R) if grid is None else grid
    mu = mu_1d[:, None]
    st = np.maximum(np.sqrt(1.0 - mu_1d ** 2)[:, None], 1e-12)
    cph = np.cos(ph_1d)[None, :]
    sph = np.sin(ph_1d)[None, :]

    s_ax_lo = np.maximum((z - P) / mu, 0.0)
    s_ax_hi = z / mu

    disc = R ** 2 - (rho * sph) ** 2
    hit = disc > 0.0
    sq = np.sqrt(np.where(hit, disc, 0.0))
    s_r_lo = np.where(hit, np.maximum((rho * cph - sq) / st, 0.0), 0.0)
    s_r_hi = np.where(hit, (rho * cph + sq) / st, -1.0)

    s1 = np.maximum(s_ax_lo, s_r_lo)
    s2 = np.minimum(s_ax_hi, s_r_hi)
    return s1, s2, hit & (s2 > s1), mu_1d, ph_1d


def cascade_kernel_point(z, rho, P, n, Sigma_h, R=R_KERNEL):
    """F(z, rho) for one field point and one parameter set. Reference implementation."""
    if P <= 0.0:
        return 0.0
    s1, s2, ok, mu, ph = _chords(z, rho, P, R)
    chord = np.where(ok, (np.exp(-Sigma_h * np.where(ok, s1, 0.0))
                          - np.exp(-Sigma_h * np.where(ok, s2, 0.0))) / Sigma_h, 0.0)
    inner = 2.0 * np.trapezoid(chord, ph, axis=1)       # over phi, even -> doubled
    return float(np.trapezoid(mu ** n * inner, mu) / (4.0 * np.pi))


def main():
    tab = np.memmap(cascade_path, dtype="float32", mode="w+",
                    shape=(NP_C, NS_C, NN_C, NZ_C, NR_C))
    z_grid = XI_GRID_C * L_KERNEL
    r_grid = RHO_GRID_C * R_KERNEL
    Sh = SIGMA_H_GRID[:, None, None]

    for iP, P_hat in enumerate(P_HAT_GRID_C):
        P = P_hat * L_KERNEL
        if P <= 0.0:
            tab[iP] = 0.0
            continue
        for iz, z in enumerate(z_grid):
            for ir, rho in enumerate(r_grid):
                s1, s2, ok, mu, ph = _chords(z, rho, P)
                if not ok.any():
                    tab[iP, :, :, iz, ir] = 0.0
                    continue
                s1 = np.where(ok, s1, 0.0); s2 = np.where(ok, s2, 0.0)
                # all Sigma_h at once; the chord geometry is shared
                chord = np.where(ok[None], (np.exp(-Sh * s1[None])
                                            - np.exp(-Sh * s2[None])) / Sh, 0.0)
                inner = 2.0 * np.trapezoid(chord, ph, axis=2)     # (NS_C, NMU)
                # ... and all n at once; only the mu weight depends on n
                mu_pow = mu[None, :] ** N_ANG_GRID[:, None]       # (NN_C, NMU)
                tab[iP, :, :, iz, ir] = np.trapezoid(
                    mu_pow[None, :, :] * inner[:, None, :], mu, axis=2) / (4.0 * np.pi)
        print(f"  P_hat = {P_hat:.3f}  (P = {P:5.1f} cm)  done", flush=True)
    tab.flush()
    print(f"wrote {cascade_path}  shape {(NP_C, NS_C, NN_C, NZ_C, NR_C)}  "
          f"({os.path.getsize(cascade_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
