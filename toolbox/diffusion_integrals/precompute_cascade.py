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

Axes: (P_hat, Sigma_h, theta_bar, xi, rho_hat) = (P/L, removal constant, mean
emission angle in degrees, z/L, r/R).

WHY THE ANGULAR AXIS IS theta_bar AND NOT n
-------------------------------------------
The lobe width goes as n^{-1/2}, so d(width)/dn -> 0 and n is a badly
conditioned fit parameter: narrowing the lobe from 17 deg to 12 deg costs
n = 20 -> 40, and every further degree costs more than the last. An optimizer
that wants a forward peak therefore runs n off to infinity, and the rail carries
no information because the parameter has no upper bound to rail against.

Reparametrizing by the MEAN EMISSION ANGLE removes both problems. Exactly,

    <cos theta> = (n+1)/(n+2)   ->   n = (2 cos theta_bar - 1)/(1 - cos theta_bar)

so theta_bar = 60 deg is n = 0 (the flattest forward lobe the family allows) and
theta_bar -> 0 is the forward delta. The whole family maps onto a BOUNDED
interval, sensitivity is uniform along it (2 deg of lobe width per grid step,
against 6.9 deg at the coarse end and 0.68 deg at the fine end of a uniform-n
axis), and a fit that still wants the forward limit now says so as a finite,
readable edge value rather than an unbounded exponent.

Linear interpolation also runs along theta_bar, i.e. along equal increments of
lobe shape, rather than along n where a single step near the top spanned a
larger shape change than ten steps near the bottom.

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

# ------------------------------------------------- angular reparametrization
def theta_to_n(theta_deg):
    """Mean emission angle (deg) -> cos^n exponent. Inverse of n_to_theta."""
    c = np.cos(np.radians(np.asarray(theta_deg, dtype=float)))
    return (2.0 * c - 1.0) / np.maximum(1.0 - c, 1e-15)


def n_to_theta(n):
    """cos^n exponent -> mean emission angle in degrees."""
    n = np.asarray(n, dtype=float)
    return np.degrees(np.arccos((n + 1.0) / (n + 2.0)))


# ---------------------------------------------------------------- grids
P_HAT_GRID_C = np.linspace(0.0, 1.0, 31)
# 1 deg per step. 60 deg is n = 0, the hard edge of the cos^n family; 6 deg is
# n = 180, far narrower than any fit has wanted (the carbon rail sat at 17.3).
#
# The step is 1 deg rather than 2 because a uniform-in-theta axis necessarily
# takes resolution AWAY from the wide end, where the fitted proton lobe sits.
# Interpolation error at n = 3.087, over the field for P = 5-30 cm:
#
#     old uniform-n axis   mean 0.16-0.50 %   peak 5.6 %
#     theta, 2 deg step    mean 0.22-0.26 %   peak 10.4 %
#     theta, 1 deg step    mean 0.06-0.08 %   peak 3.0 %
#
# so at 1 deg the reparametrization is better than the old axis at BOTH ends
# rather than buying the narrow end at the wide end's expense.
THETA_GRID_C = np.linspace(6.0, 60.0, 55)
N_ANG_GRID   = theta_to_n(THETA_GRID_C)          # derived; descending in theta
XI_GRID_C    = np.linspace(0.0, 1.0, 46)
RHO_GRID_C   = np.linspace(0.0, 6.0, 61)
SIGMA_H_GRID = np.concatenate([[0.0], np.geomspace(2e-4, 0.40, 15)])

NP_C, NS_C, NN_C, NZ_C, NR_C = (len(P_HAT_GRID_C), len(SIGMA_H_GRID),
                                len(THETA_GRID_C), len(XI_GRID_C), len(RHO_GRID_C))

_here = os.path.dirname(os.path.abspath(__file__))
cascade_path = os.path.join(_here, "cascade_dim.dat")

# Angular quadrature. The node counts are modest because the grids are ADAPTED
# to each field point: the source subtends only a narrow cone once the field
# point is far downstream, and only a narrow azimuthal wedge once rho > R, so a
# fixed grid would put nearly all its nodes where the integrand vanishes.
_NMU = 200
_NPH = 96
_MU_CLUSTER = 3          # mu nodes go as 1 - (1-mu_min) t^p, clustering at mu=1


def _angular_grid(z, rho, P, R=R_KERNEL):
    """mu and phi nodes covering exactly the directions that can see the source.

    Forward emission needs z' < z, so only the source below the field point is
    visible. For z > P the whole source lies below and the half-cone opening is
    arctan((rho+R)/(z-P)); for z <= P the visible part reaches theta -> 90 deg.
    Azimuthally the ray must pass within R of the axis, giving |sin(phi)| < R/rho
    once rho > R. The integrand is even in phi, so only [0, phi_max] is built and
    the result is doubled.

    One mu grid serves every angular node of the table, so it has to resolve the
    NARROWEST lobe on the axis. At theta_bar = 6 deg (n = 180) the lobe covers
    1 - mu < 0.006, which a uniform 160-node grid samples with a single point --
    that grid read 10.8 % high for field points inside the source. Spacing the
    nodes as 1 - (1-mu_min) t^3 puts ~30 of them inside that lobe while leaving
    the wide-angle region, where the integrand is smooth, amply covered; the
    error drops to <0.1 % at n = 180 and improves at low n as well.
    """
    if z > P:
        d = z - P
        mu_min = d / np.hypot(d, rho + R)
    else:
        mu_min = 0.0
    mu_min = max(mu_min, 1e-9)
    t = np.linspace(0.0, 1.0, _NMU)
    mu = (1.0 - (1.0 - mu_min) * t ** _MU_CLUSTER)[::-1]
    mu = np.clip(mu, mu_min, 1.0 - 1e-12)
    ph_max = np.pi if rho <= R else np.arcsin(min(R / rho, 1.0))
    ph = np.linspace(0.0, ph_max, _NPH)
    return mu, ph


def _chord_attenuated(s1, s2, ok, Sigma_h):
    """(e^{-Sigma_h s1} - e^{-Sigma_h s2}) / Sigma_h, with the Sigma_h -> 0 limit.

    At Sigma_h = 0 the quotient is 0/0; the removable limit is the unattenuated
    chord length s2 - s1. Evaluating it naively left the whole Sigma_h = 0 slab
    of the table as NaN.
    """
    Sh = np.asarray(Sigma_h, dtype=float)
    safe = np.where(Sh > 0.0, Sh, 1.0)
    att = (np.exp(-safe * s1) - np.exp(-safe * s2)) / safe
    return np.where(ok, np.where(Sh > 0.0, att, s2 - s1), 0.0)


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
    """F(z, rho) for one field point and one parameter set. Reference implementation.

    Takes the exponent n directly; use theta_to_n to enter in mean-angle terms.
    """
    if P <= 0.0:
        return 0.0
    s1, s2, ok, mu, ph = _chords(z, rho, P, R)
    chord = _chord_attenuated(np.where(ok, s1, 0.0), np.where(ok, s2, 0.0),
                              ok, Sigma_h)
    inner = 2.0 * np.trapezoid(chord, ph, axis=1)       # over phi, even -> doubled
    return float(np.trapezoid(mu ** n * inner, mu) / (4.0 * np.pi))


def main():
    tab = np.memmap(cascade_path, dtype="float32", mode="w+",
                    shape=(NP_C, NS_C, NN_C, NZ_C, NR_C))
    z_grid = XI_GRID_C * L_KERNEL
    r_grid = RHO_GRID_C * R_KERNEL
    Sh = SIGMA_H_GRID[:, None, None]

    print(f"cascade table: angular axis is theta_bar, "
          f"{THETA_GRID_C[0]:.0f}-{THETA_GRID_C[-1]:.0f} deg in {NN_C} nodes "
          f"(n = {N_ANG_GRID[0]:.1f} down to {N_ANG_GRID[-1]:.1f})")
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
                chord = _chord_attenuated(s1[None], s2[None], ok[None], Sh)
                inner = 2.0 * np.trapezoid(chord, ph, axis=2)     # (NS_C, NMU)
                # ... and all angular nodes at once; only the mu weight varies
                mu_pow = mu[None, :] ** N_ANG_GRID[:, None]       # (NN_C, NMU)
                tab[iP, :, :, iz, ir] = np.trapezoid(
                    mu_pow[None, :, :] * inner[:, None, :], mu, axis=2) / (4.0 * np.pi)
        print(f"  P_hat = {P_hat:.3f}  (P = {P:5.1f} cm)  done", flush=True)
    tab.flush()
    print(f"wrote {cascade_path}  shape {(NP_C, NS_C, NN_C, NZ_C, NR_C)}  "
          f"({os.path.getsize(cascade_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
