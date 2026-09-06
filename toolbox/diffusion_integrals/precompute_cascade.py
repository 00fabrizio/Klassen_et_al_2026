"""Builds the coupled cascade kernel table."""
import os

import numpy as np

try:
    from .kernel_config import R_KERNEL, L_KERNEL
except ImportError:
    from kernel_config import R_KERNEL, L_KERNEL


def theta_to_n(theta_deg):
    c = np.cos(np.radians(np.asarray(theta_deg, dtype=float)))
    return (2.0 * c - 1.0) / np.maximum(1.0 - c, 1e-15)


def n_to_theta(n):
    n = np.asarray(n, dtype=float)
    return np.degrees(np.arccos((n + 1.0) / (n + 2.0)))


P_HAT_GRID_C = np.linspace(0.0, 1.0, 31)


THETA_GRID_C = np.linspace(6.0, 60.0, 55)
N_ANG_GRID   = theta_to_n(THETA_GRID_C)
XI_GRID_C    = np.linspace(0.0, 1.0, 46)
RHO_GRID_C   = np.linspace(0.0, 6.0, 61)
SIGMA_H_GRID = np.concatenate([[0.0], np.geomspace(2e-4, 0.40, 15)])

NP_C, NS_C, NN_C, NZ_C, NR_C = (len(P_HAT_GRID_C), len(SIGMA_H_GRID),
                                len(THETA_GRID_C), len(XI_GRID_C), len(RHO_GRID_C))

_here = os.path.dirname(os.path.abspath(__file__))
cascade_path = os.path.join(_here, "cascade_dim.dat")


_NMU = 200
_NPH = 96
_MU_CLUSTER = 3


def _angular_grid(z, rho, P, R=R_KERNEL):
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
    Sh = np.asarray(Sigma_h, dtype=float)
    safe = np.where(Sh > 0.0, Sh, 1.0)
    att = (np.exp(-safe * s1) - np.exp(-safe * s2)) / safe
    return np.where(ok, np.where(Sh > 0.0, att, s2 - s1), 0.0)


def _chords(z, rho, P, R=R_KERNEL, grid=None):
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
    if P <= 0.0:
        return 0.0
    s1, s2, ok, mu, ph = _chords(z, rho, P, R)
    chord = _chord_attenuated(np.where(ok, s1, 0.0), np.where(ok, s2, 0.0),
                              ok, Sigma_h)
    inner = 2.0 * np.trapezoid(chord, ph, axis=1)
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

                chord = _chord_attenuated(s1[None], s2[None], ok[None], Sh)
                inner = 2.0 * np.trapezoid(chord, ph, axis=2)

                mu_pow = mu[None, :] ** N_ANG_GRID[:, None]
                tab[iP, :, :, iz, ir] = np.trapezoid(
                    mu_pow[None, :, :] * inner[:, None, :], mu, axis=2) / (4.0 * np.pi)
        print(f"  P_hat = {P_hat:.3f}  (P = {P:5.1f} cm)  done", flush=True)
    tab.flush()
    print(f"wrote {cascade_path}  shape {(NP_C, NS_C, NN_C, NZ_C, NR_C)}  "
          f"({os.path.getsize(cascade_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
