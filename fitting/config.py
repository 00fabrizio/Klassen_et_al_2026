"""Fit configuration per species: data, starting values, bounds, holds, optimiser.

Units are the model's: neutron energies (Epk) in GeV, primary energies (E_th) in
MeV (proton) or MeV/u (carbon), lengths in cm.

Bounds come from two sources:
  * physics: amplitudes, gamma, Sigma, kappa, a, w_c, n and E_th are >= 0;
  * the kernel tables, whose interpolators clip (cascade, slow) or return 0
    (evaporation) outside their grids, so a bound there marks the end of the
    table, not a physical limit:
      Sigma      <= 0.40 /cm   SIGMA_H_GRID      (precompute_cascade.py)
      theta_bar  in [6, 60] deg THETA_GRID_C     (precompute_cascade.py)
      n_ang      in [0, 179.9]  the same grid expressed as n_ang
      kappa_ev   <= 1.5 /cm    KAPPA_HAT_GRID    (kernel_config.py, R_KERNEL = 1)
      kappa_slow <= 0.5 /cm    KAPPA_SLOW_GRID   (precompute_slow.py)
  * gamma_i <= 9: P = gamma * xt is clipped at L = 45 cm, so beyond a few the
    parameter stops acting (flat direction); 9 is the value used in every
    earlier fit.
"""
import numpy as np

from toolbox.diffusion_integrals.precompute_cascade import (
    SIGMA_H_GRID, THETA_GRID_C, theta_to_n,
)

INF = np.inf

# Column order of fitting_params/params_{species}.csv and of the model's
# entry points (model.Model.spectral_energy_fluence[_theta]).
PARAMS = {
    'proton': ['A1', 'gamma1', 'Sigma', 'n_ang', 'd1', 'a', 'w_c',
               'A2', 'gamma2', 'd2', 'kappa_ev', 'Epk',
               'A3', 'gamma3', 'A4', 'gamma4', 'kappa_slow', 'E_th', 'n'],
    'carbon': ['A1', 'gamma1', 'Sigma', 'theta_bar', 'd1', 'a', 'w_c',
               'A2', 'gamma2', 'd2', 'kappa_ev', 'Epk',
               'A3', 'gamma3', 'A4', 'gamma4', 'kappa_slow', 'E_th', 'n'],
}

ENTRY = {'proton': 'spectral_energy_fluence',          # cascade lobe as n_ang
         'carbon': 'spectral_energy_fluence_theta'}    # cascade lobe as theta_bar

# Training set: 10 of the 50 simulated primary energies, evenly spaced in index
# (0, 5, 10, 16, 21, 27, 32, 38, 43, 49). The other 40 are the holdout.
TRAIN_IDX = np.linspace(0, 49, 10).astype(int)

# Grid thinning: every second depth, radius (full 0-5.5 cm range) and neutron
# energy bin, ~80k points per primary energy.
THIN = dict(z=2, rho=2, en=2)

_COMMON_BOUNDS = {
    'A1': (0, INF), 'gamma1': (0, 9),
    'Sigma': (float(SIGMA_H_GRID[0]), float(SIGMA_H_GRID[-1])),
    'd1': (-INF, INF), 'a': (0, INF), 'w_c': (0, INF),
    'A2': (0, INF), 'gamma2': (0, 9), 'd2': (-INF, 10),
    'kappa_ev': (0.01, 1.5), 'Epk': (0, INF),
    'A3': (0, INF), 'gamma3': (0, 9), 'A4': (0, INF), 'gamma4': (0, 9),
    'kappa_slow': (1e-4, 0.5),
    'E_th': (20.0, 5000.0), 'n': (0.05, 8.0),
}

# Held parameters, each with the reason it is not fitted.
_HOLD_EPK = {'Epk': (0.004, 'evaporation peak fixed at 4 MeV, the peak visible '
                            'in the proton MC; left free it runs to hundreds of '
                            'MeV and turns evaporation into a hard halo')}

# Production values (fitting_params/params_*.csv, 2026-08-14) as start 0.
_P0_PROTON = dict(
    A1=0.0284561333317475, gamma1=0.743339305848289, Sigma=0.0217669010431835,
    n_ang=3.086825079703629, d1=-0.3178529260655601, a=0.5159585252568071,
    w_c=0.3492389574946068, A2=0.0015918571139998, gamma2=0.90444902689715,
    d2=-0.1905736186891524, kappa_ev=0.308278351652505, Epk=0.004,
    A3=6.3780931388557255e-06, gamma3=0.949366503652806, A4=426.9062465414815,
    gamma4=1.0884775789192016, kappa_slow=0.1850824019955174,
    E_th=102.17404838182227, n=2.1980529303268512)

_P0_CARBON = dict(
    A1=3.217774717627983, gamma1=0.9035849785188486, Sigma=0.0,
    theta_bar=16.307707367889716, d1=-0.050551895661357, a=0.952822555017134,
    w_c=0.1950262779495929, A2=0.0096827355877511, gamma2=1.2962162099676635,
    d2=-0.6424821685533537, kappa_ev=0.1805402856795102, Epk=0.004,
    A3=3.23128e-05, gamma3=1.7803, A4=2498.62904525102,
    gamma4=1.7855783065530184, kappa_slow=0.1264723332257396,
    E_th=194.88881445456553, n=0.963881495407092)


def _displaced(p0, lobe, lobe_val, extra):
    """Start 1: deliberately moved off start 0 in the nonlinear directions."""
    d = dict(p0)
    d.update({lobe: lobe_val, 'A1': p0['A1'] * 0.4, 'A2': p0['A2'] * 3.0,
              'kappa_ev': 0.30 if p0['kappa_ev'] < 0.25 else 0.15,
              'kappa_slow': p0['kappa_slow'] * 1.3,
              'gamma3': 1.5 * p0['gamma3'], 'gamma4': 0.8 * p0['gamma4']})
    d.update(extra)
    return d


SPECIES = {
    'proton': dict(
        bounds=dict(_COMMON_BOUNDS,
                    n_ang=(0.0, float(theta_to_n(THETA_GRID_C[0])))),
        hold=dict(_HOLD_EPK),
        starts=[_P0_PROTON,
                _displaced(_P0_PROTON, 'n_ang', 5.0,
                           dict(Sigma=0.06, E_th=180.0, n=1.2))],
    ),
    'carbon': dict(
        bounds=dict(_COMMON_BOUNDS,
                    theta_bar=(float(THETA_GRID_C[0]), float(THETA_GRID_C[-1]))),
        hold=dict(_HOLD_EPK),
        starts=[_P0_CARBON,
                _displaced(_P0_CARBON, 'theta_bar', 24.0,
                           dict(Sigma=0.02, E_th=350.0, n=1.0))],
    ),
}

# Stage 1: joint nonlinear least squares, scipy.optimize.curve_fit.
OPTIMIZER = dict(method='trf', ftol=1e-12, xtol=1e-12, gtol=1e-12,
                 max_nfev=20000)

# Stage 2: epithermal (A3, gamma3) against the 1 eV bin +- 10 bins (0.31-3.2 eV),
# where the epithermal regime dominates. The energy-fluence objective of stage 1
# barely constrains this pair (it carries ~0 % of E dPhi/dE), so it drifts to a
# rail there. A3 enters linearly and is solved in closed form for every gamma3
# on a geometric grid, which gives the global optimum of the band.
STAGE2 = dict(nb=10, thin_z=4, thin_rho=5,
              gamma3_grid=dict(lo=0.15, hi=20.0, num=90))

# Tolerance for flagging a parameter as sitting on a bound (relative).
RAIL_RTOL = 1e-6
