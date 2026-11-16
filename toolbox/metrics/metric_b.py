"""Unsigned counterpart of Eq. 9 ("metric B") and related map statistics.

For each primary energy, total fluence Phi and kerma K are integrated over
neutron energy (weights dE/E and dE*k/E on the cached E dPhi/dE) for MC and the
model, giving maps over (z, rho). From these maps:

  A     Eq. 9 as published: sum (AM - MC) V / sum MC V            (signed)
  B     sum |AM - MC| V / sum MC V                                (unsigned)
  A10   A restricted to bins with MC > 10 % of the map maximum    (R1-17)
  B10   B restricted the same way
  f10   share of the total volume that passes the 10 % cut
  dmax  max |AM - MC| / max MC over the map                       (T19)
  Ac, Bc, dmaxc  A, B, dmax restricted to the core: rho <= R (field
        radius, 1 cm) and z <= primary range (data/<species>_range.csv)
  fc    share of the total volume in the core
  mc    share of the MC integral (sum MC V) in the core

All in percent. Same kernel tables and parameters as bias_curves.py.

Bin conventions (npy_data/z.npy = 1, 2, ..., 45 cm; npy_data/rho.npy = 0.1, ...,
5.5 cm, the OUTER edges of the 1 mm annuli, so the volumes use edges 0, 0.1, ...):
  * radial: a bin is in the core when its outer edge rho <= R = 1 cm, i.e. the
    10 annuli 0-1 cm. R coincides with a bin edge, so no bin straddles it.
  * axial: a bin is in the core when its z label <= the primary range. The
    labels are the grid coordinates at which the model is evaluated and the MC
    is compared; if they are upper bin edges the bin containing the range is
    excluded, so the core is cut short by up to 1 cm. 'Bc_z1' is the same
    statistic with z <= range + 1 cm (that bin included) as a sensitivity check.
  * primary range: data/<species>_range.csv, the CSDA range R(E0) of Eq.
    ep-of-z (the integral of dE/S), which the notebooks call R_bragg.

Run as a script, it writes toolbox/metrics/results/metric_b_<species>.csv
(per energy) and metric_b_summary.csv (mean / median / max over the 50 energies).
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from toolbox.metrics.bias_curves import _load, _ROOT

STATS = ('A', 'B', 'A10', 'B10', 'f10', 'dmax',
         'Ac', 'Bc', 'fc', 'mc', 'dmaxc', 'Bc_z1')
FIELD_RADIUS = 1.0  # cm, primary field radius R


def _maps(job):
    species, i, Ep, nr, vals, entry = job

    import numpy as np
    from toolbox.production_range import build_xt_table
    from toolbox.metrics.kerma_coeffs import k_coeff_pGy_cm2_from_GeV
    from model import Model

    root = _ROOT
    z = np.load(f'{root}/npy_data/z.npy')
    rho = np.load(f'{root}/npy_data/rho.npy')[:nr]
    en_low = np.load(f'{root}/npy_data/en_low.npy')
    dE = np.load(f'{root}/npy_data/en_upp.npy') - en_low
    kc = k_coeff_pGy_cm2_from_GeV(en_low)
    nz, nE = len(z), len(en_low)

    edges = np.concatenate([[0.0], rho])
    V = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)

    mc = np.asarray(np.load(f'{root}/npy_data/{species}_mc.npy',
                            mmap_mode='r')[i][:, :nr, :])
    M = Model(species=species,
              XT=build_xt_table(species, np.array([Ep]), en_low),
              iEp=np.repeat(np.arange(1), nz * nr * nE),
              iEn=np.tile(np.arange(nE), nz * nr))
    EP, Z, R, En = [a.ravel() for a in
                    np.meshgrid([Ep], z, rho, en_low, indexing='ij')]
    pr = getattr(M, entry)((Z, R, En, EP), *vals).reshape(nz, nr, nE)

    import pandas as pd
    rng = pd.read_csv(f'{root}/data/{species}_range.csv')
    R_range = float(rng.loc[np.isclose(rng['Ep'], Ep), 'Range'].iloc[0])
    core = (z[:, None] <= R_range) & (rho[None, :] <= FIELD_RADIUS + 1e-9)
    core1 = (z[:, None] <= R_range + 1.0) & (rho[None, :] <= FIELD_RADIUS + 1e-9)

    out = []
    for w in (dE / en_low, dE * kc / en_low):
        m = np.sum(mc * w, axis=2)
        q = np.sum(pr * w, axis=2)
        out.append(_stats(m, q, np.broadcast_to(V, m.shape), core, core1))
    return out


def _stats(m, q, V, core, core1):
    d = q - m
    s = {'A': np.sum(d * V) / np.sum(m * V) * 100,
         'B': np.sum(np.abs(d) * V) / np.sum(m * V) * 100}
    cut = m > 0.1 * m.max()
    s['A10'] = np.sum(d[cut] * V[cut]) / np.sum(m[cut] * V[cut]) * 100
    s['B10'] = np.sum(np.abs(d[cut]) * V[cut]) / np.sum(m[cut] * V[cut]) * 100
    s['f10'] = np.sum(V[cut]) / np.sum(V) * 100
    s['dmax'] = np.abs(d).max() / m.max() * 100
    c = core
    s['Ac'] = np.sum(d[c] * V[c]) / np.sum(m[c] * V[c]) * 100
    s['Bc'] = np.sum(np.abs(d[c]) * V[c]) / np.sum(m[c] * V[c]) * 100
    s['fc'] = np.sum(V[c]) / np.sum(V) * 100
    s['mc'] = np.sum(m[c] * V[c]) / np.sum(m * V) * 100
    s['dmaxc'] = np.abs(d[c]).max() / m.max() * 100
    c1 = core1
    s['Bc_z1'] = np.sum(np.abs(d[c1]) * V[c1]) / np.sum(m[c1] * V[c1]) * 100
    return s


def metric_b(species, param_file=None, nr=55, workers=4):
    """Return Ep and two dicts (Phi, K) mapping each stat to an array over Ep."""
    vals, entry = _load(species, param_file)
    Ep_all = np.load(f'{_ROOT}/npy_data/{species}_energies.npy')
    jobs = [(species, i, float(E), nr, vals, entry)
            for i, E in enumerate(Ep_all)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_maps, jobs, chunksize=1))
    phi = {k: np.array([o[0][k] for o in out]) for k in STATS}
    kerma = {k: np.array([o[1][k] for o in out]) for k in STATS}
    return Ep_all, phi, kerma


def _summary(v, Ep):
    a = np.abs(v)
    return a.mean(), np.median(a), a.max(), Ep[a.argmax()]


if __name__ == '__main__':
    import pandas as pd
    from fitting.config import TRAIN_IDX
    from toolbox.metrics.bias_curves import bias_curves

    outdir = os.path.join(_ROOT, 'toolbox', 'metrics', 'results')
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for sp in ('proton', 'carbon'):
        Ep, phi, kerma = metric_b(sp, workers=2)
        # A is Eq. 9 as published; it must equal the scored bias_curves values
        _, dphi, dk = bias_curves(sp, nr=55, workers=2)
        print(f'{sp}: max |A - bias_curves|  Phi {np.abs(phi["A"] - dphi).max():.2e}'
              f'  K {np.abs(kerma["A"] - dk).max():.2e}  (percentage points)')
        fitted = np.zeros(len(Ep), int)
        fitted[TRAIN_IDX] = 1
        cols = {'Ep': Ep, 'fitted': fitted}
        for lab, d in (('Phi', phi), ('K', kerma)):
            for k in STATS:
                cols[f'{lab}_{k}'] = d[k]
            for k in STATS:
                if k in ('f10', 'fc', 'mc'):
                    continue
                mean, med, mx, at = _summary(d[k], Ep)
                rows.append(dict(species=sp, quantity=lab, stat=k, mean_abs=mean,
                                 median_abs=med, max_abs=mx, max_at_Ep=at))
        pd.DataFrame(cols).to_csv(os.path.join(outdir, f'metric_b_{sp}.csv'),
                                  index=False, float_format='%.6g')
        for lab, d in (('Phi', phi), ('K', kerma)):
            for k in ('fc', 'mc', 'f10'):
                v = d[k]
                rows.append(dict(species=sp, quantity=lab, stat=k, mean_abs=v.mean(),
                                 median_abs=np.median(v), max_abs=v.max(),
                                 max_at_Ep=Ep[v.argmax()]))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(outdir, 'metric_b_summary.csv'), index=False,
                float_format='%.6g')
    print(summ.to_string(index=False, float_format=lambda x: f'{x:8.2f}'))
