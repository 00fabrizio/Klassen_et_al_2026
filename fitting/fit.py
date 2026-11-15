"""Fit the 19 model parameters of one species to the FLUKA MC data.

Pipeline (both stages use the training energies only):

  stage 1  joint nonlinear least squares of every parameter that is not held
           (config.SPECIES[...]['hold']), scipy.optimize.curve_fit, method 'trf',
           unweighted residuals on E dPhi/dE (npy_data/{species}_mc.npy), grid
           thinned as in config.THIN.
  stage 2  (A3, gamma3) refitted against the 1 eV band, closed-form A3 on a
           gamma3 grid (config.STAGE2). Everything else stays at stage 1.

Command line (from the repository root):

    python -m fitting.fit --species proton --out fitting/results/proton_refit.csv
    python -m fitting.fit --species carbon --start all --out ...
    python -m fitting.fit --species proton --energy-weights 2,0,1,1,1,0,3,1,1,0 \
        --p0 fitting/results/proton_refit.csv --out ...      # bootstrap replicate

From Python:

    from fitting.fit import run
    res = run('proton', counts={0: 2, 10: 1, ...}, p0='fitting/results/x.csv')
    res['params']          # dict, the 19 values in CSV column order

Writes <out>.csv in the layout of fitting_params/params_{species}.csv and
<out>.json with the metadata (costs, nfev, status, rails, wall time, versions,
git hash, energies and their weights, held parameters, starting values).
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import least_squares

from fitting import config as C
from model import Model
from toolbox.production_range import build_xt_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- data

def energies(species):
    return np.load(f'{ROOT}/npy_data/{species}_energies.npy')


def normalise_counts(species, counts=None):
    """{energy index: multiplicity}. None means the 10 training energies once.

    A multiplicity is how often an energy was drawn in a bootstrap resample; it
    is applied as a weight sqrt(count) on that energy's residuals, which is the
    same objective as stacking the energy count times. Energies with count 0
    are left out.
    """
    if counts is None:
        counts = {int(i): 1 for i in C.TRAIN_IDX}
    counts = {int(i): int(c) for i, c in dict(counts).items() if int(c) > 0}
    n = len(energies(species))
    bad = [i for i in counts if not 0 <= i < n]
    if bad or not counts:
        raise ValueError(f'energy indices out of range or empty: {bad}')
    return dict(sorted(counts.items()))


def _grid(thin_z, thin_rho, en_slice):
    z = np.load(f'{ROOT}/npy_data/z.npy')[::thin_z]
    rho = np.load(f'{ROOT}/npy_data/rho.npy')[::thin_rho]
    en = np.load(f'{ROOT}/npy_data/en_low.npy')[en_slice]
    return z, rho, en


def _problem(species, idx, z, rho, en, mc_slices):
    """Model, X tuple and MC vector for energies idx on the given grid."""
    Ep = energies(species)[idx]
    nEp, nz, nr, nE = len(idx), len(z), len(rho), len(en)
    M = Model(species=species, XT=build_xt_table(species, Ep, en),
              iEp=np.repeat(np.arange(nEp), nz * nr * nE),
              iEn=np.tile(np.arange(nE), nEp * nz * nr))
    EP, Z, R, En = [a.ravel() for a in np.meshgrid(Ep, z, rho, en, indexing='ij')]
    mc = np.load(f'{ROOT}/npy_data/{species}_mc.npy', mmap_mode='r')
    MC = np.concatenate([np.asarray(mc[i][mc_slices], dtype=float).ravel()
                         for i in idx])
    return M, (Z, R, En, EP), MC, nz * nr * nE


# ---------------------------------------------------------------- params

def read_params(path, species):
    p = pd.read_csv(path, index_col=0).loc['opt params'].astype(float)
    names = C.PARAMS[species]
    if list(p.index) != names:
        raise ValueError(f'{path}: columns {list(p.index)} != {names}')
    return {k: float(v) for k, v in p.items()}


def write_params(path, species, params):
    names = C.PARAMS[species]
    df = pd.DataFrame([[params[k] for k in names]], columns=names,
                      index=['opt params'])
    df.to_csv(path)


def _rails(names, vals, bounds):
    out = []
    for k, v in zip(names, vals):
        lo, hi = bounds[k]
        for side, b in (('lower', lo), ('upper', hi)):
            if np.isfinite(b) and abs(v - b) <= C.RAIL_RTOL * max(1.0, abs(b)):
                out.append(f'{k}={v:.6g} at {side} bound {b:g}')
    return out


# ---------------------------------------------------------------- stages

def stage1(species, p0, counts, free, max_nfev=None, verbose=0):
    """Joint curve_fit of the parameters in `free`; the rest held at p0."""
    names = C.PARAMS[species]
    bounds = C.SPECIES[species]['bounds']
    idx = np.array(list(counts))
    z, rho, en = _grid(C.THIN['z'], C.THIN['rho'], slice(None, None, C.THIN['en']))
    M, X, MC, npt = _problem(species, idx, z, rho, en,
                             (slice(None, None, C.THIN['z']),
                              slice(None, None, C.THIN['rho']),
                              slice(None, None, C.THIN['en'])))
    entry = getattr(M, C.ENTRY[species])
    w = np.repeat([counts[i] for i in idx], npt).astype(float)
    # Bootstrap multiplicities only: sigma = 1/sqrt(count) per point. This is NOT
    # an MC-uncertainty weighting (which is not used, see CLAUDE.md); with every
    # count 1 the fit is the plain unweighted least squares.
    sigma = None if np.all(w == 1) else 1.0 / np.sqrt(w)

    full = dict(p0)
    fi = [names.index(k) for k in free]

    def f(X, *v):
        q = [full[k] for k in names]
        for j, val in zip(fi, v):
            q[j] = val
        return entry(X, *q)

    lo = [bounds[k][0] for k in free]
    hi = [bounds[k][1] for k in free]
    x0 = np.clip([p0[k] for k in free], lo, hi)
    opt = {k: v for k, v in C.OPTIMIZER.items() if k != "method"}
    opt["method"] = C.OPTIMIZER["method"]
    if max_nfev:
        opt['max_nfev'] = int(max_nfev)
    sw = 1.0 if sigma is None else 1.0 / sigma

    def resid(v):
        return sw * (f(X, *v) - MC)

    cost0 = 0.5 * float(np.sum(resid(x0) ** 2))
    t0 = time.time()
    # Exactly what curve_fit(f, X, MC, p0, sigma, bounds, method='trf') runs
    # (scipy/optimize/_minpack_py.py): least_squares on (f - y)/sigma with a
    # 2-point finite-difference Jacobian. Called directly so that a run that
    # stops at max_nfev still returns its parameters and status instead of
    # raising, which a bootstrap loop needs.
    r = least_squares(resid, x0, jac='2-point', bounds=(lo, hi),
                      verbose=verbose, **opt)
    wall = time.time() - t0
    out = dict(full)
    out.update({k: float(v) for k, v in zip(free, r.x)})
    return out, dict(cost_start=cost0, cost=float(r.cost), nfev=int(r.nfev),
                     njev=int(r.njev), status=int(r.status),
                     success=bool(r.success), message=str(r.message),
                     wall_s=wall, n_points=int(MC.size),
                     rails=_rails(free, r.x, bounds)), (M, X, MC, w, entry)


def stage2(species, params, counts):
    """(A3, gamma3) against the 1 eV band, closed-form A3 on a gamma3 grid."""
    s = C.STAGE2
    en_all = np.load(f'{ROOT}/npy_data/en_low.npy')
    k1 = int(np.argmin(np.abs(en_all - 1e-9)))
    sl = slice(k1 - s['nb'], k1 + s['nb'] + 1)
    idx = np.array(list(counts))
    z, rho, en = _grid(s['thin_z'], s['thin_rho'], sl)
    M, X, MC, npt = _problem(species, idx, z, rho, en,
                             (slice(None, None, s['thin_z']),
                              slice(None, None, s['thin_rho']), sl))
    w = np.repeat([counts[i] for i in idx], npt).astype(float)
    names = C.PARAMS[species]
    entry = getattr(M, C.ENTRY[species])
    q = dict(params, A3=0.0)
    rest = entry(X, *[q[k] for k in names])
    _, _, _, EP = X
    pf = M.prefac2(EP, params['E_th'], params['n'])

    G = np.geomspace(s['gamma3_grid']['lo'], s['gamma3_grid']['hi'],
                     s['gamma3_grid']['num'])
    cost, a3 = np.empty(G.size), np.empty(G.size)
    for j, g3 in enumerate(G):
        K = pf * M.ep(X, 1.0, g3, params['kappa_slow'])
        a = max(float(np.sum(w * K * (MC - rest)) / np.sum(w * K * K)), 0.0)
        a3[j] = a
        cost[j] = 0.5 * float(np.sum(w * (rest + a * K - MC) ** 2))
    b = int(np.argmin(cost))
    out = dict(params, A3=float(a3[b]), gamma3=float(G[b]))
    q = dict(params)
    cost_in = 0.5 * float(np.sum(w * (entry(X, *[q[k] for k in names]) - MC) ** 2))
    rails = []
    if b in (0, G.size - 1):
        rails.append(f'gamma3={G[b]:.6g} at edge of the stage-2 grid')
    if a3[b] == 0.0:
        rails.append('A3=0 at lower bound')
    return out, dict(cost_band_in=cost_in, cost_band=float(cost[b]),
                     band_eV=[float(en[0] * 1e9), float(en[-1] * 1e9)],
                     n_points=int(MC.size), grid=dict(s['gamma3_grid']),
                     rails=rails)


# ---------------------------------------------------------------- driver

def environment():
    try:
        h = subprocess.run(['git', '-C', ROOT, 'rev-parse', 'HEAD'],
                           capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', ROOT, 'status', '--porcelain',
                                '--untracked-files=no'],
                               capture_output=True, text=True).stdout.strip()
    except OSError:
        h, dirty = '', ''
    return dict(python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, pandas=pd.__version__,
                platform=platform.platform(), git=h, git_dirty=bool(dirty))


def resolve_p0(species, p0=None, start=0):
    if p0 is None:
        return dict(C.SPECIES[species]['starts'][start]), f'config start {start}'
    if isinstance(p0, str):
        return read_params(p0, species), p0
    return dict(p0), 'dict'


def run(species, counts=None, p0=None, start=0, hold=None, free=None,
        do_stage2=True, max_nfev=None, verbose=0):
    """Run the full pipeline once. Returns a dict (see module docstring).

    counts   {energy index: multiplicity}; default the 10 training energies.
    p0       None (config start `start`), a CSV path, or a dict of 19 values.
    hold     {name: (value, reason)} replacing the config holds; held values
             override p0.
    free     explicit list of free parameters (all others held at p0); for
             reproduction runs such as freeing only A3, A4.
    """
    names = C.PARAMS[species]
    counts = normalise_counts(species, counts)
    p0d, p0_src = resolve_p0(species, p0, start)
    hold = C.SPECIES[species]['hold'] if hold is None else hold
    for k, (v, _) in hold.items():
        p0d[k] = v
    if free is None:
        free = [k for k in names if k not in hold]
    t0 = time.time()
    p1, info1, _ = stage1(species, p0d, counts, free, max_nfev, verbose)
    res = dict(species=species, params=p1, params_stage1=dict(p1),
               stage1=info1, stage2=None)
    if do_stage2 and 'A3' in free and 'gamma3' in free:
        p2, info2 = stage2(species, p1, counts)
        res['params'] = p2
        res['stage2'] = info2
    Ep = energies(species)
    res.update(
        rails=info1['rails'] if res['stage2'] is None else
        [r for r in info1['rails'] if not r.startswith(('A3=', 'gamma3='))]
        + res['stage2']['rails'],
        wall_s=time.time() - t0,
        energies={int(i): dict(E0=float(Ep[i]), count=c) for i, c in counts.items()},
        free=list(free),
        held={k: dict(value=p0d[k], reason=r) for k, (v, r) in hold.items()},
        held_other=[k for k in names if k not in free and k not in hold],
        p0=p0d, p0_source=p0_src,
        optimizer=dict(C.OPTIMIZER, **({'max_nfev': int(max_nfev)} if max_nfev else {})),
        bounds={k: list(C.SPECIES[species]['bounds'][k]) for k in free},
        thinning=dict(C.THIN), environment=environment(),
    )
    return res


def save(res, out_csv):
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    write_params(out_csv, res['species'], res['params'])
    meta = json.loads(json.dumps(res, default=lambda o: None if o is None else
                                 (o.item() if hasattr(o, 'item') else str(o))))
    meta['bounds'] = {k: [None if not np.isfinite(b) else b for b in v]
                      for k, v in res['bounds'].items()}
    with open(os.path.splitext(out_csv)[0] + '.json', 'w') as fh:
        json.dump(meta, fh, indent=1, allow_nan=False)


def _parse_counts(species, s_w, s_idx):
    idx = (np.array([int(x) for x in s_idx.split(',')]) if s_idx
           else C.TRAIN_IDX)
    if not s_w:
        return {int(i): 1 for i in idx}
    w = [int(x) for x in s_w.split(',')]
    if len(w) != len(idx):
        raise SystemExit(f'--energy-weights has {len(w)} entries, '
                         f'energy set has {len(idx)}')
    out = {}
    for i, c in zip(idx, w):
        out[int(i)] = out.get(int(i), 0) + c
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--species', required=True, choices=list(C.PARAMS))
    ap.add_argument('--out', required=True, help='output CSV (JSON alongside)')
    ap.add_argument('--start', default='0',
                    help="config start: 0, 1 or 'all' (lowest stage-1 cost wins)")
    ap.add_argument('--p0', help='CSV with starting values (warm start); '
                                 'overrides --start')
    ap.add_argument('--energy-idx', help='comma-separated energy indices '
                                         '(default: the 10 training energies)')
    ap.add_argument('--energy-weights',
                    help='comma-separated integer multiplicities, one per '
                         'energy of the set (bootstrap); 0 drops an energy')
    ap.add_argument('--free', help='comma-separated free parameters; all '
                                   'others held at p0 (reproduction runs)')
    ap.add_argument('--no-stage2', action='store_true')
    ap.add_argument('--max-nfev', type=int)
    ap.add_argument('--verbose', type=int, default=0)
    a = ap.parse_args(argv)

    counts = _parse_counts(a.species, a.energy_weights, a.energy_idx)
    free = a.free.split(',') if a.free else None
    starts = ([None] if a.p0 else
              range(len(C.SPECIES[a.species]['starts'])) if a.start == 'all'
              else [int(a.start)])
    best = None
    for s in starts:
        res = run(a.species, counts=counts, p0=a.p0, start=s or 0, free=free,
                  do_stage2=not a.no_stage2, max_nfev=a.max_nfev,
                  verbose=a.verbose)
        st = res['stage1']
        print(f'{a.species} start {res["p0_source"]}: stage-1 cost '
              f'{st["cost_start"]:.6g} -> {st["cost"]:.6g}, nfev {st["nfev"]}, '
              f'status {st["status"]}, {res["wall_s"] / 60:.1f} min, '
              f'rails {res["rails"]}', flush=True)
        if len(starts) > 1:
            save(res, os.path.splitext(a.out)[0] + f'_start{s}.csv')
        if best is None or st['cost'] < best['stage1']['cost']:
            best = res
    save(best, a.out)
    for k in C.PARAMS[a.species]:
        print(f'  {k:10s} {best["p0"][k]:>14.6g} -> {best["params"][k]:>14.6g}')
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
