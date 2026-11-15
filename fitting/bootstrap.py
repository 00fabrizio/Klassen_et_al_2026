"""Parameter uncertainties of the fit: cluster bootstrap over primary energies,
and cheap cross-checks at the point estimate (T35, R2-M4).

Why an energy bootstrap. The fit is unweighted least squares on E dPhi/dE and the
residuals are dominated by model inadequacy, not by FLUKA counting noise (`checks`
reports chi2/dof with the corrected statistical errors); the bins of one energy
share the same primaries and are strongly correlated, and 11-23 % of the bins are
empty (no defined error). The error-weighted covariance therefore cannot set the
parameter uncertainty. What does vary between plausible training sets is the choice
of primary energies: each replicate draws 10 energies with replacement from the 10
training energies (multinomial counts, applied as --energy-weights) and refits with
the full routine (stage 1 + stage 2), warm-started from the point estimate, with
the bounds and holds of fitting/config.py.

Commands (from the repository root; one fit takes ~6-7 min and ~1.7 GB, so run at
most 3 workers at once):

    # replicates; one output file per replicate, resumable (finished ones are skipped)
    python -m fitting.bootstrap run --species carbon -B 200 --seed 35 \
        --p0 fitting_params/params_carbon.csv --worker 0 --nworkers 3
    python -m fitting.bootstrap run ... --worker 1 --nworkers 3      # and 2

    # per-parameter SE, intervals, % at bound, correlation matrix + heatmap
    python -m fitting.bootstrap summarize --species carbon \
        --p0 fitting_params/params_carbon.csv

    # naive curve_fit covariance, cluster-robust sandwich, chi2 with MC errors
    python -m fitting.bootstrap checks --species carbon \
        --p0 fitting_params/params_carbon.csv

Replicate r uses numpy SeedSequence([seed, r]) -> multinomial(10, uniform) counts
over the 10 training energies, so any replicate can be redone alone. A replicate
that raises is stored as rep_XXXX.fail.json (traceback); one that returns with a
least_squares status <= 0 (max_nfev reached) is kept but flagged non-converged in
the summary. Nothing is dropped silently: `summarize` prints and saves the counts.
"""
import argparse
import glob
import json
import os
import time
import traceback

import numpy as np
import pandas as pd

from fitting import config as C
from fitting import fit as F

ROOT = F.ROOT
RAW = f'{ROOT}/fitting/results'


def rep_dir(species, out_dir=None):
    return out_dir or f'{RAW}/bootstrap_{species}/reps'


def draw_counts(seed, r, n=None):
    """Multiplicities over the training energies for replicate r (sum = n)."""
    n = n or len(C.TRAIN_IDX)
    rng = np.random.default_rng(np.random.SeedSequence([seed, r]))
    return rng.multinomial(n, np.full(len(C.TRAIN_IDX), 1.0 / len(C.TRAIN_IDX)))


# ---------------------------------------------------------------- run

def run_one(species, r, seed, p0, d):
    base = f'{d}/rep_{r:04d}'
    if os.path.exists(base + '.csv') or os.path.exists(base + '.fail.json'):
        return 'skip'
    c = draw_counts(seed, r)
    counts = {int(i): int(k) for i, k in zip(C.TRAIN_IDX, c) if k > 0}
    t0 = time.time()
    try:
        res = F.run(species, counts=counts, p0=p0)
        res['bootstrap'] = dict(seed=seed, rep=r, counts=[int(k) for k in c],
                                train_idx=[int(i) for i in C.TRAIN_IDX],
                                p0_csv=str(p0))
        F.save(res, base + '.csv')
        st = res['stage1']
        return (f'rep {r}: counts {c.tolist()} status {st["status"]} nfev {st["nfev"]} '
                f'{res["wall_s"] / 60:.1f} min rails {res["rails"]}')
    except Exception as e:                                   # noqa: BLE001
        with open(base + '.fail.json', 'w') as fh:
            json.dump(dict(rep=r, seed=seed, counts=[int(k) for k in c],
                           error=repr(e), traceback=traceback.format_exc(),
                           wall_s=time.time() - t0), fh, indent=1)
        return f'rep {r}: FAILED {e!r}'


def cmd_run(a):
    d = rep_dir(a.species, a.out_dir)
    os.makedirs(d, exist_ok=True)
    if a.retry_failed:
        for f in glob.glob(f'{d}/rep_*.fail.json'):
            os.remove(f)
    for r in range(a.worker, a.B, a.nworkers):
        print(run_one(a.species, r, a.seed, a.p0, d), flush=True)


# ---------------------------------------------------------------- collect

def collect(species, d=None):
    """One row per replicate: counts, status, wall time, rails, 19 parameters."""
    d = rep_dir(species, d)
    names = C.PARAMS[species]
    rows = []
    for f in sorted(glob.glob(f'{d}/rep_*.json')):
        if f.endswith('.fail.json'):
            j = json.load(open(f))
            rows.append(dict(rep=j['rep'], ok=False, converged=False,
                             counts=j['counts'], error=j['error']))
            continue
        j = json.load(open(f))
        p = pd.read_csv(f[:-5] + '.csv', index_col=0).loc['opt params']
        st = j['stage1']
        row = dict(rep=j['bootstrap']['rep'], ok=True,
                   converged=st['status'] > 0, status=st['status'],
                   nfev=st['nfev'], cost=st['cost'], wall_min=j['wall_s'] / 60,
                   counts=j['bootstrap']['counts'], rails=';'.join(j['rails']))
        row.update({k: float(p[k]) for k in names})
        rows.append(row)
    return pd.DataFrame(rows).sort_values('rep').reset_index(drop=True)


def at_bound(species, k, v):
    lo, hi = C.SPECIES[species]['bounds'][k]
    out = np.zeros(len(v), bool)
    for b in (lo, hi):
        if np.isfinite(b):
            out |= np.abs(v - b) <= C.RAIL_RTOL * max(1.0, abs(b))
    return out


def cmd_summarize(a):
    species = a.species
    names = C.PARAMS[species]
    free = [k for k in names if k not in C.SPECIES[species]['hold']]
    d = rep_dir(species, a.out_dir)
    df = collect(species, d)
    pt = F.read_params(a.p0, species)
    n_all = len(df)
    good = df[df.ok & df.converged]
    nf, nn = int((~df.ok).sum()), int((df.ok & ~df.converged).sum())
    print(f'{species}: {n_all} replicates, {len(good)} converged, '
          f'{nn} not converged (status <= 0), {nf} failed')
    out_dir = a.summary_dir or f'{RAW}/bootstrap_{species}'
    os.makedirs(out_dir, exist_ok=True)
    cols = ['rep', 'ok', 'converged', 'status', 'nfev', 'cost', 'wall_min',
            'counts', 'rails'] + names
    df2 = df.copy()
    df2['counts'] = df2['counts'].apply(lambda c: ' '.join(str(x) for x in c))
    df2[[c for c in cols if c in df2]].to_csv(f'{out_dir}/replicates.csv', index=False)

    rows = []
    for k in free:
        v = good[k].values
        row = dict(parameter=k, point=pt[k], boot_mean=v.mean(), boot_median=np.median(v),
                   se=v.std(ddof=1),
                   p16=np.percentile(v, 16), p84=np.percentile(v, 84),
                   p2_5=np.percentile(v, 2.5), p97_5=np.percentile(v, 97.5),
                   pct_at_bound=100 * at_bound(species, k, v).mean(),
                   point_at_bound=bool(at_bound(species, k, np.array([pt[k]]))[0]))
        rows.append(row)
    s = pd.DataFrame(rows)
    s.to_csv(f'{out_dir}/summary.csv', index=False)

    # correlation matrix over the parameters that vary in the replicates
    var = [k for k in free if good[k].std() > 0]
    R = good[var].corr()
    R.to_csv(f'{out_dir}/correlation.csv')
    Rs = good[var].corr(method='spearman')
    Rs.to_csv(f'{out_dir}/correlation_spearman.csv')
    flagged = []
    for i, ki in enumerate(var):
        for kj in var[i + 1:]:
            rp, rs = R.loc[ki, kj], Rs.loc[ki, kj]
            if abs(rp) > 0.9 or abs(rs) > 0.9:
                flagged.append(dict(a=ki, b=kj, pearson=rp, spearman=rs))
    pd.DataFrame(flagged, columns=['a', 'b', 'pearson', 'spearman']).to_csv(
        f'{out_dir}/correlation_flagged.csv', index=False)
    heatmap(R, var, f'{out_dir}/correlation.png', f'{species}: bootstrap correlation '
            f'(B={len(good)})')
    meta = dict(species=species, B_requested=a.B, n_replicates=n_all,
                n_converged=len(good), n_nonconverged=nn, n_failed=nf,
                nonconverged_reps=df[df.ok & ~df.converged].rep.tolist(),
                failed_reps=df[~df.ok].rep.tolist(),
                wall_min_median=float(good.wall_min.median()) if len(good) else None,
                wall_min_mean=float(good.wall_min.mean()) if len(good) else None,
                n_flagged_pairs=len(flagged), point_estimate=a.p0)
    json.dump(meta, open(f'{out_dir}/summary.json', 'w'), indent=1)
    pd.set_option('display.width', 250, 'display.max_columns', 30)
    print(s.to_string(float_format=lambda x: f'{x:.5g}'))
    print(f'{len(flagged)} pairs with |r| > 0.9:')
    for f in flagged:
        print(f'  {f["a"]:10s} {f["b"]:10s} pearson {f["pearson"]:+.3f} '
              f'spearman {f["spearman"]:+.3f}')
    print(json.dumps(meta))


def heatmap(R, var, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(R.values, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(len(var)), var, rotation=90)
    ax.set_yticks(range(len(var)), var)
    for i in range(len(var)):
        for j in range(len(var)):
            v = R.values[i, j]
            if i != j and abs(v) > 0.5:
                ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=6,
                        color='white' if abs(v) > 0.8 else 'black')
    fig.colorbar(im, ax=ax, label='Pearson correlation')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- checks

def _stack(species, idx, thin_z, thin_rho, en_slice):
    """Model, X, MC, cluster id, grid indices for the given grid (no weights)."""
    z, rho, en = F._grid(thin_z, thin_rho, en_slice)
    M, X, MC, npt = F._problem(species, idx, z, rho, en,
                               (slice(None, None, thin_z), slice(None, None, thin_rho),
                                en_slice))
    return M, X, MC, npt, (z, rho, en)


def _jacobian(fun, x0, lo, hi):
    from scipy.optimize._numdiff import approx_derivative
    return approx_derivative(fun, x0, method='3-point', bounds=(lo, hi))


def _inv(JtJ):
    """Pseudo-inverse with the curve_fit threshold (SVD)."""
    U, s, Vt = np.linalg.svd(JtJ)
    thr = np.finfo(float).eps * max(JtJ.shape) * s[0]
    keep = s > thr
    inv = (Vt[keep].T / s[keep]) @ U[:, keep].T
    return inv, float(s[0] / s[keep][-1]), int((~keep).sum())


def covariances(species, params, free, kind):
    """Naive and cluster-sandwich covariances of `free` at `params`.

    kind 'stage1': the stage-1 training grid, `free` varied jointly.
    kind 'band'  : the stage-2 band (A3, gamma3 only), same clusters.
    """
    names = C.PARAMS[species]
    bounds = C.SPECIES[species]['bounds']
    idx = np.asarray(C.TRAIN_IDX)
    if kind == 'stage1':
        M, X, MC, npt, _ = _stack(species, idx, C.THIN['z'], C.THIN['rho'],
                                  slice(None, None, C.THIN['en']))
    else:
        s = C.STAGE2
        en_all = np.load(f'{ROOT}/npy_data/en_low.npy')
        k1 = int(np.argmin(np.abs(en_all - 1e-9)))
        M, X, MC, npt, _ = _stack(species, idx, s['thin_z'], s['thin_rho'],
                                  slice(k1 - s['nb'], k1 + s['nb'] + 1))
    entry = getattr(M, C.ENTRY[species])
    full = dict(params)
    fi = [names.index(k) for k in free]

    def model(v):
        q = [full[k] for k in names]
        for j, val in zip(fi, v):
            q[j] = val
        return entry(X, *q)

    lo = np.array([bounds[k][0] for k in free])
    hi = np.array([bounds[k][1] for k in free])
    x0 = np.array([full[k] for k in free])
    r = model(x0) - MC
    J = _jacobian(lambda v: model(v), x0, lo, hi)
    N, p = J.shape
    JtJ = J.T @ J
    A, cond, ndrop = _inv(JtJ)
    s2 = float(r @ r) / (N - p)
    naive = A * s2
    G = len(idx)
    S = np.stack([J[g * npt:(g + 1) * npt].T @ r[g * npt:(g + 1) * npt]
                  for g in range(G)])                       # G x p cluster scores
    meat = S.T @ S
    sand = (G / (G - 1)) * A @ meat @ A
    return dict(names=list(free), x0=x0, naive=naive, sandwich=sand, cond=cond,
                n_dropped_sv=ndrop, N=N, p=p, s2=s2, score_sum_rel=float(
                    np.linalg.norm(S.sum(0)) / max(np.linalg.norm(S, axis=1).max(), 1e-300)),
                cost=0.5 * float(r @ r), at_bound=[bool(at_bound(species, k, np.array([
                    full[k]]))[0]) for k in free])


def chi2_report(species, params):
    """chi2/dof with the FLUKA relative errors on the stage-1 grid, training energies."""
    from toolbox.metrics.mc_stats import BANDS
    names = C.PARAMS[species]
    idx = np.asarray(C.TRAIN_IDX)
    th = (C.THIN['z'], C.THIN['rho'], C.THIN['en'])
    M, X, MC, npt, (z, rho, en) = _stack(species, idx, th[0], th[1], slice(None, None, th[2]))
    entry = getattr(M, C.ENTRY[species])
    r = (entry(X, *[params[k] for k in names]) - MC).reshape(len(idx), len(z), len(rho), len(en))
    mc = MC.reshape(r.shape)
    rel = np.load(f'{ROOT}/npy_data/{species}_relerr.npy', mmap_mode='r')
    relg = np.stack([np.asarray(rel[i][::th[0], ::th[1], ::th[2]], float) for i in idx])
    sig = relg * mc
    ok = (mc > 0) & np.isfinite(sig) & (sig > 0)
    nfree = len([k for k in names if k not in C.SPECIES[species]['hold']])
    tot = r.size
    out = dict(n_points=int(tot), share_zero_bins=float(np.mean(mc <= 0)),
               share_no_sigma=float(np.mean(~ok)), n_used=int(ok.sum()))
    chi2 = float(np.sum((r[ok] / sig[ok]) ** 2))
    out.update(chi2=chi2, dof=int(ok.sum()) - nfree, chi2_dof=chi2 / (ok.sum() - nfree))
    z_all = np.abs(r) / np.where(ok, sig, np.nan)
    en_l = en[None, None, None, :] * np.ones_like(r)
    e_share = np.abs(r) ** 2
    bands = {}
    for name, lo, hi in BANDS:
        m = ok & (en_l >= lo) & (en_l < hi)
        v = z_all[m]
        if v.size == 0:
            continue
        bands[name] = dict(n=int(v.size), median=float(np.median(v)),
                           p84=float(np.percentile(v, 84)), p95=float(np.percentile(v, 95)),
                           frac_gt3=float(np.mean(v > 3)), frac_gt10=float(np.mean(v > 10)),
                           chi2_dof=float(np.mean(v ** 2)),
                           share_of_sse=float(np.sum(r[m] ** 2) / np.sum(r ** 2)),
                           share_zero=float(np.mean(mc[(en_l >= lo) & (en_l < hi)] <= 0)))
    out['bands'] = bands
    per_e = {}
    for g, i in enumerate(idx):
        m = ok[g]
        v = z_all[g][m]
        per_e[int(i)] = dict(E0=float(F.energies(species)[i]), n=int(m.sum()),
                             median=float(np.median(v)), p95=float(np.percentile(v, 95)),
                             chi2_dof=float(np.mean(v ** 2)))
    out['per_energy'] = per_e
    # bins within an energy are not independent: the lag-1 correlation of the
    # standardised residuals along neutron energy at fixed (z, rho), the same
    # statistic for the MC noise is unknown, so only report the residual one
    rr = np.where(ok, r / np.where(ok, sig, 1.0), np.nan)
    a_, b_ = rr[..., :-1].ravel(), rr[..., 1:].ravel()
    m2 = np.isfinite(a_) & np.isfinite(b_)
    out['lag1_corr_resid_along_en'] = float(np.corrcoef(a_[m2], b_[m2])[0, 1])
    return out


def cmd_checks(a):
    species = a.species
    params = F.read_params(a.p0, species)
    names = C.PARAMS[species]
    free = [k for k in names if k not in C.SPECIES[species]['hold']]
    main = [k for k in free if k not in ('A3', 'gamma3')]
    t0 = time.time()
    c1 = covariances(species, params, main, 'stage1')
    c2 = covariances(species, params, ['A3', 'gamma3'], 'band')
    c1f = covariances(species, params, free, 'stage1')
    rows = []
    for k in free:
        row = dict(parameter=k, point=params[k])
        if k in main:
            j = main.index(k)
            row['naive_se'] = float(np.sqrt(max(c1['naive'][j, j], 0)))
            row['sandwich_se'] = float(np.sqrt(max(c1['sandwich'][j, j], 0)))
        else:
            j = ['A3', 'gamma3'].index(k)
            row['naive_se'] = float(np.sqrt(max(c2['naive'][j, j], 0)))
            row['sandwich_se'] = float(np.sqrt(max(c2['sandwich'][j, j], 0)))
        jf = free.index(k)
        row['naive_se_full18'] = float(np.sqrt(max(c1f['naive'][jf, jf], 0)))
        row['sandwich_se_full18'] = float(np.sqrt(max(c1f['sandwich'][jf, jf], 0)))
        row['at_bound'] = at_bound(species, k, np.array([params[k]]))[0].item()
        rows.append(row)
    s = pd.DataFrame(rows)
    out_dir = a.summary_dir or f'{RAW}/bootstrap_{species}'
    os.makedirs(out_dir, exist_ok=True)
    tag = a.tag or species
    s.to_csv(f'{out_dir}/checks_se_{tag}.csv', index=False)
    Rn = c1['naive'] / np.sqrt(np.outer(np.diag(c1['naive']), np.diag(c1['naive'])))
    pd.DataFrame(Rn, index=main, columns=main).to_csv(f'{out_dir}/checks_naive_corr_{tag}.csv')
    ch = chi2_report(species, params)
    meta = dict(point_estimate=a.p0, tag=tag, wall_s=time.time() - t0,
                stage1=dict(N=c1['N'], p=c1['p'], cond_JtJ=c1['cond'],
                            dropped_sv=c1['n_dropped_sv'], s2=c1['s2'], cost=c1['cost'],
                            score_sum_rel=c1['score_sum_rel']),
                stage1_full18=dict(cond_JtJ=c1f['cond'], dropped_sv=c1f['n_dropped_sv']),
                band=dict(N=c2['N'], p=c2['p'], cond_JtJ=c2['cond'], s2=c2['s2'],
                          cost=c2['cost']),
                chi2=ch,
                notes='SE blocks: 16 stage-1 parameters (A3, gamma3 fixed at their '
                      'final values) and (A3, gamma3) on the stage-2 band; *_full18 '
                      'varies all 18 free parameters jointly on the stage-1 grid. '
                      'Sandwich: 10 clusters (primary energies), G/(G-1) factor; its '
                      'rank is at most G-1 = 9.')
    json.dump(meta, open(f'{out_dir}/checks_{tag}.json', 'w'), indent=1)
    pd.set_option('display.width', 250, 'display.max_columns', 30)
    print(s.to_string(float_format=lambda x: f'{x:.5g}'))
    print(json.dumps({k: v for k, v in meta.items() if k != 'chi2'}, indent=1))
    print(json.dumps(ch, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('run', 'summarize', 'checks'):
        p = sub.add_parser(name)
        p.add_argument('--species', required=True, choices=list(C.PARAMS))
        p.add_argument('--p0', required=True, help='point estimate CSV')
        p.add_argument('--out-dir', help='replicate directory (default '
                       'fitting/results/bootstrap_<species>/reps)')
        p.add_argument('--summary-dir', help='default fitting/results/bootstrap_<species>')
        p.add_argument('-B', type=int, default=200)
        p.add_argument('--seed', type=int, default=35)
        p.add_argument('--worker', type=int, default=0)
        p.add_argument('--nworkers', type=int, default=1)
        p.add_argument('--retry-failed', action='store_true')
        p.add_argument('--tag', help='label for checks output files')
    a = ap.parse_args(argv)
    {'run': cmd_run, 'summarize': cmd_summarize, 'checks': cmd_checks}[a.cmd](a)


if __name__ == '__main__':
    main()
