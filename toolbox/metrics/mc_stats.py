"""Statistical uncertainty of the FLUKA reference data (T45, R2-m5).

Reads the relative statistical errors of the USRTRACK scoring from the raw
`data/<species>/<E0>/*_tab.lis` files. FLUKA writes that column in PERCENT (the
relative error of the mean over the independent runs merged by usxsuw, 0 for an
empty bin); here it is converted to a fraction.

Run from the repository root:

    .venv/bin/python -m toolbox.metrics.mc_stats

Output
  npy_data/{species}_relerr.npy   relative error (fraction), same shape as
                                  {species}_mc.npy = (E0, z, rho, En); NaN where
                                  the bin has no content
  npy_data/{species}_err.npy      absolute error of E dPhi/dE (relerr * mc), same
                                  shape; the old files were 100x too large
  npy_data/mc_stats_{species}.csv per-E0 uncertainty of the integrated quantities
  stdout                          the summaries quoted in the revision

Summaries
  (a) per-bin relative error over nonzero bins, per energy band, and the share of
      empty bins;
  (b) the same restricted to bright bins (mc above 1 % of the largest mc of that
      E0 and band), because the large errors sit in the faint tail;
  (c) propagated uncertainty of the integrated fluence Phi and kerma K per
      (z, rho) cell and of the volume-integrated totals per E0, with the weights
      of bias_curves.py (dE/En for Phi, dE*k/En for K on E dPhi/dE).

Assumption: the 250 energy bins and the (z, rho) cells are treated as
INDEPENDENT (errors added in quadrature). They are scored from the same
primaries, so neighbouring bins are positively correlated and the propagated
uncertainty of an integral is therefore a lower bound; the fully correlated
case (errors added linearly) is reported as the upper bound. The error of one bin is the FLUKA error of the mean
over the merged runs and is itself noisy where few runs contributed.

Primary histories and run counts are not in the .lis files; they are read from
the FLUKA binary headers and stored in data/mc_histories.csv.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from toolbox.dataloader import load_relerr  # noqa: E402
from toolbox.metrics.kerma_coeffs import k_coeff_pGy_cm2_from_GeV  # noqa: E402

NPY = os.path.join(ROOT, 'npy_data')
SPECIES = ('proton', 'carbon')

# Band edges in GeV. Epithermal edges are those of the model (derivation.tex:
# E_low = 1e-10 GeV = 0.1 eV, E_upp = 1e-4 GeV = 0.1 MeV); evaporation up to
# 20 MeV, cascade above. A bin belongs to the band of its lower edge.
BANDS = (('thermal', 0.0, 1e-10),
         ('epithermal', 1e-10, 1e-4),
         ('evaporation', 1e-4, 2e-2),
         ('cascade', 2e-2, np.inf))
BRIGHT = 1e-2  # fraction of the E0/band maximum of mc


def _folder(E):
    return f'{E:g}'


def build_relerr(species):
    """Write {species}_relerr.npy and {species}_err.npy.

    Returns the relerr memmap and the E0 indices whose tab.lis carries no
    statistical information (error column 0 in every bin with content: 2 proton
    and 5 carbon energies, see data/mc_histories.csv). Those E0 are set to NaN
    throughout, in relerr and err, instead of reporting a spurious 0 %.
    """
    from numpy.lib.format import open_memmap
    Ep = np.load(os.path.join(NPY, f'{species}_energies.npy'))
    mc = np.load(os.path.join(NPY, f'{species}_mc.npy'), mmap_mode='r')
    rel_out = open_memmap(os.path.join(NPY, f'{species}_relerr.npy'), mode='w+',
                          dtype=np.float64, shape=mc.shape)
    err_out = open_memmap(os.path.join(NPY, f'{species}_err.npy'), mode='w+',
                          dtype=np.float64, shape=mc.shape)
    no_err = []
    os.chdir(ROOT)  # load_relerr reads data/ relative to the working directory
    for i, E in enumerate(Ep):
        rel = load_relerr(species, _folder(E))
        if rel.shape != mc.shape[1:]:
            raise ValueError(f'{species} {E}: shape {rel.shape}')
        m = np.asarray(mc[i])
        if np.any((m == 0) != np.isnan(rel)):
            raise ValueError(f'{species} {E}: empty bins and NaN differ')
        has = ~np.isnan(rel)
        if has.any() and not np.any(rel[has] > 0):
            no_err.append(i)
            rel[:] = np.nan
        rel_out[i] = rel
        err_out[i] = np.where(m == 0, 0.0, rel * m)
    rel_out.flush()
    err_out.flush()
    return np.load(os.path.join(NPY, f'{species}_relerr.npy'), mmap_mode='r'), no_err


def band_masks(en_low):
    return {name: (en_low >= lo) & (en_low < hi) for name, lo, hi in BANDS}


def band_summary(rel, mc, en_low, valid):
    """Per-band distribution of the relative error over nonzero bins.

    `valid`: indices of the E0 with error information. Empty bins are those with
    mc == 0, counted over the same E0.
    """
    rel, mc = rel[valid], mc[valid]
    rows = {}
    masks = band_masks(en_low)
    for name, _, _ in BANDS:
        sel = masks[name]
        r = np.asarray(rel[..., sel])
        m = np.asarray(mc[..., sel])
        n_all = r.size
        nz = m > 0
        v = r[nz]
        # bright bins: mc above 1 % of the maximum of that E0 within the band
        peak = np.max(m, axis=(1, 2, 3), keepdims=True)
        bright = nz & (m > BRIGHT * peak)
        vb = r[bright]
        rows[name] = dict(
            n_bins=n_all, zero_frac=1 - v.size / n_all,
            med=np.median(v), p68=np.percentile(v, 68), p95=np.percentile(v, 95),
            med_b=np.median(vb), p68_b=np.percentile(vb, 68),
            p95_b=np.percentile(vb, 95), bright_frac=vb.size / v.size,
            # error 0 on a bin with content (single contributing batch)
            n_err0=int(np.sum(nz & (r == 0))))
    r = np.asarray(rel)
    nz = np.asarray(mc) > 0
    rows['all'] = dict(n_bins=r.size, zero_frac=1 - nz.sum() / r.size,
                       med=np.median(r[nz]), p68=np.percentile(r[nz], 68),
                       p95=np.percentile(r[nz], 95))
    return rows


def integrated(rel, mc, en_low, en_upp, rho, Ep, valid):
    """Propagated relative uncertainty of Phi and K, per E0."""
    dE = en_upp - en_low
    kc = k_coeff_pGy_cm2_from_GeV(en_low)
    edges = np.concatenate([[0.0], rho])
    V = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    out = []
    for i, E in enumerate(Ep):
        if i not in valid:
            continue
        m = np.asarray(mc[i])
        s = np.nan_to_num(np.asarray(rel[i]))
        rec = dict(E0=E)
        for lab, w in (('phi', dE / en_low), ('k', dE * kc / en_low)):
            Q = np.sum(m * w, axis=2)                       # (z, rho)
            sig = np.sqrt(np.sum((m * w * s) ** 2, axis=2))  # bins independent
            sig_ub = np.sum(m * w * s, axis=2)               # fully correlated
            ok = Q > 0
            c = sig[ok] / Q[ok]
            tot = np.sum(Q * V)
            sig_tot = np.sqrt(np.sum((sig * V) ** 2))
            rec[f'{lab}_cell_med'] = np.median(c)
            rec[f'{lab}_cell_p95'] = np.percentile(c, 95)
            rec[f'{lab}_cell_zero'] = 1 - ok.mean()
            rec[f'{lab}_tot'] = sig_tot / tot
            # upper bound: all bins and cells fully positively correlated
            rec[f'{lab}_cell_med_ub'] = np.median(sig_ub[ok] / Q[ok])
            rec[f'{lab}_tot_ub'] = np.sum(sig_ub * V) / tot
        out.append(rec)
    return out


def histories(species):
    """Primaries and runs per E0 from data/mc_histories.csv (or empty dict)."""
    f = os.path.join(ROOT, 'data', 'mc_histories.csv')
    if not os.path.exists(f):
        return {}
    d = {}
    for line in open(f).read().splitlines()[1:]:
        sp, E, prim, runs, _ = line.split(',')[:5]
        if sp == species:
            d[float(E)] = (int(prim) if prim else None, int(runs) if runs else None)
    return d


def main():
    en_low = np.load(os.path.join(NPY, 'en_low.npy'))
    en_upp = np.load(os.path.join(NPY, 'en_upp.npy'))
    rho = np.load(os.path.join(NPY, 'rho.npy'))
    for species in SPECIES:
        print(f'\n===== {species} =====')
        Ep = np.load(os.path.join(NPY, f'{species}_energies.npy'))
        rel, no_err = build_relerr(species)
        valid = np.array([i for i in range(len(Ep)) if i not in no_err])
        mc = np.load(os.path.join(NPY, f'{species}_mc.npy'), mmap_mode='r')
        empty = np.asarray(mc) == 0
        print(f'shape {rel.shape}; empty bins {empty.sum()} of {empty.size} '
              f'({100 * empty.mean():.1f} %, all E0)')
        print(f'E0 without error information (error column 0 everywhere): '
              f'{[float(Ep[i]) for i in no_err]}  -> excluded below, '
              f'{len(valid)} of {len(Ep)} E0 used')

        h = histories(species)
        if h:
            prims = sorted(set(v[0] for v in h.values() if v[0]))
            print('histories per E0 (value: #energies):',
                  {p: sum(1 for v in h.values() if v[0] == p) for p in prims},
                  ' unknown:', sum(1 for v in h.values() if not v[0]))
            print('runs per E0 (value: #energies):',
                  {r: sum(1 for v in h.values() if v[1] == r)
                   for r in sorted(set(v[1] for v in h.values() if v[1]))})

        print('\nper-bin relative error over nonzero bins [%], by band')
        print(f'{"band":12s} {"empty":>6s} {"median":>7s} {"p68":>7s} {"p95":>7s}'
              f' | bright (> 1 % of E0/band max): {"share":>6s} {"median":>7s}'
              f' {"p68":>7s} {"p95":>7s}')
        for name, r in band_summary(rel, mc, en_low, valid).items():
            line = (f'{name:12s} {100 * r["zero_frac"]:6.1f} {100 * r["med"]:7.2f} '
                    f'{100 * r["p68"]:7.2f} {100 * r["p95"]:7.2f}')
            if 'med_b' in r:
                line += (f' |                                  {100 * r["bright_frac"]:6.1f} '
                         f'{100 * r["med_b"]:7.2f} {100 * r["p68_b"]:7.2f} '
                         f'{100 * r["p95_b"]:7.2f}   (error 0 on content: {r["n_err0"]})')
            print(line)

        rec = integrated(rel, mc, en_low, en_upp, rho, Ep, set(valid.tolist()))
        keys = list(rec[0])
        with open(os.path.join(NPY, f'mc_stats_{species}.csv'), 'w') as fh:
            fh.write(','.join(keys) + '\n')
            for r in rec:
                fh.write(','.join(f'{r[k]:.6g}' for k in keys) + '\n')
        print('\nintegrated quantities, relative uncertainty [%] (bins independent)')
        arr = {k: np.array([r[k] for r in rec]) for k in keys}
        for lab in ('phi', 'k'):
            print(f'  {lab:3s} per cell (cells with Phi > 0): median over cells, '
                  f'range over E0 {100 * arr[lab + "_cell_med"].min():.3f} .. '
                  f'{100 * arr[lab + "_cell_med"].max():.3f}; '
                  f'95th pct max {100 * arr[lab + "_cell_p95"].max():.2f}; '
                  f'empty cells max {100 * arr[lab + "_cell_zero"].max():.2f}')
            print(f'      fully correlated upper bound: cell median '
                  f'{100 * arr[lab + "_cell_med_ub"].min():.2f} .. '
                  f'{100 * arr[lab + "_cell_med_ub"].max():.2f}, total '
                  f'{100 * arr[lab + "_tot_ub"].min():.3f} .. '
                  f'{100 * arr[lab + "_tot_ub"].max():.3f} %')
            print(f'      volume-integrated total per E0: '
                  f'{100 * arr[lab + "_tot"].min():.4f} .. '
                  f'{100 * arr[lab + "_tot"].max():.4f} '
                  f'(median {100 * np.median(arr[lab + "_tot"]):.4f})')
        E = arr['E0']
        for lo, hi in ((E.min(), np.median(E)), (np.median(E), E.max() + 1)):
            sel = (E >= lo) & (E < hi)
            print(f'  E0 {lo:6.1f} .. {hi:6.1f} (n={sel.sum()}): '
                  f'total Phi {100 * arr["phi_tot"][sel].min():.4f} .. '
                  f'{100 * arr["phi_tot"][sel].max():.4f} %, '
                  f'total K {100 * arr["k_tot"][sel].min():.4f} .. '
                  f'{100 * arr["k_tot"][sel].max():.4f} %; '
                  f'cell median Phi {100 * np.median(arr["phi_cell_med"][sel]):.2f} %, '
                  f'K {100 * np.median(arr["k_cell_med"][sel]):.2f} %')


if __name__ == '__main__':
    main()
