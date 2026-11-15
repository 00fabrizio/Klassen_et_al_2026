"""Score parameter sets with the Eq. 10/11 metrics and draw AM/MC maps.

    python -m fitting.evaluate --species proton \
        --set production=fitting_params/params_proton.csv \
        --set refit=fitting/results/proton_refit.csv [--maps]

Metrics are toolbox/metrics/bias_curves.py (volume-weighted bias of total
fluence, Delta_Phi, and kerma, Delta_K, full 5.5 cm radius), summarised over all
50 primary energies, the 10 training energies and the 40 holdout energies.
Maps are E dPhi/dE in (z, rho) at four neutron energies for three primary
energies, linear colour scale, MC | AM | |AM - MC| per set, sets side by side.
"""
import argparse
import json
import os

import numpy as np

from fitting import config as C
from fitting.fit import ROOT, energies, read_params
from toolbox.metrics.bias_curves import bias_curves

MAP_EP_IDX = (0, 24, 49)                       # low / mid / high primary energy
MAP_EN = {'proton': [(0.025e-9, 'meV'), (1e-9, 'eV'), (1e-3, 'MeV'), (15e-3, 'MeV')],
          'carbon': [(0.025e-9, 'meV'), (1e-9, 'eV'), (1e-3, 'MeV'), (30e-3, 'MeV')]}


def summarise(v, mask):
    a = np.abs(v[mask])
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                max=float(a.max()), n_out40=int(np.sum(a > 40)), n=int(a.size))


def metrics(species, csv, workers=3):
    Ep, dphi, dk = bias_curves(species, param_file=csv, workers=workers)
    train = np.zeros(len(Ep), bool)
    train[C.TRAIN_IDX] = True
    out = {}
    for split, m in (('all', np.ones_like(train)), ('train', train),
                     ('holdout', ~train)):
        out[split] = dict(dphi=summarise(dphi, m), dk=summarise(dk, m))
    out['per_energy'] = dict(E0=Ep.tolist(), dphi=dphi.tolist(), dk=dk.tolist())
    return out


def _cube(species, params, Ep):
    from model import Model
    from toolbox.production_range import build_xt_table
    z = np.load(f'{ROOT}/npy_data/z.npy')
    rho = np.load(f'{ROOT}/npy_data/rho.npy')
    en = np.load(f'{ROOT}/npy_data/en_low.npy')
    nz, nr, nE = len(z), len(rho), len(en)
    M = Model(species=species, XT=build_xt_table(species, np.array([Ep]), en),
              iEp=np.repeat(np.arange(1), nz * nr * nE),
              iEn=np.tile(np.arange(nE), nz * nr))
    EP, Z, R, En = [a.ravel() for a in np.meshgrid([Ep], z, rho, en, indexing='ij')]
    vals = [params[k] for k in C.PARAMS[species]]
    return getattr(M, C.ENTRY[species])((Z, R, En, EP), *vals).reshape(nz, nr, nE)


def draw_maps(species, sets, out_dir):
    """One PNG per primary energy, the sets side by side. Returns paths and
    core AM/MC medians (points above 50 % of the slice's MC maximum)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from toolbox.figures import maps

    z = np.load(f'{ROOT}/npy_data/z.npy')
    rho = np.load(f'{ROOT}/npy_data/rho.npy')
    en = np.load(f'{ROOT}/npy_data/en_low.npy')
    kk = [(int(np.argmin(np.abs(np.log(en / e)))), u) for e, u in MAP_EN[species]]
    E_all = energies(species)
    mc_all = np.load(f'{ROOT}/npy_data/{species}_mc.npy', mmap_mode='r')
    os.makedirs(out_dir, exist_ok=True)
    paths, core = [], {}
    for i in MAP_EP_IDX:
        Ep = float(E_all[i])
        mc = np.asarray(mc_all[i], dtype=float)
        blocks = []
        for lab, p in sets.items():
            am = _cube(species, p, Ep)
            rows = []
            for k, unit in kk:
                v = {'meV': en[k] * 1e12, 'eV': en[k] * 1e9, 'MeV': en[k] * 1e3}[unit]
                rows.append((rf'$E_{{\mathrm{{n}}}}={v:.2g}$ {unit}',
                             mc[:, :, k], am[:, :, k]))
                c = mc[:, :, k] > 0.5 * mc[:, :, k].max()
                core[f'{lab}|E0={Ep:.1f}|En={v:.2g} {unit}'] = float(
                    np.median(am[:, :, k][c] / mc[:, :, k][c]))
            blocks.append(dict(header=f'{lab}, $E_0 = {Ep:.0f}$ MeV'
                                      + ('/u' if species == 'carbon' else ''),
                               rows=rows))
        fig = maps.draw(blocks, z, rho,
                        quantity=r'$E_{\mathrm{n}}\,\phi(E_{\mathrm{n}})'
                                 r'\;(\mathrm{cm^{-2}\,primary^{-1}})$')
        path = os.path.join(out_dir, f'{species}_maps_E0idx{i:02d}.png')
        fig.savefig(path, dpi=90)
        plt.close(fig)
        paths.append(path)
    return paths, core


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--species', required=True, choices=list(C.PARAMS))
    ap.add_argument('--set', action='append', required=True,
                    help='label=path/to/params.csv (repeatable)')
    ap.add_argument('--maps', action='store_true')
    ap.add_argument('--map-dir', default=os.path.join(ROOT, 'fitting', 'results', 'maps'))
    ap.add_argument('--out', help='JSON with the metrics')
    ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args(argv)

    sets = dict(s.split('=', 1) for s in a.set)
    res = {}
    print(f'{"set":>14s} {"split":>8s} | {"dPhi mean":>9s} {"median":>7s} {"max":>7s} '
          f'{">40%":>4s} | {"dK mean":>8s} {"median":>7s} {"max":>7s} {">40%":>4s}')
    for lab, csv in sets.items():
        res[lab] = metrics(a.species, csv, a.workers)
        for split in ('all', 'train', 'holdout'):
            p, k = res[lab][split]['dphi'], res[lab][split]['dk']
            print(f'{lab:>14s} {split:>8s} | {p["mean"]:9.2f} {p["median"]:7.2f} '
                  f'{p["max"]:7.2f} {p["n_out40"]:4d} | {k["mean"]:8.2f} '
                  f'{k["median"]:7.2f} {k["max"]:7.2f} {k["n_out40"]:4d}')
    if a.maps:
        paths, core = draw_maps(a.species, {l: read_params(c, a.species)
                                            for l, c in sets.items()}, a.map_dir)
        res['maps'] = paths
        res['core_am_over_mc'] = core
        for p in paths:
            print('wrote', p)
        for k, v in core.items():
            print(f'   core AM/MC {k:45s} {v:.2f}')
    if a.out:
        with open(a.out, 'w') as fh:
            json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
