"""Generates figures/volume_weighted_difference.pdf (Figure 7).

Unsigned volume-weighted difference B (Eq. 9) of the total fluence Phi and the
neutron kerma K, per primary energy. Top row: the whole scoring volume; bottom
row: the core region (rho <= R = 1 cm, z <= primary range), see
toolbox/metrics/metric_b.py. Filled markers are the 10 primary energies the fit
saw (fitting.config.TRAIN_IDX), hollow markers the 40 holdout energies.

Reads toolbox/metrics/results/metric_b_<species>.csv; regenerate those with
`python -m toolbox.metrics.metric_b` after any change of parameters, kernel
tables or kerma coefficients (this script refuses stale files). Also prints
mean, median and maximum B for all / fitted / holdout energies per species.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

from toolbox.figures import style

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get('FIGURES_DIR', os.path.join(ROOT, 'figures'))
RES = os.path.join(ROOT, 'toolbox', 'metrics', 'results')
os.makedirs(OUT, exist_ok=True)
style.apply()

SPECIES = [(r'$^{1}\mathrm{H}$',  'proton', style.PROTON, 'D'),
           (r'$^{12}\mathrm{C}$', 'carbon', style.CARBON, 'd')]
# (stat, row label) per row; columns are Phi and K
ROWS = [('B', 'full volume'), ('Bc', 'core')]
COLS = [('Phi', r'total fluence $\Phi$', r'\Phi'),
        ('K', r'neutron kerma $K$', 'K')]


def load(sp):
    f = os.path.join(RES, f'metric_b_{sp}.csv')
    par = os.path.join(ROOT, 'fitting_params', f'params_{sp}.csv')
    if os.path.getmtime(f) < os.path.getmtime(par):
        raise SystemExit(f'{f} is older than {par}: rerun python -m toolbox.metrics.metric_b')
    return pd.read_csv(f)


def main():
    res = {sp: load(sp) for _, sp, _, _ in SPECIES}

    for _, sp, _, _ in SPECIES:
        d = res[sp]
        fit = d['fitted'].to_numpy().astype(bool)
        for split, m in (('all', np.ones_like(fit)), ('fitted', fit), ('holdout', ~fit)):
            row = [f'{sp:7s} {split:8s} n={m.sum():2d}']
            for q, _, _ in COLS:
                for st, lab in ROWS:
                    a = d.loc[m, f'{q}_{st}'].abs()
                    row.append(f'{q} {lab}: mean {a.mean():5.2f} med {a.median():5.2f} max {a.max():5.2f}')
            print('  '.join(row))

    lim = 1.08 * max(res[sp][f'{q}_{st}'].max()
                     for _, sp, _, _ in SPECIES for q, _, _ in COLS for st, _ in ROWS)

    fig = plt.figure(figsize=(15.5, 10.5))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.0, style.GAP, 1.0], wspace=0.0,
                           hspace=0.12)
    axes = {}
    for r, (st, rlab) in enumerate(ROWS):
        for c, (q, title, sym) in enumerate(COLS):
            ax = fig.add_subplot(gs[r, 2 * c])
            axes[r, c] = ax
            for lab, sp, col, mk in SPECIES:
                d = res[sp]
                fit = d['fitted'].to_numpy().astype(bool)
                Ep, y = d['Ep'].to_numpy(), d[f'{q}_{st}'].to_numpy()
                ax.plot(Ep[~fit], y[~fit], linestyle='none', marker=mk, ms=7, color=col,
                        markerfacecolor='none', markeredgewidth=1.6, label=lab)
                ax.plot(Ep[fit], y[fit], linestyle='none', marker=mk, ms=7, color=col,
                        markeredgewidth=1.6)
            ax.set_ylim(0, lim)
            ax.grid(alpha=0.25)
            sub = '' if st == 'B' else r'^{\mathrm{core}}'
            ax.set_ylabel(rf'$B_{{{sym}}}{sub}$ (%)' if c == 0 else '')
            if c == 1:
                ax.tick_params(labelleft=False)
            if r == 0:
                ax.set_title(title)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(r'$E_0$ (MeV/u)')
            ax.text(0.97, 0.95, rlab, transform=ax.transAxes, ha='right', va='top',
                    fontsize=style.SIZES['legend_fontsize'])

    ax = axes[0, 0]
    ax.plot([], [], linestyle='none', marker='D', ms=7, color='0.25', label='fitted')
    ax.plot([], [], linestyle='none', marker='D', ms=7, color='0.25',
            markerfacecolor='none', markeredgewidth=1.6, label='holdout')
    style.boxed_legend(ax, loc='upper left', ncol=2, columnspacing=1.2)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.95)
    style.save(fig, 'volume_weighted_difference', OUT)
    print(f'\nwrote volume_weighted_difference.pdf / .png to {OUT}')


if __name__ == '__main__':
    main()
