"""Generates figures/volume_weighted_difference.pdf.

Filled markers are the 10 primary energies the fit saw (fitting.config.TRAIN_IDX),
hollow markers the 40 holdout energies, which never entered it. Also prints mean,
median and maximum |Delta| and the count outside +-40 % for all / training /
holdout energies per species (the same summary as fitting/evaluate.py).
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fitting.config import TRAIN_IDX
from toolbox.figures import style
from toolbox.metrics.bias_curves import bias_curves

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get('FIGURES_DIR', os.path.join(ROOT, 'figures'))
os.makedirs(OUT, exist_ok=True)
style.apply()

NR = 55
SPECIES = [(r'$^{1}\mathrm{H}$',  'proton', style.PROTON, 'D'),
           (r'$^{12}\mathrm{C}$', 'carbon', style.CARBON, 'd')]


def main():
    res = {}
    for lab, sp, col, mk in SPECIES:
        Ep, dphi, dk = bias_curves(sp, nr=NR, workers=2)
        res[sp] = (Ep, dphi, dk)
        train = np.zeros(len(Ep), bool)
        train[TRAIN_IDX] = True
        inside = (Ep >= Ep[train].min()) & (Ep <= Ep[train].max())
        print(f'{sp}: training range {Ep[train].min():.1f} - {Ep[train].max():.1f} MeV/u, '
              f'{np.sum(~train & ~inside)} holdout energies outside it')
        for split, m in (('all', np.ones_like(train)), ('train', train),
                         ('holdout', ~train), ('holdout, inside train range', ~train & inside)):
            row = [f'{sp:7s} {split:28s} n={m.sum():2d}']
            for nm, v in (('dPhi', dphi), ('dK', dk)):
                a = np.abs(v[m])
                row.append(f'{nm}: mean {a.mean():5.2f} median {np.median(a):5.2f} '
                           f'max {a.max():5.2f} >40%: {int(np.sum(a > 40))}')
            print('  '.join(row))

    fig, ax_phi, ax_k = style.two_panel()
    PANELS = [(ax_phi, 1, r'total fluence $\Phi$', r'$\Delta_{\Phi}$ (%)'),
              (ax_k,   2, r'neutron kerma $K$',    r'$\Delta_{K}$ (%)')]


    lim = 1.12 * max(np.abs(res[sp][i]).max() for _, sp, _, _ in SPECIES for i in (1, 2))

    train_mask = np.zeros(50, bool)
    train_mask[TRAIN_IDX] = True

    for ax, idx, title, ylab in PANELS:
        for lab, sp, col, mk in SPECIES:
            Ep, y = res[sp][0], res[sp][idx]
            # holdout: hollow (the settled figure-7 marker); training: filled
            ax.plot(Ep[~train_mask], y[~train_mask], linestyle='none', marker=mk, ms=7,
                    color=col, markerfacecolor='none', markeredgewidth=1.6,
                    label=lab)
            ax.plot(Ep[train_mask], y[train_mask], linestyle='none', marker=mk, ms=7,
                    color=col, markeredgewidth=1.6)
        ax.axhline(0, color='k', lw=0.9)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(r'$E_0$ (MeV/u)')
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        # neutral proxies explain the fill; the species handles above stay hollow
        ax.plot([], [], linestyle='none', marker='D', ms=7, color='0.25', label='fitted')
        ax.plot([], [], linestyle='none', marker='D', ms=7, color='0.25',
                markerfacecolor='none', markeredgewidth=1.6, label='holdout')
        style.boxed_legend(ax, ncol=2, columnspacing=1.2)


    fig.subplots_adjust(**{**style.MARGINS, 'top': 0.91})
    style.save(fig, 'volume_weighted_difference', OUT)
    print(f'\nwrote volume_weighted_difference.pdf / .png to {OUT}')


if __name__ == '__main__':
    main()
