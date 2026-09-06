"""Generates figures/volume_weighted_difference.pdf."""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from toolbox.figures import style
from toolbox.metrics.bias_curves import bias_curves

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get('FIGURES_DIR', os.path.join(ROOT, 'figures'))
os.makedirs(OUT, exist_ok=True)
style.apply()

NR = 55
SPECIES = [(r'$^{1}\mathrm{H}$',  'proton', style.PROTON, 'D'),
           (r'$^{12}\mathrm{C}$', 'carbon', style.CARBON, 'd')]

res = {}
for lab, sp, col, mk in SPECIES:
    Ep, dphi, dk = bias_curves(sp, nr=NR)
    res[sp] = (Ep, dphi, dk)
    print(f'{lab:22s} mean |D_Phi| = {np.abs(dphi).mean():5.2f} %   '
          f'mean |D_K| = {np.abs(dk).mean():5.2f} %')

fig, ax_phi, ax_k = style.two_panel()
PANELS = [(ax_phi, 1, r'total fluence $\Phi$', r'$\Delta_{\Phi}$ (%)'),
          (ax_k,   2, r'neutron kerma $K$',    r'$\Delta_{K}$ (%)')]


lim = 1.12 * max(np.abs(res[sp][i]).max() for _, sp, _, _ in SPECIES for i in (1, 2))

for ax, idx, title, ylab in PANELS:
    for lab, sp, col, mk in SPECIES:
        ax.plot(res[sp][0], res[sp][idx], linestyle='none', marker=mk, ms=7,
                color=col, markerfacecolor='none', markeredgewidth=1.6,
                label=lab)
    ax.axhline(0, color='k', lw=0.9)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r'$E_0$ (MeV/u)')
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    style.boxed_legend(ax)


fig.subplots_adjust(**{**style.MARGINS, 'top': 0.91})
style.save(fig, 'volume_weighted_difference', OUT)
print(f'\nwrote volume_weighted_difference.pdf / .png to {OUT}')
