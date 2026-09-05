"""Generates figures/volume_weighted_difference.pdf -- the volume-weighted bias
of the total fluence and the neutron kerma against primary energy.

Both quantities come from toolbox.metrics.bias_curves, which implements the
manuscript's definitions directly:

  Eq. 10   Phi = sum phi dE,  K = sum k_phi phi dE, on the MC energy grid.
           The npy cache holds E dPhi/dE, so phi = mc/en_low and the weights are
           dE/en and dE kc/en. Using dE and dE kc would carry a spurious factor
           of E and move evaporation from 8.6 % to 42.1 % of the kerma.

  Eq. 11   the (z, rho) map is weighted by the annular scoring volume
           V = pi (rho_j^2 - rho_{j-1}^2) dz, not by rho. The rho grid holds
           outer edges, so weighting by rho understates the innermost bin by a
           factor 2.

Points, not lines: there are 50 discrete primary energies and nothing connects
them, so a line would imply an interpolation that was never computed.
"""
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

NR = 55                                   # full scored range, rho <= 5.5 cm
SPECIES = [(r'$^{1}\mathrm{H}$',  'proton', style.PROTON, 'D'),
           (r'$^{12}\mathrm{C}$', 'carbon', style.CARBON, 'd')]

res = {}
for lab, sp, col, mk in SPECIES:
    Ep, dphi, dk = bias_curves(sp, nr=NR)
    res[sp] = (Ep, dphi, dk)
    print(f'{lab:22s} mean |D_Phi| = {np.abs(dphi).mean():5.2f} %   '
          f'mean |D_K| = {np.abs(dk).mean():5.2f} %')

fig, axes = plt.subplots(1, 2, figsize=(15.5, 6))
PANELS = [(axes[0], 1, r'total fluence $\Phi$', r'$\Delta_{\Phi}$ (%)'),
          (axes[1], 2, r'neutron kerma $K$',    r'$\Delta_{K}$ (%)')]

for ax, idx, title, ylab in PANELS:
    lim = 0.0
    for lab, sp, col, mk in SPECIES:
        v = res[sp][idx]
        ax.plot(res[sp][0], v, linestyle='none', marker=mk, ms=7,
                color=col, markerfacecolor='none', markeredgewidth=1.6,
                label=lab)
        lim = max(lim, np.abs(v).max())
    ax.axhline(0, color='k', lw=0.9)
    ax.set_ylim(-1.12 * lim, 1.12 * lim)          # symmetric about zero
    ax.set_xlabel(r'$E_0$ (MeV/u)')
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, f'volume_weighted_difference.{ext}'),
                bbox_inches='tight', dpi=150 if ext == 'png' else None)
print(f'\nwrote volume_weighted_difference.pdf / .png to {OUT}')
