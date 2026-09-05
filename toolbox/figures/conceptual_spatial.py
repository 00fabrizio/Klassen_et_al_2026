"""Generates figures/conceptual_spatial.pdf -- the schematic z- and rho-dependence
of the spatial factor of each energy regime.

Every curve is the CURRENT model kernel, evaluated on one grid:

    cascade              cascade_coupled(P, Sigma, n_ang, z, rho)
    evaporation          evaporation_diff(P, kappa_ev, z, rho)
    epithermal/thermal   slow_diff_single(P, kappa_slow, z, rho)

The version this replaces drew the cascade's two panels from two DIFFERENT
objects -- an axial buildup law for the z panel and an ad hoc Gaussian-smeared
disk for the rho panel -- so the two halves of the figure did not describe the
same function. It also used the two-group slow kernel with kappa_f and kappa_s.
Both of those belong to the superseded factorized model and the functions no
longer exist. Here each regime is one F(z, rho): the left panel is its rho = 0
cut and the right panel its z = P cut, which is what the axes claim.

Parameters are illustrative but of the same order as the fitted ones.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator

from toolbox.figures import style
from toolbox.diffusion_integrals.diffusion_kernels import (
    cascade_coupled, evaporation_diff, slow_diff_single,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get('FIGURES_DIR', os.path.join(ROOT, 'figures'))
os.makedirs(OUT, exist_ok=True)
style.apply()

# geometry
P, L, R, RHO_MAX = 20.0, 45.0, 1.0, 3.5
# transport constants
SIGMA, N_ANG = 0.02, 3.0
KAPPA_EV, KAPPA_SLOW = 0.3, 0.2

z = np.linspace(0.0, L, 400)
rho = np.linspace(0.0, RHO_MAX, 300)
Z, RHO = np.meshgrid(z, rho, indexing='ij')

F = {
    'cas':  cascade_coupled(P, SIGMA, N_ANG, Z, RHO),
    'ev':   evaporation_diff(P=P, kappa=KAPPA_EV, z=Z, r=RHO),
    'slow': slow_diff_single(P=P, kappa=KAPPA_SLOW, z=Z, r=RHO),
}
F = {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in F.items()}

iP = int(np.abs(z - P).argmin())
norm = lambda a: a / a.max() if a.max() > 0 else a

CURVES = [
    ('cas',  style.CAS,
     rf'Cascade ($\Sigma={SIGMA}\,\mathrm{{cm^{{-1}}}},\ '
     rf'n_{{\mathrm{{ang}}}}={N_ANG:.0f}$)'),
    ('ev',   style.EV,
     rf'Evaporation ($\kappa_{{\mathrm{{ev}}}}={KAPPA_EV}\,\mathrm{{cm^{{-1}}}}$)'),
    ('slow', style.SLOW,
     rf'Epithermal / thermal ($\kappa_{{\mathrm{{slow}}}}={KAPPA_SLOW}'
     rf'\,\mathrm{{cm^{{-1}}}}$)'),
]

fig = plt.figure(figsize=(15.5, 6))
gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 0.12, 1.0], wspace=0.0)
ax_z = fig.add_subplot(gs[0, 0])
ax_r = fig.add_subplot(gs[0, 2])

for key, col, lab in CURVES:
    ax_z.plot(z, norm(F[key][:, 0]), color=col, label=lab)      # rho = 0 cut
    ax_r.plot(rho, norm(F[key][iP, :]), color=col)              # z = P cut

for ax, xv, txt, xmax, step in ((ax_z, P, r'$P$', L, 5),
                                (ax_r, R, r'$R$', RHO_MAX, 0.5)):
    ax.axvline(xv, color='k', ls=':', lw=2.2)
    ax.text(xv + 0.015 * xmax, 0.12, txt, ha='left', va='center',
            fontsize=style.annot_size())
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(True, which='major', alpha=0.25)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.05)

ax_z.set_title(r'$z$ dependence ($\rho=0$)', pad=8)
ax_z.set_xlabel(r'$z$ (cm)')
ax_z.set_ylabel(r'relative spectral fluence (a.u.)')
ax_r.set_title(r'$\rho$ dependence ($z=P$)', pad=8)
ax_r.set_xlabel(r'$\rho$ (cm)')

h, l = ax_z.get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.0))
fig.subplots_adjust(left=0.07, right=0.98, bottom=0.16, top=0.82)

for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, f'conceptual_spatial.{ext}'),
                bbox_inches='tight', dpi=150 if ext == 'png' else None)
print(f'wrote conceptual_spatial.pdf / .png to {OUT}')
for key, _, lab in CURVES:
    a = F[key]
    print(f'  {key:5s} peak at z={z[a[:,0].argmax()]:5.1f} cm (rho=0), '
          f'rho-width at z=P: half-max at {rho[np.argmax(norm(a[iP,:])<0.5)]:.2f} cm')
