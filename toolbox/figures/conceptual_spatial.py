"""Generates figures/conceptual_spatial.pdf."""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from toolbox.figures import style
from toolbox.diffusion_integrals import precompute_cascade as PC
from toolbox.diffusion_integrals import precompute_kernels as PK
from toolbox.diffusion_integrals import precompute_slow as PS
from toolbox.diffusion_integrals.kernel_config import L_KERNEL, R_KERNEL

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get('FIGURES_DIR', os.path.join(ROOT, 'figures'))
os.makedirs(OUT, exist_ok=True)
style.apply()


P, L, R, RHO_MAX = 20.0, 45.0, 1.0, 5.5

SIGMA, N_ANG = 0.02, 3.0
KAPPA_EV, KAPPA_SLOW = 0.3, 0.2


z = np.linspace(0.0, L, 700)
rho = np.linspace(0.0, RHO_MAX, 450)


def kernels(z_vals, r_vals):
    cas = np.array([[PC.cascade_kernel_point(float(zz), float(rr), P, N_ANG, SIGMA)
                     for zz in z_vals] for rr in r_vals])
    ev = PK.evap_kernel(np.asarray(r_vals, float), np.asarray(z_vals, float),
                        R=R_KERNEL, kappa=KAPPA_EV, P=P)
    B = PS.axial_factor_squared_bc(np.asarray(z_vals, float), P, L_KERNEL,
                                   np.hypot(PK.k, KAPPA_SLOW))
    slow = PK._hankel_transform(np.asarray(r_vals, float), R_KERNEL, B)
    return cas, ev, slow


cas_z, ev_z, slow_z = kernels(z, np.array([0.0]))
cas_r, ev_r, slow_r = kernels(np.array([P]), rho)

CUT_Z = {'cas': cas_z[0], 'ev': ev_z[0], 'slow': slow_z[0]}
CUT_R = {'cas': cas_r[:, 0], 'ev': ev_r[:, 0], 'slow': slow_r[:, 0]}
CUT_Z = {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in CUT_Z.items()}
CUT_R = {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in CUT_R.items()}

norm = lambda a: a / a.max() if a.max() > 0 else a

# Line styles keep the three regimes apart in black and white (R1-13): solid,
# dashed, dotted. The dotted curve is drawn slightly heavier so its dots read at
# print size, and the P / R reference lines are a thin grey dash-dot so they are
# not mistaken for it.
LINESTYLE = {
    'cas':  dict(ls='-'),
    'ev':   dict(ls=(0, (6, 3))),
    'slow': dict(ls=(0, (1, 1.6)), lw=2.8),
}
REF_LINE = dict(color='0.35', ls=(0, (8, 3, 1.5, 3)), lw=1.3)

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

fig, ax_z, ax_r = style.two_panel()

for key, col, lab in CURVES:
    ax_z.plot(z, norm(CUT_Z[key]), color=col, label=lab, **LINESTYLE[key])
    ax_r.plot(rho, norm(CUT_R[key]), color=col, **LINESTYLE[key])

for ax, xv, txt, xmax, step in ((ax_z, P, r'$P$', L, 5),
                                (ax_r, R, r'$R$', RHO_MAX, 1.0)):
    ax.axvline(xv, **REF_LINE)
    ax.text(xv - 0.012 * xmax, 0.055, txt, ha='right', va='center',
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

# One row. At the shared legend size (21) these labels overflow the canvas under
# matplotlib 3.11, so the legend is set one point smaller with shorter handles
# and tighter spacing: 95 % of the canvas width.
style.top_legend(fig, ax_z, ncol=3, fontsize=style.SIZES['legend_fontsize'] - 1,
                 handlelength=2.4, columnspacing=1.2, handletextpad=0.5)

style.save(fig, 'conceptual_spatial', OUT)
print(f'wrote conceptual_spatial.pdf / .png to {OUT}')
for key, _, _ in CURVES:
    nz_, nr_ = norm(CUT_Z[key]), norm(CUT_R[key])
    below = np.where(nr_ < 0.5)[0]
    hw = f'{rho[below[0]]:.2f} cm' if len(below) else f'>{RHO_MAX:.1f} cm'
    print(f'  {key:5s} z-peak at {z[nz_.argmax()]:5.1f} cm; '
          f'rho half-max at {hw}')
