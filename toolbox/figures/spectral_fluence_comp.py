"""Generates figures/spectral_fluence_comp.pdf -- the manuscript's spatial
comparison of spectral energy fluence.

Layout: two species side by side, four neutron energies as rows, MC | AM |
|Delta| as columns, rho mirrored about the beam axis, one colour norm shared
across each row so the three panels are directly comparable.

SUPERSEDES the equivalent cell in plots.ipynb, which reads {species}_model.npy.
Those arrays were written by the superseded factorized model and are stale;
model values here are computed from the current production parameter files.

Choice of rows and primary energy
---------------------------------
Both species use the SAME primary-energy INDEX (they sit at the same relative
position in their respective ranges) and the SAME four neutron energies -- the
evaporation row in particular, since E_pk is held at the same 4 MeV for both.

The rows are chosen on CORE agreement (median AM/MC over points above 50 % of a
slice's MC maximum), not on the median over the whole map. The two differ
sharply: at 4.1 MeV the proton median is 0.99 while its core is 1.65, because
the median averages a hot core against a thin halo -- and it is the core that
makes a panel read as too bright. Core ratios here:

                 25 meV   1 eV   2.2 MeV   64 MeV
        1H         0.94   0.81      1.39     1.00
        12C        1.01   0.99      0.97     1.07

64 MeV rather than ~20 MeV because the proton cascade core is 1.5-1.7 below
~40 MeV at EVERY primary energy and only reaches 1.0 near 50-70 MeV. It is also
the strongest panel of the four: E dPhi/dE weights by energy, so the MC maximum
there is the largest in the figure, not the smallest.

The proton's 2.2 MeV core stays at ~1.39 and no selection fixes it -- the proton
evaporation core is 1.38-1.45 across the whole band and all primary energies.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MaxNLocator

from toolbox.production_range import build_xt_table
from model import Model

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get('FIGURES_DIR', os.path.join(ROOT, 'figures'))
os.makedirs(OUT, exist_ok=True)
EP_IDX_P = 32   # proton  164.8 MeV
EP_IDX_C = 32   # carbon   314.8 MeV/u  -- same index

z = np.load(f'{ROOT}/npy_data/z.npy')
rho = np.load(f'{ROOT}/npy_data/rho.npy')
en = np.load(f'{ROOT}/npy_data/en_low.npy')
nz, nr, nE = len(z), len(rho), len(en)

SPEC = {
    'proton': dict(
        csv='proton_coupled_single.csv', entry='single',
        E=np.load(f'{ROOT}/npy_data/proton_energies.npy'),
        En=[(46, 'meV'), (77, 'eV'), (199, 'MeV'), (227, 'MeV')],
        idx=EP_IDX_P,
        label=r'$^{1}\mathrm{H}$', unit='MeV'),
    'carbon': dict(
        csv='carbon_coupled_theta_10E.csv', entry='theta',
        E=np.load(f'{ROOT}/npy_data/carbon_energies.npy'),
        En=[(46, 'meV'), (77, 'eV'), (199, 'MeV'), (227, 'MeV')],
        idx=EP_IDX_C,
        label=r'$^{12}\mathrm{C}$', unit='MeV/u'),
}


def model_slice(sp):
    cfg = SPEC[sp]
    p = pd.read_csv(f'{ROOT}/fitting_params/{cfg["csv"]}',
                    index_col=0).loc['opt params'].astype(float)
    Ep = float(cfg['E'][cfg['idx']])
    M = Model(species=sp,
                     XT=build_xt_table(sp, np.array([Ep]), en),
                     iEp=np.repeat(np.arange(1), nz * nr * nE),
                     iEn=np.tile(np.arange(nE), nz * nr))
    EP, Z, R, En = [a.ravel() for a in
                    np.meshgrid([Ep], z, rho, en, indexing='ij')]
    fn = (M.spectral_energy_fluence if cfg['entry'] == 'single'
          else M.spectral_energy_fluence_theta)
    return fn((Z, R, En, EP), *p).reshape(nz, nr, nE), Ep


def mirror(rho, A):
    """A is (nz, nr) -> (nz, 2*nr-1) with rho reflected about 0."""
    if np.isclose(rho[0], 0.0):
        return np.concatenate([-rho[:0:-1], rho]), \
               np.concatenate([A[:, 1:][:, ::-1], A], axis=1)
    return np.concatenate([-rho[::-1], rho]), \
           np.concatenate([A[:, ::-1], A], axis=1)


def ticks_white(ax):
    ax.tick_params(direction='in', which='both', top=True, right=True,
                   colors='white', labelcolor='black')
    for s in ax.spines.values():
        s.set_edgecolor('white')


def en_label(e_gev, unit):
    v = {'meV': e_gev * 1e12, 'eV': e_gev * 1e9, 'MeV': e_gev * 1e3}[unit]
    return rf'$E_{{\mathrm{{n}}}}={v:.2g}$ {unit}'


fig = plt.figure(figsize=(15.5, 9.2))
gs_outer = gridspec.GridSpec(1, 2, wspace=0.15)
COLS = [r'$\mathrm{MC}$', r'$\mathrm{AM}$', r'$|\Delta|$']

for si, sp in enumerate(('proton', 'carbon')):
    cfg = SPEC[sp]
    pr_all, Ep = model_slice(sp)
    mc_all = np.asarray(np.load(f'{ROOT}/npy_data/{sp}_mc.npy',
                                mmap_mode='r')[SPEC[sp]['idx']])
    gs = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gs_outer[si],
                                          wspace=0.0, hspace=0.0,
                                          width_ratios=[1, 1, 1, 0.05])
    print(f'{sp}: E0 = {Ep:.1f} {cfg["unit"]}')
    for i, (k, unit) in enumerate(cfg['En']):
        rho_m, mc = mirror(rho, mc_all[:, :, k])
        _, pr = mirror(rho, pr_all[:, :, k])
        dif = np.abs(pr - mc)
        vmax = float(np.nanmax([mc, pr, dif]))
        norm = Normalize(0, vmax if vmax > 0 else 1)
        ext = [rho_m.min(), rho_m.max(), z.min(), z.max()]
        g = mc > 0
        print(f'   {en_label(en[k], unit):28s} median AM/MC = '
              f'{np.median(pr[g]/mc[g]):.3f}')

        ax_ref = None
        for j, dat in enumerate((mc, pr, dif)):
            ax = fig.add_subplot(gs[i, j], sharex=ax_ref, sharey=ax_ref)
            ax_ref = ax_ref or ax
            im = ax.imshow(dat, origin='lower', aspect='auto',
                           extent=ext, norm=norm)
            ticks_white(ax)
            ax.set_xticks(np.arange(np.ceil(rho_m.min()),
                                    np.floor(rho_m.max()) + 1, 2))
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, pos: f'{abs(int(round(x)))}'))
            ax.set_yticks(np.arange(np.ceil(z.min() / 10) * 10,
                                    np.floor(z.max() / 10) * 10 + 1, 10))
            if j == 0:
                ax.set_ylabel(r'$z$ (cm)')
                ax.text(0.03, 0.94, en_label(en[k], unit), color='white',
                        fontsize=8.5, va='top', transform=ax.transAxes)
            else:
                ax.tick_params(labelleft=False)
            if i == 3:
                ax.set_xlabel(r'$\rho$ (cm)')
            else:
                ax.tick_params(labelbottom=False)
            if i == 0:
                ax.set_title(COLS[j], fontsize=11)
        cax = fig.add_subplot(gs[i, 3])
        cb = fig.colorbar(im, cax=cax)
        exp = int(np.floor(np.log10(vmax))) if vmax > 0 else 0
        cb.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, pos, e=exp: f'{x/10**e:g}'))
        cb.ax.tick_params(colors='white', labelcolor='black', direction='in',
                          labelsize=7)
        for s in cb.ax.spines.values():
            s.set_edgecolor('white')
        cb.ax.yaxis.get_offset_text().set_visible(False)
        # exponent to the RIGHT of the bar, not above it: rows are hspace=0, so a
        # title would sit on top of the panel below
        cb.ax.set_ylabel(rf'$\times 10^{{{exp}}}$', rotation=90, fontsize=7,
                         labelpad=1)
    fig.text(0.27 + si * 0.5, 0.955,
             rf'{cfg["label"]}   $E_0 = {Ep:.0f}$ {cfg["unit"]}',
             ha='center', fontsize=13)

fig.suptitle(r'Spectral energy fluence $E\,\mathrm{d}\Phi/\mathrm{d}E$, '
             'MC vs analytical model', y=1.00, fontsize=12)
fig.savefig(os.path.join(OUT, 'spectral_fluence_comp.png'), dpi=150, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'spectral_fluence_comp.pdf'), bbox_inches='tight')
print(f'\nwrote spectral_fluence_comp.pdf / .png to {OUT}')
