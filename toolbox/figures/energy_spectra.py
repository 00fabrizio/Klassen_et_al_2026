"""Generates figures/energy_spectra.pdf -- MC against the analytical model in the
energy-fluence representation, on the beam axis near the Bragg peak.

Three primary energies per species, chosen from the middle 70 % of the scanned
range, evaluated at z = 0.6 R(E_0) and rho = 0.

0.6 R, not 0.8 R. The proton cascade production range is P = gamma_cas xt with
gamma_cas = 0.743, i.e. 0.73 R at the on-axis spectral peak and 0.61 R at
64 MeV, so 0.8 R lands past the kernel's plateau and the proton panels showed
the falloff rather than the fit (AM/MC 0.44 at 64 MeV against 1.03 over the
bright core). Carbon is unaffected either way, gamma_cas = 0.904.

Two faults in the version this replaces:

  * it read the analytical model from {species}_model.npy, arrays written by the
    superseded factorized model, so the AM curves were not the current model;
  * it read the primary energies with pd.read_csv on data/{species}_energies.txt,
    which consumes the first value as a header. That gives 49 energies, whose
    indices were then used against the 50-row MC array, so every panel showed a
    spectrum one primary energy below its label -- 2.6-3.2 MeV for protons,
    5.2-6.3 MeV/u for carbon. The energies come from npy_data here.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter

from toolbox.figures import style
from toolbox.production_range import build_xt_table
from model import Model

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get('FIGURES_DIR', os.path.join(ROOT, 'figures'))
os.makedirs(OUT, exist_ok=True)
style.apply()

en = np.load(f'{ROOT}/npy_data/en_low.npy')
z = np.load(f'{ROOT}/npy_data/z.npy')
rho = np.load(f'{ROOT}/npy_data/rho.npy')
nz, nr, nE = len(z), len(rho), len(en)

SPEC = {'proton': ('params_proton.csv', 'spectral_energy_fluence', r'$^{1}\mathrm{H}$'),
        'carbon': ('params_carbon.csv', 'spectral_energy_fluence_theta', r'$^{12}\mathrm{C}$')}


def pick3(E, trim=0.15):
    """Three energies spanning the middle 70 % of the scanned range."""
    N = len(E)
    idx = np.arange(int(np.floor(trim * N)), int(np.ceil((1 - trim) * N)))
    qs = np.linspace(0, 1, 5)[1:-1]
    sel = idx[(qs * (len(idx) - 1)).round().astype(int)]
    return sel


def model_spectrum(sp, Ep, iz, ir):
    csv, entry, _ = SPEC[sp]
    p = pd.read_csv(f'{ROOT}/fitting_params/{csv}', index_col=0).loc['opt params'].astype(float)
    M = Model(species=sp, XT=build_xt_table(sp, np.array([Ep]), en),
              iEp=np.repeat(np.arange(1), nz * nr * nE),
              iEn=np.tile(np.arange(nE), nz * nr))
    EP, Z, R, En = [a.ravel() for a in np.meshgrid([Ep], z, rho, en, indexing='ij')]
    return getattr(M, entry)((Z, R, En, EP), *p).reshape(nz, nr, nE)[iz, ir, :]


fig = plt.figure(figsize=(15.5, 9.2))
gs = gridspec.GridSpec(3, 3, width_ratios=[1.0, style.GAP, 1.0],
                       hspace=0.0, wspace=0.0)

axes_col = {}
for col, sp in enumerate(('proton', 'carbon')):
    E_all = np.load(f'{ROOT}/npy_data/{sp}_energies.npy')
    mc_all = np.load(f'{ROOT}/npy_data/{sp}_mc.npy', mmap_mode='r')
    rng = pd.read_csv(f'{ROOT}/data/{sp}_range.csv')
    R_of_E = pd.Series(rng['Range'].to_numpy(), index=rng['Ep'].to_numpy())
    sel = pick3(E_all)
    axes_col[sp] = []
    print(f'{sp}: E_0 = {np.round(E_all[sel], 1)} MeV/u')

    for i, iEp in enumerate(sel):
        Ep = float(E_all[iEp])
        iz = int(np.abs(z - 0.6 * float(R_of_E.loc[Ep])).argmin())
        ir = int(np.abs(rho - 0.0).argmin())
        ax = fig.add_subplot(gs[i, 2 * col])
        axes_col[sp].append(ax)

        s = np.arange(0, en.size, 3)
        ax.plot(en[s], np.asarray(mc_all[iEp])[iz, ir, s], linestyle='none',
                marker='o', ms=3.5, color='black', label='MC')
        ax.plot(en, model_spectrum(sp, Ep, iz, ir), color=style.EV, label='AM')

        ax.set_xscale('log')
        ax.text(0.02, 0.95, rf'$E_0={Ep:.0f}\,\mathrm{{MeV/u}}$',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=style.annot_size())
        if i < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r'$E_{\mathrm{n}}\;(\mathrm{GeV})$')
        if i == 1 and col == 0:
            ax.set_ylabel(r'$E_{\mathrm{n}}\,\phi(E_{\mathrm{n}})'
                          r'\;(\mathrm{cm^{-2}\,primary^{-1}})$')

for ax in fig.axes:
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1,
                                          numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax.yaxis.minorticks_on()
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 4, 5, 10]))
    ax.grid(True, which='major', alpha=0.30, lw=0.8)
    ax.grid(True, which='minor', alpha=0.15, lw=0.5)

# The rows share edges (hspace=0), so a tick label sitting at a panel's top or
# bottom edge lands on the neighbouring panel's edge too and the two collide.
# MaxNLocator's prune= is not enough: it drops the first/last tick it GENERATES,
# which can be outside the view, leaving a label still sitting on the edge. Drop
# anything within 6 % of either edge instead, and freeze the limits so the
# fixed tick list stays valid.
for ax in fig.axes:
    lo, hi = ax.get_ylim()
    t = ax.get_yticks()
    ax.set_yticks(t[(t > lo + 0.06 * (hi - lo)) & (t < hi - 0.06 * (hi - lo))])
    ax.set_ylim(lo, hi)

# legend bottom left of the first panel, at the shared legend size
axes_col['proton'][0].legend(loc='lower left', frameon=False)

fig.subplots_adjust(**{**style.MARGINS, 'left': 0.095, 'top': 0.90,
                       'bottom': 0.08})

# species labels placed like figures 5 and 6: figure text above the axes, at the
# title size, the same absolute distance above the panels
fig.canvas.draw()
GAP_IN = 0.42                                    # inches above the axes, as in 5/6
dy = GAP_IN / fig.get_figheight()
for sp in ('proton', 'carbon'):
    box = axes_col[sp][0].get_position()
    fig.text(0.5 * (box.x0 + box.x1), box.y1 + dy, SPEC[sp][2],
             ha='center', va='bottom', fontsize=plt.rcParams['axes.titlesize'])

style.save(fig, 'energy_spectra', OUT)
print(f'\nwrote energy_spectra.pdf / .png to {OUT}')
