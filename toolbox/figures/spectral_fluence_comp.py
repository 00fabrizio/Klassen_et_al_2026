"""Generates figures/spectral_fluence_comp.pdf -- the manuscript's spatial
comparison of spectral energy fluence.

Layout: two species side by side, four neutron energies as rows, MC | AM |
|Delta| as columns, rho mirrored about the beam axis, one colour norm shared
across each row. The layout itself is toolbox/figures/maps.py, shared with the
total-fluence and kerma figures, so all three carry the same panel shape, fonts
and colourbar treatment.

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

There is no figure title: the caption below the float does that, as it does for
the other two figures built on this layout.
"""
import os

import numpy as np
import pandas as pd

from toolbox.figures import maps
from toolbox.production_range import build_xt_table
from model import Model

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get('FIGURES_DIR', os.path.join(ROOT, 'figures'))
os.makedirs(OUT, exist_ok=True)

EP_IDX = 32          # the same index for both species

z = np.load(f'{ROOT}/npy_data/z.npy')
rho = np.load(f'{ROOT}/npy_data/rho.npy')
en = np.load(f'{ROOT}/npy_data/en_low.npy')
nz, nr, nE = len(z), len(rho), len(en)

# (index into en, unit to print it in)
EN_ROWS = [(46, 'meV'), (77, 'eV'), (199, 'MeV'), (227, 'MeV')]

SPEC = {
    'proton': ('params_proton.csv', 'spectral_energy_fluence', r'$^{1}\mathrm{H}$'),
    'carbon': ('params_carbon.csv', 'spectral_energy_fluence_theta', r'$^{12}\mathrm{C}$'),
}


def model_cube(sp, Ep):
    csv, entry, _ = SPEC[sp]
    p = pd.read_csv(f'{ROOT}/fitting_params/{csv}',
                    index_col=0).loc['opt params'].astype(float)
    M = Model(species=sp, XT=build_xt_table(sp, np.array([Ep]), en),
              iEp=np.repeat(np.arange(1), nz * nr * nE),
              iEn=np.tile(np.arange(nE), nz * nr))
    EP, Z, R, En = [a.ravel() for a in
                    np.meshgrid([Ep], z, rho, en, indexing='ij')]
    return getattr(M, entry)((Z, R, En, EP), *p).reshape(nz, nr, nE)


def en_label(e_gev, unit):
    v = {'meV': e_gev * 1e12, 'eV': e_gev * 1e9, 'MeV': e_gev * 1e3}[unit]
    return rf'$E_{{\mathrm{{n}}}}={v:.2g}$ {unit}'


blocks = []
for sp in ('proton', 'carbon'):
    E_all = np.load(f'{ROOT}/npy_data/{sp}_energies.npy')
    Ep = float(E_all[EP_IDX])
    mc_all = np.asarray(np.load(f'{ROOT}/npy_data/{sp}_mc.npy',
                                mmap_mode='r')[EP_IDX])
    am_all = model_cube(sp, Ep)
    rows = []
    print(f'{sp}: E0 = {Ep:.1f} MeV/u')
    for k, unit in EN_ROWS:
        mc, am = mc_all[:, :, k], am_all[:, :, k]
        rows.append((en_label(en[k], unit), mc, am))
        core = mc > 0.5 * mc.max()
        print(f'   {en_label(en[k], unit):30s} core AM/MC = '
              f'{np.median(am[core] / mc[core]):.2f}')
    # header carries the primary energy, since this figure fixes it
    blocks.append(dict(header=rf'{SPEC[sp][2]}, $E_0 = {Ep:.0f}$ MeV/u',
                       rows=rows))

fig = maps.draw(blocks, z, rho,
                quantity=r'$E_{\mathrm{n}}\,\phi(E_{\mathrm{n}})'
                         r'\;(\mathrm{cm^{-2}\,primary^{-1}})$',
                exp_per_row=True, rho_tick_step=2)
maps.save(fig, 'spectral_fluence_comp', OUT)
print(f'\nwrote spectral_fluence_comp.pdf / .png to {OUT}')
