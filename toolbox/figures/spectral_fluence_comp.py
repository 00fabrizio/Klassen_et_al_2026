"""Generates figures/spectral_fluence_comp.pdf."""
import os

import numpy as np
import pandas as pd

from toolbox.figures import maps
from toolbox.production_range import build_xt_table
from model import Model

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get('FIGURES_DIR', os.path.join(ROOT, 'figures'))
os.makedirs(OUT, exist_ok=True)

EP_IDX = 32

z = np.load(f'{ROOT}/npy_data/z.npy')
rho = np.load(f'{ROOT}/npy_data/rho.npy')
en = np.load(f'{ROOT}/npy_data/en_low.npy')
nz, nr, nE = len(z), len(rho), len(en)


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

    blocks.append(dict(header=rf'{SPEC[sp][2]}, $E_0 = {Ep:.0f}$ MeV/u',
                       rows=rows))

fig = maps.draw(blocks, z, rho,
                quantity=r'$E_{\mathrm{n}}\,\phi(E_{\mathrm{n}})'
                         r'\;(\mathrm{cm^{-2}\,primary^{-1}})$',
                exp_per_row=True, rho_tick_step=2)
maps.save(fig, 'spectral_fluence_comp', OUT)
print(f'\nwrote spectral_fluence_comp.pdf / .png to {OUT}')
