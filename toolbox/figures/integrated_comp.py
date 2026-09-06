"""Generates figures/Phi_comp.pdf and figures/K_comp.pdf."""
import os

import numpy as np
import pandas as pd

from toolbox.figures import maps
from toolbox.metrics.kerma_coeffs import k_coeff_pGy_cm2_from_GeV
from toolbox.production_range import build_xt_table
from model import Model

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get('FIGURES_DIR', os.path.join(ROOT, 'figures'))
os.makedirs(OUT, exist_ok=True)

z = np.load(f'{ROOT}/npy_data/z.npy')
rho = np.load(f'{ROOT}/npy_data/rho.npy')
en = np.load(f'{ROOT}/npy_data/en_low.npy')
dE = np.load(f'{ROOT}/npy_data/en_upp.npy') - en
kc = k_coeff_pGy_cm2_from_GeV(en)
nz, nr, nE = len(z), len(rho), len(en)


W_PHI = dE / en
W_K = dE * kc / en

SPEC = {
    'proton': ('params_proton.csv', 'spectral_energy_fluence', r'$^{1}\mathrm{H}$'),
    'carbon': ('params_carbon.csv', 'spectral_energy_fluence_theta', r'$^{12}\mathrm{C}$'),
}


def pick3(E, trim=0.15):
    N = len(E)
    idx = np.arange(int(np.floor(trim * N)), int(np.ceil((1 - trim) * N)))
    qs = np.linspace(0, 1, 5)[1:-1]
    return idx[(qs * (len(idx) - 1)).round().astype(int)]


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


def blocks_for(weight):
    out = []
    for sp in ('proton', 'carbon'):
        E_all = np.load(f'{ROOT}/npy_data/{sp}_energies.npy')
        mc_all = np.load(f'{ROOT}/npy_data/{sp}_mc.npy', mmap_mode='r')
        rows = []
        for iEp in pick3(E_all):
            Ep = float(E_all[iEp])
            mc = np.sum(np.asarray(mc_all[iEp]) * weight, axis=2)
            am = np.sum(model_cube(sp, Ep) * weight, axis=2)
            rows.append((rf'$E_0={Ep:.0f}$ MeV/u', mc, am))
            g = mc > 0
            print(f'   {sp:7s} E0={Ep:6.1f} MeV/u   median AM/MC = '
                  f'{np.median(am[g] / mc[g]):.3f}')
        out.append(dict(header=SPEC[sp][2], rows=rows))
    return out


for name, weight, quantity in (
        ('Phi_comp', W_PHI, r'$\Phi\;(\mathrm{cm^{-2}\,primary^{-1}})$'),
        ('K_comp', W_K, r'$K\;(\mathrm{pGy\,primary^{-1}})$')):
    print(f'{name}:')
    fig = maps.draw(blocks_for(weight), z, rho, quantity=quantity,
                    exp_per_row=False, rho_tick_step=2)
    maps.save(fig, name, OUT)
    print(f'   wrote {name}.pdf / .png to {OUT}\n')
