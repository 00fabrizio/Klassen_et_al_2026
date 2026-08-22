"""Generates tables/params.tex -- the manuscript's fitted-parameter table.

Written as a script because the table it replaces had gone stale: it still
carried the factorized model's values (sigma_cas, kappa_f/kappa_s,
Sigma_t = 0.346) long after those parameters had been superseded. Reading the
CSVs directly means the table cannot drift from the fits again.

Symbols follow the manuscript, with two disambiguations:

  Sigma     the cascade attenuation, unsubscripted. The code calls this Sigma_h
            to distinguish it from the factorized model's Sigma_t (0.346, which
            absorbed the 1/s^2 dilution) and from the true total cross section
            (~0.067). The manuscript carries only the coupled model and uses one
            unsubscripted Sigma throughout, introduced with the first-flight
            kernel exp(-Sigma s)/4 pi s^2; the fitted value is an EFFECTIVE
            removal constant well below sigma_tot, and the text says so.
            Sigma_a (absorption) and Sigma_r (two-group removal) keep theirs.

  n_ang     the cascade angular exponent, cos^n_ang(theta). The manuscript uses
            a bare n for this AND for the prefactor sharpness; they are different
            parameters (3.09 vs 2.20 for protons) and are separated here.

  kappa_slow  the single slow-neutron inverse diffusion length shared by the
            epithermal and thermal regimes, replacing the fixed pair
            (kappa_f, kappa_s) of the two-group form.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get('TABLES_DIR', os.path.join(ROOT, 'tables'))
os.makedirs(OUT, exist_ok=True)


def theta_to_n(t_deg):
    c = np.cos(np.radians(t_deg))
    return (2 * c - 1) / (1 - c)


def sig3(v):
    """Three significant figures, LaTeX, consistently for every parameter."""
    if v == 0:
        return '0'
    e = int(np.floor(np.log10(abs(v))))
    # Plain decimal only where it still shows exactly three significant digits.
    # e = 3 would print 2498.63 as "2499", which is four.
    if -2 <= e <= 2:
        s = f'{v:.{max(0, 2 - e)}f}'
        return f'${s}$' if v < 0 else s
    m = v / 10.0 ** e
    s = rf'{m:.2f}\times10^{{{e}}}'
    return f'${s}$'


P = pd.read_csv(f'{ROOT}/fitting_params/params_proton.csv',
                index_col=0).loc['opt params'].astype(float)
C = pd.read_csv(f'{ROOT}/fitting_params/params_carbon.csv',
                index_col=0).loc['opt params'].astype(float)

p = {k: float(v) for k, v in P.items()}
c = {k: float(v) for k, v in C.items()}
p['n_ang'] = p['n_ang']                                   # stored as exponent
c['n_ang'] = float(theta_to_n(c['theta_bar']))            # stored as mean angle

ROWS = [
    ('Global', [
        (r'$E_{\mathrm{th}}$', r'$\mathrm{MeV/u}$', 'E_th'),
        (r'$n$', '--', 'n'),
    ]),
    ('Cascade', [
        (r'$A_{\mathrm{cas}}$', r'$\mathrm{cm^{-2}\,GeV^{-1}}$', 'A1'),
        (r'$\gamma_{\mathrm{cas}}$', '--', 'gamma1'),
        (r'$\Sigma$', r'$\mathrm{cm^{-1}}$', 'Sigma'),
        (r'$n_{\mathrm{ang}}$', '--', 'n_ang'),
        (r'$d_{\mathrm{cas}}$', '--', 'd1'),
        (r'$a_{\mathrm{cas}}$', '--', 'a'),
        (r'$w_{\mathrm{cas}}$', '--', 'w_c'),
    ]),
    ('Evaporation', [
        (r'$A_{\mathrm{ev}}$', r'$\mathrm{cm^{-2}\,GeV^{-1}}$', 'A2'),
        (r'$\gamma_{\mathrm{ev}}$', '--', 'gamma2'),
        (r'$d_{\mathrm{ev}}$', '--', 'd2'),
        (r'$\kappa_{\mathrm{ev}}$', r'$\mathrm{cm^{-1}}$', 'kappa_ev'),
        (r'$E_{\mathrm{pk}}$', r'$\mathrm{GeV}$', 'Epk'),
    ]),
    ('Epithermal', [
        (r'$A_{\mathrm{ep}}$', r'$\mathrm{cm^{-2}}$', 'A3'),
        (r'$\gamma_{\mathrm{ep}}$', '--', 'gamma3'),
    ]),
    ('Thermal', [
        (r'$A_{\mathrm{th}}$', r'$\mathrm{cm^{-2}\,GeV^{-2}}$', 'A4'),
        (r'$\gamma_{\mathrm{th}}$', '--', 'gamma4'),
        (r'$\kappa_{\mathrm{slow}}$', r'$\mathrm{cm^{-1}}$', 'kappa_slow'),
    ]),
]

DEFAULT_CAPTION = (r'\caption{Optimized model parameters for proton (${}^{1}$H) '
                   r'and carbon (${}^{12}$C) beams.}')


def existing_caption(path):
    """Reuse the caption already in the file.

    The caption is TEXT and therefore the author's, not this script's. Rewriting
    it on every regeneration once silently replaced a hand-written caption, so
    the rule now is that regenerating changes numbers and never wording. A fresh
    checkout with no file yet gets DEFAULT_CAPTION.
    """
    if not os.path.exists(path):
        return DEFAULT_CAPTION
    for line in open(path):
        if line.lstrip().startswith(r'\caption'):
            return line.rstrip('\n')
    return DEFAULT_CAPTION


path = os.path.join(OUT, 'params.tex')

L = [
    r'\begin{table}[t]',
    r'\centering',
    existing_caption(path),
    r'\label{tab:model_params}',
    r'\begin{tabular}{l l r r}',
    r'\toprule',
    r'Parameter name & Unit & ${}^{1}$H & ${}^{12}$C \\',
]
for i, (sec, rows) in enumerate(ROWS):
    L.append(r'\midrule')
    L.append(rf'\multicolumn{{4}}{{l}}{{\textbf{{{sec}}}}} \\')
    for sym, unit, key in rows:
        L.append(rf'{sym} & {unit} & {sig3(p[key])} & {sig3(c[key])} \\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

open(path, 'w').write('\n'.join(L) + '\n')
print(f'wrote {path}\n')
for sec, rows in ROWS:
    print(f'  {sec}')
    for sym, unit, key in rows:
        print(f'    {key:10s} {sig3(p[key]):>22} {sig3(c[key]):>22}')
