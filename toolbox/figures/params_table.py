"""Generates tables/params.tex."""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get('TABLES_DIR', os.path.join(ROOT, 'tables'))
os.makedirs(OUT, exist_ok=True)


def theta_to_n(t_deg):
    c = np.cos(np.radians(t_deg))
    return (2 * c - 1) / (1 - c)


FLOOR = 3


def format_pair(a, b, floor=FLOOR):
    nz = [v for v in (a, b) if v != 0]
    if not nz:
        return '0', '0'
    kmax = int(np.floor(np.log10(max(abs(v) for v in nz))))
    use_sci = kmax < -2 or kmax > 2
    k = kmax if use_sci else 0
    A, B = a / 10.0 ** k, b / 10.0 ** k

    dec = max(-int(np.floor(np.log10(abs(v)))) + floor - 1
              for v in (A, B) if v != 0)
    dec = max(dec, 0)

    def cell(v):
        if v == 0:
            return '0'
        t = f'{v:.{dec}f}'
        if use_sci:
            return rf'${t}\times10^{{{k}}}$'
        return f'${t}$' if v < 0 else t

    return cell(A), cell(B)


P = pd.read_csv(f'{ROOT}/fitting_params/params_proton.csv',
                index_col=0).loc['opt params'].astype(float)
C = pd.read_csv(f'{ROOT}/fitting_params/params_carbon.csv',
                index_col=0).loc['opt params'].astype(float)

p = {k: float(v) for k, v in P.items()}
c = {k: float(v) for k, v in C.items()}
p['n_ang'] = p['n_ang']
c['n_ang'] = float(theta_to_n(c['theta_bar']))

ROWS = [
    ('Global', [

        (r'$E_{\mathrm{ref}}$', r'$\mathrm{MeV/u}$', 'E_th'),
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
    ]),

    ('Epithermal', [
        (r'$\kappa_{\mathrm{slow}}$', r'$\mathrm{cm^{-1}}$', 'kappa_slow'),
        (r'$A_{\mathrm{ep}}$', r'$\mathrm{cm^{-2}}$', 'A3'),
        (r'$\gamma_{\mathrm{ep}}$', '--', 'gamma3'),
    ]),
    ('Thermal', [
        (r'$A_{\mathrm{th}}$', r'$\mathrm{cm^{-2}\,GeV^{-2}}$', 'A4'),
        (r'$\gamma_{\mathrm{th}}$', '--', 'gamma4'),
    ]),
]

DEFAULT_CAPTION = (r'\caption{Optimized model parameters for proton (${}^{1}$H) '
                   r'and carbon (${}^{12}$C) beams.}')


def existing_caption(path):
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

    r'\setlength{\belowcaptionskip}{6pt}',
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
        ca, cb = format_pair(p[key], c[key])
        L.append(rf'{sym} & {unit} & {ca} & {cb} \\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

open(path, 'w').write('\n'.join(L) + '\n')
print(f'wrote {path}\n')
for sec, rows in ROWS:
    print(f'  {sec}')
    for sym, unit, key in rows:
        ca, cb = format_pair(p[key], c[key])
        print(f'    {key:10s} {ca:>26} {cb:>26}')
