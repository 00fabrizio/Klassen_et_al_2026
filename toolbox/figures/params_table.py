"""Generates tables/params.tex.

Each cell is value(SE) with the bootstrap standard error (fitting/results/
bootstrap_{proton,carbon}/replicates.csv, B = 140 cluster-bootstrap refits).
Transformed parameters (carbon theta_bar -> n_ang, the A4 scale) get their SE
from transforming every replicate and taking the standard deviation (ddof=1,
as summary.csv), not from linear propagation.
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


SE_DIGITS = 2


def round_se(se):
    """Round se to SE_DIGITS significant digits; return (rounded, decimals)."""
    dec = -int(np.floor(np.log10(se))) + SE_DIGITS - 1
    r = round(se, dec)
    if r >= 10.0 ** (-dec + SE_DIGITS):      # 0.096 -> 0.10 gains a digit
        dec -= 1
        r = round(se, dec)
    return r, dec


def format_pair(pairs, key=None):
    """pairs = [(value, se), (value, se)], se None or 0 for a plain value.

    The exponent is shared (taken from the larger value, as before; sci
    notation when it falls outside [-2, 2]).  The last digit of each cell is
    set by that cell's own SE (2 significant digits), written as v(s) with s
    the SE in units of the last digit.
    """
    nz = [abs(v) for v, _ in pairs if v != 0]
    if not nz:
        return ['0'] * len(pairs)
    kmax = int(np.floor(np.log10(max(nz))))
    use_sci = kmax < -2 or kmax > 2
    k = kmax if use_sci else 0

    def cell(v, se):
        if v == 0:
            return '0'
        A = v / 10.0 ** k
        S = se / 10.0 ** k
        _, dec = round_se(S)
        if dec < 0:
            raise ValueError(f'{key}: SE coarser than units digit; rescale')
        t = f'{A:.{dec}f}'
        s_int = int(round(S * 10 ** dec))
        t = f'{t}({s_int})'
        if use_sci:
            return rf'${t}\times10^{{{k}}}$'
        return f'${t}$' if v < 0 else t

    return [cell(v, se) for v, se in pairs]


P = pd.read_csv(f'{ROOT}/fitting_params/params_proton.csv',
                index_col=0).loc['opt params'].astype(float)
C = pd.read_csv(f'{ROOT}/fitting_params/params_carbon.csv',
                index_col=0).loc['opt params'].astype(float)

p = {k: float(v) for k, v in P.items()}
c = {k: float(v) for k, v in C.items()}
c['n_ang'] = float(theta_to_n(c['theta_bar']))

THERMAL_SCALE = 1e15
for d in (p, c):
    d['A4_ms'] = d['A4'] * THERMAL_SCALE


def bootstrap_se(species, point):
    """SE per table key from the replicates, transforms applied per replicate."""
    rep = pd.read_csv(f'{ROOT}/fitting/results/bootstrap_{species}/replicates.csv')
    rep = rep[rep['ok'] & rep['converged']]
    if species == 'carbon':
        rep = rep.assign(n_ang=theta_to_n(rep['theta_bar']))
    rep = rep.assign(A4_ms=rep['A4'] * THERMAL_SCALE)
    se = {k: float(rep[k].std(ddof=1)) for k in point if k in rep}
    return se, len(rep)


p_se, B_p = bootstrap_se('proton', p)
c_se, B_c = bootstrap_se('carbon', c)
assert B_p == B_c == 140, (B_p, B_c)
# Sigma sits on its bound 0: no standard error (marked in the table instead)
p_se['Sigma'] = c_se['Sigma'] = 0.0
p['Sigma'] = c['Sigma'] = 0.0

ROWS = [
    ('Global', [

        (r'$E_{\mathrm{ref}}$', r'$\mathrm{MeV/u}$', 'E_th'),
        (r'$m$', '--', 'n'),
    ]),
    ('Cascade', [
        (r'$A_{\mathrm{cas}}$', r'$\mathrm{cm^{-3}\,primary^{-1}}$', 'A1'),
        (r'$\gamma_{\mathrm{cas}}$', '--', 'gamma1'),
        (r'$\Sigma^{\dagger}$', r'$\mathrm{cm^{-1}}$', 'Sigma'),
        (r'$n_{\mathrm{ang}}$', '--', 'n_ang'),
        (r'$d_{\mathrm{cas}}$', '--', 'd1'),
        (r'$a_{\mathrm{cas}}$', '--', 'a'),
        (r'$w_{\mathrm{cas}}$', '--', 'w_c'),
    ]),
    ('Evaporation', [
        (r'$A_{\mathrm{ev}}$', r'$\mathrm{cm^{-3}\,primary^{-1}}$', 'A2'),
        (r'$\gamma_{\mathrm{ev}}$', '--', 'gamma2'),
        (r'$d_{\mathrm{ev}}$', '--', 'd2'),
        (r'$\kappa_{\mathrm{ev}}$', r'$\mathrm{cm^{-1}}$', 'kappa_ev'),
    ]),

    ('Epithermal', [
        (r'$\kappa_{\mathrm{slow}}$', r'$\mathrm{cm^{-1}}$', 'kappa_slow'),
        (r'$A_{\mathrm{ep}}$', r'$\mathrm{cm^{-5}\,primary^{-1}}$', 'A3'),
        (r'$\gamma_{\mathrm{ep}}{}^{\ddagger}$', '--', 'gamma3'),
    ]),
    ('Thermal', [
        (r'$A_{\mathrm{th}}$', r'$\mathrm{cm^{-5}\,GeV^{-2}\,primary^{-1}}$', 'A4_ms'),
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
    r'\begin{table}[!ht]',
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
        ca, cb = format_pair([(p[key], p_se[key]), (c[key], c_se[key])], key)
        L.append(rf'{sym} & {unit} & {ca} & {cb} \\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

open(path, 'w').write('\n'.join(L) + '\n')
print(f'wrote {path}\n')
for sec, rows in ROWS:
    print(f'  {sec}')
    for sym, unit, key in rows:
        ca, cb = format_pair([(p[key], p_se[key]), (c[key], c_se[key])], key)
        print(f'    {key:10s} {ca:>26} {cb:>26}')
