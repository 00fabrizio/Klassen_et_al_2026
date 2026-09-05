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

Number formatting follows two agreed rules -- see format_pair. The caption is the
author's and is preserved, never regenerated.
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


FLOOR = 3          # significant digits, minimum, for BOTH species


def format_pair(a, b, floor=FLOOR):
    """Format one row's two values under the agreed rules.

    Rule 1  both species end on the SAME decimal place, so the columns are
            directly comparable digit for digit.
    Rule 2  at least `floor` significant digits for each; more where the two
            species differ enough in magnitude that holding rule 1 demands it.

    Where the row needs scientific notation, BOTH cells carry the factor and the
    exponent is the SAME in both, chosen from the larger of the two.
    """
    nz = [v for v in (a, b) if v != 0]
    if not nz:
        return '0', '0'
    kmax = int(np.floor(np.log10(max(abs(v) for v in nz))))
    use_sci = kmax < -2 or kmax > 2
    k = kmax if use_sci else 0
    A, B = a / 10.0 ** k, b / 10.0 ** k

    # decimals such that every nonzero value carries at least `floor` digits
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
p['n_ang'] = p['n_ang']                                   # stored as exponent
c['n_ang'] = float(theta_to_n(c['theta_bar']))            # stored as mean angle

ROWS = [
    ('Global', [
        # E_ref, not E_th: 'th' is the thermal regime everywhere else in the
        # paper, and this is not a threshold -- at E_0 = E_ref the prefactor is
        # 1 - 1/e, i.e. a saturation scale. The CSV column keeps its name.
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
    # kappa_slow governs BOTH slow regimes, so it leads the first of them.
    # E_pk is not listed: it is held at 4 MeV for both species, not fitted, and
    # this table is of optimized parameters. It is defined in the evaporation
    # spectrum in the derivation; its VALUE now appears nowhere in the
    # manuscript, so the text must state it for the model to be reproducible.
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
    # The caption sits ABOVE the table, so the gap to the toprule is
    # \belowcaptionskip. iopjournal.cls honours it but sets no value, so it takes
    # the article default of 0pt and the caption lands 3 px off the rule. Scoped
    # to this float on purpose: a global setting would also add trailing space
    # under FIGURE captions, which sit below their graphic.
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
