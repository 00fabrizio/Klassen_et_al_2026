# Settled decisions on figures and tables

A running record of what has been agreed **for good** about the shared artefacts
— the manuscript's figures and tables. Anything here is decided: apply it, do not
re-litigate it, and do not quietly deviate. Add an entry when a new decision is
reached; amend an entry only when the decision itself is revisited.

Generators live in `toolbox/figures/`. Captions and prose are the author's and
are never generated or edited — see CLAUDE.md.

---

## Table 2 — optimized model parameters (`tables/params.tex`)

*Settled 2026-08-07.*

**Number formatting, two rules, in order of precedence:**

1. **The last digit falls on the same decimal place for proton and carbon**, so
   the two columns are comparable digit for digit.
2. **At least 3 significant digits for each species**; more where holding rule 1
   requires it.

Where a row needs scientific notation, **both cells carry the factor and the
exponent is identical in both**, taken from the larger of the two values —
`0.638e-5` and `3.231e-5`, not `6.38e-6` and `3.23e-5`. Scientific notation is
used when the shared exponent falls outside `[-2, 2]`; otherwise plain decimals.

Implemented in `format_pair()` in `toolbox/figures/params_table.py`.

**Known consequence, accepted:** `A_cas` spans 113x between species, so rule 1
gives carbon six significant digits against the proton's three
(`0.0285` / `3.2178`). No representation avoids this while keeping the last digit
aligned; comparability was judged the more important property.

**On the choice of 3 as the floor.** No statistical uncertainty is available for
these parameters: the fits deliberately use no MC uncertainties as weights, so
`curve_fit`'s covariance is scaled to make reduced chi-squared 1 and would
describe model misspecification, not precision. What *is* measurable is
reproducibility — rounding every parameter to k digits and re-measuring Eq. 10:

| k | max shift in dPhi | max shift in dK |
|---|---|---|
| 2 | 2.59 pp | 2.58 pp |
| 3 | 0.23 pp | 0.27 pp |
| 4 | 0.035 pp | 0.035 pp |
| 5 | 0.005 pp | 0.006 pp |

At 3 digits a reader reproducing the model from the table lands ~0.2 pp from the
metrics quoted in the text, which are given to 0.1 pp. This was raised and 3 was
chosen anyway, for convention. Worth restating if the metrics are ever quoted to
more precision.

**Row order:** Global, Cascade, Evaporation, Epithermal, Thermal.
`kappa_slow` **leads the Epithermal block**, since it governs both slow regimes
and Epithermal is the first of them.

**`E_pk` is not listed.** It is held at 4 MeV for both species rather than
fitted, and this is a table of *optimized* parameters. Consequence to watch: its
value now appears nowhere in the manuscript — the symbol is defined in the
evaporation spectrum in `derivation.tex`, but the number 4 MeV is not stated. The
text needs to give it, or the model is not reproducible from the paper.

The same "held, not fitted" logic would arguably remove carbon's `Sigma = 0`, but
that value is a fit *outcome* that was then held, not an input, so it stays.

---

## Figure — spatial spectral fluence (`figures/spectral_fluence_comp.pdf`)

*Settled 2026-08-07.*

- **Both species at the same primary-energy index** — they then sit at the same
  relative position in their respective ranges. Currently index 32: proton
  165 MeV, carbon 315 MeV/u.
- **All four neutron energies identical for both species.** The evaporation row
  in particular, since `E_pk` is held at the same 4 MeV for both.
- **Rows chosen on CORE agreement, not the median** — median AM/MC over points
  above 50 % of that slice's MC maximum. The two diverge sharply: at 4.1 MeV the
  proton median is 0.99 while its core is 1.65, because the median averages a hot
  core against a thin halo, and it is the core that makes a panel read as too
  bright.
- Current rows: **25 meV, 1 eV, 2.2 MeV, 64 MeV**. 64 MeV rather than ~20 MeV
  because the proton cascade core is 1.5-1.7 below 40 MeV at *every* primary
  energy and only reaches 1.0 near 50-70 MeV; it is also the strongest panel of
  the four, since `E dPhi/dE` weights by energy.
- Layout: two species side by side, four neutron energies as rows,
  `MC | AM | |Delta|` as columns, rho mirrored about the beam axis, one colour
  norm shared across each row.

Resulting core ratios:

| row | 1H | 12C |
|---|---|---|
| 25 meV | 0.94 | 1.01 |
| 1 eV | 0.81 | 0.99 |
| 2.2 MeV | 1.39 | 0.97 |
| 64 MeV | 1.00 | 1.07 |

**Known and not fixable by selection:** the proton 2.2 MeV core stays near 1.39 —
the proton evaporation core is 1.38-1.45 across the whole band and every primary
energy. The proton 1 eV row at 0.81 reflects that carbon has had a targeted
epithermal refit and the proton has not.
