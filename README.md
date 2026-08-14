# Analytical model for secondary neutron fluence in particle therapy

Analytical model for the secondary neutron spectral fluence produced by internal
neutrons in active-scanning proton and carbon-ion radiotherapy, together with the
FLUKA Monte Carlo data it was fitted and verified against.

Companion code for Klassen et al., *Derivation and verification of an analytical
model for secondary neutron fluence in active-scanning proton and carbon-ion
radiotherapy*.

## Layout

```
model.py                  the model: four energy regimes x a saturation prefactor
fitting_params/           fitted parameters, columns match the entry-point signature
data/                     raw FLUKA USRTRACK output (archival) + range tables
npy_data/                 parsed cache of data/  (not tracked; rebuild, see below)
tables/                   generated LaTeX tables
toolbox/                  spatial/spectral factors, diffusion kernels, data loading
  diffusion_integrals/      precomputed kernel tables + their builders
  metrics/bias_curves.py    parallel, cached Eq. 10 bias curves
  figures/                  manuscript figure and table generators
```

## Getting started

Requires `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`.

The parsed data cache is not stored in the repository — its arrays exceed
GitHub's file size limit and it is reproducible bit-exactly from `data/`.
Build it once after cloning:

```
python toolbox/build_npy_cache.py        # ~2 min
```

This reads the 100 `New_ring_*_22_tab.lis` files under `data/` and writes
`npy_data/{species}_{mc,err}.npy` plus the shared `z`, `rho`, `en_low`, `en_upp`
grids. Then build the kernel tables (see *Diffusion kernels* below).

## The model

The spectral energy fluence is a prefactor in the primary energy `E0` times a
sum over four energy regimes — cascade, evaporation, epithermal and thermal —
each a product of a spatial and a spectral factor:

```
Phi(z, rho, En; E0) = prefac(E0) * sum_i  A_i * spatial_i(z, rho; P_i) * spectrum_i(En)
```

with `P_i = gamma_i * xt(E0, En)` the production range of regime `i`.

Two entry points differ only in how the cascade angular lobe is parametrized.
Both take 19 parameters in the same order, so one fitting routine drives both:

| entry point | angular slot | used by |
|---|---|---|
| `spectral_energy_fluence` | `n_ang`, the exponent | `proton_coupled_single.csv` |
| `spectral_energy_fluence_theta` | `theta_bar`, degrees | `carbon_coupled_theta_10E.csv` |

```python
import pandas as pd
from model import Model

p = pd.read_csv('fitting_params/carbon_coupled_theta_10E.csv', index_col=0)
M = Model(species='carbon', XT=XT, iEp=iEp, iEn=iEn)
phi = M.spectral_energy_fluence_theta((Z, R, En, EP), *p.loc['opt params'])
```

### Prefactor

```
prefac(E0) = 1 - exp(-(E0 / E_th) ** n)
```

the "fraction of primaries that interact before stopping" law: substituting the
Bragg-Kleeman range `R ~ E0^p` into `1 - exp(-Sigma_inel R)` yields exactly this
form and predicts `n ~ p`. Measured from the xt tables, `p = 1.77` (proton) and
`1.71` (carbon), against a fitted proton `n = 2.20` — the right ballpark. The
scale does not follow (the fitted `E_th` implies a mean free path much shorter
than the real inelastic one), so the prefactor is best read as a correction
absorbing the `E0` dependence that the geometric scaling `P_i = gamma_i R(E0)`
fails to supply, rather than as a standalone yield law.

**Both species must sit across the knee for this to be identifiable.** An
earlier carbon fit returned `E_th = 2003 MeV/u`, which puts the whole species at
`E0/E_th = 0.05-0.21` — the foot of the curve, where `1 - exp(-x) ~ x` and the
law is indistinguishable from a weak power law. At `E_th = 195` the range is
`0.50-2.12`, the same region the protons occupy, and the curvature carries
information. This is not cosmetic: see *Carbon* below for what it fixed.

### Cascade

The cascade spatial factor is the coupled first-flight integral over the source
volume `V`,

```
F^cas(rho, z) = int_V d3x' g(theta) exp(-Sigma s) / (4 pi s^2)
```

with `s = |x - x'|` and `cos(theta) = (z - z')/s`, tabulated by
`toolbox/diffusion_integrals/precompute_cascade.py`.

#### Why it is not separable

In the MC the fast (>20 MeV) share of the energy fluence is **~88 % at every
radius** from 1 to 5.5 cm, and its mean energy is flat (62 -> 61 MeV at
E0 = 146 MeV). Those neutrons are ballistic, not moderated — a diffusing
population would soften, since n-p elastic scattering halves the energy per
collision. A Gaussian lateral factor is ~1e-6 by 5 cm, so a factorized cascade
under-predicts the fast field there by ~300x, and the fit compensates by
distorting the evaporation regime.

A separable form also forces one z-profile at every radius, while the MC peak
migrates **deeper** with radius (z = 14 -> 18 cm between rho = 1 and 5.5 cm at
E0 = 170 MeV) — a neutron reaching large rho was emitted obliquely and therefore
arrives past its emission point. On a single (E0, En) slice a factorized fit
gives 75 % relative RMS against the coupled kernel's 20 %, and the z-profile
correlation degrades 0.99 -> 0.49 with radius where the coupled kernel holds
0.98 throughout.

#### Spherical form — why there is no singularity

In cylindrical coordinates about the beam axis the integrand carries a `1/s^2`
singularity wherever the field point lies inside the source. Putting the origin
at the **field** point removes it outright. With `x' = x - s*u` the volume
element is `d3x' = s^2 ds dOmega`, and the `s^2` cancels the `1/s^2` exactly:

```
F = (1/4pi) int dOmega g(theta) int ds exp(-Sigma s)
  = (1/4pi Sigma) int dOmega cos^n(theta) [e^{-Sigma s1} - e^{-Sigma s2}]
```

Every direction contributes the attenuated **chord** the ray cuts through the
source, between entry `s1` and exit `s2`. The `s` integral is closed form, only a
2D angular quadrature remains, and nothing diverges — a nearby source element
subtends a correspondingly small solid angle. Ray-cylinder roots are

```
s_pm = [ rho cos(phi) +- sqrt(R^2 - rho^2 sin^2(phi)) ] / sin(theta)
```

intersected with the axial window `[(z-P)/mu, z/mu]`, and empty when the ray
misses the cylinder. At `Sigma = 0` the bracket is `0/0`; the removable limit is
the unattenuated chord `s2 - s1`.

The angular quadrature is **adapted per field point** — the source subtends a
narrow cone once the field point is far downstream, and a narrow azimuthal wedge
once `rho > R`, so a fixed grid puts nearly all its nodes where the integrand
vanishes. A fixed grid was 2.6x off at `z = 30`.

#### The angular lobe is parametrized by its mean angle

`g(theta) = cos^n(theta)` has width `~ n^{-1/2}`, so `d(width)/dn -> 0`: `n` is
badly conditioned as a fit parameter, and an optimizer that wants a forward peak
runs it to infinity with nothing to rail against. Carbon did exactly that,
pinning at the grid edge in every variant. The table axis is therefore the
**mean emission angle** `theta_bar`, via the exact relation

```
<cos theta> = (n+1)/(n+2)    ->    n = (2 cos theta_bar - 1) / (1 - cos theta_bar)
```

`theta_bar = 60 deg` is `n = 0`, the flattest lobe the family allows, and
`theta_bar -> 0` is the forward delta — so the whole family maps onto a
**bounded** interval and a forward-peaked solution reports a finite edge value.
Measured shape sensitivity (RMS over the field, amplitude divided out):

| | n = 3 | n = 20 | n = 80 |
|---|---|---|---|
| per `+1` in `n` | 11.9 % | 4.3 % | 1.5 % |
| per `-2 deg` in `theta_bar` | 7.1 % | 20.7 % | 41.1 % |

In `n` the gradient decays toward zero at the narrow end — flat gradient, large
steps, runaway. In `theta_bar` it grows, so the optimizer always feels the wall.
With the recast, carbon's angular parameter stops railing and lands at an
interior 16.3 deg.

`Sigma` is an **effective removal** constant, not `sigma_tot`: a fast neutron
scattering elastically off oxygen stays in the band, so removal is largely
cancelled by in-scattering and the fitted value (0.0218 cm^-1 for protons) is far
below the total-cross-section value (~0.067). Forcing the two equal costs a
factor four in residual, so the distinction is measurable and not a fitting
artifact.

### Slow neutrons

**One source volume, different transport.** The epithermal and thermal regimes
sit over the *same* uniform cylinder the cascade and evaporation regimes use.
The regimes then differ only in how neutrons are transported out of a common
source volume, and the table is **species-independent** — it carries no `Sigma`,
so one file serves both. Fitting `gamma` absorbs the difference: a uniform column
must be longer to cover the same axial extent as a profile with an exponential
tail, so the fitted `gamma` comes out larger.

**Two groups, one diffusion length.** The two-group result follows from the
one-group kernel by

```
F_2(kappa_f, kappa_s) = [F_1(kappa_f) - F_1(kappa_s)] / (kappa_s^2 - kappa_f^2)
```

exactly, including the Dirichlet boundaries — the Helmholtz operator is diagonal
in the `sin(m pi z / L)` basis, so the partial fraction runs in a `k`-independent
denominator mode by mode. Collapsing `kappa_f = kappa_s = kappa_slow` eliminates
the fast group and leaves

```
(grad^2 - kappa_slow^2)^2 phi = const * S
```

whose propagator is `1/(k^2+kappa^2)^2 = -(1/2 kappa) d/dkappa [1/(k^2+kappa^2)]`.
In free space this is `e^{-kappa r} / (8 pi kappa)` — **the `1/r` cancels**, which
is exactly why the kernel is flat near the source and broad enough to match the
MC where a single Yukawa is not. The migration length is `sqrt(2)/kappa_slow`.
`slow_sq_dim.dat` tabulates this directly, so it is one lookup rather than a
difference of two: no removable singularity, no cancellation. The collapse costs
nothing measurable and removes a parameter.

---

## Fitted parameters

Generated into `tables/params.tex` by `toolbox/figures/params_table.py`.

### Proton — `fitting_params/proton_coupled_single.csv`

Fitted on a **single** primary energy (100 MeV) with `E_pk = 4 MeV` held, then
scored on all 50.

| | | | |
|---|---|---|---|
| `A1` | 0.0284561 | `A2` | 0.00159186 |
| `gamma1` | 0.743339 | `gamma2` | 0.904449 |
| `Sigma` | 0.0217669 | `d2` | -0.190574 |
| `n_ang` | 3.08683 | `kappa_ev` | 0.308278 |
| `d1` | -0.317853 | `Epk` | 0.004 *(held)* |
| `a` | 0.515959 | `A3` | 6.37809e-06 |
| `w_c` | 0.349239 | `gamma3` | 0.949367 |
| `E_th` | 102.174 | `A4` | 426.906 |
| `n` | 2.19805 | `gamma4` | 1.08848 |
| | | `kappa_slow` | 0.185082 |

Every value is physical: `Sigma = 0.0218` cm^-1 is a 45.9 cm removal length,
`kappa_ev = 0.308` a 3.2 cm evaporation diffusion length, `kappa_slow = 0.185` a
migration length of 7.6 cm, and `n_ang = 3.09` a mean emission angle of
**36.5 deg** — a real forward lobe with a surviving wide-angle halo.

### Carbon — `fitting_params/carbon_coupled_theta_10E.csv`

Fitted on **10 primary energies** (100-425 MeV/u, 805,000 points, 17 free) with
`Sigma` held at 0 and `E_pk` held at 4 MeV.

| | | | |
|---|---|---|---|
| `A1` | 3.21777 | `A2` | 0.00968274 |
| `gamma1` | 0.903585 | `gamma2` | 1.29622 |
| `Sigma` | 0 *(held)* | `d2` | -0.642482 |
| `theta_bar` | 16.3077 deg | `kappa_ev` | 0.180540 |
| `d1` | -0.0505519 | `Epk` | 0.004 *(held)* |
| `a` | 0.952823 | `A3` | 3.23128e-05 |
| `w_c` | 0.195026 | `gamma3` | 1.7803 |
| `E_th` | 194.889 | `A4` | 2498.63 |
| `n` | 0.963881 | `gamma4` | 1.78558 |
| | | `kappa_slow` | 0.126472 |

**`Sigma` goes to zero**, and this is a physical statement rather than a grid
artifact: earlier fits bounded it at `2e-4` because the `Sigma = 0` slab of the
table was NaN; with that fixed and the bound dropped to a true zero, the fit
returns `3e-41` and a marginally *better* cost. So carbon's cascade is governed
by geometric dilution and the emission lobe alone, with no attenuation over a
45 cm phantom — the opposite limit of the same kernel from the proton's.

**Moving `E_th` from 2003 to 195 MeV/u is what made the transport constants
physical.** With the prefactor able to carry the `E0` dependence, the diffusion
constants stopped having to:

| | single energy, `E_th` = 2003 | 10 energies, `E_th` = 195 |
|---|---|---|
| `kappa_slow` | 0.00847 (`M` = 167 cm) | **0.1265** (`M` = 11.2 cm) |
| `kappa_ev` | 0.0745 (`L_ev` = 13.5 cm) | **0.1805** (`L_ev` = 5.5 cm) |

`theta_bar` moves by **0.02 deg** between the single-energy and 10-energy fits,
so the angular lobe is a genuine transport constant rather than a per-energy
adjustment.

**`A3` and `gamma3` are fitted separately, against the 1 eV band.** In the joint
fit `gamma3` railed at 9 with `A3 = 4.08e-06`, and the 1 eV field came out 4-5x
low. The cause is the objective: it minimizes unweighted least squares on
*energy* fluence, where the epithermal band carries ~0 % of the total, so the
pair is effectively unconstrained and drifts. Solving for `A3` over the full
spectrum even returns a NEGATIVE value, because the epithermal tail reaches the
100 keV region where the model already over-predicts ~1.9x. Refitting the two
against the 1 eV bin +-10 neutron-energy bins (~0.31-3.2 eV, narrow enough that
epithermal dominates):

| `gamma3` | band cost | |
|---|---|---|
| **1.780** | **4.70e-05** | fitted |
| 9.0 | 1.49e-04 | 3.2x worse (the railed value) |
| 0.668 | 2.91e-04 | 6.2x worse (the below-1 eV slow fit's value) |

`gamma3` within 10 % of the minimum spans only 1.59-1.99, so it is well
identified once fitted where it carries signal. Neither value reachable by
inspection was close: 9.0 looks shape-correct only because a source that long has
saturated to the phantom, which flattens the ratio for the wrong reason.

---

## Performance

Volume-weighted bias (Eq. 10), all 50 primary energies, full `rho <= 5.5 cm`,
from `python -m toolbox.metrics.bias_curves`:

| | mean abs. dPhi | mean abs. dK | max abs. dPhi | outside +-40 % |
|---|---|---|---|---|
| proton | 10.3 % | 2.8 % | 29.9 % | 0/50 |
| carbon | 8.8 % | 8.1 % | 13.2 % | 0/50 |

The proton set was fitted on **one** primary energy with `E_th`/`n` frozen, so 49
of the 50 are out of sample; its distinguishing property is that the bias is flat
in radius, where every factorized variant degrades sharply outward.

Carbon's curves are smooth and almost entirely **negative**, -3 % to -13 %. That
is a structural under-prediction, not scatter, and **the cascade drives it, not
the slow regimes**. Splitting the `dPhi` numerator by band at 299 MeV/u:

| band | share of MC | contribution to `dPhi` |
|---|---|---|
| thermal | 16.9 % | -0.78 pp |
| epithermal | 4.5 % | -0.90 pp |
| evaporation | 17.6 % | -1.94 pp |
| **cascade** | **61.0 %** | **-8.76 pp** |

For `dK` it is starker: the cascade is 89 % of kerma and carries -9.5 of the
-10.9. Refitting the epithermal pair was worth ~2.5 pp of `dPhi` and essentially
nothing on `dK` — that is the ceiling on what the slow regimes can buy. What
remains is 30 MeV running at 0.56 of MC by `rho` = 5.5 against 0.78 on axis: the
halo is short at large radius.

**Eq. 10 uses differential fluence.** The npy `mc` holds `E dPhi/dE`, so
`phi = mc / en_low` and the integration weights are `dE/en` and `dE kc/en`. Using
`dE` and `dE kc` carries a spurious factor of `E` and changes the band shares
badly (evaporation 8.6 % -> 42.1 % of kerma).

Note that the fitting objective (unweighted least squares on **energy** fluence,
~88 % cascade) is not the reported metric (`rho`-weighted, `z`- and
`E`-integrated bias on **particle** fluence, ~13 % slow, and on kerma, ~0 %
slow). The two do not rank parameter sets the same way — sets with *fewer*
refitted parameters have repeatedly scored better on the reported metric. This
mismatch is deliberate on the fitting side and worth keeping in mind.

---

## Diffusion kernels

The evaporation and slow-neutron spatial factors are Hankel-transform integrals
without closed form; the cascade factor is a 2D angular quadrature. All are
precomputed on dimensionless grids and interpolated at evaluation time.

```
python -m toolbox.diffusion_integrals.precompute_kernels     # ~3 min, evaporation
python -m toolbox.diffusion_integrals.precompute_slow        # ~2 min, slow neutrons
python -m toolbox.diffusion_integrals.precompute_cascade     # ~20 min, cascade
```

`evap_dim.dat` (9.9 MB) is tracked. `slow_sq_dim.dat` (19.8 MB) and
`cascade_dim.dat` (306 MB) are **not** — they regenerate from the commands
above. The cascade table's axes are
`(P_hat, Sigma, theta_bar, xi, rho_hat)` = `(31, 16, 55, 46, 61)`.

The tables are part of the model: rebuilding one changes results at unchanged
parameters. `toolbox/metrics/bias_curves.py` therefore keys its cache on the
`.dat` files' size and mtime as well as on the parameter values.

### Table accuracy

Against the direct integral, over the field for the proton set:

| `P` | mean | max |
|---|---|---|
| 5 cm | 2.4 % | 21.7 % |
| 15 cm | 0.9 % | 7.4 % |
| 30 cm | 0.5 % | 7.4 % |

Those maxima are **confined to the dim tail** and the means are what matter.
Stratifying by brightness (`theta_bar` = 16.3 deg, `P` = 15 cm):

| `F / F_max` | points | mean | max |
|---|---|---|---|
| 1e-4 - 1e-3 | 7 | 17.0 % | 39.3 % |
| 1e-3 - 1e-2 | 15 | 4.8 % | 16.3 % |
| 1e-2 - 1e-1 | 189 | 0.13 % | 3.4 % |
| 1e-1 - 1 | 70 | 0.17 % | 0.66 % |

So the table is accurate wherever the field actually is, and the large *relative*
errors sit four decades below peak where they carry no weight. Quote the
stratified figures, not the bare max.

Isolating axes by snapping one coordinate at a time to its nearest node, the
residual is `xi` and `P_hat` — never `Sigma` or `theta_bar`. It concentrates at
the kink just past the source end and in the far upstream tail, where a narrow
lobe makes the kernel vary violently with `z`. It decays with range, so it is
worst at short ranges — i.e. at **low primary energies**, which is also where the
reported `dK` is worst.

## Manuscript figures and tables

```
python toolbox/figures/spectral_fluence_comp.py    # -> FIGURES_DIR or ./figures
python toolbox/figures/params_table.py             # -> TABLES_DIR  or ./tables
```

Both compute model values from `fitting_params/`, so they cannot drift from the
fits. Row and primary-energy selection for the figure is documented in its
docstring; the short version is that rows are chosen on **core** agreement
(median AM/MC above 50 % of a slice's MC maximum), not the median over the whole
map, because the median averages a hot core against a thin halo and it is the
core that makes a panel read as too bright.

## Primary range tables

`data/primary_range/{proton,carbon}Solver/` holds precomputed primary-energy loss
curves, read by `toolbox/production_range.py`. The scripts that generate them and
the range tables `data/{species}_range.csv` are in `toolbox/primary_range/`.

## Data provenance

`data/{proton,carbon}/<energy>/New_ring_*_22_tab.lis` is unmodified FLUKA
USRTRACK output: neutron spectral fluence in annular bins, 50 primary energies
per species, 45 axial bins, 55 radial bins to 5.5 cm, 250 energy bins from
1e-13 to 1 GeV. `toolbox/dataloader.py` is the only reader; it flips the axial
axis, weights by bin energy to give energy fluence, and converts the relative
FLUKA errors to absolute.

## Open items

- **There is no tracked fitting driver.** `fit.ipynb` fitted the superseded
  factorized model and was removed with it. The protocols for both production
  sets are described above and in the git history, but nothing in the repository
  reproduces them end to end.
- **Carbon's residual bias is the cascade at large radius** — 30 MeV runs at 0.56
  of MC by `rho` = 5.5 against 0.78 on axis, and the cascade carries -8.8 pp of
  the -12 pp `dPhi` and -9.5 of -10.9 on `dK`. The halo is still short. This is
  the only lever left that can move these numbers materially.
- **The fitting objective does not match the reported metric**, which is what left
  `A3`/`gamma3` unconstrained until they were refitted by hand against the 1 eV
  band. Reweighting the objective by `1/E` would align the two and remove the
  need for per-regime patches.
- The proton set is fitted on a single primary energy. The natural next step is a
  10-energy fit with `E_th`/`n` free, as was done for carbon, and the same
  targeted epithermal refit — its 1 eV field is ~0.8 of MC where carbon's is 1.0.
- `theta_bar` is held global. It should shrink with neutron energy (higher-energy
  cascade neutrons are more forward-peaked); an equivalent factorized
  parameterization measured `Sigma ~ En^0.36`.
- Cascade table accuracy near the source end is limited by the `P_hat` and `xi`
  grids, not by the angular or `Sigma` axes — see *Table accuracy*.
- The Eq. 10 volume weighting uses `rho` rather than `pi(rho_j^2 - rho_{j-1}^2)
  dz`. The innermost bin is 2x off; the effect on the reported `dPhi` is 0.19 pp.
