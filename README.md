# Analytical model for secondary neutron fluence in particle therapy

Analytical model for the secondary neutron spectral fluence produced by internal
neutrons in active-scanning proton and carbon-ion radiotherapy, together with the
FLUKA Monte Carlo data it was fitted and verified against.

Companion code for Klassen et al., *Derivation and verification of an analytical
model for secondary neutron fluence in active-scanning proton and carbon-ion
radiotherapy*.

## Layout

```
model.py                  Option 1: four energy regimes x two prefactor variants
model_coupled.py          Option 2: same, with a coupled ballistic cascade regime
fitting_params/           fitted parameters, columns match the entry-point signature
data/                     raw FLUKA USRTRACK output (archival) + range tables
npy_data/                 parsed cache of data/  (not tracked; rebuild, see below)
toolbox/                  spatial/spectral factors, diffusion kernels, data loading
  diffusion_integrals/      precomputed kernel tables + their builders
  metrics/bias_curves.py    parallel, cached Eq. 10 bias curves
fit.ipynb                 fits the model parameters
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
grids.

## The model

The spectral energy fluence is factorized into a prefactor in the primary energy
`E0` and a sum over four energy regimes — cascade, evaporation, epithermal and
thermal — each a product of a spatial and a spectral factor:

```
Phi(z, rho, En; E0) = prefac(E0) * sum_i  A_i * spatial_i(z, rho; P_i) * spectrum_i(En)
```

with `P_i = gamma_i * xt(E0, En)` the production range of regime `i`.

### Prefactors

Two parametrizations are provided as separate entry points, because their free
parameter lists differ in both length and order and are consumed positionally by
`curve_fit`:

| entry point | prefactor | parameters |
|---|---|---|
| `spectral_energy_fluence_prefac1` | `(E0 / E_ref) ** xi` | 19, `xi` leading |
| `spectral_energy_fluence_prefac2` | `1 - exp(-(E0 / E_th) ** n)` | 20, `E_th`/`n` trailing |

`E_ref` is fixed at 120 MeV (proton) and 200 MeV/u (carbon).

`prefac2` is the "fraction of primaries that interact before stopping" law:
substituting the Bragg-Kleeman range `R ~ E0^p` into `1 - exp(-Sigma_inel R)`
yields exactly this form and predicts `n ~ p`. Measured from the xt tables,
`p = 1.77` (proton) and `1.71` (carbon), against a fitted proton `n = 2.2` — the
right ballpark. The scale does not follow (the fitted `E_th` implies a mean free
path ~15x shorter than the real inelastic one), so the prefactor is best read as
a correction absorbing the `E0` dependence that the geometric scaling
`P_i = gamma_i R(E0)` fails to supply, not as a standalone yield law.

**The two species occupy different regions of the same curve.** Protons span
`E0/E_th = 0.53 - 2.14` and cross the whole knee (local slope 1.9 -> 0.06), so the
curvature is load-bearing and a power law misses by 73 %. Carbon spans
`0.05 - 0.21` and sits in the foot, where `1 - exp(-x) ~ x` is a power law to
within 1 %. This is why the two species end up on different prefactor choices.

Columns in `fitting_params/*.csv` are ordered to match the corresponding
signature, so they can be unpacked positionally:

```python
import pandas as pd
from model import Model

p = pd.read_csv('fitting_params/carbon_prefac1_twogroup.csv', index_col=0)
M = Model(species='carbon', XT=XT, iEp=iEp, iEn=iEn)
phi = M.spectral_energy_fluence_prefac1((Z, R, En, EP), *p.loc['opt params'])
```

---

## Two model options

### Option 1 — factorized cascade (`model.py`)

The published structure. The cascade spatial factor is separable — a disk of
radius 1 cm smeared by a Gaussian of width `sigma_cas`, times an axial
buildup/attenuation profile:

```
cascade_lateral(rho; sigma_cas)  x  cascade(z; Sigma_t, P)
```

**Carbon — `prefac1`, `fitting_params/carbon_prefac1_twogroup.csv`**

`xi = 0.387`. Fitted on 10 primary energies over the full 5.5 cm radial range.
A matched `prefac2` refit scores the same to within 0.5 pp while spending an
extra parameter, and leaves `E_th` degenerate — hence the power law.

**Proton — `prefac2`, `fitting_params/proton_prefac2_twogroup_escalated.csv`**

Fitted on 10 primary energies with `E_pk` and `kappa_ev` released. They do not
stay physical: `E_pk` runs to 500 MeV (against its bound, so unidentified) and
`kappa_ev` falls to 0.033 cm^-1 — a 30.6 cm diffusion length. The evaporation
regime has stopped being evaporation and has become a broad, hard halo. Carbon
does the same thing more mildly (`E_pk = 140 MeV`). The numbers are excellent;
**the labels are not defensible and the regime must be renamed if this option is
used.** For protons in particular a real 4 MeV evaporation peak is visible in the
MC spectrum, which a 500 MeV peak contradicts.

Holding `E_pk` and `kappa_ev` at physical values instead gives only 42.9 % on the
full radial range, and refitting does not rescue it — which is what motivated
Option 2.

### Option 2 — coupled ballistic cascade (`model_coupled.py`)

Replaces the separable cascade with the coupled first-flight integral over the
source volume `V`,

```
F(rho, z) = int_V d3x' g(theta) exp(-Sigma_h s) / (4 pi s^2)
```

with `s = |x - x'|` and `cos(theta) = (z - z')/s`, tabulated by
`toolbox/diffusion_integrals/precompute_cascade.py`. Parameter mapping:
`(Sigma_t, sigma_cas) -> (Sigma_h, n_ang)`, so the vector keeps its length and
the same fitting code drives both options. Setting `g(theta) = delta(forward)`
recovers the Option 1 axial law, so that profile is the forward limit of this
kernel rather than a separate ingredient.

#### Spherical form — why there is no singularity

In cylindrical coordinates about the beam axis the integrand carries a `1/s^2`
singularity wherever the field point lies inside the source. It is integrable,
but awkwardly: evaluating a line source on a radial grid and convolving with the
disk in Hankel space is exact in principle yet inherits that singularity in the
intermediate profile, and read ~12 % low inside the source on a 0.18 cm grid.

Putting the origin at the **field** point removes it outright. With
`x' = x - s*u` the volume element is `d3x' = s^2 ds dOmega`, and the `s^2`
cancels the `1/s^2` exactly:

```
F = (1/4pi) int dOmega g(theta) int ds exp(-Sigma_h s)
  = (1/4pi Sigma_h) int dOmega cos^n(theta) [e^{-Sigma_h s1} - e^{-Sigma_h s2}]
```

Every direction contributes the attenuated **chord** the ray cuts through the
source, between entry `s1` and exit `s2`. The `s` integral is closed form, only a
2D angular quadrature remains, and nothing diverges — a nearby source element
subtends a correspondingly small solid angle. Ray-cylinder roots are

```
s_pm = [ rho cos(phi) +- sqrt(R^2 - rho^2 sin^2(phi)) ] / sin(theta)
```

intersected with the axial window `[(z-P)/mu, z/mu]`, and empty when the ray
misses the cylinder. At `Sigma_h = 0` the bracket is `0/0`; the removable limit
is the unattenuated chord `s2 - s1`.

The angular quadrature is **adapted per field point** — the source subtends a
narrow cone once the field point is far downstream, and a narrow azimuthal wedge
once `rho > R`, so a fixed grid puts nearly all its nodes where the integrand
vanishes. A fixed grid was 2.6x off at `z = 30`.

#### The angular lobe is parametrized by its mean angle, not by `n`

`g(theta) = cos^n(theta)` has width `~ n^{-1/2}`, so `d(width)/dn -> 0`: `n` is
badly conditioned as a fit parameter, and an optimizer that wants a forward peak
runs it to infinity with nothing to rail against. The table axis is therefore the
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

`cascade_coupled(P, Sigma_h, n_ang, z, r)` still takes the exponent and converts
internally, so **existing parameter files read unchanged**; `cascade_coupled_theta`
is the entry point for new fits.

**Proton — `fitting_params/proton_coupled_single.csv`** (production set)

Fitted on a **single** primary energy (100 MeV) with `E_pk = 4 MeV` held, then
scored on all 50. Entry point `spectral_energy_fluence_single`, 19 parameters:

| | | | |
|---|---|---|---|
| `A1` | 0.0284561 | `A2` | 0.00159186 |
| `gamma1` | 0.743339 | `gamma2` | 0.904449 |
| `Sigma_h` | 0.0217669 | `d2` | -0.190574 |
| `n_ang` | 3.08683 | `kappa_ev` | 0.308278 |
| `d1` | -0.317853 | `Epk` | 0.004 *(held)* |
| `a` | 0.515959 | `A3` | 6.37809e-06 |
| `w_c` | 0.349239 | `gamma3` | 0.949367 |
| `E_th` | 102.174 | `A4` | 426.906 |
| `n` | 2.19805 | `gamma4` | 1.08848 |
| | | `kappa` | 0.185082 |

Every value is physical: `Sigma_h = 0.0218` cm^-1 is a 45.9 cm removal length,
`kappa_ev = 0.308` a 3.2 cm evaporation diffusion length, `kappa = 0.185` a
migration length `M = sqrt(2)/kappa = 7.6 cm`, and `n_ang = 3.09` a mean emission
angle of **36.5 deg** — a real forward lobe with a surviving wide-angle halo.

**Which grid.** This set was fitted against the cascade table as it stood then:
angular axis uniform in `n` (`linspace(0,8,17)` plus `9,10,12,14,17,20`), uniform
160-node `mu` quadrature. It is now evaluated against the current table —
`theta_bar` 6-60 deg in 55 nodes, clustered 200-node `mu` — which shifts the
kernel by **0.06-0.37 % mean, 2.6-4.2 % max**. The parameters have **not** been
refitted since; they are the pre-recast optimum read through the current table.

**Carbon — `fitting_params/carbon_coupled_theta_10E.csv`**

Fitted on **10 primary energies** (100-425 MeV/u, 805,000 points, 17 free) with
the saturation prefactor, `Sigma_h` held at 0 and `E_pk` held at 4 MeV. Entry
point `spectral_energy_fluence_single_theta`, 19 parameters — the file carries
the two held values so it unpacks positionally like the others.

| | | | |
|---|---|---|---|
| `A1` | 3.21777 | `A2` | 0.00968274 |
| `gamma1` | 0.903585 | `gamma2` | 1.29622 |
| `Sigma_h` | 0 *(held)* | `d2` | -0.642482 |
| `theta_bar` | 16.3077 deg | `kappa_ev` | 0.180540 |
| `d1` | -0.0505519 | `Epk` | 0.004 *(held)* |
| `a` | 0.952823 | `A3` | 3.23128e-05 |
| `w_c` | 0.195026 | `gamma3` | 1.7803 |
| `E_th` | 194.889 | `A4` | 2498.63 |
| `n` | 0.963881 | `gamma4` | 1.78558 |
| | | `kappa` | 0.126472 |

**`E_th` at 200 rather than 2000 is what made the transport constants
physical.** The earlier carbon prefactor sat at `E_th` = 2003 MeV/u, which puts
the whole species at `E0/E_th` = 0.05-0.21 — the foot of the saturation curve,
where `1 - exp(-x) ~ x` and the law is indistinguishable from a weak power law
(the fitted `n` = 0.379 spans a factor 1.55 over the entire range). Starting at
200 moves the range to 0.50-2.12, across the knee, the same region the protons
occupy. The fit then keeps it (194.9), and with the prefactor carrying the `E0`
dependence the diffusion constants stop having to:

| | single energy, `E_th` = 2003 | 10 energies, `E_th` = 195 |
|---|---|---|
| `kappa` | 0.00847 (`M` = 167 cm) | **0.1265** (`M` = 11.2 cm) |
| `kappa_ev` | 0.0745 (`L_ev` = 13.5 cm) | **0.1805** (`L_ev` = 5.5 cm) |

Both are now sane for a 45 cm phantom. `theta_bar` moves by **0.02 deg** between
the single-energy fit and the 10-energy fit, so the angular lobe is a genuine
transport constant rather than a per-energy adjustment.

**`A3` and `gamma3` are fitted separately, against the 1 eV band.** In the joint
fit `gamma3` railed at 9 with `A3` = 4.08e-06, and the 1 eV field came out 4-5x
low. The cause is the objective: the fit minimizes unweighted least squares on
*energy* fluence, where the epithermal band carries ~0 % of the total, so the
pair is effectively unconstrained and drifts. Solving for `A3` over the full
spectrum even returns a NEGATIVE value, because the epithermal tail reaches the
100 keV region where the model already over-predicts ~1.9x.

Refitting the two against the 1 eV bin +-10 neutron-energy bins (~0.31-3.2 eV,
narrow enough that epithermal dominates and 100 keV cannot contaminate it), on
10 primary energies and a coarse `z`/`rho` grid with everything else held:

| `gamma3` | band cost | |
|---|---|---|
| **1.780** | **4.70e-05** | fitted |
| 9.0 | 1.49e-04 | 3.2x worse (the railed value) |
| 0.668 | 2.91e-04 | 6.2x worse (the below-1 eV slow fit's value) |

`gamma3` within 10 % of the minimum spans only 1.59-1.99, so it is well
identified once fitted where it carries signal. `A3` enters linearly, so the
optimum at each `gamma3` is closed form and the scan is global, not local.

1 eV median AM/MC goes 0.275 / 0.220 / 0.199 -> **0.783 / 0.950 / 1.040** at
249 / 299 / 348 MeV/u, and the AM - MC map turns from a uniform deficit into a
roughly balanced residual. Note that neither of the two values one might reach
for by hand was close: 9.0 looked shape-correct only because a source that long
has saturated to the phantom, which flattens the ratio for the wrong reason.

A single `gamma3` still cannot hold all three energies — the trend
0.78 / 0.95 / 1.04 climbs with `E0`, since the source is tied to a range that
grows with it. 299 is nearly exact; 249 runs ~20 % low.

*Superseded single-energy exploration (300 MeV/u), kept for the record.* All four
variants scored on the *same* (current) table, so the costs are comparable:

| variant | cost | `theta_bar` | `n` | `E_pk` | `kappa_ev` | `L_ev` | `A1` | `Sigma_h` |
|---|---|---|---|---|---|---|---|---|
| old `n` axis, `E_pk` free | 0.27976 | 17.34 deg *(railed)* | 20.0 | 499.9 MeV | 0.380 | 2.6 cm | 4.72 | 2e-4 floor |
| old `n` axis, `E_pk` = 4 MeV | 0.28854 | 17.34 deg *(railed)* | 20.0 | 4 MeV | 0.082 | 12.2 cm | 5.67 | 2e-4 floor |
| new `theta_bar`, `E_pk` free | **0.24076** | 13.28 deg | 35.4 | 486.0 MeV | 0.126 | 7.9 cm | 7.48 | 2e-4 floor |
| new `theta_bar`, `E_pk` = 4 MeV | 0.28575 | 16.28 deg | 22.9 | 4 MeV | 0.074 | 13.5 cm | 6.26 | 2e-4 floor |
| ... with `Sigma_h` free to 0 | 0.28538 | 16.29 deg | 22.9 | 4 MeV | 0.075 | 13.4 cm | 6.25 | **0** |

*proton production set for reference: `theta_bar` 36.5 deg, `E_pk` 4 MeV,
`kappa_ev` 0.308 (`L_ev` 3.2 cm), `A1` 0.0285.*

The recast does what it was for: **the angular parameter no longer rails**, in
either `E_pk` condition. With `E_pk` free it settles at an interior 13.3 deg
where the old axis pinned it at the grid edge, and the cost falls 14 %. With
`E_pk` held the cost edge is only ~1 %, because the old rail at `n = 20` already
sat close to the optimum. Both `E_pk` = 4 MeV starts converge to the same point
to 6 significant figures from very different initial values, so it is the global
optimum of this parametrization, not a local one.

Two findings from that exploration carried into the production fit, and
**neither is about the angular parametrization**:

- `Sigma_h` goes to **zero**. Earlier fits bounded it at `2e-4` because the
  `Sigma_h = 0` slab of the table was NaN; with that fixed and the bound dropped
  to a true zero, the fit returns `3e-41` and a marginally *better* cost. So at
  300 MeV/u the coupled kernel degenerates to pure geometry — `1/(4 pi s^2)`
  dilution with a `cos^23` lobe and no attenuation whatever over 45 cm. The
  proton, by contrast, sits at a well-identified `Sigma_h = 0.0218` (45.9 cm)
  with a much wider 36.5 deg lobe.
- The fourth regime insists on being a **broad halo** whichever knob is left
  open. Free `E_pk` buys a hard-spectrum halo (486 MeV peak); holding `E_pk` at
  the physical 4 MeV keeps the spectrum right but drives `kappa_ev` to 0.074, a
  13.5 cm diffusion length against the proton's 3.2 cm. The old `n` axis showed
  the same thing (0.082, 12.2 cm), so this predates the recast.

`A1` is also ~110x the proton amplitude.

#### Why it exists

In the MC the fast (>20 MeV) share of the energy fluence is **~88 % at every
radius** from 1 to 5.5 cm, and its mean energy is flat (62 -> 61 MeV at
E0 = 146 MeV). Those neutrons are ballistic, not moderated — a diffusing
population would soften, since n-p elastic scattering halves the energy per
collision. A Gaussian lateral factor is ~1e-6 by 5 cm, so the factorized cascade
under-predicts the fast field there by ~300x, and the fit compensates by
distorting the evaporation regime.

The factorized form also forces one z-profile at every radius, while the MC peak
migrates **deeper** with radius (z = 14 -> 18 cm between rho = 1 and 5.5 cm at
E0 = 170 MeV) — a neutron reaching large rho was emitted obliquely and therefore
arrives past its emission point. On a single (E0, En) slice the factorized fit
gives 75 % relative RMS against the coupled kernel's 20 %, and the z-profile
correlation degrades 0.99 -> 0.49 with radius where the coupled kernel holds
0.98 throughout.

Under Option 2, `kappa_ev` settles at 0.308 (from 0.325) even when free alongside
`Sigma_h` — with a real halo in the cascade, evaporation no longer tries to
become one.

#### On the transport constants

Three quantities that all look like attenuation coefficients and are not the same:

| | value | meaning |
|---|---|---|
| `Sigma_t` (Option 1) | 0.346 | effective — the axial factor has no geometric term, so this absorbs the 1/s^2 dilution and the angular spread as well |
| `sigma_tot` | ~0.067 | true total cross section in water at these energies |
| `Sigma_h` (Option 2) | 0.0218 | **removal** constant — elastic scattering off oxygen leaves a fast neutron in the band, so in-scattering largely cancels removal |

Forcing `Sigma_h = Sigma_t` costs a factor four in residual (84 % vs 20 %), so
the distinction is measurable and not a fitting artifact.

#### Table accuracy

Against the direct integral, over the field for the proton set:

| `P` | mean | max |
|---|---|---|
| 5 cm | 2.4 % | 21.7 % |
| 15 cm | 0.9 % | 7.4 % |
| 30 cm | 0.5 % | 7.4 % |

Those maxima are **confined to the dim tail** and the means are what matter.
Stratifying by brightness (`theta_bar` = 16.3 deg, `P` = 15 cm, the carbon case):

| `F / F_max` | points | mean | max |
|---|---|---|---|
| 1e-4 - 1e-3 | 7 | 17.0 % | 39.3 % |
| 1e-3 - 1e-2 | 15 | 4.8 % | 16.3 % |
| 1e-2 - 1e-1 | 189 | 0.13 % | 3.4 % |
| 1e-1 - 1 | 70 | 0.17 % | 0.66 % |

So the table is accurate wherever the field actually is, and the large *relative*
errors sit four decades below peak where they carry no weight in a least-squares
fit or in Eq. 10. Quote the stratified figures, not the bare max.

Isolating axes by snapping one coordinate at a time to its nearest node, the
residual is `xi` and `P_hat` — never `Sigma_h` or `theta_bar`. It concentrates at
the **kink just past the source end** and in the far upstream tail, where a
narrow lobe makes the kernel vary violently with `z` (at `theta_bar` = 16 deg a
point at `z` = 2.8, `rho` = 2.5 needs ~42 deg emission, far outside the lobe).
It decays with range, so it is worst at short ranges — i.e. at **low primary
energies**, which is also where the reported `dK` is worst. Refining
`P_HAT_GRID_C` and `XI_GRID_C` near the source end is the obvious next move and
has not been done.

---

## Performance

Volume-weighted bias (Eq. 10), all 50 primary energies, full `rho <= 5.5 cm`:

| | mean abs. dPhi | mean abs. dK | max abs. dPhi | outside +-40 % |
|---|---|---|---|---|
| **Option 1** carbon, `prefac1` | 7.2 % | 3.8 % | 9.1 % | 0/50 |
| **Option 1** proton, `prefac2` escalated | 4.6 % | 6.4 % | 33.0 % | 0/50 |
| **Option 2** proton, `proton_coupled_single.csv` | **10.3 %** | **2.8 %** | 29.9 % | 0/50 |
| **Option 2** carbon, `carbon_coupled_theta_10E.csv` | **8.8 %** | **8.1 %** | 13.2 % | 0/50 |

Carbon's Option 2 numbers are the flattest of the four in `dPhi` (max 13.2 %
against the proton's 29.9 %), but both curves are smooth and almost entirely
**negative**, -3 % to -13 %. That is a structural under-prediction, not scatter.

**The cascade is what drives it, not the slow regimes.** Splitting the `dPhi`
numerator by band at 299 MeV/u:

| band | share of MC | contribution to `dPhi` |
|---|---|---|
| thermal | 16.9 % | -0.78 pp |
| epithermal | 4.5 % | -0.90 pp |
| evaporation | 17.6 % | -1.94 pp |
| **cascade** | **61.0 %** | **-8.76 pp** |

For `dK` it is starker: the cascade is 89 % of kerma and carries -9.5 of the
-10.9. Refitting the epithermal pair (above) was worth ~2.5 pp of `dPhi` and
essentially nothing on `dK`, which is the ceiling on what the slow regimes can
buy. Band-by-band AM/MC at 299 MeV/u, after that refit:

| band | rho=0.1 | rho=1.5 | rho=3.5 | rho=5.5 |
|---|---|---|---|---|
| 25 meV | 0.98 | 1.00 | 0.95 | 0.96 |
| 1 eV | ~0.95 (refitted) | | | |
| 100 keV | 1.96 | 1.70 | 1.58 | 1.16 |
| 1 MeV | 1.04 | 1.30 | 1.19 | 1.05 |
| **30 MeV** | 0.78 | 0.75 | 0.83 | **0.56** |

Thermal and epithermal are now good. 30 MeV still falls off too fast laterally
(0.78 on axis, 0.56 at `rho` = 5.5), so the halo is short at large radius even
with the coupled kernel — that is where the remaining bias lives.

The fitted and unfitted energies lie on the same smooth curve with no visible
gap, so this is systematic, not overfitting.

Option 2's proton set, measured on the current table: `dPhi` mean 10.26 %,
median 5.25 %, max 29.86 % at 70 MeV; `dK` mean 2.76 %, median 2.10 %, max
20.32 % at 54 MeV. These are the same to the quoted precision as the values
measured before the table was rebuilt on the `theta_bar` axis — the kernel moved
0.06-0.37 %, the integrated metric did not move.

That it was fitted on a **single** primary energy with `E_th`/`n` frozen makes
49 of the 50 out of sample. Its distinguishing property is that the bias is
**flat in radius**, where every factorized variant degrades sharply outward
(23.7 % -> 42.9 %).

`dK` is worst at the low-energy end (+20 % at 54 MeV, +10 % at 60 MeV, then ~4 %
everywhere else). That is also where the cascade table is least accurate — the
`P_hat`/`xi` grids near the source end, see *Table accuracy* — so the two may be
related, but that has not been demonstrated.

**Eq. 10 uses differential fluence.** The npy `mc` holds `E dPhi/dE`, so
`phi = mc / en_low` and the integration weights are `dE/en` and `dE kc/en`.
Using `dE` and `dE kc` carries a spurious factor of `E`; it changes the band
shares badly (evaporation 8.6 % -> 42.1 % of kerma). `toolbox/metrics/
bias_curves.py` still contains the uncorrected weights and is Option 1-only.

Note that the fitting objective (unweighted least squares on **energy** fluence,
~88 % cascade) is not the reported metric (`rho`-weighted, `z`- and
`E`-integrated bias on **particle** fluence, ~38 % slow, and on kerma, ~0 %
slow). The two do not rank parameter sets the same way — several transport
constants shift noticeably depending on which is minimized, and sets with
*fewer* refitted parameters have repeatedly scored better on the reported
metric. This mismatch is deliberate on the fitting side and worth keeping in
mind when comparing rows above.

---

## Fitting

`fit.ipynb` reproduces the `prefac2` parameters for protons and carbon and writes
them to `fitting_params/`. Each of its two cells is self-contained: it loads the
cache from `npy_data/`, builds the production-range table, fits with
`scipy.optimize.curve_fit` (trust-region reflective), and reports volume-weighted
fluence and kerma deviations for the fitted energies. Initial values are taken
from the parameter file it overwrites, so the fit starts from the previous
converged solution. MC uncertainties are deliberately not used as weights — they
destabilize the fit.

## Diffusion kernels

The evaporation and epithermal/thermal spatial factors are Hankel-transform
integrals without closed form. They are precomputed on dimensionless grids and
interpolated at evaluation time.

```
python -m toolbox.diffusion_integrals.precompute_kernels     # ~6 min, both species
python -m toolbox.diffusion_integrals.precompute_slow        # ~2 min, uniform source
python -m toolbox.diffusion_integrals.precompute_slow --sq   # ~2 min, single kappa
python -m toolbox.diffusion_integrals.precompute_cascade     # ~20 min, Option 2 only
```

`evap_dim.dat` and `therm_dim_{proton,carbon}.dat` are tracked. The coupled
cascade table `cascade_dim.dat` is **not** — at 306 MB it would bloat the
history, and it regenerates from the command above. Its axes are
`(P_hat, Sigma_h, theta_bar, xi, rho_hat)` = `(31, 16, 55, 46, 61)`.

Note that the tables are part of the model: rebuilding one changes results at
unchanged parameters. `toolbox/metrics/bias_curves.py` therefore keys its cache
on the `.dat` files' size and mtime as well as on the parameter values.

### Slow neutrons

**One source volume, different transport.** The slow-neutron table
(`precompute_slow.py`) puts the epithermal and thermal regimes over the *same*
uniform cylinder used by the cascade and evaporation regimes, rather than over
the cascade axial profile. The regimes then differ only in how neutrons are
transported out of a common source volume, and the table becomes
**species-independent** — it no longer references `Sigma_t`, so one file serves
both. Fitting `gamma` absorbs the difference: a uniform column must be longer to
cover the same axial extent as a profile with an exponential tail, so the fitted
`gamma` comes out larger.

**Two groups, one diffusion length.** The two-group result follows from the
tabulated one-group kernel by

```
F_2(kappa_f, kappa_s) = [F_1(kappa_f) - F_1(kappa_s)] / (kappa_s^2 - kappa_f^2)
```

exactly, including the Dirichlet boundaries — the Helmholtz operator is diagonal
in the `sin(m pi z / L)` basis, so the partial fraction runs in a `k`-independent
denominator mode by mode. So `kappa_f` costs no table dimension.

Collapsing `kappa_f = kappa_s = kappa` eliminates the fast group and leaves

```
(grad^2 - kappa^2)^2 phi = const * S
```

whose propagator is `1/(k^2+kappa^2)^2 = -(1/2 kappa) d/dkappa [1/(k^2+kappa^2)]`.
In free space this is `e^{-kappa r} / (8 pi kappa)` — **the `1/r` cancels**, which
is exactly why the two-group kernel is flat near the source and broad enough to
match the MC where a single Yukawa is not. The migration length is
`M = sqrt(2)/kappa`. `slow_sq_dim.dat` tabulates it directly, so it is one lookup
rather than a difference of two: no removable singularity, no cancellation. This
collapse costs nothing measurable and removes a parameter.

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

- **Carbon's residual bias is the cascade at large radius** — 30 MeV runs at 0.56
  of MC by `rho` = 5.5 against 0.78 on axis, and the cascade carries -8.8 pp of
  the -12 pp `dPhi` and -9.5 of -10.9 on `dK`. The halo is still short. This is
  the only lever left that can move these numbers materially.
- **The fitting objective does not match the reported metric**, which is what
  left `A3`/`gamma3` unconstrained until they were refitted by hand against the
  1 eV band. Reweighting the objective by `1/E` would align the two and remove
  the need for per-regime patches; not done.
- Option 2's proton set is fitted on a single primary energy. The natural next
  step is a 10-energy fit with `E_th`/`n` free, as was done for carbon.
- Option 1's proton set requires renaming the fourth regime; "evaporation" with
  `E_pk = 500 MeV` is not defensible as written.
- `theta_bar` is held global. It should shrink with neutron energy (higher-energy
  cascade neutrons are more forward-peaked); the equivalent factorized
  parameterization measured `Sigma_h ~ En^0.36`.
- Cascade table accuracy near the source end is limited by the `P_hat` and `xi`
  grids, not by the angular or `Sigma_h` axes — see *Table accuracy* above.
- The Eq. 10 volume weighting uses `rho` rather than `pi(rho_j^2 - rho_{j-1}^2)
  dz`. The innermost bin is 2x off; the effect on the reported `dPhi` is 0.19 pp.
  It should be corrected before the metric goes in the paper.
