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

Replaces the separable cascade with the coupled first-flight integral

```
phi(rho, z; P, Sigma_h, n_ang)
    = int_0^P dz' S(z') cos^n(theta) exp(-Sigma_h s) / (4 pi s^2)
```

with `s` the slant distance from source point to field point, tabulated by
`toolbox/diffusion_integrals/precompute_cascade.py`. Parameter mapping:
`(Sigma_t, sigma_cas) -> (Sigma_h, n_ang)`, so the vector keeps its length and
the same fitting code drives both options. Setting `g(theta) = delta(forward)`
recovers the Option 1 axial law, so that profile is the forward limit of this
kernel rather than a separate ingredient.

**Proton — `fitting_params/proton_coupled.csv`** (limited testing; carbon not done)

`Sigma_h = 0.0218` cm^-1 (L = 46 cm), `n_ang = 3.09`, `kappa_ev = 0.308`, with
`E_pk = 4 MeV` held. Fitted on a **single** primary energy (100 MeV) with `E_th`,
`n` and the whole slow-neutron sector held fixed, then scored on all 50.

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

---

## Performance

Volume-weighted bias (Eq. 10), all 50 primary energies, full `rho <= 5.5 cm`:

| | mean abs. dPhi | mean abs. dK | max abs. dPhi | outside +-40 % |
|---|---|---|---|---|
| **Option 1** carbon, `prefac1` | 7.2 % | 3.8 % | 9.1 % | 0/50 |
| **Option 1** proton, `prefac2` escalated | 4.6 % | 6.4 % | 33.0 % | 0/50 |
| **Option 2** proton, coupled cascade | 15.7 % | 19.1 % | 73.3 % | 2/50 |

Option 2's larger number is expected: it was fitted on one primary energy with
`E_th`/`n` frozen, so 49 of the 50 are out of sample and the low-energy end is
unconstrained (+73 % at 54 MeV). Its distinguishing property is that the bias is
**flat in radius** — 16.0 % at 3.5 cm against 15.7 % at 5.5 cm — where every
factorized variant degrades sharply outward (23.7 % -> 42.9 %).

Note that the fitting objective (unweighted least squares on raw fluence) is not
the reported metric (rho-weighted, z- and E-integrated bias). The two do not
always rank parameter sets the same way; several transport constants shift
noticeably depending on which is minimized.

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
python -m toolbox.diffusion_integrals.precompute_cascade     # ~4 min, Option 2 only
```

`evap_dim.dat` and `therm_dim_{proton,carbon}.dat` are tracked. The coupled
cascade table `cascade_dim.dat` is **not** — at 68 MB it would bloat the history,
and it regenerates from the command above.

The slow-neutron table is species-specific because its source is the cascade
axial profile, which depends on that species' `Sigma_t`.

### Two-group slow neutrons

The epithermal and thermal regimes share one two-group kernel. The two-group
result follows from the tabulated one-group kernel by

```
F_2(kappa_f, kappa_s) = [F_1(kappa_f) - F_1(kappa_s)] / (kappa_s^2 - kappa_f^2)
```

since `kappa_s^2 - kappa_f^2` carries no k dependence and the Hankel integral is
linear — so `kappa_f` costs no table dimension. Both kappas are held fixed
(0.150 and 0.128 cm^-1): in a 45 cm phantom the Dirichlet boundary supplies the
observed decay, leaving `kappa` weakly identifiable on its own.

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

- Option 2 is tested on protons only, and on a single primary energy. The natural
  next step is a 10-energy fit with `E_th`/`n` free, then carbon.
- Option 1's proton set requires renaming the fourth regime; "evaporation" with
  `E_pk = 500 MeV` is not defensible as written.
- Under Option 2 the slow-neutron table still uses the Option 1 `Sigma_t` as its
  cascade source shape. That regime contributes ~0 % of the energy fluence so
  results are unaffected, but the two are formally inconsistent.
- `n_ang` is held global. It should rise with neutron energy (higher-energy
  cascade neutrons are more forward-peaked); the equivalent factorized
  parameterization measured `Sigma_h ~ En^0.36`.
