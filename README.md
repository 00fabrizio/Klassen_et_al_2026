# Analytical model for secondary neutron fluence in particle therapy

Analytical model for the secondary neutron spectral fluence produced by internal
neutrons in active-scanning proton and carbon-ion radiotherapy, together with the
FLUKA Monte Carlo data it was fitted and verified against.

Companion code for Klassen et al., *Derivation and verification of an analytical
model for secondary neutron fluence in active-scanning proton and carbon-ion
radiotherapy*.

## Layout

```
model.py                  the model: four energy regimes x two prefactor variants
fitting_params/           fitted parameters, {species}_prefac{1,2}.csv
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

`model.py` factorizes the spectral energy fluence into a global prefactor in the
primary energy and a sum over four energy regimes — cascade, evaporation,
epithermal and thermal — each a product of a spatial and a spectral factor.

Two prefactor parametrizations are provided as separate entry points, because
their free-parameter lists differ in both length and order and are consumed
positionally by `curve_fit`:

| entry point | prefactor | parameters |
|---|---|---|
| `spectral_energy_fluence_prefac1` | `(EP / E_ref) ** xi` | 19, `xi` leading |
| `spectral_energy_fluence_prefac2` | `1 - exp(-(EP / E_th) ** n)` | 20, `E_th`/`n` trailing |

`prefac1` reproduces the parametrization published in the paper; `prefac2`
replaces the power law with a saturation curve. Columns in
`fitting_params/{species}_prefac{1,2}.csv` are ordered to match the
corresponding signature, so they can be unpacked positionally:

```python
import pandas as pd
from model import Model

p = pd.read_csv('fitting_params/proton_prefac2.csv', index_col=0)
M = Model(species='proton', XT=XT, iEp=iEp, iEn=iEn)
phi = M.spectral_energy_fluence_prefac2((Z, R, En, EP), *p.loc['opt params'])
```

## Fitting

`fit.ipynb` reproduces the `prefac2` parameters for protons and carbon and
writes them to `fitting_params/`. Each of its two cells is self-contained: it
loads the cache from `npy_data/`, builds the production-range table, fits with
`scipy.optimize.curve_fit` (trust-region reflective, `Epk` held fixed), and
reports volume-weighted fluence and kerma deviations for the fitted energies.
Initial values are taken from the parameter file it overwrites, so the fit
starts from the previous converged solution.

Verification and figure notebooks are kept outside the repository; the model,
the parameters and the data here are what is needed to reproduce them.

## Diffusion kernels

The evaporation and epithermal/thermal spatial factors are Hankel-transform
integrals without closed form. They are precomputed on a dimensionless
`(P, kappa, z, rho)` grid and interpolated at evaluation time. The tables
`toolbox/diffusion_integrals/{evap,therm}_dim.dat` are tracked, but can be
regenerated from scratch to verify them:

```
python toolbox/diffusion_integrals/precompute_kernels.py    # ~6 min
```

## Primary range tables

`data/primary_range/{proton,carbon}Solver/` holds precomputed primary-energy
loss curves, read by `toolbox/production_range.py`. The scripts that generate
them and the range tables `data/{species}_range.csv` are in
`toolbox/primary_range/`.

## Data provenance

`data/{proton,carbon}/<energy>/New_ring_*_22_tab.lis` is unmodified FLUKA
USRTRACK output: neutron spectral fluence in annular bins, 50 primary energies
per species, 45 axial bins, 55 radial bins to 5.5 cm, 250 energy bins from
1e-13 to 1 GeV. `toolbox/dataloader.py` is the only reader; it flips the axial
axis, weights by bin energy to give energy fluence, and converts the relative
FLUKA errors to absolute.
