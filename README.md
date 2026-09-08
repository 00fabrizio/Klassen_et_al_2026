# Analytical model for secondary neutron fluence in particle therapy

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22650366.svg)](https://doi.org/10.5281/zenodo.22650366)

Code and data for **Klassen et al. (2026)**, an analytical model of the spectral
energy fluence of secondary neutrons produced by proton and carbon-ion beams in a
water phantom, fitted to FLUKA Monte Carlo data.

The model gives the neutron spectral energy fluence
`E_n dPhi/dE_n (z, rho, E_n; E_0)` as a sum over four regimes — cascade, evaporation, epithermal and
thermal, each modelled as a spatial factor multiplied by an energy spectrum.

## Install

```bash
git clone https://github.com/00fabrizio/Klassen_et_al_2026.git
cd Klassen_et_al_2026
pip install -r requirements.txt
```

The cascade kernel table takes about 20 minutes

```bash
python -m toolbox.diffusion_integrals.precompute_cascade
```

The evaporation and slow-neutron tables are in the repository already. Run
everything from the repository root, with it on `PYTHONPATH`.

## Quick start

[`sample_usage.ipynb`](sample_usage.ipynb) evaluates the model on a grid you
choose and plots the result in energy and in space. The short version:

```python
import numpy as np, pandas as pd
from toolbox.production_range import build_xt_table
from model import Model

species, E0 = 'proton', 164.84
z, rho = np.linspace(0.5, 45.0, 90), np.linspace(0.0, 5.5, 56)
En = np.logspace(-13, 0, 250)                      # GeV
nz, nr, nE = z.size, rho.size, En.size

p = pd.read_csv(f'fitting_params/params_{species}.csv',
                index_col=0).loc['opt params'].astype(float)
m = Model(species=species,
          XT=build_xt_table(species, np.array([E0]), En),
          iEp=np.repeat(np.arange(1), nz * nr * nE),
          iEn=np.tile(np.arange(nE), nz * nr))
EP, Z, R, E = [a.ravel() for a in np.meshgrid([E0], z, rho, En, indexing='ij')]
phi = m.spectral_energy_fluence((Z, R, E, EP), *p).reshape(nz, nr, nE)
```

`E0` must be one of the energies with a precomputed depth-energy table under
`data/primary_range/`; the notebook lists them. Carbon uses
`spectral_energy_fluence_theta`, which takes the cascade lobe as a mean emission
angle rather than an exponent; the two entry points are otherwise identical and
take their 19 parameters in the same order as the CSV columns.

Energies are **GeV** inside the model and **MeV or MeV/u** in the parameter
files. `*_mc.npy` holds `E_n dPhi/dE_n`, not `dPhi/dE_n`.

## Layout

```
model.py                        the model: four regimes, prefactor, two entry points
fitting_params/                 fitted parameters, one CSV per species
sample_usage.ipynb              worked example

toolbox/
  spectral_funcs.py             energy spectra of the four regimes
  production_range.py           production range xt(E0, En)
  dataloader.py                 reader for the raw FLUKA .lis files
  build_npy_cache.py            data/ -> npy_data/
  diffusion_integrals/          kernel tables and their interpolators
  primary_range/                depth-energy solvers for both species
  metrics/                      fluence-to-kerma coefficients, bias curves
  figures/                      generators for the manuscript figures and table

data/
  proton/, carbon/              raw FLUKA output, 50 primary energies each
  primary_range/                precomputed depth-energy tables per energy
  {proton,carbon}_range.csv     primary range vs energy
  {proton,carbon}_energies.txt  the simulated primary energies
```

`npy_data/` is a derived cache built from `data/` with
`python toolbox/build_npy_cache.py`. It is not tracked: each array exceeds
GitHub's file limit and it is reproducible bit for bit from the `.lis` files.

## Reproducing the manuscript

Every figure and the parameter table are generated from `fitting_params/`, so
they cannot drift from the fits:

```bash
FIGURES_DIR=figures python toolbox/figures/energy_spectra.py
FIGURES_DIR=figures python toolbox/figures/spectral_fluence_comp.py
FIGURES_DIR=figures python toolbox/figures/integrated_comp.py
FIGURES_DIR=figures python toolbox/figures/volume_weighted_difference.py
FIGURES_DIR=figures python toolbox/figures/conceptual_spatial.py
TABLES_DIR=tables  python toolbox/figures/params_table.py
```

These need `npy_data/`. `toolbox/metrics/bias_curves.py` computes the
volume-weighted bias of total fluence and kerma against the MC data.

## Data

`data/proton/` and `data/carbon/` hold the raw FLUKA output, one directory per
primary energy, 50 energies per species. Each `.lis` file is a neutron fluence
tally on a 45 x 55 x 250 grid in depth, radius and neutron energy, in a
cylindrical water phantom of 45 cm length and 5.5 cm scoring radius.

## Citation

If you use this code or data, please cite the paper and the archived release.
See `CITATION.cff`.

## Licence

MIT — see `LICENSE`.
