#!/usr/bin/env python3
"""
build_npy_cache.py

Rebuild the npy_data/ cache from the raw FLUKA USRTRACK .lis files in data/.

The .lis files are the archival record; npy_data/ is a bit-exact,
fast-loading cache of exactly what toolbox.dataloader.load_radial_data
returns for every available primary energy.

Writes, per species:
    npy_data/{species}_mc.npy        (nE, nz, nr, nEn) float64
    npy_data/{species}_err.npy       (nE, nz, nr, nEn) float64
    npy_data/{species}_energies.npy  (nE,)             float64

and the shared grids:
    npy_data/z.npy, rho.npy, en_low.npy, en_upp.npy

Arrays are streamed one energy at a time via open_memmap, so peak memory
stays at a single energy slab (~25 MB) rather than the full ~236 MB array.
Output is staged to .tmp files and atomically moved into place.

Run from the repository root:
    python toolbox/build_npy_cache.py
"""
import os
import sys

import numpy as np
from numpy.lib.format import open_memmap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from toolbox.dataloader import load_radial_data  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
NPY = os.path.join(ROOT, "npy_data")

SPECIES = ["proton", "carbon"]


def energy_folders(species):
    """Folder names for one species, ascending by primary energy."""
    d = os.path.join(DATA, species)
    return sorted(
        (f for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))),
        key=float,
    )


def build(species):
    folders = energy_folders(species)
    nE = len(folders)
    print(f"\n{species}: {nE} energies ({folders[0]} .. {folders[-1]})")

    # first energy fixes the grid shapes
    z, r, en_low, en_upp, MC, ERR = load_radial_data(species, [folders[0]])
    nz, nr, nEn = MC[0].shape
    print(f"  slab shape (nz, nr, nEn) = ({nz}, {nr}, {nEn})")

    mc_tmp = os.path.join(NPY, f"{species}_mc.npy.tmp")
    err_tmp = os.path.join(NPY, f"{species}_err.npy.tmp")

    mc_out = open_memmap(mc_tmp, mode="w+", dtype=np.float64,
                         shape=(nE, nz, nr, nEn))
    err_out = open_memmap(err_tmp, mode="w+", dtype=np.float64,
                          shape=(nE, nz, nr, nEn))

    mc_out[0] = MC[0]
    err_out[0] = ERR[0]

    for i, folder in enumerate(folders[1:], start=1):
        _, _, el, eu, MC_i, ERR_i = load_radial_data(species, [folder])
        if not (np.array_equal(el, en_low) and np.array_equal(eu, en_upp)):
            raise ValueError(f"{species}/{folder}: energy grid differs")
        if MC_i[0].shape != (nz, nr, nEn):
            raise ValueError(
                f"{species}/{folder}: shape {MC_i[0].shape} != {(nz, nr, nEn)}"
            )
        mc_out[i] = MC_i[0]
        err_out[i] = ERR_i[0]
        print(f"  [{i + 1:2d}/{nE}] {folder}")

    mc_out.flush()
    err_out.flush()
    del mc_out, err_out

    os.replace(mc_tmp, os.path.join(NPY, f"{species}_mc.npy"))
    os.replace(err_tmp, os.path.join(NPY, f"{species}_err.npy"))

    energies = np.array([float(f) for f in folders], dtype=np.float64)
    np.save(os.path.join(NPY, f"{species}_energies.npy"), energies)

    return z, r, en_low, en_upp


def main():
    os.makedirs(NPY, exist_ok=True)

    grids = {}
    for species in SPECIES:
        grids[species] = build(species)

    # shared grids must agree across species
    ref_species = SPECIES[0]
    z, r, en_low, en_upp = grids[ref_species]
    for species in SPECIES[1:]:
        for name, a, b in zip(("z", "rho", "en_low", "en_upp"),
                              grids[ref_species], grids[species]):
            if not np.array_equal(a, b):
                raise ValueError(f"{name} differs between species")

    np.save(os.path.join(NPY, "z.npy"), z)
    np.save(os.path.join(NPY, "rho.npy"), r)
    np.save(os.path.join(NPY, "en_low.npy"), en_low)
    np.save(os.path.join(NPY, "en_upp.npy"), en_upp)

    print("\nGrids written:")
    print(f"  z      {z.shape}  {z.min()} .. {z.max()} cm")
    print(f"  rho    {r.shape}  {r.min()} .. {r.max()} cm")
    print(f"  en_low {en_low.shape}  {en_low.min():.3g} .. {en_low.max():.3g} GeV")
    print("Done.")


if __name__ == "__main__":
    main()
