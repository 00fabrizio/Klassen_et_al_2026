"""Reader for the raw FLUKA .lis output."""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def _read_tab(species, energy):
    """Parse one USRTRACK `_tab.lis`: per-bin flux and statistical error.

    Returns z, r, en_low, en_upp, flux, pct with flux and pct shaped
    (num_z, num_r, num_e) in file order (z not yet flipped). Columns of the file:
    E_low, E_high [GeV], dPhi/dE [cm^-2 GeV^-1 per primary], relative error of
    the mean in PERCENT (0 where the bin is empty).
    """
    num_e = 250
    num_r = 55
    num_z = 45

    r = np.round(np.linspace(0.1, 5.5, num_r), 3)
    z = np.round(np.linspace(0, 44.0, num_z), 3) + 1

    filename = f'data/{species}/{energy}/New_ring_{species}_22_tab.lis'
    # Header lines start with '#'; what is left is 4 columns, 250 rows per
    # detector. Detector index = j * num_z + i (ring j, slab i); the last 25
    # detectors in the file are not part of the (z, rho) grid and are dropped.
    tab = np.loadtxt(filename, comments='#')
    tab = tab.reshape(-1, num_e, 4)[:num_r * num_z]
    flux = tab[:, :, 2].reshape(num_r, num_z, num_e).transpose(1, 0, 2).copy()
    pct = tab[:, :, 3].reshape(num_r, num_z, num_e).transpose(1, 0, 2).copy()
    en_low = tab[0, :, 0].copy()
    en_upp = tab[0, :, 1].copy()
    return z, r, en_low, en_upp, flux, pct


def load_relerr(species, energy):
    """Relative statistical error of every bin as a FRACTION, shape (z, rho, En).

    z is flipped like `load_radial_data`. Empty bins are NaN. FLUKA writes the
    error in percent, so it is divided by 100 here.
    """
    _, _, _, _, flux, pct = _read_tab(species, energy)
    rel = np.flip(pct, axis=0) / 100.0
    rel[np.flip(flux, axis=0) == 0] = np.nan
    return rel


def load_radial_data(species, energies):
    MC = []
    ERR = []
    for energy in energies:
        z, r, en_low, en_upp, mc, err = _read_tab(species, energy)

        mc = np.flip(mc, axis=0)*en_low
        err = np.flip(err, axis=0)
        # FLUKA's error column is a relative error in PERCENT. This used to
        # multiply by the percent value directly, giving 100x the absolute
        # error (T45, R2-m5). Divide by 100 to get the absolute error of E dPhi/dE.
        err = err/100.0*mc
        MC.append(mc)
        ERR.append(err)
    MC = np.stack(MC, axis=0)
    ERR = np.stack(ERR, axis=0)
    return z, r, en_low, en_upp, MC, ERR
