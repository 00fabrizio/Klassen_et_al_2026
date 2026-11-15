"""Neutron fluence-to-kerma conversion coefficients for water.

Four pieces, in MeV and fGy m^2 (returned as pGy cm^2, a factor 10), joined by
log-log interpolation:

- 11 eV to 150 MeV: ICRU Report 63 (1999), Table C.3, water column
  (``icru63_table_c3_water.txt``, transcribed from the report).
- 0.0256 eV to 11 eV: ICRU 63 water points digitised from Shrestha et al. 2022
  (PMB 67 125019) Fig. 3 (``icru63_water_digitised.txt``, about +-3 % per
  point); Table C.3 starts at 11 eV. In the table's range the digitised points
  agree with it to 0.7 % mean, 3.5 % max.
- Above 150 MeV: Shrestha et al. 2022 Eq. 5 high-energy term, u E^2 + v.
- Below 0.0256 eV: Shrestha's a1 cosh(log10 E).

Shrestha's log-normal term (b1, mu1, sigma1 in their Table 1) is not used: the
printed water values do not reproduce their own Fig. 3. a1, u and v do.
Revision 1, task T20.
"""
import os

import numpy as np

A_1 = 4.65e-07
U = 2.377e-04
V = 7.402

_HERE = os.path.dirname(os.path.abspath(__file__))
_TABLE = np.loadtxt(os.path.join(_HERE, 'icru63_table_c3_water.txt'))
_DIGIT = np.loadtxt(os.path.join(_HERE, 'icru63_water_digitised.txt'))
_ICRU = np.vstack([_DIGIT[_DIGIT[:, 0] < _TABLE[0, 0]], _TABLE])
_LOG_E = np.log(_ICRU[:, 0])
_LOG_K = np.log(_ICRU[:, 1])
E_MIN_MEV = float(_ICRU[0, 0])
E_MAX_MEV = float(_ICRU[-1, 0])


def k_coeff_fGy_m2_from_MeV(En_MeV: np.ndarray) -> np.ndarray:
    shape = np.shape(En_MeV)
    E = np.clip(np.atleast_1d(np.asarray(En_MeV, dtype=float)), 1e-30, None)
    k = np.exp(np.interp(np.log(E), _LOG_E, _LOG_K))
    low = E < E_MIN_MEV
    high = E > E_MAX_MEV
    k[low] = A_1 * np.cosh(np.log10(E[low]))
    k[high] = U * E[high] ** 2 + V
    return k.reshape(shape)


def k_coeff_pGy_cm2_from_GeV(En_GeV: np.ndarray) -> np.ndarray:
    return 10.0 * k_coeff_fGy_m2_from_MeV(np.asarray(En_GeV, dtype=float) * 1e3)
