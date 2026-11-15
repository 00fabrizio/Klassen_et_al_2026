"""Neutron fluence-to-kerma conversion coefficients for water.

Three pieces, in MeV and fGy m^2 (returned as pGy cm^2, a factor 10):

- E within the ICRU Report 63 table: log-log interpolation of the 59 ICRU 63
  water points, digitised from Shrestha et al. 2022 (PMB 67 125019) Fig. 3
  (``icru63_water_digitised.txt``, about +-3 % per point). The last point
  (148.3 MeV as digitised) is ICRU 63's 150 MeV point.
- E above the last point: Shrestha et al. 2022 Eq. 5 high-energy term,
  u E^2 + v.
- E below the first point (2.56e-8 MeV): Shrestha's a1 cosh(log10 E).

Shrestha's log-normal term (b1, mu1, sigma1 in their Table 1) is not used: the
printed water values do not reproduce their own Fig. 3 (125 % mean deviation
from ICRU 63). a1, u and v do. Revision 1, task T20.
"""
import os

import numpy as np

A_1 = 4.65e-07
U = 2.377e-04
V = 7.402

_ICRU = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'icru63_water_digitised.txt'))
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
