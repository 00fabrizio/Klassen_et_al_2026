"""Neutron fluence-to-kerma conversion coefficients."""
import numpy as np


def k_coeff_pGy_cm2_from_GeV(En_GeV: np.ndarray) -> np.ndarray:
    En_MeV = np.asarray(En_GeV, dtype=float) * 1e3

    a_1 = 4.65e-07
    b_1 = 43.30
    sigma_1 = 2.705
    mu1 = 1.571
    u = 2.377e-04
    v = 7.402
    E_thresh = 23.0

    En_MeV = np.clip(En_MeV, 1e-30, None)

    part_1 = a_1 * np.cosh(np.log10(En_MeV))
    part_2 = (
        b_1
        * np.heaviside(E_thresh - En_MeV, 0.5)
        * np.exp(-(np.log(En_MeV) - mu1) ** 2 / (2 * sigma_1**2))
        / np.sqrt(2 * np.pi * sigma_1**2)
    )
    part_3 = np.heaviside(En_MeV - E_thresh, 0.5) * (u * En_MeV**2 + v)

    return 10.0 * (part_1 + part_2 + part_3)
