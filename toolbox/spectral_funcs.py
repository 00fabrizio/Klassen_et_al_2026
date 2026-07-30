import numpy as np
from scipy.special import erf

def cnorm(x):
    """
    Standard cumulative normal function, defined via the error function
    """
    return (1 + erf(x/np.sqrt(2))) / 2

def cascade_spectrum(en, A, alpha, t_c, w_c):
    # A_t = F(en=thresh)
    return 2*A*(en/t_c)**(alpha+1) * cnorm(np.log(t_c/en)/w_c)

def evaporation_spectrum(en, A, alpha, Epk):
    """
    Energy fluence evaporation spectrum.

    Parameters
    ----------
    en : array_like
        Neutron energy (GeV)
    A : float
        Peak amplitude (phi(Epk) = A)
    alpha : float
        Low-energy power-law parameter (p = alpha + 1)
    Epk : float
        Evaporation peak energy (GeV)

    Returns
    -------
    phi_E : array_like
        Energy fluence spectrum
    """
    p = alpha + 1.0
    en_safe = np.maximum(en, 1e-300)
    log_val = p * (np.log(en_safe / Epk) + 1.0 - en_safe / Epk)
    return A * np.exp(np.clip(log_val, -745, 709))

def epithermal_spectrum(en):
    """
    Epithermal function, constant in lethargy units. Cutoff guarantees smooth transition
    into neighbouring energy regimes
    """
    low = 1e-10
    upp = 1e-04
    return cnorm(np.log(upp/en))*(1-cnorm(np.log(low/en)))

def thermal_spectrum(en):
    """
    Thermal spectrum, follows Maxwell-Boltzmann distribution
    """
    T = 2.585e-11
    return 1e+15* en**2 * np.exp(-en / T)