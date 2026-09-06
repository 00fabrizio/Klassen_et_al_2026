"""Energy spectra of the four neutron regimes."""
import numpy as np
from scipy.special import erf

def cnorm(x):
    return (1 + erf(x/np.sqrt(2))) / 2

def cascade_spectrum(en, A, alpha, t_c, w_c):

    return 2*A*(en/t_c)**(alpha+1) * cnorm(np.log(t_c/en)/w_c)

def evaporation_spectrum(en, A, alpha, Epk):
    p = alpha + 1.0
    en_safe = np.maximum(en, 1e-300)
    log_val = p * (np.log(en_safe / Epk) + 1.0 - en_safe / Epk)
    return A * np.exp(np.clip(log_val, -745, 709))

def epithermal_spectrum(en):
    low = 1e-10
    upp = 1e-04
    return cnorm(np.log(upp/en))*(1-cnorm(np.log(low/en)))

def thermal_spectrum(en):
    T = 2.585e-11
    return 1e+15* en**2 * np.exp(-en / T)
