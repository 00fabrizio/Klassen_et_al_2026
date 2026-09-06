"""Analytical model of the secondary neutron spectral energy fluence."""
import numpy as np

from toolbox.diffusion_integrals.diffusion_kernels import (
    cascade_coupled, cascade_coupled_theta, evaporation_diff, slow_diff_single,
)
from toolbox.spectral_funcs import (
    cascade_spectrum, epithermal_spectrum, evaporation_spectrum,
    thermal_spectrum,
)


class Model:

    def __init__(self, species='proton', XT=None, iEp=None, iEn=None):
        self.species = species
        self.E_ref = 120 if species == 'proton' else 200
        self.XT = XT
        self.iEp = iEp
        self.iEn = iEn

    def cas(self, X, A, gamma, Sigma, n_ang, alpha, a, w_c):
        (Z, R, E, EP) = X
        P = gamma * self.XT[self.iEp, self.iEn]
        t_c = a * EP / 1000
        return cascade_coupled(P, Sigma, n_ang, Z, R) \
            * cascade_spectrum(E, A, alpha=alpha, t_c=t_c, w_c=w_c)

    def cas_theta(self, X, A, gamma, Sigma, theta_bar, alpha, a, w_c):
        (Z, R, E, EP) = X
        P = gamma * self.XT[self.iEp, self.iEn]
        t_c = a * EP / 1000
        return cascade_coupled_theta(P, Sigma, theta_bar, Z, R) \
            * cascade_spectrum(E, A, alpha=alpha, t_c=t_c, w_c=w_c)

    def ev(self, X, A, gamma, alpha, kappa_ev, Epk):
        (Z, R, E, EP) = X
        P = gamma * self.XT[self.iEp, self.iEn]
        return np.nan_to_num(
            evaporation_diff(P=P, kappa=kappa_ev, r=R, z=Z)
            * evaporation_spectrum(E, A, alpha, Epk),
            nan=0.0, posinf=0.0, neginf=0.0,
        )

    def ep(self, X, A, gamma, kappa_slow):
        (Z, R, E, EP) = X
        P = gamma * self.XT[self.iEp, self.iEn]
        return A * slow_diff_single(P=P, kappa=kappa_slow, r=R, z=Z) \
            * epithermal_spectrum(E)

    def th(self, X, A, gamma, kappa_slow):
        (Z, R, E, EP) = X
        P = gamma * self.XT[self.iEp, self.iEn]
        return A * slow_diff_single(P=P, kappa=kappa_slow, r=R, z=Z) \
            * thermal_spectrum(E)

    def prefac2(self, EP, E_th, n):
        return 1 - np.exp(-(EP / E_th) ** n)

    def spectral_energy_fluence(self, X,
                                A1, gamma1, Sigma, n_ang, d1, a, w_c,
                                A2, gamma2, d2, kappa_ev, Epk,
                                A3, gamma3, A4, gamma4,
                                kappa_slow,
                                E_th, n):
        (Z, R, E, EP) = X
        return self.prefac2(EP, E_th, n) * (
              self.cas(X, A1, gamma1, Sigma, n_ang, d1, a, w_c)
            + self.ev(X, A2, gamma2, d2, kappa_ev, Epk)
            + self.ep(X, A3, gamma3, kappa_slow)
            + self.th(X, A4, gamma4, kappa_slow)
        )

    def spectral_energy_fluence_theta(self, X,
                                      A1, gamma1, Sigma, theta_bar, d1, a, w_c,
                                      A2, gamma2, d2, kappa_ev, Epk,
                                      A3, gamma3, A4, gamma4,
                                      kappa_slow,
                                      E_th, n):
        (Z, R, E, EP) = X
        return self.prefac2(EP, E_th, n) * (
              self.cas_theta(X, A1, gamma1, Sigma, theta_bar, d1, a, w_c)
            + self.ev(X, A2, gamma2, d2, kappa_ev, Epk)
            + self.ep(X, A3, gamma3, kappa_slow)
            + self.th(X, A4, gamma4, kappa_slow)
        )
