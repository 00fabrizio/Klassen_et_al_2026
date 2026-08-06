import numpy as np
from toolbox.diffusion_integrals.diffusion_kernels import *
from toolbox.spatial_funcs import *
from toolbox.spectral_funcs import *


class Model():
    """
    Analytical model for the secondary neutron spectral energy fluence.

    The fluence is factorized into a global prefactor in the primary energy
    EP and a sum over the four energy regimes (cascade, evaporation,
    epithermal, thermal), each a product of a spatial and a spectral factor.

    Two prefactor parametrizations are provided:

        prefac1  power law         (EP / E_ref)**xi
        prefac2  saturation curve  1 - exp(-(EP / E_th)**n)

    They are exposed as two separate entry points,
    spectral_energy_fluence_prefac1 and spectral_energy_fluence_prefac2,
    because their free-parameter lists differ in both length and order and
    are consumed positionally by curve_fit:

        prefac1  18 parameters, xi leading
        prefac2  20 parameters, E_th and n trailing

    Fitted values live in fitting_params/{species}_prefac{1,2}.csv, with
    columns in the same order as the corresponding signature.
    """

    def __init__(self, species='proton', XT=None, iEp=None, iEn=None):
        self.species = species
        self.E_ref   = 120 if species == 'proton' else 200
        self.XT  = XT
        self.iEp = iEp
        self.iEn = iEn

    # ------------------------------------------------------------------
    # energy regimes
    # ------------------------------------------------------------------
    def cas(self, X, A, gamma, Sigma_t, Sigma, alpha, a, w_c):
        (Z, R, E, EP) = X
        P   = gamma * self.XT[self.iEp, self.iEn]
        t_c = a * EP / 1000
        return cascade_lateral(R, R=1, sigma=Sigma) \
             * cascade(Z, Sigma_t=Sigma_t, P=P) \
             * cascade_spectrum(E, A, alpha=alpha, t_c=t_c, w_c=w_c)


    def ev(self, X, A, gamma, alpha, kappa, T):
        (Z, R, E, EP) = X
        P = gamma * self.XT[self.iEp, self.iEn]
        return np.nan_to_num(
            evaporation_diff(P=P, kappa=kappa, r=R, z=Z) * evaporation_spectrum(E, A, alpha, T),
            nan=0.0, posinf=0.0, neginf=0.0,
        )

    def ep(self, X, A, gamma, kappa_f, kappa_s):
        (Z, R, E, EP) = X
        P = gamma * self.XT[self.iEp, self.iEn]
        return A * slow_diff(P=P, kappa_f=kappa_f, kappa_s=kappa_s, r=R, z=Z) \
            * epithermal_spectrum(E)

    def th(self, X, A, gamma, kappa_f, kappa_s):
        (Z, R, E, EP) = X
        P = gamma * self.XT[self.iEp, self.iEn]
        return A * slow_diff(P=P, kappa_f=kappa_f, kappa_s=kappa_s, r=R, z=Z) \
            * thermal_spectrum(E)

    def get_component(self, name):
        return {
            "cas": self.cas,
            "ev": self.ev,
            "ep": self.ep,
            "th": self.th,
        }[name]

    # ------------------------------------------------------------------
    # prefactors
    # ------------------------------------------------------------------
    def prefac1(self, EP, xi):
        """Power law in the primary energy."""
        return (EP / self.E_ref) ** xi

    def prefac2(self, EP, E_th, n):
        """Saturation curve with threshold E_th and sharpness n."""
        return 1 - np.exp(-(EP / E_th) ** n)

    # ------------------------------------------------------------------
    # spectral energy fluence
    # ------------------------------------------------------------------
    def _regimes(self, X,
                 A1, gamma1, Sigma_t, Sigma, d1, a, w_c,
                 A2, gamma2, d2, kappa_ev, Epk,
                 A3, gamma3, A4, gamma4,
                 kappa_f, kappa_s):
        # epithermal and thermal share one slow-neutron spatial kernel over the
        # same cylindrical source as the other regimes; they differ only in
        # amplitude, production range and energy spectrum
        return (  self.cas(X, A1, gamma1, Sigma_t, Sigma, d1, a, w_c)
                + self.ev( X, A2, gamma2, d2, kappa_ev, Epk)
                + self.ep( X, A3, gamma3, kappa_f, kappa_s)
                + self.th( X, A4, gamma4, kappa_f, kappa_s) )

    def spectral_energy_fluence_prefac1(self, X,
                                        xi,
                                        A1, gamma1, Sigma_t, Sigma, d1, a, w_c,
                                        A2, gamma2, d2, kappa_ev, Epk,
                                        A3, gamma3, A4, gamma4,
                                        kappa_f, kappa_s):
        """Power-law prefactor. 18 parameters, xi leading."""
        (Z, R, E, EP) = X
        return self.prefac1(EP, xi) * self._regimes(
            X,
            A1, gamma1, Sigma_t, Sigma, d1, a, w_c,
            A2, gamma2, d2, kappa_ev, Epk,
            A3, gamma3, A4, gamma4,
            kappa_f, kappa_s,
        )

    def spectral_energy_fluence_prefac2(self, X,
                                        A1, gamma1, Sigma_t, Sigma, d1, a, w_c,
                                        A2, gamma2, d2, kappa_ev, Epk,
                                        A3, gamma3, A4, gamma4,
                                        kappa_f, kappa_s,
                                        E_th, n):
        """Saturation-curve prefactor. 20 parameters, E_th and n trailing."""
        (Z, R, E, EP) = X
        return self.prefac2(EP, E_th, n) * self._regimes(
            X,
            A1, gamma1, Sigma_t, Sigma, d1, a, w_c,
            A2, gamma2, d2, kappa_ev, Epk,
            A3, gamma3, A4, gamma4,
            kappa_f, kappa_s,
        )
