"""Analytical model for the secondary neutron spectral energy fluence.

The fluence is a prefactor in the primary energy E0 times a sum over four energy
regimes -- cascade, evaporation, epithermal, thermal -- each a product of a
spatial and a spectral factor:

    Phi(z, rho, En; E0) = prefac(E0) * sum_i A_i spatial_i(z, rho; P_i) spec_i(En)

with P_i = gamma_i * xt(E0, En) the production range of regime i.

Cascade
-------
The cascade spatial factor is the coupled first-flight integral over the source
volume V,

    F^cas(rho, z) = int_V d3x' g(theta) exp(-Sigma s) / (4 pi s^2)

with s = |x - x'| and cos(theta) = (z - z')/s, tabulated by
toolbox/diffusion_integrals/precompute_cascade.py. A separable form -- a
Gaussian-smeared disk times an axial profile -- cannot reproduce the field
beyond rho ~ 3 cm, because a neutron reaching large rho was emitted obliquely
and therefore arrives DEEPER than its emission point, which a rho-independent
axial factor cannot express.

The forward lobe g(theta) = cos^n_ang(theta) is exposed two ways. `cas` takes
the exponent n_ang directly; `cas_theta` takes the MEAN EMISSION ANGLE in
degrees, which is the better-conditioned coordinate for fitting: the lobe width
goes as n_ang^(-1/2), so d(width)/d(n_ang) vanishes and an optimizer wanting a
forward peak runs the exponent to infinity with nothing to rail against. In
theta_bar the whole cos^n family maps onto a bounded interval -- 60 deg is
n_ang = 0, 0 deg the forward delta -- and the shape gradient grows rather than
dies at the narrow end. The two are related exactly by

    <cos theta> = (n_ang + 1) / (n_ang + 2).

Slow neutrons
-------------
Epithermal and thermal share one kernel over the same uniform cylindrical source
the other regimes use, with a single inverse diffusion length kappa. This is the
two-group problem with both lengths collapsed onto one, giving the squared
propagator 1/(k^2 + kappa^2)^2 -- in free space e^{-kappa r}/(8 pi kappa), where
the 1/r cancels, which is why the kernel is flat near the source and broad
enough to match the MC where a single Yukawa is not. Migration length is
sqrt(2)/kappa.

Prefactor
---------
    prefac2(E0) = 1 - exp(-(E0/E_th)^n)

the "fraction of primaries that interact before stopping" law: substituting the
Bragg-Kleeman range R ~ E0^p into 1 - exp(-Sigma_inel R) gives exactly this form
and predicts n ~ p. Both production sets use it.

Fitted values live in fitting_params/params_{proton,carbon}.csv, with columns in
the same order as the corresponding signature so they can be unpacked
positionally.
"""
import numpy as np

from toolbox.diffusion_integrals.diffusion_kernels import (
    cascade_coupled, cascade_coupled_theta, evaporation_diff, slow_diff_single,
)
from toolbox.spectral_funcs import (
    cascade_spectrum, epithermal_spectrum, evaporation_spectrum,
    thermal_spectrum,
)


class Model:
    """Four energy regimes over a common cylindrical source."""

    def __init__(self, species='proton', XT=None, iEp=None, iEn=None):
        self.species = species
        self.E_ref = 120 if species == 'proton' else 200
        self.XT = XT
        self.iEp = iEp
        self.iEn = iEn

    # ------------------------------------------------------------------
    # energy regimes
    # ------------------------------------------------------------------
    def cas(self, X, A, gamma, Sigma, n_ang, alpha, a, w_c):
        """Cascade, angular lobe given as the exponent of cos^n_ang(theta)."""
        (Z, R, E, EP) = X
        P = gamma * self.XT[self.iEp, self.iEn]
        t_c = a * EP / 1000
        return cascade_coupled(P, Sigma, n_ang, Z, R) \
            * cascade_spectrum(E, A, alpha=alpha, t_c=t_c, w_c=w_c)

    def cas_theta(self, X, A, gamma, Sigma, theta_bar, alpha, a, w_c):
        """Cascade, angular lobe given as its mean emission angle in degrees.

        Same kernel as `cas`; only the angular coordinate differs. Bound
        theta_bar to the table's [6, 60] deg when fitting.
        """
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

    # ------------------------------------------------------------------
    # prefactor
    # ------------------------------------------------------------------
    def prefac2(self, EP, E_th, n):
        """Saturation curve with threshold E_th and sharpness n."""
        return 1 - np.exp(-(EP / E_th) ** n)

    # ------------------------------------------------------------------
    # spectral energy fluence -- 19 parameters, consumed positionally
    # ------------------------------------------------------------------
    def spectral_energy_fluence(self, X,
                                A1, gamma1, Sigma, n_ang, d1, a, w_c,
                                A2, gamma2, d2, kappa_ev, Epk,
                                A3, gamma3, A4, gamma4,
                                kappa_slow,
                                E_th, n):
        """Angular lobe as the exponent. Used by params_proton.csv."""
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
        """Angular lobe as the mean angle. Used by params_carbon.csv.

        Same order and length as `spectral_energy_fluence`, so one fitting
        routine drives both; only the bounds on slot 3 differ.
        """
        (Z, R, E, EP) = X
        return self.prefac2(EP, E_th, n) * (
              self.cas_theta(X, A1, gamma1, Sigma, theta_bar, d1, a, w_c)
            + self.ev(X, A2, gamma2, d2, kappa_ev, Epk)
            + self.ep(X, A3, gamma3, kappa_slow)
            + self.th(X, A4, gamma4, kappa_slow)
        )
