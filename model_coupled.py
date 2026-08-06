"""Option 2: analytical model with a coupled ballistic cascade regime.

Identical to `model.Model` except in the cascade regime. There the factorized
spatial form

    cascade_lateral(rho; sigma_cas) * cascade(z; Sigma_t, P)

is replaced by the coupled first-flight integral tabulated in
toolbox/diffusion_integrals/precompute_cascade.py,

    phi(rho, z; P, Sigma_h, n_ang)
        = int_0^P dz' S(z') cos^n(theta) exp(-Sigma_h s) / (4 pi s^2)

with s the slant distance from source point to field point. Motivation: in the
MC the fast (>20 MeV) fraction of the fluence is ~88 % at EVERY radius out to
5.5 cm and its mean energy is flat, i.e. those neutrons are ballistic, not
moderated. A Gaussian lateral factor falls as exp(-rho^2/2 sigma^2) and is ~1e-6
by 5 cm, so the factorized cascade under-predicts the fast field there by ~300x.
The factorized form also forces one z-profile at all radii, whereas the MC peak
migrates outward-and-deeper (z = 14 -> 18 cm between rho = 1 and 5.5 cm at
E0 = 170 MeV) because a neutron reaching large rho was emitted obliquely.

Parameter mapping relative to model.Model: (Sigma_t, sigma_cas) are replaced by
(Sigma_h, n_ang). Sigma_h is a REMOVAL constant -- elastic scattering off oxygen
leaves a fast neutron in the band, so in-scattering largely cancels removal and
the fitted value is far below sigma_tot. Sigma_t in the factorized model was
never a cross section at all: with no geometric term in the axial factor it had
to absorb the 1/s^2 dilution as well, which is why it fitted to ~0.35 cm^-1.

Fitted values: fitting_params/proton_coupled.csv (see README).
"""
import numpy as np

from model import Model
from toolbox.diffusion_integrals.diffusion_kernels import cascade_coupled, slow_diff_single
from toolbox.spectral_funcs import cascade_spectrum, epithermal_spectrum, thermal_spectrum


class CoupledModel(Model):
    """Model with the ballistic cascade kernel in place of the factorized one."""

    def cas(self, X, A, gamma, Sigma_h, n_ang, alpha, a, w_c):
        """Cascade regime via the coupled kernel.

        Signature mirrors Model.cas, with (Sigma_h, n_ang) where the factorized
        model takes (Sigma_t, sigma_cas), so the parameter vector keeps its
        length and position and the same fitting code drives both.
        """
        (Z, R, E, EP) = X
        P = gamma * self.XT[self.iEp, self.iEn]
        t_c = a * EP / 1000
        return cascade_coupled(P, Sigma_h, n_ang, Z, R) \
            * cascade_spectrum(E, A, alpha=alpha, t_c=t_c, w_c=w_c)

    def spectral_energy_fluence_coupled(self, X,
                                        A1, gamma1, Sigma_h, n_ang, d1, a, w_c,
                                        A2, gamma2, d2, kappa_ev, Epk,
                                        A3, gamma3, A4, gamma4,
                                        kappa_f, kappa_s,
                                        E_th, n):
        """Saturating prefactor, coupled cascade. 20 parameters, E_th/n trailing."""
        (Z, R, E, EP) = X
        return self.prefac2(EP, E_th, n) * (
              self.cas(X, A1, gamma1, Sigma_h, n_ang, d1, a, w_c)
            + self.ev( X, A2, gamma2, d2, kappa_ev, Epk)
            + self.ep( X, A3, gamma3, kappa_f, kappa_s)
            + self.th( X, A4, gamma4, kappa_f, kappa_s)
        )

    def spectral_energy_fluence_coupled_prefac1(self, X,
                                                xi,
                                                A1, gamma1, Sigma_h, n_ang, d1, a, w_c,
                                                A2, gamma2, d2, kappa_ev, Epk,
                                                A3, gamma3, A4, gamma4,
                                                kappa_f, kappa_s):
        """Power-law prefactor, coupled cascade. 19 parameters, xi leading."""
        (Z, R, E, EP) = X
        return self.prefac1(EP, xi) * (
              self.cas(X, A1, gamma1, Sigma_h, n_ang, d1, a, w_c)
            + self.ev( X, A2, gamma2, d2, kappa_ev, Epk)
            + self.ep( X, A3, gamma3, kappa_f, kappa_s)
            + self.th( X, A4, gamma4, kappa_f, kappa_s)
        )

    # ------------------------------------------------------------------
    # single-diffusion-length slow regimes
    # ------------------------------------------------------------------
    def ep_single(self, X, A, gamma, kappa):
        (Z, R, E, EP) = X
        P = gamma * self.XT[self.iEp, self.iEn]
        return A * slow_diff_single(P=P, kappa=kappa, r=R, z=Z) \
            * epithermal_spectrum(E)

    def th_single(self, X, A, gamma, kappa):
        (Z, R, E, EP) = X
        P = gamma * self.XT[self.iEp, self.iEn]
        return A * slow_diff_single(P=P, kappa=kappa, r=R, z=Z) \
            * thermal_spectrum(E)

    def spectral_energy_fluence_single(self, X,
                                       A1, gamma1, Sigma_h, n_ang, d1, a, w_c,
                                       A2, gamma2, d2, kappa_ev, Epk,
                                       A3, gamma3, A4, gamma4,
                                       kappa,
                                       E_th, n):
        """Coupled cascade + single-diffusion-length slow regimes.

        19 parameters: one kappa in place of (kappa_f, kappa_s).
        """
        (Z, R, E, EP) = X
        return self.prefac2(EP, E_th, n) * (
              self.cas(X, A1, gamma1, Sigma_h, n_ang, d1, a, w_c)
            + self.ev( X, A2, gamma2, d2, kappa_ev, Epk)
            + self.ep_single(X, A3, gamma3, kappa)
            + self.th_single(X, A4, gamma4, kappa)
        )

    def spectral_energy_fluence_single_prefac1(self, X,
                                               xi,
                                               A1, gamma1, Sigma_h, n_ang, d1, a, w_c,
                                               A2, gamma2, d2, kappa_ev, Epk,
                                               A3, gamma3, A4, gamma4,
                                               kappa):
        """Power-law prefactor, coupled cascade, single-kappa slow regimes.

        18 parameters, xi leading. E_ref is 120 MeV for protons (model.Model).
        """
        (Z, R, E, EP) = X
        return self.prefac1(EP, xi) * (
              self.cas(X, A1, gamma1, Sigma_h, n_ang, d1, a, w_c)
            + self.ev( X, A2, gamma2, d2, kappa_ev, Epk)
            + self.ep_single(X, A3, gamma3, kappa)
            + self.th_single(X, A4, gamma4, kappa)
        )

    # ------------------------------------------------------------------
    # cascade with near-threshold suppression
    # ------------------------------------------------------------------
    def cas_supp(self, X, A, gamma, Sigma_h, n_ang, alpha, a, w_c, m_supp):
        """Cascade with the yield tapered towards the kinematic bound.

        xt(E0, En) is the depth at which the primary's residual energy still
        equals En, so production of an En neutron is confined to [0, xt] and
        xt -> 0 as En -> E0. That bound is correct, but the model has been
        treating every point of [0, xt] as producing at the same rate. Near the
        far end of the column the residual energy IS En, so emitting that
        neutron requires transferring essentially all of it to one nucleon --
        allowed, but strongly suppressed. The result was a factor ~25-80
        over-prediction just below the bound (E0 = 54 MeV, En = 31-39 MeV),
        which the kerma coefficient's E^2 branch then amplified into +17 pp of
        Delta_K.

        xt/R(E0) is the fraction of the track over which production is allowed,
        so (xt/R)^m suppresses exactly those cases, equals 1 well below the
        bound and vanishes at it. One parameter; m = 0 recovers the old model.
        """
        (Z, R, E, EP) = X
        xt = self.XT[self.iEp, self.iEn]
        R0 = self.XT[self.iEp, 0]                 # xt at the lowest En ~ full range
        frac = np.clip(xt / np.maximum(R0, 1e-12), 0.0, 1.0)
        P = gamma * xt
        t_c = a * EP / 1000
        return frac ** m_supp \
            * cascade_coupled(P, Sigma_h, n_ang, Z, R) \
            * cascade_spectrum(E, A, alpha=alpha, t_c=t_c, w_c=w_c)

    def spectral_energy_fluence_supp_prefac1(self, X,
                                             xi,
                                             A1, gamma1, Sigma_h, n_ang, d1, a, w_c, m_supp,
                                             A2, gamma2, d2, kappa_ev, Epk,
                                             A3, gamma3, A4, gamma4,
                                             kappa):
        """Power law, coupled cascade with threshold suppression, single-kappa slow.

        19 parameters, xi leading, m_supp following the cascade block.
        """
        (Z, R, E, EP) = X
        return self.prefac1(EP, xi) * (
              self.cas_supp(X, A1, gamma1, Sigma_h, n_ang, d1, a, w_c, m_supp)
            + self.ev( X, A2, gamma2, d2, kappa_ev, Epk)
            + self.ep_single(X, A3, gamma3, kappa)
            + self.th_single(X, A4, gamma4, kappa)
        )
