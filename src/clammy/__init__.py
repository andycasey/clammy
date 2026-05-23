"""Joint RV + template + continuum fitting for stellar spectra.

Forward model (log-flux, additive):

    ln d(x) ~= sum_k w_k T_k(x - dx) + sum_m c_m P_m(x)

with x = ln(lambda) and a radial velocity acting as a uniform translation
dx = ln(1 + v/c). The templates T_k ARE the log of the rectified flux: T_0 = mu
(mean rectified log-flux) and T_k = phi_k (its PCA components), so the model is
linear in the weights w. The P_m are low-order Legendre continuum functions.

See module docstrings for the four tasks:
  grid    -- load the PHOENIX grid onto the shared log-lambda grid
  basis   -- Task 1: rectify + PCA template basis
  toy     -- Task 2: toy data generator
  fitter  -- Task 3: JAX fitter (FFT cross-correlation + normal equations)
"""

# JAX is the linear-algebra / fitting backend; enable double precision globally
# before any jax array is created.
from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

C_LIGHT_KMS = 299792.458  # speed of light, km/s
FWHM_OVER_SIGMA = 2.3548200450309493  # 2*sqrt(2 ln 2)


def sigma_kms_of_R(R):
    """Gaussian-LSF velocity dispersion (km/s) for a resolving power R.

    A constant resolving power R = lambda/dlambda has a constant *velocity*
    FWHM = c/R, hence sigma_v = c / (FWHM_OVER_SIGMA * R). On a log-lambda grid
    (uniform in velocity) this is a constant width in pixels, which is what makes
    the broadening a single Fourier-domain Gaussian multiply.
    """
    return C_LIGHT_KMS / (FWHM_OVER_SIGMA * R)


def R_of_sigma_kms(sigma_kms):
    """Resolving power R for a Gaussian-LSF velocity dispersion (km/s)."""
    return C_LIGHT_KMS / (FWHM_OVER_SIGMA * sigma_kms)


__all__ = ["C_LIGHT_KMS", "FWHM_OVER_SIGMA", "sigma_kms_of_R", "R_of_sigma_kms"]
