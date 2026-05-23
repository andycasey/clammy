"""Task 2: toy data generator with known (v, R, w, c).

Builds a synthetic observed spectrum from a known rectified log-flux truth:
  1. degrade to a known spectral resolution R (Gaussian LSF, applied to the
     *linear* rectified flux -- the physically correct space for convolution),
  2. shift it by a known RV (Fourier sub-pixel translation on the log grid),
  3. add a known low-order Legendre log-continuum (multiplicative in linear flux),
  4. optionally resample onto a different output log-lambda grid,
  5. add Gaussian noise at a chosen per-pixel SNR.

Returns the noisy flux, the noise sigma vector, and the ground truth.
"""
import numpy as np
import scipy.ndimage as ndi

from . import C_LIGHT_KMS, sigma_kms_of_R
from .basis import xnorm
from numpy.polynomial import legendre as npleg


def broaden_linear(r_logrect, sigma_pix):
    """Gaussian-broaden a log-rectified spectrum in *linear* flux (exact).

    The LSF convolves photons (linear flux), not log-flux, so we exponentiate,
    convolve, and take the log again. This is the deep-line-correct broadening
    the fitter's log-space convolution only approximates.
    """
    if sigma_pix <= 0:
        return np.asarray(r_logrect, float)
    rect = np.exp(np.asarray(r_logrect, float))
    rect_b = ndi.gaussian_filter1d(rect, sigma_pix, mode="nearest")
    return np.log(np.maximum(rect_b, 1e-300))


def resample_to_loglam(src_loglam, flux, sigma, dst_loglam):
    """Linear-interpolate (flux, sigma) from one log-lambda grid to another.

    Returns ``(flux_dst, sigma_dst, good)`` where ``good`` flags destination
    pixels inside the source coverage. NB: interpolation correlates the noise
    between adjacent destination pixels -- fine for point estimates, but the
    diagonal-covariance assumption (and hence chi2/dof) becomes approximate.
    """
    src_loglam = np.asarray(src_loglam, float)
    dst_loglam = np.asarray(dst_loglam, float)
    good = (dst_loglam >= src_loglam[0]) & (dst_loglam <= src_loglam[-1])
    flux_dst = np.interp(dst_loglam, src_loglam, flux)
    sigma_dst = np.interp(dst_loglam, src_loglam, sigma)
    return flux_dst, sigma_dst, good


def fourier_shift(y, shift_pix):
    """Translate ``y`` by ``shift_pix`` pixels (fractional ok) via FFT.

    Positive ``shift_pix`` shifts content to higher index: out[i] = y[i - shift].
    Periodic; edge wrap is confined to a band of width ``ceil(shift_pix)`` and is
    masked out by the fitter.
    """
    n = y.size
    freq = np.fft.rfftfreq(n)
    spec = np.fft.rfft(y) * np.exp(-2j * np.pi * freq * shift_pix)
    return np.fft.irfft(spec, n=n)


def dx_of_v(v_kms):
    """Log-wavelength shift for radial velocity ``v_kms`` (km/s)."""
    return np.log1p(np.asarray(v_kms, float) / C_LIGHT_KMS)


def make_toy(loglam, r_true, v_kms, cont_coef, snr, resolution_R=None,
             out_loglam=None, seed=0):
    """Generate one noisy toy spectrum.

    Parameters
    ----------
    loglam : (n_pix,)        log-lambda grid the truth lives on
    r_true : (n_pix,)        true rectified *log*-flux (no broadening/shift/continuum)
    v_kms : float            true radial velocity, km/s
    cont_coef : array        Legendre coeffs of the true log-continuum (x in [-1,1])
    snr : float              target per-pixel signal-to-noise of the clean flux
    resolution_R : float|None   degrade to this resolving power (Gaussian LSF, linear
                                flux). None = leave at the template resolution.
    out_loglam : (m,)|None   deliver the observation on this (different) log grid;
                             None = keep ``loglam``.
    seed : int               RNG seed

    Returns
    -------
    d     : noisy linear flux on the output grid
    sigma : per-pixel 1-sigma noise on the output grid
    truth : dict  (v_kms, resolution_R, sigma_kms, shift_pix, loglam_out, clean arrays, ...)
    """
    rng = np.random.default_rng(seed)
    loglam = np.asarray(loglam, float)
    dln = float(loglam[1] - loglam[0])
    velscale = C_LIGHT_KMS * dln
    dx = float(dx_of_v(v_kms))
    shift_pix = dx / dln

    # 1. resolution degradation (linear-flux Gaussian); convolution commutes with
    #    the RV translation, so order vs. the shift does not matter.
    sigma_kms = sigma_kms_of_R(resolution_R) if resolution_R else 0.0
    r_b = broaden_linear(r_true, sigma_kms / velscale) if sigma_kms else np.asarray(r_true, float)

    # 2-3. shift + multiplicative (log-additive) continuum
    r_shift = fourier_shift(r_b, shift_pix)
    log_cont = npleg.legval(xnorm(loglam), np.asarray(cont_coef, float))
    d_clean = np.exp(r_shift + log_cont)

    # 4. optional resample onto a different output grid
    if out_loglam is None:
        loglam_out = loglam
        clean_out = d_clean
        good = np.ones(loglam.size, bool)
    else:
        loglam_out = np.asarray(out_loglam, float)
        clean_out, _, good = resample_to_loglam(loglam, d_clean,
                                                np.zeros_like(d_clean), loglam_out)

    # 5. noise
    sigma = np.where(good, clean_out / float(snr), np.inf)
    d = clean_out + rng.normal(0.0, np.where(good, clean_out / float(snr), 0.0))

    truth = {
        "v_kms": float(v_kms),
        "dx": dx,
        "shift_pix": shift_pix,
        "resolution_R": float(resolution_R) if resolution_R else np.inf,
        "sigma_kms": float(sigma_kms),
        "cont_coef": np.asarray(cont_coef, float),
        "snr": float(snr),
        "loglam_out": loglam_out,
        "good": good,
        "d_clean": clean_out,
    }
    return d, sigma, truth


def project_onto_basis(mu, Phi, r):
    """Least-squares template weights so that ``mu + w·Phi`` best fits ``r``.

    Gives the ground-truth weight vector ``w`` (incl. the mu weight, first entry)
    for a target rectified log-flux ``r`` that may not lie exactly in the basis.
    """
    # design: [mu, Phi^T]; solve normal equations
    A = np.vstack([mu, Phi]).T  # (n_pix, K+1)
    w, *_ = np.linalg.lstsq(A, r, rcond=None)
    return w  # length K+1; w[0] multiplies mu
