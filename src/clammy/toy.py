"""Task 2: toy data generator with known (v, R, vsini, w, c, tellurics).

Builds a synthetic observed spectrum from a known rectified log-flux truth:
  1. broaden by a known projected rotation velocity vsini (Gray rotational
     profile, applied to the *linear* rectified flux),
  2. degrade to a known spectral resolution R (Gaussian LSF, applied to the
     *linear* rectified flux -- the physically correct space for convolution),
  3. shift it by a known RV (Fourier sub-pixel translation on the log grid),
  4. optionally imprint a telluric absorption spectrum in the OBSERVED
     (topocentric) frame -- i.e. NOT RV-shifted -- as an additive log-flux term,
     optionally itself instrument-broadened,
  5. add a known low-order Legendre log-continuum (multiplicative in linear flux),
  6. optionally resample onto a different output log-lambda grid,
  7. add Gaussian noise at a chosen per-pixel SNR.

Rotation and the LSF are both convolutions, which commute with the RV
translation, so they are applied to the rest-frame rectified flux before the
shift. Tellurics live in the observed frame and so are added *after* the shift,
with no RV translation -- exactly the asymmetry the fitter exploits.

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


def rotation_kernel(vsini_pix, n, epsilon=0.6):
    """Normalized Gray rotational broadening kernel on an integer-pixel grid.

    The classic Gray (2005) rotation profile, written in velocity offset Dv with
    x = Dv / vsini for |x| <= 1:

        G(Dv) proportional to  2 (1 - eps) sqrt(1 - x^2) + (pi eps / 2) (1 - x^2)

    with linear limb-darkening coefficient ``epsilon``. On a log-lambda grid the
    velocity per pixel is constant, so vsini is a constant number of pixels
    (``vsini_pix``) and the kernel is a fixed array of integer-pixel offsets. We
    build it in zero-phase / ``fftshift`` ordering (offsets 0, 1, ..., n/2, ...,
    -2, -1) so that convolving via ``rfft`` introduces no net shift, and
    normalize it to unit *sum* on the grid (matching the fitter's discrete
    transfer function exactly).
    """
    j = np.fft.fftfreq(n) * n  # 0,1,...,n/2,...,-2,-1  (integer pixel offsets)
    x = j / float(vsini_pix)
    one_minus_x2 = np.clip(1.0 - x * x, 0.0, None)
    inside = np.abs(j) <= vsini_pix
    k = np.where(inside,
                 2.0 * (1.0 - epsilon) * np.sqrt(one_minus_x2)
                 + (0.5 * np.pi * epsilon) * one_minus_x2,
                 0.0)
    s = k.sum()
    return k / s if s > 0 else k


def broaden_rotation_linear(r_logrect, vsini_pix, epsilon=0.6):
    """Rotationally broaden a log-rectified spectrum in *linear* flux (Gray).

    Analogous to :func:`broaden_linear` but with the Gray rotation kernel instead
    of a Gaussian: exponentiate to linear rectified flux, circularly convolve with
    the (zero-phase) rotation kernel, take the log again. Rotation, like the LSF,
    is a convolution of the linear flux, so this is the physically correct toy
    broadening. The kernel is zero-phase, so it introduces no net velocity shift.
    """
    if vsini_pix is None or vsini_pix <= 0:
        return np.asarray(r_logrect, float)
    rect = np.exp(np.asarray(r_logrect, float))
    n = rect.size
    k = rotation_kernel(vsini_pix, n, epsilon=epsilon)
    rect_b = np.fft.irfft(np.fft.rfft(rect) * np.fft.rfft(k), n=n)
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
             out_loglam=None, seed=0, *, vsini_kms=None, epsilon=0.6,
             tell_r=None, tell_broaden=True):
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
    vsini_kms : float|None   keyword-only. Projected rotation velocity (km/s) to
                             apply to the STELLAR flux via the Gray profile, in
                             linear flux, before the RV shift. None = no rotation.
    epsilon : float          keyword-only. Linear limb-darkening coefficient for
                             the Gray rotation profile (default 0.6).
    tell_r : (n_pix,)|None   keyword-only. A telluric rectified *log*-flux on
                             ``loglam`` in the OBSERVED (topocentric) frame. If
                             given it is added to the log-flux WITHOUT any RV
                             shift, before the continuum and noise -- so a fit can
                             recover the injected telluric weights. None = none.
    tell_broaden : bool      keyword-only. If True (default) and ``resolution_R``
                             is set, the injected telluric is instrument-broadened
                             by the same resolution as the star (tellurics share
                             the instrument LSF). Set False to inject an
                             already-broadened telluric.

    Returns
    -------
    d     : noisy linear flux on the output grid
    sigma : per-pixel 1-sigma noise on the output grid
    truth : dict  (v_kms, resolution_R, sigma_kms, vsini_kms, vsini_pix, shift_pix,
                  loglam_out, clean arrays, telluric info, ...)
    """
    rng = np.random.default_rng(seed)
    loglam = np.asarray(loglam, float)
    dln = float(loglam[1] - loglam[0])
    velscale = C_LIGHT_KMS * dln
    dx = float(dx_of_v(v_kms))
    shift_pix = dx / dln

    # 1. rotational (Gray) broadening of the stellar flux (linear-flux conv);
    #    a convolution, so it commutes with the RV translation -> apply first.
    vsini_pix = (float(vsini_kms) / velscale) if vsini_kms else 0.0
    r_rot = (broaden_rotation_linear(r_true, vsini_pix, epsilon=epsilon)
             if vsini_pix > 0 else np.asarray(r_true, float))

    # 2. resolution degradation (linear-flux Gaussian); also a convolution, so it
    #    commutes with the RV translation, and order vs. the shift does not matter.
    sigma_kms = sigma_kms_of_R(resolution_R) if resolution_R else 0.0
    sigma_pix = sigma_kms / velscale
    r_b = broaden_linear(r_rot, sigma_pix) if sigma_pix > 0 else r_rot

    # 3. shift the stellar log-flux to the observed frame
    r_shift = fourier_shift(r_b, shift_pix)

    # 4. telluric absorption in the OBSERVED frame (NO RV shift). The instrument
    #    LSF broadens tellurics too, so optionally broaden by the same resolution.
    if tell_r is not None:
        tell_r = np.asarray(tell_r, float)
        tell_b = (broaden_linear(tell_r, sigma_pix)
                  if (tell_broaden and sigma_pix > 0) else tell_r)
        log_star_plus_tell = r_shift + tell_b
    else:
        tell_b = None
        log_star_plus_tell = r_shift

    # 5. multiplicative (log-additive) continuum
    log_cont = npleg.legval(xnorm(loglam), np.asarray(cont_coef, float))
    d_clean = np.exp(log_star_plus_tell + log_cont)

    # 6. optional resample onto a different output grid
    if out_loglam is None:
        loglam_out = loglam
        clean_out = d_clean
        good = np.ones(loglam.size, bool)
    else:
        loglam_out = np.asarray(out_loglam, float)
        clean_out, _, good = resample_to_loglam(loglam, d_clean,
                                                np.zeros_like(d_clean), loglam_out)

    # 7. noise
    sigma = np.where(good, clean_out / float(snr), np.inf)
    d = clean_out + rng.normal(0.0, np.where(good, clean_out / float(snr), 0.0))

    truth = {
        "v_kms": float(v_kms),
        "dx": dx,
        "shift_pix": shift_pix,
        "resolution_R": float(resolution_R) if resolution_R else np.inf,
        "sigma_kms": float(sigma_kms),
        "vsini_kms": float(vsini_kms) if vsini_kms else 0.0,
        "vsini_pix": float(vsini_pix),
        "epsilon": float(epsilon),
        "cont_coef": np.asarray(cont_coef, float),
        "snr": float(snr),
        "loglam_out": loglam_out,
        "good": good,
        "d_clean": clean_out,
        "has_telluric": tell_r is not None,
        "tell_r": tell_r,
        "tell_r_broadened": tell_b,
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
