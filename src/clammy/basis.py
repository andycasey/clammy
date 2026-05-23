"""
Build a low-dimensional, rectified template basis from the grid.

Pipeline:
  1. log-flux  y = ln(flux)             (flux is strictly positive here)
  2. rectify   r = y - continuum(y)     robust low-order Legendre in log-space,
               so each rectified spectrum is ~0 in the continuum and dips
               negative in absorption lines. This isolates *line* information;
               the smooth pseudo-continuum is handled by the fitter's Legendre
               block, which keeps the template/continuum roles separate.
  3. PCA on the rectified spectra (Gram-matrix trick, since n_spec << n_pix),
     keep enough components for `var_target` of the variance.

The saved basis is (mean spectrum mu, components Phi). In the forward model the
"templates" are [mu, Phi_1, ..., Phi_K]; mu enters with a free weight so it is
shifted by the RV like any other template.
"""
import numpy as np
from numpy.polynomial import legendre as npleg


def xnorm(loglam):
    """Map the log-lambda grid to [-1, 1] for Legendre evaluation."""
    a, b = float(loglam[0]), float(loglam[-1])
    return 2.0 * (np.asarray(loglam, float) - a) / (b - a) - 1.0


def fit_log_continuum(logflux, xn, order=5, niter=5, low=1.5, high=3.0):
    """Robust Legendre continuum of a single log-flux spectrum.

    Asymmetric sigma clipping rejects absorption (points well *below* the fit)
    harder than it rejects positive excursions, so the fit tracks the upper
    (continuum) envelope.
    """
    good = np.isfinite(logflux)
    coef = npleg.legfit(xn[good], logflux[good], order)
    for _ in range(niter):
        model = npleg.legval(xn, coef)
        resid = logflux - model
        s = np.std(resid[good])
        if s == 0:
            break
        newgood = np.isfinite(logflux) & (resid > -low * s) & (resid < high * s)
        if newgood.sum() < order + 2 or np.array_equal(newgood, good):
            good = newgood
            break
        good = newgood
        coef = npleg.legfit(xn[good], logflux[good], order)
    return npleg.legval(xn, coef), coef


def rectify_grid(loglam, flux, order=5, **clip_kw):
    """Return ``(R, conts)``: log-rectified spectra and the continuum models."""
    xn = xnorm(loglam)
    logflux = np.log(flux)
    R = np.empty_like(logflux)
    conts = np.empty_like(logflux)
    for i in range(flux.shape[0]):
        cont, _ = fit_log_continuum(logflux[i], xn, order=order, **clip_kw)
        conts[i] = cont
        R[i] = logflux[i] - cont
    return R, conts


def build_pca(R, var_target=0.99, max_k=None):
    """PCA of the rectified spectra.

    Uses the (n_spec x n_spec) Gram matrix R_c R_c^T rather than the
    (n_pix x n_pix) covariance, since n_pix >> n_spec.

    Returns
    -------
    mu     : (n_pix,)        mean rectified spectrum
    Phi    : (K, n_pix)      unit-norm principal directions (templates)
    info   : dict            'evals', 'var_ratio', 'cumvar', 'K'
    """
    n = R.shape[0]
    mu = R.mean(axis=0)
    Xc = R - mu
    gram = Xc @ Xc.T  # (n, n); eigenvalues are those of Xc^T Xc too
    evals, evecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0.0, None)
    evecs = evecs[:, order]

    var_ratio = evals / evals.sum()
    cumvar = np.cumsum(var_ratio)
    K = int(np.searchsorted(cumvar, var_target) + 1)
    if max_k is not None:
        K = min(K, max_k)
    K = max(K, 1)

    # principal directions in pixel space: v_k = Xc^T u_k / sqrt(eval_k)
    comps = Xc.T @ evecs[:, :K]  # (n_pix, K)
    norms = np.linalg.norm(comps, axis=0)
    norms[norms == 0] = 1.0
    Phi = (comps / norms).T  # (K, n_pix), unit norm

    info = {"evals": evals, "var_ratio": var_ratio, "cumvar": cumvar, "K": K}
    return mu, Phi, info


def save_basis(path, loglam, mu, Phi, params, info, rectify_order):
    """Save the basis to a ``.npz`` file."""
    np.savez(
        path,
        loglam=loglam,
        mu=mu,
        Phi=Phi,
        evals=info["evals"],
        var_ratio=info["var_ratio"],
        cumvar=info["cumvar"],
        K=info["K"],
        rectify_order=rectify_order,
        teff=params["teff"],
        logg=params["logg"],
        feh=params["feh"],
    )


def load_basis(path):
    """Load a basis ``.npz`` as a dict of arrays/scalars."""
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}
