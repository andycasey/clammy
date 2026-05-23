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

Frames
------
The same rectify+PCA machinery serves two physically distinct kinds of basis,
distinguished only by the ``frame`` recorded alongside them:

* a STELLAR basis (``frame="rest"``) lives in the stellar *rest* frame and is
  RV-shifted by the fitter to match the observed spectrum;
* a TELLURIC basis (``frame="observed"``) lives in the *observed* (topocentric)
  frame -- Earth's atmosphere imprints absorption at fixed observed wavelengths
  regardless of the star's motion -- so it must NOT be RV-shifted.

We only build and store the basis here; the fitter reads the ``frame`` metadata
to decide whether to apply the RV shift. ``build_telluric_basis`` is the
documented entry point for the telluric case.

npz key contract
----------------
Every saved basis carries the common keys::

    loglam, mu, Phi, evals, var_ratio, cumvar, K, rectify_order, frame, resolution

A stellar basis additionally stores ``teff, logg, feh``. ``resolution`` is the
native resolving power R = lambda / delta-lambda of the model grid the basis was
built from (assumed constant, i.e. a constant velocity width on the log-lambda
grid); ``np.inf`` is the sentinel meaning "effectively unresolved / infinite
native resolution" and reproduces the historical full-broadening behaviour. The
fitter uses it to broaden the template differentially from its native resolution
up to the observed instrument resolution in quadrature
(``sigma_diff^2 = max(0, sigma_obs^2 - sigma_native^2)``). The stellar and
telluric grids may have different native resolutions, so each basis carries its
own.
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


def build_nmf(R, K, max_iter=500, tol=1e-4, random_state=0):
    """NMF of the rectified spectra, operated on τ = −R ≥ 0.

    Returns the same ``(mu, Phi, info)`` signature as :func:`build_pca` so
    the result can be handed directly to :func:`save_basis` and the fitter.
    ``Phi`` rows are the NMF components negated back to log-flux space
    (absorption templates, ≤ 0 in lines, matching the PCA convention).

    Initialised via NNDSVD (truncated SVD of τ), then refined with Lee &
    Seung multiplicative updates until the relative Frobenius-norm change
    falls below ``tol`` (checked every 10 iterations) or ``max_iter`` is
    reached.

    Parameters
    ----------
    R : (n_spec, n_pix) ndarray
        Rectified log-flux spectra (≤ 0 in absorption lines).
    K : int
        Number of NMF components (exact, unlike the variance-target of PCA).
    max_iter : int
        Maximum multiplicative-update iterations.
    tol : float
        Relative Frobenius-norm convergence tolerance.
    random_state : int
        Seed for reproducible random initialisation fallback.

    Returns
    -------
    mu     : (n_pix,)    mean rectified spectrum
    Phi    : (K, n_pix)  unit-norm NMF absorption templates in log-flux space
    info   : dict        'K', 'evals' (zeros), 'var_ratio', 'cumvar'
    """
    eps = 1e-10
    rng = np.random.default_rng(random_state)

    mu = R.mean(axis=0)
    tau = np.maximum(-R, 0.0)   # non-negative optical depth, (n_spec, n_pix)
    n_spec, n_pix = tau.shape
    K = max(1, min(K, n_spec, n_pix))

    # NNDSVD initialisation (Boutsidis & Gallopoulos 2008)
    try:
        from scipy.sparse.linalg import svds
        k_svd = min(K, min(n_spec, n_pix) - 1)
        v0 = rng.standard_normal(min(n_spec, n_pix))
        U, S, Vt = svds(tau, k=k_svd, v0=v0)
        # svds returns ascending order; flip to descending
        U, S, Vt = U[:, ::-1], S[::-1], Vt[::-1, :]
        W = np.empty((n_spec, K))
        H = np.empty((K, n_pix))
        for k in range(K):
            if k == 0:
                W[:, k] = np.sqrt(S[0]) * np.abs(U[:, 0])
                H[k, :] = np.sqrt(S[0]) * np.abs(Vt[0, :])
            else:
                up = np.maximum( U[:, k], 0.0); um = np.maximum(-U[:, k], 0.0)
                vp = np.maximum( Vt[k, :], 0.0); vm = np.maximum(-Vt[k, :], 0.0)
                np_val = np.linalg.norm(up) * np.linalg.norm(vp)
                nm_val = np.linalg.norm(um) * np.linalg.norm(vm)
                if np_val >= nm_val:
                    nu = np.linalg.norm(up) + eps; nv = np.linalg.norm(vp) + eps
                    f = np.sqrt(S[k] * np_val)
                    W[:, k] = f * up / nu; H[k, :] = f * vp / nv
                else:
                    nu = np.linalg.norm(um) + eps; nv = np.linalg.norm(vm) + eps
                    f = np.sqrt(S[k] * nm_val)
                    W[:, k] = f * um / nu; H[k, :] = f * vm / nv
        W = np.maximum(W, eps)
        H = np.maximum(H, eps)
    except Exception:
        W = rng.uniform(0.0, 1.0, (n_spec, K))
        H = rng.uniform(0.0, 1.0, (K, n_pix))

    # Lee & Seung multiplicative updates
    prev_loss = np.inf
    for it in range(max_iter):
        WH = W @ H
        H *= (W.T @ tau) / (W.T @ WH + eps)
        np.maximum(H, eps, out=H)
        WH = W @ H
        W *= (tau @ H.T) / (WH @ H.T + eps)
        np.maximum(W, eps, out=W)
        if (it + 1) % 10 == 0:
            loss = np.linalg.norm(tau - W @ H, "fro") ** 2
            converged = prev_loss < np.inf and abs(prev_loss - loss) / (prev_loss + eps) < tol
            prev_loss = loss
            if converged:
                print(f"  NMF converged at iteration {it + 1}, loss={loss:.4e}")
                break

    # Normalise H rows to unit norm (mirrors PCA convention)
    norms = np.linalg.norm(H, axis=1)
    norms[norms == 0] = 1.0
    Phi = -(H / norms[:, None])  # log-flux absorption templates, ≤ 0 in lines

    # Variance metrics: total fraction of τ² explained, split evenly
    total_var = float(np.sum(tau ** 2))
    recon_err = float(np.linalg.norm(tau - W @ H, "fro") ** 2)
    frac = max(0.0, 1.0 - recon_err / (total_var + eps))
    var_ratio = np.full(K, frac / K)
    cumvar = np.cumsum(var_ratio)

    info = {"K": K, "evals": np.zeros(K), "var_ratio": var_ratio, "cumvar": cumvar}
    return mu, Phi, info


def build_telluric_nmf(loglam, flux, K, order=5, **kw):
    """Thin wrapper: rectify ``flux`` then run :func:`build_nmf`.

    Mirrors :func:`build_telluric_basis` but uses NMF instead of PCA.
    Pass ``frame="observed"`` and ``params=None`` to :func:`save_basis`.
    """
    R, _ = rectify_grid(loglam, flux, order=order)
    return build_nmf(R, K, **kw)


def build_telluric_basis(loglam, flux, order=5, var_target=0.99, max_k=None, **clip_kw):
    """Build a telluric template basis from a grid of telluric models.

    This is a thin convenience wrapper that runs the *identical* rectify+PCA
    pipeline used for the stellar basis (:func:`rectify_grid` then
    :func:`build_pca`) -- the math does not care what produced the spectra. The
    one thing that differs is *interpretation*:

        The returned basis is meant to be applied in the OBSERVED (topocentric)
        frame, with NO radial-velocity shift.

    Telluric absorption is imprinted by Earth's atmosphere at fixed observed
    wavelengths, independent of the star's motion, so unlike the stellar basis
    (which lives in the rest frame and is RV-shifted to match the data) the
    telluric basis is held fixed in the observed frame. Persist this by passing
    ``frame="observed"`` (and ``params=None``) to :func:`save_basis`; the fitter
    reads that ``frame`` flag to skip the RV shift.

    Parameters
    ----------
    loglam : (n_pix,) ndarray
        Shared natural-log wavelength grid.
    flux : (n_spec, n_pix) ndarray
        Strictly positive telluric model flux (e.g. from
        :func:`clammy.grid.load_spectra`).
    order : int
        Legendre order for the robust log-space continuum rectification.
    var_target : float
        Cumulative-variance target for the PCA truncation.
    max_k : int | None
        Optional cap on the number of retained components.
    **clip_kw
        Forwarded to :func:`fit_log_continuum` (``niter``, ``low``, ``high``).

    Returns
    -------
    mu   : (n_pix,)        mean rectified spectrum
    Phi  : (K, n_pix)      unit-norm principal directions (templates)
    info : dict            'evals', 'var_ratio', 'cumvar', 'K'

    Notes
    -----
    The return signature mirrors the stellar path exactly, so a caller can hand
    ``(mu, Phi, info)`` straight to :func:`save_basis` (with
    ``params=None, frame="observed"``).
    """
    R, _ = rectify_grid(loglam, flux, order=order, **clip_kw)
    return build_pca(R, var_target=var_target, max_k=max_k)


def save_basis(path, loglam, mu, Phi, params, info, rectify_order, frame="rest",
               resolution=np.inf, method="pca"):
    """Save the basis to a ``.npz`` file.

    The archive uses the npz key contract documented in the module docstring:
    the common keys ``loglam, mu, Phi, evals, var_ratio, cumvar, K,
    rectify_order, frame, resolution`` are always written, plus ``teff, logg,
    feh`` for a stellar basis.

    Parameters
    ----------
    path : str
        Output ``.npz`` path.
    loglam, mu, Phi : ndarray
        Shared log-lambda grid, mean spectrum, and principal components.
    params : dict | None
        Stellar labels ``{'teff', 'logg', 'feh'}``. For an unlabelled
        (telluric) basis pass ``None``; if ``None`` (or missing those keys) the
        label arrays are simply omitted from the archive.
    info : dict
        Output of :func:`build_pca` (``'evals'``, ``'var_ratio'``,
        ``'cumvar'``, ``'K'``).
    rectify_order : int
        Legendre order used for rectification.
    frame : str
        Reference frame the basis lives in: ``"rest"`` for a stellar basis
        (RV-shifted by the fitter) or ``"observed"`` for a telluric basis (held
        fixed in the topocentric frame, never RV-shifted). Always stored.
    resolution : float
        Native resolving power R = lambda / delta-lambda of the model grid this
        basis was built from (assumed constant -- a constant R is a constant
        velocity width on the log-lambda grid, which the fitter relies on). This
        is a build-time *input* (the CLI passes it through), not something
        derived from the grid here. Always stored as a float scalar. The default
        ``np.inf`` is the sentinel for "effectively unresolved / infinite native
        resolution" and reproduces the historical full-broadening behaviour
        (e.g. raw PHOENIX HiRes, far higher resolution than any observed
        instrument).
    """
    arrays = dict(
        loglam=loglam,
        mu=mu,
        Phi=Phi,
        evals=info["evals"],
        var_ratio=info["var_ratio"],
        cumvar=info["cumvar"],
        K=info["K"],
        rectify_order=rectify_order,
        frame=frame,
        resolution=float(resolution),
        method=method,
    )
    # Stellar labels are optional: present for a stellar basis, absent for a
    # telluric one. Only store them if all three are available.
    if params is not None and all(k in params for k in ("teff", "logg", "feh")):
        arrays.update(teff=params["teff"], logg=params["logg"], feh=params["feh"])
    np.savez(path, **arrays)


def load_basis(path):
    """Load a basis ``.npz`` as a dict of arrays/scalars.

    Returns the npz key contract documented in the module docstring. Backward
    compatible with bases written before the ``frame`` and ``resolution`` keys
    existed:

    * a missing ``frame`` defaults to ``"rest"`` (pre-frame bases are stellar);
    * a missing ``resolution`` defaults to ``np.inf`` (the historical
      full-broadening behaviour).

    The ``frame`` value is returned as a clean python ``str`` and ``resolution``
    as a clean python ``float`` (np.savez stores these as 0-d arrays / np
    scalars) so callers can compare/use them directly.
    """
    with np.load(path, allow_pickle=False) as d:
        out = {k: d[k] for k in d.files}
    if "frame" in out:
        out["frame"] = str(np.asarray(out["frame"]).item())
    else:
        out["frame"] = "rest"
    if "resolution" in out:
        out["resolution"] = float(np.asarray(out["resolution"]).item())
    else:
        out["resolution"] = float(np.inf)
    if "method" in out:
        out["method"] = str(np.asarray(out["method"]).item())
    else:
        out["method"] = "pca"
    return out
