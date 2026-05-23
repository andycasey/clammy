"""Task 3: joint RV (+ resolution) + template-weight + continuum fitter (JAX).

Model in log-flux:   ln d(x) ~= G_R * [mu, Phi](x - dx) @ w  +  P(x) @ c
with templates T = [mu, Phi_1..Phi_K], Legendre continuum columns P_0..P_L, a
radial-velocity translation by dx = ln(1+v/c), and an optional Gaussian LSF G_R
of resolving power R.

Separability: at fixed (dx, R) the model is linear in theta = (w, c), so we solve
the linear (normal-equations) problem inside and scan the nonlinear (dx, R)
outside:

    N = A^T W A = [[ M_TT  M_TP ],[ M_TP^T  M_PP ]],   r = A^T W ln d = [b_T ; b_P]
    theta = N^{-1} r,   chi2 = ||ln d||^2_W - r^T N^{-1} r.

W = diag inverse variances of ln d (sigma_lnd ~= sigma_d/d); masked px -> 0.

Speed:
 * b_T(dx) and M_TP(dx) are cross-correlations of fixed vectors against a shifted
   template -> all shifts at once via FFT.  M_PP, b_P are shift/R-independent.
 * A constant resolving power R is a Gaussian of constant width in velocity, i.e.
   constant pixels on the log-lambda grid, so broadening is one Fourier-domain
   multiply  T_hat(f) -> T_hat(f) * exp(-2 pi^2 f^2 sigma_pix^2)  ("on the fly").
   (For a wavelength-dependent R or non-Gaussian LSF there is no closed Fourier
   form; precompute banded convolution matrices on an R-grid and interpolate.)
 * M_TT is recomputed exactly at the refined optimum for the final solve.

Resolution caveat: the LSF convolves *linear* flux, but the inner solve is linear
in (w,c) only if we broaden the log-rectified basis (continuum stays additive).
This log-space convolution matches the exact linear-flux convolution to first
order; it slightly over-deepens saturated line cores.  The toy generator uses the
exact linear-flux convolution, so the validation directly measures any residual
(v, R) bias from this approximation.

Edges: the FFT correlation is circular, so the data weights are zeroed in a band
of width = max lag at both ends, making the circular correlation equal to the
linear one over the valid region for all evaluated shifts.
"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from numpy.polynomial import legendre as npleg

from . import C_LIGHT_KMS, sigma_kms_of_R, R_of_sigma_kms
from .basis import xnorm


def _legendre_design(loglam, order):
    """(n_pix, order+1) Legendre design matrix over the log-lambda range."""
    return npleg.legvander(xnorm(loglam), order)


@partial(jax.jit, static_argnums=())
def _quad_forms(MTP, bT, M_TT0, M_PP, b_P, ridge):
    """Batched r^T N^{-1} r over a stack of shifts (one per row of MTP/bT)."""
    Ktot = M_TT0.shape[0]
    L1 = M_PP.shape[0]
    eye = jnp.eye(Ktot + L1)

    def one(MTP_s, bT_s):
        top = jnp.concatenate([M_TT0, MTP_s], axis=1)
        bot = jnp.concatenate([MTP_s.T, M_PP], axis=1)
        N = jnp.concatenate([top, bot], axis=0) + ridge * eye
        r = jnp.concatenate([bT_s, b_P])
        return r @ jnp.linalg.solve(N, r)

    return jax.vmap(one)(MTP, bT)


def _solve_exact(A, W, lnd, ridge):
    """Exact linear solve at a fixed design A. Returns theta, cov, chi2."""
    N = A.T @ (W[:, None] * A) + ridge * jnp.eye(A.shape[1])
    r = A.T @ (W * lnd)
    cov = jnp.linalg.inv(N)
    theta = cov @ r
    chi2 = jnp.sum(W * lnd * lnd) - r @ theta
    return theta, cov, chi2


@partial(jax.jit, static_argnames=("n", "fit_R"))
def _profiled_chi2(x, Tf, Pj, Wj, lndj, freq, const, ridge, n, fit_R):
    """Profiled chi2 as a function of the nonlinear params x = [p] or [p, sigma_pix].

    The shift p (pixels) enters as a Fourier phase ramp; the broadening sigma_pix
    as a Fourier-domain Gaussian (only when fit_R). The template weights and
    continuum are profiled out by the exact linear solve, so this is the marginal
    surface. Differentiable end-to-end -> JAX supplies grad and Hessian. Big arrays
    are arguments (not closed-over constants) so the jit compiles once and is
    reused across fits of identical shape.
    """
    phase = jnp.exp(-2j * jnp.pi * freq * x[0])
    if fit_R:
        g = jnp.exp(-2.0 * (jnp.pi ** 2) * (freq ** 2) * (x[1] ** 2))  # even in sigma_pix
        Tfb = Tf * g[:, None]
    else:
        Tfb = Tf
    T_sb = jnp.fft.irfft(Tfb * phase[:, None], n=n, axis=0)
    A = jnp.concatenate([T_sb, Pj], axis=1)
    N = A.T @ (Wj[:, None] * A) + ridge * jnp.eye(A.shape[1])
    r = A.T @ (Wj * lndj)
    return const - r @ jnp.linalg.solve(N, r)


@partial(jax.jit, static_argnames=("n", "fit_R", "max_iter", "max_ls"))
def _newton_jit(x0, Tf, Pj, Wj, lndj, freq, const, ridge, n, fit_R,
                max_iter=15, max_ls=25, c1=1e-4, shrink=0.5, ftol=1e-2):
    """Damped-Newton minimiser of the profiled chi2, fully fused in XLA.

    Each step uses the JAX-autodiff gradient and (one) Hessian; the Hessian is
    floored to positive-definite (Levenberg-Marquardt), the step is forced to be a
    descent direction, and its length satisfies the Armijo sufficient-decrease
    condition via a backtracking line search. A `while_loop` stops as soon as the
    chi2 decrease drops below `ftol` -- there is no point refining far below
    Delta-chi2 = 1, and the FFT-based gradient/Hessian are the expensive part, so
    we take as few steps as possible (~3-5 from the coarse-scan start). The whole
    optimiser is one jitted program (no per-iteration host syncs).
    Returns (x_opt, Hessian_at_opt).
    """
    def obj(x):
        return _profiled_chi2(x, Tf, Pj, Wj, lndj, freq, const, ridge, n=n, fit_R=fit_R)

    vg = jax.value_and_grad(obj)
    hess = jax.hessian(obj)
    eye = jnp.eye(x0.size)

    # Fixed-metric ("modified") Newton: the Hessian of a chi2 surface is ~constant
    # over the quadratic basin, so we factor it ONCE at the coarse-scan start and
    # reuse it as the step metric. Near a quadratic minimum H0 ~= H*, so the step
    # is essentially the exact Newton step and converges in ~1-3 iterations -- but
    # each iteration costs only a gradient + line search, not a fresh (expensive)
    # Hessian. The Hessian is recomputed once at the optimum for the covariance.
    H0 = 0.5 * (hess(x0) + hess(x0).T)
    ev0 = jnp.linalg.eigvalsh(H0)
    lm = jnp.maximum(0.0, 1e-8 * jnp.max(jnp.abs(ev0)) - jnp.min(ev0))
    M = H0 + lm * eye

    def cond(state):
        _x, _fp, done, it = state
        return (~done) & (it < max_iter)

    def step(state):
        x, f_prev, _done, it = state
        f0, g = vg(x)
        dx = -jnp.linalg.solve(M, g)
        gtd = g @ dx
        dx = jnp.where(gtd >= 0, -g, dx)        # ensure descent direction
        gtd = g @ dx

        def lcond(st):
            t, f, i = st
            return (f > f0 + c1 * t * gtd) & (i < max_ls)

        def lbody(st):
            t, _f, i = st
            tn = t * shrink
            return tn, obj(x + tn * dx), i + 1

        t, f_new, _ = jax.lax.while_loop(lcond, lbody, (1.0, obj(x + dx), 0))
        return x + t * dx, f_new, (f_prev - f_new) < ftol, it + 1

    x, _, _, _ = jax.lax.while_loop(cond, step, (x0, obj(x0), False, 0))
    H = hess(x)
    return x, 0.5 * (H + H.T)


def fit_rv(
    d,
    sigma,
    loglam,
    mu,
    Phi,
    cont_order=3,
    vmin=-400.0,
    vmax=400.0,
    mask=None,
    ridge=0.0,
    resolution_R=None,
    fit_resolution=False,
    R_bounds=(2000.0, 60000.0),
    n_R=10,
    return_model=True,
):
    """Fit radial velocity (and optionally resolution), template weights, continuum.

    Parameters
    ----------
    d, sigma : (n_pix,)   observed linear flux and 1-sigma noise (on `loglam`)
    loglam   : (n_pix,)   log-lambda grid (must match the basis)
    mu, Phi  : mean template (n_pix,) and PCA basis (K, n_pix)
    cont_order : int      Legendre continuum order L
    vmin, vmax : float    RV search range, km/s
    mask     : (n_pix,) bool | None   True = use pixel
    ridge    : float      Tikhonov ridge on N
    resolution_R : float|None  fix the LSF resolving power R (broaden templates once)
    fit_resolution : bool      jointly fit R alongside v (2-D outer search)
    R_bounds, n_R : R search range and coarse grid size when fit_resolution=True

    Returns
    -------
    dict: v_kms, v_err_kms, w (len K+1; w[0]*mu), c (len L+1), cov (theta cov),
    chi2, dof, chi2_dof, v_grid, chi2_grid, p_star, n_good, and -- when R is fit --
    resolution_R, R_err, sigma_kms, sigma_kms_err, cov_vR (2x2 in (v,R)), rho_vR,
    R_grid, chi2_2d.  With return_model: model flux, residuals.
    """
    d = np.asarray(d, float)
    sigma = np.asarray(sigma, float)
    loglam = np.asarray(loglam, float)
    n = d.size
    dln = float(loglam[1] - loglam[0])
    velscale = C_LIGHT_KMS * dln

    def sigpix_of_R(R):
        return sigma_kms_of_R(R) / velscale

    # --- lag window covering the RV search range -------------------------------
    p_lo = int(np.floor(np.log1p(vmin / C_LIGHT_KMS) / dln))
    p_hi = int(np.ceil(np.log1p(vmax / C_LIGHT_KMS) / dln))
    n_max = max(abs(p_lo), abs(p_hi)) + 4

    # --- weights on ln d, with edge + bad-pixel masking ------------------------
    good = np.isfinite(d) & np.isfinite(sigma) & (d > 0) & (sigma > 0)
    if mask is not None:
        good &= np.asarray(mask, bool)
    good[:n_max] = False
    good[-n_max:] = False
    lnd = np.where(good, np.log(np.where(d > 0, d, 1.0)), 0.0)
    W = np.zeros(n)
    W[good] = (d[good] / sigma[good]) ** 2

    # --- design blocks ----------------------------------------------------------
    T = np.vstack([mu, Phi]).T
    Ktot = T.shape[1]
    P = _legendre_design(loglam, cont_order)
    L1 = P.shape[1]

    Wj, lndj, Tj, Pj = map(jnp.asarray, (W, lnd, T, P))
    M_PP = Pj.T @ (Wj[:, None] * Pj)        # shift/R-independent
    b_P = Pj.T @ (Wj * lndj)                # shift/R-independent
    const = float(jnp.sum(Wj * lndj * lndj))

    Tf = jnp.fft.rfft(Tj, axis=0)           # template FFTs (nf, Ktot)
    S_lnd = jnp.fft.rfft(Wj * lndj)
    S_P = jnp.fft.rfft(Wj[:, None] * Pj, axis=0)
    freq = jnp.fft.rfftfreq(n)

    lags = np.arange(p_lo, p_hi + 1)
    lag_idx = jnp.asarray(lags % n)

    def _gauss(sigma_pix):
        if sigma_pix <= 0:
            return None
        return jnp.exp(-2.0 * (np.pi ** 2) * (freq ** 2) * (sigma_pix ** 2))

    def _Tf_broad(sigma_pix):
        g = _gauss(sigma_pix)
        return Tf if g is None else Tf * g[:, None]

    def _coarse_quad(sigma_pix):
        """chi2(p) - const over the lag grid, at fixed broadening (uses M_TT(0))."""
        Tfb = _Tf_broad(sigma_pix)
        T_u = jnp.fft.irfft(Tfb, n=n, axis=0)
        M_TT0 = T_u.T @ (Wj[:, None] * T_u)
        bT = jnp.take(jnp.fft.irfft(jnp.conj(Tfb) * S_lnd[:, None], n=n, axis=0),
                      lag_idx, axis=0)
        cols = [jnp.take(jnp.fft.irfft(jnp.conj(Tfb) * S_P[:, m][:, None], n=n, axis=0),
                         lag_idx, axis=0) for m in range(L1)]
        MTP = jnp.stack(cols, axis=-1)
        return np.asarray(_quad_forms(MTP, bT, M_TT0, M_PP, b_P, float(ridge)))

    def _exact_at(p, sigma_pix=0.0):
        """Exact profiled chi2 at sub-pixel shift p and broadening sigma_pix."""
        Tfb = _Tf_broad(sigma_pix)
        phase = jnp.exp(-2j * np.pi * freq * float(p))
        T_sb = jnp.fft.irfft(Tfb * phase[:, None], n=n, axis=0)
        A = jnp.concatenate([T_sb, Pj], axis=1)
        theta, cov, chi2 = _solve_exact(A, Wj, lndj, float(ridge))
        return float(chi2), theta, cov, A

    # p (and sigma_pix) refined by the jitted damped-Newton optimiser below.
    const_j, ridge_j = jnp.asarray(const), jnp.asarray(float(ridge))

    def _refine(Tf_use, fit_R, x0):
        x_star, H = _newton_jit(jnp.asarray(x0), Tf_use, Pj, Wj, lndj, freq,
                                const_j, ridge_j, n=n, fit_R=fit_R)
        return np.asarray(x_star), np.asarray(H)

    chi2_2d = None
    resolution_limited = False
    if fit_resolution:
        # coarse 2-D scan over (R, dx) just to find the basin of the minimum
        sp_grid = np.unique(np.concatenate(
            [[0.0], np.linspace(sigpix_of_R(R_bounds[1]), sigpix_of_R(R_bounds[0]), n_R)]))
        chi2_2d = const - np.array([_coarse_quad(sp) for sp in sp_grid])  # (n_sp, S)
        i0, j0 = np.unravel_index(int(np.argmin(chi2_2d)), chi2_2d.shape)

        if i0 == 0:
            # data consistent with no broadening -> R is only a lower limit; fit v.
            resolution_limited = True
            sp_star = 0.0
            x_star, H = _refine(Tf, False, [float(lags[j0])])
            p_star = float(x_star[0])
            cov_ps = np.array([[2.0 / float(H[0, 0]) if H[0, 0] > 0 else np.nan, 0.0],
                               [0.0, np.nan]])
        else:
            x_star, H = _refine(Tf, True, [float(lags[j0]), float(sp_grid[i0])])
            p_star = float(x_star[0])
            sp_star = abs(float(x_star[1]))
            if np.all(np.linalg.eigvalsh(H) > 0):
                cov_ps = 2.0 * np.linalg.inv(H)        # cov in (p, sigma_pix), Delta-chi2=1
                if x_star[1] < 0:  # objective even in sigma -> keep cross-term sign sane
                    cov_ps[0, 1] = -cov_ps[0, 1]
                    cov_ps[1, 0] = -cov_ps[1, 0]
            else:
                cov_ps = np.full((2, 2), np.nan)
    else:
        sp_star = 0.0 if resolution_R is None else sigpix_of_R(resolution_R)
        chi2_grid_v = const - _coarse_quad(sp_star)
        j0 = int(np.argmin(chi2_grid_v))
        Tf_use = Tf if sp_star == 0 else Tf * _gauss(sp_star)[:, None]
        x_star, H = _refine(Tf_use, False, [float(lags[j0])])
        p_star = float(x_star[0])
        cov_ps = np.array([[2.0 / float(H[0, 0]) if H[0, 0] > 0 else np.nan, 0.0],
                           [0.0, 0.0]])

    # --- exact final solve at the optimum --------------------------------------
    dx_star = p_star * dln
    v_star = C_LIGHT_KMS * np.expm1(dx_star)
    chi2_exact, theta, cov, A = _exact_at(p_star, sp_star)
    theta = np.asarray(theta)
    w, c = theta[:Ktot], theta[Ktot:]
    n_good = int(good.sum())
    D = Ktot + L1
    n_nonlin = 2 if fit_resolution else 1
    dof = n_good - D - n_nonlin

    # --- propagate (p, sigma_pix) covariance to (v, R) -------------------------
    dv_dp = C_LIGHT_KMS * np.exp(dx_star) * dln
    v_err = np.sqrt(cov_ps[0, 0]) * dv_dp if np.isfinite(cov_ps[0, 0]) else np.nan
    sigma_kms_star = sp_star * velscale

    out = {
        "v_kms": float(v_star),
        "v_err_kms": float(v_err),
        "w": w,
        "c": c,
        "cov": np.asarray(cov),
        "chi2": float(chi2_exact),
        "dof": dof,
        "chi2_dof": float(chi2_exact) / dof if dof > 0 else np.nan,
        "v_grid": C_LIGHT_KMS * np.expm1(lags * dln),
        "chi2_grid": (chi2_2d[np.argmin(np.min(chi2_2d, axis=1))]
                      if fit_resolution else chi2_grid_v),
        "p_star": float(p_star),
        "n_good": n_good,
        "lag_window": (int(p_lo), int(p_hi)),
        "sigma_kms": float(sigma_kms_star),
        "resolution_R": float(R_of_sigma_kms(sigma_kms_star)) if sigma_kms_star > 0 else np.inf,
    }
    if fit_resolution:
        # Jacobian d(v,R)/d(p,sigma_pix): R = c/(FWHM*sigma_pix*velscale) -> dR/dsig = -R/sigma_pix
        R_star = out["resolution_R"]
        dR_dsp = -R_star / sp_star if sp_star > 0 else np.nan
        J = np.array([[dv_dp, 0.0], [0.0, dR_dsp]])
        cov_vR = J @ cov_ps @ J.T
        sigErr = np.sqrt(cov_ps[1, 1]) if np.isfinite(cov_ps[1, 1]) else np.nan
        out.update({
            "R_err": float(np.sqrt(cov_vR[1, 1])) if np.isfinite(cov_vR[1, 1]) else np.nan,
            "sigma_kms_err": float(sigErr * velscale) if np.isfinite(sigErr) else np.nan,
            "cov_vR": cov_vR,
            "rho_vR": float(cov_vR[0, 1] / np.sqrt(cov_vR[0, 0] * cov_vR[1, 1]))
                      if np.isfinite(cov_vR[0, 0] * cov_vR[1, 1]) and cov_vR[0, 0] * cov_vR[1, 1] > 0
                      else np.nan,
            "R_grid": C_LIGHT_KMS / (velscale * sp_grid[1:] * 2.3548200450309493),
            "chi2_2d": chi2_2d,
            "sp_grid": sp_grid,
            "resolution_limited": resolution_limited,
        })
    if return_model:
        ln_model = np.asarray(A @ jnp.asarray(theta))
        out["ln_model"] = ln_model
        out["model"] = np.exp(ln_model)
        out["lnd"] = lnd
        out["resid_lnd"] = np.where(good, lnd - ln_model, np.nan)
        out["good"] = good
    return out
