"""Task 3: joint RV (+ resolution + vsini) + template-weight + continuum fitter.

Model in log-flux (stellar + tellurics + continuum):

    ln d(x) ~= G_star * [mu, Phi](x - dx) @ w           # stellar, RV-shifted
             + G_tell * [tell_mu, tell_Phi](x) @ u        # tellurics, NOT shifted
             + P(x) @ c                                    # Legendre continuum

with x = ln(lambda), a radial-velocity translation by dx = ln(1 + v/c), a stellar
transfer ``G_star`` (instrument LSF * rotation) and a telluric transfer ``G_tell``
(instrument LSF only). The stellar templates T = [mu, Phi_1..Phi_K] live in the
*rest* frame and are RV-shifted; the telluric templates S = [tell_mu,
tell_Phi_1..tell_Phi_J] live in the *observed* (topocentric) frame and are NOT
shifted -- Earth's atmosphere imprints absorption at fixed observed wavelengths
regardless of the star's motion. So tellurics enter as shift-INDEPENDENT columns,
exactly like the Legendre continuum block P. Continuum stays additive in log-flux.

Fixed block F = [G_tell * S, P]
-------------------------------
The shift-independent part of the design is generalized from "P alone" to
``F = [S_tell_broadened, P]``: the broadened telluric columns sit alongside the
Legendre continuum. F depends only on the telluric instrument broadening (NOT on
dx or vsini), so M_FF = F^T W F and b_F = F^T W ln d are recomputed only when the
telluric broadening changes (i.e. at most once per coarse-grid resolution). When
no telluric basis is supplied, F = P and every code path is identical in result
to the continuum-only implementation.

Separability: at fixed nonlinear params (dx, broadening) the model is linear in
theta = (w, u, c), so we solve the linear (normal-equations) problem inside and
scan the nonlinear params outside:

    N = A^T W A = [[ M_TT  M_TF ],[ M_TF^T  M_FF ]],   r = A^T W ln d = [b_T ; b_F]
    theta = N^{-1} r,   chi2 = ||ln d||^2_W - r^T N^{-1} r.

W = diag inverse variances of ln d (sigma_lnd ~= sigma_d/d); masked px -> 0.

Speed:
 * b_T(dx) and M_TF(dx) are cross-correlations of fixed F columns against a
   shifted stellar template -> all shifts at once via FFT. M_FF, b_F are
   shift/vsini-independent (they change only with the telluric broadening).
 * A constant resolving power R is a Gaussian of constant width in velocity, i.e.
   constant pixels on the log-lambda grid, so instrument broadening is one
   Fourier-domain multiply T_hat(f) -> T_hat(f) * exp(-2 pi^2 f^2 sigma_pix^2).
   Rotation (vsini) is a second Fourier-domain multiply by the rfft of the Gray
   kernel (see ``_g_rot``). Both applied "on the fly" inside the inner solve.
 * M_TT is recomputed exactly at the refined optimum for the final solve.

Differential native-resolution broadening
------------------------------------------
Each basis has a NATIVE resolving power (``R_native_star``, ``R_native_tell``).
To broaden a basis of native resolution R_nat up to a target OBSERVED resolution
with sigma-in-pixels ``sigma_obs_pix``, we apply a Gaussian of DIFFERENTIAL width

    sigma_diff_pix = sqrt( max(0, sigma_obs_pix^2 - sigma_nat_pix^2) ),
    sigma_nat_pix  = sigma_kms_of_R(R_nat) / velscale.

With R_nat = inf, sigma_nat_pix = 0 and sigma_diff_pix = sigma_obs_pix -- EXACTLY
the single-Gaussian behaviour. The stellar and telluric native resolutions may
differ, so the differential Gaussian applied to each block differs even though
the physical instrument resolution ``sigma_obs_pix`` is a single shared parameter.
Negatives are clamped to 0 (one cannot sharpen a basis below its native
resolution; requesting a target finer than native broadens by 0).

Rotational broadening (Gray profile)
------------------------------------
Stellar templates may additionally be convolved by a rotation kernel of projected
rotation velocity vsini and linear limb-darkening coefficient epsilon. The Gray
profile in velocity offset Dv (|Dv| <= vsini, x = Dv/vsini) is

    G(Dv) propto 2 (1 - eps) sqrt(1 - x^2) + (pi eps / 2) (1 - x^2).

We build the kernel directly in real space on the constant integer-pixel offset
array j (zero-phase / fftshift ordering 0,1,...,n/2,...,-2,-1), mask to |v| <=
vsini, normalize to unit SUM, and rfft it once into a real transfer function
``g_rot(freq; vsini_pix, epsilon)`` that multiplies the stellar template FFTs --
just like the Gaussian. The kernel is continuous in vsini, so the transfer
function is differentiable in vsini (verified against a direct real-space
convolution to <1%). Rotation is applied to the STELLAR block only -- never to
tellurics or continuum.

Resolution caveat: the LSF/rotation convolve *linear* flux, but the inner solve
is linear in (w,u,c) only if we broaden the log-rectified basis (continuum stays
additive). This log-space convolution matches the exact linear-flux convolution
to first order; it slightly over-deepens saturated line cores. The toy generator
uses the exact linear-flux convolution, so the validation directly measures any
residual (v, R, vsini) bias from this approximation.

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


def _g_rot(freq, vsini_pix, epsilon, j):
    """Real Fourier-domain transfer function of the Gray rotation kernel.

    Builds the Gray rotation profile in real space on the constant integer-pixel
    offset array ``j`` (zero-phase / fftshift ordering: 0, 1, ..., n/2, ..., -2,
    -1), in velocity units of pixels so that the half-width is ``vsini_pix``:

        x = j / vsini_pix,
        kernel propto 2 (1 - eps) sqrt(1 - x^2) + (pi eps / 2) (1 - x^2),  |x| <= 1

    masked to |j| <= vsini_pix, normalized to unit SUM, then rfft'd to a real
    transfer function (the kernel is zero-phase, so its rfft is real up to tiny
    roundoff -- we take the real part). Continuous and hence autodiff-
    differentiable in ``vsini_pix``: the kernel amplitude is continuous in vsini.
    Multiplying a template FFT by this transfer function applies the rotation
    convolution.

    Autodiff safety: ``sqrt(1 - x^2)`` has an infinite derivative as x -> 1 (the
    kernel edge), which would back-propagate a NaN through the masked-out branch
    of ``jnp.where``. We use the "double where" trick -- the sqrt operand is forced
    to a safe positive constant wherever it would be <= 0 *before* the sqrt, and
    the mask zeroes those entries afterwards -- so both the value and the gradient
    are finite everywhere. When ``vsini_pix <= 0`` (rotation OFF) the function
    returns an all-ones (identity) transfer function and the 1/vsini division is
    guarded, so no NaN leaks into the value or the autodiff at vsini_pix = 0.
    """
    on = vsini_pix > 0
    vs_safe = jnp.where(on, vsini_pix, 1.0)   # avoid 0-division under jit/grad
    x = j / vs_safe
    one_minus_x2 = 1.0 - x * x
    inside = one_minus_x2 > 0.0
    # double-where: sqrt only on strictly-positive operands (the masked-out
    # branch sees a constant 1.0), so grad of sqrt never sees 0 -> no NaN grad.
    safe_o = jnp.where(inside, one_minus_x2, 1.0)
    k = jnp.where(inside,
                  2.0 * (1.0 - epsilon) * jnp.sqrt(safe_o)
                  + (0.5 * jnp.pi * epsilon) * one_minus_x2,
                  0.0)
    k = k / jnp.sum(k)
    tf = jnp.real(jnp.fft.rfft(k))
    return jnp.where(on, tf, jnp.ones_like(tf))


def _diff_sigma(sigma_obs_pix, sigma_nat_pix):
    """Differential broadening sigma to take a native-R basis to sigma_obs_pix.

    sqrt(max(0, sigma_obs_pix^2 - sigma_nat_pix^2)); clamped at 0 (cannot sharpen).
    """
    return jnp.sqrt(jnp.clip(sigma_obs_pix ** 2 - sigma_nat_pix ** 2, 0.0, None))


def _gauss_tf(freq, sigma_pix):
    """Real Gaussian transfer function exp(-2 pi^2 f^2 sigma_pix^2) (even)."""
    return jnp.exp(-2.0 * (jnp.pi ** 2) * (freq ** 2) * (sigma_pix ** 2))


@partial(jax.jit, static_argnums=())
def _quad_forms(MTF, bT, M_TT0, M_FF, b_F, ridge):
    """Batched r^T N^{-1} r over a stack of shifts (one per row of MTF/bT)."""
    Ktot = M_TT0.shape[0]
    Lf = M_FF.shape[0]
    eye = jnp.eye(Ktot + Lf)

    def one(MTF_s, bT_s):
        top = jnp.concatenate([M_TT0, MTF_s], axis=1)
        bot = jnp.concatenate([MTF_s.T, M_FF], axis=1)
        N = jnp.concatenate([top, bot], axis=0) + ridge * eye
        r = jnp.concatenate([bT_s, b_F])
        return r @ jnp.linalg.solve(N, r)

    return jax.vmap(one)(MTF, bT)


def _solve_exact(A, W, lnd, ridge):
    """Exact linear solve at a fixed design A. Returns theta, cov, chi2."""
    N = A.T @ (W[:, None] * A) + ridge * jnp.eye(A.shape[1])
    r = A.T @ (W * lnd)
    cov = jnp.linalg.inv(N)
    theta = cov @ r
    chi2 = jnp.sum(W * lnd * lnd) - r @ theta
    return theta, cov, chi2


@partial(jax.jit, static_argnames=("n", "fit_R", "fit_vsini"))
def _profiled_chi2(x, Tf, Sf, Pj, Wj, lndj, freq, j_off, const, ridge,
                   sig_nat_star, sig_nat_tell, sig_obs_fixed,
                   vsini_fixed, epsilon, n, fit_R, fit_vsini):
    """Profiled chi2 as a function of the variable-length nonlinear vector x.

    The nonlinear vector is ``x = [p]`` followed by ``sigma_obs_pix`` (only if
    ``fit_R``) and ``vsini_pix`` (only if ``fit_vsini``), in that order. Inactive
    broadening params take their fixed values (``sig_obs_fixed`` / ``vsini_fixed``;
    0 means OFF). The RV shift p enters as a Fourier phase ramp.

    Transfer functions are reconstructed each call:
      * stellar:  Gauss(diff_star(sigma_obs_pix)) * g_rot(vsini_pix)
      * telluric: Gauss(diff_tell(sigma_obs_pix))
    where diff_*(.) is the differential native-resolution Gaussian. The design is
    ``A = [stellar_shifted_broadened, telluric_broadened, P]`` (the telluric block
    is absent when ``Sf`` has zero columns). The stellar template weights, telluric
    weights, and continuum are profiled out by the exact linear solve, so this is
    the marginal surface -- differentiable end-to-end so JAX supplies grad and
    Hessian. Big arrays are arguments (not closed-over) so jit compiles once and is
    reused across fits of identical shape.
    """
    idx = 1
    if fit_R:
        sigma_obs_pix = x[idx]
        idx += 1
    else:
        sigma_obs_pix = sig_obs_fixed
    if fit_vsini:
        vsini_pix = x[idx]
        idx += 1
    else:
        vsini_pix = vsini_fixed

    # stellar transfer = instrument differential Gaussian * rotation
    g_star = _gauss_tf(freq, _diff_sigma(sigma_obs_pix, sig_nat_star))
    g_star = g_star * _g_rot(freq, vsini_pix, epsilon, j_off)
    Tfb = Tf * g_star[:, None]

    phase = jnp.exp(-2j * jnp.pi * freq * x[0])
    T_sb = jnp.fft.irfft(Tfb * phase[:, None], n=n, axis=0)

    # telluric block (shift-independent), broadened by the differential Gaussian
    g_tell = _gauss_tf(freq, _diff_sigma(sigma_obs_pix, sig_nat_tell))
    S_b = jnp.fft.irfft(Sf * g_tell[:, None], n=n, axis=0)

    A = jnp.concatenate([T_sb, S_b, Pj], axis=1)
    N = A.T @ (Wj[:, None] * A) + ridge * jnp.eye(A.shape[1])
    r = A.T @ (Wj * lndj)
    return const - r @ jnp.linalg.solve(N, r)


@partial(jax.jit, static_argnames=("n", "fit_R", "fit_vsini", "max_iter", "max_ls"))
def _newton_jit(x0, Tf, Sf, Pj, Wj, lndj, freq, j_off, const, ridge,
                sig_nat_star, sig_nat_tell, sig_obs_fixed, vsini_fixed, epsilon,
                n, fit_R, fit_vsini, max_iter=15, max_ls=25, c1=1e-4,
                shrink=0.5, ftol=1e-2):
    """Damped-Newton minimiser of the profiled chi2, fully fused in XLA.

    Operates on the variable-length nonlinear vector x = [p (, sigma_obs_pix)
    (, vsini_pix)] (slots present per the ``fit_R``/``fit_vsini`` static flags).
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
        return _profiled_chi2(x, Tf, Sf, Pj, Wj, lndj, freq, j_off, const, ridge,
                              sig_nat_star, sig_nat_tell, sig_obs_fixed,
                              vsini_fixed, epsilon, n=n, fit_R=fit_R,
                              fit_vsini=fit_vsini)

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
    R_native_star=np.inf,
    R_native_tell=np.inf,
    tell_basis=None,
    vsini=None,
    fit_vsini=False,
    vsini_bounds=(1.0, 300.0),
    n_vsini=8,
    epsilon=0.6,
    return_model=True,
):
    """Fit radial velocity (+ optional resolution, vsini, tellurics), weights, continuum.

    Parameters
    ----------
    d, sigma : (n_pix,)   observed linear flux and 1-sigma noise (on `loglam`)
    loglam   : (n_pix,)   log-lambda grid (must match the basis)
    mu, Phi  : mean stellar template (n_pix,) and PCA basis (K, n_pix), rest frame
    cont_order : int      Legendre continuum order L
    vmin, vmax : float    RV search range, km/s
    mask     : (n_pix,) bool | None   True = use pixel
    ridge    : float      Tikhonov ridge on N

    resolution_R : float|None  FIXED observed instrument resolving power R (broaden
                               both bases once, differentially). None = instrument
                               broadening OFF unless `fit_resolution`.
    fit_resolution : bool      FIT the observed instrument resolution alongside v.
    R_bounds, n_R : R coarse-grid range/size when `fit_resolution`.
    R_native_star, R_native_tell : float   native resolving power of the stellar
                               and telluric bases (default inf -> no native floor,
                               so the differential Gaussian equals the full sigma).

    tell_basis : (tell_mu, tell_Phi) | None   OPTIONAL telluric basis in the
                               OBSERVED (topocentric) frame, on the SAME log-lambda
                               grid as the data. NOT RV-shifted; gets instrument
                               (not rotational) broadening. Adds telluric weights
                               `u` to the solution. None = no telluric block.

    vsini : float|None         FIXED projected rotation velocity (km/s) for the
                               stellar block. None = no rotation unless `fit_vsini`.
    fit_vsini : bool           FIT vsini alongside v.
    vsini_bounds, n_vsini : vsini coarse-grid range/size when `fit_vsini`.
    epsilon : float            linear limb-darkening coefficient for the Gray
                               rotation profile (default 0.6).

    Modes
    -----
    Instrument resolution: if `fit_resolution` -> a fitted nonlinear param; elif
    `resolution_R is not None` -> fixed sigma_obs_pix = sigma_kms_of_R(resolution_R)
    / velscale; else -> OFF (zero differential Gaussian to BOTH blocks). vsini is
    determined the same way (fit / fixed / off=0). The four named modes:
      1. fit_resolution=True, fit_vsini=True  -> outer vector [p, sigma_obs_pix,
         vsini_pix]; instrument differential to both bases, rotation to stellar.
      2. resolution_R=R0, no fits            -> outer [p]; pre-broaden once at R0.
      3. all off                             -> outer [p]; today's default fit.
      4. fit_vsini=True, instrument off      -> outer [p, vsini_pix]; rotation only.
    Combinations (e.g. fixed resolution_R + fit_vsini) are supported.

    Returns
    -------
    dict, always: v_kms, v_err_kms, w (len K+1; w[0]*mu), c (len L+1), cov (theta
    cov), chi2, dof, chi2_dof, v_grid, chi2_grid, p_star, n_good, lag_window,
    sigma_kms, resolution_R. With return_model: model, ln_model, lnd, resid_lnd,
    good. When R is fit: R_err, sigma_kms_err, cov_vR, rho_vR, R_grid, chi2_2d,
    sp_grid, resolution_limited. When vsini is fit: vsini_kms, vsini_err_kms,
    vsini_limited (True when the optimum is at the lower bound -> lower limit /
    unresolved rotation); a fixed `vsini` is echoed as vsini_kms. When a telluric
    basis is used: u (telluric weights). The joint covariance among the active
    nonlinear params (v and whichever of R/vsini are fit) is `cov_nl`; `cov_vR`
    keeps its 2x2 (v,R) semantics when only R is fit.
    """
    d = np.asarray(d, float)
    sigma = np.asarray(sigma, float)
    loglam = np.asarray(loglam, float)
    n = d.size
    dln = float(loglam[1] - loglam[0])
    velscale = C_LIGHT_KMS * dln

    def sigpix_of_R(R):
        return sigma_kms_of_R(R) / velscale

    # native-resolution sigmas (pixels): 0 when R_native = inf (the default).
    sig_nat_star = float(sigpix_of_R(R_native_star)) if np.isfinite(R_native_star) else 0.0
    sig_nat_tell = float(sigpix_of_R(R_native_tell)) if np.isfinite(R_native_tell) else 0.0

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

    # telluric block S = [tell_mu, tell_Phi]^T (observed frame, NOT shifted).
    use_tell = tell_basis is not None
    if use_tell:
        tell_mu, tell_Phi = tell_basis
        tell_mu = np.asarray(tell_mu, float)
        tell_Phi = np.atleast_2d(np.asarray(tell_Phi, float))
        S = np.vstack([tell_mu, tell_Phi]).T          # (n_pix, J_tell)
    else:
        S = np.zeros((n, 0))
    J_tell = S.shape[1]

    Wj, lndj, Tj, Pj, Sj = map(jnp.asarray, (W, lnd, T, P, S))
    const = float(jnp.sum(Wj * lndj * lndj))

    Tf = jnp.fft.rfft(Tj, axis=0)           # stellar template FFTs (nf, Ktot)
    Sf = jnp.fft.rfft(Sj, axis=0) if J_tell else jnp.zeros((Tf.shape[0], 0),
                                                           dtype=Tf.dtype)
    S_lnd = jnp.fft.rfft(Wj * lndj)
    freq = jnp.fft.rfftfreq(n)
    # integer-pixel offset array for the rotation kernel (zero-phase ordering).
    j_off = jnp.asarray(np.fft.fftfreq(n) * n)

    lags = np.arange(p_lo, p_hi + 1)
    lag_idx = jnp.asarray(lags % n)

    # --- broadening transfer-function helpers (numpy/jax, host side) -----------
    def gauss_np(sigma_pix):
        if sigma_pix <= 0:
            return None
        return jnp.exp(-2.0 * (np.pi ** 2) * (freq ** 2) * (sigma_pix ** 2))

    def diff_star(sigma_obs_pix):
        return float(np.sqrt(max(0.0, sigma_obs_pix ** 2 - sig_nat_star ** 2)))

    def diff_tell(sigma_obs_pix):
        return float(np.sqrt(max(0.0, sigma_obs_pix ** 2 - sig_nat_tell ** 2)))

    def star_tf(sigma_obs_pix, vsini_pix):
        """Stellar transfer = instrument differential Gaussian * rotation."""
        g = gauss_np(diff_star(sigma_obs_pix))
        gT = jnp.ones_like(freq) if g is None else g
        if vsini_pix and vsini_pix > 0:
            gT = gT * _g_rot(freq, float(vsini_pix), float(epsilon), j_off)
        return gT

    def tell_tf(sigma_obs_pix):
        """Telluric transfer = instrument differential Gaussian (no rotation)."""
        g = gauss_np(diff_tell(sigma_obs_pix))
        return jnp.ones_like(freq) if g is None else g

    # --- fixed-block (telluric + continuum) normal-eqn pieces ------------------
    # F = [S_tell_broadened, P]; M_FF, b_F depend only on telluric broadening.
    def fixed_block(sigma_obs_pix):
        if J_tell:
            S_b = jnp.fft.irfft(Sf * tell_tf(sigma_obs_pix)[:, None], n=n, axis=0)
            Fj = jnp.concatenate([S_b, Pj], axis=1)
        else:
            Fj = Pj
        M_FF = Fj.T @ (Wj[:, None] * Fj)
        b_F = Fj.T @ (Wj * lndj)
        S_F = jnp.fft.rfft(Wj[:, None] * Fj, axis=0)   # for cross-corr in coarse scan
        return Fj, M_FF, b_F, S_F

    Lf = J_tell + L1   # width of the fixed block

    def _coarse_quad(sigma_obs_pix, vsini_pix, fixed):
        """chi2(p) - const over the lag grid, at fixed broadening (uses M_TT(0))."""
        _Fj, M_FF, b_F, S_F = fixed
        gT = star_tf(sigma_obs_pix, vsini_pix)
        Tfb = Tf * gT[:, None]
        T_u = jnp.fft.irfft(Tfb, n=n, axis=0)
        M_TT0 = T_u.T @ (Wj[:, None] * T_u)
        bT = jnp.take(jnp.fft.irfft(jnp.conj(Tfb) * S_lnd[:, None], n=n, axis=0),
                      lag_idx, axis=0)
        cols = [jnp.take(jnp.fft.irfft(jnp.conj(Tfb) * S_F[:, m][:, None], n=n, axis=0),
                         lag_idx, axis=0) for m in range(Lf)]
        MTF = jnp.stack(cols, axis=-1)
        return np.asarray(_quad_forms(MTF, bT, M_TT0, M_FF, b_F, float(ridge)))

    def _exact_at(p, sigma_obs_pix, vsini_pix):
        """Exact profiled solve at sub-pixel shift p and given broadening."""
        gT = star_tf(sigma_obs_pix, vsini_pix)
        phase = jnp.exp(-2j * np.pi * freq * float(p))
        T_sb = jnp.fft.irfft(Tf * gT[:, None] * phase[:, None], n=n, axis=0)
        if J_tell:
            S_b = jnp.fft.irfft(Sf * tell_tf(sigma_obs_pix)[:, None], n=n, axis=0)
            A = jnp.concatenate([T_sb, S_b, Pj], axis=1)
        else:
            A = jnp.concatenate([T_sb, Pj], axis=1)
        theta, cov, chi2 = _solve_exact(A, Wj, lndj, float(ridge))
        return float(chi2), theta, cov, A

    # --- decide the nonlinear-parameter layout ---------------------------------
    # sigma_obs_pix: fit / fixed / off(0); vsini_pix: fit / fixed / off(0).
    sig_obs_fixed = 0.0 if (resolution_R is None or fit_resolution) else float(sigpix_of_R(resolution_R))
    vsini_fixed = 0.0 if (vsini is None or fit_vsini) else float(vsini) / velscale

    const_j, ridge_j = jnp.asarray(const), jnp.asarray(float(ridge))
    sig_nat_star_j, sig_nat_tell_j = jnp.asarray(sig_nat_star), jnp.asarray(sig_nat_tell)
    eps_j = jnp.asarray(float(epsilon))

    def _refine(x0, fit_R, fit_vs, sig_obs_fix, vsini_fix):
        x_star, H = _newton_jit(
            jnp.asarray(x0), Tf, Sf, Pj, Wj, lndj, freq, j_off, const_j, ridge_j,
            sig_nat_star_j, sig_nat_tell_j, jnp.asarray(float(sig_obs_fix)),
            jnp.asarray(float(vsini_fix)), eps_j, n=n, fit_R=fit_R, fit_vsini=fit_vs)
        return np.asarray(x_star), np.asarray(H)

    # --- coarse scan over the active broadening params + RV lags ---------------
    sp_grid = np.unique(np.concatenate(
        [[0.0], np.linspace(sigpix_of_R(R_bounds[1]), sigpix_of_R(R_bounds[0]), n_R)])) \
        if fit_resolution else None
    vs_grid = np.unique(np.concatenate(
        [[0.0], np.linspace(vsini_bounds[0] / velscale, vsini_bounds[1] / velscale, n_vsini)])) \
        if fit_vsini else None

    chi2_2d = None              # (n_sp, S) coarse R-surface (back-compat single-R)
    resolution_limited = False
    vsini_limited = False

    # iterate over the broadening grid (Cartesian product of active params).
    sp_vals = sp_grid if fit_resolution else np.array([sig_obs_fixed])
    vs_vals = vs_grid if fit_vsini else np.array([vsini_fixed])

    # fixed-block pieces depend only on the telluric instrument broadening, which
    # depends on sigma_obs_pix; cache per sigma_obs value across the vsini grid.
    fixed_cache = {}

    def get_fixed(sp):
        key = round(float(sp), 12)
        if key not in fixed_cache:
            fixed_cache[key] = fixed_block(sp)
        return fixed_cache[key]

    best = None  # (chi2, i_sp, i_vs, j_lag)
    coarse_cube = np.empty((len(sp_vals), len(vs_vals), lags.size))
    for ia, sp in enumerate(sp_vals):
        fixed = get_fixed(sp)
        for ib, vsp in enumerate(vs_vals):
            row = const - _coarse_quad(sp, vsp, fixed)   # (S,)
            coarse_cube[ia, ib] = row
            jbest = int(np.argmin(row))
            if best is None or row[jbest] < best[0]:
                best = (row[jbest], ia, ib, jbest)

    _, i_sp, i_vs, j0 = best

    # detect lower-limit (unresolved) situations at the coarse-grid minimum.
    if fit_resolution and i_sp == 0:
        resolution_limited = True
    if fit_vsini and i_vs == 0:
        vsini_limited = True

    # --- build the refinement starting vector & static flags -------------------
    # When the coarse minimum sits at the zero (off) edge of an active param, we
    # drop that param from the refinement (report it as a lower limit) -- exactly
    # the existing resolution_limited behaviour, generalized to vsini too.
    refine_R = fit_resolution and not resolution_limited
    refine_vs = fit_vsini and not vsini_limited

    x0 = [float(lags[j0])]
    if refine_R:
        x0.append(float(sp_vals[i_sp]))
    if refine_vs:
        x0.append(float(vs_vals[i_vs]))
    sig_obs_fix_use = sig_obs_fixed if not refine_R else 0.0
    vsini_fix_use = vsini_fixed if not refine_vs else 0.0
    x_star, H = _refine(x0, refine_R, refine_vs, sig_obs_fix_use, vsini_fix_use)

    # --- unpack the refined nonlinear params -----------------------------------
    p_star = float(x_star[0])
    idx = 1
    if refine_R:
        sp_star = abs(float(x_star[idx])); idx += 1
    elif fit_resolution:
        sp_star = 0.0
    else:
        sp_star = sig_obs_fixed
    if refine_vs:
        vsp_star = abs(float(x_star[idx])); idx += 1
    elif fit_vsini:
        vsp_star = 0.0
    else:
        vsp_star = vsini_fixed

    # --- exact final solve at the optimum --------------------------------------
    dx_star = p_star * dln
    v_star = C_LIGHT_KMS * np.expm1(dx_star)
    chi2_exact, theta, cov, A = _exact_at(p_star, sp_star, vsp_star)
    theta = np.asarray(theta)
    w = theta[:Ktot]
    u = theta[Ktot:Ktot + J_tell]
    c = theta[Ktot + J_tell:]
    n_good = int(good.sum())
    D = Ktot + J_tell + L1
    # Count every REQUESTED nonlinear dimension (v, plus R/vsini if those are
    # fit), not just the ones actually refined -- a search costs a degree of
    # freedom even when its optimum lands on a bound (lower limit). This matches
    # the previous single-broadening convention (n_nonlin = 2 when fit_resolution).
    n_nonlin = 1 + int(fit_resolution) + int(fit_vsini)
    dof = n_good - D - n_nonlin

    # --- covariance of the active nonlinear params (Delta-chi2 = 1) ------------
    # H is over x_star = [p (, sigma_obs_pix) (, vsini_pix)]; invert to get cov.
    if np.all(np.linalg.eigvalsh(H) > 0):
        cov_x = 2.0 * np.linalg.inv(H)
        # objective is even in each broadening param -> keep cross-term signs sane
        for k in range(1, x_star.size):
            if x_star[k] < 0:
                cov_x[0, k] = -cov_x[0, k]; cov_x[k, 0] = -cov_x[k, 0]
    else:
        cov_x = np.full((x_star.size, x_star.size), np.nan)

    dv_dp = C_LIGHT_KMS * np.exp(dx_star) * dln
    v_err = np.sqrt(cov_x[0, 0]) * dv_dp if np.isfinite(cov_x[0, 0]) else np.nan
    sigma_kms_star = sp_star * velscale
    vsini_kms_star = vsp_star * velscale

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
        "chi2_grid": coarse_cube[i_sp, i_vs],
        "p_star": float(p_star),
        "n_good": n_good,
        "lag_window": (int(p_lo), int(p_hi)),
        "sigma_kms": float(sigma_kms_star),
        "resolution_R": float(R_of_sigma_kms(sigma_kms_star)) if sigma_kms_star > 0 else np.inf,
    }

    # --- build the (p, sigma_obs, vsini) -> (v, R, vsini) Jacobian + cov --------
    # column order of cov_x matches x_star = [p (, sigma_obs_pix) (, vsini_pix)].
    nl_labels = ["v"]
    col = 1
    sp_col = vs_col = None
    if refine_R:
        sp_col = col; col += 1; nl_labels.append("R")
    if refine_vs:
        vs_col = col; col += 1; nl_labels.append("vsini")

    # full joint covariance among the active nonlinear params, in (v, R, vsini).
    m = x_star.size
    J = np.zeros((m, m))
    J[0, 0] = dv_dp
    if sp_col is not None:
        R_star = out["resolution_R"]
        J[sp_col, sp_col] = (-R_star / sp_star) if sp_star > 0 else np.nan
    if vs_col is not None:
        J[vs_col, vs_col] = velscale  # vsini_kms = vsini_pix * velscale
    cov_nl = J @ cov_x @ J.T
    out["cov_nl"] = cov_nl
    out["nl_labels"] = nl_labels

    if use_tell:
        out["u"] = u

    if fit_resolution:
        # back-compat single-R reporting (cov_vR is the 2x2 (v,R) block).
        sigErr = np.sqrt(cov_x[sp_col, sp_col]) if (sp_col is not None and
                  np.isfinite(cov_x[sp_col, sp_col])) else np.nan
        # extract a 2x2 (v,R) covariance from cov_nl (zeros if R is a lower limit).
        cov_vR = np.full((2, 2), np.nan)
        cov_vR[0, 0] = cov_nl[0, 0]
        if sp_col is not None:
            cov_vR[1, 1] = cov_nl[sp_col, sp_col]
            cov_vR[0, 1] = cov_nl[0, sp_col]
            cov_vR[1, 0] = cov_nl[sp_col, 0]
        else:
            cov_vR[1, 1] = np.nan
        out.update({
            "R_err": float(np.sqrt(cov_vR[1, 1])) if np.isfinite(cov_vR[1, 1]) else np.nan,
            "sigma_kms_err": float(sigErr * velscale) if np.isfinite(sigErr) else np.nan,
            "cov_vR": cov_vR,
            "rho_vR": float(cov_vR[0, 1] / np.sqrt(cov_vR[0, 0] * cov_vR[1, 1]))
                      if np.isfinite(cov_vR[0, 0] * cov_vR[1, 1]) and cov_vR[0, 0] * cov_vR[1, 1] > 0
                      else np.nan,
            "R_grid": C_LIGHT_KMS / (velscale * sp_grid[1:] * 2.3548200450309493),
            "chi2_2d": coarse_cube[:, i_vs, :],  # (n_sp, S) at the best vsini slice
            "sp_grid": sp_grid,
            "resolution_limited": resolution_limited,
        })

    if fit_vsini:
        vsErr = np.sqrt(cov_x[vs_col, vs_col]) if (vs_col is not None and
                 np.isfinite(cov_x[vs_col, vs_col])) else np.nan
        out.update({
            "vsini_kms": float(vsini_kms_star),
            "vsini_err_kms": float(vsErr * velscale) if np.isfinite(vsErr) else np.nan,
            "vsini_limited": vsini_limited,
            "vsini_grid": vs_grid * velscale,
        })
    elif vsini is not None:
        # echo the fixed vsini that was applied to the stellar block.
        out["vsini_kms"] = float(vsini)

    if return_model:
        ln_model = np.asarray(A @ jnp.asarray(theta))
        out["ln_model"] = ln_model
        out["model"] = np.exp(ln_model)
        out["lnd"] = lnd
        out["resid_lnd"] = np.where(good, lnd - ln_model, np.nan)
        out["good"] = good
    return out
