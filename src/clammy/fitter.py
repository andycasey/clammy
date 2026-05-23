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


@partial(jax.jit, static_argnames=("n", "fit_R", "fit_vsini", "fit_tell"))
def _profiled_chi2(x, Tf, Sf, Pj, Wj, lndj, freq, j_off, const, ridge,
                   sig_nat_star, sig_nat_tell, sig_obs_fixed,
                   vsini_fixed, epsilon, n, fit_R, fit_vsini, fit_tell):
    """Profiled chi2 as a function of the variable-length nonlinear vector x.

    The nonlinear vector is ``x = [p]`` followed by ``sigma_obs_pix`` (only if
    ``fit_R``), ``vsini_pix`` (only if ``fit_vsini``), and ``p_tell`` (only if
    ``fit_tell``), in that order. Inactive broadening params take their fixed
    values (``sig_obs_fixed`` / ``vsini_fixed``; 0 means OFF). The stellar RV
    shift p enters as a Fourier phase ramp; the telluric lag p_tell (when active)
    enters as a SEPARATE phase ramp on the telluric block.

    Transfer functions are reconstructed each call:
      * stellar:  Gauss(diff_star(sigma_obs_pix)) * g_rot(vsini_pix)
      * telluric: Gauss(diff_tell(sigma_obs_pix))
    where diff_*(.) is the differential native-resolution Gaussian. The design is
    ``A = [stellar_shifted_broadened, telluric_broadened(, shifted), P]`` (the
    telluric block is absent when ``Sf`` has zero columns). The telluric block is
    instrument-broadened only (NO rotation) and, when ``fit_tell``, additionally
    shifted by p_tell via ``exp(-2 pi i freq p_tell)``. The stellar template
    weights, telluric weights, and continuum are profiled out by the exact linear
    solve, so this is the marginal surface -- differentiable end-to-end so JAX
    supplies grad and Hessian. Big arrays are arguments (not closed-over) so jit
    compiles once and is reused across fits of identical shape.
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

    # telluric block, broadened by the differential Gaussian (no rotation), and
    # optionally shifted by the telluric lag p_tell (the instrument wavelength
    # zero-point) via a second Fourier phase ramp.
    Sfb = Sf * _gauss_tf(freq, _diff_sigma(sigma_obs_pix, sig_nat_tell))[:, None]
    if fit_tell:
        tell_phase = jnp.exp(-2j * jnp.pi * freq * x[idx])
        Sfb = Sfb * tell_phase[:, None]
    S_b = jnp.fft.irfft(Sfb, n=n, axis=0)

    A = jnp.concatenate([T_sb, S_b, Pj], axis=1)
    N = A.T @ (Wj[:, None] * A) + ridge * jnp.eye(A.shape[1])
    r = A.T @ (Wj * lndj)
    return const - r @ jnp.linalg.solve(N, r)


@partial(jax.jit, static_argnames=("n", "fit_R", "fit_vsini", "fit_tell", "max_iter", "max_ls"))
def _newton_jit(x0, lo, hi, Tf, Sf, Pj, Wj, lndj, freq, j_off, const, ridge,
                sig_nat_star, sig_nat_tell, sig_obs_fixed, vsini_fixed, epsilon,
                n, fit_R, fit_vsini, fit_tell, max_iter=15, max_ls=25, c1=1e-4,
                shrink=0.5, ftol=1e-2):
    """Damped-Newton minimiser of the profiled chi2, fully fused in XLA.

    Operates on the variable-length nonlinear vector x = [p (, sigma_obs_pix)
    (, vsini_pix) (, p_tell)] (slots present per the ``fit_R``/``fit_vsini``/
    ``fit_tell`` static flags). Each step uses the JAX-autodiff gradient and (one)
    Hessian; the Hessian is floored to positive-definite (Levenberg-Marquardt),
    the step is forced to be a descent direction, and its length satisfies the
    Armijo sufficient-decrease condition via a backtracking line search. A
    `while_loop` stops as soon as the chi2 decrease drops below `ftol` -- there is
    no point refining far below Delta-chi2 = 1, and the FFT-based gradient/Hessian
    are the expensive part, so we take as few steps as possible (~3-5 from the
    coarse-scan start). The whole optimiser is one jitted program (no
    per-iteration host syncs). Returns (x_opt, Hessian_at_opt).
    """
    def obj(x):
        return _profiled_chi2(x, Tf, Sf, Pj, Wj, lndj, freq, j_off, const, ridge,
                              sig_nat_star, sig_nat_tell, sig_obs_fixed,
                              vsini_fixed, epsilon, n=n, fit_R=fit_R,
                              fit_vsini=fit_vsini, fit_tell=fit_tell)

    vg = jax.value_and_grad(obj)
    hess = jax.hessian(obj)
    eye = jnp.eye(x0.size)
    # box-constrain to the requested bounds: lo/hi are +/-inf for the unbounded
    # parameters (RV lag, telluric lag) and the R/vsini limits for the broadening
    # ones. Every trial point in the line search is projected back into the box, so
    # the refined value can never drift outside the bounds the caller gave.
    x0 = jnp.clip(x0, lo, hi)

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
            return tn, obj(jnp.clip(x + tn * dx, lo, hi)), i + 1

        t, f_new, _ = jax.lax.while_loop(
            lcond, lbody, (1.0, obj(jnp.clip(x + dx, lo, hi)), 0))
        return jnp.clip(x + t * dx, lo, hi), f_new, (f_prev - f_new) < ftol, it + 1

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
    fit_telluric_shift=False,
    vsini=None,
    fit_vsini=False,
    vsini_bounds=(1.0, 300.0),
    n_vsini=8,
    epsilon=0.6,
    return_model=True,
    rescale_errors=False,
    n_conv_iter=0,
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
    fit_telluric_shift : bool  OPTIONAL. When True (requires `tell_basis`), fit a
                               telluric velocity lag `p_tell` jointly with the
                               stellar RV. Tellurics are at rest in the observer
                               frame, so a nonzero telluric velocity IS the
                               instrument wavelength zero-point (e.g. DEIMOS slit
                               flexure). The stellar RV inherits the same
                               zero-point, so the corrected stellar velocity comes
                               from the lag DIFFERENCE (p_star - p_tell). The
                               coarse scan keeps the telluric at p_tell=0 (in the
                               fixed block); p_tell is initialised to 0 and refined
                               jointly by the damped-Newton step. The telluric
                               block is then shifted by `exp(-2 pi i freq p_tell)`,
                               instrument-broadened only (no rotation). Reports
                               `v_tell_kms`, `v_corr_kms` (and their errors).

    vsini : float|None         FIXED projected rotation velocity (km/s) for the
                               stellar block. None = no rotation unless `fit_vsini`.
    fit_vsini : bool           FIT vsini alongside v.
    vsini_bounds, n_vsini : vsini coarse-grid range/size when `fit_vsini`.
    epsilon : float            linear limb-darkening coefficient for the Gray
                               rotation profile (default 0.6).

    n_conv_iter : int          OPTIONAL. Iterations of the linear-flux convolution
                               correction (default 0 -> OFF, byte-identical to the
                               historical log-space-broadened fit). See the
                               "Iterated linear-flux convolution correction" note
                               below. With n_conv_iter >= 1 the model is broadened
                               in LINEAR flux (the physically correct space for the
                               instrument LSF), which removes the deep-core misfit
                               and the ~few-percent R bias of the log-space
                               approximation.

    Iterated linear-flux convolution correction (``n_conv_iter``)
    -------------------------------------------------------------
    The inner solve is linear in (w, u, c) only because we broaden the LOG-rectified
    basis -- a Fourier multiply on ``ln``-flux. The instrument LSF, however,
    physically convolves the LINEAR flux. Writing the per-component log-model as the
    log-space-broadened block

        A_c = G_c (x)_log m_c        (Fourier multiply of the transfer G_c on ln-flux)

    where ``m_c`` is the UNBROADENED (RV/lag-shifted) component log-flux, the EXACT
    (linear-flux-broadened) block is

        E_c = ln( G_c (x) exp(m_c) )  (convolve the LINEAR flux, then take the log).

    The two agree to first order; the dropped second-order term ``delta_c = E_c - A_c``
    is the ``+1/2 Var_LSF`` curvature term and peaks sharply at deep, saturated line
    cores (where it reaches many sigma). Continuum is smooth, so it needs no
    correction; ``delta = delta_star + delta_tell``.

    Because ``E = A + delta``, fitting the existing log-space machinery to the
    CORRECTED data ``lnd_eff = lnd - delta`` makes the exact model ``E`` match the
    data. We iterate a fixed point:

      iter 0 : lnd_eff = lnd (delta = 0) -> the historical fit (coarse scan + Newton
               + exact solve) gives the nonlinear params and theta = (w, u, c).
      iter k : recompute delta from the current solution (the small correction does
               not move the basin), set lnd_eff = lnd - delta, WARM-START the Newton
               from the previous optimum (SKIP the coarse scan), re-solve exactly,
               recompute delta. Stop after ``n_conv_iter`` iters (or earlier if delta
               stops changing). Each extra iter costs a warm Newton refine + a couple
               of FFTs -- no new coarse scan.

    After convergence the EXACT model ``exp(E) = exp(A + delta)`` is reported, ``R``
    is fit against the corrected data (so its log-space bias vanishes), and the
    block decomposition (``ln_star``/``ln_tell``/``ln_cont``) holds the EXACT
    broadened blocks ``E_star``/``E_tell``/``cont`` so it still sums to ``ln_model``.

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
    basis is used: u (telluric weights). When `fit_telluric_shift`: v_tell_kms,
    v_tell_err_kms (the telluric velocity = instrument wavelength zero-point) and
    v_corr_kms, v_corr_err_kms (the stellar RV with that zero-point removed, from
    the lag difference p_star - p_tell). v_kms always remains the RAW stellar shift
    (observed frame). The joint covariance among the active
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

    # telluric block S = [tell_mu, tell_Phi]^T (observed frame). NOT shifted unless
    # fit_telluric_shift -> then it carries its own (small) wavelength-zero-point lag.
    use_tell = tell_basis is not None
    if fit_telluric_shift and not use_tell:
        raise ValueError(
            "fit_telluric_shift=True requires a telluric basis (tell_basis); "
            "got tell_basis=None."
        )
    fit_tell = bool(fit_telluric_shift and use_tell)
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

    def _exact_at(p, sigma_obs_pix, vsini_pix, p_tell=0.0, lnd_arg=lndj):
        """Exact profiled solve at sub-pixel shift p and given broadening.

        When the telluric block is active it is broadened and (if a telluric lag
        ``p_tell`` is supplied) shifted by ``exp(-2 pi i freq p_tell)`` -- matching
        the shift the profiled chi2 / Newton refine applies. ``lnd_arg`` defaults to
        the raw data; the linear-flux convolution loop passes ``lnd_eff = lnd -
        delta`` so the exact solve (and its chi2) are taken against the corrected
        log-data.
        """
        gT = star_tf(sigma_obs_pix, vsini_pix)
        phase = jnp.exp(-2j * np.pi * freq * float(p))
        T_sb = jnp.fft.irfft(Tf * gT[:, None] * phase[:, None], n=n, axis=0)
        if J_tell:
            Sfb = Sf * tell_tf(sigma_obs_pix)[:, None]
            if p_tell:
                tell_phase = jnp.exp(-2j * np.pi * freq * float(p_tell))
                Sfb = Sfb * tell_phase[:, None]
            S_b = jnp.fft.irfft(Sfb, n=n, axis=0)
            A = jnp.concatenate([T_sb, S_b, Pj], axis=1)
        else:
            A = jnp.concatenate([T_sb, Pj], axis=1)
        theta, cov, chi2 = _solve_exact(A, Wj, lnd_arg, float(ridge))
        return float(chi2), theta, cov, A

    def _delta(theta, p, sigma_obs_pix, vsini_pix, p_tell=0.0):
        """Per-component linear-flux convolution correction delta = E - A.

        For each broadened component c, ``A_c = G_c (x)_log m_c`` is the LOG-space
        broadened block the design already uses, and ``E_c = ln(G_c (x) exp(m_c))``
        is the EXACT block obtained by convolving the LINEAR flux. With the SAME
        transfer functions the fitter builds (``star_tf`` = instrument differential
        Gaussian * rotation; ``tell_tf`` = instrument differential Gaussian), at the
        current nonlinear params, the correction is

            delta_c = E_c - A_c   (the dropped +1/2 Var_LSF curvature term).

        ``m_star`` is the UNBROADENED, RV-shifted stellar log-flux (= the stellar
        columns shifted by the RV phase, NO broadening multiply, weighted by w);
        ``m_tell`` is the UNBROADENED, lag-shifted telluric log-flux (weighted by u).
        The continuum is smooth -> no correction. Returns ``(delta_star, delta_tell,
        E_star, E_tell)`` (eager numpy), where ``E_c = A_c + delta_c`` are the EXACT
        broadened blocks used for the reported decomposition.

        Masked-region guard: the correction is only used at GOOD pixels (the masked
        edges/gaps are W = 0). A resampled basis can extrapolate to enormous log-flux
        there (e.g. m_tell ~ +100 -> exp ~ 1e46), and convolving such a spike rings
        the circular FFT into NEGATIVE linear flux -> log(neg) = NaN that would poison
        the warm Newton. So we exponentiate from the log-flux with the masked region
        set to the continuum level (m_c = 0 -> linear flux 1, the neutral "no
        absorption" value), floor the convolved linear flux to a tiny positive value
        before the log (as the toy generator does), and zero ``delta`` outside the
        good region. The instrument Gaussian is only a few pixels wide -- far narrower
        than the masked edge band -- so this leaves the correction at good pixels
        physically intact.
        """
        theta = jnp.asarray(theta)
        w_ = theta[:Ktot]
        u_ = theta[Ktot:Ktot + J_tell]
        phase = jnp.exp(-2j * np.pi * freq * float(p))
        goodj = jnp.asarray(good)
        FLOOR = 1e-300                  # linear-flux floor before the log (toy-style)

        # --- stellar block -----------------------------------------------------
        gT = star_tf(sigma_obs_pix, vsini_pix)
        # unbroadened, RV-shifted stellar log-flux m_star(x) = sum_k w_k T_k(x - dx)
        m_star = jnp.fft.irfft((Tf * phase[:, None]) @ w_, n=n)
        # log-space broadened block A_star = G_star (x)_log m_star, applied to the
        # SAME shifted-weighted combination (matches the design's stellar columns).
        A_star = jnp.fft.irfft((Tf * gT[:, None] * phase[:, None]) @ w_, n=n)
        # exact (linear-flux) broadened block E_star = ln(G_star (x) exp(m_star)),
        # with the masked region forced to continuum (m=0) so edge spikes cannot ring.
        es = jnp.exp(jnp.where(goodj, m_star, 0.0))
        E_star_raw = jnp.log(jnp.maximum(jnp.fft.irfft(jnp.fft.rfft(es) * gT, n=n), FLOOR))
        delta_star = np.where(good, np.asarray(E_star_raw - A_star), 0.0)
        # report E_star = A_star + delta_star so the block decomposition sums EXACTLY
        # to ln_model = A@theta + delta everywhere (incl. the delta=0 masked region).
        E_star = np.asarray(A_star) + delta_star

        # --- telluric block (instrument broadening only, no rotation) ----------
        if J_tell:
            tell_phase = (jnp.exp(-2j * np.pi * freq * float(p_tell))
                          if p_tell else jnp.ones_like(freq))
            gS = tell_tf(sigma_obs_pix)
            m_tell = jnp.fft.irfft((Sf * tell_phase[:, None]) @ u_, n=n)
            A_tell = jnp.fft.irfft((Sf * gS[:, None] * tell_phase[:, None]) @ u_, n=n)
            et = jnp.exp(jnp.where(goodj, m_tell, 0.0))
            E_tell_raw = jnp.log(jnp.maximum(jnp.fft.irfft(jnp.fft.rfft(et) * gS, n=n), FLOOR))
            delta_tell = np.where(good, np.asarray(E_tell_raw - A_tell), 0.0)
            E_tell = np.asarray(A_tell) + delta_tell
        else:
            delta_tell = np.zeros(n)
            E_tell = np.zeros(n)
        return delta_star, delta_tell, E_star, E_tell

    # --- decide the nonlinear-parameter layout ---------------------------------
    # sigma_obs_pix: fit / fixed / off(0); vsini_pix: fit / fixed / off(0).
    sig_obs_fixed = 0.0 if (resolution_R is None or fit_resolution) else float(sigpix_of_R(resolution_R))
    vsini_fixed = 0.0 if (vsini is None or fit_vsini) else float(vsini) / velscale

    const_j, ridge_j = jnp.asarray(const), jnp.asarray(float(ridge))
    sig_nat_star_j, sig_nat_tell_j = jnp.asarray(sig_nat_star), jnp.asarray(sig_nat_tell)
    eps_j = jnp.asarray(float(epsilon))

    def _refine(x0, fit_R, fit_vs, sig_obs_fix, vsini_fix, fit_t,
                lnd_arg=lndj, const_arg=const_j):
        # box bounds matching the x0 layout [p (, sigma_obs)(, vsini)(, p_tell)].
        # R in [RMIN, RMAX] <-> sigma_obs in [sigpix(RMAX), sigpix(RMIN)].
        # ``lnd_arg``/``const_arg`` default to the raw data; the linear-flux
        # convolution loop passes the corrected ``lnd_eff`` (and its const) here so
        # the Newton refine fits the corrected data (a warm restart from x0).
        lo, hi = [-np.inf], [np.inf]
        if fit_R:
            lo.append(float(sigpix_of_R(R_bounds[1])))
            hi.append(float(sigpix_of_R(R_bounds[0])))
        if fit_vs:
            lo.append(float(vsini_bounds[0] / velscale))
            hi.append(float(vsini_bounds[1] / velscale))
        if fit_t:
            lo.append(-np.inf)
            hi.append(np.inf)
        x_star, H = _newton_jit(
            jnp.asarray(x0), jnp.asarray(lo), jnp.asarray(hi),
            Tf, Sf, Pj, Wj, lnd_arg, freq, j_off, const_arg, ridge_j,
            sig_nat_star_j, sig_nat_tell_j, jnp.asarray(float(sig_obs_fix)),
            jnp.asarray(float(vsini_fix)), eps_j, n=n, fit_R=fit_R, fit_vsini=fit_vs,
            fit_tell=fit_t)
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
    # the telluric lag (wavelength zero-point) is small, so we add it to the
    # refinement initialised at 0 (the value it held in the coarse fixed block)
    # and let Newton move it jointly -- no coarse grid dimension for it.
    if fit_tell:
        x0.append(0.0)
    sig_obs_fix_use = sig_obs_fixed if not refine_R else 0.0
    vsini_fix_use = vsini_fixed if not refine_vs else 0.0
    x_star, H = _refine(x0, refine_R, refine_vs, sig_obs_fix_use, vsini_fix_use,
                        fit_tell)

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
    if fit_tell:
        p_tell_star = float(x_star[idx]); tell_col = idx; idx += 1
    else:
        p_tell_star = 0.0; tell_col = None

    # --- exact final solve at the optimum (iter 0; delta = 0) ------------------
    chi2_exact, theta, cov, A = _exact_at(p_star, sp_star, vsp_star, p_tell_star)
    theta = np.asarray(theta)

    # --- iterated linear-flux convolution correction ---------------------------
    # iter 0 above used lnd_eff = lnd (delta = 0) -> byte-identical to the historical
    # fit. When n_conv_iter >= 1 we fit the EXACT (linear-flux-broadened) model by
    # the fixed point: correct the data lnd_eff = lnd - delta (delta = E - A peaks at
    # deep cores), WARM-START the Newton from the previous optimum (no coarse scan --
    # the small correction does not move the basin), re-solve exactly, recompute
    # delta. Stop after n_conv_iter iters or when delta stops changing.
    delta_star = delta_tell = None        # block corrections (None until computed)
    E_star_block = E_tell_block = None     # exact broadened blocks for reporting
    if n_conv_iter >= 1:
        x_prev = np.asarray(x_star, float)
        for _it in range(int(n_conv_iter)):
            # delta from the CURRENT solution (its nonlinear params + theta).
            dstar, dtell, E_star_block, E_tell_block = _delta(
                theta, p_star, sp_star, vsp_star, p_tell_star)
            delta = dstar + dtell                  # continuum needs no correction
            delta_prev = delta_star + delta_tell if delta_star is not None else None
            delta_star, delta_tell = dstar, dtell

            # corrected log-data and its lnd-dependent normal-equation pieces.
            lnd_eff = np.where(good, lnd - delta, 0.0)
            lnd_eff_j = jnp.asarray(lnd_eff)
            const_eff_j = jnp.sum(Wj * lnd_eff_j * lnd_eff_j)

            # warm refine (Newton from x_prev, NO coarse scan) on the corrected data,
            # then the exact profiled solve on the corrected data.
            x_star, H = _refine(x_prev, refine_R, refine_vs, sig_obs_fix_use,
                                vsini_fix_use, fit_tell,
                                lnd_arg=lnd_eff_j, const_arg=const_eff_j)
            p_star = float(x_star[0])
            jj = 1
            if refine_R:
                sp_star = abs(float(x_star[jj])); jj += 1
            if refine_vs:
                vsp_star = abs(float(x_star[jj])); jj += 1
            if fit_tell:
                p_tell_star = float(x_star[jj]); jj += 1
            # chi2_exact = sum W (lnd_eff - A)^2 = sum W (lnd - A - delta)^2 -- the
            # EXACT-model chi2, since lnd_eff - A = lnd - (A + delta) = lnd - E.
            chi2_exact, theta, cov, A = _exact_at(
                p_star, sp_star, vsp_star, p_tell_star, lnd_arg=lnd_eff_j)
            theta = np.asarray(theta)
            x_prev = np.asarray(x_star, float)

            # early stop if delta has effectively converged.
            if delta_prev is not None:
                dchg = float(np.max(np.abs((delta_star + delta_tell) - delta_prev)))
                if dchg < 1e-8:
                    break
        # recompute the FINAL delta / exact blocks at the converged solution so the
        # reported model (exp(A + delta)) and block decomposition are consistent
        # with the params we report. (The chi2 from the last in-loop solve used the
        # PRE-refine delta; the refine then moved theta/params, so we recompute the
        # exact-model chi2 = sum W (lnd - A@theta - delta_final)^2 against the FINAL
        # delta so chi2 and the reported model E = A@theta + delta agree exactly.)
        delta_star, delta_tell, E_star_block, E_tell_block = _delta(
            theta, p_star, sp_star, vsp_star, p_tell_star)
        A_np_final = np.asarray(A)
        ln_model_final = A_np_final @ theta + (delta_star + delta_tell)
        chi2_exact = float(np.sum(np.asarray(W) * np.where(
            good, lnd - ln_model_final, 0.0) ** 2))

    dx_star = p_star * dln
    v_star = C_LIGHT_KMS * np.expm1(dx_star)
    w = theta[:Ktot]
    u = theta[Ktot:Ktot + J_tell]
    c = theta[Ktot + J_tell:]
    n_good = int(good.sum())
    D = Ktot + J_tell + L1
    # Count every REQUESTED nonlinear dimension (v, plus R/vsini if those are
    # fit), not just the ones actually refined -- a search costs a degree of
    # freedom even when its optimum lands on a bound (lower limit). This matches
    # the previous single-broadening convention (n_nonlin = 2 when fit_resolution).
    n_nonlin = 1 + int(fit_resolution) + int(fit_vsini) + int(fit_tell)
    dof = n_good - D - n_nonlin

    # --- covariance of the active nonlinear params (Delta-chi2 = 1) ------------
    # H is over x_star = [p (, sigma_obs_pix) (, vsini_pix) (, p_tell)]; invert to
    # get cov. The objective is EVEN in each broadening param (R, vsini) -- we
    # report their abs() -- so we flip the cross-term sign when one was refined to
    # the negative branch. The telluric lag p_tell is NOT even (it is a genuine
    # signed shift like p), so it is excluded from the sign-fix.
    if np.all(np.linalg.eigvalsh(H) > 0):
        cov_x = 2.0 * np.linalg.inv(H)
        for k in range(1, x_star.size):
            if k == tell_col:
                continue
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

    # --- EXACT chi2 curves vs the fitted broadening params (at the optimum) -----
    # The coarse-scan surface uses the M_TT(0) (zero-shift) approximation, whose
    # broadening (R/vsini) minimum is biased at large RV shifts. For an honest
    # diagnostic we recompute chi2 EXACTLY along R and vsini at the refined optimum
    # (fixing p, the other broadening, and the telluric lag) -- one FFT + solve each,
    # so its minimum coincides with the reported value.
    if fit_resolution:
        R_grid_fine = np.geomspace(R_bounds[0], R_bounds[1], 40)
        out["R_curve"] = (R_grid_fine, np.array([
            _exact_at(p_star, float(sigpix_of_R(R)), vsp_star, p_tell_star)[0]
            for R in R_grid_fine]))
    if fit_vsini:
        vs_grid_fine = np.linspace(vsini_bounds[0], vsini_bounds[1], 40)
        out["vsini_curve"] = (vs_grid_fine, np.array([
            _exact_at(p_star, sp_star, float(vk / velscale), p_tell_star)[0]
            for vk in vs_grid_fine]))

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

    if fit_tell:
        # Telluric lag p_tell = the instrument wavelength zero-point (tellurics are
        # at rest in the observer frame). Report it as a velocity, and report the
        # stellar RV with that zero-point removed via the LAG DIFFERENCE so the
        # shared zero-point cancels: v_corr = c * expm1((p_star - p_tell) * dln).
        # Errors propagate from the joint (p_star, p_tell) 2x2 sub-block of cov_x.
        dx_tell = p_tell_star * dln
        v_tell = C_LIGHT_KMS * np.expm1(dx_tell)
        dvt_dpt = C_LIGHT_KMS * np.exp(dx_tell) * dln
        var_pt = cov_x[tell_col, tell_col]
        v_tell_err = (np.sqrt(var_pt) * abs(dvt_dpt)
                      if np.isfinite(var_pt) and var_pt >= 0 else np.nan)

        q = p_star - p_tell_star
        dxq = q * dln
        v_corr = C_LIGHT_KMS * np.expm1(dxq)
        dvc_dp = C_LIGHT_KMS * np.exp(dxq) * dln       # d v_corr / d p_star
        # d v_corr / d p_tell = -dvc_dp; propagate through the joint 2x2 cov.
        cov_pp = np.array([[cov_x[0, 0], cov_x[0, tell_col]],
                           [cov_x[tell_col, 0], cov_x[tell_col, tell_col]]])
        g = np.array([dvc_dp, -dvc_dp])
        var_corr = float(g @ cov_pp @ g)
        v_corr_err = np.sqrt(var_corr) if np.isfinite(var_corr) and var_corr >= 0 else np.nan

        out.update({
            "v_tell_kms": float(v_tell),
            "v_tell_err_kms": float(v_tell_err),
            "v_corr_kms": float(v_corr),
            "v_corr_err_kms": float(v_corr_err),
            "p_tell_star": float(p_tell_star),
        })

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

    # Optionally rescale the formal (curvature) errors so the reduced chi-square is
    # 1, i.e. multiply by sqrt(chi2/dof). This folds the fit's own misfit (template
    # mismatch, correlated residuals) into the reported statistical errors; it is a
    # rescaling, not an additive systematic floor.
    if rescale_errors and dof > 0 and np.isfinite(out["chi2_dof"]) and out["chi2_dof"] > 0:
        s = float(np.sqrt(out["chi2_dof"]))
        for k in ("v_err_kms", "R_err", "sigma_kms_err",
                  "vsini_err_kms", "v_tell_err_kms", "v_corr_err_kms"):
            if k in out and np.isfinite(out[k]):
                out[k] = float(out[k] * s)
        out["error_rescale"] = s

    if return_model:
        A_np = np.asarray(A)
        if delta_star is None:
            # n_conv_iter = 0: the historical log-space-broadened model (unchanged).
            ln_model = A_np @ theta
            out["ln_model"] = ln_model
            out["model"] = np.exp(ln_model)
            out["lnd"] = lnd
            out["resid_lnd"] = np.where(good, lnd - ln_model, np.nan)
            out["good"] = good
            # decompose the additive log-model into its blocks (shifted/broadened):
            # stellar absorption, telluric absorption, and the Legendre continuum.
            out["ln_star"] = A_np[:, :Ktot] @ theta[:Ktot]
            out["ln_cont"] = A_np[:, Ktot + J_tell:] @ theta[Ktot + J_tell:]
            if J_tell:
                out["ln_tell"] = A_np[:, Ktot:Ktot + J_tell] @ theta[Ktot:Ktot + J_tell]
        else:
            # n_conv_iter >= 1: report the EXACT (linear-flux-broadened) model
            # E = A + delta. ln_model = A@theta + delta; model = exp(E); the fit
            # residual is lnd - E = lnd_eff - A (the residual on the corrected data).
            delta = delta_star + delta_tell
            ln_model = A_np @ theta + delta
            out["ln_model"] = ln_model
            out["model"] = np.exp(ln_model)
            out["lnd"] = lnd
            out["resid_lnd"] = np.where(good, lnd - ln_model, np.nan)
            out["good"] = good
            # block decomposition uses the EXACT broadened blocks so it still sums
            # to ln_model: ln_star = E_star = A_star + delta_star, ln_tell = E_tell,
            # ln_cont = cont (smooth -> uncorrected).
            out["ln_star"] = E_star_block
            out["ln_cont"] = A_np[:, Ktot + J_tell:] @ theta[Ktot + J_tell:]
            if J_tell:
                out["ln_tell"] = E_tell_block
    return out
