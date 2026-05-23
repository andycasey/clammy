"""Command-line interface for clammy.

    clammy build --templates DIR --basis pca --var 0.99 --k-max K -o basis.npz
    clammy build --kind telluric --templates DIR --resolution R -o tell.npz
    clammy fit   basis.npz spectrum.fits [--telluric-basis tell.npz] [...]

`build` rectifies a spectral grid and reduces it to a saved template basis. With
``--kind stellar`` (default) it builds a rest-frame stellar basis (RV-shifted by
the fitter); with ``--kind telluric`` it builds an OBSERVED-frame telluric basis
(held fixed in the topocentric frame, never RV-shifted). Both record their native
resolving power (``--resolution``) so the fitter can broaden differentially.

`fit` recovers radial velocity (+ optionally instrument resolution and/or vsini),
template weights, continuum -- and, when a ``--telluric-basis`` is supplied, a set
of telluric weights -- from an observed spectrum.
"""
import argparse
import json
import os
import sys

import numpy as np

from . import basis, grid, fitter, toy, paths


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def _resolution_arg(args):
    """Native resolving power to store: the --resolution value or np.inf."""
    return float(args.resolution) if args.resolution is not None else np.inf


def cmd_build(args):
    if args.basis != "pca":
        raise SystemExit(f"unknown --basis {args.basis!r} (only 'pca' is implemented)")
    resolution = _resolution_arg(args)

    if args.kind == "telluric":
        # Telluric model grids carry no stellar labels, so use the
        # label-agnostic loader and the (custom) glob pattern.
        loglam, flux, files = grid.load_spectra(args.templates, pattern=args.pattern)
        print(
            f"loaded {flux.shape[0]} telluric spectra x {flux.shape[1]} px "
            f"from {args.templates} (pattern {args.pattern!r})"
        )
        mu, Phi, info = basis.build_telluric_basis(
            loglam, flux, order=args.order, var_target=args.var, max_k=args.k_max
        )
        K = info["K"]
        print(
            f"PCA: kept K={K} components for "
            f"{info['cumvar'][K - 1] * 100:.2f}% variance"
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        basis.save_basis(
            args.out,
            loglam,
            mu,
            Phi,
            params=None,
            info=info,
            rectify_order=args.order,
            frame="observed",
            resolution=resolution,
        )
        print(
            "NOTE: telluric basis built/stored in the OBSERVED (topocentric) "
            "frame -- it is NOT RV-shifted by `clammy fit`."
        )
        print(
            f"wrote telluric basis -> {args.out} "
            f"(frame=observed, resolution={resolution})"
        )
        return

    # stellar (default): unchanged pipeline, now recording frame + resolution.
    loglam, flux, params, files = grid.load_grid(args.templates)
    print(f"loaded {flux.shape[0]} spectra x {flux.shape[1]} px from {args.templates}")
    R, _ = basis.rectify_grid(loglam, flux, order=args.order)
    mu, Phi, info = basis.build_pca(R, var_target=args.var, max_k=args.k_max)
    K = info["K"]
    print(f"PCA: kept K={K} components for {info['cumvar'][K - 1] * 100:.2f}% variance")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    basis.save_basis(
        args.out,
        loglam,
        mu,
        Phi,
        params=params,
        info=info,
        rectify_order=args.order,
        frame="rest",
        resolution=resolution,
    )
    print(f"wrote basis -> {args.out} (frame=rest, resolution={resolution})")


# --------------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------------- #
def _column(table, name):
    """Pull a 1-D array from an astropy Table, handling single-row array-cell
    tables (dmost-style) and ordinary per-pixel-row tables."""
    if len(table) == 1 and np.ndim(table[name][0]) > 0:
        return np.asarray(table[name][0], float)
    return np.asarray(table[name], float)


def vac_to_air(wave_vac):
    """Vacuum -> air wavelengths (Angstrom), matching the dmost/PHOENIX template grid."""
    w = np.asarray(wave_vac, float)
    s2 = (1.0e4 / w) ** 2
    n = 1.0 + 0.0000834254 + 0.02406147 / (130.0 - s2) + 0.00015998 / (38.9 - s2)
    return w / n


def read_spectrum(path, wave_col, flux_col, sigma_col, ivar_col, mask_col, snr,
                  log_wave, wave_frame):
    """Return (loglam, flux, sigma, good) for an observed spectrum.

    Supports .npz (arrays keyed by name) and FITS tables (incl. the dmost single-row
    array-cell layout and PypeIt ``OneSpec`` per-pixel tables). Uncertainties come
    from ``sigma_col`` if present, else ``ivar_col`` (``sigma = 1/sqrt(ivar)``), else
    ``--snr``. A 1=good ``mask_col`` is honoured (returned as ``good``; else None).
    The wavelength axis is linear-A or natural-log (auto-detected: median > 50 ->
    linear), and is converted vacuum->air to match the (air) template grid when
    ``wave_frame='vacuum'`` -- or ``'auto'`` and the file looks like a PypeIt product
    (e.g. a Keck/DEIMOS OneSpec spectrum, which stores vacuum wavelengths).
    """
    meta = {}
    if path.endswith(".npz"):
        d = np.load(path)
        present = set(d.files)
        col = lambda c: np.asarray(d[c], float)
    else:
        from astropy.table import Table

        t = Table.read(path)
        meta = {str(k).upper(): t.meta[k] for k in t.meta}
        present = set(t.colnames)
        col = lambda c: _column(t, c)

    wave = col(wave_col)
    flux = col(flux_col)

    if sigma_col in present:
        sigma = col(sigma_col)
    elif ivar_col in present:
        iv = col(ivar_col)
        sigma = np.where(iv > 0, 1.0 / np.sqrt(np.where(iv > 0, iv, 1.0)), np.inf)
    elif snr is not None:
        sigma = np.abs(flux) / float(snr)
    else:
        raise SystemExit(
            f"no '{sigma_col}' or '{ivar_col}' column; pass --sigma-col/--ivar-col or --snr"
        )

    good = (np.asarray(col(mask_col), float) > 0) if mask_col in present else None

    if log_wave is None:
        log_wave = np.nanmedian(wave) < 50.0  # log axis is ~8-9; linear is thousands
    wave_ang = np.exp(wave) if log_wave else np.asarray(wave, float)

    if wave_frame == "auto":
        vacuum = any(k in meta for k in ("PYP_SPEC", "DMODCLS")) or str(
            meta.get("INSTRUME", "")
        ).upper().startswith(("DEIMOS", "KECK"))
    else:
        vacuum = wave_frame == "vacuum"
    if vacuum:
        wave_ang = vac_to_air(wave_ang)
        print("converted observed wavelengths vacuum -> air (matching the air template grid)")

    return np.log(wave_ang), flux, sigma, good


def _load_telluric_basis(path, loglam_fit):
    """Load a telluric basis and resample it onto the fitting grid.

    Returns ``(tell_mu, tell_Phi, R_native_tell)``. The telluric basis must be in
    the OBSERVED frame (warn loudly otherwise). Its ``mu`` and every row of
    ``Phi`` are interpolated with ``np.interp`` onto ``loglam_fit`` when the grids
    differ; if they are identical the arrays are passed through unchanged.
    """
    tb = basis.load_basis(path)
    if tb["frame"] != "observed":
        print(
            f"WARNING: telluric basis frame is {tb['frame']!r}, expected "
            "'observed'. Telluric columns are NOT RV-shifted; a rest-frame basis "
            "here is almost certainly a mistake.",
            file=sys.stderr,
        )
    loglam_t = np.asarray(tb["loglam"], float)
    mu_t = np.asarray(tb["mu"], float)
    Phi_t = np.atleast_2d(np.asarray(tb["Phi"], float))

    same = loglam_t.shape == loglam_fit.shape and np.allclose(
        loglam_t, loglam_fit, atol=0, rtol=1e-12
    )
    if same:
        tell_mu, tell_Phi = mu_t, Phi_t
    else:
        tell_mu = np.interp(loglam_fit, loglam_t, mu_t)
        tell_Phi = np.vstack(
            [np.interp(loglam_fit, loglam_t, row) for row in Phi_t]
        )
        print(
            f"resampled telluric basis onto the fitting grid "
            f"({loglam_t.size} -> {loglam_fit.size} px)"
        )
    return tell_mu, tell_Phi, float(tb["resolution"])


def cmd_fit(args):
    b = basis.load_basis(args.basis_file)
    loglam_b, mu, Phi = b["loglam"], b["mu"], b["Phi"]
    R_native_star = float(b["resolution"])

    loglam_o, flux_o, sigma_o, good_o = read_spectrum(
        args.spectrum_file,
        args.wave_col,
        args.flux_col,
        args.sigma_col,
        args.ivar_col,
        args.mask_col,
        args.snr,
        args.log_wave,
        args.wave_frame,
    )

    # resample onto the basis grid if the observation is on a different grid
    same = loglam_o.shape == loglam_b.shape and np.allclose(
        loglam_o, loglam_b, atol=0, rtol=1e-12
    )
    if same:
        flux, sigma, mask = flux_o, sigma_o, good_o
    else:
        flux, sigma, good = toy.resample_to_loglam(loglam_o, flux_o, sigma_o, loglam_b)
        mask = good
        if good_o is not None:
            # carry the observed-frame bad-pixel mask onto the basis grid: keep only
            # destination pixels that interpolate from all-good source pixels.
            mask = mask & (np.interp(loglam_b, loglam_o, good_o.astype(float)) > 0.999)
        print(
            f"resampled observation onto basis grid "
            f"({int(np.asarray(mask).sum())} of {loglam_b.size} px usable)"
        )

    # optional telluric basis (observed frame, NOT RV-shifted), resampled to grid.
    tell_basis = None
    R_native_tell = np.inf
    if args.telluric_basis:
        tell_mu, tell_Phi, R_native_tell = _load_telluric_basis(
            args.telluric_basis, np.asarray(loglam_b, float)
        )
        tell_basis = (tell_mu, tell_Phi)

    fit_kwargs = dict(
        cont_order=args.cont_order,
        vmin=args.vmin,
        vmax=args.vmax,
        mask=mask,
        ridge=args.ridge,
        resolution_R=args.resolution_R,
        fit_resolution=args.fit_resolution,
        R_bounds=tuple(args.R_bounds),
        R_native_star=R_native_star,
        R_native_tell=R_native_tell,
        tell_basis=tell_basis,
        fit_telluric_shift=args.fit_telluric_shift,
        vsini=args.vsini,
        fit_vsini=args.fit_vsini,
        vsini_bounds=tuple(args.vsini_bounds),
        epsilon=args.epsilon,
        rescale_errors=args.rescale_errors,
    )
    res = fitter.fit_rv(flux, sigma, loglam_b, mu, Phi, **fit_kwargs)
    if args.rescale_errors:
        print(f"\n(errors below rescaled by sqrt(chi2/dof) = {res.get('error_rescale', 1.0):.2f})")

    print(f"\nv        = {res['v_kms']:+.3f} +/- {res['v_err_kms']:.3f} km/s")
    if args.fit_resolution:
        rl = "  (lower limit; unresolved)" if res.get("resolution_limited") else ""
        print(f"R        = {res['resolution_R']:.0f} +/- {res['R_err']:.0f}{rl}")
        if np.isfinite(res.get("rho_vR", np.nan)):
            print(f"rho(v,R) = {res['rho_vR']:+.3f}")
    elif args.resolution_R is not None:
        print(f"R        = {res['resolution_R']:.0f}   (fixed)")
    if args.fit_vsini:
        if res.get("vsini_limited"):
            print(
                f"vsini    = {res['vsini_kms']:.2f} km/s  "
                "(lower limit; unresolved)"
            )
        else:
            print(f"vsini    = {res['vsini_kms']:.2f} +/- {res['vsini_err_kms']:.2f} km/s")
    elif args.vsini is not None:
        print(f"vsini    = {res['vsini_kms']:.2f} km/s   (fixed)")
    if args.fit_telluric_shift:
        print(f"v_tell   = {res['v_tell_kms']:+.3f} +/- {res['v_tell_err_kms']:.3f} km/s   (telluric wavelength zero-point)")
        print(f"v_corr   = {res['v_corr_kms']:+.3f} +/- {res['v_corr_err_kms']:.3f} km/s   (zero-point-corrected RV)")
    if args.vbary is not None:
        _base = res.get("v_corr_kms", res["v_kms"])
        _lbl = "barycentric+telluric" if "v_corr_kms" in res else "barycentric"
        print(f"v_final  = {_base + args.vbary:+.3f} km/s   ({_lbl}-corrected; vbary={args.vbary:+.3f})")
    print(
        f"chi2/dof = {res['chi2_dof']:.3f}   "
        f"(chi2={res['chi2']:.1f}, dof={res['dof']}, n={res['n_good']})"
    )
    print(f"weights  = {np.array2string(res['w'], precision=4, suppress_small=True)}")
    if "u" in res:
        print(
            f"telluric = {np.array2string(res['u'], precision=4, suppress_small=True)}"
        )
    print(f"continuum= {np.array2string(res['c'], precision=4, suppress_small=True)}")

    if args.out:
        payload = {
            k: res[k]
            for k in ("v_kms", "v_err_kms", "chi2", "dof", "chi2_dof", "n_good")
        }
        payload["w"] = res["w"].tolist()
        payload["c"] = res["c"].tolist()
        if args.fit_resolution:
            payload.update(
                resolution_R=res["resolution_R"],
                R_err=res["R_err"],
                rho_vR=res.get("rho_vR"),
                resolution_limited=bool(res.get("resolution_limited")),
            )
        elif args.resolution_R is not None:
            payload["resolution_R"] = res["resolution_R"]
        if args.fit_vsini:
            payload.update(
                vsini_kms=res["vsini_kms"],
                vsini_err_kms=res["vsini_err_kms"],
                vsini_limited=bool(res.get("vsini_limited")),
            )
        elif args.vsini is not None:
            payload["vsini_kms"] = res["vsini_kms"]
        if args.fit_telluric_shift:
            payload.update(
                v_tell_kms=res["v_tell_kms"],
                v_tell_err_kms=res["v_tell_err_kms"],
                v_corr_kms=res["v_corr_kms"],
                v_corr_err_kms=res["v_corr_err_kms"],
            )
        if "u" in res:
            payload["u"] = res["u"].tolist()
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote results -> {args.out}")

    ref = (args.ref_v, args.ref_v_err, args.ref_name) if args.ref_v is not None else None
    if args.plot:
        _plot_fit(loglam_b, flux, sigma, res, args.plot,
                  source=os.path.basename(args.spectrum_file), ref=ref, vbary=args.vbary)
        print(f"wrote plot -> {args.plot}")

    if args.scan_plot:
        _plot_scan(res, args.scan_plot, source=os.path.basename(args.spectrum_file))
        print(f"wrote scan plot -> {args.scan_plot}")


def _plot_fit(loglam, flux, sigma, res, path, source=None, ref=None, vbary=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lam = np.exp(loglam)
    sl = slice(0, len(lam), max(1, len(lam) // 12000))
    fig, ax = plt.subplots(
        2,
        1,
        figsize=(33, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax[0].plot(lam[sl], flux[sl], lw=0.4, color="black", label="data")
    ax[0].plot(lam[sl], res["model"][sl], lw=0.5, color="tab:red", label="model")
    # the fitted Legendre continuum (smooth, multiplicative): exp(P . c) in linear flux
    cont = np.exp(np.polynomial.legendre.legval(basis.xnorm(loglam), res["c"]))
    ax[0].plot(lam[sl], cont[sl], lw=1.0, color="tab:orange", ls=":", label="continuum")
    # restrict both panels to where there is real (in-coverage, finite) data; the
    # resampled data is edge-clamped (finite) outside coverage, so use the `good` mask.
    valid = np.isfinite(flux)
    good = res.get("good")
    if good is not None:
        valid &= np.asarray(good, bool)
    if valid.any():
        lam_v = lam[valid]
        ax[0].set_xlim(lam_v.min(), lam_v.max())  # sharex -> applies to both panels
        lo, hi = np.percentile(flux[valid], [1, 99])
        margin = 0.10 * (hi - lo)
        ax[0].set_ylim(0.0, hi + margin)  # floor at 0, keep the (1-99 pctile + 10%) max
    # title: the RV chain raw -> telluric-zero-point-corrected -> +barycentric, then
    # R/vsini/chi2 and an optional external reference velocity (e.g. the Geha value).
    v_e = res.get("v_err_kms", float("nan"))
    title = f"v={res['v_kms']:+.2f}±{v_e:.2f}"
    base, base_e = res["v_kms"], v_e
    if "v_corr_kms" in res:
        ce = res.get("v_corr_err_kms", float("nan"))
        title += f" -> tell-corr {res['v_corr_kms']:+.2f}±{ce:.2f}"
        base, base_e = res["v_corr_kms"], ce
    if vbary is not None:
        lbl = "bary+tell" if "v_corr_kms" in res else "bary"
        title += f" -> {lbl} {base + vbary:+.2f}±{base_e:.2f}"
    title += " km/s"
    if "rho_vR" in res:
        title += f", R={res['resolution_R']:.0f}"
    if "vsini_kms" in res:
        title += f", vsini={res['vsini_kms']:.1f}"
    title += f", chi2/dof={res['chi2_dof']:.2f}"
    if ref is not None:
        rv, rverr, rname = ref
        title += f"   |   {rname} v={rv:+.2f}"
        if rverr is not None:
            title += f"±{rverr:.2f}"
        title += " km/s"
    ax[0].set_title((f"{source}\n" if source else "") + title)
    ax[0].set_ylabel("flux")
    ax[0].legend()
    # we fit in log-flux, but show the linear-flux chi residual (data - model)/sigma
    chi = (flux - res["model"]) / sigma
    if good is not None:
        chi = np.where(np.asarray(good, bool), chi, np.nan)
    ax[1].plot(lam[sl], chi[sl], lw=0.4, color="0.3")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].axhline(3, color="k", lw=0.5, ls="--")
    ax[1].axhline(-3, color="k", lw=0.5, ls="--")
    ax[1].set_ylim(-5, 5)
    ax[1].set_ylabel("(flux - model) / sigma")
    ax[1].set_xlabel("wavelength [A]")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_scan(res, path, source=None):
    """Coarse-scan profiled chi2: chi2(v, R) when R was fit, else chi2(v)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = np.asarray(res["v_grid"], float)
    if "chi2_2d" in res and "R_grid" in res:
        chi2 = np.asarray(res["chi2_2d"], float)  # (n_sp, S); row 0 = no broadening
        Rg = np.asarray(res["R_grid"], float)  # (n_sp-1,), aligned with rows 1:
        dchi = chi2 - np.nanmin(chi2)
        fig, ax = plt.subplots(
            2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [1, 2]}
        )
        ax[0].plot(v, dchi.min(axis=0), lw=0.8, color="0.2")
        ax[0].axvline(res["v_kms"], color="C3", lw=0.8)
        ax[0].set_ylabel(r"$\Delta\chi^2$ (profiled over R)")
        from matplotlib.colors import LogNorm

        d2 = np.clip(dchi[1:], 1.0, None)  # color by log of Delta-chi2 (compress range)
        pm = ax[1].pcolormesh(
            v, Rg, d2, shading="auto", cmap="viridis",
            norm=LogNorm(vmin=1.0, vmax=max(10.0, float(np.nanmax(d2)))),
        )
        ax[1].plot(res["v_kms"], res["resolution_R"], "x", color="tab:red", ms=10, mew=2)
        ax[1].set_ylabel("R")
        ax[1].set_xlabel("v [km/s]")
        fig.colorbar(pm, ax=ax[1], label=r"$\Delta\chi^2$ (log scale)")
        title = f"chi2(v, R) scan: v={res['v_kms']:.2f} km/s, R={res['resolution_R']:.0f}"
    else:
        dchi = np.asarray(res["chi2_grid"], float)
        dchi = dchi - np.nanmin(dchi)
        fig, axx = plt.subplots(figsize=(10, 4))
        axx.plot(v, dchi, lw=0.8, color="0.2")
        axx.axvline(res["v_kms"], color="C3", lw=0.8)
        axx.set_xlabel("v [km/s]")
        axx.set_ylabel(r"$\Delta\chi^2$")
        ax = [axx]
        title = f"chi2(v) scan: v={res['v_kms']:.2f} km/s"
    ax[0].set_title((f"{source}\n" if source else "") + title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
_FIT_EPILOG = """\
broadening modes (instrument resolution R and rotation vsini):
  1. fit R + vsini together   --fit-resolution --fit-vsini
         instrument LSF applied to BOTH stellar and telluric blocks;
         rotation applied to the STELLAR block only.
  2. fixed R, no vsini        --resolution-R 7000
         instrument LSF (fixed) applied to both blocks; no rotation.
  3. nothing                  (no broadening flags)
         assume the templates are already at the right resolution.
  4. vsini only               --fit-vsini   (optionally --vsini for a fixed value)
         assume the instrument resolution is right; fit extra stellar
         rotational broadening.
Each basis carries its native resolving power; the fitter broadens
DIFFERENTIALLY in quadrature (sigma_diff^2 = max(0, sigma_obs^2 - sigma_native^2)),
so stellar and telluric grids built at different resolutions are handled
correctly.
"""


def build_parser():
    p = argparse.ArgumentParser(
        prog="clammy",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    pb = sub.add_parser("build", help="build a template basis from a spectral grid")
    pb.add_argument(
        "--kind",
        choices=["stellar", "telluric"],
        default="stellar",
        help="stellar basis (rest frame, RV-shifted) or telluric basis "
        "(observed frame, NOT shifted)",
    )
    pb.add_argument(
        "--templates",
        "-t",
        default=paths.CONVOLVED,
        help="directory of template spectra (default: grid/convolved). For "
        "--kind stellar these are dmost_lte_*.fits; for --kind telluric, any "
        "files matching --pattern",
    )
    pb.add_argument(
        "--pattern",
        default="*.fits",
        help="glob pattern for --kind telluric (load_spectra; default *.fits)",
    )
    pb.add_argument("--basis", default="pca", choices=["pca"], help="basis type")
    pb.add_argument("--var", type=float, default=0.99, help="cumulative-variance target")
    pb.add_argument("--k-max", type=int, default=None, help="cap on number of components")
    pb.add_argument("--order", type=int, default=5, help="rectification Legendre order")
    pb.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="native resolving power R of the input grid (stored with the basis; "
        "default: np.inf = effectively unresolved)",
    )
    pb.add_argument(
        "--out",
        "-O",
        "-o",
        dest="out",
        default="basis.npz",
        help="output basis .npz",
    )
    pb.set_defaults(func=cmd_build)

    pf = sub.add_parser(
        "fit",
        help="fit RV/weights/continuum[/resolution/vsini/tellurics] to a spectrum",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_FIT_EPILOG,
    )
    pf.add_argument("basis_file", help="stellar basis .npz from `clammy build`")
    pf.add_argument("spectrum_file", help="observed spectrum (.npz or FITS table)")
    pf.add_argument(
        "--telluric-basis",
        default=None,
        help="optional OBSERVED-frame telluric basis .npz (from "
        "`clammy build --kind telluric`); adds telluric columns that are NOT "
        "RV-shifted, resampled onto the fitting grid if needed",
    )
    pf.add_argument(
        "--fit-telluric-shift",
        action="store_true",
        help="also fit a velocity shift of the telluric block (the instrument "
        "wavelength zero-point); reports v_tell and the zero-point-corrected v_corr",
    )
    pf.add_argument("--cont-order", type=int, default=3, help="Legendre continuum order")
    pf.add_argument("--vmin", type=float, default=-500.0, help="RV search min [km/s]")
    pf.add_argument("--vmax", type=float, default=500.0, help="RV search max [km/s]")
    pf.add_argument(
        "--resolution-R",
        dest="resolution_R",
        type=float,
        default=None,
        help="FIXED observed instrument resolving power R (broaden both bases "
        "once; mode 2). Ignored if --fit-resolution is given",
    )
    pf.add_argument(
        "--fit-resolution",
        action="store_true",
        help="also fit the observed instrument resolution R (mode 1)",
    )
    pf.add_argument(
        "--R-bounds",
        type=float,
        nargs=2,
        default=(2000.0, 60000.0),
        metavar=("RMIN", "RMAX"),
        help="R search range when --fit-resolution",
    )
    pf.add_argument(
        "--fit-vsini",
        action="store_true",
        help="fit projected rotational broadening vsini, stellar block only "
        "(mode 4; combine with --fit-resolution for mode 1)",
    )
    pf.add_argument(
        "--vsini",
        type=float,
        default=None,
        help="FIXED vsini [km/s] applied to the stellar block (ignored if "
        "--fit-vsini is given)",
    )
    pf.add_argument(
        "--vsini-bounds",
        type=float,
        nargs=2,
        default=(1.0, 300.0),
        metavar=("VMIN", "VMAX"),
        help="vsini search range [km/s] when --fit-vsini (default 1 300)",
    )
    pf.add_argument(
        "--epsilon",
        type=float,
        default=0.6,
        help="linear limb-darkening coefficient for the Gray rotation profile "
        "(default 0.6)",
    )
    pf.add_argument("--ridge", type=float, default=0.0, help="Tikhonov ridge on the normal matrix")
    pf.add_argument("--wave-col", default="wave")
    pf.add_argument("--flux-col", default="flux")
    pf.add_argument("--sigma-col", default="sigma")
    pf.add_argument("--ivar-col", default="ivar",
                    help="inverse-variance column; sigma=1/sqrt(ivar) when no sigma column")
    pf.add_argument("--mask-col", default="mask",
                    help="good-pixel mask column (1=good), honoured if present")
    pf.add_argument("--wave-frame", choices=["air", "vacuum", "auto"], default="auto",
                    help="wavelength frame of the input; converted vacuum->air to match the "
                    "template grid. 'auto' treats PypeIt/DEIMOS products as vacuum (default)")
    pf.add_argument(
        "--snr",
        type=float,
        default=None,
        help="if no sigma column, assume sigma = |flux| / snr",
    )
    grp = pf.add_mutually_exclusive_group()
    grp.add_argument(
        "--log-wave",
        dest="log_wave",
        action="store_true",
        default=None,
        help="treat the wavelength column as natural-log",
    )
    grp.add_argument(
        "--linear-wave",
        dest="log_wave",
        action="store_false",
        help="treat the wavelength column as linear (Angstrom)",
    )
    pf.add_argument("--out", "-o", default=None, help="write results JSON")
    pf.add_argument("--plot", default=None, help="write a data/model/residual plot")
    pf.add_argument("--scan-plot", default=None,
                    help="write the coarse-scan chi2 surface: chi2(v, R) if --fit-resolution, else chi2(v)")
    pf.add_argument("--ref-v", type=float, default=None,
                    help="reference RV [km/s] to show in the plot title (e.g. a catalog value)")
    pf.add_argument("--ref-v-err", type=float, default=None, help="uncertainty on --ref-v [km/s]")
    pf.add_argument("--ref-name", default="ref", help="label for --ref-v in the plot title")
    pf.add_argument("--vbary", type=float, default=None,
                    help="barycentric velocity correction [km/s] to add to the (telluric-)corrected RV in the title/report")
    pf.add_argument("--rescale-errors", action="store_true",
                    help="rescale the formal RV errors by sqrt(chi2/dof) (reduced chi2 -> 1)")
    pf.set_defaults(func=cmd_fit)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
