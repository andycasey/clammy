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


def _nmf_components_arg(args):
    """Return the NMF K, or raise if --nmf-components was not supplied."""
    if args.nmf_components is None:
        raise SystemExit("--basis nmf requires --nmf-components K")
    return int(args.nmf_components)


def cmd_build(args):
    resolution = _resolution_arg(args)

    if args.kind == "telluric":
        loglam, flux, files = grid.load_spectra(args.templates, pattern=args.pattern)
        print(
            f"loaded {flux.shape[0]} telluric spectra x {flux.shape[1]} px "
            f"from {args.templates} (pattern {args.pattern!r})"
        )
        if args.basis == "nmf":
            K = _nmf_components_arg(args)
            mu, Phi, info = basis.build_telluric_nmf(loglam, flux, K, order=args.order)
            tag = f"NMF K={info['K']}, {info['cumvar'][-1] * 100:.2f}% variance"
        else:
            mu, Phi, info = basis.build_telluric_basis(
                loglam, flux, order=args.order, var_target=args.var, max_k=args.k_max
            )
            K = info["K"]
            tag = f"PCA K={K}, {info['cumvar'][K - 1] * 100:.2f}% variance"
        print(tag)
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
            method=args.basis,
        )
        print(
            "NOTE: telluric basis built/stored in the OBSERVED (topocentric) "
            "frame -- it is NOT RV-shifted by `clammy fit`."
        )
        print(f"wrote telluric basis -> {args.out} (frame=observed, resolution={resolution})")
        return

    # stellar
    loglam, flux, params, files = grid.load_grid(args.templates)
    print(f"loaded {flux.shape[0]} spectra x {flux.shape[1]} px from {args.templates}")
    R, _ = basis.rectify_grid(loglam, flux, order=args.order)
    if args.basis == "nmf":
        K = _nmf_components_arg(args)
        mu, Phi, info = basis.build_nmf(R, K)
        tag = f"NMF K={info['K']}, {info['cumvar'][-1] * 100:.2f}% variance"
    else:
        mu, Phi, info = basis.build_pca(R, var_target=args.var, max_k=args.k_max)
        K = info["K"]
        tag = f"PCA K={K}, {info['cumvar'][K - 1] * 100:.2f}% variance"
    print(tag)

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
        method=args.basis,
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


def _parse_hdu(val):
    """--hdu as an int extension index (e.g. '42', matching hdu[42]) or an EXTNAME
    string (e.g. 'SPAT0248-SLIT0177-MSC03'); None (no --hdu) passes through."""
    if val is None:
        return None
    s = str(val)
    return int(s) if s.lstrip("+-").isdigit() else s


def vac_to_air(wave_vac):
    """Vacuum -> air wavelengths (Angstrom), matching the dmost/PHOENIX template grid."""
    w = np.asarray(wave_vac, float)
    s2 = (1.0e4 / w) ** 2
    n = 1.0 + 0.0000834254 + 0.02406147 / (130.0 - s2) + 0.00015998 / (38.9 - s2)
    return w / n


def read_spectrum(path, wave_col, flux_col, sigma_col, ivar_col, mask_col, snr,
                  log_wave, wave_frame, hdu=None, max_spike=30.0):
    """Return (loglam, flux, sigma, good, ext_label) for an observed spectrum.

    ``ext_label`` is the slit/extension name (e.g. ``SPAT0248-SLIT0177-MSC03``)
    when the file is a multi-extension spec1d -- used to label the fit log, plot
    title, and auto-named PNGs so per-slit runs don't overwrite each other -- and
    ``None`` for .npz or a single-spectrum FITS (output naming then unchanged).

    Supports .npz (arrays keyed by name) and FITS tables: the dmost single-row
    array-cell layout, PypeIt ``OneSpec`` per-pixel tables, and PypeIt ``spec1d``
    multi-extension files (one ``SpecObj`` table per slit/object). For ``spec1d``,
    pass ``hdu`` (an integer extension index matching ``hdu[N]`` -- e.g. 42 -- or an
    ``EXTNAME`` string like ``SPAT0248-SLIT0177-MSC03``) to choose the slit, and the
    optimal-extraction columns (``OPT_WAVE``/``OPT_COUNTS``/``OPT_COUNTS_IVAR``/
    ``OPT_MASK``) are picked up automatically when the generic defaults are absent.

    Uncertainties come from ``sigma_col`` if present, else ``ivar_col``
    (``sigma = 1/sqrt(ivar)``), else ``--snr``; non-positive sigma (masked pixels)
    is set to ``inf`` so it carries no weight. A 1=good ``mask_col`` is honoured
    (returned as ``good``; else None). ``max_spike>0`` additionally rejects
    catastrophic positive flux spikes (cosmic rays / hot pixels the pipeline mask
    missed) more than ``max_spike`` times a robust running continuum -- raw PypeIt
    counts carry a handful, and in the weighted log-flux fit they would otherwise
    dominate everything (and overflow ``exp``); the cut is conservative enough never
    to trip on clean / continuum-normalized data. The wavelength axis is linear-A or natural-log
    (auto-detected: median > 50 -> linear), and is converted vacuum->air to match the
    (air) template grid when ``wave_frame='vacuum'`` -- or ``'auto'`` and the file
    looks like a PypeIt product (e.g. a Keck/DEIMOS spectrum, which stores vacuum
    wavelengths).
    """
    meta, ext_label = {}, None
    if path.endswith(".npz"):
        d = np.load(path)
        present = set(d.files)
        col = lambda c: np.asarray(d[c], float)
    else:
        from astropy.io import fits
        from astropy.table import Table

        t = Table.read(path, hdu=hdu) if hdu is not None else Table.read(path)
        # Identify which extension was actually read. spec1d files hold one slit
        # per HDU, and without this nothing downstream (log, plot title, PNG name)
        # records which slit was fit -- and a default/auto run silently takes
        # hdu=1. Also merge the PRIMARY header so the vacuum/air auto-detection
        # keys (PYP_SPEC/INSTRUME/DMODCLS) are visible; they live in HDU 0, not
        # the slit table.
        ext_name = str(t.meta.get("EXTNAME", "")).strip()
        try:
            with fits.open(path) as hdul:
                ph = hdul[0].header
                meta = {str(k).upper(): ph[k] for k in ph if k}
                multi_ext = sum(isinstance(h, fits.BinTableHDU) for h in hdul) > 1
                if isinstance(hdu, str):
                    ext_idx = hdul.index_of(hdu)
                elif isinstance(hdu, (int, np.integer)):
                    ext_idx = int(hdu)
                elif ext_name:
                    ext_idx = hdul.index_of(ext_name)
                else:
                    ext_idx = None
                if not ext_name and ext_idx is not None:
                    ext_name = str(hdul[ext_idx].name).strip()
        except Exception:
            meta, multi_ext = {}, False
            ext_idx = hdu if isinstance(hdu, int) else None
        meta.update({str(k).upper(): t.meta[k] for k in t.meta})
        present = set(t.colnames)
        col = lambda c: _column(t, c)

        # Only label genuine multi-extension (spec1d) files, so single-spectrum
        # FITS keep their plain "<spectrum>-fit.png" output names.
        if multi_ext:
            ext_label = ext_name or (f"hdu{ext_idx}" if ext_idx is not None else None)
            if ext_label:
                where = f"hdu={ext_idx}" if ext_idx is not None else f"hdu={hdu!r}"
                print(f"read extension {ext_label!r} ({where})")

        # PypeIt spec1d (SpecObj) tables name their columns OPT_*/BOX_*; fall back to
        # the optimal-extraction columns whenever a generic default name is absent.
        _alt = lambda name, opt: name if name in present else (opt if opt in present else name)
        wave_col, flux_col, sigma_col, ivar_col, mask_col = (
            _alt(wave_col, "OPT_WAVE"),
            _alt(flux_col, "OPT_COUNTS"),
            _alt(sigma_col, "OPT_COUNTS_SIG"),
            _alt(ivar_col, "OPT_COUNTS_IVAR"),
            _alt(mask_col, "OPT_MASK"),
        )
        if wave_col.startswith("OPT_") or flux_col.startswith("OPT_"):
            print(f"PypeIt spec1d columns detected: using "
                  f"{wave_col}/{flux_col}/{ivar_col}/{mask_col}")

    wave = col(wave_col)
    flux = col(flux_col)

    if sigma_col in present:
        s = col(sigma_col)
        sigma = np.where(s > 0, s, np.inf)  # masked pixels carry sigma=0 -> ignore
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

    if max_spike and max_spike > 0:
        from scipy.ndimage import median_filter

        f = np.asarray(flux, float)
        cont = median_filter(np.nan_to_num(f, nan=0.0), size=51)  # robust to isolated CRs
        spike = np.isfinite(f) & (cont > 0) & (f > max_spike * cont)
        if spike.any():
            good = (np.ones(f.shape, bool) if good is None else good) & ~spike
            print(
                f"rejected {int(spike.sum())} catastrophic flux spike(s) "
                f"(> {max_spike:g}x local continuum; cosmic rays / hot pixels)"
            )

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

    return np.log(wave_ang), flux, sigma, good, ext_label


def _auto_vbary(path):
    """Heliocentric velocity correction [km/s] inferred from a FITS header, or None.

    Reads the target coordinates (RA_OBJ/DEC_OBJ, else RA/DEC), the epoch (MJD, else
    MJD-OBS), and the observatory location (LON/LAT/ALT-OBS, else the named TELESCOP
    site) and returns the heliocentric correction via astropy. Returns None for .npz
    or whenever the needed keywords (or astropy) are unavailable -- the fit then
    simply reports no barycentric-corrected velocity.
    """
    if not path or path.endswith(".npz"):
        return None
    try:
        from astropy.io import fits
        from astropy.coordinates import SkyCoord, EarthLocation
        from astropy.time import Time
        import astropy.units as u

        with fits.open(path) as hdul:
            hdrs = [h.header.copy() for h in hdul]

        def get(*keys):
            for h in hdrs:
                for k in keys:
                    if k in h and h[k] not in ("", None):
                        return h[k]
            return None

        ra, dec, mjd = get("RA_OBJ", "RA"), get("DEC_OBJ", "DEC"), get("MJD", "MJD-OBS")
        if ra is None or dec is None or mjd is None:
            return None

        loc, lon, lat, alt = None, get("LON-OBS"), get("LAT-OBS"), get("ALT-OBS")
        if lon is not None and lat is not None:
            # FITS LON-OBS is usually West-positive; EarthLocation wants East-positive.
            # (Only the small diurnal term depends on this, so a best-effort sign is OK.)
            lon_e = -float(lon) if float(lon) > 0 else float(lon)
            loc = EarthLocation.from_geodetic(
                lon_e * u.deg, float(lat) * u.deg, (float(alt or 0.0)) * u.m)
        else:
            tel = str(get("TELESCOP") or "")
            if tel:
                loc = EarthLocation.of_site(tel)  # may need a one-off site download
        if loc is None:
            return None

        unit = (u.deg, u.deg) if isinstance(ra, (int, float)) else (u.hourangle, u.deg)
        sc = SkyCoord(ra, dec, unit=unit)
        v = sc.radial_velocity_correction(
            "heliocentric", obstime=Time(float(mjd), format="mjd"), location=loc)
        return float(v.to("km/s").value)
    except Exception:
        return None


def _load_telluric_basis(path, loglam_fit):
    """Load a telluric basis and resample it onto the fitting grid.

    Returns ``(tell_mu, tell_Phi, R_native_tell, method)``. The telluric basis
    must be in the OBSERVED frame (warn loudly otherwise). Its ``mu`` and every
    row of ``Phi`` are interpolated with ``np.interp`` onto ``loglam_fit`` when
    the grids differ; if they are identical the arrays are passed through unchanged.
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
    return tell_mu, tell_Phi, float(tb["resolution"]), str(tb.get("method", "pca"))


def cmd_fit(args):
    paths = args.spectrum_files
    if len(paths) > 1 and (args.plot or args.scan_plot or args.out):
        raise SystemExit(
            "with multiple spectra use --plots (auto-named per input); "
            "--plot/--scan-plot/--out each take a single output file"
        )

    # Load the (shared) bases once; the per-spectrum fits below reuse the JIT-compiled
    # kernels (identical array shapes on the basis grid), so spectra after the first
    # skip XLA compilation.
    b = basis.load_basis(args.basis_file)
    loglam_b, mu, Phi = b["loglam"], b["mu"], b["Phi"]
    R_native_star = float(b["resolution"])
    nnls_stellar = (b.get("method", "pca") == "nmf")

    tell_basis = None
    R_native_tell = np.inf
    nnls_telluric = False
    if args.telluric_basis:
        tell_mu, tell_Phi, R_native_tell, tell_method = _load_telluric_basis(
            args.telluric_basis, np.asarray(loglam_b, float)
        )
        tell_basis = (tell_mu, tell_Phi)
        nnls_telluric = (tell_method == "nmf")

    if nnls_stellar or nnls_telluric:
        which = " + ".join(
            (["stellar"] if nnls_stellar else []) + (["telluric"] if nnls_telluric else [])
        )
        print(f"NMF basis detected ({which}): weights will be non-negative (NNLS post-refine).")

    ref = (args.ref_v, args.ref_v_err, args.ref_name) if args.ref_v is not None else None

    for i, path in enumerate(paths):
        if len(paths) > 1:
            print(f"\n=== [{i + 1}/{len(paths)}] {os.path.basename(path)} ===")
        try:
            _fit_one(path, args, loglam_b, mu, Phi, R_native_star,
                     tell_basis, R_native_tell, ref,
                     nnls_stellar=nnls_stellar, nnls_telluric=nnls_telluric)
        except Exception as exc:  # don't let one bad spectrum abort a batch
            if len(paths) == 1:
                raise
            print(f"  ERROR fitting {path}: {exc}", file=sys.stderr)


def _fit_one(path, args, loglam_b, mu, Phi, R_native_star, tell_basis, R_native_tell, ref,
             nnls_stellar=False, nnls_telluric=False):
    """Fit one spectrum and write its report, JSON, and plots."""
    loglam_o, flux_o, sigma_o, good_o, ext_label = read_spectrum(
        path,
        args.wave_col,
        args.flux_col,
        args.sigma_col,
        args.ivar_col,
        args.mask_col,
        args.snr,
        args.log_wave,
        args.wave_frame,
        hdu=_parse_hdu(args.hdu),
        max_spike=args.max_spike,
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

    # restrict the fit -- and, via the `good` mask the fitter returns, the plotted
    # range -- to the [wl_min, wl_max] observed-wavelength window. Out-of-window
    # pixels are masked (zero weight), so they enter neither chi2 nor the plot.
    if args.wl_min is not None or args.wl_max is not None:
        lam_b = np.exp(np.asarray(loglam_b, float))
        lo = -np.inf if args.wl_min is None else float(args.wl_min)
        hi = np.inf if args.wl_max is None else float(args.wl_max)
        window = (lam_b >= lo) & (lam_b <= hi)
        mask = window if mask is None else (np.asarray(mask, bool) & window)
        print(
            f"restricted to {lo:g}-{hi:g} A "
            f"({int(window.sum())} of {lam_b.size} basis px in window)"
        )

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
        n_conv_iter=args.conv_iter,
        nnls_stellar=nnls_stellar,
        nnls_telluric=nnls_telluric,
        weight_scheme=args.weight_scheme,
    )
    res = fitter.fit_rv(flux, sigma, loglam_b, mu, Phi, **fit_kwargs)
    if args.rescale_errors:
        print(f"\n(errors below rescaled by sqrt(chi2/dof) = {res.get('error_rescale', 1.0):.2f})")

    # barycentric correction: use --vbary if given, else infer it from the FITS header.
    vbary = args.vbary if args.vbary is not None else _auto_vbary(path)

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
    if vbary is not None:
        _base = res.get("v_corr_kms", res["v_kms"])
        _lbl = "barycentric+telluric" if "v_corr_kms" in res else "barycentric"
        _src = "given" if args.vbary is not None else "auto from header"
        print(f"v_final  = {_base + vbary:+.3f} km/s   ({_lbl}-corrected; vbary={vbary:+.3f}, {_src})")
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
        if ext_label:
            payload["extension"] = ext_label
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
        if vbary is not None:
            payload["vbary_kms"] = float(vbary)
            payload["v_final_kms"] = float(res.get("v_corr_kms", res["v_kms"]) + vbary)
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote results -> {args.out}")

    # output plots: explicit --plot/--scan-plot, or auto-named "<spectrum>-{fit,scan}.png"
    # next to each input when --plots is given. For multi-slit spec1d files the slit
    # label is folded into the auto-name (so per-slit runs don't overwrite) and the
    # plot title (so the figure says which slit it is); explicit names are untouched.
    stem = os.path.splitext(path)[0] + (f"-{ext_label}" if ext_label else "")
    source = os.path.basename(path) + (f" [{ext_label}]" if ext_label else "")
    fit_png = args.plot or (f"{stem}-fit.png" if args.plots else None)
    scan_png = args.scan_plot or (f"{stem}-scan.png" if args.plots else None)
    if fit_png:
        _plot_fit(loglam_b, flux, sigma, res, fit_png,
                  source=source, ref=ref, vbary=vbary)
        print(f"wrote plot -> {fit_png}")
    if scan_png:
        _plot_scan(res, scan_png, source=source)
        print(f"wrote scan plot -> {scan_png}")


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
    # decompose the (multiplicative) model: continuum and the telluric absorption
    # applied ON the continuum. ln_cont/ln_tell come from the fitter; fall back to
    # recomputing the continuum from c if absent.
    cont_ln = (np.asarray(res["ln_cont"]) if "ln_cont" in res
               else np.polynomial.legendre.legval(basis.xnorm(loglam), res["c"]))
    # the FULL model -- continuum × stellar × telluric -- is what the residual
    # panel measures; always draw it so the top panel shows what is actually fit.
    # The telluric × continuum overlay omits the stellar absorption, so it sits
    # above the data where stellar lines are -- don't read it as the fit. Layering:
    # data (black) and the full model (red) are the primary curves; the telluric ×
    # continuum and continuum overlays are secondary and recede (thin, alpha).
    ax[0].plot(lam[sl], res["model"][sl], lw=0.9, color="tab:red", label="model")
    if "ln_tell" in res:
        ax[0].plot(lam[sl], np.exp(np.asarray(res["ln_tell"]) + cont_ln)[sl],
                   lw=0.5, alpha=0.5, color="tab:blue", label="telluric × continuum")
    ax[0].plot(lam[sl], np.exp(cont_ln)[sl], lw=0.8, alpha=0.7, color="tab:orange",
               ls="--", label="continuum")
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
    """Profiled Delta-chi2 vs each fitted nonlinear parameter.

    chi2(v), chi2(R) and chi2(vsini) are each recomputed EXACTLY at the optimum
    (``v_curve``/``R_curve``/``vsini_curve`` from the fitter), so their minima
    coincide with the reported values. The coarse (M_TT(0)-approximation) surface
    is only a fallback for ``v`` when ``v_curve`` is absent -- it can be biased (even
    negative) at large lags, so its minimum is not necessarily a real chi2 minimum.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (x, Delta-chi2, x_opt, xlabel, ylabel, log-x?)
    panels = []
    if "v_curve" in res:
        v, cv = res["v_curve"]
        v = np.asarray(v, float); cv = np.asarray(cv, float)
        panels.append((v, cv - np.nanmin(cv), res["v_kms"], "v [km/s]",
                       r"$\Delta\chi^2$ (exact)", False))
    else:
        v = np.asarray(res["v_grid"], float)
        cv = np.asarray(res["chi2_grid"], float)
        panels.append((v, cv - np.nanmin(cv), res["v_kms"], "v [km/s]",
                       r"$\Delta\chi^2$ (coarse)", False))
    if "R_curve" in res:
        Rg, cR = res["R_curve"]
        panels.append((np.asarray(Rg, float), np.asarray(cR, float) - np.nanmin(cR),
                       res["resolution_R"], "R", r"$\Delta\chi^2$ (exact)", True))
    if "vsini_curve" in res:
        vg, cvs = res["vsini_curve"]
        panels.append((np.asarray(vg, float), np.asarray(cvs, float) - np.nanmin(cvs),
                       res["vsini_kms"], "vsini [km/s]", r"$\Delta\chi^2$ (exact)", False))

    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 3.0 * len(panels)))
    axes = np.atleast_1d(axes)
    for ax, (x, d, xopt, xlabel, ylabel, logx) in zip(axes, panels):
        ax.plot(x, d, lw=1.0, color="0.2")
        ax.axvline(xopt, color="tab:red", lw=1.2, label=f"fit = {xopt:.4g}")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper center", fontsize=8)
    title = f"v={res['v_kms']:+.2f} km/s"
    if "R_curve" in res:
        title += f", R={res['resolution_R']:.0f}"
    if "vsini_curve" in res:
        title += f", vsini={res['vsini_kms']:.1f}"
    axes[0].set_title((f"{source}\n" if source else "") + title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
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
    pb.add_argument("--basis", default="pca", choices=["pca", "nmf"], help="basis type")
    pb.add_argument("--var", type=float, default=0.99, help="cumulative-variance target (PCA)")
    pb.add_argument("--k-max", type=int, default=None, help="cap on number of components (PCA)")
    pb.add_argument(
        "--nmf-components",
        type=int,
        default=None,
        metavar="K",
        help="number of NMF components (required when --basis nmf)",
    )
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
    pf.add_argument("spectrum_files", nargs="+", metavar="spectrum",
                    help="observed spectrum/spectra (.npz or FITS table); multiple are "
                    "fit in one process, reusing the JIT-compiled kernels")
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
    pf.add_argument("--wl-min", dest="wl_min", type=float, default=6775.0,
                    help="only fit/plot observed wavelengths >= this [A] (default 6775); "
                    "out-of-window pixels are masked from the fit and the plot")
    pf.add_argument("--wl-max", dest="wl_max", type=float, default=8700.0,
                    help="only fit/plot observed wavelengths <= this [A] (default 8700)")
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
    pf.add_argument(
        "--hdu",
        default=None,
        help="FITS extension to read from a multi-extension file (e.g. a PypeIt "
        "spec1d): an integer index matching hdu[N] (e.g. 42) or an EXTNAME string "
        "like SPAT0248-SLIT0177-MSC03. Default: the first table HDU. The PypeIt "
        "OPT_WAVE/OPT_COUNTS/OPT_COUNTS_IVAR/OPT_MASK columns are then auto-detected",
    )
    pf.add_argument(
        "--max-spike",
        type=float,
        default=30.0,
        metavar="FACTOR",
        help="reject positive flux spikes (unmasked cosmic rays / hot pixels) more "
        "than FACTOR x a robust running continuum; needed so raw PypeIt counts don't "
        "blow up the log-flux fit. Conservative (never trips on clean data); 0=off",
    )
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
    pf.add_argument("--out", "-o", default=None, help="write results JSON (single spectrum)")
    pf.add_argument("--plot", default=None, help="write a data/model/residual plot (single spectrum)")
    pf.add_argument("--scan-plot", default=None,
                    help="write the coarse-scan chi2 surface (single spectrum): chi2(v, R) if --fit-resolution, else chi2(v)")
    pf.add_argument("--plots", action="store_true",
                    help="save <spectrum>-fit.png and <spectrum>-scan.png next to each input "
                    "(works for any number of spectra)")
    pf.add_argument("--ref-v", type=float, default=None,
                    help="reference RV [km/s] to show in the plot title (e.g. a catalog value)")
    pf.add_argument("--ref-v-err", type=float, default=None, help="uncertainty on --ref-v [km/s]")
    pf.add_argument("--ref-name", default="ref", help="label for --ref-v in the plot title")
    pf.add_argument("--vbary", type=float, default=None,
                    help="barycentric velocity correction [km/s]; if omitted it is computed "
                    "automatically from the FITS header (RA/Dec/MJD/site) when possible")
    pf.add_argument("--rescale-errors", action="store_true",
                    help="rescale the formal RV errors by sqrt(chi2/dof) (reduced chi2 -> 1)")
    pf.add_argument("--conv-iter", type=int, default=0, metavar="N",
                    help="iterate the EXACT linear-flux convolution N times to fix deep line "
                    "cores (0=off, fast log-space approx; ~8-12 converges). Unbiases R.")
    pf.add_argument("--weight", default="snr2", choices=["snr2", "ivar"],
                    dest="weight_scheme",
                    help="pixel weighting in log-flux space. 'snr2' (default): W = (d/sigma)^2 "
                    "= SNR^2 -- the delta-method log-flux weight; down-weights deep absorption "
                    "cores by d^2 so they are nearly ignored. 'ivar': W = (d_cont/sigma)^2 "
                    "where d_cont is the 95th-percentile flux (continuum proxy) -- all pixels "
                    "get the same weight scale regardless of local flux; use when sigma is "
                    "read-noise dominated and you want the cores to influence the fit equally.")
    pf.set_defaults(func=cmd_fit)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
