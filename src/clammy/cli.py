"""Command-line interface for clammy.

    clammy build --templates DIR --basis pca --var 0.99 --k-max K -o basis.npz
    clammy fit   basis.npz spectrum.fits [--fit-resolution ...]

`build` rectifies a spectral grid and reduces it to a saved template basis;
`fit` recovers radial velocity (+ optionally resolution), template weights, and
continuum from an observed spectrum.
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
def cmd_build(args):
    loglam, flux, params, files = grid.load_grid(args.templates)
    print(f"loaded {flux.shape[0]} spectra x {flux.shape[1]} px from {args.templates}")

    if args.basis != "pca":
        raise SystemExit(f"unknown --basis {args.basis!r} (only 'pca' is implemented)")
    R, _ = basis.rectify_grid(loglam, flux, order=args.order)
    mu, Phi, info = basis.build_pca(R, var_target=args.var, max_k=args.k_max)
    K = info["K"]
    print(f"PCA: kept K={K} components for {info['cumvar'][K - 1] * 100:.2f}% variance")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    basis.save_basis(args.out, loglam, mu, Phi, params, info, args.order)
    print(f"wrote basis -> {args.out}")


# --------------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------------- #
def _column(table, name):
    """Pull a 1-D array from an astropy Table, handling single-row array-cell
    tables (dmost-style) and ordinary per-pixel-row tables."""
    if len(table) == 1 and np.ndim(table[name][0]) > 0:
        return np.asarray(table[name][0], float)
    return np.asarray(table[name], float)


def read_spectrum(path, wave_col, flux_col, sigma_col, snr, log_wave):
    """Return (loglam, flux, sigma) for an observed spectrum.

    Supports .npz (arrays keyed by the column names) and FITS tables. The
    wavelength axis may be linear (Angstrom) or natural-log; it is auto-detected
    (median > 50 -> linear) unless --log-wave/--linear-wave forces it.
    """
    if path.endswith(".npz"):
        d = np.load(path)
        wave = np.asarray(d[wave_col], float)
        flux = np.asarray(d[flux_col], float)
        sigma = np.asarray(d[sigma_col], float) if sigma_col in d.files else None
    else:
        from astropy.table import Table
        t = Table.read(path)
        wave = _column(t, wave_col)
        flux = _column(t, flux_col)
        sigma = _column(t, sigma_col) if sigma_col in t.colnames else None

    if log_wave is None:
        log_wave = np.nanmedian(wave) < 50.0  # log axis is ~8-9; linear is thousands
    loglam = wave if log_wave else np.log(wave)

    if sigma is None:
        if snr is None:
            raise SystemExit("no sigma column found; pass --sigma-col or --snr")
        sigma = np.abs(flux) / float(snr)
    return loglam, flux, sigma


def cmd_fit(args):
    b = basis.load_basis(args.basis_file)
    loglam_b, mu, Phi = b["loglam"], b["mu"], b["Phi"]

    loglam_o, flux_o, sigma_o = read_spectrum(
        args.spectrum_file, args.wave_col, args.flux_col, args.sigma_col,
        args.snr, args.log_wave)

    # resample onto the basis grid if the observation is on a different grid
    same = (loglam_o.shape == loglam_b.shape and
            np.allclose(loglam_o, loglam_b, atol=0, rtol=1e-12))
    if same:
        flux, sigma, mask = flux_o, sigma_o, None
    else:
        flux, sigma, good = toy.resample_to_loglam(loglam_o, flux_o, sigma_o, loglam_b)
        mask = good
        print(f"resampled observation onto basis grid ({good.sum()} of {loglam_b.size} px in range)")

    res = fitter.fit_rv(flux, sigma, loglam_b, mu, Phi, cont_order=args.cont_order,
                        vmin=args.vmin, vmax=args.vmax, mask=mask,
                        fit_resolution=args.fit_resolution,
                        R_bounds=tuple(args.R_bounds), ridge=args.ridge)

    print(f"\nv        = {res['v_kms']:+.3f} +/- {res['v_err_kms']:.3f} km/s")
    if args.fit_resolution:
        rl = "  (lower limit; unresolved)" if res.get("resolution_limited") else ""
        print(f"R        = {res['resolution_R']:.0f} +/- {res['R_err']:.0f}{rl}")
        print(f"rho(v,R) = {res['rho_vR']:+.3f}")
    print(f"chi2/dof = {res['chi2_dof']:.3f}   (chi2={res['chi2']:.1f}, dof={res['dof']}, n={res['n_good']})")
    print(f"weights  = {np.array2string(res['w'], precision=4, suppress_small=True)}")
    print(f"continuum= {np.array2string(res['c'], precision=4, suppress_small=True)}")

    if args.out:
        payload = {k: res[k] for k in ("v_kms", "v_err_kms", "chi2", "dof", "chi2_dof", "n_good")}
        payload["w"] = res["w"].tolist()
        payload["c"] = res["c"].tolist()
        if args.fit_resolution:
            payload.update(resolution_R=res["resolution_R"], R_err=res["R_err"],
                           rho_vR=res["rho_vR"])
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote results -> {args.out}")

    if args.plot:
        _plot_fit(loglam_b, flux, res, args.plot)
        print(f"wrote plot -> {args.plot}")


def _plot_fit(loglam, flux, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lam = np.exp(loglam)
    sl = slice(0, len(lam), max(1, len(lam) // 4000))
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(lam[sl], flux[sl], lw=0.4, color="0.5", label="data")
    ax[0].plot(lam[sl], res["model"][sl], lw=0.5, color="C3", label="model")
    title = f"v={res['v_kms']:.2f} km/s, chi2/dof={res['chi2_dof']:.2f}"
    if "rho_vR" in res:
        title += f", R={res['resolution_R']:.0f}"
    ax[0].set_title(title); ax[0].set_ylabel("flux"); ax[0].legend()
    ax[1].plot(lam[sl], res["resid_lnd"][sl], lw=0.4); ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_ylabel("ln-flux resid"); ax[1].set_xlabel("wavelength [A]")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(prog="clammy", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    pb = sub.add_parser("build", help="build a template basis from a spectral grid")
    pb.add_argument("--templates", "-t", default=paths.CONVOLVED,
                    help="directory of dmost_lte_*.fits templates (default: grid/convolved)")
    pb.add_argument("--basis", default="pca", choices=["pca"], help="basis type")
    pb.add_argument("--var", type=float, default=0.99, help="cumulative-variance target")
    pb.add_argument("--k-max", type=int, default=None, help="cap on number of components")
    pb.add_argument("--order", type=int, default=5, help="rectification Legendre order")
    pb.add_argument("--out", "-O", "-o", dest="out", default="basis.npz",
                    help="output basis .npz")
    pb.set_defaults(func=cmd_build)

    pf = sub.add_parser("fit", help="fit RV/weights/continuum[/resolution] to a spectrum")
    pf.add_argument("basis_file", help="basis .npz from `clammy build`")
    pf.add_argument("spectrum_file", help="observed spectrum (.npz or FITS table)")
    pf.add_argument("--cont-order", type=int, default=3, help="Legendre continuum order")
    pf.add_argument("--vmin", type=float, default=-500.0, help="RV search min [km/s]")
    pf.add_argument("--vmax", type=float, default=500.0, help="RV search max [km/s]")
    pf.add_argument("--fit-resolution", action="store_true", help="also fit the resolution R")
    pf.add_argument("--R-bounds", type=float, nargs=2, default=(2000.0, 60000.0),
                    metavar=("RMIN", "RMAX"), help="R search range when --fit-resolution")
    pf.add_argument("--ridge", type=float, default=0.0, help="Tikhonov ridge on the normal matrix")
    pf.add_argument("--wave-col", default="wave"); pf.add_argument("--flux-col", default="flux")
    pf.add_argument("--sigma-col", default="sigma")
    pf.add_argument("--snr", type=float, default=None,
                    help="if no sigma column, assume sigma = |flux| / snr")
    grp = pf.add_mutually_exclusive_group()
    grp.add_argument("--log-wave", dest="log_wave", action="store_true", default=None,
                     help="treat the wavelength column as natural-log")
    grp.add_argument("--linear-wave", dest="log_wave", action="store_false",
                     help="treat the wavelength column as linear (Angstrom)")
    pf.add_argument("--out", "-o", default=None, help="write results JSON")
    pf.add_argument("--plot", default=None, help="write a data/model/residual plot")
    pf.set_defaults(func=cmd_fit)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
