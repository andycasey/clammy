#!/usr/bin/env python
"""Reconstruct telluric transmission spectra from PypeIt's TellPCA model.

This separates the (faithful) spectrum reconstruction from any basis decomposition:
it writes the full set of model transmission spectra to ONE ``.npz`` so you can
either build a clammy PCA telluric basis from it --

    clammy build --kind telluric --templates <out.npz> --resolution 120000 -o tell.npz

-- or apply a different factorisation (e.g. NMF) to the same spectra later.

Correct transformations applied (PypeIt's TellPCA is stored in arsinh(optical depth),
not transmission, and in vacuum wavelengths):
  1. arsinh(tau) -> transmission, per training coefficient vector ``c`` (HDU3):
         arsinh_tau = [1, c] . pca          (pca row 0 = mean, rows 1.. = eigenvectors)
         tau        = max(0, sinh(arsinh_tau))
         T          = exp(-tau)              in (0, 1]
  2. vacuum -> air wavelengths (the dmost/DEIMOS stellar grid is in air), using the
     same dispersion relation as ``process_phoenix_templates.py``;
  3. a small floor on T so the downstream ``ln`` (clammy rectification) stays finite.

Spectra are kept on the model's native (full-resolution, full-range) grid by default;
``--wmin/--wmax`` trim the wavelength range and ``--n-spectra`` limits the count.

Output ``.npz`` (clammy-build ready; extra keys are ignored by the loader):
  wave        (n_pix,)          natural-log AIR wavelength (clammy grid convention)
  flux        (n_spec, n_pix)   transmission in (0, 1], strictly positive
  coeff       (n_spec, n_comp)  the TellPCA coefficient vector per spectrum (labels)
  resolution  scalar            nominal resolving power (120000)
"""
import argparse
import os

import numpy as np
from astropy.io import fits

from clammy import paths

PYPEIT_R = 120000.0
DEFAULT_TELLPCA = os.path.join(paths.ROOT, "grid", "telluric",
                               "TellPCA_3000_10500_R120000.fits")


def vac_to_air(wave_vac):
    """Vacuum -> air wavelengths (Angstrom), matching ``process_phoenix_templates``."""
    s2 = (1.0e4 / np.asarray(wave_vac, float)) ** 2
    n = (1.0 + 0.0000834254 + 0.02406147 / (130.0 - s2)
         + 0.00015998 / (38.9 - s2))
    return wave_vac / n


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tellpca", default=DEFAULT_TELLPCA,
                    help="PypeIt TellPCA_*.fits model (default: grid/telluric/...)")
    ap.add_argument("-o", "--out",
                    default=os.path.join(paths.ROOT, "grid", "telluric", "telluric_grid.npz"),
                    help="output .npz of reconstructed spectra")
    ap.add_argument("--n-spectra", type=int, default=0,
                    help="number of spectra to reconstruct (0 = all in the file)")
    ap.add_argument("--wmin", type=float, default=None, help="trim to >= this AIR wavelength [A]")
    ap.add_argument("--wmax", type=float, default=None, help="trim to <= this AIR wavelength [A]")
    ap.add_argument("--floor", type=float, default=1e-8,
                    help="floor on transmission (keeps saturated cores > 0 for ln)")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                    help="storage dtype for the flux matrix (float32 halves the size)")
    ap.add_argument("--no-vac-to-air", action="store_true",
                    help="keep vacuum wavelengths (skip the vacuum->air conversion)")
    ap.add_argument("--no-compress", action="store_true", help="write uncompressed .npz")
    args = ap.parse_args(argv)

    # --- TellPCA model --------------------------------------------------------------
    with fits.open(args.tellpca) as h:
        pca = np.asarray(h[0].data, float)         # (1 + n_comp, n_pix), arsinh(tau)
        wave_vac = np.asarray(h[1].data, float)    # (n_pix,), vacuum Angstrom
        coeff = np.asarray(h[3].data, float)       # (n_train, n_comp)
        n_comp = int(h[0].header["NCOMP"])
    n_train = coeff.shape[0]
    print(f"TellPCA: {n_comp} comps, {wave_vac.size} px, "
          f"{wave_vac[0]:.1f}-{wave_vac[-1]:.1f} A vacuum, {n_train} training vectors")

    # --- wavelengths: vacuum -> air -> natural-log, trimmed, ascending --------------
    wave_air = wave_vac if args.no_vac_to_air else vac_to_air(wave_vac)
    keep = np.ones(wave_air.size, bool)
    if args.wmin is not None:
        keep &= wave_air >= args.wmin
    if args.wmax is not None:
        keep &= wave_air <= args.wmax
    wave_air, pca = wave_air[keep], pca[:, keep]
    loglam = np.log(wave_air)
    srt = np.argsort(loglam)
    loglam, pca = loglam[srt], pca[:, srt]

    # --- reconstruct (chunked matmul to bound peak memory) --------------------------
    n = n_train if args.n_spectra in (0, None) or args.n_spectra >= n_train else args.n_spectra
    coeff = coeff[:n]
    dt = np.float32 if args.dtype == "float32" else np.float64
    full = np.hstack([np.ones((n, 1)), coeff])      # (n, 1 + n_comp); the leading 1 = mean
    flux = np.empty((n, loglam.size), dtype=dt)
    print(f"reconstructing {n} of {n_train} spectra x {loglam.size} px ({args.dtype}) ...")
    chunk = max(1, int(2e8 // loglam.size))         # ~1.6 GB float64 transient per chunk
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        arsinh_tau = full[s:e] @ pca                # (rows, n_pix)
        tau = np.sinh(arsinh_tau)
        np.clip(tau, 0.0, None, out=tau)
        T = np.exp(-tau)
        np.clip(T, args.floor, None, out=T)
        flux[s:e] = T.astype(dt, copy=False)

    # --- save -----------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    payload = dict(wave=loglam.astype(np.float64), flux=flux,
                   coeff=coeff.astype(np.float64), resolution=np.float64(PYPEIT_R))
    (np.savez if args.no_compress else np.savez_compressed)(args.out, **payload)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {n} spectra x {loglam.size} px -> {args.out}  ({size_mb:.1f} MB)")
    print(f"  next: clammy build --kind telluric --templates {args.out} "
          f"--resolution {PYPEIT_R:.0f} -o outputs/telluric_basis.npz")


if __name__ == "__main__":
    main()
