#!/usr/bin/env python
"""
Smooth + trim downloaded PHOENIX HiRes templates for DEIMOS / dmost.

Mirrors the "SMOOTH AND TRIM FOR DEIMOS DMOST" section of
prepare_pheonix_templates.ipynb: read the shared (vacuum) wavelength grid,
convert to air, trim to the DEIMOS range, then for each lte*.fits log-rebin to
log-lambda and write dmost_lte_{teff}_{logg}_{feh}_.fits.

Reads raw HiRes templates (lte*.fits + WAVE_*.fits) from grid/original/
and writes the convolved/resampled dmost_lte_*.fits to grid/convolved/.

Environment variables (see clammy.paths):
  ORIGINAL_DIR   input dir of lte*.fits + WAVE_*.fits (default: grid/original)
  CONVOLVED_DIR  output dir for dmost_lte_*.fits      (default: grid/convolved)
  CLOBBER        set to 1 to overwrite existing outputs (default: 0)
  SMOOTH         set to 1 to log-rebin the gaussian-smoothed spectrum.
                 The notebook computes `smooth` but log-rebins the *unsmoothed*
                 data, so the default (0) reproduces the notebook's actual output.
"""
import os
import glob
import numpy as np
from astropy.io import fits
from astropy.table import Table
import scipy.ndimage as scipynd
import ppxf.ppxf_util as ppxf

from clammy import paths

ORIG_DIR = paths.ORIGINAL
OUT_DIR = paths.CONVOLVED
WAVE_FILE = os.path.join(ORIG_DIR, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
CLOBBER = bool(int(os.environ.get("CLOBBER", "0")))
SMOOTH = bool(int(os.environ.get("SMOOTH", "0")))


def convert_zfeh(feh):
    return 0.02 * 10**(feh)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # READ UNIVERSAL WAVELENGTH FILE (vacuum)
    vwave = fits.open(WAVE_FILE)[0].data

    # CONVERT TO AIR WAVELENGTHS
    s2 = (1.e4 / vwave)**2
    wave = vwave / (1. + 0.0000834254 + (0.02406147 / (130. - s2))
                    + (0.00015998 / (38.9 - s2)))

    # TRIM TO DEIMOS WAVELENGTH RANGE
    mdeimos = (wave >= 6000) & (wave <= 9600)
    wave_deimos = wave[mdeimos]

    files = sorted(glob.glob(os.path.join(ORIG_DIR, "lte*.fits")))
    print("processing {} templates from {}".format(len(files), ORIG_DIR))

    for f in files:
        hdu = fits.open(f)
        data = hdu[0].data
        header = hdu[0].header

        # PARSE HEADER
        logg = header['PHXLOGG']
        teff = header['PHXTEFF']
        feh = header['PHXM_H']
        iso = '{:0.6f}'.format(convert_zfeh(feh))

        outfile = os.path.join(
            OUT_DIR,
            'dmost_lte_{:0.0f}_{}_{}_.fits'.format(teff, logg, feh))

        if os.path.isfile(outfile) and not CLOBBER:
            print("skip  {}".format(outfile))
            continue

        # TRIM
        data_deimos = data[mdeimos]
        # SMOOTH
        smooth = scipynd.gaussian_filter1d(data_deimos, 3)

        spec = smooth if SMOOTH else data_deimos

        # REBIN TO LOG LAMBDA
        # NB: the notebook passed 0.6 positionally; in old ppxf that was
        # `oversample`, but in current ppxf (>=7) the 3rd positional is
        # `velscale`. Use the keyword so we reproduce the notebook exactly
        # (oversample=0.6 -> int(N*0.6) pixels, vscale~0.652 km/s).
        logSpec, logLam, vscale = ppxf.log_rebin(
            [np.min(wave_deimos), np.max(wave_deimos)], spec, oversample=0.6)

        ttable = Table([[logLam], [logSpec], [logg], [teff], [feh], [iso], [vscale]],
                       names=('wave', 'flux', 'logg', 'teff', 'feh', 'Z', 'vscale'))
        ttable.write(outfile, overwrite=CLOBBER)
        print("wrote {}  (vscale={})".format(outfile, vscale))


if __name__ == "__main__":
    main()
