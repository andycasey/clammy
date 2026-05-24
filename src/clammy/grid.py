"""Load the processed PHOENIX (dmost) grid onto the shared log-lambda grid.

The ``dmost_lte_{teff}_{logg}_{feh}_.fits`` files (produced by
``process_phoenix_templates.py``) are single-row astropy tables with columns
``wave`` (natural-log wavelength, uniform spacing) and ``flux`` (linear,
*not* rectified), plus the scalar labels ``teff``, ``logg``, ``feh``.

All files share one identical log-lambda grid, so we load them into a single
``(n_spec, n_pix)`` flux matrix.

For grids that carry no stellar labels (e.g. telluric model grids, which live
in the observed/topocentric frame rather than any stellar rest frame), use the
label-agnostic :func:`load_spectra` instead of :func:`load_grid`.
"""
import glob
import os
import re

import numpy as np
from astropy.table import Table

_FNAME_RE = re.compile(r"dmost_lte_(\d+)_([0-9.]+)_(-?[0-9.]+)_\.fits$")


def list_grid(templ_dir="."):
    """Return ``[(path, teff, logg, feh), ...]`` sorted by filename."""
    rows = []
    for f in sorted(glob.glob(os.path.join(templ_dir, "dmost_lte_*.fits"))):
        m = _FNAME_RE.search(os.path.basename(f))
        if m is None:
            continue
        rows.append((f, float(m.group(1)), float(m.group(2)), float(m.group(3))))
    if not rows:
        raise FileNotFoundError(f"no dmost_lte_*.fits in {templ_dir!r}")
    return rows


def load_one(path):
    """Return ``(loglam, flux)`` for a single dmost file."""
    t = Table.read(path)
    return np.asarray(t["wave"][0], float), np.asarray(t["flux"][0], float)


def load_grid(templ_dir=".", exclude=None):
    """Load the whole grid.

    Parameters
    ----------
    templ_dir : str
        Directory holding the ``dmost_lte_*.fits`` files.
    exclude : set[str] | None
        Basenames to leave out (e.g. a held-out test spectrum).

    Returns
    -------
    loglam : (n_pix,) ndarray         shared natural-log wavelength grid
    flux   : (n_spec, n_pix) ndarray  linear flux
    params : dict[str, ndarray]       'teff', 'logg', 'feh', each (n_spec,)
    files  : list[str]                basenames, aligned with rows of `flux`
    """
    exclude = set(exclude or ())
    rows = [r for r in list_grid(templ_dir) if os.path.basename(r[0]) not in exclude]

    loglam = None
    flux = np.empty((len(rows), 0))
    fluxes = []
    for path, *_ in rows:
        w, fl = load_one(path)
        if loglam is None:
            loglam = w
        elif w.shape != loglam.shape or not np.allclose(w, loglam, atol=0, rtol=1e-12):
            raise ValueError(f"{path} is not on the shared log-lambda grid")
        fluxes.append(fl)

    flux = np.asarray(fluxes)
    params = {
        "teff": np.array([r[1] for r in rows]),
        "logg": np.array([r[2] for r in rows]),
        "feh": np.array([r[3] for r in rows]),
    }
    files = [os.path.basename(r[0]) for r in rows]
    return loglam, flux, params, files


def _load_one_spectrum(path, wave_col, flux_col):
    """Return ``(loglam, flux)`` for a single file, FITS or ``.npz``.

    Both backends are handled the same way as elsewhere in the package:

    * ``.npz`` archives store the wavelength/flux columns as plain arrays under
      ``wave_col``/``flux_col``.
    * FITS files are read as astropy tables. The dmost convention packs the
      whole spectrum into a single array-valued cell of a one-row table (see
      ``load_one`` and ``cli._column``); ordinary one-value-per-row tables are
      also supported. We auto-detect the single-row array-cell layout and
      unwrap it, otherwise we take the column as-is.
    """
    if str(path).endswith(".npz"):
        with np.load(path, allow_pickle=False) as d:
            return np.asarray(d[wave_col], float), np.asarray(d[flux_col], float)

    t = Table.read(path)

    def _col(name):
        # dmost-style single-row table with array-valued cells -> unwrap row 0;
        # otherwise treat the column as one value per pixel.
        if len(t) == 1 and np.ndim(t[name][0]) > 0:
            return np.asarray(t[name][0], float)
        return np.asarray(t[name], float)

    return _col(wave_col), _col(flux_col)


def _load_spectra_matrix(path, wave_col, flux_col):
    """Load a whole grid from ONE file: ``(n_pix,)`` wave + ``(n_spec, n_pix)`` flux.

    The natural single-file format for a model grid (e.g. a reconstructed telluric
    spectrum set). ``.npz`` keys the arrays by ``wave_col``/``flux_col``; a FITS
    table is read with the same columns. A 1-D flux (a single spectrum) is promoted
    to one row, and a transposed ``(n_pix, n_spec)`` matrix is tolerated.
    """
    if str(path).endswith(".npz"):
        with np.load(path, allow_pickle=False) as d:
            w = np.asarray(d[wave_col], float)
            fl = np.asarray(d[flux_col], float)
    else:
        t = Table.read(path)
        wv = t[wave_col]
        w = np.asarray(wv[0] if (len(t) == 1 and np.ndim(wv[0]) > 0) else wv, float)
        fl = np.asarray(t[flux_col], float)
    fl = np.atleast_2d(fl)
    if fl.shape[1] != w.shape[0] and fl.shape[0] == w.shape[0]:
        fl = fl.T  # tolerate a (n_pix, n_spec) matrix
    if fl.shape[1] != w.shape[0]:
        raise ValueError(
            f"{path}: flux shape {fl.shape} is incompatible with wave {w.shape}"
        )
    if not np.all(np.isfinite(fl)) or np.any(fl <= 0.0):
        raise ValueError(
            f"{path}: flux must be finite and strictly positive for the "
            "log-rectification step (got non-positive or non-finite values)"
        )
    files = [f"{os.path.basename(path)}[{i}]" for i in range(fl.shape[0])]
    return w, fl, files


def load_spectra(templ_dir=".", pattern="*.fits", wave_col="wave", flux_col="flux"):
    """Load an unlabelled grid of spectra onto a shared log-lambda grid.

    This is the label-agnostic sibling of :func:`load_grid`. Where
    ``load_grid`` parses stellar labels (``teff``/``logg``/``feh``) out of the
    ``dmost_lte_*.fits`` filenames, ``load_spectra`` makes no assumption about
    filenames or metadata: it simply ingests every file matching ``pattern``
    and stacks their flux. This is what a *telluric* grid needs, since telluric
    models carry no stellar labels (and, conceptually, no rest-frame identity at
    all -- they live in the observed/topocentric frame).

    Parameters
    ----------
    templ_dir : str
        Directory to glob for spectra.
    pattern : str
        Glob pattern relative to ``templ_dir`` (e.g. ``"*.fits"`` or
        ``"telluric_*.npz"``).
    wave_col, flux_col : str
        Column/array names holding the natural-log wavelength and the linear
        flux. FITS tables may use the dmost single-row array-cell layout (see
        :func:`load_one` / ``cli._column``); ``.npz`` archives key plain arrays
        by these names.

    Returns
    -------
    loglam : (n_pix,) ndarray         shared natural-log wavelength grid
    flux   : (n_spec, n_pix) ndarray  linear flux, strictly positive
    files  : list[str]                basenames, aligned with rows of `flux`

    Notes
    -----
    As with :func:`load_grid`, every file must lie on one identical log-lambda
    grid (checked to ``rtol=1e-12``). Because the downstream rectification step
    takes ``ln(flux)``, the flux must be strictly positive; we validate this and
    raise a clear error rather than silently producing NaNs/infs. No parameter
    arrays are returned -- pass ``params=None`` to ``basis.save_basis`` for an
    unlabelled (telluric) basis.
    """
    # A single multi-spectrum file (an .npz/FITS holding a (n_spec, n_pix) flux
    # matrix plus a (n_pix,) wavelength array) is accepted directly: the natural
    # one-file format for a model grid such as a reconstructed telluric spectrum
    # set, avoiding an explosion into thousands of per-spectrum files.
    if os.path.isfile(templ_dir):
        return _load_spectra_matrix(templ_dir, wave_col, flux_col)

    files_full = sorted(glob.glob(os.path.join(templ_dir, pattern)))
    if not files_full:
        raise FileNotFoundError(f"no files matching {pattern!r} in {templ_dir!r}")

    loglam = None
    fluxes = []
    file_labels = []
    for path in files_full:
        # An npz holding a 2-D flux matrix (the natural single-file format for a
        # model grid like the PypeIt TellPCA reconstruction) is loaded via the
        # matrix loader so the full (n_spec, n_pix) block is ingested at once.
        if str(path).endswith(".npz"):
            w, fl_mat, lbls = _load_spectra_matrix(path, wave_col, flux_col)
            if loglam is None:
                loglam = w
            elif w.shape != loglam.shape or not np.allclose(w, loglam, atol=0, rtol=1e-12):
                raise ValueError(f"{path} is not on the shared log-lambda grid")
            for row, lbl in zip(fl_mat, lbls):
                fluxes.append(row)
                file_labels.append(lbl)
            continue

        w, fl = _load_one_spectrum(path, wave_col, flux_col)
        if loglam is None:
            loglam = w
        elif w.shape != loglam.shape or not np.allclose(w, loglam, atol=0, rtol=1e-12):
            raise ValueError(f"{path} is not on the shared log-lambda grid")
        if not np.all(np.isfinite(fl)) or np.any(fl <= 0.0):
            raise ValueError(
                f"{path}: flux must be finite and strictly positive for the "
                "log-rectification step (got non-positive or non-finite values)"
            )
        fluxes.append(fl)
        file_labels.append(os.path.basename(path))

    flux = np.asarray(fluxes)
    return loglam, flux, file_labels
