"""Load the processed PHOENIX (dmost) grid onto the shared log-lambda grid.

The ``dmost_lte_{teff}_{logg}_{feh}_.fits`` files (produced by
``process_phoenix_templates.py``) are single-row astropy tables with columns
``wave`` (natural-log wavelength, uniform spacing) and ``flux`` (linear,
*not* rectified), plus the scalar labels ``teff``, ``logg``, ``feh``.

All files share one identical log-lambda grid, so we load them into a single
``(n_spec, n_pix)`` flux matrix.
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
