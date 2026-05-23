# Regenerating the telluric basis from the PypeIt TellPCA product

The telluric basis used by `clammy fit --telluric-basis` is derived from PypeIt's
**TellPCA** atmospheric model. None of the intermediate data is committed (it is all
under the `grid/telluric/` and `outputs/*.npz` gitignore rules); regenerate it with
the three steps below.

## 0. Download the PypeIt TellPCA model

File: `TellPCA_3000_10500_R120000.fits` (3000–10500 Å, R=120,000, ~36 MiB).
Save it to `grid/telluric/`.

- Easiest: install PypeIt (in a throwaway env, not the project `.venv`) and use its
  downloader, which resolves the current cache host:

  ```bash
  pip install pypeit            # in a scratch venv
  pypeit_install_telluric TellPCA_3000_10500_R120000.fits
  # then copy the cached file into grid/telluric/
  ```

  Equivalent Python: `from pypeit import cache;
  cache.fetch_remote_file("TellPCA_3000_10500_R120000.fits", "telluric/atm_grids",
  remote_host="s3_cloud", install_script=True)`.

- Direct (host may change): `https://s3-west.nrp-nautilus.io/pypeit/telluric/atm_grids/TellPCA_3000_10500_R120000.fits`
  (sha256 `3521a29ac2849b75f5bbacdf7b23aa934e6ef3ad8a04b0efe95ec03e84f666b0`).

Source/citation: Noll et al. 2012 (A&A 543, A92) for the underlying LBLRTM/HITRAN sky
model; the PCA grid ships with PypeIt (BSD-3). It is a 10-component model: HDU0 row 0
is the mean and rows 1–10 the eigenvectors, all in **arsinh(optical depth)** (not
transmission); HDU1 is the log-spaced **vacuum** wavelength grid; HDU3 holds 5000
training coefficient vectors.

## 1. Reconstruct the transmission spectra → one grid file

`scripts/reconstruct_telluric_grid.py` evaluates the model into physical transmission
and applies the corrections clammy needs:

- arsinh(τ) → transmission: `T = exp(-max(0, sinh(mean + c · eigvecs)))` for each of
  the 5000 training coefficient vectors `c`;
- **vacuum → air** wavelengths (the dmost/DEIMOS stellar grid is in air), using the
  same dispersion relation as `process_phoenix_templates.py`;
- a small positive floor so the downstream `ln` stays finite.

```bash
python scripts/reconstruct_telluric_grid.py --wmin 5900 --wmax 9700 \
    -o grid/telluric/telluric_grid.npz
```

This reconstructs **all 5000** spectra (default `--n-spectra 0`) trimmed to the DEIMOS
window (a little wider than the 6000–9600 Å stellar grid so resampling has margin),
as a single `.npz` (`wave` = natural-log air λ, `flux` = (5000, n_pix) transmission,
`coeff` = per-spectrum coefficients, `resolution` = 120000). Output ≈ 1.8 GB
(`float32`, ~149k px). Useful flags: `--wmin/--wmax`, `--n-spectra`, `--floor`,
`--dtype`, `--no-vac-to-air`.

The `.npz` is deliberately a *method-agnostic* spectrum set — feed it to a PCA below,
or factorise it some other way (e.g. NMF) later.

## 2. Build the PCA basis

```bash
clammy build --kind telluric --templates grid/telluric/telluric_grid.npz \
    --resolution 120000 --var 0.999 -o outputs/telluric_basis.npz
```

`build_telluric_basis` rectifies in log-flux and runs PCA; the result is stored with
`frame="observed"` (never RV-shifted) and `resolution=120000` (used for differential
broadening at fit time). With all 5000 spectra over the trimmed window this keeps
**K=14** components for ~99.91% variance, → `outputs/telluric_basis.npz` (~18 MB).

Memory: the rectify + Gram-matrix path holds several `(5000 × n_pix)` float64 copies
(~24 GB peak for the trimmed window). On less RAM, narrow the window further or use
`--n-spectra` to subsample (the model is only 10-D, so a few hundred draws still span
it well).

## 3. Use it in a fit

```bash
clammy fit outputs/basis.npz spectrum.fits --telluric-basis outputs/telluric_basis.npz
```

The telluric block enters the observed (topocentric) frame, instrument-broadened only
and not RV-shifted, and is resampled onto the fitting grid automatically.
