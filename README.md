# Linear Models for Stellar Nuisances

Analysing stellar spectra requires handling of various nuisances: effects that may
not be related to our primary goal, but would bias our analysis if we ignored them.
Depending on your science goal, those nuisances might include: continuum-normalization,
radial velocity shifts, instrumental broadening, and rotational broadening.

This code uses linear models to simultaneously fit: radial velocity; stellar absorption
(from a low-dimensional representation); continuum; and broadening.
The linear algebra, the FFTs, and the optimisation all run on `jax`.

A full derivation (linear template fit → RV scan → FFT acceleration → joint
RV+weights and its convexity → continuum → convolution/R) is in
[`docs/formulation.tex`](docs/formulation.pdf), with strict consistent notation.

## Principle

Resample to a uniform **log-wavelength** grid `x = ln λ`, so a radial velocity `v`
acts as a pure translation `T(x) → T(x − Δx)` with `Δx = ln(1 + v/c)`. Working in
**log-flux** makes the (continuum × rectified-flux) model additive:

```
ln d(x) ≈ Σ_k w_k · T_k(x − Δx)  +  Σ_m c_m · P_m(x)
          └─ rectified templates ─┘   └─ Legendre log-continuum ─┘
```

The templates `T_k = [μ, Φ_1, …, Φ_K]` are the basis of *rectified* model grid 
(e.g., perhaps a mean and PCA); `P_m` are Legendre polynomials. At fixed `Δx` the model is
**linear** in `θ = (w, c)`, so we **solve linearly inside, scan `Δx` outside**:

```
N(Δx) = AᵀΣ⁻¹A = [[M_TT  M_TP],[M_TPᵀ  M_PP]],   r(Δx) = AᵀΣ⁻¹ ln d
θ(Δx) = N⁻¹ r,   χ²(Δx) = ‖ln d‖²_Σ⁻¹ − rᵀ N⁻¹ r
```

with `Σ⁻¹` the inverse variances of `ln d` (`σ_lnd ≈ σ_d/d`; masked px → weight 0).

**Speed**: `b_T(Δx)` and `M_TP(Δx)` are cross-correlations of fixed vectors against
a shifted template → computed for **all shifts at once via FFT**. `M_PP`, `b_P` are
shift-independent; `M_TT` is only weakly shift-dependent (through the weights) so
`M_TT(0)` is used during the coarse scan, then `M_TT` is recomputed **exactly** at
the refined shift. Circular-FFT wrap is eliminated by zeroing the data weights in a
band of width = max lag at each edge, making the circular correlation equal to the
linear one over the valid region.

**Resolution / convolution**: at constant resolving power `R` the LSF is a Gaussian
of constant width *in velocity* → constant pixels on the log grid → one Fourier
multiply `T̂(ω) → T̂(ω)·exp(−2π²ω²σ_pix²)`, with `σ_pix = c/(2.355·velscale·R)`. It can
be **fixed** (known LSF, pre-broaden once) or **fit jointly with v** (`fit_resolution=True`)
as a second nonlinear parameter — the model stays linear in `(w, c)` at fixed
`(Δx, R)`. Caveat: the LSF convolves *linear* flux, but to keep the inner solve
linear we broaden the *log*-rectified templates; this matches the exact linear-flux
convolution to first order and slightly over-deepens saturated cores (the toy uses
the exact linear-flux convolution, so the validation measures the residual ~2% `R`
bias). For wavelength-dependent `R`/non-Gaussian LSFs, precompute banded convolution
matrices on an `R`-grid and interpolate.

**Refinement / optimisation**: a coarse FFT scan over `Δx` (and a coarse `R` grid)
locates the global basin (the profiled χ² is multimodal in the nonlinear
parameters, though convex in `(w,c)` at fixed `Δx,R`); then a **JAX-autodiff
modified-Newton** with a backtracking Armijo line search polishes `(Δx[, R])` on the
*profiled* χ². The parameter covariance is the autodiff Hessian of the profiled χ²
at the optimum (Δχ² = 1), propagated to `(v, R)`. Typical warm cost ≈ 2 s/fit (CPU;
the bottleneck is the 216k-point FFT, ~10× faster on GPU).

## Layout

`clammy` is a src-layout Python package; install it editable (`pip install -e .`)
and you get both the `clammy` import and the `clammy` CLI.

```
marla-geha/
├─ pyproject.toml             package metadata + the `clammy` console script
├─ .venv/                     uv venv (clammy installed -e, + numpy scipy astropy matplotlib jax)
├─ src/clammy/                the package (grid, basis, toy, fitter, cli, paths)
├─ scripts/                   drivers: build_basis.py, validate.py, validate_resolution.py,
│                             download_phoenix_templates.sh, process_phoenix_templates.py
├─ outputs/                   basis.npz + all diagnostic/validation plots
├─ docs/                      formulation.tex/pdf
└─ grid/
   ├─ original/               raw PHOENIX HiRes: lte*.fits, WAVE_*.fits
   └─ convolved/              smoothed + log-rebinned: dmost_lte_*.fits
```

All paths are derived from `clammy/paths.py` (overridable via the
`ORIGINAL_DIR` / `CONVOLVED_DIR` / `OUTPUTS_DIR` env vars), so code runs from any cwd.

Install:

```bash
uv pip install -e .            # or: pip install -e .   (add [templates] for ppxf)
```

## Data

198 PHOENIX-ACES synthetic spectra (`convolved/dmost_lte_{teff}_{logg}_{feh}_.fits`),
already log-rebinned to one shared grid: **216,058 px, 6000–9600 Å (air), 0.652
km/s/px**. Coverage: Teff ∈ {2500…8000 K} (11), log g ∈ {1, 3, 5}, [Fe/H] ∈ {0,
−0.5, −1, −2, −3, −4}. The raw HiRes inputs live in `original/`. Downloaded by
`download_phoenix_templates.sh` and convolved/resampled by
`process_phoenix_templates.py` (`original/ → convolved/`).

## Package `clammy/`

| module | task | what it does |
|--------|------|--------------|
| `grid.py`   | —      | load the grid into a `(n_spec, n_pix)` flux matrix on the shared log-λ grid |
| `basis.py`  | Task 1 | robust log-space continuum rectification + PCA (Gram-matrix trick); save Φ, μ, grid |
| `toy.py`    | Task 2 | Fourier sub-pixel RV shift + linear-flux R-broadening + Legendre log-continuum + noise; optional resample to a different grid |
| `fitter.py` | Task 3 | the JAX fitter: FFT cross-correlations, `vmap`'d normal-equation scan, joint `(v[,R])` autodiff modified-Newton refine + covariance |
| `cli.py`    | —      | the `clammy build` / `clammy fit` command-line interface |
| `paths.py`  | —      | single source-of-truth for `original/`, `convolved/`, `outputs/` (env-overridable) |

Rectification removes the smooth pseudo-continuum so PCA captures **lines**, while
the fitter's Legendre block carries the continuum — keeping the two roles separate
avoids the template/continuum degeneracy. The mean spectrum μ is a free-weight
template, so it is RV-shifted like any component.

## Usage

### CLI

```bash
# build a template basis from a spectral grid
clammy build --templates grid/convolved --basis pca --var 0.99 --k-max 30 -o outputs/basis.npz

# fit a spectrum (.npz or FITS table; wavelength linear-Å or log, auto-detected)
clammy fit outputs/basis.npz spectrum.npz                       # RV + weights + continuum
clammy fit outputs/basis.npz spectrum.fits --fit-resolution \
           --vmin -300 --vmax 300 --plot fit.png -o fit.json    # also fit R
```

`fit` reads `wave`/`flux`/`sigma` columns (override with `--wave-col` etc.; use
`--snr` if there is no uncertainty column), resamples onto the basis grid if
needed, and prints `v[, R]`, χ²/dof, weights, and continuum coefficients.

### Scripts (drivers / reproduce the figures)

```bash
# (re)build the PHOENIX grid:  original/ -> convolved/   (needs the [templates] extra)
scripts/download_phoenix_templates.sh

python scripts/build_basis.py --var 0.99 --order 5     # basis + diagnostic plots
python scripts/validate.py                             # RV/weights/continuum recovery
python scripts/validate_resolution.py                  # joint (v,R) + different-grid handling
```

### Python API

```python
from clammy import grid, basis, toy, fitter, paths
b = basis.load_basis(f"{paths.OUTPUTS}/basis.npz")

# RV + weights + continuum
d, sigma, truth = toy.make_toy(b["loglam"], r_true, v_kms=42.0,
                               cont_coef=[0.6,-0.3,0.1], snr=25)
res = fitter.fit_rv(d, sigma, b["loglam"], b["mu"], b["Phi"],
                    cont_order=3, vmin=-400, vmax=400)
res["v_kms"], res["v_err_kms"], res["w"], res["c"], res["cov"], res["chi2_dof"]

# also fit the spectral resolution R (degraded + on any grid)
d, sigma, truth = toy.make_toy(b["loglam"], r_true, v_kms=42.0,
                               cont_coef=[0.6,-0.3,0.1], snr=50, resolution_R=6000.)
res = fitter.fit_rv(d, sigma, b["loglam"], b["mu"], b["Phi"],
                    cont_order=3, fit_resolution=True)
res["v_kms"], res["resolution_R"], res["R_err"], res["cov_vR"]
```


## Validation summary

Leave-one-out (basis never saw the test star; metal-poor RGB, Teff 4500 / log g 1 / [Fe/H] −2):

- **RV** — bias ≤ 10 m/s at SNR 5, errors well-calibrated (realised scatter ≈ predicted), χ²/dof ≈ 1 at low SNR; at high SNR χ²/dof rises (the expected **PCA-truncation floor**, since K=14 components cannot perfectly represent a held-out star) while RV bias stays ~1–2 m/s. RV-vs-velocity sweep (−300…+300 km/s): max\|bias\| = 3 m/s. *(In-basis truth gives RV bias < 0.4 m/s and χ²/dof = 1 at all SNR.)*
- **Continuum / basis** — continuum order ≥ 3 suffices (χ²/dof plateaus); larger K lowers the truncation floor (χ²/dof: 3.0 at K=1 → 1.6 at K=14 → 1.1 at K=30).
- **Resolution** — `v` and `R` are nearly orthogonal (ρ_vR ≈ 0) and errors are well-calibrated (realised scatter ≈ predicted for both). Two bias regimes: with an *in-basis* truth, `R` is biased only **−2%** (the log-space-vs-linear-flux convolution approximation; a log-broadened toy is recovered to 0.1%). With a *held-out* star (realistic template mismatch), `R` is biased **+11…16%** and freeing `R` also leaks mismatch into `v` (tens–hundreds of m/s, dominating at high SNR). **Takeaway: R from this fit is degenerate with template completeness — trust it to ~15% unless the basis represents the target well.** A different/coarser observed grid is handled by resampling onto the template grid (point estimates recovered; resampling correlates noise, so χ²/dof and error bars need the proper covariance).

## References

- Tonry & Davis (1979) — cross-correlation RVs.
- Cappellari (2017) — pPXF (additive + multiplicative polynomials; canonical implementation).
- Bolton et al. (2012) — BOSS PCA-template + redshift-scan classification.
- Geha et al. (2026) - DIEMOS analysis
