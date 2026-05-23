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

Resample to a uniform log-wavelength grid $`x = \ln\lambda`$, so a radial
velocity $`v`$ acts as a pure translation $`T_k(x) \to T_k(x - \Delta x)`$ with
$`\Delta x = \ln(1 + v/c)`$. Working in log-flux $`Y \equiv \ln d`$ makes the
(continuum $`\times`$ rectified-flux) model additive:

```math
Y(x)  \approx  \underbrace{\sum_{k=0}^{K} w_k T_k(x - \Delta x)}_{\text{rectified line basis}}  +  \underbrace{\sum_{m=0}^{L} c_m P_m(x)}_{\text{log-continuum}}
```

The templates $`\mathbf{T} = [\mu, \phi_1, \dots, \phi_K]`$ are the basis of the
*rectified* model grid (e.g. a mean $`\mu \equiv T_0`$ and PCA components
$`\phi_k \equiv T_k`$); $`P_m`$ are Legendre polynomials. At fixed $`\Delta x`$ the model
is linear in $`\boldsymbol\theta = (\mathbf{w}, \mathbf{c})`$, so we solve linearly
inside, scan $`\Delta x`$ outside. With the design
$`\mathbf{A}(\Delta x) = [ \mathbf{T}(\Delta x) \mid \mathbf{P} ]`$ and the data
covariance $`\mathbf{C} = \mathrm{diag}(\sigma_{Y,i}^2)`$ of $`Y`$ ($`\sigma_{Y,i} \approx \sigma_i/d_i`$;
masked px $`\to (\mathbf{C}^{-1})_{ii} = 0`$):

```math
\mathbf{N}(\Delta x) = \mathbf{A}^{\mathsf{T}}\mathbf{C}^{-1}\mathbf{A} = \begin{bmatrix} \mathbf{M}_{TT} & \mathbf{M}_{TP} \\ \mathbf{M}_{TP}^{\mathsf{T}} & \mathbf{M}_{PP} \end{bmatrix} \qquad \mathbf{r}(\Delta x) = \mathbf{A}^{\mathsf{T}}\mathbf{C}^{-1}\mathbf{Y}
```

```math
\widehat{\boldsymbol\theta}(\Delta x) = \mathbf{N}^{-1}\mathbf{r} \qquad \chi^2(\Delta x) = \mathbf{Y}^{\mathsf{T}}\mathbf{C}^{-1}\mathbf{Y} - \mathbf{r}^{\mathsf{T}}\mathbf{N}^{-1}\mathbf{r}
```

Here $`\mathbf{C}`$ is the diagonal data covariance of $`Y`$
($`\sigma_Y \approx \sigma_d/d`$; masked px get zero weight, $`(\mathbf{C}^{-1})_{ii} = 0`$),
so $`\mathbf{C}^{-1}`$ is the inverse-variance weight. The radial-velocity estimate is the global minimiser
$`\widehat{\Delta x} = \arg\min_{\Delta x} \chi^2(\Delta x)`$, with
$`\widehat v = c (e^{\widehat{\Delta x}} - 1)`$.

**Speed**: both $`\mathbf{b}_T(\Delta x)`$ and $`\mathbf{M}_{TP}(\Delta x)`$ are
cross-correlations of fixed vectors against a shifted template → computed for all
shifts at once via FFT. $`\mathbf{M}_{PP}`$, $`\mathbf{b}_P`$ are shift-independent;
$`\mathbf{M}_{TT}`$ is only weakly shift-dependent (through the weights) so
$`\mathbf{M}_{TT}(0)`$ is used during the coarse scan, then $`\mathbf{M}_{TT}`$ is
recomputed exactly at the refined shift. Circular-FFT wrap is eliminated by zeroing
the data weights in a band of width = max lag at each edge, making the circular
correlation equal to the linear one over the valid region.

**Resolution / convolution**: at constant resolving power $`R`$ the LSF is a Gaussian
of constant width *in velocity* → constant pixels on the log grid → one Fourier
multiply $`\widehat T_k(\omega) \to \widehat T_k(\omega) \exp(-2\pi^2\omega^2\sigma_{\rm pix}^2)`$,
with $`\sigma_{\rm pix} = c/(f \mathrm{velscale} R)`$, where $`\mathrm{velscale} = c \delta`$
is the velocity per pixel and $`f = 2\sqrt{2\ln 2}`$. It can be fixed (known LSF,
pre-broaden once) or fit jointly with $`v`$ (`fit_resolution=True`) as a second
nonlinear parameter — the model stays linear in $`(\mathbf{w}, \mathbf{c})`$ at fixed
$`(\Delta x, R)`$. Caveat: the LSF convolves *linear* flux, but to keep the inner
solve linear we broaden the *log*-rectified templates; this matches the exact
linear-flux convolution to first order and slightly over-deepens saturated cores
(the toy uses the exact linear-flux convolution, so the validation measures the
residual ~2% $`R`$ bias). This core bias is removable with `--conv-iter N`
(`n_conv_iter`): a fixed-point that subtracts the exact-minus-log convolution
correction $`\delta = \ln(\mathrm{LSF}\otimes e^{M}) - \mathrm{LSF}\!\cdot\!M`$ from
the data and re-fits, so the *exact* linear-flux model matches the data and $`R`$ is
unbiased — on an in-basis test χ²/dof falls ~40 → ~1 and the deep-core residuals
~60σ → ~4σ (it converges in ~8–12 iterations, so it is off by default). For
wavelength-dependent $`R`$/non-Gaussian LSFs, precompute banded convolution matrices
on an $`R`$-grid and interpolate.

### Extended forward model: tellurics, rotation, and native resolution

The same separable structure absorbs three further nuisances. The full log-flux
forward model is

```math
Y(x) \approx \underbrace{G_{\rm rot}(v\sin i) G_{\rm inst}}_{\text{stellar}} \sum_{k=0}^{K} w_k T_k(x - \Delta x)  +  \underbrace{G_{\rm inst}}_{\text{telluric}} \sum_{j=0}^{J} u_j S_j(x)  +  \sum_{m=0}^{L} c_m P_m(x)
```

with $`G_{\rm inst}`$ the instrument LSF and $`G_{\rm rot}(v\sin i)`$ the rotation
kernel (both Fourier multiplies).

**Tellurics.** Earth's atmosphere imprints absorption at *fixed observed
wavelengths*, independent of the star's motion. So the telluric templates
$`S_j = [\mathrm{tell} \mu, \mathrm{tell} \phi_1,\dots]`$ live in the observed
(topocentric) frame and are not RV-shifted — they enter as shift-independent
columns alongside the Legendre continuum $`P_m`$, broadened by the instrument LSF
only. They contribute telluric weights $`\mathbf{u}`$ to
$`\boldsymbol\theta = (\mathbf{w}, \mathbf{u}, \mathbf{c})`$. The fixed block is
generalised from $`\mathbf{P}`$ to $`\mathbf{F} = [ G_{\rm inst}\mathbf{S} \mid \mathbf{P} ]`$,
so $`\mathbf{M}_{FF}, \mathbf{b}_F`$ change only when the telluric broadening
changes; when no telluric basis is supplied, $`\mathbf{F} = \mathbf{P}`$ and every
code path reduces to the stellar-only case.

**Telluric shift / wavelength zero-point.** Spectrographs like DEIMOS carry a
slit-dependent wavelength zero-point error (uncorrected flexure): an overall shift of
the observed wavelength scale. Since the tellurics are at rest in the observer frame,
their apparent shift measures that zero-point directly. With `--fit-telluric-shift`
(`fit_telluric_shift=True`) the telluric block gets its own velocity
$`\Delta x_{\rm tell}`$ — a Fourier phase ramp on its FFTs, just like the stellar RV —
refined jointly with the other nonlinear parameters. The stellar shift absorbs both
the true RV and the zero-point ($`p_{\rm star} = p_{\rm true} + p_0`$) while the
tellurics measure the zero-point alone ($`p_{\rm tell} = p_0`$), so the corrected RV
is the lag difference

```math
v_{\rm corr} = c \left( e^{(p_{\rm star} - p_{\rm tell})\delta} - 1 \right),
```

reported as `v_corr_kms` alongside the telluric velocity `v_tell_kms`. Adding a
barycentric correction then places it in the heliocentric frame — reproducing the
telluric/sky-line wavelength recalibration used by DEIMOS pipelines (e.g. dmost).

**Rotation** ($`v\sin i`$). Stellar templates may additionally be convolved by a
Gray rotational profile of projected rotation velocity $`v\sin i`$ and linear
limb-darkening coefficient $`\epsilon`$ (default $`0.6`$),

```math
G(\Delta v)  \propto  2 (1-\epsilon)\sqrt{1-x^2}  +  \tfrac{\pi\epsilon}{2} (1-x^2) \qquad x = \Delta v / (v\sin i),\ |x| \le 1.
```

On the log grid this is a fixed-pixel kernel, rfft'd once into a real transfer
function that multiplies the stellar template FFTs — applied to the stellar block
only, never to tellurics or continuum. It can be fixed or fit (`fit_vsini=True`);
reported as `vsini_kms` ± err, with an unresolved lower-limit flag when the
optimum lands at the search floor.

**Native resolution.** Each basis records the native resolving power $`R`$ it was
built from. The fitter broadens differentially in quadrature, so a basis of
native dispersion $`\sigma_{\rm native}`$ is taken to the observed instrument
dispersion $`\sigma_{\rm obs}`$ via

```math
\sigma_{\rm diff}^2 = \max\big(0,  \sigma_{\rm obs}^2 - \sigma_{\rm native}^2\big).
```

Stellar and telluric grids may be built at different native resolutions; each
carries its own, so the differential Gaussian applied to each block differs even
though the physical instrument resolution is a single shared parameter.
$`R_{\rm native} = \infty`$ (the default) gives $`\sigma_{\rm native} = 0`$ and
reproduces the full single-Gaussian broadening.

**Refinement / optimisation**: a coarse FFT scan over $`\Delta x`$ (and a coarse $`R`$
grid) locates the global basin (the profiled $`\chi^2`$ is multimodal in the
nonlinear parameters, though convex in $`(\mathbf{w}, \mathbf{c})`$ at fixed
$`(\Delta x, R)`$); then a JAX-autodiff modified-Newton with a backtracking Armijo
line search polishes $`(\Delta x[, R])`$ on the *profiled* $`\chi^2`$. Rotation
$`v\sin i`$ and the telluric shift $`\Delta x_{\rm tell}`$ are refined the same way;
the telluric zero-point is small and unimodal, so (unlike $`\Delta x`$) it is *not*
coarse-scanned — just initialised at 0 and Newton-refined. The parameter covariance
is the autodiff Hessian of the profiled $`\chi^2`$ at the optimum
($`\Delta\chi^2 = 1`$), propagated to $`(v, R, v\sin i, v_{\rm tell})`$. Typical warm
cost ≈ 2 s/fit (CPU; the bottleneck is the 216k-point FFT, ~10× faster on GPU).

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

## Usage

### CLI

```bash
# build a STELLAR template basis (rest frame, RV-shifted by the fitter)
clammy build --templates grid/convolved --basis pca --var 0.99 --k-max 30 -o outputs/basis.npz

# record the grid's native resolving power so the fitter can broaden differentially
clammy build --templates grid/convolved --resolution 100000 -o outputs/basis.npz

# build a TELLURIC basis from a telluric MODEL grid -- stored in the OBSERVED
# (topocentric) frame, so the fitter NEVER RV-shifts it
clammy build --kind telluric --templates tell_models --pattern "*.fits" \
             --resolution 7000 -o outputs/tell.npz

# fit a spectrum (.npz or FITS table; wavelength linear-Å or log, auto-detected)
clammy fit outputs/basis.npz spectrum.npz                       # RV + weights + continuum
clammy fit outputs/basis.npz spectrum.fits --fit-resolution \
           --vmin -300 --vmax 300 --plot fit.png -o fit.json    # also fit R

# add an observed-frame telluric block (NOT RV-shifted); fit vsini too
clammy fit outputs/basis.npz spectrum.fits --telluric-basis outputs/tell.npz --fit-vsini
```

`fit` reads `wave`/`flux`/`sigma` columns (override with `--wave-col` etc.; use
`--snr` if there is no uncertainty column), resamples onto the basis grid if
needed, and prints `v[, R][, vsini]`, χ²/dof, stellar weights, telluric weights
(when a `--telluric-basis` is given), and continuum coefficients.

**The four broadening modes** (instrument resolution `R` and rotation `vsini`):

```bash
clammy fit basis.npz spec.fits --fit-resolution --fit-vsini  # 1. fit R + vsini together
clammy fit basis.npz spec.fits --resolution-R 7000           # 2. fixed R, no vsini
clammy fit basis.npz spec.fits                               # 3. nothing (already at R)
clammy fit basis.npz spec.fits --fit-vsini                   # 4. vsini only (R assumed right)
```

1. **fit R + vsini together** — the instrument LSF is applied to both the stellar
   and telluric blocks; the rotation kernel to the stellar block only.
2. **fixed R, no vsini** — a known instrument LSF is applied (once) to both
   blocks; no rotation.
3. **nothing** — assume the templates are already at the right resolution.
4. **vsini only** — assume the instrument resolution is right; fit the extra
   stellar rotational broadening. Use `--vsini FLOAT` for a *fixed* vsini instead,
   `--vsini-bounds VMIN VMAX` for the search range, and `--epsilon` for the
   limb-darkening coefficient.

Because each basis stores its native resolving power, the differential broadening
$`\sigma_{\rm diff}^2 = \max(0, \sigma_{\rm obs}^2 - \sigma_{\rm native}^2)`$ is
correct even when the stellar and telluric grids were built at different
resolutions.

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
from clammy import basis, fitter, grid, toy, paths

b = basis.load_basis(f"{paths.OUTPUTS}/basis.npz")

# a rectified-log-flux "truth" that lies in the basis (mean + a couple of PCs)
r_true = b["mu"] + 0.5 * b["Phi"][0] - 0.3 * b["Phi"][1]

# RV + weights + continuum
d, sigma, truth = toy.make_toy(
    b["loglam"],
    r_true,
    v_kms=42.0,
    cont_coef=[0.6, -0.3, 0.1],
    snr=25,
)
res = fitter.fit_rv(
    d,
    sigma,
    b["loglam"],
    b["mu"],
    b["Phi"],
    cont_order=3,
    vmin=-400,
    vmax=400,
)
res["v_kms"], res["v_err_kms"], res["w"], res["c"], res["cov"], res["chi2_dof"]

# also fit the spectral resolution R (degraded + on any grid)
d, sigma, truth = toy.make_toy(
    b["loglam"],
    r_true,
    v_kms=42.0,
    cont_coef=[0.6, -0.3, 0.1],
    snr=50,
    resolution_R=6000.0,
)
res = fitter.fit_rv(
    d,
    sigma,
    b["loglam"],
    b["mu"],
    b["Phi"],
    cont_order=3,
    fit_resolution=True,
)
res["v_kms"], res["resolution_R"], res["R_err"], res["cov_vR"]
```

**The four broadening modes** map to these `fit_rv` kwargs (mirroring the CLI):

```python
# 1. fit instrument R + vsini together
fitter.fit_rv(d, sigma, b["loglam"], b["mu"], b["Phi"], fit_resolution=True, fit_vsini=True)

# 2. fixed instrument R, no vsini
fitter.fit_rv(d, sigma, b["loglam"], b["mu"], b["Phi"], resolution_R=7000.0)

# 3. nothing (templates already at the right resolution)
fitter.fit_rv(d, sigma, b["loglam"], b["mu"], b["Phi"])

# 4. vsini only (instrument resolution assumed correct)
fitter.fit_rv(d, sigma, b["loglam"], b["mu"], b["Phi"], fit_vsini=True)
res["vsini_kms"], res["vsini_err_kms"], res["vsini_limited"]  # lower-limit flag
```

**Tellurics + native resolution.** Build an observed-frame telluric basis the same
way as the stellar one (the math is identical; only the `frame`/`resolution`
metadata and the absence of stellar labels differ), then pass it to `fit_rv` as a
shift-independent block:

```python
import numpy as np

# build an OBSERVED-frame telluric basis from a telluric model grid
loglam_t, flux_t, files = grid.load_spectra("tell_models", pattern="*.fits")
tell_mu, tell_Phi, info = basis.build_telluric_basis(loglam_t, flux_t, var_target=0.99)
basis.save_basis(
    f"{paths.OUTPUTS}/tell.npz",
    loglam_t,
    tell_mu,
    tell_Phi,
    params=None,  # telluric: no stellar labels
    info=info,
    rectify_order=5,
    frame="observed",  # NEVER RV-shifted by the fitter
    resolution=7000.0,  # native resolving power of the telluric grid
)

# load both bases and fit -- the telluric basis must be on the SAME grid as the
# data/stellar basis (resample with np.interp first if it is not)
tb = basis.load_basis(f"{paths.OUTPUTS}/tell.npz")
res = fitter.fit_rv(
    d,
    sigma,
    b["loglam"],
    b["mu"],
    b["Phi"],
    resolution_R=7000.0,
    tell_basis=(tb["mu"], tb["Phi"]),  # NOT RV-shifted; instrument-broadened only
    R_native_star=b["resolution"],  # differential broadening from native R
    R_native_tell=tb["resolution"],
)
res["u"]  # telluric template weights
```

The fitter broadens each block differentially in quadrature,
$`\sigma_{\rm diff}^2 = \max(0, \sigma_{\rm obs}^2 - \sigma_{\rm native}^2)`$, so
stellar and telluric grids built at different native resolutions are handled
correctly; the default `np.inf` means "effectively unresolved" and reproduces full
broadening.


## Validation summary

Leave-one-out (basis never saw the test star; metal-poor RGB, Teff 4500 / log g 1 / [Fe/H] −2):

- **RV** — bias ≤ 10 m/s at SNR 5, errors well-calibrated (realised scatter ≈ predicted), χ²/dof ≈ 1 at low SNR; at high SNR χ²/dof rises (the expected PCA-truncation floor, since K=14 components cannot perfectly represent a held-out star) while RV bias stays ~1–2 m/s. RV-vs-velocity sweep (−300…+300 km/s): max\|bias\| = 3 m/s. *(In-basis truth gives RV bias < 0.4 m/s and χ²/dof = 1 at all SNR.)*
- **Continuum / basis** — continuum order ≥ 3 suffices (χ²/dof plateaus); larger K lowers the truncation floor (χ²/dof: 3.0 at K=1 → 1.6 at K=14 → 1.1 at K=30).
- **Resolution** — `v` and `R` are nearly orthogonal (ρ_vR ≈ 0) and errors are well-calibrated (realised scatter ≈ predicted for both). Two bias regimes: with an *in-basis* truth, `R` is biased only −2% (the log-space-vs-linear-flux convolution approximation; a log-broadened toy is recovered to 0.1%). With a *held-out* star (realistic template mismatch), `R` is biased +11…16% and freeing `R` also leaks mismatch into `v` (tens–hundreds of m/s, dominating at high SNR). Takeaway: R from this fit is degenerate with template completeness — trust it to ~15% unless the basis represents the target well. A different/coarser observed grid is handled by resampling onto the template grid (point estimates recovered; resampling correlates noise, so χ²/dof and error bars need the proper covariance).
- **Tellurics** — on toy data with telluric absorption injected in the observed frame, adding the observed-frame telluric block sharply reduces the RV bias that an unmodelled telluric otherwise induces (the unmodelled case shows a strongly inflated χ²/dof that the telluric columns absorb back to ≈ 1), since the telluric features are held fixed in the observed frame and so do not pull on the RV. The telluric weights `u` are returned alongside the stellar weights.
- **vsini** — the Gray rotational profile (limb-darkening `epsilon`, default 0.6) is recovered when the coarse search brackets the true `vsini` (demonstrated on toy data); when the rotation is below the velocity sampling the optimum lands at the search floor and `vsini` is reported as a `vsini_limited` lower limit. As with `R`, freeing `vsini` against a held-out star trades against template mismatch, so treat a fitted `vsini` as informative only when the basis represents the target well.

## References

- Tonry & Davis (1979) — cross-correlation RVs.
- Cappellari (2017) — pPXF (additive + multiplicative polynomials; canonical implementation).
- Bolton et al. (2012) — BOSS PCA-template + redshift-scan classification.
- Geha et al. (2026) - DIEMOS analysis
