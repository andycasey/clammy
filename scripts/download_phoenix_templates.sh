#!/usr/bin/env bash
#
# download_phoenix_templates.sh
#
# Download the PHOENIX-ACES-AGSS-COND-2011 HiRes synthetic spectra used by dmost,
# in parallel, into grid/original/ (override with ORIGINAL_DIR), then
# smooth/trim/log-rebin them into grid/convolved/.
#
# This mirrors the wget loop in prepare_pheonix_templates.ipynb. The active grid
# there is GRID #1 (SN > 25); GRIDS #2 and #3 are included below, commented out.
#
# Usage:
#   ./download_phoenix_templates.sh          # 8 parallel downloads
#   NJOBS=16 ./download_phoenix_templates.sh # override parallelism
#
set -uo pipefail

ROOT="ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011"

# Number of simultaneous downloads
NJOBS="${NJOBS:-8}"

# ---- GRID #1:  SN > 25  (active grid in the notebook) ----
ZZ=(-0.0 -0.5 -1.0 -2.0 -3.0 -4.0)
LOGG=(1.00 3.00 5.00)
TEFF=(2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 8000)

# ---- GRID #2:  10 < SN < 25 ----
# ZZ=(-0.0 -1.0 -2.0 -3.0 -4.0)
# LOGG=(1.00 3.00 5.00)
# TEFF=(3000 3500 4000 5000 6000 7000 8000)

# ---- GRID #3:  SN < 10 ----
# ZZ=(-0.0 -1.0 -3.0)
# LOGG=(2.00 5.00)
# TEFF=(3000 3500 4000 5000 6000 7000)

# Build the list of URLs, matching the notebook's filename construction:
#   'lte0{t}-{lg}{z}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'  in dir  Z{z}/
urls=()
for z in "${ZZ[@]}"; do
  for lg in "${LOGG[@]}"; do
    for t in "${TEFF[@]}"; do
      f="lte0${t}-${lg}${z}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
      urls+=("${ROOT}/Z${z}/${f}")
    done
  done
done

# The shared wavelength grid (needed downstream by the notebook's smoothing step).
urls+=("ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")

HERE="$(cd "$(dirname "$0")" && pwd)"
ORIGINAL_DIR="${ORIGINAL_DIR:-$HERE/../grid/original}"
mkdir -p "$ORIGINAL_DIR"
cd "$ORIGINAL_DIR"

echo "Downloading ${#urls[@]} files with up to ${NJOBS} parallel jobs into $(pwd)"

# Download in parallel. curl -C - resumes partial files and skips already-complete
# ones, so the script is safe to re-run.
printf '%s\n' "${urls[@]}" | xargs -P "${NJOBS}" -n 1 -I {} bash -c '
  url="$1"
  f="${url##*/}"
  if curl -fsS --retry 5 --retry-delay 2 -C - -o "$f" "$url"; then
    echo "ok    $f"
  else
    echo "FAIL  $f"
  fi
' _ {}

echo "Downloads finished. Any lines above starting with FAIL did not download; re-run to retry."

# ---- Smooth + trim into grid/convolved/dmost_lte_*.fits ----
# Set PROCESS=0 to download only and skip this step.
if [ "${PROCESS:-1}" != "0" ]; then
  PY="$HERE/../.venv/bin/python"
  [ -x "$PY" ] || PY="$(command -v python3 || command -v python || true)"
  if [ -n "$PY" ]; then
    echo "Smoothing + trimming templates (-> grid/convolved/) ..."
    "$PY" "$HERE/process_phoenix_templates.py"
  else
    echo "python not found; skipping smooth/trim. Run process_phoenix_templates.py manually."
  fi
fi
