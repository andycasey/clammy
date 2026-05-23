"""Canonical project paths, derived from this file's location.

Layout (src-layout package; data under grid/):

    marla-geha/                  ROOT
    |-- pyproject.toml
    |-- src/clammy/              the package
    |-- scripts/                 driver scripts
    |-- outputs/                 basis.npz + figures
    |-- docs/                    formulation.tex/pdf
    `-- grid/
        |-- original/            raw PHOENIX HiRes: lte*.fits, WAVE_*.fits
        `-- convolved/           smoothed + log-rebinned: dmost_lte_*.fits

Everything is computed relative to the package, so it works from any cwd and
regardless of where clammy is installed in editable mode. Override any of these
with the matching environment variable if needed.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))   # .../src/clammy
SRC = os.path.dirname(_HERE)                          # .../src
ROOT = os.path.dirname(SRC)                           # .../marla-geha
_PHX = os.path.join(ROOT, "grid")

ORIGINAL = os.environ.get("ORIGINAL_DIR", os.path.join(_PHX, "original"))
CONVOLVED = os.environ.get("CONVOLVED_DIR", os.path.join(_PHX, "convolved"))
OUTPUTS = os.environ.get("OUTPUTS_DIR", os.path.join(ROOT, "outputs"))

__all__ = ["ROOT", "SRC", "ORIGINAL", "CONVOLVED", "OUTPUTS"]
