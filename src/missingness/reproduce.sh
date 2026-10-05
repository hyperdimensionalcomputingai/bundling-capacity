#!/bin/sh
# Reproduce the parked missingness work. Run from the repository root, after data/scale/ exists.
set -eu
project="--project src/missingness --locked"
uv run $project python data/missingness/generate.py   # mask and pairs, from data/scale/signatures.parquet
uv run $project pytest -q src/missingness/tests
uv run $project python src/missingness/pairwise.py
uv run $project python src/missingness/normalization.py
uv run $project python src/missingness/charts.py
