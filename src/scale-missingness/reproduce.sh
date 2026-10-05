#!/bin/sh
# Reproduce all four experiments. Run from the repository root.
set -eu
project="--project src/scale-missingness --locked"
uv run $project python data/scale-missingness/generate.py   # deterministic fixture
uv run $project pytest -q src/scale-missingness/tests
uv run $project python src/scale-missingness/run.py          # store, then Experiments 1 and 3
uv run $project python src/scale-missingness/pairwise.py     # Experiment 2
uv run $project python src/scale-missingness/ivfpq.py        # Experiment 4
uv run $project python src/scale-missingness/summarize.py
uv run $project python src/scale-missingness/charts.py
