#!/bin/sh
# Reproduce both experiments. Run from the repository root.
set -eu
project="--project src/scale --locked"
uv run $project python data/scale/generate.py   # deterministic fixture
uv run $project pytest -q src/scale/tests
uv run $project python src/scale/run.py         # store, then Experiment 1
uv run $project python src/scale/ivfpq.py       # Experiment 2
uv run $project python src/scale/summarize.py
uv run $project python src/scale/charts.py
