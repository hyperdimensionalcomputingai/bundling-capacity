#!/bin/sh
# Reproduce the scale study. Run from the repository root.
set -eu
project="--project src/scale --locked"
uv run $project ruff check src/scale data/scale
uv run $project pytest -q src/scale/tests
uv run $project python src/scale/run.py   # records, encoder check, scan, summaries, figures, report
