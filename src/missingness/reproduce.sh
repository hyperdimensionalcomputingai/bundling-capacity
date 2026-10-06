#!/bin/sh
# Reproduce Experiments 1-5 and the earlier pairwise experiment.
# Run from the repository root, after data/scale/ exists (src/scale/reproduce.sh builds it).
set -eu
project="--project src/missingness --locked"
uv run $project python data/missingness/generate.py    # original inputs, MCAR masks, duplicates
uv run $project pytest -q src/missingness/tests
uv run $project python src/missingness/experiment1_weight.py   # Experiment 1
uv run $project python src/missingness/experiment2_dial.py     # Experiment 2
uv run $project python src/missingness/experiment3_linkage.py  # Experiment 3
uv run $project python src/missingness/experiment4_build.py    # Experiment 4
uv run $project python src/missingness/experiment5_index.py    # Experiment 5 (IVF_PQ indexes, ~11 GB store)
uv run $project python src/missingness/pairwise.py             # earlier pairwise experiment
uv run $project python src/missingness/charts.py
