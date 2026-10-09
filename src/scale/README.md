# When does chance look like a match?

This study looks **outside** one bundle. Each record is the raw sum of five bound categorical facts, so every fact can still be unbound. We compare every record with every other and ask how often an unrelated pair scores above a match threshold by chance, for **one search** and for **the whole dataset**, at D = 2,048, 4,096 and 8,192. The companion [`qa-encoding`](../qa-encoding/README.md) study looks **inside** one bundle.

- **[Report](REPORT.md):** results, figures and tables.
- **[Methodology](METHODOLOGY.md):** the question, the four-step theory and the experiment.
- **[Data](../../data/scale/README.md):** the 40,000 synthetic records.

## What this means for HDC practitioners

These hold for five uniformly drawn categorical properties, raw-sum bundling and cosine similarity; the [report](REPORT.md) has the measurements.

1. **A shared fact is worth 1/5 of the score.** A pair sharing q of 5 properties averages q/5, at any dimension.
2. **Ask which job you are doing.** One search makes N comparisons; checking a whole dataset makes about N²/2. The chance per comparison is the same, so the dataset fills up first: the birthday paradox.
3. **At 2,048 dimensions, one-fact matches are fine for search but not for deduplication.** With 40,000 records, a single search returns a false match about 4 times in 100,000, while the dataset as a whole holds at least one false pair in about half the trials. Both match the binomial prediction.
4. **4,096 or 8,192 dimensions, or two-fact matches, make chance collisions astronomically rare.** No false match occurred at 4,096 or 8,192, or for two-fact matches at any dimension, as predicted.
5. **Size on paper first.** Expected false matches = comparisons × p(T), with p(T) from the binomial model, predicted every measured setting.

## Reproduce

From the repository root:

```sh
sh src/scale/reproduce.sh
```

This runs Ruff, the focused tests, then `run.py`, which regenerates the records, runs every trial, and writes the summaries, figures and report. It takes about six minutes on a recent laptop CPU.

| File | Role |
| --- | --- |
| `run.py` | Records, trials, summaries, plotnine figures, manifest, then the report |
| `study.py` | Encoder (TorchHD role and value hypervectors, int8 facts, raw sum), theory (thresholds and binomial tail) and the all-pairs check |
| `report.py` | Writes `REPORT.md` from the summary |
| `tests/test_study.py` | Encoder and unbinding, shared-property counts, theory, and the all-pairs check against brute force |

Outputs go to `results/scale/`; the [report](REPORT.md#files) lists them.
