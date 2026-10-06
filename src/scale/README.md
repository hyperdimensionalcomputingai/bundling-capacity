# MAP similarity among 920,000 records

This directory holds the scale study. The companion [`qa-encoding`](../qa-encoding/README.md) study looks **inside** one bundle: how many answers can be packed in and read back. This study looks **outside** it: how a record's bundle behaves among a million others. Missing values have their own study in [`src/missingness`](../missingness/README.md).

- **[Report](REPORT.md):** the two experiments, what we learned, and the takeaways.
- **[Methodology](METHODOLOGY.md):** fixture, encoder, baseline, threshold contract, storage, and a note on how far an earlier capacity estimate applies.
- **[Fixture](../../data/scale/README.md):** the 920,000 factorial PERSON signatures and query panels.

Binding uses $\otimes$: a fact is $h_{\mathrm{fact}} = h_{\mathrm{role}} \otimes h_{\mathrm{value}}$. Bundling uses $\oplus$: a record combines its facts as $h_{\mathrm{record}} = \bigoplus_{f \in \mathcal{F}} h_{\mathrm{fact},f}$, where $\mathcal{F}$ is the set of six encoded facts. Here binding is element-wise multiplication and bundling is an arithmetic sum without a sign threshold. The stored bundles retain that sum; cosine comparison divides by their norms when scoring. The [methodology](METHODOLOGY.md#encoder) gives the coordinate equations.

## The two experiments

| # | Question | Script | Main output |
| --- | --- | --- | --- |
| 1 | How high do unrelated records score by chance among 920,000 candidates, and how does D control it? | `run.py` | `experiment1_scale.csv`, `experiment1_scale.png` |
| 2 | How much does a LanceDB IVF_PQ index lose against exact search? | `ivfpq.py` | `ivfpq_summary.csv`, `experiment2_ivfpq.png` |

## Reproduce

From the repository root:

```sh
sh src/scale/reproduce.sh
```

The script regenerates the fixture, runs the tests, then runs `run.py`, `ivfpq.py`, `summarize.py` and `charts.py`.

`run.py` has three stages, which can also be run separately with `--stage build|calibrate|evaluate`:
- **build** writes the LanceDB store. It is skipped when `store.json` matches the fixture hashes and grid.
- **calibrate** freezes the thresholds.
- **evaluate** is one streamed pass covering Experiment 1. It refuses to run if `thresholds.json` doesn't match the current fixture and settings.

**Storage.**
- LanceDB holds every atom as float16, a few MB, plus one materialized table of 920,000 record bundles at D = 2,048 for Experiment 2, about 3.8 GB.
- Every other bundle is encoded on the fly, in float32, from the stored atoms.
- The bundles are small integers, so float16 storage is exact.
- The store is gitignored. Delete `results/scale/experiment.lancedb` to reclaim the space.

| Module | Role |
| --- | --- |
| `person.py` | Records, the MAP-I encoder, exact integer-dot cosine, and the S_exact baseline |
| `store.py` | LanceDB schemas and I/O: float16 at the write boundary, float32 on read |
| `experiment.py` | The streamed exact search behind Experiment 1 |
| `run.py` | Build, calibration and evaluation stages, plus the manifest |
| `ivfpq.py` | Experiment 2 |
| `summarize.py` | Tables for Experiment 1, with bootstrap and Clopper–Pearson uncertainty |
| `charts.py` | Figures as PNG and editable SVG |
| `tests/` | Ordinal kernel, orthogonal-atom exactness (S_MAP = S_exact), float16 round trip, and a streamed pass against brute force on a small LanceDB store |

## Saved evidence

Everything below is in `results/scale/`.

| File | Contents |
| --- | --- |
| `thresholds.json` | Frozen T per (D, seed), with calibration recall, the rule, fixture hashes and settings |
| `per_query.parquet` | One row per (D, seed, N, query): tier pair counts, highest unrelated score, threshold events, top-10 agreement against S_exact, head ordering |
| `cell_summary.csv` | Per (D, seed, N): MAP-error quantiles, score tails and threshold rates |
| `experiment1_scale.csv`, `experiment1_uncertainty.csv` | Experiment 1, averaged over seeds with seed ranges, and query-level bootstrap intervals and Clopper–Pearson bounds |
| `ivfpq_summary.csv`, `ivfpq_per_query.parquet`, `ivfpq_index.json` | Experiment 2 |
| `encoder_diagnostics.csv` | Realized age-level cosines and independent-atom overlap for every (D, seed) |
| `manifest.json` | Versions, seed formulas, storage design, fixture hashes, settings and stage timings |

IVF_PQ training has no fixed seed in this script, so fresh index builds can produce slightly different results. The saved index measurements describe the recorded run.

These are reproducible and gitignored:
- `experiment.lancedb/`
- `histograms.parquet`: nonzero 1e-4 bins
- `exact_head.parquet`: each query's exact top 100

## Scope

The fixture is a controlled factorial state space, not a sample of people. Results hold for this encoder, fixture, grid and query panel. No billion-record extrapolation is made here; that is left for future work.
