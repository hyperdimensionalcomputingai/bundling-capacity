# MAP similarity among 920,000 records, with missing values

This directory implements [HYP-83](https://linear.app/hyperdimensionalcomputing/issue/HYP-83/research-blog-test-map-similarity-under-missingness-and-million-record). The companion [`qa-encoding`](../qa-encoding/README.md) study looks **inside** one bundle: how many answers can be packed in and read back. This study looks **outside** it: how a record's bundle behaves among a million others, and when some of its fields have no value.

- **[Report](REPORT.md):** the four experiments, what we learned, and the takeaways.
- **[Methodology](METHODOLOGY.md):** fixture, encoder, baselines, threshold contract, storage, and a note on how far the earlier chat's model applies.
- **[Fixture](../../data/scale-missingness/README.md):** the 920,000 factorial PERSON signatures, query panels, pairs and mask.

## The four experiments

| # | Question | Script | Main output |
| --- | --- | --- | --- |
| 1 | How high do unrelated records score by chance among 920,000 candidates, and how does D control it? | `run.py` | `experiment1_scale.csv`, `experiment1_scale.png` |
| 2 | What does a missing field do to one comparison, and does `"null"` vs `"no_value"` vs `""` matter? | `pairwise.py` | `pairwise_summary.csv`, `experiment2_pairwise.png` |
| 3 | Does that pairwise effect produce wrong results in a million-record search? | `run.py` | `experiment3_missingness_retrieval.csv` |
| 4 | How much does a LanceDB IVF_PQ index lose against exact search? | `ivfpq.py` | `ivfpq_summary.csv`, `experiment4_ivfpq.png` |

## Reproduce

From the repository root:

```sh
sh src/scale-missingness/reproduce.sh
```

The script regenerates the fixture, runs the tests, then runs `run.py`, `pairwise.py`, `ivfpq.py`, `summarize.py` and `charts.py`.

`run.py` has three stages, which can also be run separately with `--stage build|calibrate|evaluate`:
- **build** writes the LanceDB store. It is skipped when `store.json` matches the fixture hashes and grid.
- **calibrate** freezes the thresholds.
- **evaluate** is one streamed pass covering Experiments 1 and 3. It refuses to run if `thresholds.json` doesn't match the current fixture and settings.

**Storage.**
- LanceDB holds every atom as float16, a few MB, plus one materialized table of 920,000 complete-record bundles at D = 2,048 for Experiment 4, about 3.8 GB.
- Every other bundle is encoded on the fly, in float32, from the stored atoms.
- The bundles are small integers, so float16 storage is exact.
- The store is gitignored. Delete `results/scale-missingness/experiment.lancedb` to reclaim the space.

| Module | Role |
| --- | --- |
| `person.py` | Records, the MAP-I encoder (omit or null-token strategies), exact integer-dot cosine, and the S_content / S_encoder baselines |
| `store.py` | LanceDB schemas and I/O: float16 at the write boundary, float32 on read |
| `experiment.py` | The streamed exact search behind Experiments 1 and 3 |
| `run.py` | Build, calibration and evaluation stages, plus the manifest |
| `pairwise.py` | Experiment 2 |
| `ivfpq.py` | Experiment 4 |
| `summarize.py` | Tables for Experiments 1 and 3, with bootstrap and Clopper–Pearson uncertainty |
| `charts.py` | Figures as PNG and editable SVG |
| `tests/` | Ordinal kernel, orthogonal-atom exactness (S_MAP = S_encoder), strategy equivalences, float16 round trip, and a streamed pass against brute force on a small LanceDB store |

## Saved evidence

Everything below is in `results/scale-missingness/`.

| File | Contents |
| --- | --- |
| `thresholds.json` | Frozen T per (D, seed, condition), with calibration recall, the rule, fixture hashes and settings |
| `per_query.parquet` | One row per (D, seed, condition, N, query): tier pair counts, highest unrelated score, threshold events, top-10 agreement against S_content and S_encoder, head ordering |
| `cell_summary.csv` | Per (D, seed, condition, N): MAP-error quantiles, score tails and threshold rates |
| `experiment1_scale.csv`, `experiment1_uncertainty.csv` | Experiment 1, averaged over seeds with seed ranges, and query-level bootstrap intervals and Clopper–Pearson bounds |
| `pairwise.parquet`, `pairwise_summary.csv` | Experiment 2: every pair's MAP cosine next to its expected value |
| `experiment3_missingness_retrieval.csv`, `semantic_shift.csv` | Experiment 3 |
| `ivfpq_summary.csv`, `ivfpq_per_query.parquet`, `ivfpq_index.json` | Experiment 4 |
| `encoder_diagnostics.csv` | Realized age-level cosines and independent-atom overlap for every (D, seed) |
| `manifest.json` | Versions, seed formulas, storage design, fixture hashes, settings and stage timings |

These are reproducible and gitignored:
- `experiment.lancedb/`
- `histograms.parquet`: nonzero 1e-4 bins
- `exact_head.parquet`: each query's exact top 100

## Scope

The fixture is a controlled factorial state space, not a sample of people. Results hold for this encoder, fixture, grid and query panel. No billion-record extrapolation is made here; that is Phase 2 of HYP-83, after review.
