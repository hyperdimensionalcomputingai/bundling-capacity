# Ordered MAP bundle experiment

This is the implementation of [QA_ENCODING_EXPERIMENT_PLAN.md](../../QA_ENCODING_EXPERIMENT_PLAN.md). It measures scored-answer recovery, whole-profile geometry, and whether similar profiles remain close while adding 5 to 50 facts to a single MAP bundle. The 50 item keys and one ordered five-level answer scale are reused across every profile at a given dimension and seed.

## Reproduce

From the repository root:

```sh
uv run --project src/qa-encoding --locked python src/qa-encoding/run.py
uv run --project src/qa-encoding --locked python src/qa-encoding/perturbation.py
uv run --project src/qa-encoding --locked python src/qa-encoding/charts.py
```

The run uses dimensions 512, 2,048, 4,096, and 8,192; seeds 11, 23, 37, 53, and 71; and nested widths 5, 10, 20, 30, 40, and 50. It reads the fixed source CSVs with Polars, writes explicit PyArrow schemas to LanceDB, stores hypervectors as fixed-size float16 lists, and casts them to float32 for TorchHD computation. Item and profile metadata remain ordinary columns. The sum used for unbinding is never sign-thresholded.

The run checks the Q19/Q49 scoring example, the five-level geometry, storage schemas, and a filtered exact flat LanceDB cleanup search at the first, middle, and last item for every dimension/seed/width. The complete sweep compares each unbound vector with the same five **stored** answer vectors in TorchHD. This gives exact five-way cosine cleanup without repeating LanceDB query planning for all 96,100 item readouts. LanceDB does not have an ANN index in this study. The query comparison checks all five answer positions, not just the winner.

## Saved evidence

| File or table | Contents |
| --- | --- |
| `experiment.lancedb/responses` | Respondent ID, item ID, raw and scored ground truth. |
| `experiment.lancedb/symbols_d*` | Independent item keys and five shared ordered levels for each seed. |
| `experiment.lancedb/profiles_d*` | Bundles at every width for each profile and seed. |
| `item_readouts.parquet` | Correct/predicted scored and raw answers, ordinal error, and winner margin for every query. |
| `profile_pairs.parquet` | Bundle cosine, matched-item response difference, and ideal level similarity for every profile pair. |
| `recovery_by_seed.csv`, `geometry_by_seed.csv` | Figure data and seed variation. |
| `encoder_diagnostics.csv` | Off-diagonal item-key cosine and all level-pair cosines. |
| `manifest.json` | Seeds, source hashes, versions, dtypes, and encoder settings. |
| `perturbation_pairs.parquet` | Every controlled raw-answer change and its whole-bundle and ideal answer-level cosine. |
| `perturbation_summary.csv`, `perturbation_design.json` | Whole-bundle and ideal answer-level cosine by width, plus the study settings. |
| `perturbation_distinct_profiles.csv` | Context from pairs of different synthetic profiles, grouped by raw answer difference. |

The analysis uses 465 unique profile pairs per condition. Spearman correlation compares the ordering of all pairs, with average ranks for ties. Nearest-neighbor agreement and three-neighbor overlap are computed for each of the 31 profiles, with respondent order breaking exact ties. The two references compare responses at **matching item positions**. Reversing both profiles' responses on the same item leaves absolute difference unchanged.

The perturbation study fixes the dimension at 4,096 and uses widths 5, 10, 20, 40, and 50. It changes exactly 20% of raw answers by one level for similar pairs and 60% for a farther control, then re-scores and compares whole bundles. Variants are shared across five MAP seeds, and each profile contributes equal weight within a condition and width. The question is whether the similarity of the 20%-changed pairs falls as more facts enter the bundle.

Figures are available as PNG and editable SVG: [answer recovery](../../results/qa-encoding/recovery.png), [profile geometry](../../results/qa-encoding/profile_geometry.png), and [similarity as bundles grow](../../results/qa-encoding/perturbation_similarity.png). The [report](../../REPORT.md) interprets them for a research audience.

The fixture is synthetic and generated from five factor anchors. The observed ceiling at high dimension and strong profile ranking on this structured fixture should be read as results of this design, not a general MAP capacity threshold.
