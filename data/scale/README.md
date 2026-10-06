# Controlled factorial PERSON fixture

These files hold **920,000 synthetic directory-style records**, one for every combination of four fields:

| Field | Values | Meaning |
| --- | --- | --- |
| `age_band` | 10 ordered bands, 18–24 to 65+ | Ordinal |
| `job_category` | 8 broad categories | Nominal |
| `home_region` | 5 regions | Nominal |
| `interests` | exactly 3 distinct values from 25 | Unordered set |

10 × 8 × 5 × C(25, 3) = 920,000. Together these rows **exhaust the declared state space**. Every signature appears exactly once. They are not a sample of people, and there are no names, IDs or free text to encode. A complete record contributes six bound facts: age, job, region, and three interests. `domains.json` maps integer codes to readable labels.

## Files

| File | Used by | Contents |
| --- | --- | --- |
| `signatures.parquet` | all experiments | `record_index` plus the six integer codes, in the balanced order below |
| `query_panels.parquet` | both experiments | The 200-query calibration panel and 400-query evaluation panel, as record indices |

The missingness mask and pairwise pairs that were first generated here now live in [`data/missingness`](../missingness/README.md), built from these signatures.

## Balanced nested prefixes

Experiment 1 compares candidate counts N = 100, 1,000, 10,000, 100,000 and 920,000 by taking the first N rows. Rows cycle through the 400 (age, job, region) cells so that every prefix has near-equal counts of each age, job and region, within one of each other. For example, the first 100 rows contain every age band 10 times and every region 20 times. `generate.py` checks the balance for every prefix used.

## Query panels

Both panels balance jobs and regions and cover all ten age bands. **Endpoint ages (bands 1 and 10) make up 40% of each panel.** Only an endpoint-age query has candidates with zero exact similarity: they have the opposite endpoint age, a different job and region, and no shared interests. The calibration panel sets thresholds and the evaluation panel measures them. The two panels share no records.

## Regenerate

From the repository root:

```sh
uv run --project src/scale --locked python data/scale/generate.py
```

Generation is deterministic, with seed string `hyp83-person-v2` and torch generators. The run records the SHA-256 of every Parquet file and refuses to reuse frozen thresholds if any file changes.
