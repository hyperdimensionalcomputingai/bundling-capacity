# Missingness inputs

Inputs for [`src/missingness`](../../src/missingness/README.md). Every file indexes records of the scale study's [factorial PERSON fixture](../scale/README.md) by `record_index`; `generate.py` reads `data/scale/signatures.parquet` and `query_panels.parquet` and rebuilds them deterministically, each random stream named by a seed string and its purpose.

| File | Contents |
| --- | --- |
| `mcar.parquet` | Experiments 2, 4 and 5: one missing-completely-at-random mask per rate (0.1, 0.3, 0.5). Age, job and region are each missing with probability r, and each interest independently with probability r |
| `duplicates.parquet` | Experiment 3: a noisy duplicate of each query-panel record and of 20,000 training records, at every rate. Each value is replaced by a different one with probability 0.1 (the same corruption at every rate), then the duplicate gets its own missingness. Missing values are -1 |
| `pairwise_pairs.parquet` | The earlier pairwise experiment: 2,000 same-person and 2,000 unrelated pairs. Each pair fixes the order for removing scalar fields and each record's null spelling for the mixed-spelling strategy |
| `missingness.parquet` | The earlier retrieval check: one fixed mask: each field of each record independently missing with probability 0.3 |
| `settings.json` | Seed string, null spellings, rates, error rate and pair and duplicate counts |

These two earlier files are reproduced byte for byte from the original combined generator.

- **Same-person pairs** are one random record paired with itself.
- **Unrelated pairs** are a band-1 record and a band-10 record with a different job, a different region and no shared interests. Their content similarity is 0, and stays 0 however many scalar fields are removed.

Regenerate from the repository root:

```sh
uv run --project src/missingness --locked python data/missingness/generate.py
```
