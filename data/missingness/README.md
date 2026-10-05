# Missingness inputs

Inputs for the parked missing-value work in [`src/missingness`](../../src/missingness/README.md). Both files index records of the scale study's [factorial PERSON fixture](../scale/README.md) by `record_index`; `generate.py` reads `data/scale/signatures.parquet` and reproduces them byte for byte.

| File | Contents |
| --- | --- |
| `pairwise_pairs.parquet` | 2,000 same-person and 2,000 unrelated pairs. Each pair fixes the order for removing scalar fields and each record's null spelling for the mixed-spelling strategy |
| `missingness.parquet` | One fixed mask: each field of each record independently missing with probability 0.3 |
| `settings.json` | Seed string, null spellings, missing rate and pairs per type |

- **Same-person pairs** are one random record paired with itself.
- **Unrelated pairs** are a band-1 record and a band-10 record with a different job, a different region and no shared interests. Their content similarity is 0, and stays 0 however many scalar fields are removed.

Regenerate from the repository root:

```sh
uv run --project src/missingness --locked python data/missingness/generate.py
```
