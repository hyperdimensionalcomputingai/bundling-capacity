# Synthetic categorical records

`records.parquet` holds **1,000,000 synthetic records**, each with five categorical properties drawn uniformly and independently:

| Column | Values | Type |
| --- | --- | --- |
| `record_id` | 0 to 999,999, in sequence order | int32 |
| `region` | 20 | int16 code |
| `education` | 10 | int16 code |
| `occupation` | 100 | int16 code |
| `interest_cluster` | 200 | int16 code |
| `employer` | 1,000 | int16 code |

The property names make the example readable. Codes carry no meaning, and the study gives every value an independent random vector, so different values have no semantic similarity. This is a controlled synthetic population, not a model of real demographic frequencies or of relationships between properties.

## How the study uses it

- **Nested prefixes.** N = 10,000, 100,000 and 1,000,000 are the first N rows.
- **Query panel.** The first 100 records are the queries.
- **Duplicates.** There are four billion possible combinations, so independent sampling produces some duplicate attribute records: 132 in this sequence. They are kept. A query's own `record_id` is the only record excluded from its comparisons.
- **Overlap shares.** Under these cardinalities a random pair shares no property with probability 0.841, and exactly one with probability 0.151. `settings.json` lists the share for every k.

## Regenerate

From the repository root:

```sh
uv run --project src/scale --locked python data/scale/generate.py
```

`generate.py` draws each property in turn with `torch.randint` from one generator seeded with the data seed, 101. It writes `records.parquet` and `settings.json`, which records the seed, cardinalities, torch version, pair-overlap shares and duplicate count. The study's `run.py` calls it first, so a full run always starts from these records.
