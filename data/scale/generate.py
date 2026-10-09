"""Generate the synthetic categorical records for the scale study.

One fixed sequence of 1,000,000 records, each with five categorical properties drawn
uniformly and independently. The study takes nested prefixes of this sequence
(N = 10,000, 100,000 and 1,000,000) and uses the first 100 records as its query panel,
so the order of rows matters and is fixed by the data seed.

Run from the repository root:

    uv run --project src/scale --locked python data/scale/generate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import torch

HERE = Path(__file__).resolve().parent
RECORDS_PATH = HERE / "records.parquet"
SETTINGS_PATH = HERE / "settings.json"

# Design choices for this study, not estimates of real-world frequencies. Five properties
# keep the majority-sign bundle free of ties; the cardinalities span two to three orders
# of magnitude so that shared values are common for some properties and rare for others.
PROPERTIES = {
    "region": 20,
    "education": 10,
    "occupation": 100,
    "interest_cluster": 200,
    "employer": 1_000,
}
RECORD_COUNT = 1_000_000
DATA_SEED = 101


def generate() -> pl.DataFrame:
    """Draw every property uniformly and independently, one property at a time.

    Drawing per property from a single torch generator keeps the sequence deterministic
    for a given seed and torch version. Duplicate attribute records are allowed; they are
    a natural consequence of independent sampling and land in the k = 5 group.
    """
    generator = torch.Generator().manual_seed(DATA_SEED)
    columns = {"record_id": pl.Series(range(RECORD_COUNT), dtype=pl.Int32)}
    for name, cardinality in PROPERTIES.items():
        draws = torch.randint(0, cardinality, (RECORD_COUNT,), generator=generator)
        columns[name] = pl.Series(draws.tolist(), dtype=pl.Int16)
    return pl.DataFrame(columns)


def check(records: pl.DataFrame) -> dict:
    """Sanity checks, plus the pair-overlap shares the methodology quotes."""
    if records.height != RECORD_COUNT or records["record_id"].to_list() != list(
        range(RECORD_COUNT)
    ):
        raise AssertionError("Records must be numbered 0..N-1 in sequence order")
    for name, cardinality in PROPERTIES.items():
        low, high = records[name].min(), records[name].max()
        if low != 0 or high != cardinality - 1:
            raise AssertionError(f"{name} codes should span 0..{cardinality - 1}")
    # Probability that a random pair shares exactly k properties, under uniform draws.
    match = [1 / c for c in PROPERTIES.values()]
    shares = [1.0]
    for p in match:
        shares = [
            (shares[k] if k < len(shares) else 0) * (1 - p)
            + (shares[k - 1] * p if k >= 1 else 0)
            for k in range(len(shares) + 1)
        ]
    duplicates = records.height - records.select(list(PROPERTIES)).n_unique()
    return {
        "pair_share_by_k": shares,
        "duplicate_attribute_records": duplicates,
    }


def main():
    records = generate()
    facts = check(records)
    records.write_parquet(RECORDS_PATH, compression="zstd", statistics=False)
    settings = {
        "record_count": RECORD_COUNT,
        "data_seed": DATA_SEED,
        "properties": PROPERTIES,
        "sampling": "uniform and independent per property, torch.randint, one generator",
        "torch_version": torch.__version__,
        **facts,
    }
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2) + "\n")
    print(json.dumps(settings, indent=2))


if __name__ == "__main__":
    main()
