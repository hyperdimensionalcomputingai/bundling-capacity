"""Build the missingness inputs on top of the scale study's factorial PERSON fixture.

Parked from HYP-83, where one generator wrote both. Every random stream is named
(seed string plus a purpose), so splitting the generator reproduces the original
files byte for byte. Reads data/scale/signatures.parquet; run from the repository root:

    uv run --project src/missingness python data/missingness/generate.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import torch

HERE = Path(__file__).resolve().parent
SIGNATURES = HERE.parent / "scale" / "signatures.parquet"
SEED = "hyp83-person-v2"  # the scale fixture's seed string, kept so the files stay identical
FIELDS = ("age_band", "job_category", "home_region", "interests")
SCALAR_FIELDS = FIELDS[:3]
NULL_SPELLINGS = ("null", "no_value", "")

# HYP-83 Experiment 3: every field independently missing at this rate, one fixed mask.
MISSING_RATE = 0.3
# HYP-83 Experiment 2: pairs per pair type.
PAIRS = 2_000


def rng(*parts: object) -> torch.Generator:
    digest = hashlib.sha256("|".join(str(p) for p in (SEED, *parts)).encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "big") >> 1)


def missingness(records: int) -> pl.DataFrame:
    """The 30% mask: each field of each record independently missing at MISSING_RATE."""
    columns = {"record_index": torch.arange(records, dtype=torch.int32).numpy()}
    for field in FIELDS:
        columns[f"{field}_missing"] = (
            torch.rand(records, generator=rng("mask", field)) < MISSING_RATE
        ).numpy()
    return pl.DataFrame(columns)


def pairwise_pairs(frame: pl.DataFrame) -> pl.DataFrame:
    """Pairwise pairs.

    same_person: a record and itself. unrelated: two records with zero content
    similarity (age bands 1 and 10, different job and region, disjoint interests),
    which stays zero however many scalar fields are removed. Each pair also fixes
    the order in which scalar fields go missing and, for the mixed-spelling
    strategy, how each record spells each missing field.
    """
    g = rng("pairs")
    records = frame.height
    sig = frame.select(
        "age_band", "job_category", "home_region", "interest_1", "interest_2", "interest_3"
    ).to_torch()
    youngest, oldest = (torch.nonzero(sig[:, 0] == a).flatten() for a in (0, 9))
    rows = []
    for pair in range(PAIRS):
        a = int(torch.randint(records, (1,), generator=g))
        rows.append(("same_person", pair, a, a))
    pair = 0
    while pair < PAIRS:
        a = int(youngest[torch.randint(len(youngest), (1,), generator=g)])
        b = int(oldest[torch.randint(len(oldest), (1,), generator=g)])
        if (
            sig[a, 1] != sig[b, 1]
            and sig[a, 2] != sig[b, 2]
            and not set(sig[a, 3:].tolist()) & set(sig[b, 3:].tolist())
        ):
            rows.append(("unrelated", pair, a, b))
            pair += 1
    order = torch.stack([torch.randperm(len(SCALAR_FIELDS), generator=g) for _ in rows])
    spelling = torch.randint(len(NULL_SPELLINGS), (len(rows), 2, len(SCALAR_FIELDS)), generator=g)
    return pl.DataFrame(
        {
            "pair_type": [r[0] for r in rows],
            "pair_id": [r[1] for r in rows],
            "record_a": [r[2] for r in rows],
            "record_b": [r[3] for r in rows],
            **{
                f"drop_{i + 1}": order[:, i].to(torch.int8).numpy()
                for i in range(len(SCALAR_FIELDS))
            },
            **{
                f"spelling_{side}_{field}": spelling[:, s, f].to(torch.int8).numpy()
                for s, side in enumerate("ab")
                for f, field in enumerate(SCALAR_FIELDS)
            },
        }
    )


def main():
    frame = pl.read_parquet(SIGNATURES)
    if frame.height != 920_000:
        raise AssertionError("Expected the 920,000-signature scale fixture")
    missingness(frame.height).write_parquet(HERE / "missingness.parquet", statistics=False)
    pairwise_pairs(frame).write_parquet(HERE / "pairwise_pairs.parquet", statistics=False)
    (HERE / "settings.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "null_spellings": NULL_SPELLINGS,
                "missing_rate": MISSING_RATE,
                "pairs_per_type": PAIRS,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote the 30% mask and {PAIRS * 2:,} pairs to {HERE}")


if __name__ == "__main__":
    main()
