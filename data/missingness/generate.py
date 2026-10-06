"""Build the missingness inputs on top of the scale study's factorial PERSON fixture.

  missingness.parquet, pairwise_pairs.parquet   the original study's mask and pairs.
      Every random stream is named (seed string plus a purpose), so splitting the
      generator from the scale fixture reproduces them byte for byte.
  mcar.parquet         Experiments 2, 4 and 5: one missing-completely-at-random
      mask per rate. Age, job and region are each missing with probability r, and
      each of the three interests independently with probability r.
  duplicates.parquet   Experiment 3: a noisy duplicate of each query-panel
      record and of 20,000 training records, at every rate. Each value is replaced
      by a different one with probability ERROR_RATE, the same corruption at every
      rate; then the duplicate gets its own missingness at that rate.

Reads data/scale/signatures.parquet and query_panels.parquet; run from the repository root:

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

# Original retrieval check: every field independently missing at this rate, one fixed mask.
MISSING_RATE = 0.3
# Original pairwise experiment: pairs per pair type.
PAIRS = 2_000

# Design choices for this run, not tuned values. Three rates span light to
# heavy missingness; 10% value errors make duplicates imperfect without swamping them.
MCAR_RATES = (0.1, 0.3, 0.5)
ERROR_RATE = 0.1
TRAINING_DUPLICATES = 20_000  # labelled pairs for the "known m" weights in Experiment 3
SIZES = {"age_band": 10, "job_category": 8, "home_region": 5, "interests": 25}


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


def mcar(records: int) -> pl.DataFrame:
    """One mask per rate. Each scalar field, and each interest slot, is missing independently."""
    parts = []
    for rate in MCAR_RATES:
        columns = {
            "record_index": torch.arange(records, dtype=torch.int32).numpy(),
            "rate": [rate] * records,
        }
        for slot in (*SCALAR_FIELDS, "interest_1", "interest_2", "interest_3"):
            columns[f"{slot}_missing"] = (
                torch.rand(records, generator=rng("mcar", rate, slot)) < rate
            ).numpy()
        parts.append(pl.DataFrame(columns))
    return pl.concat(parts)


def corrupt(values: torch.Tensor, interests: torch.Tensor, g: torch.Generator):
    """Replace each value by a different one with probability ERROR_RATE. Interests stay distinct."""
    values, interests = values.clone().long(), interests.clone().long()
    for f, field in enumerate(SCALAR_FIELDS):
        hit = torch.rand(len(values), generator=g) < ERROR_RATE
        shift = torch.randint(1, SIZES[field], (len(values),), generator=g)
        values[:, f] = torch.where(hit, (values[:, f] + shift) % SIZES[field], values[:, f])
    for k in range(3):
        hit = torch.rand(len(values), generator=g) < ERROR_RATE
        for i in torch.nonzero(hit).flatten().tolist():
            taken = set(interests[i].tolist())
            choices = [v for v in range(SIZES["interests"]) if v not in taken]
            interests[i, k] = choices[int(torch.randint(len(choices), (1,), generator=g))]
    return values, interests


def duplicates(frame: pl.DataFrame, panels: pl.DataFrame) -> pl.DataFrame:
    """Noisy duplicates of the query panels and of training records, at every MCAR rate."""
    g = rng("duplicates")
    panel_records = set(panels["record_index"].to_list())
    pool = torch.tensor([i for i in range(frame.height) if i not in panel_records])
    training = pool[torch.randperm(len(pool), generator=g)[:TRAINING_DUPLICATES]].sort().values
    sources = (
        [(p, r) for p, r in zip(panels["panel"], panels["record_index"])]
        + [("training", int(r)) for r in training]
    )
    index = torch.tensor([r for _, r in sources])
    sig = frame.select(
        "age_band", "job_category", "home_region", "interest_1", "interest_2", "interest_3"
    ).to_torch()[index]
    values, interests = corrupt(sig[:, :3], sig[:, 3:], g)
    parts = []
    for rate in MCAR_RATES:
        m = rng("duplicate-mask", rate)
        drop = torch.rand(len(sources), 6, generator=m) < rate
        v = torch.where(drop[:, :3], -1, values)
        i = torch.where(drop[:, 3:], -1, interests)
        parts.append(
            pl.DataFrame(
                {
                    "purpose": [p for p, _ in sources],
                    "source_record": index.to(torch.int32).numpy(),
                    "rate": [rate] * len(sources),
                    **{f: v[:, j].to(torch.int8).numpy() for j, f in enumerate(SCALAR_FIELDS)},
                    **{f"interest_{k + 1}": i[:, k].to(torch.int8).numpy() for k in range(3)},
                }
            )
        )
    return pl.concat(parts)


def main():
    frame = pl.read_parquet(SIGNATURES)
    if frame.height != 920_000:
        raise AssertionError("Expected the 920,000-signature scale fixture")
    missingness(frame.height).write_parquet(HERE / "missingness.parquet", statistics=False)
    pairwise_pairs(frame).write_parquet(HERE / "pairwise_pairs.parquet", statistics=False)
    mcar(frame.height).write_parquet(HERE / "mcar.parquet", statistics=False)
    panels = pl.read_parquet(SIGNATURES.parent / "query_panels.parquet")
    duplicates(frame, panels).write_parquet(HERE / "duplicates.parquet", statistics=False)
    (HERE / "settings.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "null_spellings": NULL_SPELLINGS,
                "missing_rate": MISSING_RATE,
                "pairs_per_type": PAIRS,
                "mcar_rates": MCAR_RATES,
                "error_rate": ERROR_RATE,
                "training_duplicates": TRAINING_DUPLICATES,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote the original mask and pairs, MCAR masks and duplicates to {HERE}")


if __name__ == "__main__":
    main()
