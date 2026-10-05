"""Prototype of HYP-118 Experiment 1: does normalizing each property change how much missing interests cost?

One query and a handful of hand-built candidates, encoded two ways from the same
atoms (plan: EXPERIMENT_UPDATE.md in this directory):

  without_normalization  the current encoder: every known value is one bound fact,
                         so three interests contribute three terms
  with_normalization     each known property is scaled to unit length before
                         bundling, so every observed property has the same norm

Candidates:

  A_k  matches the query on age, job and region, and knows k = 0..3 of the
       query's three interests (the rest are unknown and omitted, not mismatched)
  B    complete: matches age, region and all three interests, but has a different job

The A_k curve isolates the cost of missing interests; B is the reference A has to
beat. Each MAP cosine is reported next to its exact value if atoms were orthogonal.

Records are built directly; no fixture or LanceDB store is needed. Atoms come from
`person.codebook`, the same deterministic seeding the store holds. The store keeps
only seeds 11, 23 and 37, and this experiment repeats over 100 seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
import torch

import person
from person import MISSING, PRESENT, Records

DIMENSION = 2048  # the dimension Experiment 3 runs at
SEEDS = tuple(range(1, 101))  # repeats, because without normalization A_1 and B differ by ~0.017
WORKED_SEED = 11  # the seed shown as the worked example; not chosen for its result

# Arbitrary concrete values. Every compared value either matches exactly or comes
# from an independent atom, so the expected scores do not depend on which values these are.
QUERY = {"age_band": 4, "job_category": 2, "home_region": 1, "interests": (3, 9, 17)}
B_JOB = 5
KNOWN_INTERESTS = (0, 1, 2, 3)
ENCODINGS = ("without_normalization", "with_normalization")


def build_records() -> tuple[Records, list[dict]]:
    """Row 0 is the query, then A_0..A_3, then B. A_k keeps the first k query interests."""
    rows = [("query", None, QUERY["job_category"], QUERY["interests"])]
    rows += [
        (f"A_{k}", "A", QUERY["job_category"], QUERY["interests"][:k]) for k in KNOWN_INTERESTS
    ]
    rows.append(("B", "B", B_JOB, QUERY["interests"]))
    n = len(rows)
    value = torch.tensor(
        [[QUERY["age_band"], job, QUERY["home_region"]] for _, _, job, _ in rows], dtype=torch.int8
    )
    interests = torch.full((n, person.MAX_INTERESTS), -1, dtype=torch.int8)
    state = torch.full((n, 4), PRESENT, dtype=torch.int8)
    for i, (*_, known) in enumerate(rows):
        interests[i, : len(known)] = torch.tensor(known)
        if not known:
            state[i, 3] = MISSING  # A_0: the whole interest property is unknown
    records = Records(torch.arange(n), state, value, interests, torch.zeros(n, 4, dtype=torch.int8))
    meta = [
        {"candidate": name, "candidate_type": kind, "known_interests": len(known)}
        for name, kind, _, known in rows[1:]
    ]
    return records, meta


def score(book: person.Codebook, records: Records) -> dict[str, torch.Tensor]:
    """Cosine of the query (row 0) against every candidate, for both encodings."""
    out = {}
    for name, sums in (
        ("without_normalization", person.encode_sum(book, records, "omit")),
        ("with_normalization", person.encode_normalized(book, records)),
    ):
        out[name] = person.cosine(sums[:1], sums[1:])[0]
    return out


def expected(records: Records) -> dict[str, torch.Tensor]:
    """Exact cosines with orthogonal atoms (the plan's predictions)."""
    query = records.take(torch.zeros(len(records) - 1, dtype=torch.long))
    candidates = records.take(torch.arange(1, len(records)))
    return {
        "without_normalization": person.baselines(query, candidates, "omit", paired=True).encoder,
        "with_normalization": person.normalized_baseline(query, candidates),
    }


def run(out: Path, log=print):
    records, meta = build_records()
    ideal = expected(records)
    rows = []
    for seed in SEEDS:
        measured = score(person.codebook(DIMENSION, seed), records)
        for encoding in ENCODINGS:
            rows.append(
                pl.DataFrame(meta).with_columns(
                    encoding=pl.lit(encoding),
                    dimension=pl.lit(DIMENSION),
                    seed=pl.lit(seed),
                    s_map=pl.Series(measured[encoding].numpy()),
                    s_encoder=pl.Series(ideal[encoding].numpy()),
                )
            )
    frame = pl.concat(rows)
    frame.write_parquet(out / "normalization.parquet")

    keys = ["encoding", "candidate", "candidate_type", "known_interests"]
    summary = (
        frame.group_by(keys)
        .agg(
            pl.len().alias("seeds"),
            pl.col("s_encoder").first().alias("expected_cosine"),
            pl.col("s_map").filter(pl.col("seed") == WORKED_SEED).first().alias("seed_11_cosine"),
            pl.col("s_map").mean().alias("measured_mean"),
            pl.col("s_map").std().alias("measured_sd"),
            pl.col("s_map").min().alias("measured_min"),
            pl.col("s_map").max().alias("measured_max"),
        )
        .sort("encoding", "candidate", descending=[True, False])
    )
    summary.write_csv(out / "normalization_summary.csv", float_precision=5)

    # Each A_k against B in the same seed and encoding. The predicted winner comes from
    # the orthogonal-atom scores; a seed "holds" when the measured scores agree with it.
    b = frame.filter(pl.col("candidate") == "B").select(
        "encoding", "seed", b_map=pl.col("s_map"), b_encoder=pl.col("s_encoder")
    )
    rankings = (
        frame.filter(pl.col("candidate_type") == "A")
        .join(b, on=["encoding", "seed"])
        .with_columns(
            predicted_winner=pl.when(pl.col("s_encoder") > pl.col("b_encoder"))
            .then(pl.lit("A"))
            .otherwise(pl.lit("B")),
            gap=pl.col("s_map") - pl.col("b_map"),
        )
        .with_columns(holds=(pl.col("gap") > 0) == (pl.col("predicted_winner") == "A"))
        .group_by("encoding", "candidate", "known_interests", "predicted_winner")
        .agg(
            (pl.col("s_encoder") - pl.col("b_encoder")).first().alias("expected_gap_a_minus_b"),
            pl.col("gap").mean().alias("measured_gap_mean"),
            pl.col("gap").std().alias("measured_gap_sd"),
            pl.col("holds").mean().alias("share_of_seeds_prediction_holds"),
            pl.col("holds").filter(pl.col("seed") == WORKED_SEED).first().alias("holds_at_seed_11"),
        )
        .sort("encoding", "known_interests", descending=[True, False])
    )
    rankings.write_csv(out / "normalization_rankings.csv", float_precision=5)

    config = {
        "plan": "src/missingness/EXPERIMENT_UPDATE.md",
        "dimension": DIMENSION,
        "seeds_inclusive": [SEEDS[0], SEEDS[-1]],
        "worked_seed": WORKED_SEED,
        "atom_seeding": "torch.Generator().manual_seed(1000 * seed + offset) per family",
        "query": QUERY,
        "candidate_b_job": B_JOB,
        "candidate_a": "first k of the query's interests known, k in 0..3; the rest omitted",
        "dtype": "float32",
        "normalization": "each known property's bound facts divided by their L2 norm; no sign threshold",
    }
    (out / "normalization_config.json").write_text(json.dumps(config, indent=2) + "\n")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3, tbl_width_chars=160):
        log(
            summary.select(
                keys[:2] + ["expected_cosine", "seed_11_cosine", "measured_mean", "measured_sd"]
            )
        )
        log(rankings.drop("candidate"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=person.OUT)
    run(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
