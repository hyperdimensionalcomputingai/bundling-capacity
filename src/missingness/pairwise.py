"""HYP-83 Experiment 2 (parked): what a missing value does to one pairwise comparison.

For each fixed pair in data/missingness/pairwise_pairs.parquet and each
k = 0..3 missing scalar fields (age, job, region; removed in the pair's fixed
order, interests always kept so a complete record has six facts):

  same_person  a complete record against itself with k fields missing
               -> how much does missingness dilute self-similarity?
  unrelated    two records with zero content similarity, both missing the same
               k fields -> does a shared null make strangers look alike?

Strategies: omit the fact; a shared token spelled "null", "no_value" or "" (each
its own random vector bound to the field role); or mixed spellings, where each
record's null is spelled independently from those three. Each MAP cosine is
reported next to S_encoder, its exact value if atoms were orthogonal.
"""

from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

import polars as pl
import torch

import person

DIMENSIONS = (512, 2048, 8192)
SEEDS = (11, 23, 37)
MISSING = (0, 1, 2, 3)
# name -> spelling index for every null, or None for each record's own mixed spelling
STRATEGIES = {
    "omit": ("omit", 0),
    "token_null": ("token", 0),
    "token_no_value": ("token", 1),
    "token_empty_string": ("token", 2),
    "token_mixed_spellings": ("token", None),
}


def sparse(
    records: person.Records, order: torch.Tensor, k: torch.Tensor, spelling
) -> person.Records:
    """Remove the first k[i] fields of row i's drop order; spell every null as `spelling`."""
    position = order.long().argsort(1)  # where each scalar field falls in the order
    missing = torch.cat([position < k[:, None], torch.zeros(len(records), 1, dtype=torch.bool)], 1)
    if isinstance(spelling, int):
        spelling = torch.full((len(records), 4), spelling)
    return records.without(missing, spelling)


def run(out: Path, log=print):
    pairs = pl.read_parquet(person.DATA / "pairwise_pairs.parquet")
    complete = person.complete_records(person.load_fixture())
    a = complete.take(pairs["record_a"].to_torch().long())
    b = complete.take(pairs["record_b"].to_torch().long())
    order = pairs.select("drop_1", "drop_2", "drop_3").to_torch()
    pad = torch.zeros(len(pairs), 1, dtype=torch.int8)  # interests are never removed here
    mixed = {
        side: torch.cat(
            [pairs.select(f"spelling_{side}_{f}" for f in person.FIELDS[:3]).to_torch(), pad], 1
        )
        for side in "ab"
    }
    same_person = torch.tensor(pairs["pair_type"].to_numpy() == "same_person")
    rows = []
    for dimension, seed in itertools.product(DIMENSIONS, SEEDS):
        # The same atoms the scale study stores in LanceDB (tested there); regenerated here
        # so this experiment needs no store.
        book = person.codebook(dimension, seed)
        for k, (name, (strategy, spelling)) in itertools.product(MISSING, STRATEGIES.items()):
            sa, sb = (mixed["a"], mixed["b"]) if spelling is None else (spelling, spelling)
            # same_person keeps side a complete; unrelated pairs lose the same k fields on both sides.
            left = sparse(a, order, torch.where(same_person, 0, k), sa)
            right = sparse(b, order, torch.full((len(pairs),), k), sb)
            s_map = person.cosine(
                person.encode_sum(book, left, strategy),
                person.encode_sum(book, right, strategy),
                paired=True,
            )
            base = person.baselines(left, right, strategy, paired=True)
            rows.append(
                pl.DataFrame(
                    {
                        "pair_type": pairs["pair_type"],
                        "pair_id": pairs["pair_id"],
                        "missing_fields": k,
                        "strategy": name,
                        "dimension": dimension,
                        "seed": seed,
                        "s_map": s_map.numpy(),
                        "s_encoder": base.encoder.numpy(),
                        "s_content": base.content.numpy(),
                    }
                )
            )
        log(f"pairwise D={dimension:,} seed={seed}")
    frame = pl.concat(rows)
    frame.write_parquet(out / "pairwise.parquet")
    (
        frame.group_by("pair_type", "missing_fields", "strategy", "dimension")
        .agg(
            pl.len().alias("comparisons"),
            pl.col("s_content").mean().alias("content_similarity"),
            pl.col("s_encoder").mean().alias("expected_cosine"),
            pl.col("s_map").mean().alias("measured_cosine"),
            pl.col("s_map").quantile(0.05).alias("measured_p05"),
            pl.col("s_map").quantile(0.95).alias("measured_p95"),
            (pl.col("s_map") - pl.col("s_encoder")).abs().quantile(0.99).alias("abs_map_error_p99"),
        )
        .sort("pair_type", "strategy", "missing_fields", "dimension")
        .write_csv(out / "pairwise_summary.csv", float_precision=5)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=person.OUT)
    args = parser.parse_args()
    torch.set_num_threads(os.cpu_count())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run(args.output_dir, lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
