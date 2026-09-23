"""Test whether similar questionnaire bundles stay close as facts accumulate."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

import lancedb
import polars as pl
import torch
import torchhd

from run import ROOT, SEEDS, bundles_at_widths, encoder, read_vectors, source


DIMENSION = 4096
WIDTHS = (5, 10, 20, 40, 50)
VARIANTS_PER_PROFILE = 24
CONDITIONS = {"near": 0.2, "far": 0.6}


def variants(raw_answers: list[int], respondent_id: str, width: int, condition: str):
    """Unique exact one-step changes, shared across all encoder seeds."""
    count = round(width * CONDITIONS[condition])
    rng = random.Random(int.from_bytes(hashlib.sha256(
        f"qa-encoding-v1:{respondent_id}:{width}:{condition}".encode()
    ).digest()[:8], "big"))

    def moves(positions):
        choices = [(1,) if raw_answers[i] == 1 else (-1,) if raw_answers[i] == 5
                   else (-1, 1) for i in positions]
        for directions in itertools.product(*choices):
            yield tuple(zip(positions, directions))

    # At five facts the finite set is smaller than 24. Enumerate it rather
    # than repeatedly sampling the same variant and overstating precision.
    if width == 5:
        options = [change for positions in itertools.combinations(range(width), count)
                   for change in moves(positions)]
        rng.shuffle(options)
        return options[:VARIANTS_PER_PROFILE]

    output = set()
    while len(output) < VARIANTS_PER_PROFILE:
        positions = tuple(sorted(rng.sample(range(width), count)))
        output.add(tuple((i, rng.choice((1,) if raw_answers[i] == 1 else
                                        (-1,) if raw_answers[i] == 5 else (-1, 1)))
                         for i in positions))
    return sorted(output)


def check_sources(data_dir: Path, result_dir: Path):
    manifest = json.loads((result_dir / "manifest.json").read_text())
    if DIMENSION not in manifest["dimensions"] or not set(WIDTHS) <= set(manifest["widths"]):
        raise ValueError("Run the base experiment with D=4096 and all required widths first")
    if set(SEEDS) != set(manifest["seeds"]):
        raise ValueError("The base experiment must include all five MAP seeds")
    for name, expected in manifest["source_sha256"].items():
        if hashlib.sha256((data_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source data changed since the base experiment: {name}")


def run(data_dir: Path, result_dir: Path):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    check_sources(data_dir, result_dir)
    _, respondent_ids, raw, scored, reversed_items, _ = source(data_dir)
    db = lancedb.connect(str(result_dir / "experiment.lancedb"))
    metadata, vectors = read_vectors(db.open_table(f"profiles_d{DIMENSION}"), DIMENSION, symbols=False)
    original = {(m["seed"], m["bundle_width"], m["respondent_id"]): vectors[i]
                for i, m in enumerate(metadata)}
    if len(original) != len(metadata):
        raise ValueError("Duplicate stored profile bundles")

    # Raw perturbations are chosen once and reused for every MAP seed.
    changes = {(p, width, condition): variants(raw[p, :width].tolist(), respondent_id,
                                                width, condition)
               for p, respondent_id in enumerate(respondent_ids)
               for width in WIDTHS for condition in CONDITIONS}
    rows = []
    for seed in SEEDS:
        keys, levels, level_cosines = encoder(DIMENSION, seed)
        bound = torchhd.bind(keys[:, None, :], levels[None, :, :])
        for width in WIDTHS:
            for p, respondent_id in enumerate(respondent_ids):
                base = original[seed, width, respondent_id]
                raw_answers = raw[p, :width].tolist()
                for condition in CONDITIONS:
                    for variant_id, change in enumerate(changes[p, width, condition]):
                        positions = [i for i, _ in change]
                        new_raw = [raw_answers[i] + step for i, step in change]
                        old_scored = scored[p, positions].long() - 1
                        new_scored = torch.tensor([
                            6 - answer if reversed_items[i] else answer
                            for i, answer in zip(positions, new_raw)
                        ], dtype=torch.long) - 1
                        # A MAP bundle is an unthresholded sum. Replacing each
                        # selected bound fact is exact, including float16 storage.
                        altered = base + (bound[positions, new_scored] -
                                          bound[positions, old_scored]).sum(dim=0)
                        if p == 0 and condition == "near" and variant_id == 0:
                            changed = scored[p, :width].clone()
                            changed[positions] = new_scored.to(changed.dtype) + 1
                            rebuilt = bundles_at_widths(keys, levels, changed[None, :], (width,))[width][0]
                            if not torch.equal(altered, rebuilt):
                                raise AssertionError("Replacement differs from fresh MAP bundle")
                        cosine = float(torchhd.cosine_similarity(base, altered))
                        oracle = (width - len(positions) +
                                  float(level_cosines[old_scored, new_scored].sum())) / width
                        rows.append({
                            "seed": seed, "bundle_width": width,
                            "respondent_id": respondent_id, "condition": condition,
                            "variant_id": variant_id,
                            "changed_positions": json.dumps([i + 1 for i in positions]),
                            "changed_raw_answers": json.dumps(new_raw),
                            "fraction_changed": len(positions) / width,
                            "bundle_cosine": cosine, "oracle_codebook_cosine": oracle,
                        })
        print(f"MAP seed {seed}: {len(rows):,} pair measurements so far", flush=True)

    pairs = pl.DataFrame(rows)
    pairs.write_parquet(result_dir / "perturbation_pairs.parquet")
    # Profile first: the number of unique five-fact variants depends on
    # endpoint answers, and should not change that profile's weight.
    profile_means = (pairs.group_by("bundle_width", "condition", "seed", "respondent_id")
                     .agg(pl.col("bundle_cosine").mean().alias("mean_cosine"),
                          pl.col("oracle_codebook_cosine").mean().alias("mean_oracle"),
                          pl.len().alias("variants")))
    summary = (profile_means.group_by("bundle_width", "condition")
               .agg(pl.col("mean_cosine").mean().alias("bundle_cosine"),
                    pl.col("mean_oracle").mean().alias("oracle_cosine"),
                    pl.col("variants").sum().alias("pairs"),
                    pl.len().alias("profile_seed_units"))
               .sort("bundle_width", "condition"))
    summary.write_csv(result_dir / "perturbation_summary.csv")
    distinct = (pl.read_parquet(result_dir / "profile_pairs.parquet")
                .filter(pl.col("dimension") == DIMENSION,
                        pl.col("bundle_width").is_in(WIDTHS))
                .group_by("bundle_width")
                .agg(pl.len().alias("distinct_pairs"),
                     pl.col("mean_absolute_response_difference").min().alias("smallest_raw_difference"),
                     (pl.col("mean_absolute_response_difference") <= .200001)
                     .sum().alias("naturally_near_pairs"))
                .sort("bundle_width"))
    distinct.write_csv(result_dir / "perturbation_distinct_profiles.csv")
    (result_dir / "perturbation_design.json").write_text(json.dumps({
        "dimension": DIMENSION, "bundle_widths": WIDTHS,
        "conditions": {"near": "20% of raw answers change by one level",
                       "far": "60% of raw answers change by one level"},
        "variants_per_profile_maximum": VARIANTS_PER_PROFILE,
        "seeds": SEEDS,
        "aggregation": "average variants within profile and seed, then average profile-seed units",
        "raw_perturbations_reused_across_seeds": True,
    }, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "qa-encoding")
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "qa-encoding")
    args = parser.parse_args()
    run(args.data_dir, args.results)
