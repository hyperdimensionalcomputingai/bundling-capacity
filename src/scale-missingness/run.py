"""Experiments 1 and 3: build the LanceDB store, calibrate thresholds, freeze them, evaluate.

Stages (all run by default, in order):

  build       every atom for every (D, seed), plus the complete-record bundles
              Experiment 4 indexes (D=2048, seed 11), as float16 in LanceDB
  calibrate   calibration panel against the first 100,000 candidates; writes thresholds.json
  evaluate    evaluation panel against all 920,000 candidates, every codebook and
              condition in one streamed pass. Refuses to run unless thresholds.json
              matches the fixture and settings.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import experiment as ex
import lancedb
import person
import polars as pl
import pyarrow as pa
import scipy
import store
import torch
import torchhd

OUT = person.ROOT / "results" / "scale-missingness"
UNITS = list(itertools.product(ex.DIMENSIONS, ex.SEEDS))
INDEXED = (2048, 11)  # the one stored record table, used by Experiment 4


def settings() -> dict:
    return {
        "dimensions": ex.DIMENSIONS,
        "seeds": ex.SEEDS,
        "prefixes": ex.PREFIXES,
        "conditions": [[c.name, c.strategy, c.masked, list(c.dimensions)] for c in ex.CONDITIONS],
        "calibration_candidates": ex.CALIBRATION_CANDIDATES,
        "related_cutoff": ex.RELATED_CUTOFF,
        "tiers": ex.TIER_NAMES,
        "beta": ex.BETA,
        "epsilon": ex.EPSILON,
        "reference_threshold": ex.REFERENCE_T,
        "top_k": ex.TOP_K,
        "head": ex.HEAD,
        "histogram_bin": ex.BIN,
        "chunk": ex.CHUNK,
    }


def condition_records():
    signatures = person.load_fixture()
    complete = person.complete_records(signatures)
    masked = person.masked_records(signatures)
    return {c: masked if c.masked else complete for c in ex.CONDITIONS}, complete


def panel(name: str) -> tuple[list[int], torch.Tensor]:
    rows = (
        pl.read_parquet(person.DATA / "query_panels.parquet")
        .filter(pl.col("panel") == name)
        .sort("query_rank")
    )
    return rows["record_index"].to_list(), rows["query_rank"].to_torch()


def build(out: Path, log=print):
    """Write atoms and the indexed record table; skipped when the store already matches."""
    marker = out / "store.json"
    expected = {
        "fixture_sha256": person.fixture_hashes(),
        "dimensions": ex.DIMENSIONS,
        "seeds": ex.SEEDS,
        "indexed": INDEXED,
    }
    if marker.exists() and json.loads(marker.read_text()) == json.loads(json.dumps(expected)):
        log("store is current; skipping build")
        return
    db = store.connect(out / "experiment.lancedb")
    for dimension in ex.DIMENSIONS:
        store.write_symbols(db, dimension, ex.SEEDS)
    started = time.time()
    _, complete = condition_records()
    store.write_records(
        db, store.records_table(*INDEXED), store.read_codebook(db, *INDEXED), complete, ex.CHUNK
    )
    log(
        f"stored atoms for {len(UNITS)} codebooks and {len(complete):,} bundles at D={INDEXED[0]} "
        f"in {time.time() - started:.0f}s"
    )
    marker.write_text(json.dumps(expected, indent=2) + "\n")


def calibrate(out: Path, log=print):
    db = store.connect(out / "experiment.lancedb")
    records, _ = condition_records()
    offsets, ranks = panel("calibration")
    state, _, _ = ex.run_pass(
        db, records, offsets, ranks, UNITS, candidates=ex.CALIBRATION_CANDIDATES, log=log
    )
    rows = []
    for (u, c), unit in state.items():
        threshold, pairs, achieved = ex.calibrated_threshold(unit.hist.segment("related_score", 0))
        rows.append(
            {
                "dimension": u[0],
                "seed": u[1],
                "condition": c.name,
                "threshold": threshold,
                "calibration_related_pairs": pairs,
                "calibration_recall": achieved,
            }
        )
    (out / "thresholds.json").write_text(
        json.dumps(
            {
                "rule": "T = largest 1e-4 grid value with at most beta of the calibration panel's related pairs "
                "(S_content >= 2/3) against the first 100,000 candidates scoring below T; calibrated per "
                "(D, seed, condition) and held fixed across candidate counts",
                "fixture_sha256": person.fixture_hashes(),
                "settings": settings(),
                "thresholds": rows,
            },
            indent=2,
        )
        + "\n"
    )
    log(f"calibrated {len(rows)} thresholds")


def load_thresholds(out: Path) -> dict:
    frozen = json.loads((out / "thresholds.json").read_text())
    if frozen["fixture_sha256"] != person.fixture_hashes():
        raise ValueError("Fixture changed after thresholds were frozen; recalibrate")
    if frozen["settings"] != json.loads(json.dumps(settings())):
        raise ValueError("Settings changed after thresholds were frozen; recalibrate")
    return {
        (r["dimension"], r["seed"], r["condition"]): r["threshold"] for r in frozen["thresholds"]
    }


def evaluate(out: Path, log=print):
    frozen = load_thresholds(out)
    db = store.connect(out / "experiment.lancedb")
    records, complete = condition_records()
    offsets, ranks = panel("evaluation")
    thresholds = {
        (u, c): {"calibrated": frozen[(*u, c.name)], "reference": ex.REFERENCE_T}
        for u in UNITS
        for c in ex.CONDITIONS
        if c.applies(u)
    }
    state, sources, per_query = ex.run_pass(
        db, records, offsets, ranks, UNITS, candidates=len(complete), thresholds=thresholds, log=log
    )
    ex.per_query_frame(per_query).write_parquet(out / "per_query.parquet")
    ex.histogram_frame(state).write_parquet(out / "histograms.parquet")
    head = []
    for (u, c), unit in state.items():
        q, k = unit.head["index"].shape
        head.append(
            pl.DataFrame(
                {
                    "dimension": [u[0]] * (q * k),
                    "seed": [u[1]] * (q * k),
                    "condition": [c.name] * (q * k),
                    "query_rank": ranks.repeat_interleave(k).tolist(),
                    "rank": list(range(k)) * q,
                    "record_index": unit.head["index"].reshape(-1).tolist(),
                    "score": unit.head["score"].reshape(-1).tolist(),
                    "content": unit.head["content"].reshape(-1).tolist(),
                }
            )
        )
    pl.concat(head).write_parquet(out / "exact_head.parquet")
    pl.DataFrame(
        [
            {
                "condition": c.name,
                "tier": t,
                "pairs": int(s.pairs[i]),
                "semantic_shift_mean": float(s.shift_sum[i]) / int(s.pairs[i])
                if int(s.pairs[i])
                else None,
            }
            for c, s in sources.items()
            for i, t in enumerate(ex.TIER_NAMES)
        ]
    ).write_csv(out / "semantic_shift.csv", float_precision=6)
    log(f"evaluated {len(state)} codebook-condition units")


def encoder_diagnostics(out: Path):
    db = store.connect(out / "experiment.lancedb")
    rows = []
    for dimension, seed in UNITS:
        book = store.read_codebook(db, dimension, seed)
        atoms = torch.cat([book.atoms[f] for f in person.ATOM_FAMILIES if f != "age_level"])
        cos = torchhd.cosine_similarity(atoms, atoms)
        off = cos[torch.triu_indices(len(atoms), len(atoms), 1).unbind()].abs()
        level = book.level_similarity
        rows.append(
            {
                "dimension": dimension,
                "seed": seed,
                "independent_atoms": len(atoms),
                "mean_abs_atom_cosine": float(off.mean()),
                "max_abs_atom_cosine": float(off.max()),
                **{
                    f"age_{a + 1}_{b + 1}_cosine": float(level[a, b])
                    for a in range(10)
                    for b in range(a + 1, 10)
                },
            }
        )
    pl.DataFrame(rows).write_csv(out / "encoder_diagnostics.csv", float_precision=6)


def write_manifest(out: Path, timings: dict):
    path = out / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest.update(
        {
            "design": "MAP-I: bipolar role and value atoms, element-wise binding, additive bundle, cosine comparison; "
            "exact streamed search",
            "fixture": "controlled factorial PERSON fixture, 10 x 8 x 5 x C(25,3) = 920,000 signatures",
            "fixture_sha256": person.fixture_hashes(),
            "settings": settings(),
            "storage": "LanceDB fixed-size float16 lists: every atom (symbols_d{D}) and the complete-record bundles at "
            "D=2048, seed 11 (records_d2048_s11). Coordinates are small integers, so float16 is exact. "
            "All other bundles are encoded on the fly in float32 from the stored atoms.",
            "compute_dtype": "float32; float64 only for running sums over >1e8 pairs",
            "atom_seed_formula": "torch.Generator().manual_seed(1000 * seed + offset); offsets 1 roles, 2 age levels, "
            "3-5 job/region/interest values, 7 null tokens",
            "zero_content_records": "excluded from candidates and queries under both strategies",
            "self_match": "a query's own record is excluded from its candidates",
            "torch_threads": torch.get_num_threads(),
            "versions": {
                "torch": torch.__version__,
                "torchhd": torchhd.__version__,
                "polars": pl.__version__,
                "pyarrow": pa.__version__,
                "scipy": scipy.__version__,
                "lancedb": lancedb.__version__,
            },
        }
    )
    manifest.setdefault("stage_seconds", {}).update(timings)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", choices=("all", "build", "calibrate", "evaluate"), default="all")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    torch.set_num_threads(os.cpu_count())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = lambda message: print(message, flush=True)
    timings = {}
    for stage, step in (("build", build), ("calibrate", calibrate), ("evaluate", evaluate)):
        if args.stage in ("all", stage):
            started = time.time()
            step(args.output_dir, log)
            timings[stage] = round(time.time() - started)
    if args.stage in ("all", "evaluate"):
        encoder_diagnostics(args.output_dir)
    write_manifest(args.output_dir, timings)


if __name__ == "__main__":
    main()
