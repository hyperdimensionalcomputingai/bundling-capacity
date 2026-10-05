"""Experiment 4: how much recall does a LanceDB IVF_PQ index lose against exact search?

Runs after run.py. It indexes the stored complete-record table (D=2048, seed 11)
and searches it with the evaluation panel. The exact top 10 from Experiment 1 is
the ground truth. Stored coordinates are small integers, so float16 storage is
exact and any loss here belongs to the index.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from pathlib import Path

import experiment as ex
import lancedb
import person
import polars as pl
import store
import torch
from lancedb.index import IvfPq
from run import INDEXED, OUT, load_thresholds, panel

NPROBES = (10, 20, 50, 100)
REFINE = (None, 10, 50)
K = ex.TOP_K


def run(out: Path, log=print):
    dimension, seed = INDEXED
    threshold = load_thresholds(out)[(dimension, seed, "complete")]
    db = store.connect(out / "experiment.lancedb")
    table = db.open_table(store.records_table(dimension, seed))
    complete = person.complete_records(person.load_fixture())
    offsets, ranks = panel("evaluation")
    _, q_sum = store.read_rows(table, dimension, offsets)

    def exact(q: int, candidates: list[int]) -> torch.Tensor:
        """Exact cosine from stored integer vectors, so identical vectors tie exactly."""
        return person.cosine(q_sum[q : q + 1], store.read_rows(table, dimension, candidates)[1])[0]

    truth = pl.read_parquet(out / "exact_head.parquet").filter(
        (pl.col("dimension") == dimension)
        & (pl.col("seed") == seed)
        & (pl.col("condition") == "complete")
        & (pl.col("rank") < K)
    )
    truth = {
        r: g.sort("rank")["record_index"].to_list() for (r,), g in truth.group_by("query_rank")
    }
    exact_kth = {q: float(exact(q, truth[r]).min()) for q, r in enumerate(ranks.tolist())}
    truth_related = {}
    for q, (offset, r) in enumerate(zip(offsets, ranks.tolist())):
        base = person.baselines(complete.take([offset]), complete.take(truth[r]), "omit")
        truth_related[q] = int((ex.tiers(base.content)[0] == 3).sum())

    rows = table.count_rows()
    config = IvfPq(
        distance_type="cosine",
        num_partitions=round(math.sqrt(rows)),
        num_sub_vectors=dimension // 16,
    )
    started = time.time()
    table.create_index("vector", config=config)
    build_seconds = round(time.time() - started)
    log(f"IVF_PQ index on {rows:,} rows built in {build_seconds}s")

    results = []
    for nprobes, refine in itertools.product(NPROBES, REFINE):
        for q, (offset, rank) in enumerate(zip(offsets, ranks.tolist())):
            t0 = time.perf_counter()
            search = (
                table.search(q_sum[q].tolist(), vector_column_name="vector")
                .distance_type("cosine")
                .nprobes(nprobes)
                .limit(K + 1)
                .select(["record_index", "_distance"])
            )
            if refine is not None:
                search = search.refine_factor(refine)
            found = search.to_arrow()
            latency = (time.perf_counter() - t0) * 1000
            kept = [
                (i, 1 - d)
                for i, d in zip(found["record_index"].to_pylist(), found["_distance"].to_pylist())
                if i != offset
            ][:K]
            returned = [i for i, _ in kept]
            index_score = torch.tensor([s for _, s in kept])
            exact_score = exact(q, returned)
            tier = ex.tiers(
                person.baselines(complete.take([offset]), complete.take(returned), "omit").content
            )[0]
            results.append(
                {
                    "nprobes": nprobes,
                    "refine_factor": refine,
                    "query_rank": rank,
                    "returned": len(returned),
                    "recall_at_10": float((exact_score >= exact_kth[q] - 1e-6).sum()) / K,
                    "related_in_exact_top10": truth_related[q],
                    "related_returned": int((tier == 3).sum()),
                    "unrelated_above_t_index_score": int(
                        ((tier == 0) & (index_score >= threshold)).sum()
                    ),
                    "mean_abs_score_error": float((index_score - exact_score).abs().mean()),
                    "max_abs_score_error": float((index_score - exact_score).abs().max()),
                    "latency_ms": latency,
                }
            )
    frame = pl.DataFrame(results, infer_schema_length=None)
    frame.write_parquet(out / "ivfpq_per_query.parquet")
    (
        frame.group_by("nprobes", "refine_factor")
        .agg(
            pl.len().alias("queries"),
            pl.col("recall_at_10").mean(),
            pl.col("recall_at_10").quantile(0.05).alias("recall_at_10_p05"),
            (pl.col("recall_at_10") < 1).mean().alias("queries_with_any_miss"),
            (pl.col("related_in_exact_top10") - pl.col("related_returned"))
            .clip(0)
            .sum()
            .alias("related_missed"),
            pl.col("related_in_exact_top10").sum(),
            pl.col("unrelated_above_t_index_score").sum(),
            pl.col("mean_abs_score_error").mean(),
            pl.col("max_abs_score_error").max(),
            pl.col("latency_ms").median().alias("median_latency_ms"),
        )
        .sort("nprobes", "refine_factor", nulls_last=False)
        .write_csv(out / "ivfpq_summary.csv", float_precision=5)
    )
    (out / "ivfpq_index.json").write_text(
        json.dumps(
            {
                "table": store.records_table(dimension, seed),
                "rows": rows,
                "distance": "cosine",
                "storage": "float16",
                "num_partitions": config.num_partitions,
                "num_sub_vectors": config.num_sub_vectors,
                "num_bits": config.num_bits,
                "max_iterations": config.max_iterations,
                "sample_rate": config.sample_rate,
                "build_seconds": build_seconds,
                "nprobes": NPROBES,
                "refine_factor": REFINE,
                "ground_truth": "exact top 10 from Experiment 1 (self excluded)",
                "recall": "tie-aware: a returned record counts when its exact score >= the exact 10th score - 1e-6, "
                "both recomputed from the stored integer vectors",
                "latency": "logged for context only; single-threaded Python loop, warm cache; not a studied outcome",
                "lancedb_version": lancedb.__version__,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    torch.set_num_threads(os.cpu_count())
    run(args.output_dir, lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
