"""HYP-118 Experiment 5: approximate (IVF) search over normalized, partly-null vectors.

Experiment 4 searched 100,000 records exactly (flat). The scale study measured
IVF_PQ on complete, unnormalized integer bundles. This experiment closes the gap:
LanceDB IVF indexes over the full 920,000 records with 30% of values missing at
random, encoded with per-property L2 normalization (D = 2,048, seed 11), float16.

LanceDB trains an IVF index for one distance type (the k-means partitions and the
PQ codebooks are fit under it), and that type must match the one used to search.
So each configuration gets its own metric-matched index:

  A  unit_cosine      unit-length vectors, IVF_PQ trained and searched with cosine
                      (the recommended configuration)
  B  unit_dot         the same unit-length vectors, IVF_PQ with dot; LanceDB documents
                      dot as equivalent to cosine for unit vectors
  C  pivoted_dot      vectors scaled by Singhal's pivoted length (s = 1.2), so norms
                      vary by coverage; IVF_PQ with dot, the metric this policy needs
  D  unit_cosine_rq   the same unit-length vectors with IVF_RQ (RaBitQ, 1 bit per
                      dimension, LanceDB's default), cosine

Index settings are LanceDB's documented defaults, with the two that size the index
set explicitly so they can't drift between versions and match the scale study's
IVF_PQ: num_partitions = round(sqrt(rows)) and num_sub_vectors = D / 16 (16
dimensions per sub-vector, the SIMD-friendly choice), with 8-bit PQ codes, 50
k-means iterations and sample rate 256 left at their defaults. The distance type
is the only setting that differs between indexes.

Ground truth is exact search over the same stored float16 vectors under the same
metric. Recall@10 is tie-aware: a returned record counts when its exact score
reaches the exact 10th best. Precision@10 against hidden-truth relevance (Experiment
2) is reported for every index setting and for exact flat search over the same
vectors, so the gap between them is what the index costs in result quality. Sweep:
nprobes 10, 20, 50, 100 × refine_factor none, 10, 50 (as in the scale study) and 200.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import lancedb
import polars as pl
import pyarrow as pa
import torch
from lancedb.index import IvfPq, IvfRq

import person
import policy as P
from experiment4_build import DIMENSION, SEED, SLOPE, exact_scores, write_table

RATE = 0.3
K = 10
NPROBES = (10, 20, 50, 100)
REFINE = (None, 10, 50, 200)
CHUNK = 32_768  # exact-search chunk


def stored(vectors: torch.Tensor) -> torch.Tensor:
    """What a float16 column holds, back in float32."""
    return vectors.half().float()


def exact_top(
    queries: torch.Tensor, pool, encode, metric: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact top-K scores and pool positions over the stored float16 vectors."""
    q = queries / queries.norm(dim=1, keepdim=True) if metric == "cosine" else queries
    best = torch.full((len(q), K), -torch.inf)
    where = torch.full((len(q), K), -1, dtype=torch.long)
    for start in range(0, len(pool), CHUNK):
        v = stored(encode(pool.take(slice(start, start + CHUNK))))
        if metric == "cosine":
            v = v / v.norm(dim=1, keepdim=True)
        s = q @ v.T
        top = s.topk(K, 1)
        merged = torch.cat([best, top.values], 1)
        positions = torch.cat([where, top.indices + start], 1)
        keep = merged.topk(K, 1).indices
        best, where = merged.gather(1, keep), positions.gather(1, keep)
    return best, where


def returned_scores(queries, pool, position, encode, metric, returned) -> torch.Tensor:
    """Exact scores (stored vectors, same metric) of each query's returned records."""
    out = torch.full((len(queries), K), torch.nan)
    for i, rows in enumerate(returned):
        idx = torch.tensor([position[r] for r in rows[:K]], dtype=torch.long)
        if len(idx) == 0:
            continue
        v = stored(encode(pool.take(idx)))
        q = queries[i]
        if metric == "cosine":
            v, q = v / v.norm(dim=1, keepdim=True), q / q.norm()
        out[i, : len(idx)] = v @ q
    return out


def sweep(
    table,
    name,
    queries,
    q_records,
    pool,
    position,
    encode,
    metric,
    kth_exact,
    kth_truth,
    complete,
    log,
):
    """Every (nprobes, refine) setting: recall against exact search, precision against truth."""
    rows = []
    for nprobes in NPROBES:
        for refine in REFINE:
            returned, scores, latency = [], [], []
            for q in queries:
                t0 = time.perf_counter()
                search = (
                    table.search(q.tolist(), vector_column_name="vector")
                    .distance_type(metric)
                    .nprobes(nprobes)
                    .limit(K)
                    .select(["record_index", "_distance"])
                )
                if refine is not None:
                    search = search.refine_factor(refine)
                found = search.to_arrow()
                latency.append((time.perf_counter() - t0) * 1000)
                returned.append(found["record_index"].to_pylist())
                d = torch.tensor(found["_distance"].to_pylist())
                scores.append(1 - d)  # cosine distance is 1 - cosine; dot distance is 1 - dot
            exact = returned_scores(queries, pool, position, encode, metric, returned)
            truth = exact_scores(q_records, complete, position, P.relevance, returned)
            index_scores = torch.stack(
                [torch.nn.functional.pad(s, (0, K - len(s)), value=torch.nan) for s in scores]
            )
            recall = (exact >= kth_exact[:, None] - 1e-6).float().mean(1)
            error = (index_scores - exact).abs()
            rows.append(
                {
                    "index": name,
                    "metric": metric,
                    "nprobes": nprobes,
                    "refine_factor": refine,
                    "recall_at_10": float(recall.mean()),
                    "queries_with_any_miss": float((recall < 1).float().mean()),
                    "precision_at_10": float(
                        (truth >= kth_truth[:, None] - P.TOLERANCE).float().mean()
                    ),
                    "mean_abs_score_error": float(error.nanmean()),
                    "max_abs_score_error": float(error[~error.isnan()].max()),
                    "median_latency_ms": float(torch.tensor(latency).median()),
                }
            )
            log(
                f"  {name} nprobes={nprobes} refine={refine}: recall@10={rows[-1]['recall_at_10']:.3f}"
            )
    return rows


# Normalizer check: the cosine metric divides by each MAP vector's realized norm, which
# carries cross-term noise (about 1.4% at D = 2,048 for a complete record). Dividing by
# sqrt(populated fields) instead is exact. Flat search, both ways, over these settings.
NORMALIZER_CHECK = (
    (0.3, 1.0, 2048),
    (0.3, 1.0, 8192),
    (0.1, 1.0, 2048),
    (0.5, 1.0, 2048),
    (0.3, 0.1, 2048),
    (0.3, 0.01, 2048),
)


def normalizer_check(log=print) -> pl.DataFrame:
    """Precision@10 of flat MAP search normalized by realized norm (cosine) or by field count."""
    from experiment2_dial import subsample

    signatures = person.load_fixture()
    complete_all = person.complete_records(signatures)
    evaluation, calibration = person.query_panel("evaluation"), person.query_panel("calibration")
    queries = complete_all.take(evaluation)
    rows = []
    for rate, density, dimension in NORMALIZER_CHECK:
        keep = subsample(len(complete_all), density, torch.cat([calibration, evaluation]))
        pool, complete = person.mcar_records(signatures, rate).take(keep), complete_all.take(keep)
        known = torch.nonzero(P.known(pool)[0].sum(1) > 0).flatten()
        pool, complete = pool.take(known), complete.take(known)
        kth = P.scan_kth(queries, pool, complete, {}, K)
        book = person.codebook(dimension, SEED)
        q = person.encode_normalized(book, queries)
        q = q / q.norm(dim=1, keepdim=True)
        row = {"rate": rate, "density": density, "dimension": dimension, "pool_size": len(pool)}
        for norm in ("realized", "field_count"):

            def encode(r, norm=norm, book=book):
                h = person.encode_normalized(book, r)
                d = h.norm(dim=1) if norm == "realized" else P.known(r)[0].sum(1).float().sqrt()
                return h / d[:, None]

            _, where = exact_top(q, pool, encode, "dot")
            truth = torch.stack(
                [
                    P.relevance(P.compare(queries.take([i]), complete.take(where[i])))[0]
                    for i in range(len(queries))
                ]
            )
            row[f"precision_{norm}"] = float(
                (truth >= kth["relevance"][:, None] - P.TOLERANCE).float().mean()
            )
        rows.append(row)
        log(f"normalizer check {row}")
    return pl.DataFrame(rows).with_columns(
        gap=pl.col("precision_field_count") - pl.col("precision_realized")
    )


def run(out: Path, log=print):
    signatures = person.load_fixture()
    complete_all = person.complete_records(signatures)
    pool = person.mcar_records(signatures, RATE)
    keep = torch.nonzero(P.known(pool)[0].sum(1) > 0).flatten()
    pool, complete = pool.take(keep), complete_all.take(keep)
    position = {int(r): i for i, r in enumerate(pool.record_index.tolist())}
    coverage = P.known(pool)[0].sum(1)
    pivot = P.mean_length(pool)
    q_records = complete_all.take(person.query_panel("evaluation"))
    book = person.codebook(DIMENSION, SEED)

    unit = lambda v: v / v.norm(dim=1, keepdim=True).clamp(min=1e-9)
    l2 = lambda r: person.encode_normalized(book, r)
    length = lambda cov: (1 - SLOPE) * pivot + SLOPE * cov.float().sqrt()
    encoders = {
        "unit": lambda r: unit(l2(r)),
        "pivoted": lambda r: l2(r) / length(P.known(r)[0].sum(1))[:, None],
    }
    q_vectors = unit(l2(q_records))
    # Exact policy precision: what each policy gets with no approximation (hidden truth).
    policies = {"unit_cosine": P.cosine, "pivoted_dot": P.pivoted(SLOPE, pivot)}
    kth = P.scan_kth(q_records, pool, complete, policies, K)
    exact_policy = P.scan_top(q_records, pool, complete, policies, kth, K)

    store = out / "experiment5.lancedb"
    shutil.rmtree(store, ignore_errors=True)
    db = lancedb.connect(store)
    partitions, sub_vectors = round(len(pool) ** 0.5), DIMENSION // 16
    pq = lambda metric: IvfPq(
        distance_type=metric, num_partitions=partitions, num_sub_vectors=sub_vectors
    )
    configs = (
        ("unit_cosine", "unit", "cosine", pq("cosine")),
        ("unit_dot", "unit", "dot", pq("dot")),
        ("pivoted_dot", "pivoted", "dot", pq("dot")),
        (
            "unit_cosine_rq",
            "unit",
            "cosine",
            IvfRq(distance_type="cosine", num_partitions=partitions),
        ),
    )
    rows, settings, flat = [], {}, {}
    for name, enc, metric, config in configs:
        started = time.time()
        table = write_table(
            db, name, pool, coverage, lambda r, cov, e=enc: encoders[e](r), pa.float16()
        )
        written = time.time() - started
        started = time.time()
        table.create_index("vector", config=config, name="vector_idx")
        built = time.time() - started
        kind = type(config).__name__.replace("Ivf", "IVF_").upper()
        log(
            f"{name}: wrote {len(pool):,} rows in {written:.0f}s, built {kind} ({metric}) in {built:.0f}s"
        )
        settings[name] = {
            "index": kind,
            "metric": metric,
            "build_seconds": round(built),
            "stats": str(table.index_stats("vector_idx")),
        }
        kth_exact, where = exact_top(q_vectors, pool, encoders[enc], metric)
        truth = torch.stack(
            [
                P.relevance(P.compare(q_records.take([i]), complete.take(where[i])))[0]
                for i in range(len(q_records))
            ]
        )
        flat[name] = float((truth >= kth["relevance"][:, None] - P.TOLERANCE).float().mean())
        rows += sweep(
            table,
            name,
            q_vectors,
            q_records,
            pool,
            position,
            encoders[enc],
            metric,
            kth_exact[:, -1],
            kth["relevance"],
            complete,
            log,
        )
    summary = pl.DataFrame(rows, infer_schema_length=None)
    policy_of = {
        "unit_cosine": "unit_cosine",
        "unit_dot": "unit_cosine",
        "unit_cosine_rq": "unit_cosine",
        "pivoted_dot": "pivoted_dot",
    }
    summary = summary.with_columns(
        flat_precision_at_10=pl.col("index").replace_strict(flat),
        exact_policy_precision_at_10=pl.col("index").replace_strict(
            {n: float(exact_policy[p].hits.mean()) / K for n, p in policy_of.items()}
        ),
    )
    summary.write_csv(out / "experiment5_summary.csv", float_precision=5)
    normalizer_check(log).write_csv(out / "experiment5_normalizer.csv", float_precision=5)
    (out / "experiment5_config.json").write_text(
        json.dumps(
            {
                "dimension": DIMENSION,
                "seed": SEED,
                "rate": RATE,
                "pool_size": len(pool),
                "pivot": pivot,
                "slope": SLOPE,
                "storage": "float16",
                "queries": "400 complete evaluation-panel records",
                "ground_truth": "exact search over the same stored float16 vectors, same metric; recall ties count as hits",
                "num_partitions": partitions,
                "num_sub_vectors": sub_vectors,
                "index_defaults": "8-bit PQ codes, 1-bit RQ codes, 50 k-means iterations, sample rate 256 (LanceDB defaults)",
                "indexes": settings,
                "latency": "single-threaded Python loop, warm cache; context only",
                "lancedb_version": lancedb.__version__,
            },
            indent=2,
        )
        + "\n"
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=220, float_precision=3):
        log(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=person.OUT)
    args = parser.parse_args()
    torch.set_num_threads(os.cpu_count())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run(args.output_dir, lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
