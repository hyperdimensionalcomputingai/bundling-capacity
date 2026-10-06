"""Experiment 4: building it with MAP vectors in LanceDB.

Experiments 2 and 3 scored policies exactly, as if atoms were orthogonal. Here the
same policies are built from real MAP vectors (D = 2,048, seed 11), stored in
LanceDB, and searched with a single dot product:

  cosine_f32      L2 property normalization, each record scaled to unit length, float32
  cosine_f16      the same vectors stored as float16
  mismatch_f16    L2 property normalization, unscaled: dot ranks like "missing is a mismatch"
  pivoted_f16     each record divided by Singhal's pivoted length (1 - s) pivot + s sqrt(p_c),
                  from a stored coverage count; s = 1.2, the slope Experiment 2 fitted
                  for the sparse pool at 30% missingness
  sign_f16        majority-sign property normalization (integer-valued, exact in float16),
                  searched by cosine
  gower_rerank    Gower cannot be one dot product: its denominator depends on the pair.
                  It is applied as a reranker over cosine_f16's top 100, from stored codes.
  weighted_f16    cosine_f16's index searched with a query whose property terms carry
                  the label-free Fellegi-Sunter agreement weights (Experiment 3): one
                  stored index, any query-side weighting

Each search is compared with the exact policy it implements (the share of its top
10 that belongs in the exact top 10, ties included), with hidden-truth relevance
as in Experiment 2, and with the float32 scores of the same vectors. Storage size
and latency are recorded for context.

Pool: the first 100,000 fixture records (a balanced prefix) at 30% MCAR
missingness, about the density of Experiment 2's 10% pool; queries: the 400
complete evaluation-panel records. Searches are flat (no index), so any
difference from the in-memory float32 ranking comes from storage precision.
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

import person
import policy as P
from experiment3_linkage import assumed_m, level_frequencies

DIMENSION = 2048
SEED = 11
RATE = 0.3
POOL = 100_000
SLOPE = 1.2
K = 10
RERANK = 100
CHUNK = 8192
SIGN_SEEDS = tuple(range(1, 21))  # the majority-sign cost table


def schema(dtype) -> pa.Schema:
    fields = [
        pa.field("record_index", pa.int32()),
        pa.field("coverage", pa.int8()),
        *[pa.field(f, pa.int8()) for f in ("age_band", "job_category", "home_region")],
        *[pa.field(f"interest_{k}", pa.int8()) for k in (1, 2, 3)],
        pa.field("vector", pa.list_(dtype, DIMENSION)),
    ]
    return pa.schema(fields)


def to_column(vectors: torch.Tensor, dtype) -> pa.FixedSizeListArray:
    """The storage boundary: float32 vectors cast to the table's type."""
    torch_type = torch.float16 if dtype == pa.float16() else torch.float32
    flat = vectors.to(torch_type).contiguous().reshape(-1).numpy()
    return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=dtype), DIMENSION)


def write_table(db, name: str, records: person.Records, coverage, encode, dtype):
    s = schema(dtype)

    def batches():
        for start in range(0, len(records), CHUNK):
            r = records.take(slice(start, start + CHUNK))
            yield pa.RecordBatch.from_arrays(
                [
                    pa.array(r.record_index.numpy(), pa.int32()),
                    pa.array(coverage[start : start + CHUNK].numpy(), pa.int8()),
                    *[pa.array(r.value[:, f].numpy(), pa.int8()) for f in range(3)],
                    *[pa.array(r.interests[:, k].numpy(), pa.int8()) for k in range(3)],
                    to_column(encode(r, coverage[start : start + CHUNK]), dtype),
                ],
                schema=s,
            )

    table = db.create_table(name, pa.RecordBatchReader.from_batches(s, batches()), mode="overwrite")
    if table.schema != s:
        raise AssertionError(f"LanceDB changed the schema of {name}")
    return table


def search(table, vectors: torch.Tensor, distance: str, k: int) -> tuple[list, list, list]:
    """Flat search for every query: returned record indices, stored codes, scores, latency (ms)."""
    found, scores, latency = [], [], []
    for v in vectors:
        t0 = time.perf_counter()
        rows = (
            table.search(v.tolist(), vector_column_name="vector")
            .distance_type(distance)
            .limit(k)
            .select(["record_index", "_distance"])
            .to_arrow()
        )
        latency.append((time.perf_counter() - t0) * 1000)
        found.append(rows["record_index"].to_pylist())
        scores.append([1 - d for d in rows["_distance"].to_pylist()])
    return found, scores, latency


def exact_scores(queries, pool, position, score, returned) -> torch.Tensor:
    """Exact policy scores of each query's returned records, (q, k)."""
    out = torch.full((len(queries), K), torch.nan)
    for i, rows in enumerate(returned):
        idx = torch.tensor([position[r] for r in rows[:K]], dtype=torch.long)
        out[i, : len(idx)] = score(P.compare(queries.take([i]), pool.take(idx)))[0]
    return out


def agreement(returned_exact: torch.Tensor, kth: torch.Tensor) -> torch.Tensor:
    """Share of the returned top 10 whose exact score reaches the exact 10th best (ties count)."""
    return (returned_exact >= kth[:, None] - P.TOLERANCE).float().mean(1)


def sign_cost() -> pl.DataFrame:
    """Interests' own similarity, query knowing 3 against a candidate knowing k: L2 vs sign."""
    rows = []
    complete = person.complete_records(person.load_fixture())
    for seed in SIGN_SEEDS:
        book = person.codebook(DIMENSION, seed)
        g = torch.Generator().manual_seed(seed)
        base = complete.take(torch.randint(len(complete), (200,), generator=g))
        q_l2 = person.property_terms(book, base, normalized=True)[3]
        for k in (1, 2, 3):
            drop = torch.zeros(len(base), 3, dtype=torch.bool)
            drop[:, k:] = True
            c = person.Records(
                base.record_index,
                base.state,
                base.value,
                torch.where(drop, -1, base.interests).to(torch.int8),
                base.spelling,
            )
            c_l2 = person.property_terms(book, c, normalized=True)[3]
            only = lambda r: r.without(torch.tensor([[True, True, True, False]]).expand(len(r), 4))
            q_sign, c_sign = person.encode_sign(book, only(base)), person.encode_sign(book, only(c))
            rows.append(
                {
                    "seed": seed,
                    "candidate_interests": k,
                    "l2": float(person.cosine(q_l2, c_l2, paired=True).mean()),
                    "sign": float(person.cosine(q_sign, c_sign, paired=True).mean()),
                }
            )
    return (
        pl.DataFrame(rows)
        .group_by("candidate_interests")
        .agg(
            pl.col("l2").mean().alias("l2_measured"),
            pl.col("sign").mean().alias("sign_measured"),
            pl.col("sign").std().alias("sign_sd"),
        )
        .with_columns(l2_expected=(pl.col("candidate_interests") / 3).sqrt())
        .sort("candidate_interests")
    )


def run(out: Path, log=print):
    signatures = person.load_fixture()
    complete_all = person.complete_records(signatures)
    pool = person.mcar_records(signatures, RATE).take(slice(0, POOL))
    complete = complete_all.take(slice(0, POOL))
    keep = torch.nonzero(
        P.known(pool)[0].sum(1) > 0
    ).flatten()  # records with nothing known can't be found
    pool, complete = pool.take(keep), complete.take(keep)
    position = {int(r): i for i, r in enumerate(pool.record_index.tolist())}
    coverage = P.known(pool)[0].sum(1)
    pivot = P.mean_length(pool)
    queries = complete_all.take(person.query_panel("evaluation"))
    book = person.codebook(DIMENSION, SEED)

    # Query-side weights: label-free Fellegi-Sunter agreement weights from random pairs.
    g = torch.Generator().manual_seed(7)
    u = level_frequencies(
        P.levels(P.compare(queries, pool.take(torch.randint(len(pool), (2000,), generator=g))))
    )
    m = assumed_m(u)
    weights = torch.tensor([float((m[f][0] / u[f][0]).log()) for f in range(4)])

    l2 = lambda r: person.encode_normalized(book, r)
    unit = lambda v: v / v.norm(dim=1, keepdim=True).clamp(min=1e-9)
    length = lambda cov: (1 - SLOPE) * pivot + SLOPE * cov.float().sqrt()
    tables = {
        "cosine_f32": (lambda r, cov: unit(l2(r)), pa.float32(), "dot"),
        "cosine_f16": (lambda r, cov: unit(l2(r)), pa.float16(), "dot"),
        "mismatch_f16": (lambda r, cov: l2(r), pa.float16(), "dot"),
        "pivoted_f16": (lambda r, cov: l2(r) / length(cov)[:, None], pa.float16(), "dot"),
        "sign_f16": (lambda r, cov: person.encode_sign(book, r), pa.float16(), "cosine"),
    }
    store = out / "experiment4.lancedb"
    shutil.rmtree(store, ignore_errors=True)
    db = lancedb.connect(store)
    q_l2 = unit(l2(queries))
    q_terms = person.property_terms(book, queries, normalized=True)
    q_weighted = unit((q_terms * weights[:, None, None]).sum(0))
    query_vectors = {
        "cosine_f32": q_l2,
        "cosine_f16": q_l2,
        "mismatch_f16": q_l2,
        "pivoted_f16": q_l2,
        "sign_f16": person.encode_sign(book, queries),
        "weighted_f16": q_weighted,
    }
    exact = {
        "cosine_f32": P.cosine,
        "cosine_f16": P.cosine,
        "mismatch_f16": P.mismatch,
        "pivoted_f16": P.pivoted(SLOPE, pivot),
        "sign_f16": P.cosine,  # majority sign approximates the L2 cosine policy
        "weighted_f16": P.weighted_cosine(weights),
        "gower_rerank": P.gower,
    }
    kth = P.scan_kth(queries, pool, complete, exact, K)
    exact_top = P.scan_top(
        queries, pool, complete, exact, kth, K
    )  # each exact policy's own precision
    results, storage = {}, {}
    for name, (encode, dtype, distance) in tables.items():
        started = time.time()
        table = write_table(db, name, pool, coverage, encode, dtype)
        storage[name] = sum(
            f.stat().st_size for f in (store / f"{name}.lance").rglob("*") if f.is_file()
        )
        results[name] = search(table, query_vectors[name], distance, K)
        log(
            f"{name}: wrote and searched in {time.time() - started:.0f}s, {storage[name] / 1e6:.0f} MB"
        )
    cosine16 = db.open_table("cosine_f16")
    results["weighted_f16"] = search(cosine16, q_weighted, "dot", K)
    found, _, latency = search(cosine16, q_l2, "dot", RERANK)
    reranked = []
    t0 = time.perf_counter()
    for i, rows in enumerate(found):
        idx = torch.tensor([position[r] for r in rows], dtype=torch.long)
        s = P.gower(P.compare(queries.take([i]), pool.take(idx)))[0]
        order = s.argsort(descending=True, stable=True)[:K]
        reranked.append([rows[j] for j in order.tolist()])
    rerank_ms = (time.perf_counter() - t0) * 1000 / len(found)
    results["gower_rerank"] = (reranked, None, [x + rerank_ms for x in latency])

    # float32 reference scores of the same vectors, for the float16 error.
    rows = []
    for name, (returned, scores, lat) in results.items():
        returned_exact = exact_scores(queries, pool, position, exact[name], returned)
        returned_truth = exact_scores(queries, complete, position, P.relevance, returned)
        row = {
            "implementation": name,
            "exact_policy": {"sign_f16": "cosine"}.get(name, name.split("_")[0]),
            "agreement_with_exact_top10": float(agreement(returned_exact, kth[name]).mean()),
            "precision_at_10": float(agreement(returned_truth, kth["relevance"]).mean()),
            "exact_policy_precision_at_10": float(exact_top[name].hits.mean()) / K,
            "median_latency_ms": float(torch.tensor(lat).median()),
            "storage_mb": storage.get(name, storage["cosine_f16"])
            / 1e6,  # weighted and rerank reuse cosine_f16
        }
        rows.append(row)
    summary = pl.DataFrame(rows)

    # float16 against float32: same vectors, same queries.
    f32, f16 = results["cosine_f32"], results["cosine_f16"]
    overlap = [len(set(a) & set(b)) / K for a, b in zip(f32[0], f16[0])]
    score_error = [abs(x - y) for a, b in zip(f32[1], f16[1]) for x, y in zip(a, b)]
    precision = {
        "top10_overlap_f16_vs_f32": sum(overlap) / len(overlap),
        "queries_with_identical_top10": sum(o == 1 for o in overlap) / len(overlap),
        "max_abs_score_difference_rank_aligned": max(score_error),
    }
    summary.write_csv(out / "experiment4_summary.csv", float_precision=5)
    sign = sign_cost()
    sign.write_csv(out / "experiment4_sign_cost.csv", float_precision=5)
    (out / "experiment4_config.json").write_text(
        json.dumps(
            {
                "dimension": DIMENSION,
                "seed": SEED,
                "rate": RATE,
                "pool": f"first {POOL:,} fixture records, minus {POOL - len(pool)} with nothing known",
                "pool_size": len(pool),
                "pivot": pivot,
                "slope": SLOPE,
                "k": K,
                "rerank_depth": RERANK,
                "query_weights": dict(zip(("age", "job", "region", "interests"), weights.tolist())),
                "search": "LanceDB flat search (no index); dot distance is 1 - dot",
                "float16_vs_float32": precision,
                "lancedb_version": lancedb.__version__,
            },
            indent=2,
        )
        + "\n"
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=3):
        log(summary)
        log(sign)
        log(precision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=person.OUT)
    args = parser.parse_args()
    torch.set_num_threads(os.cpu_count())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run(args.output_dir, lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
