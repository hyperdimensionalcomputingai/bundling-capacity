"""Experiment 2: the missing-value dial (Gower, cosine, mismatch, pivoted).

Every policy scores the same per-property similarities; they differ only in the
denominator, that is, in what a missing property costs (policy.py). Here they
compete in search: 400 complete query records against a pool in which every
record has fields missing completely at random.

Two notions of relevance:

  hidden truth  each pool record's *complete* version, scored with equal property
                weights. Precision@10 is the share of a policy's top 10 (masked
                data) that belongs in the top 10 it would return with nothing
                missing. MCAR masking is independent of truth, so relevant records
                have the same coverage mix as the pool: Singhal's diagnostic
                compares that mix with the mix a policy retrieves.
  own copy      the query person's own masked record. Recall@1 and Recall@10,
                next to the random reference k / pool size.

Singhal's pivoted normalizer is fitted per (rate, density) on the separate
200-query calibration panel, by precision@10 over a grid of slopes, then held
fixed for the evaluation panel. Slopes within 0.001 of the best count as ties,
resolved toward cosine (slope 1). Slope 1 is cosine and slope 0 ranks like
mismatch; slopes above 1 move toward treating missing properties as neutral.

Grid (design choices for this run): MCAR rates 0.1, 0.3, 0.5 and pool densities
100%, 10% and 1% of the 920,000 records. The factorial fixture contains every
near-variant of every record, which real populations do not; the sparser pools
test whether conclusions depend on that density. Query-panel records are always
kept in the pool so each query's own copy is present.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import polars as pl
import torch

import person
import policy as P

RATES = (0.1, 0.3, 0.5)
DENSITIES = (1.0, 0.1, 0.01)
SLOPES = tuple(
    round(0.1 * i, 1) for i in range(17)
)  # 0.0 .. 1.6; steeper slopes flood sparse records
K = 10
BOOTSTRAP = 2000
SAMPLE_SEED = 118  # the fixed pool subsamples
FIT_MARGIN = (
    0.001  # calibration precision differences smaller than this don't move the slope off cosine
)


def subsample(n: int, density: float, keep: torch.Tensor) -> torch.Tensor:
    """A fixed random subset of record indices at `density`, always including `keep`."""
    if density >= 1:
        return torch.arange(n)
    g = torch.Generator().manual_seed(SAMPLE_SEED + round(density * 1000))
    picked = torch.randperm(n, generator=g)[: round(density * n)]
    return torch.unique(torch.cat([picked, keep]))


def pivots(pivot: float) -> dict:
    return {f"pivoted_{s:.1f}": P.pivoted(s, pivot) for s in SLOPES}


def fit_slope(queries, pool, complete, pivot) -> tuple[float, pl.DataFrame]:
    """Precision@10 on the calibration panel for every slope; returns the best slope and the curve."""
    policies = pivots(pivot)
    kth = P.scan_kth(queries, pool, complete, policies, K)
    top = P.scan_top(queries, pool, complete, policies, kth, K)
    curve = pl.DataFrame(
        {
            "slope": list(SLOPES),
            "precision_at_10": [float(top[f"pivoted_{s:.1f}"].hits.mean()) / K for s in SLOPES],
        }
    )
    # Prefer cosine: among slopes within FIT_MARGIN of the best, take the one closest to 1.
    top = curve["precision_at_10"].max()
    near = curve.filter(pl.col("precision_at_10") >= top - FIT_MARGIN)
    best = near.sort((pl.col("slope") - 1).abs())["slope"][0]
    return float(best), curve


def bootstrap(values: torch.Tensor, g: torch.Generator) -> tuple[float, float]:
    picks = torch.randint(0, len(values), (BOOTSTRAP, len(values)), generator=g)
    low, high = torch.quantile(values[picks].mean(1), torch.tensor([0.025, 0.975]))
    return float(low), float(high)


def run(out: Path, log=print):
    signatures = person.load_fixture()
    complete_all = person.complete_records(signatures)
    calibration, evaluation = person.query_panel("calibration"), person.query_panel("evaluation")
    panels = torch.cat([calibration, evaluation])
    rows, curves, summary = [], [], []
    g = torch.Generator().manual_seed(83)
    for rate, density in itertools.product(RATES, DENSITIES):
        keep = subsample(len(complete_all), density, panels)
        pool = person.mcar_records(signatures, rate).take(keep)
        complete = complete_all.take(keep)
        pivot = P.mean_length(pool)
        slope, curve = fit_slope(complete_all.take(calibration), pool, complete, pivot)
        curves.append(
            curve.with_columns(
                rate=pl.lit(rate), density=pl.lit(density), panel=pl.lit("calibration")
            )
        )

        # Evaluation: the three fixed policies, the fitted pivot, and the whole slope curve for the figure.
        queries = complete_all.take(evaluation)
        own = person.mcar_records(signatures, rate).take(evaluation)
        policies = {
            "mismatch": P.mismatch,
            "cosine": P.cosine,
            "gower": P.gower,
            "pivoted_fitted": P.pivoted(slope, pivot),
            **pivots(pivot),
        }
        kth = P.scan_kth(queries, pool, complete, policies, K)
        top = P.scan_top(queries, pool, complete, policies, kth, K)
        ranks = P.rank_relevant(queries, pool, own, policies, K, exclude=evaluation)
        curves.append(
            pl.DataFrame(
                {
                    "slope": list(SLOPES),
                    "precision_at_10": [
                        float(top[f"pivoted_{s:.1f}"].hits.mean()) / K for s in SLOPES
                    ],
                }
            ).with_columns(rate=pl.lit(rate), density=pl.lit(density), panel=pl.lit("evaluation"))
        )
        n = len(pool)
        for name in ("mismatch", "cosine", "gower", "pivoted_fitted", "relevance"):
            t = top[name]
            precision = t.hits / K
            frame = {
                "rate": rate,
                "density": density,
                "pool_size": n,
                "policy": name,
                "query_record": evaluation.tolist(),
                "precision_at_10": precision.tolist(),
                **{
                    f"top10_coverage_{c}": (t.coverage[:, c] / K).tolist()
                    for c in range(P.COVERAGE_BINS)
                },
            }
            if name != "relevance":
                r = ranks[name]
                frame.update(
                    own_recall_at_1=r.recall(1).tolist(),
                    own_recall_at_10=r.recall(K).tolist(),
                    own_reciprocal_rank=r.reciprocal_rank().tolist(),
                )
            rows.append(pl.DataFrame(frame))
            if name == "relevance":
                continue
            r = ranks[name]
            p_low, p_high = bootstrap(precision, g)
            r_low, r_high = bootstrap(r.recall(K), g)
            summary.append(
                {
                    "rate": rate,
                    "density": density,
                    "pool_size": n,
                    "policy": name,
                    "slope": slope
                    if name == "pivoted_fitted"
                    else {"cosine": 1.0, "mismatch": 0.0}.get(name),
                    "precision_at_10": float(precision.mean()),
                    "precision_at_10_ci_low": p_low,
                    "precision_at_10_ci_high": p_high,
                    "own_recall_at_1": float(r.recall(1).mean()),
                    "own_recall_at_10": float(r.recall(K).mean()),
                    "own_recall_at_10_ci_low": r_low,
                    "own_recall_at_10_ci_high": r_high,
                    "own_mrr": float(r.reciprocal_rank().mean()),
                    "random_recall_at_10": K / n,
                    "top10_share_one_property": float(t.coverage[:, 1].mean()) / K,
                    "top10_share_complete": float(t.coverage[:, 4].mean()) / K,
                    "relevant_share_one_property": float(top["relevance"].coverage[:, 1].mean())
                    / K,
                    "relevant_share_complete": float(top["relevance"].coverage[:, 4].mean()) / K,
                }
            )
        log(f"rate={rate} density={density:.0%} pool={n:,} fitted slope={slope}")
    pl.concat(rows, how="diagonal").write_parquet(out / "experiment2_per_query.parquet")
    pl.concat(curves).write_csv(out / "experiment2_slope_curve.csv", float_precision=5)
    pl.DataFrame(summary).write_csv(out / "experiment2_summary.csv", float_precision=5)
    (out / "experiment2_config.json").write_text(
        json.dumps(
            {
                "rates": RATES,
                "densities": DENSITIES,
                "slopes": SLOPES,
                "k": K,
                "sample_seed": SAMPLE_SEED,
                "queries": "complete records of the scale study's evaluation panel (400); slope fitted on its calibration panel (200)",
                "relevance": "hidden truth: equal-weight property similarity of the complete records; top 10 with ties",
                "own_copy": "the query person's masked record; ties placed uniformly in expectation",
                "pivot": "mean sqrt(known properties) over the pool (Singhal's average old normalizer)",
                "slope_fit": f"max calibration precision@10; within {FIT_MARGIN} of the best, the slope closest to 1",
                "scores": "exact policy values (orthogonal atoms); the MAP implementation is Experiment 4",
                "bootstrap": f"{BOOTSTRAP} resamples over evaluation queries",
            },
            indent=2,
        )
        + "\n"
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
