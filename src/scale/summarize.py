"""Tables for Experiment 1 from the streamed histograms and per-query results.

Experiment 2 (ivfpq.py) writes its own summary.

Uncertainty: the sampling unit is the evaluation query. Seeds reuse the same
queries and pairs sharing a query are dependent, so pairs are never treated as
independent trials. Query-level rates get a bootstrap over queries (seed-averaged
per query); events never observed get a one-sided Clopper-Pearson 95% upper bound.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
import torch
from scipy.stats import beta as beta_dist

import experiment as ex
from run import OUT

BOOTSTRAP = 2000
CELL = ["dimension", "seed", "candidates"]


def quantile(counts: torch.Tensor, lo: float, q: float, *, upper: bool = True) -> float:
    """Bin edge holding the q-quantile: upper edge for upper tails, lower edge for lower tails."""
    total = int(counts.sum())
    if total == 0:
        return float("nan")
    index = int(torch.searchsorted(torch.cumsum(counts, 0), torch.tensor(q * total)))
    return lo + (index + int(upper)) * ex.BIN


def clopper_pearson_upper(events: int, n: int, level: float = 0.95) -> float:
    return 1.0 if events >= n else float(beta_dist.ppf(level, events + 1, n - events))


def histogram_rows(hist: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for (d, s), part in hist.group_by("dimension", "seed"):
        for n in ex.PREFIXES:
            dense = {}
            for (metric,), m in part.filter(pl.col("segment_end") <= n).group_by("metric"):
                lo, hi = ex.METRICS[metric]
                counts = torch.zeros(round((hi - lo) / ex.BIN), dtype=torch.long)
                summed = m.group_by("bin").agg(pl.col("count").sum())
                counts[summed["bin"].to_torch().long()] = summed["count"].to_torch().long()
                dense[metric] = counts
            err = dense["abs_numerical_error"]
            row = {
                "dimension": d,
                "seed": s,
                "candidates": n,
                "abs_map_error_median": quantile(err, 0, 0.5),
                "abs_map_error_p99": quantile(err, 0, 0.99),
                "related_score_p01": quantile(dense["related_score"], -1, 0.01, upper=False),
            }
            unrelated = dense.get("unrelated_score")
            if unrelated is not None and int(unrelated.sum()):
                half = len(unrelated) // 2
                row.update(
                    unrelated_score_p999=quantile(unrelated, -1, 0.999),
                    unrelated_abs_score_p99=quantile(
                        unrelated[half:] + unrelated[:half].flip(0), 0, 0.99
                    ),
                )
            rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def query_level(per_query: pl.DataFrame) -> pl.DataFrame:
    """Per (cell, query) outcomes."""
    return per_query.with_columns(
        (pl.col("false_pos_calibrated") > 0).alias("queries_with_false_match"),
        (pl.col("false_pos_abs_calibrated") > 0).alias("queries_with_abs_violation"),
        (pl.col("false_pos_reference") > 0).alias("queries_with_false_match_reference_t"),
        (pl.col("bundle_top1_exact") < pl.col("exact_best") - ex.TOLERANCE).alias("top1_changed"),
    )


def cell_summary(q: pl.DataFrame) -> pl.DataFrame:
    return (
        q.group_by(CELL)
        .agg(
            pl.len().alias("queries"),
            pl.col("pairs_unrelated_0").sum().alias("unrelated_pairs"),
            pl.col("pairs_related_[2/3,1]").sum().alias("related_pairs"),
            pl.col("max_unrelated")
            .filter(pl.col("pairs_unrelated_0") > 0)
            .median()
            .alias("median_query_max_unrelated"),
            pl.col("max_unrelated")
            .filter(pl.col("pairs_unrelated_0") > 0)
            .max()
            .alias("max_unrelated"),
            pl.col("false_pos_calibrated").sum().alias("false_matches"),
            pl.col("false_neg_calibrated").sum().alias("false_negatives"),
            # A query can only have a false match if it has unrelated candidates at all.
            (pl.col("pairs_unrelated_0") > 0).sum().alias("queries_with_unrelated_candidates"),
            pl.col(
                "queries_with_false_match",
                "queries_with_abs_violation",
                "queries_with_false_match_reference_t",
            )
            .filter(pl.col("pairs_unrelated_0") > 0)
            .mean(),
            pl.col("precision_at_10").mean(),
            pl.col("top1_changed").mean(),
            pl.col("head_discordant").mean(),
        )
        .with_columns(
            (1 - pl.col("false_negatives") / pl.col("related_pairs")).alias("evaluation_recall")
        )
    )


def bootstrap(q: pl.DataFrame, keys: list[str], metrics: list[str]) -> pl.DataFrame:
    """Seed-averaged per-query outcomes, resampled over queries.

    False-match rates use only queries that have unrelated candidates; the rest use every query.
    """
    false_match = [m for m in metrics if m.startswith("queries_with")]
    if false_match and len(false_match) < len(metrics):
        return (
            bootstrap(q.filter(pl.col("pairs_unrelated_0") > 0), keys, false_match)
            .rename({"queries": "queries_with_unrelated_candidates"})
            .join(bootstrap(q, keys, [m for m in metrics if m not in false_match]), on=keys)
        )
    if false_match:
        q = q.filter(pl.col("pairs_unrelated_0") > 0)
    per_query = q.group_by(keys + ["query_rank"]).agg(
        [pl.col(m).cast(pl.Float32).mean() for m in metrics]
    )
    g = torch.Generator().manual_seed(83)
    rows = []
    for group, part in per_query.group_by(keys, maintain_order=True):
        row, n = dict(zip(keys, group)), part.height
        picks = torch.randint(0, n, (BOOTSTRAP, n), generator=g)
        row["queries"] = n
        for m in metrics:
            values = part[m].to_torch()
            low, high = torch.quantile(values[picks].mean(1), torch.tensor([0.025, 0.975]))
            row.update(
                {m: float(values.mean()), f"{m}_ci_low": float(low), f"{m}_ci_high": float(high)}
            )
            if m.startswith("queries_with"):
                events = int((values > 0).sum())
                row[f"{m}_observed"] = events
                row[f"{m}_upper95"] = clopper_pearson_upper(events, n)
        rows.append(row)
    return pl.DataFrame(rows).sort(keys)


def seed_spread(cells: pl.DataFrame, keys: list[str], metrics: list[str]) -> pl.DataFrame:
    return (
        cells.group_by(keys)
        .agg(
            [pl.col(m).mean() for m in metrics]
            + [pl.col(m).min().alias(f"{m}_seed_min") for m in metrics]
            + [pl.col(m).max().alias(f"{m}_seed_max") for m in metrics]
        )
        .sort(keys)
    )


def run(out: Path):
    thresholds = pl.DataFrame(
        json.loads((out / "thresholds.json").read_text())["thresholds"]
    ).select("dimension", "seed", pl.col("threshold").alias("calibrated_threshold"))
    q = query_level(pl.read_parquet(out / "per_query.parquet"))
    cells = (
        histogram_rows(pl.read_parquet(out / "histograms.parquet"))
        .join(cell_summary(q), on=CELL)
        .join(thresholds, on=["dimension", "seed"])
        .sort(CELL)
    )
    cells.write_csv(out / "cell_summary.csv", float_precision=6)

    # Experiment 1: every D and N.
    scale_metrics = [
        "abs_map_error_median",
        "abs_map_error_p99",
        "median_query_max_unrelated",
        "max_unrelated",
        "unrelated_score_p999",
        "related_score_p01",
        "calibrated_threshold",
        "evaluation_recall",
        "queries_with_false_match",
        "queries_with_abs_violation",
        "queries_with_false_match_reference_t",
        "precision_at_10",
        "top1_changed",
        "head_discordant",
    ]
    seed_spread(cells, ["dimension", "candidates"], scale_metrics).write_csv(
        out / "experiment1_scale.csv", float_precision=6
    )
    uncertainty = [
        "queries_with_false_match",
        "queries_with_false_match_reference_t",
        "top1_changed",
    ]
    bootstrap(
        q.filter(pl.col("candidates") == ex.PREFIXES[-1]),
        ["dimension"],
        uncertainty,
    ).write_csv(out / "experiment1_uncertainty.csv", float_precision=6)

    print(f"Wrote summaries for {cells.height:,} cells to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    run(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
