"""Report figures: Experiment 1 (chance similarity at scale) and Experiment 2 (IVF_PQ)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import FixedLocator, NullLocator, PercentFormatter

import experiment as ex
from run import OUT

# One-hue ordinal ramp for dimension (validated: monotone lightness, >= 0.06 steps, light end >= 2:1).
RAMP = ("#7fb6f7", "#6094d4", "#4273b1", "#255390", "#07356f")
DIM_COLORS = dict(zip(ex.DIMENSIONS, RAMP))
# Categorical slots 1-3 (validated; aqua is under 3:1 contrast, so every line is labelled directly).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
N_LABELS = [f"{n // 1000}k" if n >= 1000 else str(n) for n in ex.PREFIXES]


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": INK,
            "axes.facecolor": SURFACE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "figure.facecolor": "white",
            "grid.color": GRID,
            "grid.linewidth": 1,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "lines.linewidth": 2,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )


def save(fig, path: Path):
    for ext in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=190, facecolor="white")
    svg = path.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def frame_axes(ax):
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)


def log_x(ax, values, labels):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(values))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(labels)


def header(fig, title: str, subtitle: str):
    fig.text(0.06, 0.975, title, fontsize=16, weight="bold", va="top")
    fig.text(0.06, 0.915, subtitle, fontsize=10.5, color=MUTED, va="top")


def footnote(fig, text: str):
    fig.text(0.06, 0.015, text, color=MUTED, fontsize=8.5)


def experiment1(scale: pl.DataFrame, out: Path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.6, 5.4))
    fig.subplots_adjust(left=0.07, right=0.9, top=0.78, bottom=0.17, wspace=0.28)
    header(
        fig,
        "Chance similarity stays small at a million records",
        "Left: how far a MAP cosine strays from the similarity the encoder intends. "
        "Right: the highest score any unrelated record reaches",
    )
    full = scale.filter(pl.col("candidates") == ex.PREFIXES[-1]).sort("dimension")
    d = full["dimension"].to_numpy()
    for column, label, color in (
        ("abs_map_error_median", "median pair", RAMP[1]),
        ("abs_map_error_p99", "99th percentile pair", RAMP[4]),
    ):
        left.fill_between(
            d,
            full[f"{column}_seed_min"],
            full[f"{column}_seed_max"],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        left.plot(d, full[column], color=color, marker="o", markersize=6)
        left.text(d[-1] * 1.1, full[column][-1], label, va="center", fontsize=9.5)
    left.plot(d, 1 / np.sqrt(d), color=MUTED, linestyle=(0, (2, 3)), linewidth=1.3)
    left.text(d[0] * 1.05, 1 / np.sqrt(d[0]) * 1.06, "1/√D", color=MUTED, fontsize=9.5)
    log_x(left, ex.DIMENSIONS, [f"{x:,}" for x in ex.DIMENSIONS])
    left.set_xlim(420, ex.DIMENSIONS[-1] * 2.6)
    left.set_ylim(0, None)
    left.set_xlabel("Dimensions (D)", labelpad=8)
    left.set_title("MAP error |S_MAP − S_encoder|", loc="left", fontsize=11, pad=10)
    frame_axes(left)

    for dimension, color in DIM_COLORS.items():
        rows = scale.filter(pl.col("dimension") == dimension).sort("candidates")
        n = rows["candidates"].to_numpy()
        right.fill_between(
            n,
            rows["median_query_max_unrelated_seed_min"],
            rows["median_query_max_unrelated_seed_max"],
            color=color,
            alpha=0.15,
            linewidth=0,
        )
        right.plot(
            n,
            rows["median_query_max_unrelated"],
            color=color,
            marker="o",
            markersize=5,
            label=f"D = {dimension:,}",
        )
    right.axhline(ex.REFERENCE_T, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    right.text(
        ex.PREFIXES[0] * 1.1,
        ex.REFERENCE_T + 0.006,
        "earlier chat's T = 0.2157",
        color=MUTED,
        fontsize=9,
    )
    log_x(right, ex.PREFIXES, N_LABELS)
    right.set_ylim(0, None)
    right.set_xlabel("Candidate records searched (N)", labelpad=8)
    right.set_title("Highest unrelated score, typical query", loc="left", fontsize=11, pad=10)
    frame_axes(right)
    handles, labels = right.get_legend_handles_labels()
    fig.legend(
        handles[::-1], labels[::-1], frameon=False, loc="center left", bbox_to_anchor=(0.905, 0.48)
    )
    footnote(
        fig,
        f"Complete records · 400 evaluation queries · {len(ex.SEEDS)} seeds (shading: seed range) · right: "
        "median over endpoint-age queries, the only ones with unrelated candidates · the calibrated match "
        f"threshold is {full['calibrated_threshold'].min():.2f}–{full['calibrated_threshold'].max():.2f}",
    )
    save(fig, out / "experiment1_scale")


def experiment2(summary: pl.DataFrame, index: dict, out: Path):
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    fig.subplots_adjust(left=0.1, right=0.8, top=0.78, bottom=0.17)
    header(
        fig,
        "What an IVF_PQ index gives up against exact search",
        "Share of the exact top 10 that LanceDB IVF_PQ returns, by partitions probed",
    )
    for refine, color, label in (
        (None, BLUE, "no refine"),
        (10, ORANGE, "refine_factor 10"),
        (50, AQUA, "refine_factor 50"),
    ):
        rows = (
            summary.filter(pl.col("refine_factor").is_null())
            if refine is None
            else summary.filter(pl.col("refine_factor") == refine)
        ).sort("nprobes")
        ax.plot(rows["nprobes"], rows["recall_at_10"], color=color, marker="o", markersize=6)
        ax.text(
            rows["nprobes"][-1] * 1.12,
            rows["recall_at_10"][-1] - (0.006 if refine == 50 else 0),
            label,
            va="center",
            fontsize=9.5,
        )
    ax.axhline(1, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(10, 1.005, "exact search", color=MUTED, fontsize=9, va="bottom")
    log_x(ax, [10, 20, 50, 100], ["10", "20", "50", "100"])
    ax.set_xlim(8, 190)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel(f"nprobes (of {index['num_partitions']} partitions)", labelpad=8)
    ax.set_ylabel("Recall@10", labelpad=8)
    frame_axes(ax)
    footnote(
        fig,
        f"D = 2,048, seed 11, {index['rows']:,} complete records · 400 evaluation queries · ties at the 10th "
        f"exact score count as hits · num_sub_vectors = {index['num_sub_vectors']} · float16 storage is exact",
    )
    save(fig, out / "experiment2_ivfpq")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    out = parser.parse_args().output_dir
    style()
    experiment1(pl.read_csv(out / "experiment1_scale.csv"), out)
    if (out / "ivfpq_summary.csv").exists():
        experiment2(
            pl.read_csv(out / "ivfpq_summary.csv"),
            json.loads((out / "ivfpq_index.json").read_text()),
            out,
        )
    print(f"Saved figures in {out}")


if __name__ == "__main__":
    main()
