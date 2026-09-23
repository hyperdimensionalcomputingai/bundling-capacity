"""Compact research figures from the saved experiment summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import polars as pl


ROOT = Path(__file__).resolve().parents[2]
COLORS = {512: "#B65B43", 2048: "#C58A32", 4096: "#3C8590", 8192: "#325578"}
INK = "#20343F"
MUTED = "#586C74"
WIDTHS = (5, 10, 20, 30, 40, 50)


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
        "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
        "axes.facecolor": "#F3F6F5", "figure.facecolor": "white",
        "grid.color": "white", "grid.linewidth": 1.5,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "svg.fonttype": "none", "savefig.bbox": "tight",
    })


def save(fig, path: Path):
    for ext in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=190, facecolor="white")
    svg = path.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def aggregate(frame: pl.DataFrame, metric: str, keys: list[str]):
    return (frame.group_by(keys)
            .agg(pl.col(metric).mean().alias("mean"),
                 pl.col(metric).min().alias("low"),
                 pl.col(metric).max().alias("high"))
            .sort(keys))


def curve(ax, data: pl.DataFrame, dimension: int, *, label: str | None = None):
    rows = data.filter(pl.col("dimension") == dimension).sort("bundle_width")
    x = rows["bundle_width"].to_list()
    mean, low, high = (rows[name].to_list() for name in ("mean", "low", "high"))
    if len(x) != 6:
        raise ValueError(f"Incomplete bundle-width sweep at D={dimension}")
    ax.fill_between(x, low, high, color=COLORS[dimension], alpha=.15, linewidth=0)
    ax.plot(x, mean, color=COLORS[dimension], linewidth=2.3, marker="o",
            markersize=4.5, label=label or f"{dimension:,}")


def axes_style(ax, *, percent: bool = False):
    ax.set_xticks(WIDTHS)
    ax.set_xlim(3, 52)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1))


def recovery(frame: pl.DataFrame, output: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.7))
    fig.subplots_adjust(left=.085, right=.99, top=.70, bottom=.19, wspace=.2)
    fig.text(.085, .96, "What survives as the bundle fills?", fontsize=19, weight="bold", va="top")
    metrics = [
        ("exact_accuracy", "Exact answer recovered", True),
        ("mean_absolute_error", "Average rating error", False),
    ]
    for ax, (metric, title, percent) in zip(axes, metrics):
        summary = aggregate(frame, metric, ["dimension", "bundle_width"])
        for dimension in COLORS:
            curve(ax, summary, dimension)
        axes_style(ax, percent=percent)
        ax.set_title(title, loc="left", weight="bold", pad=10)
        ax.set_xlabel("Facts in the bundle", labelpad=9)
        if percent:
            ax.set_ylim(.75, 1.012)
            ax.set_yticks([.75, .8, .85, .9, .95, 1])
        else:
            ax.set_ylim(bottom=0)
            ax.set_ylabel("Scale points (1–5)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Dimensions", ncol=4, frameon=False,
               loc="upper left", bbox_to_anchor=(.08, .90))
    fig.text(.99, .03, "Vertical accuracy scale starts at 75% · lines: five-seed means · shading: seed range",
             ha="right", color=MUTED, fontsize=9)
    save(fig, output / "recovery")


def geometry(frame: pl.DataFrame, output: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), sharex=True)
    fig.subplots_adjust(left=.085, right=.99, top=.76, bottom=.13, hspace=.34, wspace=.20)
    fig.text(.085, .97, "Do whole-profile relationships survive?", fontsize=19, weight="bold", va="top")
    reference_titles = {"response_difference": "Response distance",
                        "oracle_codebook": "Oracle level similarity"}
    for row, reference in enumerate(reference_titles):
        subset = frame.filter(pl.col("reference") == reference)
        for col, (metric, title) in enumerate((
            ("spearman", "Pair ranking · Spearman ρ"),
            ("top3_neighbor_overlap", "Three nearest neighbors retained"),
        )):
            ax = axes[row, col]
            summary = aggregate(subset, metric, ["dimension", "bundle_width"])
            for dimension in COLORS:
                curve(ax, summary, dimension)
            axes_style(ax, percent=col == 1)
            ax.set_title(title if row == 0 else "", loc="left", weight="bold", pad=10)
            if col == 0:
                ax.set_ylabel(reference_titles[reference], labelpad=11)
            if col == 1:
                ax.set_ylim(.75, 1.01)
                ax.set_yticks([.75, .8, .85, .9, .95, 1])
            else:
                ax.set_ylim(.94, 1.002)
                ax.set_yticks([.94, .96, .98, 1])
            if row == 1:
                ax.set_xlabel("Facts in the bundle", labelpad=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Dimensions", ncol=4, frameon=False,
               loc="upper left", bbox_to_anchor=(.08, .90))
    fig.text(.99, .035, "Vertical scales are zoomed · lines: five-seed means · shading: seed range",
             ha="right", color=MUTED, fontsize=9)
    save(fig, output / "profile_geometry")


def perturbation(frame: pl.DataFrame, output: Path):
    fig, ax = plt.subplots(figsize=(9.1, 5.6))
    fig.subplots_adjust(left=.095, right=.82, top=.80, bottom=.20)
    fig.text(.095, .965, "Does profile similarity fade as bundles grow?",
             fontsize=17, weight="bold", va="top")
    colors = {"near": "#247B80", "far": "#B3603C"}
    labels = {"near": "20% changed", "far": "60% changed"}
    for condition in ("near", "far"):
        band = (frame.filter(pl.col("condition") == condition)
                .group_by("bundle_width")
                .agg(pl.col("bundle_cosine").median().alias("median"),
                     pl.col("bundle_cosine").quantile(.05).alias("low"),
                     pl.col("bundle_cosine").quantile(.95).alias("high"))
                .sort("bundle_width"))
        x = band["bundle_width"].to_list()
        ax.fill_between(x, band["low"].to_list(), band["high"].to_list(),
                        color=colors[condition], alpha=.16, linewidth=0)
        ax.plot(x, band["median"].to_list(), color=colors[condition],
                linewidth=2.7, marker="o", markersize=5)
        ax.text(51.5, band["median"][-1], labels[condition],
                color=colors[condition], va="center", fontsize=10, weight="bold")
    ax.set_xticks([5, 10, 20, 40, 50])
    ax.set_xlim(3, 66)
    ax.set_ylim(.825, .97)
    ax.set_yticks([.85, .90, .95])
    ax.set_xlabel("Facts in the bundle", labelpad=9)
    ax.set_ylabel("Cosine similarity", labelpad=10)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)
    fig.text(.095, .095, "Five MAP seeds · shading covers the middle 90% of pairs",
             color=MUTED, fontsize=9)
    save(fig, output / "perturbation_similarity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "qa-encoding")
    args = parser.parse_args()
    frame = pl.read_csv(args.results / "recovery_by_seed.csv")
    rank = pl.read_csv(args.results / "geometry_by_seed.csv")
    expected_recovery = 4 * 5 * 6
    expected_rank = 4 * 5 * 6 * 2
    if frame.height != expected_recovery or rank.height != expected_rank:
        raise ValueError("Expected four dimensions, five seeds and six bundle widths")
    style()
    recovery(frame, args.results)
    geometry(rank, args.results)
    perturbation(pl.read_parquet(args.results / "perturbation_pairs.parquet"), args.results)
    print(f"Saved three PNG/SVG figures in {args.results}")


if __name__ == "__main__":
    main()
