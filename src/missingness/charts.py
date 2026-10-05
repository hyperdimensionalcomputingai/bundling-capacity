"""Figures for the parked missingness work: HYP-83 Experiment 2 (pairwise) and the
HYP-118 Experiment 1 prototype (property normalization)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

import pairwise
import person

# Categorical slots 1-3 (validated; aqua is under 3:1 contrast, so every line is labelled directly).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


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


def header(fig, title: str, subtitle: str):
    fig.text(0.06, 0.975, title, fontsize=16, weight="bold", va="top")
    fig.text(0.06, 0.915, subtitle, fontsize=10.5, color=MUTED, va="top")


def footnote(fig, text: str):
    fig.text(0.06, 0.015, text, color=MUTED, fontsize=8.5)


PAIRWISE = (
    ("omit", "omit the fact", BLUE, "o", "-"),
    ("token_null", '"null" token', ORANGE, "o", "-"),
    ("token_no_value", '"no_value" token', ORANGE, "s", "none"),
    ("token_empty_string", '"" token', ORANGE, "^", "none"),
    ("token_mixed_spellings", "mixed spellings", AQUA, "D", "--"),
)


def experiment2(summary: pl.DataFrame, out: Path, dimension: int = 2048):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.86, top=0.78, bottom=0.17, wspace=0.1)
    header(
        fig,
        "A null token dilutes a record and makes strangers look alike",
        "Cosine of one pair as scalar fields go missing: lines are the exact expectation, "
        "markers the measured MAP cosine (5th–95th percentile bars)",
    )
    panels = (
        ("same_person", "Same person, one copy missing k fields"),
        ("unrelated", "Unrelated people, both missing the same k fields"),
    )
    k_values = sorted(summary["missing_fields"].unique().to_list())
    for ax, (pair_type, title) in zip(axes, panels):
        part = summary.filter(
            (pl.col("pair_type") == pair_type) & (pl.col("dimension") == dimension)
        )
        for offset, (strategy, label, color, marker, line) in enumerate(PAIRWISE):
            rows = part.filter(pl.col("strategy") == strategy).sort("missing_fields")
            k = rows["missing_fields"].to_numpy() + (offset - 2) * 0.05
            if line != "none":
                ax.plot(
                    rows["missing_fields"],
                    rows["expected_cosine"],
                    color=color,
                    linewidth=1.6,
                    alpha=0.9,
                    linestyle=(0, (4, 3)) if line == "--" else "-",
                )
            mean = rows["measured_cosine"].to_numpy()
            ax.errorbar(
                k,
                mean,
                yerr=[
                    mean - rows["measured_p05"].to_numpy(),
                    rows["measured_p95"].to_numpy() - mean,
                ],
                fmt=marker,
                color=color,
                markersize=6,
                capsize=0,
                elinewidth=1.2,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=label,
            )
        ax.set_xticks(k_values)
        ax.set_xlabel("Scalar fields missing (of age, job, region)", labelpad=8)
        ax.set_title(title, loc="left", fontsize=11, pad=10)
        frame_axes(ax)
    axes[0].set_ylabel("Cosine similarity", labelpad=8)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="center left", bbox_to_anchor=(0.865, 0.5))
    footnote(
        fig,
        f"D = {dimension:,} · 2,000 pairs per type × {len(pairwise.SEEDS)} seeds · interests always present, so a "
        "complete record has 6 facts · the three single-spelling tokens coincide by construction",
    )
    save(fig, out / "experiment2_pairwise")


NORMALIZATION = (
    ("without_normalization", "Without normalization", BLUE),
    ("with_normalization", "With normalization", ORANGE),
)


def experiment5(summary: pl.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    fig.subplots_adjust(left=0.1, right=0.74, top=0.8, bottom=0.17)
    header(
        fig,
        "Normalizing each property shrinks the cost of missing interests",
        "Cosine to the query as candidate A knows more of its 3 interests. Dashed: candidate B, "
        "complete but with a different job",
    )
    for encoding, label, color in NORMALIZATION:
        part = summary.filter(pl.col("encoding") == encoding)
        a = part.filter(pl.col("candidate_type") == "A").sort("known_interests")
        b = part.filter(pl.col("candidate") == "B")
        k = a["known_interests"].to_numpy()
        ax.plot(k, a["expected_cosine"], color=color, linewidth=1.6, alpha=0.9)
        mean = a["measured_mean"].to_numpy()
        ax.errorbar(
            k,
            mean,
            yerr=[mean - a["measured_min"].to_numpy(), a["measured_max"].to_numpy() - mean],
            fmt="o",
            color=color,
            markersize=7,
            capsize=0,
            elinewidth=1.2,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        b_mean = b["measured_mean"][0]
        ax.axhspan(b["measured_min"][0], b["measured_max"][0], color=color, alpha=0.1, linewidth=0)
        ax.axhline(b_mean, color=color, linestyle=(0, (4, 3)), linewidth=1.3)
        ax.text(3.15, b_mean, f"B, {label.lower()}: {b_mean:.3f}", va="center", fontsize=9.5)
        # Direct labels in the empty space above (with) and below (without) the A_0 point.
        above = encoding == "with_normalization"
        ax.text(
            -0.12 if above else 0.06,
            mean[0] + (0.065 if above else -0.015),
            label,
            va="bottom" if above else "top",
            color=color,
            fontsize=9.5,
            weight="bold",
        )
    ax.set_xticks(k)
    ax.set_xticklabels([f"{n} of 3" for n in k])
    ax.set_xlim(-0.25, 3.1)
    ax.set_xlabel("Interests candidate A has on record (all match the query)", labelpad=8)
    ax.set_ylabel("Cosine similarity to the query", labelpad=8)
    frame_axes(ax)
    footnote(
        fig,
        f"D = 2,048 · lines: exact value with orthogonal atoms · markers: mean of {summary['seeds'][0]} "
        "seeds (bars and bands: seed range) · A matches age, job and region; unknown interests are omitted",
    )
    save(fig, out / "experiment5_normalization")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=person.OUT)
    out = parser.parse_args().output_dir
    style()
    experiment2(pl.read_csv(out / "pairwise_summary.csv"), out)
    experiment5(pl.read_csv(out / "normalization_summary.csv"), out)
    print(f"Saved figures in {out}")


if __name__ == "__main__":
    main()
