"""Figures for HYP-118 Experiments 1-4, and HYP-83's parked pairwise experiment.

Static PNG and editable SVG, in the scale study's style. Categorical colours are
the validated slots 1-5 in fixed order; three of them sit under 3:1 contrast on the
surface, so every series is labelled directly as well as in a legend.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.ticker import PercentFormatter

import pairwise
import person

# Categorical slots 1-5, validated together (CVD and normal-vision separation pass; aqua,
# yellow and magenta are under 3:1 contrast, so every series is labelled directly).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
RAMP = ("#86b6ef", "#3987e5", "#1c5cab")  # one-hue ordinal ramp for 1, 2, 3 listed interests
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


def hyp83_pairwise(summary: pl.DataFrame, out: Path, dimension: int = 2048):
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
    save(fig, out / "hyp83_pairwise")


NORMALIZATION = (
    ("without_normalization", "Without normalization", BLUE),
    ("with_normalization", "With normalization", ORANGE),
)


def experiment1_ranking(summary: pl.DataFrame, out: Path):
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
    save(fig, out / "experiment1_ranking")


def experiment1_influence(shares: pl.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.87, top=0.78, bottom=0.17, wspace=0.3)
    header(
        fig,
        "Without normalization, listing more interests makes interests count for more",
        "Share of a same-person match supplied by the interests property; age, job and region "
        "known on both sides",
    )
    for ax, (encoding, title, _) in zip(axes, NORMALIZATION):
        part = shares.filter(pl.col("encoding") == encoding)
        for n_q, color in zip((1, 2, 3), RAMP):
            rows = part.filter(pl.col("query_interests") == n_q).sort("candidate_interests")
            k = rows["candidate_interests"].to_numpy()
            ax.plot(k, rows["expected_share"], color=color, linewidth=1.6)
            ax.plot(
                k,
                rows["measured_share"],
                "o",
                color=color,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=f"query lists {n_q}",
            )
            ax.text(
                3.12, rows["measured_share"][-1], f"query lists {n_q}", va="center", fontsize=9.5
            )
        ax.axhline(0.25, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.1)
        ax.text(1.0, 0.255, "one of four properties", color=MUTED, fontsize=9, va="bottom")
        ax.set_xticks([1, 2, 3])
        ax.set_xlim(0.85, 3.1)
        ax.set_title(title, loc="left", fontsize=11, pad=10)
        ax.set_xlabel("Interests the candidate lists", labelpad=8)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        frame_axes(ax)
    axes[0].set_ylabel("Interests' share of the match", labelpad=8)
    footnote(
        fig,
        f"D = 2,048 · lines: exact value with orthogonal atoms · markers: mean over "
        f"{int(shares['seeds'][0])} seeds × 500 records · both sides list their first interests, so "
        "the shorter list is shared",
    )
    save(fig, out / "experiment1_influence")


POLICY_STYLE = {
    "cosine": ("cosine", BLUE),
    "mismatch": ("mismatch", ORANGE),
    "gower": ("Gower", AQUA),
    "pivoted_fitted": ("pivoted (fitted)", YELLOW),
    "relevance": ("relevant records", MUTED),
}


def experiment2_dial(curve: pl.DataFrame, summary: pl.DataFrame, out: Path, rate: float = 0.3):
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12.6, 5.4), gridspec_kw={"width_ratios": [1.1, 1]}
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.17, wspace=0.28)
    header(
        fig,
        "Treating missing values as neutral floods search with sparse records",
        f"{rate:.0%} of values missing at random. Left: precision@10 along Singhal's pivot "
        "(1 = cosine). Right: what the top 10 is made of, 1% pool",
    )
    densities = (
        (1.0, "full pool", RAMP[2]),
        (0.1, "10% pool", RAMP[1]),
        (0.01, "1% pool", RAMP[0]),
    )
    # Labels sit just above or below each line's left end, where the lines are furthest apart.
    offsets = {1.0: 0.012, 0.1: -0.012, 0.01: 0.012}
    for density, label, color in densities:
        rows = curve.filter(
            (pl.col("rate") == rate)
            & (pl.col("density") == density)
            & (pl.col("panel") == "evaluation")
        ).sort("slope")
        left.plot(rows["slope"], rows["precision_at_10"], color=color, marker="o", markersize=4)
        dy = offsets[density]
        left.text(
            0.02,
            rows["precision_at_10"][0] + dy,
            label,
            fontsize=9.5,
            va="bottom" if dy > 0 else "top",
        )
    left.axvline(1, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.1)
    left.text(1.02, 0.02, "cosine", color=MUTED, fontsize=9, transform=left.get_xaxis_transform())
    left.text(0.0, 0.02, "mismatch", color=MUTED, fontsize=9, transform=left.get_xaxis_transform())
    left.set_xlim(-0.05, 1.7)
    left.set_xticks([0, 0.5, 1, 1.5])
    left.set_ylim(0, None)
    left.yaxis.set_major_formatter(PercentFormatter(1))
    left.set_xlabel("Pivot slope (above 1: missing values count less)", labelpad=8)
    left.set_ylabel("Precision@10 against the hidden complete records", labelpad=8)
    frame_axes(left)

    cell = summary.filter((pl.col("rate") == rate) & (pl.col("density") == 0.01))
    bars = [
        ("relevance", cell["relevant_share_complete"][0], cell["relevant_share_one_property"][0])
    ]
    for policy in ("mismatch", "cosine", "pivoted_fitted", "gower"):
        row = cell.filter(pl.col("policy") == policy)
        bars.append((policy, row["top10_share_complete"][0], row["top10_share_one_property"][0]))
    y = list(range(len(bars)))[::-1]
    for yi, (policy, complete, one) in zip(y, bars):
        label, color = POLICY_STYLE[policy]
        right.barh(yi, complete, color=color, height=0.55)
        right.text(
            complete + 0.015,
            yi,
            f"{complete:.0%} complete · {one:.0%} one property",
            va="center",
            fontsize=9,
        )
    right.set_yticks(y)
    right.set_yticklabels([POLICY_STYLE[p][0] for p, *_ in bars])
    right.set_xlim(0, 1.45)
    right.set_xticks([0, 0.5, 1])
    right.xaxis.set_major_formatter(PercentFormatter(1))
    right.set_xlabel("Share of top-10 slots held by complete records (1% pool)", labelpad=8)
    right.grid(axis="x")
    right.set_axisbelow(True)
    right.tick_params(length=0, pad=6)
    footnote(
        fig,
        "400 complete queries · relevant = top 10 by equal-weight similarity of the complete records "
        "(ties shared) · exact policy scores · pivot slope fitted on a separate 200-query panel",
    )
    save(fig, out / "experiment2_dial")


SCORER_STYLE = (
    ("cosine", "cosine (equal weights)", BLUE),
    ("fs_known", "Fellegi-Sunter, labelled m", ORANGE),
    ("fs_em", "Fellegi-Sunter, EM m", AQUA),
    ("fs_linear", "linear Fellegi-Sunter", YELLOW),
    ("cosine_weighted", "cosine + informativeness weights", MAGENTA),
)


def experiment3_linkage(summary: pl.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))
    fig.subplots_adjust(left=0.07, right=0.8, top=0.78, bottom=0.17, wspace=0.2)
    header(
        fig,
        "Weight by informativeness, but keep cosine's treatment of missing values",
        "Recall@1 of each query person's noisy duplicate (10% of values wrong), by share of values missing",
    )
    for ax, (density, title) in zip(
        axes, ((1.0, "Full pool, 920,000 records"), (0.01, "1% pool, about 9,800 records"))
    ):
        part = summary.filter(pl.col("density") == density)
        for scorer, label, color in SCORER_STYLE:
            rows = part.filter(pl.col("scorer") == scorer).sort("rate")
            ax.plot(
                rows["rate"],
                rows["recall_at_1"],
                color=color,
                marker="o",
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=label,
            )
        ax.set_xticks([0.1, 0.3, 0.5])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_ylim(0, None)
        ax.set_xlabel("Values missing at random", labelpad=8)
        ax.set_title(title, loc="left", fontsize=11, pad=10)
        frame_axes(ax)
    axes[0].set_ylabel("Recall@1", labelpad=8)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=False, loc="center left", bbox_to_anchor=(0.805, 0.5), fontsize=9.5
    )
    footnote(
        fig,
        "400 complete queries · exact comparisons · Fellegi-Sunter scores a missing property as 0 "
        "(neutral); the cosine scorers divide by sqrt(p_q · p_c)",
    )
    save(fig, out / "experiment3_linkage")


BUILD_LABELS = {
    "cosine_f32": "cosine, float32",
    "cosine_f16": "cosine, float16",
    "weighted_f16": "cosine + query weights, float16",
    "mismatch_f16": "mismatch, float16",
    "pivoted_f16": "pivoted (s = 1.2), float16",
    "sign_f16": "majority-sign cosine, float16",
    "gower_rerank": "Gower rerank of cosine top 100",
}


def experiment4_build(summary: pl.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    fig.subplots_adjust(left=0.3, right=0.95, top=0.78, bottom=0.17)
    header(
        fig,
        "One stored index, a dot product, and float16 serve the useful policies",
        "Precision@10 of LanceDB flat search over MAP vectors (bars) and of the exact policy it "
        "implements (ticks)",
    )
    order = [k for k in BUILD_LABELS if k in summary["implementation"].to_list()]
    y = list(range(len(order)))[::-1]
    for yi, name in zip(y, order):
        row = summary.filter(pl.col("implementation") == name)
        color = MAGENTA if name == "weighted_f16" else AQUA if name == "gower_rerank" else BLUE
        ax.barh(yi, row["precision_at_10"][0], color=color, height=0.55)
        ax.plot(
            [row["exact_policy_precision_at_10"][0]] * 2,
            [yi - 0.36, yi + 0.36],
            color=INK,
            linewidth=1.6,
        )
        ax.text(
            max(row["precision_at_10"][0], row["exact_policy_precision_at_10"][0]) + 0.008,
            yi,
            f"{row['precision_at_10'][0]:.1%} · agrees with exact top 10 {row['agreement_with_exact_top10'][0]:.0%}",
            va="center",
            fontsize=9,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([BUILD_LABELS[n] for n in order])
    ax.set_xlim(0, 0.62)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Precision@10 against the hidden complete records", labelpad=8)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)
    footnote(
        fig,
        "D = 2,048, seed 11 · first 100,000 records, 30% missing at random · 400 complete queries · "
        "float16 and float32 return the same top 10 for 96.5% of queries",
    )
    save(fig, out / "experiment4_build")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=person.OUT)
    out = parser.parse_args().output_dir
    style()
    experiment1_ranking(pl.read_csv(out / "experiment1_ranking_summary.csv"), out)
    experiment1_influence(pl.read_csv(out / "experiment1_influence.csv"), out)
    experiment2_dial(
        pl.read_csv(out / "experiment2_slope_curve.csv"),
        pl.read_csv(out / "experiment2_summary.csv"),
        out,
    )
    experiment3_linkage(pl.read_csv(out / "experiment3_summary.csv"), out)
    if (out / "experiment4_summary.csv").exists():
        experiment4_build(pl.read_csv(out / "experiment4_summary.csv"), out)
    hyp83_pairwise(pl.read_csv(out / "pairwise_summary.csv"), out)
    print(f"Saved figures in {out}")


if __name__ == "__main__":
    main()
