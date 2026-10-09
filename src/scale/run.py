"""Run the scale study end to end: records, encoder check, scan, summaries, figures, report.

From the repository root:

    uv run --project src/scale --locked python src/scale/run.py

Each (D, vector seed) is one pass through the million-record sequence, with summaries
taken as the scan reaches each prefix N. Everything written to results/scale and
src/scale/REPORT.md is regenerated from scratch.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullLocator

import study
from report import write_report
from study import MU, Encoder

KEYS = ["dimension", "seed", "n", "k"]
LARGEST = max(study.PREFIXES)
# The encoder check: record pairs per (D, k), each with exactly k shared properties.
VALIDATION_PAIRS = 20_000
VALIDATION_SEED = 11
VALIDATION_DATA_SEED = 202  # separate from the population's data seed 101


def regenerate_records() -> dict:
    """Run data/scale/generate.py so the records always come from the fixed data seed."""
    path = study.ROOT / "data" / "scale" / "generate.py"
    spec = importlib.util.spec_from_file_location("generate", path)
    generate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate)
    if (
        tuple(generate.PROPERTIES) != study.PROPERTIES
        or tuple(generate.PROPERTIES.values()) != study.CARDINALITIES
    ):
        raise AssertionError("generate.py and study.py disagree on the properties")
    generate.main()
    return json.loads(generate.SETTINGS_PATH.read_text())


# ----------------------------------------------------------------------------- encoder check


def validate_encoder() -> pl.DataFrame:
    """Controlled pairs with exactly k shared properties, encoded with the study's vectors.

    Each pair draws a random query record, copies k randomly chosen properties into the
    candidate and gives the other properties a different value. Value vectors are reused
    across pairs, as in the population, so this checks the encoder in realistic conditions.
    """
    rows = []
    cardinalities = torch.tensor(study.CARDINALITIES)
    for dimension in study.DIMENSIONS:
        encoder = Encoder.make(dimension, VALIDATION_SEED)
        for k in study.GROUPS:
            generator = torch.Generator().manual_seed(VALIDATION_DATA_SEED * 10 + k)
            shape = (VALIDATION_PAIRS, study.FACTS)
            query = (torch.rand(shape, generator=generator) * cardinalities).long()
            order = torch.rand(shape, generator=generator).argsort(1)
            shared = torch.zeros(shape, dtype=torch.bool).scatter(1, order[:, :k], True)
            offset = 1 + (torch.rand(shape, generator=generator) * (cardinalities - 1)).long()
            candidate = torch.where(shared, query, (query + offset) % cardinalities)
            assert ((query == candidate).sum(1) == k).all()
            q, c = encoder.encode(query), encoder.encode(candidate)
            scores = study.cosines(q, c).diagonal()
            rows.append(
                {
                    "dimension": dimension,
                    "seed": VALIDATION_SEED,
                    "k": k,
                    "pairs": VALIDATION_PAIRS,
                    "predicted_mean": MU[k],
                    "measured_mean": float(scores.mean()),
                    # Theory predicts the spread for zero overlap only; shared facts narrow it.
                    "predicted_std": study.spread(dimension) if k == 0 else None,
                    "measured_std": float(scores.std(correction=0)),
                }
            )
    return pl.DataFrame(rows)


# ----------------------------------------------------------------------------- summaries


def add_theory(cells: pl.DataFrame) -> pl.DataFrame:
    """Predicted mean, spread, maximum reference and expected exceedances per row."""
    references = study.exceedance_scores()
    extra = []
    for row in cells.select("dimension", "k", "count").iter_rows(named=True):
        d, k, m = row["dimension"], row["k"], row["count"]
        zero = k == 0
        entry = {
            "predicted_mean": MU[k],
            "predicted_std": study.spread(d) if zero else None,
            "max_reference": study.maximum_reference(m, d) if zero else None,
        }
        for name, s in references.items():
            entry[f"expected_zero_at_or_above_{name}"] = m * study.tail_probability(s, d) if zero else None
        extra.append(entry)
    return pl.concat([cells, pl.DataFrame(extra)], how="horizontal_extend")


def seed_summary(cells: pl.DataFrame) -> pl.DataFrame:
    """Average over vector seeds, with the seed range, per (D, N, k)."""
    measured = ["mean", "std", "max", "zero_at_or_above_half_mu1", "zero_at_or_above_mu1"]
    theory = [c for c in cells.columns if c.startswith(("predicted_", "expected_"))]
    return (
        cells.group_by(["dimension", "n", "k"])
        .agg(
            pl.col("count").first(),
            pl.col("count").n_unique().alias("count_variants"),
            *[pl.col(c).first() for c in [*theory, "max_reference"]],
            *[pl.col(c).mean().alias(f"{c}_seed_mean") for c in measured],
            *[pl.col(c).min().alias(f"{c}_seed_min") for c in measured],
            *[pl.col(c).max().alias(f"{c}_seed_max") for c in measured],
        )
        .sort(["dimension", "n", "k"])
    )


# ----------------------------------------------------------------------------- figures

# Categorical slots for k = 0, 1, 2 (validated light-mode palette; aqua is under 3:1
# contrast against the surface, so every series also carries a direct label and legend).
K_COLORS = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a"}
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
K_LABELS = {0: "zero shared", 1: "one shared", 2: "two shared"}


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


def save(fig, name: str):
    for ext in ("png", "svg"):
        fig.savefig(study.OUT / f"{name}.{ext}", dpi=190, facecolor="white")
    svg = study.OUT / f"{name}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def dimension_title(d: int) -> str:
    return f"D = {d:,}"


def figure_shared_properties(summary: pl.DataFrame):
    """Figure 1: mean cosine by k at N = 1,000,000, one panel per D."""
    data = summary.filter((pl.col("n") == LARGEST) & (pl.col("k") <= 2))
    fig, axes = plt.subplots(1, len(study.DIMENSIONS), figsize=(13, 4.1), sharey=True)
    for ax, d in zip(axes, study.DIMENSIONS):
        rows = data.filter(pl.col("dimension") == d).sort("k")
        for row in rows.iter_rows(named=True):
            k, color = row["k"], K_COLORS[row["k"]]
            # Theory: the predicted mean as a tick; +/- one predicted std as a band (k = 0 only).
            if row["predicted_std"] is not None:
                ax.fill_between(
                    [k - 0.32, k - 0.08],
                    row["predicted_mean"] - row["predicted_std"],
                    row["predicted_mean"] + row["predicted_std"],
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                )
            ax.plot([k - 0.32, k - 0.08], [row["predicted_mean"]] * 2, color=color, lw=2)
            # Measured: seed-averaged mean, +/- the seed-averaged standard deviation.
            ax.errorbar(
                k + 0.2,
                row["mean_seed_mean"],
                yerr=row["std_seed_mean"],
                fmt="o",
                color=color,
                markersize=7,
                capsize=4,
                elinewidth=2,
                markeredgecolor="white",
                markeredgewidth=1.2,
            )
        ax.set_title(dimension_title(d), fontsize=11, loc="left")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["0", "1", "2"])
        ax.set_xlim(-0.6, 2.6)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.set_xlabel("shared properties, k")
    axes[0].set_ylabel("cosine similarity")
    handles = [
        Patch(facecolor=MUTED, alpha=0.25, label="predicted mean (left; band ± 1/√D for k = 0)"),
        Line2D(
            [], [], color=MUTED, marker="o", linestyle="none", label="measured mean ± 1 std (right)"
        ),
        *[Patch(facecolor=K_COLORS[k], label=f"k = {k}, {K_LABELS[k]}") for k in (0, 1, 2)],
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.suptitle(
        f"Cosine rises by {MU[1]:.1f} per shared property; the spread narrows as D grows",
        x=0.06,
        ha="left",
        fontsize=12.5,
        y=1.02,
    )
    fig.text(
        0.06,
        0.945,
        "N = 1,000,000 candidates, 100 queries, mean of vector seeds 11, 23 and 37. "
        "Bars and bands show the spread of individual scores, not confidence intervals.",
        ha="left",
        fontsize=9.5,
        color=MUTED,
    )
    save(fig, "figure1_shared_properties")


def figure_zero_overlap_maximum(summary: pl.DataFrame):
    """Figure 2: highest zero-overlap cosine against N, with the k = 1 and 2 score bands."""
    fig, axes = plt.subplots(1, len(study.DIMENSIONS), figsize=(13, 4.3), sharey=True)
    ns = list(study.PREFIXES)
    for ax, d in zip(axes, study.DIMENSIONS):
        rows = summary.filter(pl.col("dimension") == d)
        for k in (1, 2):
            g = rows.filter(pl.col("k") == k).sort("n")
            mean, std = g["mean_seed_mean"], g["std_seed_mean"]
            ax.fill_between(ns, mean - std, mean + std, color=K_COLORS[k], alpha=0.16, lw=0)
            ax.plot(ns, mean, color=K_COLORS[k], lw=1.6)
        zero = rows.filter(pl.col("k") == 0).sort("n")
        ax.plot(ns, zero["max_reference"], color=INK, lw=1.4, linestyle=(0, (4, 3)), zorder=4)
        ax.fill_between(
            ns, zero["max_seed_min"], zero["max_seed_max"], color=K_COLORS[0], alpha=0.3, lw=0
        )
        ax.plot(
            ns,
            zero["max_seed_mean"],
            color=K_COLORS[0],
            marker="o",
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=1.2,
        )
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(ns))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(["10k", "100k", "1M"])
        ax.set_xlim(ns[0] / 1.6, ns[-1] * 1.6)
        ax.set_title(dimension_title(d), fontsize=11, loc="left")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.set_xlabel("candidates, N")
        if d == study.DIMENSIONS[0]:
            last = rows.filter(pl.col("n") == LARGEST)
            for k in (1, 2):
                y = last.filter(pl.col("k") == k)["mean_seed_mean"][0]
                ax.annotate(
                    f"k = {k} mean",
                    (ns[0], y),
                    xytext=(0, 5),
                    textcoords="offset points",
                    fontsize=8.5,
                    color=INK,
                )
    axes[0].set_ylabel("cosine similarity")
    handles = [
        Line2D(
            [],
            [],
            color=K_COLORS[0],
            marker="o",
            markeredgecolor="white",
            label="highest zero-overlap score (seed mean; band = seed range)",
        ),
        Line2D([], [], color=INK, linestyle=(0, (4, 3)), label="theoretical maximum reference"),
        Patch(facecolor=K_COLORS[1], alpha=0.35, label="one shared: mean ± 1 std"),
        Patch(facecolor=K_COLORS[2], alpha=0.35, label="two shared: mean ± 1 std"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.5, -0.09),
    )
    fig.suptitle(
        "The highest zero-overlap score creeps up with N and falls sharply with D",
        x=0.06,
        ha="left",
        fontsize=12.5,
        y=1.02,
    )
    fig.text(
        0.06,
        0.945,
        "Exact cosine, 100 queries against every other record in each prefix; "
        "vector seeds 11, 23 and 37.",
        ha="left",
        fontsize=9.5,
        color=MUTED,
    )
    save(fig, "figure2_zero_overlap_maximum")


# ----------------------------------------------------------------------------- main


def versions() -> dict:
    names = ["torch", "torch-hd", "polars", "pyarrow", "scipy", "matplotlib"]
    return {
        "python": platform.python_version(),
        **{n: importlib.metadata.version(n) for n in names},
        "machine": f"{platform.system()} {platform.machine()}",
        "torch_threads": torch.get_num_threads(),
    }


def main():
    started = time.perf_counter()
    timings = {}
    study.OUT.mkdir(parents=True, exist_ok=True)

    t = time.perf_counter()
    data_settings = regenerate_records()
    codes = study.load_codes()
    timings["records_s"] = time.perf_counter() - t

    t = time.perf_counter()
    validation = validate_encoder()
    validation.write_csv(study.OUT / "encoder_validation.csv")
    timings["encoder_validation_s"] = time.perf_counter() - t

    per_query, scans = [], {}
    for d in study.DIMENSIONS:
        for seed in study.VECTOR_SEEDS:
            t = time.perf_counter()
            per_query.append(study.scan(Encoder.make(d, seed), codes))
            scans[f"d{d}_s{seed}"] = round(time.perf_counter() - t, 2)
            print(f"D={d:>6,} seed={seed}: {scans[f'd{d}_s{seed}']:.1f} s", flush=True)
    per_query = pl.concat(per_query)
    per_query.write_parquet(study.OUT / "per_query.parquet", compression="zstd")
    timings["scan_s"] = scans

    cells = add_theory(study.pool(per_query, KEYS))
    cells.write_csv(study.OUT / "cell_summary.csv")
    summary = seed_summary(cells)
    if summary["count_variants"].max() != 1:
        raise AssertionError("Group counts depend only on the records, never on the seed")
    summary = summary.drop("count_variants")
    summary.write_csv(study.OUT / "seed_summary.csv")

    t = time.perf_counter()
    style()
    figure_shared_properties(summary)
    figure_zero_overlap_maximum(summary)
    timings["figures_s"] = time.perf_counter() - t

    timings["total_s"] = time.perf_counter() - started
    manifest = {
        "settings": {
            "dimensions": study.DIMENSIONS,
            "prefixes": study.PREFIXES,
            "vector_seeds": study.VECTOR_SEEDS,
            "vector_draw": "torch.Generator().manual_seed(seed * 1_000_003 + D); "
            "torchhd.random roles, then values per property in order",
            "query_panel": f"first {study.QUERY_COUNT} records; self excluded, duplicates kept",
            "batch": study.BATCH,
            "facts": study.PROPERTIES,
            "cardinalities": study.CARDINALITIES,
            "bundling": "raw coordinate-wise sum of all five bound facts (no sign)",
            "score": "cosine: integer dot / product of the two record lengths",
            "predicted_means": MU,
            "exceedance_scores": study.exceedance_scores(),
            "validation": {
                "pairs_per_k": VALIDATION_PAIRS,
                "vector_seed": VALIDATION_SEED,
                "data_seed": VALIDATION_DATA_SEED,
            },
        },
        "records": data_settings,
        "versions": versions(),
        "timings": timings,
    }
    (study.OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    write_report(summary, validation, data_settings, timings)
    print(f"Done in {timings['total_s']:.0f} s")


if __name__ == "__main__":
    main()
