"""Run the collision study end to end: records, all-pairs trials, summaries, figures, report.

From the repository root:

    uv run --project src/scale --locked python src/scale/run.py

Each trial draws fresh hypervectors for one dimension and compares every record with every
other record once. The nested populations (the first 4,000, 13,000 and 40,000 records) are
summarized from the same pass. Everything in results/scale and src/scale/REPORT.md is
regenerated from scratch.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import math
import platform
import time

import polars as pl
from plotnine import (
    aes,
    element_blank,
    element_line,
    element_rect,
    element_text,
    facet_wrap,
    geom_col,
    geom_line,
    geom_point,
    geom_step,
    geom_text,
    geom_vline,
    ggplot,
    labs,
    position_dodge,
    scale_color_manual,
    scale_fill_manual,
    scale_x_continuous,
    scale_x_log10,
    scale_y_continuous,
    scale_y_log10,
    theme,
    theme_minimal,
)

import study
from report import write_report

# ----------------------------------------------------------------------------- records


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


# ----------------------------------------------------------------------------- summaries


def summarize(trials: pl.DataFrame) -> pl.DataFrame:
    """Per (D, N): measured counts and chances over trials, beside the binomial predictions."""
    rows = []
    for (d, n), group in trials.group_by(["dimension", "n"], maintain_order=True):
        first = group.row(0, named=True)
        row = {
            "dimension": d,
            "n": n,
            "trials": group.height,
            "pairs": first["pairs"],
            "unrelated_pairs": first["unrelated_pairs"],
            "storage_bytes_per_record": d,  # int8 raw sums
            "highest_unrelated_mean": group["highest_unrelated"].mean(),
            "highest_unrelated_max": group["highest_unrelated"].max(),
        }
        for q in study.MATCHES:
            row[f"threshold_{q}"] = first[f"threshold_{q}"]
            row[f"false_pairs_{q}_mean"] = group[f"false_pairs_{q}"].mean()
            row[f"dataset_with_false_{q}"] = (group[f"false_pairs_{q}"] > 0).mean()
            row[f"search_with_false_{q}"] = (group[f"searches_with_false_{q}"] / n).mean()
            row[f"miss_rate_{q}"] = group[f"misses_{q}"].sum() / group[f"true_pairs_{q}"].sum()
        row.update(study.predictions(first))
        rows.append(row)
    return pl.DataFrame(rows).sort(["dimension", "n"])


# ----------------------------------------------------------------------------- figures

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GROUP_COLORS = {"0": "#2a78d6", "1": "#eb6834", "2": "#1baf7a"}
GROUP_LABELS = {"0": "share nothing", "1": "share 1 fact", "2": "share 2 facts"}
JOB_COLORS = {"search": "#4a3aa7", "dataset": "#e34948"}
JOB_LABELS = {
    "search": "someone unrelated looks like one person",
    "dataset": "any two unrelated people look alike",
}


def dimension_label(d: int) -> str:
    return f"{d:,} dimensions ({d // 1024} KB)"


# Figures are drawn at roughly the width of the article column, so their text stays legible
# when a phone shrinks them to fit. Panels stack vertically instead of sitting side by side.
COLUMN_WIDTH = 4.8


def house_theme():
    return theme_minimal(base_size=13) + theme(
        figure_size=(COLUMN_WIDTH, 4.2),
        plot_background=element_rect(fill="white", color="white"),
        panel_background=element_rect(fill=SURFACE, color=SURFACE),
        panel_grid_major=element_line(color=GRID, size=0.5),
        panel_grid_minor=element_blank(),
        strip_text=element_text(color=INK, size=14, ha="left"),
        axis_title=element_text(color=MUTED, size=13),
        axis_text=element_text(color=MUTED, size=13),
        legend_position="top",
        legend_title=element_blank(),
        legend_text=element_text(color=INK, size=13),
        panel_spacing_y=0.04,
    )


def count_label(x: float) -> str:
    """A pair count in short form: 100, 100K, 100M."""
    for size, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if x >= size:
            return f"{x / size:,.0f}{suffix}"
    return f"{x:,.0f}"


def figure_scores(histograms: pl.DataFrame, summary: pl.DataFrame):
    """Figure 1: how scores spread for each overlap group, one dimension per row."""
    largest = max(study.SIZES)
    order = [dimension_label(d) for d in study.DIMENSIONS]
    data = histograms.filter(pl.col("count") > 0).with_columns(
        pl.col("k").cast(pl.Utf8),
        pl.col("dimension")
        .map_elements(dimension_label, return_dtype=pl.Utf8)
        .cast(pl.Enum(order))
        .alias("panel"),
    )
    lines = (
        summary.filter(pl.col("n") == largest)
        .select("dimension", "threshold_1", "threshold_2")
        .unpivot(index="dimension", variable_name="which", value_name="t")
        .with_columns(
            pl.col("dimension")
            .map_elements(dimension_label, return_dtype=pl.Utf8)
            .cast(pl.Enum(order))
            .alias("panel"),
            pl.when(pl.col("which") == "threshold_1")
            .then(pl.lit("1-fact\nthreshold"))
            .otherwise(pl.lit("2-fact\nthreshold"))
            .alias("label"),
        )
    )
    plot = (
        ggplot(data.to_pandas(), aes("score", "count", color="k"))
        + geom_step(size=0.9)
        + geom_vline(
            aes(xintercept="t"), data=lines.to_pandas(), color=INK, linetype="dashed", size=0.5
        )
        + geom_text(
            aes(x="t", y=1e9, label="label"),
            data=lines.to_pandas(),
            color=INK,
            size=12,
            ha="left",
            va="top",
            nudge_x=0.01,
            inherit_aes=False,
        )
        + facet_wrap("panel", ncol=1)
        + scale_color_manual(values=GROUP_COLORS, labels=GROUP_LABELS)
        + scale_y_log10(
            breaks=[1e2, 1e5, 1e8], labels=lambda v: [count_label(x) for x in v], limits=(5, 2e9)
        )
        + scale_x_continuous(limits=(-0.15, 0.55), breaks=[0, 0.2, 0.4])
        + labs(x="cosine similarity", y="pairs (log scale)")
        + house_theme()
        + theme(figure_size=(COLUMN_WIDTH, 7.4), legend_text=element_text(color=INK, size=12))
    )
    save(plot, "figure1_scores")


def percent_label(x: float) -> str:
    """A chance as a percentage to two significant figures, so tiny values stay readable."""
    if x >= 0.1:
        return f"{x * 100:.0f}%"
    return f"{x * 100:.2g}%"


def figure_false_matches(summary: pl.DataFrame, dimension: int = 2_048):
    """Figure 2: chance of at least one false positive, for one person and for any two people."""
    rows = []
    for r in summary.filter(pl.col("dimension") == dimension).iter_rows(named=True):
        for job in ("search", "dataset"):
            rows += [
                {"n": r["n"], "job": job, "kind": "predicted", "chance": r[f"{job}_chance_1"]},
                {"n": r["n"], "job": job, "kind": "measured", "chance": r[f"{job}_with_false_1"]},
            ]
    data = pl.DataFrame(rows)
    predicted = data.filter(pl.col("kind") == "predicted").to_pandas()
    measured = data.filter((pl.col("kind") == "measured") & (pl.col("chance") > 0)).to_pandas()
    largest = max(study.SIZES)
    ends = predicted[predicted["n"] == largest].copy()
    ends["label"] = ends["chance"].map(percent_label)
    # Name each line in the empty band between them, at the middle dataset, instead of a legend.
    middle = sorted(study.SIZES)[1]
    names = (
        pl.DataFrame({"job": ["dataset", "search"], "label": ["any two people", "one person"]})
        .join(data.filter((pl.col("kind") == "predicted") & (pl.col("n") == middle)), on="job")
        .with_columns(
            pl.when(pl.col("job") == "dataset")
            .then(pl.col("chance") / 10**0.9)
            .otherwise(pl.col("chance") * 10**0.9)
            .alias("chance")
        )
    )
    plot = (
        ggplot(predicted, aes("n", "chance", color="job"))
        + geom_line(size=1.1)
        + geom_point(data=measured, size=3.4, fill="white", stroke=1.3, shape="o")
        + geom_text(aes(label="label"), data=ends, size=13, ha="left", nudge_x=0.07)
        + geom_text(aes(label="label"), data=names.to_pandas(), size=14, ha="center", va="center")
        + scale_color_manual(values=JOB_COLORS, guide=None)
        + scale_x_log10(
            breaks=list(study.SIZES),
            labels=lambda v: [f"{x:,.0f}" for x in v],
            limits=(min(study.SIZES) / 1.15, largest * 2.6),
        )
        + scale_y_log10(
            breaks=[1e-6, 1e-4, 1e-2, 1], labels=lambda v: [percent_label(x) for x in v]
        )
        + labs(x="records in the dataset", y="chance of at least one false positive")
        + house_theme()
    )
    save(plot, "figure2_false_matches")


def figure_predicted_vs_measured(summary: pl.DataFrame, dimension: int = 2_048):
    """Figure 3: predicted and measured chances, one panel per question, stacked."""
    panels = {
        "search": "One person",
        "dataset": "Any two people",
    }
    rows = []
    for r in summary.filter(pl.col("dimension") == dimension).iter_rows(named=True):
        for job, predicted, measured in (
            ("search", r["search_chance_1"], r["search_with_false_1"]),
            ("dataset", r["dataset_chance_1"], r["dataset_with_false_1"]),
        ):
            for kind, chance in (("predicted", predicted), ("measured", measured)):
                rows.append(
                    {"panel": panels[job], "n": f"{r['n']:,}", "kind": kind, "chance": chance}
                )
    sizes = [f"{n:,}" for n in study.SIZES]
    data = pl.DataFrame(rows).with_columns(
        pl.col("panel").cast(pl.Enum(list(panels.values()))),
        pl.col("n").cast(pl.Enum(sizes)),
        pl.col("kind").cast(pl.Enum(["predicted", "measured"])),
        pl.col("chance").map_elements(bar_label, return_dtype=pl.Utf8).alias("label"),
    )
    dodge = position_dodge(width=0.8)
    plot = (
        ggplot(data.to_pandas(), aes("n", "chance", fill="kind"))
        + geom_col(position=dodge, width=0.75)
        + geom_text(aes(label="label"), position=dodge, va="bottom", size=11, color=INK)
        + facet_wrap("panel", ncol=1, scales="free_y")
        + scale_fill_manual(
            values={"predicted": "#b9b6ae", "measured": "#4a3aa7"},
            labels={"predicted": "predicted by theory", "measured": "measured"},
        )
        + scale_y_continuous(expand=(0, 0, 0.2, 0))
        + labs(x="records", y="", fill="")
        + house_theme()
        + theme(
            figure_size=(COLUMN_WIDTH, 6.4),
            panel_spacing_y=0.06,
            panel_grid_major=element_blank(),
            axis_text_y=element_blank(),
            axis_text_x=element_text(color=INK, size=13),
            panel_border=element_rect(color=MUTED, fill=None, linetype="dashed", size=0.8),
            strip_background=element_rect(color=MUTED, fill=SURFACE, linetype="dashed", size=0.8),
        )
    )
    save(plot, "figure3_predicted_vs_measured")


def bar_label(x: float) -> str:
    """A chance as a percentage to two significant figures, so tiny values stay readable."""
    if x == 0:
        return "0%"
    return f"{x * 100:.2g}%"


def figures_only():
    """Redraw every figure from the saved CSVs, without rerunning the trials."""
    summary = pl.read_csv(study.OUT / "summary.csv")
    figure_scores(pl.read_csv(study.OUT / "score_histograms.csv"), summary)
    figure_false_matches(summary)
    figure_predicted_vs_measured(summary)


def short_chance(x: float) -> str:
    """A probability as a percentage, or as a power of ten once it is tiny."""
    if x >= 0.01:
        return f"{x:.0%}"
    exponent = math.floor(math.log10(x))
    mantissa = round(x / 10**exponent)
    if mantissa == 10:
        mantissa, exponent = 1, exponent + 1
    power = f"10^{{{exponent}}}"
    return f"${power}$" if mantissa == 1 else rf"${mantissa}\times{power}$"


def save(plot, name: str):
    for ext in ("png", "svg"):
        plot.save(study.OUT / f"{name}.{ext}", dpi=190, verbose=False)


# ----------------------------------------------------------------------------- main


def versions() -> dict:
    names = ["torch", "torch-hd", "polars", "pyarrow", "scipy", "plotnine"]
    return {
        "python": platform.python_version(),
        **{n: importlib.metadata.version(n) for n in names},
        "machine": f"{platform.system()} {platform.machine()}",
    }


def main():
    started = time.perf_counter()
    study.OUT.mkdir(parents=True, exist_ok=True)
    for stale in study.OUT.iterdir():
        stale.unlink()
    data_settings = regenerate_records()
    codes = study.load_codes()

    trials, histograms, timings = [], [], {}
    for d in study.DIMENSIONS:
        t = time.perf_counter()
        for seed in range(study.TRIALS[d]):
            result, hist = study.all_pairs(study.Encoder.make(d, seed), codes, histogram=seed == 0)
            trials.append(result)
            if hist is not None:
                histograms.append(hist)
        timings[f"d{d}_s"] = round(time.perf_counter() - t, 1)
        print(f"D={d:>5,}: {study.TRIALS[d]} trials in {timings[f'd{d}_s']:.0f} s", flush=True)
    trials = pl.concat(trials)
    histograms = pl.concat(histograms)
    trials.write_csv(study.OUT / "trials.csv")
    histograms.write_csv(study.OUT / "score_histograms.csv")
    summary = summarize(trials)
    summary.write_csv(study.OUT / "summary.csv")

    figure_scores(histograms, summary)
    figure_false_matches(summary)
    figure_predicted_vs_measured(summary)

    timings["total_s"] = round(time.perf_counter() - started, 1)
    manifest = {
        "settings": {
            "dimensions": study.DIMENSIONS,
            "sizes": study.SIZES,
            "trials": study.TRIALS,
            "matches": study.MATCHES,
            "recall": study.RECALL,
            "vector_draw": "torch.Generator().manual_seed(seed * 1_000_003 + D), seed = trial",
            "bundling": "raw coordinate-wise sum of five bound facts (no sign), int8",
            "score": "cosine: integer dot / product of the two record lengths",
            "threshold": "q/5 - z_0.999 / sqrt(D)",
            "tail": "binomial: H ~ Binomial(D, 1/2), score (2H - D) / D",
        },
        "records": data_settings,
        "versions": versions(),
        "timings": timings,
    }
    (study.OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    write_report(summary, data_settings, timings)
    print(f"Done in {timings['total_s']:.0f} s")


if __name__ == "__main__":
    import sys

    figures_only() if "--figures-only" in sys.argv else main()
