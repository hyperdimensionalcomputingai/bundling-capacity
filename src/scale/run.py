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
    geom_line,
    geom_point,
    geom_step,
    geom_text,
    geom_vline,
    ggplot,
    labs,
    scale_color_manual,
    scale_x_continuous,
    scale_x_log10,
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
GROUP_LABELS = {"0": "share nothing", "1": "share 1 property", "2": "share 2 properties"}
JOB_COLORS = {"search": "#4a3aa7", "dataset": "#e34948"}
JOB_LABELS = {"search": "searching for one person", "dataset": "checking the whole dataset"}


def dimension_label(d: int) -> str:
    return f"{d:,} dimensions ({d // 1024} KB per hypervector)"


def house_theme():
    return theme_minimal(base_size=11) + theme(
        figure_size=(12.5, 4.4),
        plot_background=element_rect(fill="white", color="white"),
        panel_background=element_rect(fill=SURFACE, color=SURFACE),
        panel_grid_major=element_line(color=GRID, size=0.5),
        panel_grid_minor=element_blank(),
        strip_text=element_text(color=INK, size=11, ha="left"),
        axis_title=element_text(color=MUTED, size=10),
        axis_text=element_text(color=MUTED, size=9),
        plot_title=element_text(color=INK, size=13, ha="left"),
        plot_subtitle=element_text(color=MUTED, size=10, ha="left"),
        legend_position="bottom",
        legend_title=element_blank(),
        legend_text=element_text(color=INK, size=10),
        panel_spacing_x=0.03,
    )


def figure_scores(histograms: pl.DataFrame, summary: pl.DataFrame):
    """Figure 1: how scores spread for each overlap group, with both thresholds marked."""
    largest = max(study.SIZES)
    data = histograms.filter(pl.col("count") > 0).with_columns(
        pl.col("k").cast(pl.Utf8),
        pl.col("dimension").map_elements(dimension_label, return_dtype=pl.Utf8).alias("panel"),
    )
    lines = (
        summary.filter(pl.col("n") == largest)
        .select("dimension", "threshold_1", "threshold_2")
        .unpivot(index="dimension", variable_name="which", value_name="t")
        .with_columns(
            pl.col("dimension").map_elements(dimension_label, return_dtype=pl.Utf8).alias("panel"),
            pl.when(pl.col("which") == "threshold_1")
            .then(pl.lit("1-fact\nthreshold"))
            .otherwise(pl.lit("2-fact\nthreshold"))
            .alias("label"),
        )
    )
    order = [dimension_label(d) for d in study.DIMENSIONS]
    data = data.with_columns(pl.col("panel").cast(pl.Enum(order)))
    lines = lines.with_columns(pl.col("panel").cast(pl.Enum(order)))
    plot = (
        ggplot(data.to_pandas(), aes("score", "count", color="k"))
        + geom_step(size=0.9)
        + geom_vline(
            aes(xintercept="t"), data=lines.to_pandas(), color=INK, linetype="dashed", size=0.5
        )
        + geom_text(
            aes(x="t", y=3e7, label="label"),
            data=lines.to_pandas(),
            color=INK,
            size=8,
            ha="left",
            nudge_x=0.012,
            inherit_aes=False,
        )
        + facet_wrap("panel", nrow=1)
        + scale_color_manual(values=GROUP_COLORS, labels=GROUP_LABELS)
        + scale_y_log10(labels=lambda v: [f"{x:,.0f}" for x in v])
        + scale_x_continuous(limits=(-0.25, 0.6), breaks=[-0.2, 0, 0.2, 0.4, 0.6])
        + labs(
            x="cosine similarity",
            y="pairs (log scale)",
            title="Unrelated pairs pile up near zero; the gap to the thresholds widens with dimension",
            subtitle=f"Every pair among {largest:,} records, first trial at each dimension. "
            "Thresholds keep about 99.9% of true matches.",
        )
        + house_theme()
    )
    save(plot, "figure1_scores")


def figure_false_matches(summary: pl.DataFrame):
    """Figure 2: chance of at least one false match, one search vs the whole dataset."""
    order = [dimension_label(d) for d in study.DIMENSIONS]
    rows = []
    for r in summary.iter_rows(named=True):
        panel = dimension_label(r["dimension"])
        rows += [
            {
                "panel": panel,
                "n": r["n"],
                "job": "search",
                "kind": "predicted",
                "chance": r["search_chance_1"],
            },
            {
                "panel": panel,
                "n": r["n"],
                "job": "dataset",
                "kind": "predicted",
                "chance": r["dataset_chance_1"],
            },
            {
                "panel": panel,
                "n": r["n"],
                "job": "search",
                "kind": "measured",
                "chance": r["search_with_false_1"],
                "trials": r["trials"],
            },
            {
                "panel": panel,
                "n": r["n"],
                "job": "dataset",
                "kind": "measured",
                "chance": r["dataset_with_false_1"],
                "trials": r["trials"],
            },
        ]
    data = pl.DataFrame(rows).with_columns(pl.col("panel").cast(pl.Enum(order)))
    predicted = data.filter(pl.col("kind") == "predicted").to_pandas()
    measured = data.filter((pl.col("kind") == "measured") & (pl.col("chance") > 0)).to_pandas()
    largest = max(study.SIZES)
    ends = predicted[predicted["n"] == largest].copy()
    ends["label"] = ends["chance"].map(short_chance)
    none_seen = (
        summary.filter((pl.col("n") == largest) & (pl.col("dataset_with_false_1") == 0))
        .with_columns(
            pl.col("dimension")
            .map_elements(dimension_label, return_dtype=pl.Utf8)
            .cast(pl.Enum(order))
            .alias("panel"),
            pl.format("none seen in\n{} trials", pl.col("trials")).alias("label"),
            (pl.col("dataset_chance_1") / 8).alias("y"),
        )
        .to_pandas()
    )
    plot = (
        ggplot(predicted, aes("n", "chance", color="job"))
        + geom_line(size=1)
        + geom_point(data=measured, size=3.2, fill="white", stroke=1.2, shape="o")
        + geom_text(
            aes(label="label"), data=ends, size=9, ha="left", nudge_x=0.06, show_legend=False
        )
        + geom_text(
            aes(x=largest * 1.12, y="y", label="label"),
            data=none_seen,
            color=MUTED,
            size=8,
            ha="left",
            va="top",
            inherit_aes=False,
        )
        + facet_wrap("panel", nrow=1, scales="free_y")
        + scale_color_manual(values=JOB_COLORS, labels=JOB_LABELS)
        + scale_x_log10(
            breaks=list(study.SIZES),
            labels=lambda v: [f"{x / 1000:,.0f}K" for x in v],
            limits=(min(study.SIZES) / 1.15, largest * 2.4),
        )
        + scale_y_log10(labels=lambda v: [short_chance(x) for x in v])
        + labs(
            x="records in the dataset",
            y="chance of at least one false match (log scale)",
            title="One search stays clean; the whole dataset fills up first",
            subtitle="Lines: binomial prediction, labelled at 40K records. Circles: measured over "
            "trials. A match shares at least one of five properties. Each panel has its own scale.",
        )
        + house_theme()
    )
    save(plot, "figure2_false_matches")


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
    main()
