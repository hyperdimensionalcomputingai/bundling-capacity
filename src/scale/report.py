"""Write src/scale/REPORT.md and results/scale/table_n1m.csv from the saved summaries.

The prose quotes numbers computed here, so a rerun keeps text and results in step. Values
are rounded to three decimals at most; comparison counts are whole numbers.
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl

import study
from study import MU

REPORT_PATH = Path(__file__).resolve().parent / "REPORT.md"
LARGEST = max(study.PREFIXES)


def f3(x: float | None) -> str:
    if x is None:
        return "—"
    text = f"{x:.3f}"
    return "0.000" if text == "-0.000" else text


def span(low: float, high: float) -> str:
    """A range; written with "to" when an end is negative, so the dash is not misread."""
    a, b = f3(low), f3(high)
    return f"{a} to {b}".replace("-", "−") if a.startswith("-") else f"{a}–{b}"


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def whole(x: float) -> str:
    return f"{round(x):,}"


def expected_count(x: float) -> str:
    if x >= 1:
        return whole(x)
    return "<0.001" if x < 0.0005 else f3(x)


def dims(d: int) -> str:
    return f"{d:,}"


def row(frame: pl.DataFrame, **match) -> dict:
    selected = frame
    for key, value in match.items():
        selected = selected.filter(pl.col(key) == value)
    if selected.height != 1:
        raise ValueError(f"Expected one row for {match}, found {selected.height}")
    return selected.row(0, named=True)


def main_table(summary: pl.DataFrame) -> tuple[pl.DataFrame, str]:
    """The study's one table: N = 1,000,000, k = 0, 1 and 2, seed averages and ranges."""
    data = summary.filter((pl.col("n") == LARGEST) & (pl.col("k") <= 2)).sort(["dimension", "k"])
    lines = [
        (
            "| D | k | Comparisons | Predicted mean | Measured mean (seed range) "
            "| Predicted std | Measured std (seed range) | Highest score (seed range) "
            "| Maximum reference |"
        ),
        "|---:|---:|---:|---:|---|---:|---|---|---:|",
    ]
    for r in data.iter_rows(named=True):
        zero = r["k"] == 0
        lines.append(
            f"| {dims(r['dimension'])} | {r['k']} | {whole(r['count'])} "
            f"| {f3(r['predicted_mean'])} "
            f"| {f3(r['mean_seed_mean'])} ({span(r['mean_seed_min'], r['mean_seed_max'])}) "
            f"| {f3(r['predicted_std'])} "
            f"| {f3(r['std_seed_mean'])} ({span(r['std_seed_min'], r['std_seed_max'])}) "
            + (
                f"| {f3(r['max_seed_mean'])} ({span(r['max_seed_min'], r['max_seed_max'])}) "
                f"| {f3(r['max_reference'])} |"
                if zero
                else "| — | — |"
            )
        )
    table = data.select(
        "dimension",
        "k",
        "count",
        "predicted_mean",
        "mean_seed_mean",
        "mean_seed_min",
        "mean_seed_max",
        "predicted_std",
        "std_seed_mean",
        "std_seed_min",
        "std_seed_max",
        *[
            pl.when(pl.col("k") == 0).then(pl.col(c)).alias(c)
            for c in ("max_seed_mean", "max_seed_min", "max_seed_max", "max_reference")
        ],
    )
    return table, "\n".join(lines)


def maximum_table(summary: pl.DataFrame) -> tuple[str, list[dict]]:
    """Part 2 at N = 1,000,000: the highest zero-overlap score against the k = 1 band."""
    lines = [
        (
            "| D | Zero-overlap comparisons | Maximum reference | Highest zero-overlap score "
            "(seed range) | k = 1 mean − 1 std | k = 1 mean | Gap below the k = 1 mean, "
            "in k = 1 stds |"
        ),
        "|---:|---:|---:|---|---:|---:|---:|",
    ]
    facts = []
    for d in study.DIMENSIONS:
        zero = row(summary, dimension=d, n=LARGEST, k=0)
        one = row(summary, dimension=d, n=LARGEST, k=1)
        two = row(summary, dimension=d, n=LARGEST, k=2)
        gap = (one["mean_seed_mean"] - zero["max_seed_mean"]) / one["std_seed_mean"]
        facts.append(
            {
                "d": d,
                "max": zero["max_seed_mean"],
                "max_low": zero["max_seed_min"],
                "max_high": zero["max_seed_max"],
                "reference": zero["max_reference"],
                "one_mean": one["mean_seed_mean"],
                "one_low": one["mean_seed_mean"] - one["std_seed_mean"],
                "two_low": two["mean_seed_mean"] - two["std_seed_mean"],
                "gap": gap,
                "sigmas": zero["max_seed_mean"] / zero["predicted_std"],
            }
        )
        lines.append(
            f"| {dims(d)} | {whole(zero['count'])} | {f3(zero['max_reference'])} "
            f"| {f3(zero['max_seed_mean'])} ({span(zero['max_seed_min'], zero['max_seed_max'])}) "
            f"| {f3(facts[-1]['one_low'])} | {f3(one['mean_seed_mean'])} | {f3(gap)} |"
        )
    return "\n".join(lines), facts


def growth_table(summary: pl.DataFrame) -> str:
    lines = [
        "| D | " + " | ".join(f"N = {whole(n)}" for n in study.PREFIXES) + " |",
        "|---:|" + "---|" * len(study.PREFIXES),
    ]
    for d in study.DIMENSIONS:
        cells = []
        for n in study.PREFIXES:
            r = row(summary, dimension=d, n=n, k=0)
            cells.append(f"{f3(r['max_seed_mean'])} (reference {f3(r['max_reference'])})")
        lines.append(f"| {dims(d)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def exceedance_table(summary: pl.DataFrame) -> tuple[str, list[dict]]:
    lines = [
        (
            f"| D | Expected ≥ {f3(MU[1] / 2)} | Measured ≥ {f3(MU[1] / 2)} (seed range) "
            f"| Expected ≥ {f3(MU[1])} | Measured ≥ {f3(MU[1])} (seed range) |"
        ),
        "|---:|---:|---|---:|---|",
    ]
    facts = []
    for d in study.DIMENSIONS:
        r = row(summary, dimension=d, n=LARGEST, k=0)
        cells = []
        for name in ("half_mu1", "mu1"):
            expected = r[f"expected_zero_at_or_above_{name}"]
            measured = r[f"zero_at_or_above_{name}_seed_mean"]
            low, high = (
                r[f"zero_at_or_above_{name}_seed_min"],
                r[f"zero_at_or_above_{name}_seed_max"],
            )
            cells += [expected_count(expected), f"{whole(measured)} ({whole(low)}–{whole(high)})"]
            facts.append({"d": d, "name": name, "expected": expected, "measured": measured})
        lines.append(f"| {dims(d)} | " + " | ".join(cells) + " |")
    return "\n".join(lines), facts


def validation_table(validation: pl.DataFrame) -> str:
    lines = [
        "| D | k | Predicted mean | Measured mean | Predicted std | Measured std |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in validation.sort(["dimension", "k"]).iter_rows(named=True):
        lines.append(
            f"| {dims(r['dimension'])} | {r['k']} | {f3(r['predicted_mean'])} "
            f"| {f3(r['measured_mean'])} | {f3(r['predicted_std'])} | {f3(r['measured_std'])} |"
        )
    return "\n".join(lines)


def sizing_rule(summary: pl.DataFrame, maxima: list[dict]) -> tuple[str, dict]:
    """A back-of-envelope rule from the normal approximation, checked against the measurements.

    The highest of M near-independent normal scores with spread 1/sqrt(D) sits about
    sqrt(2 ln M) spreads above zero. The one-shared mean sits mu_1 * sqrt(D) spreads above
    zero. Their difference approximates the gap in Table 2, and solving for D gives the
    dimension at which the gap reaches a chosen size. It is an orientation aid, not a
    replacement for the binomial reference.
    """
    m0 = row(summary, dimension=study.DIMENSIONS[0], n=LARGEST, k=0)["count"]
    z = math.sqrt(2 * math.log(m0))
    lines = [
        "| D | μ₁√D | Rule-of-thumb gap, μ₁√D − √(2 ln M₀) | Measured gap (Table 2) |",
        "|---:|---:|---:|---:|",
    ]
    for f in maxima:
        lift = MU[1] * math.sqrt(f["d"])
        lines.append(f"| {dims(f['d'])} | {f3(lift)} | {f3(lift - z)} | {f3(f['gap'])} |")
    needed = {g: ((z + g) / MU[1]) ** 2 for g in (0, 3)}
    return "\n".join(lines), {"m0": m0, "z": z, "needed": needed}


def relation(fact: dict) -> str:
    """Where the highest zero-overlap score sits relative to the k = 1 and k = 2 bands."""
    if fact["max"] >= fact["two_low"]:
        return "reaches into the two-shared band (above its mean minus one standard deviation)"
    if fact["max"] >= fact["one_mean"]:
        return "exceeds the typical one-shared score"
    if fact["max"] >= fact["one_low"]:
        return "stays below the one-shared mean but inside its one-standard-deviation band"
    return "stays below the one-shared band"


def write_report(summary, validation, data_settings, timings):
    table, table_md = main_table(summary)
    table.write_csv(study.OUT / "table_n1m.csv")
    maximum_md, maxima = maximum_table(summary)
    exceed_md, exceed = exceedance_table(summary)

    at_n = summary.filter((pl.col("n") == LARGEST) & (pl.col("k") <= 2))
    mean_error = (at_n["mean_seed_mean"] - at_n["predicted_mean"]).abs().max()
    zero_n = at_n.filter(pl.col("k") == 0)
    ratio = zero_n["std_seed_mean"] / zero_n["predicted_std"]
    every_cell = summary.filter(pl.col("k") == 0)
    max_error = (every_cell["max_seed_mean"] - every_cell["max_reference"]).abs().max()
    counts = {
        k: row(summary, dimension=study.DIMENSIONS[0], n=LARGEST, k=k)["count"]
        for k in study.GROUPS
    }
    total = sum(counts.values())
    by_d = {f["d"]: f for f in maxima}
    first, last = study.DIMENSIONS[0], study.DIMENSIONS[-1]
    small = {d: row(summary, dimension=d, n=min(study.PREFIXES), k=0) for d in (first, last)}
    std_8k = row(summary, dimension=8_192, n=LARGEST, k=0)
    std_10k = row(summary, dimension=10_000, n=LARGEST, k=0)
    narrower_predicted = 1 - std_10k["predicted_std"] / std_8k["predicted_std"]
    narrower_measured = 1 - std_10k["std_seed_mean"] / std_8k["std_seed_mean"]
    ex = {(f["d"], f["name"]): f for f in exceed}
    separated = [d for d in study.DIMENSIONS if by_d[d]["gap"] >= 3]
    sigmas = [f["sigmas"] for f in maxima]
    validation_error = (validation["measured_mean"] - validation["predicted_mean"]).abs().max()
    zero_val = validation.filter(pl.col("k") == 0)
    val_ratio = zero_val["measured_std"] / zero_val["predicted_std"]
    scan_seconds = sum(timings["scan_s"].values())
    rises = ", ".join(
        f"{f3(row(summary, dimension=d, n=LARGEST, k=0)['max_reference'] - row(summary, dimension=d, n=min(study.PREFIXES), k=0)['max_reference'])} at D = {dims(d)}"
        for d in study.DIMENSIONS
    )
    middle = by_d[study.DIMENSIONS[1]]
    middle_sentence = (
        "" if middle["d"] in separated else f"At D = {dims(middle['d'])} it {relation(middle)}. "
    )
    compared = [
        f"{f3(f['measured'] / f['expected'])} times the expectation at "
        f"{f3(MU[1] / 2 if f['name'] == 'half_mu1' else MU[1])} for D = {dims(f['d'])}"
        for f in exceed
        if f["expected"] >= 1
    ]
    quiet = [d for d in study.DIMENSIONS if ex[(d, "half_mu1")]["expected"] < 0.01]
    exceed_text = (
        "Where the expected count is at least one, the measured count is "
        + "; ".join(compared)
        + f". From D = {dims(quiet[0])}, the binomial model expects fewer than 0.01 "
        f"zero-overlap comparisons at or above {f3(MU[1] / 2)} among {whole(counts[0])}; "
        + (
            "none occurred."
            if all(ex[(d, "half_mu1")]["measured"] == 0 for d in quiet)
            else "some occurred."
        )
    )
    sizing_md, sizing = sizing_rule(summary, maxima)
    rise_pct = [
        100
        * (
            row(summary, dimension=d, n=LARGEST, k=0)["max_seed_mean"]
            / row(summary, dimension=d, n=min(study.PREFIXES), k=0)["max_seed_mean"]
            - 1
        )
        for d in study.DIMENSIONS
    ]
    growth = 100 * (
        math.sqrt(math.log(sizing["m0"]))
        / math.sqrt(math.log(row(summary, dimension=first, n=min(study.PREFIXES), k=0)["count"]))
        - 1
    )

    text = f"""# Bundled categorical records at scale

This report covers the scale study described in the [methodology](METHODOLOGY.md): exact cosine comparisons between records that bundle five categorical properties as a raw sum, against populations of up to one million records, at five hypervector dimensions. All values are rounded to three decimals; comparison counts are whole numbers.

## In brief

A hypervector record can be compared with a million others by cosine similarity. Two records that share nothing still score a little above or below zero by chance, and with enough candidates, one of those chance scores gets surprisingly high. This study measured how high, and whether it can be confused with the score of a record that genuinely shares one or two properties. Theory predicted the answers almost exactly. Each shared property adds about {f3(MU[1])} to the cosine. Chance scores scatter by about 1/√D. The highest chance score among {whole(counts[0])} comparisons sits about {f3(sum(sigmas) / len(sigmas))} of those spreads above zero. At D = {dims(first)} that is {f3(by_d[first]["max"])}, more than one shared property is worth. From D = {dims(separated[0])} it stays clearly below.

## Background for readers new to hypervectors

- **Hypervectors.** A hypervector is a long list of D numbers. Every role and value hypervector here has entries of +1 or −1, chosen at random. Two independent random hypervectors are almost orthogonal: their cosine similarity is close to 0, scattering by about 1/√D (0.044 at D = 512, 0.010 at D = 10,000). That scatter is the noise floor of everything below.
- **Binding ($\\otimes$).** Element-wise multiplication. Binding a property's role vector (say, "employer") with a value vector (say, employer no. 412) gives a new random-looking vector, a *fact*, unrelated to either input and to facts about other properties. Two records with the same employer produce the identical employer fact.
- **Bundling ($\\oplus$).** Adds the five facts coordinate by coordinate into one record vector of the same length. No sign is applied, so each coordinate is −5, −3, −1, 1, 3 or 5, and any fact can still be unbound from the record.
- **Cosine similarity.** The dot product of two records divided by the product of their lengths: 1 for identical records, about 0 for unrelated ones.
- **Why one shared fact is worth {f3(MU[1])}.** Multiplying two records coordinate by coordinate, a shared fact meets itself and contributes +1; every other product pairs unrelated terms and averages 0. The dot product is therefore about D per shared fact, and each record's length is about √(5D). One shared fact gives D / 5D = 1/5; k shared facts give k/5.
- **Why more candidates mean higher chance scores.** Each zero-overlap comparison is one random draw from a narrow bell curve around 0. The more draws, the further the most extreme one reaches into the tail. That tail thins very fast, so the maximum grows only with the square root of the logarithm of the number of comparisons.

## What a practitioner can expect

These rules of thumb hold under the measured conditions: five uniformly drawn categorical properties, one raw-sum bundle per record, exact cosine.

1. **Each shared property is worth {f3(MU[1])} cosine.** Records sharing zero, one, two or three properties score {f3(MU[0])}, {f3(MU[1])}, {f3(MU[2])} and {f3(MU[3])} on average, at every D and N tested. The measured means match these predictions to within {f3(mean_error)}.
2. **D sets the noise; N does not.** The spread of individual scores is about 1/√D: {f3(row(summary, dimension=first, n=LARGEST, k=0)["std_seed_mean"])} at D = {dims(first)} and {f3(row(summary, dimension=last, n=LARGEST, k=0)["std_seed_mean"])} at D = {dims(last)}. Adding candidates only adds more draws from the same distribution.
3. **The highest zero-overlap score rises slowly with N.** From 10,000 to 1,000,000 candidates (100 times as many), it rose from {f3(small[first]["max_seed_mean"])} to {f3(by_d[first]["max"])} at D = {dims(first)}, and from {f3(small[last]["max_seed_mean"])} to {f3(by_d[last]["max"])} at D = {dims(last)}. At a million candidates it sat {f3(min(sigmas))}–{f3(max(sigmas))} standard deviations above zero.
4. **At D = {dims(first)}, a million candidates are too many to keep chance scores below one shared property.** The highest zero-overlap score ({f3(by_d[first]["max"])}) exceeds the typical one-shared score ({f3(by_d[first]["one_mean"])}). {middle_sentence}From D = {dims(separated[0])} it sits at least {f3(min(by_d[d]["gap"] for d in separated))} one-shared standard deviations below the one-shared mean.
5. **The binomial model predicts the extremes well enough to size D in advance.** Across every D and N, the measured maximum stayed within {f3(max_error)} of the theoretical reference.
6. **Gains taper beyond 8,192 dimensions.** Moving to 10,000 dimensions narrowed the zero-overlap spread by {pct(narrower_measured)} (predicted {pct(narrower_predicted)}) and lowered the highest zero-overlap score from {f3(by_d[8_192]["max"])} to {f3(by_d[10_000]["max"])}.

None of this sets a match threshold. Whether one shared property counts as relevant is an application decision; the study measures how faithfully cosine reports the number of shared properties.

## Setup

- **Records.** {whole(data_settings["record_count"])} synthetic records with five independent, uniformly drawn properties: region (20 values), education (10), occupation (100), interest cluster (200) and employer (1,000). Data seed {data_settings["data_seed"]}. The sequence contains {whole(data_settings["duplicate_attribute_records"])} duplicate attribute records.
- **Encoder.** Each property has a random bipolar role vector and each value a random bipolar value vector. A record is the coordinate-wise sum of its five bound facts, with no sign applied, $h_{{\\mathrm{{record}}}} = h_{{\\mathrm{{fact}},1}} \\oplus \\cdots \\oplus h_{{\\mathrm{{fact}},5}}$ with $h_{{\\mathrm{{fact}},i}} = h_{{\\mathrm{{role}},i}} \\otimes h_{{\\mathrm{{value}},i}}$.
- **Comparisons.** The first 100 records are queries. Each is compared, by exact cosine, with every other record in nested prefixes N = 10,000, 100,000 and 1,000,000. That makes {whole(total)} directed comparisons at the largest prefix for each D and vector seed. A query's own record is excluded.
- **Grid.** D = 512, 2,048, 4,096, 8,192 and 10,000; vector seeds 11, 23 and 37. Each (D, seed) is one pass through the million records. The 15 passes took {whole(scan_seconds)} seconds in total on one laptop CPU.
- **Groups.** Pairs are grouped by k, the number of properties with identical values, from the categorical records alone. At N = 1,000,000 the panel has {whole(counts[0])} comparisons with k = 0, {whole(counts[1])} with k = 1, {whole(counts[2])} with k = 2, {whole(counts[3])} with k = 3, {whole(counts[4])} with k = 4 and {whole(counts[5])} with k = 5. None of the 100 queries happens to have a duplicate attribute record. These counts are the same for every D and seed.

### Terms used in this report

- **k, shared properties.** The number of properties on which two records have identical values, counted from the raw categorical records before any encoding. Groups are called *zero shared* (k = 0, also *zero-overlap*), *one shared* (k = 1) and *two shared* (k = 2).
- **Predicted mean and spread.** $\\mu_k = k/5$, and for zero overlap $\\sigma_0 = 1/\\sqrt{{D}}$, the standard deviation of a single score under the ideal model of independent random vectors. Spreads for shared properties are measured, not predicted.
- **Measured mean and spread.** Pooled over every comparison in a group (all 100 queries), per vector seed, then averaged over the three seeds. "Std" always means the spread of individual scores, never an uncertainty of the mean.
- **Highest zero-overlap score.** The single largest cosine among all zero-overlap comparisons for the whole query panel at a given N, D and seed.
- **Maximum reference.** The cosine that an ideal model expects about one zero-overlap comparison in M₀ to reach. It is a reference level for the maximum, not a bound.
- **Exceedances.** How many zero-overlap comparisons reach a fixed score level, compared with the number the binomial model expects.
- **Vector seed.** Selects one random set of role and value vectors. Different seeds are different, equally valid encoders of the same records.

## Part 1: cosine reflects the number of shared properties

![Mean cosine by shared-property count at N = 1,000,000](../../results/scale/figure1_shared_properties.png)

**Figure 1.** Mean cosine for zero, one and two shared properties at N = 1,000,000, one panel per D. Each predicted mean is a tick with a band of ± one predicted standard deviation. Each measured mean is a dot with bars of ± the measured standard deviation, averaged over seeds. Bars and bands show the spread of individual scores, not confidence intervals.

{table_md}

**Table 1.** N = 1,000,000; seed averages, with the range over vector seeds 11, 23 and 37 in parentheses. The comparison counts are identical across D and seeds.

The measured means differ from k/5 by at most {f3(mean_error)}. The zero-overlap standard deviations are {f3(ratio.min())}–{f3(ratio.max())} times 1/√D; shared facts narrow the spread a little, because shared terms always contribute the same +1. The ladder of means does not depend on D; increasing D only narrows each group. Growing N from 10,000 to 1,000,000 leaves the means and spreads unchanged to three decimals, as expected. The saved summaries also hold k = 3 and k = 4, whose measured means ({f3(row(summary, dimension=last, n=LARGEST, k=3)["mean_seed_mean"])} and {f3(row(summary, dimension=last, n=LARGEST, k=4)["mean_seed_mean"])} at D = {dims(last)}) agree with {f3(MU[3])} and {f3(MU[4])}. Only {whole(counts[4])} comparisons have k = 4, so its spread is noisy.

## Part 2: how high zero-overlap similarity gets

![Highest zero-overlap cosine against N](../../results/scale/figure2_zero_overlap_maximum.png)

**Figure 2.** The highest zero-overlap cosine across the 100-query panel at each N, one panel per D. Dots are seed means and the dark band is the seed range. The dashed line is the theoretical maximum reference for the actual number of zero-overlap comparisons. The shaded bands are the measured one-shared and two-shared means ± one standard deviation.

{maximum_md}

**Table 2.** N = 1,000,000; seed averages. The gap column is (k = 1 mean − highest zero-overlap score) / k = 1 standard deviation; a negative gap means the highest zero-overlap score lies above the one-shared mean.

At each D, the highest zero-overlap score:

{chr(10).join(f"- **D = {dims(f['d'])}:** {relation(f)}." for f in maxima)}

Beating a group's mean is a comparison with a typical score. It does not mean every record in that group was outranked. This study does not rank results, and it does not label one shared property a false match.

How the highest zero-overlap score grows with N (seed means, with the theoretical reference):

{growth_table(summary)}

The maximum reference is the smallest attainable cosine whose ideal tail probability is at most 1/M₀, where M₀ is the number of zero-overlap comparisons. Every measured maximum lies within {f3(max_error)} of it. Multiplying the comparisons by 100 raises the reference by {rises}.

### A back-of-envelope sizing rule

The normal approximation gives a rule simple enough to quote. The highest of M₀ near-independent chance scores sits about √(2 ln M₀) noise spreads above zero, where one spread is 1/√D. With M₀ = {whole(sizing["m0"])}, that is {f3(sizing["z"])} spreads. The one-shared mean sits μ₁√D spreads above zero, so the gap between them, in spreads, is roughly μ₁√D − √(2 ln M₀):

{sizing_md}

**Table 4.** The rule against the measured gap from Table 2. The rule is slightly pessimistic, mainly because the √(2 ln M) approximation overshoots the expected maximum of a normal sample at this M₀.

Solving for D: the highest chance score reaches the one-shared mean at D ≈ (√(2 ln M₀)/μ₁)² ≈ {whole(sizing["needed"][0])}. It sits three spreads below that mean at D ≈ ((√(2 ln M₀) + 3)/μ₁)² ≈ {whole(sizing["needed"][3])}. Both are consistent with the measurements: at D = {dims(first)} the measured maximum is above the one-shared mean, and at D = {dims(separated[0])} it sits {f3(by_d[separated[0]]['gap'])} one-shared spreads below it. Because √(2 ln M) grows so slowly, a hundred times more comparisons raise it by only about {growth:.1f}% here. The measured maximum rose by {min(rise_pct):.1f}–{max(rise_pct):.1f}% across D, between N = 10,000 and N = 1,000,000. The rule describes this fixture's five facts and uniform values; it is not a capacity estimate for other populations.

### Expected exceedances

Expected exceedance counts test the tail directly, without assuming independent comparisons. The table counts the zero-overlap comparisons at N = 1,000,000 scoring at least half the one-shared mean ({f3(MU[1] / 2)}) and at least the one-shared mean ({f3(MU[1])}). These are fixed score levels, not match thresholds.

{exceed_md}

**Table 3.** Expected counts are $M_0\\,p_0(s)$ from the binomial survival probability. Measured counts are seed means, with seed ranges.

{exceed_text}

## Where theory agrees and where it differs

- **Means and spreads agree.** Agreement is within {f3(mean_error)} for the means and within {pct(max(abs(1 - ratio.min()), abs(ratio.max() - 1)))} for the zero-overlap spread, at every D.
- **The binomial tail is an approximation for raw sums.** It is exact for ±1 records. A raw-sum zero-overlap score has the same mean (0) and spread (1/√D), but each coordinate's product ranges from −25 to 25, so its far tail is a little heavier than the binomial's. Table 3 shows this at D = 512: more zero-overlap comparisons reach the one-shared mean than the binomial model expects.
- **Maxima agree to within {f3(max_error)}.** The maximum reference is illustrative because comparisons are dependent: the 100 queries reuse the same value vectors, and so do the candidates. In practice it tracked the measured maximum closely.
- **Small offsets come from reused vectors.** At D = 512 the zero-overlap mean sits at {f3(row(summary, dimension=first, n=LARGEST, k=0)["mean_seed_mean"])} rather than 0. One fixed set of role and value vectors adds a small, seed-specific correlation to every comparison. The effect shrinks with D and is invisible at three decimals from D = 2,048.
- **Seed ranges are wider than sampling noise alone.** Seed-to-seed differences in the maximum (Table 1) reflect different vector sets, not just different draws. A single deployment has one vector set, so expect its maximum to land anywhere in a range like these.

## Encoder validation

Controlled pairs check the encoder before any population result. For each D and k, {whole(validation["pairs"][0])} random record pairs share exactly k properties and differ in the rest. They are encoded with seed 11's role and value vectors. Measured means are within {f3(validation_error)} of k/5, and zero-overlap spreads are {f3(val_ratio.min())}–{f3(val_ratio.max())} times 1/√D. Identical attribute records (k = 5) always score exactly 1. The [methodology](METHODOLOGY.md#encoder-validation) describes the check; `results/scale/encoder_validation.csv` holds the full precision.

{validation_table(validation)}

## Conditions and limits

The results describe five independent, uniformly distributed categorical properties, raw-sum bundling of exactly five facts and exact cosine. The tested grid is D from 512 to 10,000, N up to 1,000,000 and a fixed 100-query panel. Real records have skewed value frequencies and correlated properties, which change how often pairs share values. The per-fact cosine of {f3(MU[1])} also changes with the number of facts and the bundling method. The theoretical tails describe an ideal model, and agreement here does not validate them far beyond the measured range or define a universal capacity. An application still needs its own relevance rule or match threshold.

## Notes for a write-up

**Claims the results support, with their evidence:**
- Each shared property adds about {f3(MU[1])} cosine under five-fact raw-sum bundling, at every D (Table 1, Figure 1).
- Increasing N does not shift the typical score of any group; it only raises the highest chance score, slowly (Figure 2, growth table).
- Increasing D narrows every group by about 1/√D, which is what pushes the highest chance score down (Figures 1 and 2).
- At D = {dims(first)} and a million candidates, the highest zero-overlap score ({f3(by_d[first]["max"])}) is above the typical one-shared score ({f3(by_d[first]["one_mean"])}). From D = 4,096 it is at least {f3(min(by_d[d]["gap"] for d in separated))} one-shared spreads below it (Table 2).
- Simple probability theory predicted the means, spreads, maxima and tail counts closely (Tables 1–3). The measurements validate the theory over this measured range.
- Going from 8,192 to 10,000 dimensions buys about {pct(narrower_measured)} less noise: a real but tapering gain.

**Claims to avoid:**
- Do not call one-shared scores false matches, or zero-overlap scores errors; whether one shared property is relevant is an application decision.
- Do not say a one-shared record was "outranked". The comparison is with that group's typical score, not with every record in it.
- Do not extrapolate a record capacity, or say D = 4,096 "supports a million records" in general. The conclusions depend on five facts, uniform values and raw-sum bundling.
- Do not present the spreads in the figures as confidence intervals; they are the scatter of individual scores.

**Numbers worth quoting:** the per-fact step ({f3(MU[1])}), the noise floor (0.044 at D = 512, 0.010 at D = 10,000), the highest chance score at a million candidates ({f3(by_d[first]["max"])} at D = {dims(first)}, {f3(by_d[last]["max"])} at D = {dims(last)}), the theory's accuracy on that maximum (within {f3(max_error)}), and the run cost: about 100 million comparisons per setting, all 15 settings in {whole(scan_seconds)} seconds on a laptop CPU.

**Figures:** Figure 1 explains the score ladder and the role of D. Figure 2 carries the scale story. For a single figure, use Figure 2.

## Files

All in `results/scale/`:

| File | Contents |
|---|---|
| `per_query.parquet` | One row per (D, seed, N, query, k): float64 sums of cosines and their squares and the maximum cosine, plus mean, standard deviation and maximum cosine, and zero-overlap exceedance counts |
| `cell_summary.csv` | Pooled per (D, seed, N, k), with the theoretical mean, spread, maximum reference and expected exceedances |
| `seed_summary.csv` | Seed averages and ranges per (D, N, k) |
| `table_n1m.csv` | Table 1 at full precision |
| `encoder_validation.csv` | The controlled-pair check |
| `figure1_shared_properties.png/.svg`, `figure2_zero_overlap_maximum.png/.svg` | Figures 1 and 2 |
| `manifest.json` | Settings, seeds, library versions and timings |

Regenerate everything, including this report, with `sh src/scale/reproduce.sh` from the repository root.
"""
    REPORT_PATH.write_text(text)
