# Methodology: bundled categorical records at scale

This study has two parts. It shows what to expect when comparing bundled hypervectors against a growing population, with probability theory as the baseline. The results are in the [report](REPORT.md).

It replaces earlier scale experiments, which remain in git history.

## The question

> As the candidate population grows, how high do zero-overlap scores get, and how well do bundled cosine scores distinguish records sharing zero, one or two properties?

We compare the cosine similarity of two complete record hypervectors. Both query and candidate bundle the same five categorical properties. For example, a query and candidate may share their region while differing in education, occupation, interest cluster and employer. That one shared fact should contribute similarity.

The study measures how faithfully cosine reflects the number of shared properties as N and D change. It does not declare how many shared properties make a valid application match or whether two records describe the same person.

The questionnaire study looked inside additive bundles. This study keeps the same raw sums, so every fact can still be unbound, and looks at comparisons between records. The previous study does not establish that behaviour, so the encoder gets its own small validation.

Approximate indexes are out of scope. The study uses exact cosine comparisons so it measures the representation's behaviour directly.

## Group pairs by the number of shared properties

Count identical values only within the same property. Let k be the number of shared properties across the five facts.

| Group | Definition | What it tells us |
|---|---|---|
| **Zero shared properties** | k = 0 | similarity arising without a shared fact |
| **One shared property** | k = 1 | similarity contributed by one shared fact |
| **Two shared properties** | k = 2 | similarity contributed by two shared facts |
| **Three, four or five shared properties** | k = 3, 4 or 5, recorded separately | additional overlap, including identical attribute records at k = 5 |

Every pair belongs to exactly one k group, decided from the original categorical values independently of the measured hypervector cosine. The main figures focus on k = 0, 1 and 2. Zero overlap means no deliberately shared fact; its measured cosine can still differ from zero. One shared property is genuine overlap, not an error.

## Theory predicts, the experiment checks

The explanation needs three ideas:

1. **Zero-overlap bundles are almost orthogonal.** Their cosine fluctuates around zero, with a standard deviation of about 1/√D: 0.044 at 512 dimensions, 0.011 at 8,192 and 0.010 at 10,000.
2. **Shared facts raise the expected cosine.** For five-fact raw-sum bundles, one matching property gives an expected cosine of 1/5 = 0.2; two give 0.4. Increasing D narrows the fluctuations around these values.
3. **More comparisons give more opportunities for an unusually high score.** Even if the typical zero-overlap score stays near zero, its highest observed score can rise as the candidate population grows. Compare that maximum with the scores of records sharing one or two properties.

Theory establishes the expected separation and tail probabilities under an ideal random-vector model. Measurements show how well those predictions describe a population that reuses the same property and value vectors throughout.

### The equations

Each property has an independent random bipolar role vector, and each categorical value has an independent random bipolar value vector within that property. Binding uses $\otimes$:

$$
h_{\mathrm{fact},i} = h_{\mathrm{role},i} \otimes h_{\mathrm{value},i}.
$$

Bundling uses $\oplus$:

$$
h_{\mathrm{record}} = h_{\mathrm{fact},1} \oplus h_{\mathrm{fact},2} \oplus h_{\mathrm{fact},3} \oplus h_{\mathrm{fact},4} \oplus h_{\mathrm{fact},5}.
$$

Here binding is element-wise multiplication. Bundling is the coordinate-wise sum of all five facts, with no sign, so each coordinate is −5, −3, −1, 1, 3 or 5. Queries and candidates use this same encoder. Cosine divides the dot product by the product of the two records' lengths.

- **Expected cosine:** each shared fact meets itself in the dot product and adds D; every other product averages 0. Each record's length is about $\sqrt{5D}$, so $\mu_k = k/5$: 0, 0.2, 0.4, 0.6, 0.8 and 1 for k = 0–5.
- **Spread:** for zero overlap, $\sigma_0 = 1/\sqrt{D}$. Shared facts narrow the spread a little; those spreads are measured, not predicted.
- **Per-comparison tail probability (zero overlap):** model the score as $(2H - D)/D$ with $H \sim \operatorname{Binomial}(D, 1/2)$, and compute $p_0(s) = P(H \geq \lceil D(1 + s)/2 \rceil)$ from the binomial survival probability. The model is exact for ±1 records; for raw sums it has the right mean and spread, and its far tail is a little light. The normal approximation $p_k(s) \approx \Phi(-(s - \mu_k)/\sigma_k)$ explains the bell-curve intuition, but is not used to calculate extreme tails. A reference cosine is a score level, not an application match threshold.
- **Expected exceedances:** among $M_k$ comparisons in group k, the expected number scoring at least s is $M_k p_k(s)$. This connects the per-pair distribution to the scale of the population without defining false or missed matches.

For orientation, the expected cosine is independent of D, while the predicted standard deviation narrows as D increases:

| Shared properties | Expected cosine | Standard deviation at D = 512 | Standard deviation at D = 8,192 | Standard deviation at D = 10,000 |
|---|---|---|---|---|
| 0 | 0 | about 0.044 | about 0.011 | about 0.0100 |
| 1 | 0.2 | measured | measured | measured |
| 2 | 0.4 | measured | measured | measured |

These are theoretical predictions, not measured results. The implementation recomputes them from the formulas. Tail probabilities describe ideal limits; they do not establish a universal number of records that fit in D dimensions.

Expected exceedance counts do not require comparisons to be independent. Predicting the highest cosine does require an additional approximation: use the score quantile with tail probability about $1/M_0$ as a reference for the maximum across $M_0$ zero-overlap comparisons. Repeated queries and reused value vectors make those comparisons dependent, so the maximum reference is illustrative rather than a guarantee.

## Records

Each synthetic person has five categorical properties:

| Property | Values |
|---|---|
| Region | 20 |
| Education | 10 |
| Occupation | 100 |
| Interest cluster | 200 |
| Employer | 1,000 |

Employer supplies the fifth categorical property, preserving an odd number of facts. These names make the example readable; all values use independent random vectors, with no semantic similarity between different values.

Draw values uniformly and independently. This is a controlled synthetic population, not a model of real demographic frequencies or relationships between properties. Its four billion possible combinations allow duplicate records under independent sampling. Keep duplicates in the k = 5 group; exclude only a query's own record ID from its comparisons.

Generate one record sequence with a fixed data seed, 101. Reuse it for every dimension and vector seed. Roughly 84% of random pairs have zero overlap and 15% share exactly one property, giving both parts substantial comparison populations.

## Runs

Both parts use the same records and exact cosine comparisons. They summarize typical scores by shared-property count and the highest zero-overlap scores.

- D ∈ {512, 2,048, 4,096, 8,192, 10,000}.
- N ∈ {10,000, 100,000, 1,000,000}, as nested prefixes of the record sequence.
- Vector seeds 11, 23 and 37, each drawing a new set of role and value vectors.
- A fixed panel of 100 queries, the first 100 records, compared with every other record in each prefix.

The 10,000-dimensional setting shows the size of the final improvement beyond 8,192. Theory predicts about 9.5% narrower fluctuations at that step; compare the measurements to show how the gains taper as D increases.

There are Q(N − 1) directed query–candidate comparisons, with Q = 100: about 100 million at the largest prefix. This is a query-panel study, not a scan of all N(N − 1)/2 unordered population pairs. Use the actual group counts when calculating every theoretical expectation.

Candidates are encoded and compared in batches of 5,000, with summaries accumulated as the scan reaches each N. Each (D, vector seed) needs one pass through the million-record sequence, rather than a separate run for each prefix. The panel stays at 100 queries: seed-to-seed variation was small enough for a clear conclusion, so it was not expanded.

Measurements are saved per query and per seed, along with averages and seed ranges. Reporting is kept simple, with no bootstrap framework.

## Part 1: Does cosine reflect the number of shared properties?

For each k, N, D and vector seed, measure the comparison count, mean cosine and standard deviation. Compare these with $\mu_k$ and $\sigma_k$. Show k = 0, 1 and 2 in the main figure; retain all groups in the saved summaries.

This establishes what zero, one and two shared facts look like in bundled cosine scores, how much those scores fluctuate, and how D affects their separation. Growing N gives more observations; it is not expected to change the ideal mean or spread within a group.

## Part 2: How high does zero-overlap similarity get at scale?

Measure the highest zero-overlap cosine across the query panel at each N and compare it with the theoretical maximum reference based on the actual number of zero-overlap comparisons.

Alongside the maximum, show the measured mean and standard deviation for k = 1 and 2. This reveals whether the most extreme zero-overlap score approaches or exceeds the typical scores of records with shared facts.

Exceeding a group's mean is a comparison with a typical score, not proof that all records in that group were outranked. The study describes score separation; it does not add a ranking benchmark or label one-property overlap as a false match.

## Outputs

- **Figure 1:** mean cosine against shared-property count k = 0, 1 and 2 at N = 1,000,000, one panel per D. Show measured standard deviations and theoretical means and standard deviations. Label the spread explicitly; it is not a confidence interval.
- **Figure 2:** highest zero-overlap cosine against N, one panel per D, with its theoretical maximum reference and the measured means of k = 1 and 2, shaded by one standard deviation. Show seed ranges for the observed maximum.
- **One table:** at N = 1,000,000, D, comparison counts and predicted and measured means and standard deviations for k = 0, 1 and 2, plus the zero-overlap maximum. Show seed averages and ranges.
- **A short report:** explain the setup, the two figures, where theory agrees or differs, and what a practitioner can expect under these conditions.

The report also tabulates how the maximum grows with N, and the expected and measured exceedances. The encoder validation is reported as a table rather than a figure. There is no extrapolated record-capacity table or capacity solver.

## Encoder validation

Before any population result, controlled pairs check the encoder against the theory. For each D and each k = 0–5, 20,000 random record pairs share exactly k properties: the shared properties are chosen at random, and every other property takes a different value. The pairs are encoded with vector seed 11's role and value vectors. Values are reused across pairs, as they are in the population, so the check runs under realistic conditions, not the ideal independent-vector model.

Measured means fall within 0.001 of k/5, and zero-overlap spreads within about 1% of $1/\sqrt{D}$, at every D. Identical attribute records always score exactly 1. The deviations of the mean are a few times larger than independent sampling would allow. They share a sign within each D, because reusing one vector set adds a small, seed-specific correlation; the population results show the same offset. The full table is in the [report](REPORT.md#encoder-validation) and `results/scale/encoder_validation.csv`.

## Implementation

- **One run script.** `run.py` regenerates the records, runs the encoder validation, scans every (D, seed), and writes the summaries, figures, table and report. `study.py` holds the encoder, the theory and the scan; `report.py` writes the report from the saved summaries. `data/scale/generate.py` generates the records.
- **No stored hypervectors.** Role and value vectors are regenerated deterministically from the vector seed and D (`torch.Generator().manual_seed(seed * 1_000_003 + D)`, drawn with `torchhd.random`). Records are a Parquet file of categorical codes. LanceDB is not used: exact comparison against an in-memory batch needs no index, and the vectors take seconds to rebuild.
- **Arithmetic.** Bound facts are precomputed per property as int8, and a batch of records is encoded by lookup and an int8 sum, with no sign. Dot products are exact integers at these D; each cosine is computed in float64. Per query and k, the scan accumulates the cosines, their squares and their maximum in float64.
- **Deterministic seeds and a manifest.** `manifest.json` records settings, library versions and timings.
- **Not included.** The study has no approximate-index benchmark, ordinal encoding, majority-sign record bundles, field-similarity baseline, calibrated thresholds, score histograms or ranking diagnostics. Bit-packing is out of scope.

## Acceptance checks

These are implemented in `tests/test_study.py`, which runs, together with Ruff, before the full study.

- The encoder binds each categorical value to its own property role and sums exactly five facts with no sign, identically for queries and candidates. Unbinding a role from a record recovers its value.
- A small controlled-pair check compares measured means and spreads with the theoretical predictions for k = 0–5. Measured means agree with k/5.
- Hand-built records verify shared-property counts k = 0–5, duplicate handling and self-exclusion.
- A small end-to-end run agrees with direct brute-force cosines, group counts, means, standard deviations, maxima and prefix summaries.
- Theoretical means, spreads and binomial maximum references are computed from the formulas, respecting the discrete cosine values, rather than copied from the illustrative table.
- One script regenerates the records, measurements, table, figures and report. Run the focused checks and Ruff before the full study.

## What the results establish

The study measures how increasing N and D affects exact similarity comparisons for five categorical facts under raw-sum bundling, through one million candidates and about 100 million comparisons per setting.

Theory provides a baseline for typical similarities and extreme scores. Agreement in the measured range supports that baseline there; it does not validate its extreme tails or establish a universal storage capacity.

The conclusions depend on the number of facts, property frequencies and bundling method. An application would need to choose its own relevance rule or match threshold. State the measured conditions alongside the results. Other conditions are future questions, not extra experiments in this rework.
