# Methodology: MAP similarity at million-record scale

This document defines the fixture, the encoder, the baseline and the two experiments. Settings are copied from the code (`person.py`, `experiment.py`, `ivfpq.py`). `results/scale/manifest.json` records versions, fixture hashes and stage timings. The findings are in the [report](REPORT.md).

## Why two experiments, and why these

The study answers one question: how does **unintended similarity** behave as the number of stored records and the dimension change, and what does an approximate index give up on top?

| # | Question |
| --- | --- |
| 1 | How high do unrelated records score by chance among 920,000 candidates, and how does D control it? |
| 2 | How much does a LanceDB IVF_PQ index lose against exact search? |

Every record is complete. Missing values were part of this study's first design, but their effect follows from which terms are encoded and how records are normalized, not from N. That work now has its own study in [`src/missingness`](../missingness/README.md).

## Fixture

The input is a controlled factorial `PERSON` fixture ([`data/scale`](../../data/scale/README.md)). It contains every combination of:
- 10 ordered age bands
- 8 job categories
- 5 home regions
- 3 distinct interests from 25

That's 920,000 integer signatures. They exhaust the declared state space and are not a sample of people. Rows are in a balanced order, so every nested prefix is balanced by age, job and region. Record IDs, names and a constant `entity_type` are not encoded; a constant term would add the same similarity to every pair.

## Encoder

MAP-I throughout:
- **Atoms:** bipolar vectors.
- **Binding ($\otimes$):** combine a role hypervector and a value hypervector, implemented as element-wise multiplication.
- **Bundling ($\oplus$):** combine the bound facts into $h_{\mathrm{record}}$, implemented as an arithmetic sum with no sign threshold, compared by cosine.
- **Roles:** four independent random vectors.
- **Values:** independent random vectors for the 8 jobs, 5 regions and 25 interests.
- **Age levels:** ordinal. A random start vector has D/2 of its coordinates split into nine near-equal parts, and each step up one band flips one more part. The intended similarity is `k_age(i, j) = 1 − |i − j| / 9`, so bands 1 and 10 are orthogonal. `encoder_diagnostics.csv` records the realized cosines, which differ from the intended ones by less than 5/D.
- **Seeds:** `torch.Generator().manual_seed(1000 × seed + offset)` per atom family.

The operators name the conceptual HDC operations. Their implementation in these experiments is ordinary coordinate arithmetic. For coordinate $j$,

$$
\begin{aligned}
[h_{\mathrm{role}} \otimes h_{\mathrm{value}}]_j &= [h_{\mathrm{role}}]_j [h_{\mathrm{value}}]_j, \\
[h_a \oplus h_b]_j &= [h_a]_j + [h_b]_j.
\end{aligned}
$$

The plus sign in the second equation adds two scalar coordinates; $\oplus$ denotes bundling the hypervectors. Stored bundles keep the unthresholded sum. Cosine comparison divides by the vector norms when scoring, rather than storing normalized vectors.

A complete record has six bound facts: one each for age, job and region, plus one for each of the three interests in the set $I$. Each fact is formed by binding its role to its value; all interests reuse the same interest role. We write

$$
\begin{aligned}
h_{\mathrm{record}} &= h_{\mathrm{age\,fact}} \oplus h_{\mathrm{job\,fact}} \oplus h_{\mathrm{region\,fact}} \\
&\quad \oplus \bigoplus_{i \in I} h_{\mathrm{interest\,fact},i}.
\end{aligned}
$$

For example, $h_{\mathrm{age\,fact}} = h_{\mathrm{age\,role}} \otimes h_{\mathrm{age},a}$ for age band $a$, and $h_{\mathrm{interest\,fact},i} = h_{\mathrm{interest\,role}} \otimes h_{\mathrm{interest},i}$ for interest $i$.

## Baselines

For any pair, the intended similarity is computed exactly from the integer records, without touching a hypervector.

**Exact similarity (`S_exact`)** is the cosine the encoder intends: the intended age-level overlap and no accidental overlap between independent random atoms. The plus signs below add scalar similarity contributions, rather than denoting hypervector bundling:

```text
[k_age + same job + same region + |shared interests|] / 6
```

Both records have six unit facts, so the denominator is 6.

**MAP similarity (`S_MAP`)** is $S_{\mathrm{MAP}} = (h_a \cdot h_b) / (\lVert h_a \rVert \, \lVert h_b \rVert)$ computed from the integer-valued sums. The dot products are exact in float32 (at most 36·D), so identical records tie exactly.

**MAP error** is `S_MAP − S_exact`: the noise from using finite, random vectors.

## Storage

Every atom, for every (D, seed), is stored in LanceDB (`experiment.lancedb`) as a fixed-size float16 list with an explicit PyArrow schema, a few MB in total. It is cast to float32 on read. The encoder builds its fact tables from the atoms it reads back.

Record bundles are encoded on the fly, in float32 and in bounded chunks, from those stored atoms. Only the table Experiment 2 indexes is materialized: complete records at D = 2,048, seed 11, about 3.8 GB.

Bundles are sums of six ±1 facts, so every coordinate is an integer in [−6, 6] and float16 stores it exactly. The write boundary refuses anything float16 can't hold exactly, and the tests check the round trip bit for bit. All computation is float32. The only float64 values are running sums over more than 10⁸ pairs.

## Experiment 1: Accidental similarity at million-record scale

**Grid:**
- D ∈ {512, 1,024, 2,048, 4,096, 8,192}
- seeds 11, 23 and 37
- N ∈ {100, 1,000, 10,000, 100,000, 920,000}

**Queries.** A 400-query evaluation panel, balanced by job and region, covering all ten age bands with 40% endpoint ages. Among complete records, only a band-1 or band-10 query has `S_exact = 0` candidates: the opposite endpoint age, a different job and region, and no shared interests. A query's own record is excluded.

**Search.** Exact cosine, streamed in chunks of 8,192 that never cross a prefix boundary. Results at smaller N are snapshots taken during the one 920,000-candidate pass.

**Measured:**
- `|S_MAP − S_exact|` median and p99 over all scored pairs
- each query's highest unrelated score
- the p99.9 of unrelated scores
- top-10 agreement with the exact ranking (ties handled: a retrieved record counts if its `S_exact` is at least the 10th best available)
- whether the top result is the best available
- the share of wrong-order pairs among the top 100

**Threshold contract, declared before the run:**
- **related** means `S_exact ≥ 2/3`
- **β = 0.01**: T is the largest 1e-4 grid value with at most 1% of related pairs scoring below it
- T is calibrated on a separate 200-query panel against the first 100,000 candidates, a balanced prefix. A pair's cosine doesn't depend on N, so the related-pair distribution is the same.
- T is frozen per (D, seed) and held fixed across N
- **ε = 0.01 per query**

At T the experiment reports:
- recall on the evaluation panel
- the share of queries with an unrelated candidate at or above T (one-sided)
- the share with `|S_MAP| ≥ T` (two-sided)

The threshold from an earlier back-of-envelope capacity estimate, T = 0.2157, is also evaluated, as a fixed reference. `thresholds.json` stores every T with the fixture hashes and settings, and the evaluation refuses to run if either has changed.

## Experiment 2: IVF_PQ against exact search

**Index.** The stored D = 2,048, seed-11 table, with a LanceDB `IVF_PQ` index:
- cosine distance
- `num_partitions` = round(√rows)
- `num_sub_vectors` = D / 16 = 128
- 8 bits
- LanceDB defaults for everything else, recorded in `ivfpq_index.json`

**Sweep.** `nprobes` ∈ {10, 20, 50, 100} × `refine_factor` ∈ {none, 10, 50}.

**Ground truth** is Experiment 1's exact top 10. Recall@10 handles ties: a returned record counts when its exact score, recomputed from the stored vectors, is at least the exact 10th score.

**Also reported:**
- related records missed
- unrelated records crossing T by the index's own score
- index-score error

Latency is logged for context only. Storage is exact, so any loss belongs to the index. IVF_PQ training has no fixed seed in this script; fresh index builds can produce slightly different recall and score errors. The saved measurements describe the recorded index run.

## Uncertainty and scope

The sampling unit is the query in both experiments. Seeds reuse the same queries, and records sharing values share bound terms, so pairs are not treated as independent trials:
- query-level rates are averaged over seeds per query, then **bootstrapped over queries** (2,000 resamples)
- events never observed are reported with a one-sided **Clopper–Pearson 95% upper bound**; for 400 queries that's about 0.75%, not zero
- seed ranges are shown next to means
- histogram quantiles use 1e-4 bins, rounded in the conservative direction; per-query maxima and threshold counts are exact

**Scope.** Results hold for this encoder, fixture, grid and query panel. 400 queries against 920,000 candidates is not the 4.2 × 10¹¹ pairs among all records, so no result is an all-pairs guarantee. Candidate-count effects are extreme-value pressure on each query, not records filling the space.

## Applicability of the earlier capacity estimate

An earlier back-of-envelope capacity estimate assumed a different encoder.

| | Earlier estimate | This study |
| --- | --- | --- |
| Record vector | $\operatorname{sign}\left(\bigoplus_f h_{\mathrm{fact},f}\right)$, bipolar | $\bigoplus_f h_{\mathrm{fact},f}$, compared by cosine |
| Record shape | five attributes | four fields, six facts |
| Unrelated-pair cosine | `1 − 2·Hamming/D`, an exact binomial for independent bipolar vectors | a normalized sum of cross-terms between atoms shared across records; spread near 1/√D, not binomial |
| Related-pair similarity | compressed by majority bundling (arcsine) | expected cosine is `S_exact` itself |
| Threshold example | T ≈ 0.2157 at D = 2,048, β = 0.001, ε = 0.01 | T calibrated for related pairs at `S_exact ≥ 2/3`; 0.2157 evaluated as a reference only |

Here $h_{\mathrm{fact},f} = h_{\mathrm{role},f} \otimes h_{\mathrm{value},f}$. The notation $\oplus$ denotes additive bundling in both columns; the earlier encoder applies a coordinate-wise sign to that sum afterwards, while this study retains the sum.

Two consequences follow:
1. The estimate's binomial tail does not describe these additive records.
2. The arcsine mapping does not define their related-pair baseline.

Auditing the estimate's 12.2-billion and 17.3-billion figures is left for future work.
