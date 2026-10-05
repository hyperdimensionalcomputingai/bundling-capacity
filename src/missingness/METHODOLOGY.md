# Methodology: missing data and property normalization

This document defines the encodings, the missing-value policies, and the four experiments of [HYP-118](https://linear.app/hyperdimensionalcomputing/issue/HYP-118/research-blog-how-to-handle-missing-data-via-normalization). Settings are copied from the code (`person.py`, `policy.py`, `experiment1_weight.py` to `experiment4_build.py`); each experiment also writes its settings to `results/missingness/experiment<N>_config.json`. The findings are in the [report](REPORT.md).

## Records and inputs

Records come from the scale study's controlled factorial `PERSON` fixture ([`data/scale`](../../data/scale/README.md)): 920,000 records, one for every combination of 10 ordered age bands, 8 job categories, 5 home regions and 3 distinct interests from 25. Queries are the scale study's panels: 400 evaluation records and a separate 200-record calibration panel used only for fitting.

[`data/missingness/generate.py`](../../data/missingness/generate.py) adds, deterministically:

| File | Contents |
| --- | --- |
| `mcar.parquet` | One missing-completely-at-random (MCAR) mask per rate r ∈ {0.1, 0.3, 0.5}: age, job and region each missing with probability r, and each of the three interests independently with probability r. The interest property is missing only when all three are. |
| `duplicates.parquet` | A noisy duplicate of every query-panel record and of 20,000 training records. Each value is replaced by a different one with probability 0.1, the same corruption at every rate; each duplicate then gets its own MCAR missingness at each rate. |

These rates and the 10% error rate are design choices for this run, not tuned values.

## Two encodings

Each value is bound to its property's role, $h_{\mathrm{role},p} \otimes h_{p,v}$. Write $V_p$ for a record's known values of property $p$, $h_p = \bigoplus_{v \in V_p} h_{p,v}$ for their bundle, and $P$ for the properties the record knows.

| Encoding | Record |
| --- | --- |
| without normalization (the HYP-83 encoder, omit strategy) | $h_{\mathrm{record}} = \bigoplus_{p \in P} \bigoplus_{v \in V_p} h_{\mathrm{role},p} \otimes h_{p,v}$ |
| with per-property normalization | $h_{\mathrm{record}} = \bigoplus_{p \in P} h_{\mathrm{role},p} \otimes \big(h_p / \lVert h_p \rVert\big)$ |
| majority-sign normalization (Experiment 4) | $h_{\mathrm{record}} = \bigoplus_{p \in P} h_{\mathrm{role},p} \otimes \operatorname{sign}\big(h_p + \tfrac{1}{2} h_{\mathrm{tie},p}\big)$ |

Binding is element-wise multiplication. Bundling is computed as an arithmetic sum with no sign threshold; normalization is a separate step applied to one property's bundle before it is bound. Binding a bipolar role preserves norms, so `person.encode_normalized` divides each property's bound facts by their L2 norm. In the sign encoding, half a fixed random tie-break vector $h_{\mathrm{tie},p}$ settles the zero coordinates of an even-sized bundle and never changes an odd one. A property with no known value contributes nothing in every encoding.

All atoms are bipolar MAP vectors seeded as in the scale study (`torch.Generator().manual_seed(1000 × seed + offset)`); tie-break atoms use offset 8. Computation is float32.

## Exact similarity and missing-value policies

With orthogonal atoms, the dot product of two normalized bundles is a sum of per-property similarities over the properties both records know. The plus signs below add scalar contributions, not hypervectors:

```text
S(q, c) = s_age + s_job + s_region + s_interests       (each only if both records know it)

s_age        = 1 - |i - j| / 9                          the intended ordinal age kernel
s_job        = [same job]
s_region     = [same region]
s_interests  = |shared| / sqrt(n_q * n_c)              Ochiai (set-cosine) over known interests
```

A missing-value policy is a choice of denominator for that numerator (`policy.py`). With $p_q$ and $p_c$ the numbers of known properties:

| Policy | Score | A missing property counts as | Source |
| --- | --- | --- | --- |
| Gower | S / (properties both know) | absent: fully neutral | Gower (1971) |
| cosine | S / √(p_q · p_c) | partly neutral | what normalized-bundle cosine gives |
| mismatch | S / p_q | a mismatch | |
| pivoted | S / (√p_q · ((1 − s) · pivot + s · √p_c)) | tunable by slope s | Singhal, Buckley & Mitra (1996) |

The pivot is the pool's mean √p_c, Singhal's "average old normalizer". Slope 1 is cosine; slope 0 ranks like mismatch; slopes above 1 move toward neutral.

Experiments 2 and 3 use these exact scores, so they measure the policies themselves. Experiment 4 builds them from MAP vectors.

## Ties

Exact scores are ratios of small integers, so ties are common and are handled in expectation: scores within 1e-6 tie, and a record tied with e others at a cut is placed uniformly among them. Recall@k of one relevant record is clamp((k − greater) / (equal + 1), 0, 1); reciprocal rank is averaged over the tied positions; a top k's coverage mix and relevant count give tied records an equal share of the remaining slots.

## Experiment 1: Who sets the weight?

**Part A, ranking.** One query (age band, job, region, three interests) and hand-built candidates: A_k matches age, job and region and knows k ∈ {0, 1, 2, 3} of the query's interests; B is complete, matches age, region and all three interests, and has a different job. Both encodings at D = 2,048, seeds 1–100, seed 11 as the worked example; the plan is [EXPERIMENT_UPDATE.md](EXPERIMENT_UPDATE.md).

| Encoding | A_k | B |
| --- | --- | --- |
| without normalization | √((3 + k) / 6) | 5/6 |
| with normalization | (3 + √(k/3)) / (2√(3 + [k > 0])) | 3/4 |

**Part B, influence.** For 500 random fixture records per seed (seeds 1–20), a query copy that knows its first n_q interests is compared with a copy of the same person that knows its first n_c (age, job and region known on both). Every known value agrees, so the cosine splits exactly into per-property contributions $h_{p}(q) \cdot h_{p}(c) / (\lVert h_q \rVert \lVert h_c \rVert)$, and the interests' share of the match is measured next to its exact value: shared / (3 + shared) without normalization and x / (3 + x) with it, where shared = min(n_q, n_c) and x = shared / √(n_q n_c).

## Experiment 2: The missing-value dial

**Search.** 400 complete queries against a pool masked at rate r, at three pool densities: all 920,000 records, and fixed random 10% and 1% subsamples (query-panel records always kept). The factorial fixture contains every near-variant of every record, which real populations don't; the subsamples test whether conclusions depend on that density.

**Hidden-truth relevance.** Each pool record also has its complete version. Relevance is the equal-weight property similarity of the complete records, S / 4, and the relevant set is each query's top 10 by it, ties included. Precision@10 is the expected share of a policy's top 10 (scored on masked data) that is relevant. Because masking is MCAR, relevance is independent of coverage: Singhal's diagnostic compares the coverage mix of the relevant set with the mix each policy retrieves.

**Own copy.** Each query person's own masked record, ranked against everyone else: Recall@1, Recall@10 (random reference 10 / pool size) and MRR.

**Fitting the pivot.** Per (rate, density), slopes 0.0–1.6 in steps of 0.1 are scored by precision@10 on the calibration panel. Slopes within 0.001 of the best count as ties and resolve toward cosine; the chosen slope is then fixed for the evaluation panel. Steeper slopes approach the point where a one-property record's normalizer reaches zero.

**Grid.** r ∈ {0.1, 0.3, 0.5} × density ∈ {100%, 10%, 1%}. Bootstrap intervals use 2,000 resamples over evaluation queries.

## Experiment 3: Weights from informativeness

**Task.** Find each query person's noisy duplicate among everyone else. Queries are complete; duplicates have 10% wrong values and MCAR missingness at rate r; distractors are every other fixture record at the same rate; the query's original record is excluded.

**Comparison levels** (Splink style; −1 when either side is missing):

| Property | Levels |
| --- | --- |
| age | same band · one band apart · further |
| job, region | agree · disagree |
| interests | all comparable interests shared (shared = min(n_q, n_c)) · some shared · none |

**Weights.** Fellegi–Sunter weights are log(m / u) per level, and a property either side is missing scores 0.
- u: level frequencies among random pairs (calibration queries against 5,000 random pool records). True matches are a negligible share of random pairs, so no labels are needed.
- m, labelled ("known"): level frequencies between 20,000 training records and their duplicates.
- m, EM: estimated without labels, with u held fixed and conditional independence across properties. Each blocking rule fixes one property and EM estimates m for the others, whose non-match levels stay random under the rule: same region gives m for age, job and interests; same job gives m for region. Up to 500 random blocked pool records per calibration query, plus the query's duplicate when the rule admits it. Blocking on overall similarity instead makes every blocked pair look like a match and EM collapses to "all pairs match".
- m, assumed ("u-only"): agreement 0.9 (about one minus the error rate), the rest of m spread like u. Like IDF, the weight then comes from how common an agreement is by chance. This uses our known error rate, so it is optimistic.

Add-half smoothing keeps every level probability above zero.

**Scorers.** cosine (equal weights); Fellegi–Sunter with labelled, EM and u-only m; linear Fellegi–Sunter, which collapses the u-only weights to agree (a_p) and disagree (d_p) and scores Σ over known-by-both of d_p + (a_p − d_p) · s_p; and weighted cosine, Σ a_p · s_p / √(p_q p_c). With a complete query, linear Fellegi–Sunter is a query-weighted dot product plus a per-record bias that depends only on which properties the candidate knows, so one stored coordinate serves it. Weighted cosine needs only the query's property terms scaled by a_p.

**Measures.** Recall@1, Recall@10 and MRR of the duplicate (pool size 920,000 or about 9,800; random Recall@10 is 10 / pool size), and the average precision of "the top result is the duplicate" decisions ranked by top-1 score. **Grid:** r ∈ {0.1, 0.3, 0.5} × the full pool and the 1% subsample.

## Experiment 4: Building it with MAP vectors in LanceDB

**Pool.** The first 100,000 fixture records (a balanced prefix) at 30% MCAR missingness, minus the 61 records with nothing known: about the density of Experiment 2's 10% pool. Queries: the 400 complete evaluation records. D = 2,048, seed 11.

**Tables.** Each is a LanceDB table with an explicit PyArrow schema: record index, a coverage count (known properties), the stored value codes, and a fixed-size vector column.

| Table | Stored vector | Search |
| --- | --- | --- |
| cosine_f32, cosine_f16 | L2-normalized bundle scaled to unit length | dot |
| mismatch_f16 | L2-normalized bundle, unscaled | dot |
| pivoted_f16 | L2-normalized bundle divided by (1 − s) · pivot + s · √coverage, s = 1.2 | dot |
| sign_f16 | majority-sign bundle (integers, exact in float16) | cosine |

Query vectors are unit-length normalized bundles, or the sign bundle for sign_f16. The weighted search reuses cosine_f16 with a query whose property terms are scaled by the u-only agreement weights. Gower is applied as a reranker of cosine_f16's top 100, from the stored codes. Searches are flat (no index), so differences from the in-memory float32 ranking come from storage precision; LanceDB's dot distance is 1 − dot.

**Measures.** Agreement with the exact policy's top 10 (the share of returned records whose exact score reaches the exact 10th best, ties included); precision@10 against hidden-truth relevance, next to the exact policy's own precision on the same pool; float16 against float32 (top-10 overlap, identical top-10 share, largest rank-aligned score difference); storage size; median latency of a single-threaded Python loop, for context only. A separate table measures the interests property's similarity under L2 and majority-sign normalization when a three-interest query meets a candidate that knows 1, 2 or 3 of them (seeds 1–20, 200 records each).

## Uncertainty and scope

The sampling unit is the evaluation query. Experiment 2 reports 95% bootstrap intervals over queries; elsewhere spreads are over seeds (Experiment 1) or are point estimates on 400 queries. Pool subsamples, MCAR masks and duplicates are fixed and seeded.

Results hold for this encoder, fixture and grid. The fixture is a controlled factorial space with four properties and low-cardinality values; MCAR is the friendliest missingness mechanism; Experiment 3's error model is uniform. Real records with more fields, skewed values or informative missingness can move every number here.
