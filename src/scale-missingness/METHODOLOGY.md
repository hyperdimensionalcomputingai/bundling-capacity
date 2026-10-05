# Methodology: MAP similarity under missingness and at million-record scale

This document defines the fixture, the encoder, the baselines and the four experiments. The design is recorded on [HYP-83](https://linear.app/hyperdimensionalcomputing/issue/HYP-83/research-blog-test-map-similarity-under-missingness-and-million-record). Settings are copied from the code (`person.py`, `experiment.py`, `pairwise.py`, `ivfpq.py`). `results/scale-missingness/manifest.json` records versions, fixture hashes and stage timings. The findings are in the [report](REPORT.md).

## Why four experiments, and why these

The study answers two questions:
1. When some of a Person's fields have no value, what happens to **pairwise** similarity, and is `"null"` a better null than `"no_value"` or `""`?
2. How does **unintended similarity** behave as the number of stored records and the dimension change?

Each experiment answers one question, and no measurement is repeated across experiments:

| # | Question | Needs scale? |
| --- | --- | --- |
| 1 | How high do unrelated records score by chance among 920,000 candidates, and how does D control it? | yes |
| 2 | What does a missing value do to one comparison, and does the null's spelling matter? | no, pairwise |
| 3 | Does Experiment 2's pairwise effect produce wrong results in a million-record search? | yes, one cell |
| 4 | How much does a LanceDB IVF_PQ index lose against exact search? | yes, one index |

Missingness is **not** swept across rates, dimensions and candidate counts. Its pairwise effect follows exactly from which terms are encoded, and doesn't depend on N. MAP adds noise of about 1/√D on top of that, whatever the missing-value strategy. Experiment 2 measures the effect directly; Experiment 3 checks it once at scale.

## Fixture

The input is a controlled factorial `PERSON` fixture ([`data/scale-missingness`](../../data/scale-missingness/README.md)). It contains every combination of:
- 10 ordered age bands
- 8 job categories
- 5 home regions
- 3 distinct interests from 25

That's 920,000 integer signatures. They exhaust the declared state space and are not a sample of people. Rows are in a balanced order, so every nested prefix is balanced by age, job and region. Record IDs, names and a constant `entity_type` are not encoded; a constant term would add the same similarity to every pair.

## Encoder

MAP-I throughout:
- **Atoms:** bipolar vectors.
- **Binding:** element-wise multiplication of role and value.
- **Record:** the sum of its bound facts, with no sign threshold, compared by cosine.
- **Roles:** four independent random vectors.
- **Values:** independent random vectors for the 8 jobs, 5 regions and 25 interests.
- **Age levels:** ordinal. A random start vector has D/2 of its coordinates split into nine near-equal parts, and each step up one band flips one more part. The intended similarity is `k_age(i, j) = 1 − |i − j| / 9`, so bands 1 and 10 are orthogonal. `encoder_diagnostics.csv` records the realized cosines, which differ from the intended ones by less than 5/D.
- **Seeds:** `torch.Generator().manual_seed(1000 × seed + offset)` per atom family.

A complete record has six facts:

```text
AGE_ROLE ⊙ AGE[age] + JOB_ROLE ⊙ JOB[job] + REGION_ROLE ⊙ REGION[region] + Σ INTEREST_ROLE ⊙ INTEREST[i]
```

### Two ways to encode a missing field

| Strategy | A missing field contributes |
| --- | --- |
| **omit** | nothing |
| **token** | `ROLE ⊙ NULL[spelling]`: one shared vector per spelling (`"null"`, `"no_value"`, `""`), each an independent random atom |

A null spelling is simply another symbol, so every spelling is a token. What changes the result is whether a token is used at all, and whether two records use the same one. A missing interest set counts as **one** token fact. Three copies of one vector would add at full strength, so two records with unknown interests would share 9 of 12 squared-norm units from missingness alone.

In practice a pipeline maps absent keys, `None`, `""` and textual markers either to omission or to a token. This study measures both outcomes rather than prescribing a normalizer.

## Baselines

For any pair, two similarities are computed exactly from the integer records, without touching a hypervector:

**Content similarity (`S_content`)** counts only what both records actually know:

```text
[k_age + same job + same region + |shared interests|]  (each only if both know the field)
  / sqrt(w_A · w_B)
```

`w` is a record's number of content facts. For two complete records the denominator is 6. **Shared missingness never counts as similarity.**

**Encoder similarity (`S_encoder`)** is the exact cosine of the terms actually encoded, if all atoms were orthogonal. With tokens, each field both records mark missing with the **same spelling** adds 1 to the numerator, and each token counts as a term. Under omit, `S_encoder = S_content`.

**MAP similarity (`S_MAP`)** is `a · b / (‖a‖ ‖b‖)` computed from the integer-valued sums. The dot products are exact in float32 (at most 36·D), so identical records tie exactly.

Every result keeps two effects separate:
- **MAP error:** `S_MAP − S_encoder`, the finite-dimensional noise
- **Semantic shift:** `S_encoder − S_content`, the similarity the null strategy adds

## Storage

Every atom, for every (D, seed), is stored in LanceDB (`experiment.lancedb`) as a fixed-size float16 list with an explicit PyArrow schema, a few MB in total. It is cast to float32 on read. The encoder builds its fact tables from the atoms it reads back.

Record bundles are encoded on the fly, in float32 and in bounded chunks, from those stored atoms. Only the table Experiment 4 indexes is materialized: complete records at D = 2,048, seed 11, about 3.8 GB.

Bundles are sums of six ±1 facts, so every coordinate is an integer in [−6, 6] and float16 stores it exactly. The write boundary refuses anything float16 can't hold exactly, and the tests check the round trip bit for bit. All computation is float32. The only float64 values are running sums over more than 10⁸ pairs.

## Experiment 1: Accidental similarity at million-record scale

**Grid:**
- complete records only
- D ∈ {512, 1,024, 2,048, 4,096, 8,192}
- seeds 11, 23 and 37
- N ∈ {100, 1,000, 10,000, 100,000, 920,000}

**Queries.** A 400-query evaluation panel, balanced by job and region, covering all ten age bands with 40% endpoint ages. Among complete records, only a band-1 or band-10 query has `S_content = 0` candidates: the opposite endpoint age, a different job and region, and no shared interests. A query's own record is excluded.

**Search.** Exact cosine, streamed in chunks of 8,192 that never cross a prefix boundary. Results at smaller N are snapshots taken during the one 920,000-candidate pass.

**Measured:**
- `|S_MAP − S_encoder|` median and p99 over all scored pairs
- each query's highest unrelated score
- the p99.9 of unrelated scores
- top-10 agreement with the source ranking (ties handled: a retrieved record counts if its `S_content` is at least the 10th best available)
- whether the top result is the best available
- the share of wrong-order pairs among the top 100

**Threshold contract, declared before the run:**
- **related** means `S_content ≥ 2/3`
- **β = 0.01**: T is the largest 1e-4 grid value with at most 1% of related pairs scoring below it
- T is calibrated on a separate 200-query panel against the first 100,000 candidates, a balanced prefix. A pair's cosine doesn't depend on N, so the related-pair distribution is the same.
- T is frozen per (D, seed, condition) and held fixed across N
- **ε = 0.01 per query**

At T the experiment reports:
- recall on the evaluation panel
- the share of queries with an unrelated candidate at or above T (one-sided)
- the share with `|S_MAP| ≥ T` (two-sided)

The earlier chat's T = 0.2157 is also evaluated, as a fixed reference. `thresholds.json` stores every T with the fixture hashes and settings, and the evaluation refuses to run if either has changed.

## Experiment 2: One comparison with missing fields

**Pairs.** 2,000 fixed pairs of each type (`pairwise_pairs.parquet`):
- **same person:** a complete record against itself with k fields removed. This measures **dilution**.
- **unrelated people:** two records with `S_content = 0` (age bands 1 and 10, different job and region, disjoint interests), both with the same k fields removed. This measures **false similarity**. Their content similarity stays 0 at every k.

**Missing fields.** k ∈ {0, 1, 2, 3} of the scalar fields (age, job, region), removed in each pair's fixed random order. Interests are always kept, so a complete record has 6 facts and the expected curves are simple.

**Strategies:**
- omit
- a token spelled `"null"`, `"no_value"` or `""`
- **mixed spellings**, where each record's null is spelled independently from those three

**Grid:** D ∈ {512, 2,048, 8,192}, seeds 11, 23 and 37.

**Expected values** (the `S_encoder` curves, for orthogonal atoms):

| Pair | omit | shared token | mixed spellings |
| --- | --- | --- | --- |
| same person, k missing on one side | √((6−k)/6) | (6−k)/6 | (6−k)/6 |
| unrelated, both missing the same k | 0 | k/6 | about k/18 on average |

MAP cosines are reported next to these values, with the 5th–95th percentile spread.

## Experiment 3: Does it matter in search?

**Setup.** One fixed mask (`missingness.parquet`) marks each field of each record missing independently with probability 0.3, for queries and candidates alike. The two strategies are compared:
- omit
- a shared `"null"` token

Both run at D = 2,048 with seeds 11, 23 and 37, against all 920,000 candidates. They use Experiment 1's panels, tiers and threshold rule, calibrated separately for each strategy. A complete-record run at the same D is the reference. Records with no remaining content (0.3⁴ ≈ 0.8% of records) are excluded under both strategies.

**Measured:**
- the mean `S_encoder − S_content` of unrelated pairs
- the share of queries with an unrelated record above T
- top-10 agreement and top-1 change against `S_content`, and against `S_encoder`, which isolates MAP noise from the strategy's semantic shift

## Experiment 4: IVF_PQ against exact search

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

The sampling unit is the query in Experiments 1, 3 and 4, and the pair in Experiment 2. Seeds reuse the same queries and pairs, and records sharing values share bound terms, so pairs are not treated as independent trials:
- query-level rates are averaged over seeds per query, then **bootstrapped over queries** (2,000 resamples)
- events never observed are reported with a one-sided **Clopper–Pearson 95% upper bound**; for 400 queries that's about 0.75%, not zero
- seed ranges are shown next to means
- histogram quantiles use 1e-4 bins, rounded in the conservative direction; per-query maxima and threshold counts are exact

**Scope.** Results hold for this encoder, fixture, grid and query panel. 400 queries against 920,000 candidates is not the 4.2 × 10¹¹ pairs among all records, so no result is an all-pairs guarantee. Candidate-count effects are extreme-value pressure on each query, not records filling the space.

## Applicability of the earlier chat's model

The [earlier capacity discussion](https://chatgpt.com/s/cx_6a97c2dbe6d0819199ddece9e394882b) assumed a different encoder.

| | Earlier chat | This study |
| --- | --- | --- |
| Record vector | `sign(Σ role ⊙ value)`, bipolar | `Σ role ⊙ value`, compared by cosine |
| Record shape | five attributes | four fields, six facts when complete |
| Unrelated-pair cosine | `1 − 2·Hamming/D`, an exact binomial for independent bipolar vectors | a normalized sum of cross-terms between atoms shared across records; spread near 1/√D, not binomial |
| Related-pair similarity | compressed by majority bundling (arcsine) | expected cosine is `S_encoder` itself |
| Threshold example | T ≈ 0.2157 at D = 2,048, β = 0.001, ε = 0.01 | T calibrated for related pairs at `S_content ≥ 2/3`; 0.2157 evaluated as a reference only |

Three consequences follow:
1. The chat's binomial tail does not describe these additive records.
2. The arcsine mapping does not define their related-pair baseline.
3. Dividing a population estimate by the number of attributes is not meaningful when missingness changes the number of facts.

Auditing the 12.2-billion and 17.3-billion figures is Phase 2 of HYP-83, after review.
