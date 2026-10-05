# Missing data and normalization (parked starting material)

This directory holds the missing-value work split out of the HYP-83 [scale study](../scale/README.md). It is the starting point for [HYP-118](https://linear.app/hyperdimensionalcomputing/issue/HYP-118/research-blog-how-to-handle-missing-data-via-normalization), a separate research post on handling missing data via normalization. The HYP-118 experiments are not built yet; what is here runs, but it is an earlier design, not the new suite.

| File | What it is |
| --- | --- |
| `EXPERIMENT_UPDATE.md` | The plan for the property-normalization comparison, written after David's feedback |
| `normalization.py` | Prototype of HYP-118 Experiment 1: the A_k / B comparison with and without per-property normalization, D = 2,048, seeds 1–100 |
| `pairwise.py` | HYP-83's pairwise experiment: omitting a missing field against a null token (`"null"`, `"no_value"`, `""`, mixed spellings) |
| `person.py` | Records with missing values, the MAP-I encoder (omit or token), `encode_normalized`, and the exact baselines |
| `charts.py` | The two figures below |
| `tests/` | Orthogonal-atom exactness with and without normalization, strategy equivalences and the plan's predictions |

HYP-83's retrieval check at 30% missingness (omit against a shared token, among 920,000 records) is not parked as code: it was woven into the scale study's streamed search. The code is at commit `450c4e6` and its saved tables are in `results/missingness/`.

Binding uses $\otimes$ and bundling uses $\oplus$. Without normalization a record is $\bigoplus_{p} \bigoplus_{v \in V_p} h_{\mathrm{role},p} \otimes h_{p,v}$; with per-property normalization it is $\bigoplus_{p} h_{\mathrm{role},p} \otimes (h_p / \lVert h_p \rVert)$, where $h_p = \bigoplus_{v \in V_p} h_{p,v}$. Bundling is computed as an arithmetic sum with no sign threshold; normalization is the separate division by the L2 norm.

## Reproduce

From the repository root, after the scale fixture exists in `data/scale/`:

```sh
sh src/missingness/reproduce.sh
```

Atoms are regenerated from their seeds with `person.codebook`, so no LanceDB store is needed. They are identical to the atoms the scale study stores in LanceDB, which `src/scale/tests` checks.

## Saved evidence (`results/missingness/`)

| File | Contents |
| --- | --- |
| `normalization.parquet`, `normalization_summary.csv`, `normalization_rankings.csv`, `normalization_config.json`, `experiment5_normalization.png` | The normalization prototype: every (seed, encoding, candidate) cosine next to its exact value, summaries, how often each predicted ranking holds, settings, figure |
| `pairwise.parquet`, `pairwise_summary.csv`, `experiment2_pairwise.png` | HYP-83's pairwise experiment |
| `experiment3_missingness_retrieval.csv`, `semantic_shift.csv` | HYP-83's 30%-missing retrieval check (saved tables only) |

File names keep their HYP-83 experiment numbers until HYP-118 settles its own.

## Prototype methodology: property normalization

The plan, with the reasoning for each choice, is [`EXPERIMENT_UPDATE.md`](EXPERIMENT_UPDATE.md).

**Why.** The current encoder gives each known value one bound fact. Age, job and region each contribute one fact, but a record with three interests contributes three. So the number of listed interests, in the query and in each candidate, sets how much the interests property counts. Cosine divides by each record's overall norm, but that doesn't give every property equal influence.

**Two encodings from the same atoms.** Write $h_p$ for the bundle of property $p$'s known values, $h_p = \bigoplus_{v \in V_p} h_{p,v}$, and $P$ for the record's observed properties.

| Encoding | Record |
| --- | --- |
| **without normalization** (the current encoder, omit strategy) | $\bigoplus_{p \in P} \bigoplus_{v \in V_p} h_{\mathrm{role},p} \otimes h_{p,v}$ |
| **with normalization** | $\bigoplus_{p \in P} h_{\mathrm{role},p} \otimes \big(h_p / \lVert h_p \rVert\big)$ |

In both, bundling is computed as an arithmetic sum with no sign threshold. Normalization is a separate step, dividing a property's bundle by its L2 norm before binding. That preserves the property's direction and gives every observed property a unit-length term. Binding a bipolar role preserves norms, so `encode_normalized` divides each property's bound facts from `encode_sum` by their norm. A property with no known value contributes nothing.

**Records.** These are built directly; the fixture isn't used. The query has an age band, a job, a region and three interests. Candidate **A_k** matches the query on age, job and region and knows the first k ∈ {0, 1, 2, 3} of its interests; the rest are omitted, not encoded as mismatches. Candidate **B** is complete and matches age, region and all three interests, but has a different job. The values themselves are arbitrary: each compared value either matches exactly or comes from an independent atom.

**Expected values** (orthogonal atoms; `normalized_baseline` for the normalized encoding):

| Encoding | A_k | B |
| --- | --- | --- |
| without normalization | √((3 + k) / 6) | 5/6 |
| with normalization | (3 + √(k/3)) / (2√(3 + [k > 0])) | 3/4 |

Without normalization, the query's interests make up half its weight (3 of 6 terms); with normalization, a quarter. With normalization, A_k's interests property earns √(k/3) of the interests credit, so partial knowledge earns partial credit. An unknown property adds nothing to the dot product but also shortens the record, so it costs less than a mismatched property, which adds nothing to the dot product and still contributes to the norm.

**Grid.** D = 2,048, seeds 1–100 with seed 11 as the worked example, float32. Atoms come from `person.codebook`, seeded as in the scale study's LanceDB store, which holds only seeds 11, 23 and 37. For each encoding and each k, `normalization_rankings.csv` reports the predicted winner of A_k versus B, the measured gap, and the share of seeds in which the measured ranking agrees with the prediction.
