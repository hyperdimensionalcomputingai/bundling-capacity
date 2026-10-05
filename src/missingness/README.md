# Missing data and normalization

This directory implements [HYP-118](https://linear.app/hyperdimensionalcomputing/issue/HYP-118/research-blog-how-to-handle-missing-data-via-normalization): how to handle missing values when records are hypervector bundles. The [scale study](../scale/README.md) asks how a complete record behaves among a million others; this one asks what to do when records are incomplete. The answer turns out to be about **normalization**, and it lines up with what statistics, information retrieval and record linkage worked out decades ago.

- **[Report](REPORT.md):** the four experiments, what we found, and the takeaways.
- **[Methodology](METHODOLOGY.md):** encodings, missing-value policies, relevance, ties, and each experiment's design.
- **[Inputs](../../data/missingness/README.md):** missingness masks and noisy duplicates built on the scale study's fixture.

## What this means for HDC practitioners

If you encode records as hypervectors (customer profiles, patient records, product listings) some fields will be empty. You don't need the experiments to use what they found:

1. **Normalize each field before you bundle it.** If one person lists three interests and another lists one, a plain bundle lets the first person's interests count three times as much. Bundle a field's values, scale that field to unit length, then bind it to its role. Now every field you know counts the same, whatever it holds.
2. **Leave missing values out, and compare with cosine.** Don't invent a "missing" vector, and don't treat a blank as a perfect match either. Cosine on normalized fields charges a record a little for what it doesn't know, and that turned out to be the right amount in almost every setting we tried. Treating blanks as neutral let near-empty records take most of the top 10.
3. **Give rare agreements more weight.** Two people sharing a region (1 in 5 by chance) is weak evidence; sharing three interests (about 1 in 37) is strong. Weight each field by how surprising an agreement is, which you can estimate from random pairs of your own records without labels. Put the weights on the *query*, not the stored records: one index then serves any weighting.
4. **Store unit-length vectors in float16 and search by cosine.** Normalized vectors lose nothing measurable in float16, which halves storage. A dot product over unit-length vectors is the same thing as cosine. Only if you want a different length normalization ("missing is a mismatch", or a tuned pivot) do you scale each record by it when you write it and search by raw dot product, because a cosine metric would divide that scaling back out.
5. **Keep partial credit.** Thresholding each field to ±1 keeps vectors compact but can't tell one shared interest from two. Use L2 normalization when partial matches matter.
6. **Show coverage next to the score.** A 0.87 from a record that knows two of four fields and a 0.87 from a complete record are different claims. Store how many fields each record knows; it lets you report, filter or rerank on completeness.

**What this doesn't settle:** if missingness *means* something (people who skip "income" differ from those who don't), absence is evidence, and an explicit per-field "missing" token can be the right tool. That trade-off is worth its own section in a practitioner post.

### Ideas for the "what's possible" post

- **Search with your own missing-value policy.** Cosine, mismatch and tuned middle grounds are a write-time scaling choice; neutral (Gower) is a rerank step over the top results.
- **One index, many weightings.** Different users, tasks or queries can bring their own field weights at query time.
- **Label-free deduplication.** Weights from random pairs, plus normalized bundles, give a fixed-size vector per record and a matching score that needs no training data.
- **Auditable results.** Coverage beside every score shows how much of a match is evidence and how much is absence.

## The four experiments

| # | Question | Script | Main output |
| --- | --- | --- | --- |
| 1 | Who sets a property's weight: the schema, or how many values a record lists? | `experiment1_weight.py` | `experiment1_ranking.png`, `experiment1_influence.png` |
| 2 | What should a missing property cost in search: neutral (Gower), partly (cosine), a mismatch, or a fitted pivot (Singhal)? | `experiment2_dial.py` | `experiment2_summary.csv`, `experiment2_dial.png` |
| 3 | Do informativeness weights (Fellegi–Sunter) find noisy duplicates better than equal weights? | `experiment3_linkage.py` | `experiment3_summary.csv`, `experiment3_linkage.png` |
| 4 | Do the policies survive real MAP vectors, float16 storage and a single dot product in LanceDB? | `experiment4_build.py` | `experiment4_summary.csv`, `experiment4_build.png` |

Binding uses $\otimes$ and bundling uses $\oplus$. Without normalization a record is $\bigoplus_{p} \bigoplus_{v \in V_p} h_{\mathrm{role},p} \otimes h_{p,v}$; with per-property normalization it is $\bigoplus_{p} h_{\mathrm{role},p} \otimes (h_p / \lVert h_p \rVert)$, where $h_p = \bigoplus_{v \in V_p} h_{p,v}$. Bundling is computed as an arithmetic sum with no sign threshold; normalization is the separate division by the L2 norm. Experiments 2 and 3 score policies exactly (as if atoms were orthogonal); Experiments 1 and 4 use MAP vectors at D = 2,048.

## Reproduce

From the repository root, after the scale fixture exists in `data/scale/`:

```sh
sh src/missingness/reproduce.sh
```

The script regenerates the inputs, runs the tests, all four experiments, the parked pairwise experiment, and the charts. Atoms are regenerated from their seeds with `person.codebook`, identical to those the scale study stores. Experiment 4 writes a LanceDB store of about 2.5 GB to `results/missingness/experiment4.lancedb`; it is gitignored and safe to delete.

| Module | Role |
| --- | --- |
| `person.py` | Records with missing values, MCAR and duplicate loaders, the MAP-I encoder (omit or token), per-property terms, L2 and majority-sign normalization, exact HYP-83 baselines |
| `policy.py` | Exact per-property similarities, the missing-value policies, Fellegi–Sunter levels and scorers, and the streamed, tie-aware search behind Experiments 2–4 |
| `experiment1_weight.py` … `experiment4_build.py` | The four experiments |
| `pairwise.py` | HYP-83's pairwise omit-versus-token experiment (parked) |
| `charts.py` | Figures as PNG and editable SVG |
| `tests/` | Orthogonal-atom exactness for both encodings and for the cosine policy, policy denominators, tie-aware ranks against brute force, EM recovery, majority-sign properties, mask rates |

## Saved evidence

Everything below is in `results/missingness/`.

| File | Contents |
| --- | --- |
| `experiment1_ranking.parquet`, `experiment1_ranking_summary.csv`, `experiment1_rankings.csv`, `experiment1_influence.csv`, `experiment1_config.json` | Every A_k / B cosine next to its exact value, how often each predicted ranking holds, and the interests' measured share of same-person matches |
| `experiment2_per_query.parquet`, `experiment2_summary.csv`, `experiment2_slope_curve.csv`, `experiment2_config.json` | Per-query precision, own-copy ranks and top-10 coverage mix per policy; summaries with bootstrap intervals; precision along the pivot slope |
| `experiment3_per_query.parquet`, `experiment3_summary.csv`, `experiment3_weights.csv`, `experiment3_config.json` | Duplicate ranks per scorer; u, labelled, EM and assumed m with their weights |
| `experiment4_summary.csv`, `experiment4_sign_cost.csv`, `experiment4_config.json` | Each LanceDB implementation against its exact policy and hidden truth; L2 against majority-sign partial credit; float16 against float32 |
| `pairwise.parquet`, `pairwise_summary.csv`, `hyp83_pairwise.png`, `hyp83_retrieval.csv`, `hyp83_semantic_shift.csv` | Parked HYP-83 results: omit against null tokens. The retrieval check's code is at commit `450c4e6`. |

[`EXPERIMENT_UPDATE.md`](EXPERIMENT_UPDATE.md) is the plan behind Experiment 1's ranking comparison, written after David's feedback.

## Scope

The fixture is a controlled factorial state space with four low-cardinality properties, not a sample of people; missingness is completely at random; Experiment 3's value errors are uniform. Results hold for this encoder, fixture and grid. See the report's limits.
