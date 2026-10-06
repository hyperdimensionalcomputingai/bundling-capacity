# Missing data and normalization

Real records have gaps. One customer lists three interests, another lists one, a third never filled in their job title. This study asks how to handle those gaps when each record is stored as a hypervector, and how to search millions of such records well.

The short answer: **it comes down to normalization**, meaning how you scale each part of a record before you compare it. Statistics, search engines and record linkage each worked out pieces of this decades ago, and the same ideas carry over to hypervectors.

- **[Report](REPORT.md):** the five experiments, with figures and numbers.
- **[Methodology](METHODOLOGY.md):** exactly how each experiment was set up.
- **[Scale study](../scale/README.md):** how complete records behave among a million others.

## What this means if you build with hypervectors

A quick refresher on the vocabulary: each field value (a job, a region, an interest) gets its own random vector. You **bind** a value to its field's "role" vector, and you **bundle** (add up) all the bound values to get one vector per record. Records are then compared by cosine similarity.

### When you encode a record

1. **Leave missing values out.** Don't create a special "missing" vector for blanks that are just gaps in the data. If you do, two strangers who are both missing the same fields start to look alike.
2. **Scale each field to the same length before you bundle it.** Otherwise a field's weight depends on how many values it holds: someone who lists three interests gets interests counted three times as heavily as someone who lists one. Add up a field's values, scale that sum to length 1, then bind it to its role.
3. **Scale with the L2 norm, not by rounding to ±1.** Rounding keeps vectors compact, but it can't tell "shares one interest" from "shares two".

### When you compare records

4. **Use cosine similarity.** It charges a record a little for the fields it doesn't have, because a match on two fields is less certain than a match on four. That turned out to be the right amount in almost every setting we tried. The tempting alternative, treating a blank as "no penalty at all", let near-empty records fill most of the top 10 results.
5. **Give rare agreements more weight.** Two people in the same region (1 chance in 5) is weak evidence; two people with the same three interests (about 1 in 37) is strong. You can measure how often each field agrees between random pairs of your own records, no labels needed, and weight the fields accordingly. Apply the weights to the *query*, so one stored index serves any weighting.

### When you store and search

6. **Store unit-length vectors as float16.** It halves storage and changed almost nothing in our tests (the same top 10 as float32 for 96.5% of queries).
7. **Store how many fields each record has filled in, next to the vector.** A score of 0.87 from a record that knows two of four fields means something different from 0.87 from a complete record. Keeping this count lets you show it, filter on it or rerank with it.
8. **Index with IVF_PQ trained for cosine, and always turn on refinement.** A LanceDB index is trained for one distance metric, so train it with the one you search with. For unit-length vectors use cosine (L2 works as well), not dot: a dot index on the same vectors found far fewer of the true top 10 (43% against 79% without refinement). Set `refine_factor` to about 50: without it, the cosine index returned about 70% of the true top 10; with it, about 94%, and those results were just as relevant as an exact search. If you ever need vectors of varying length, append one extra coordinate so you can use an L2 index instead of a dot index (the report shows how).
9. **Expect many records to share the same vector.** Records with the same known fields encode identically: with 30% of values missing, more than half of our records had an exact twin. Group them under one key and break ties by record ID or by how complete each record is.

## When a blank field means something

Everything above assumes that a blank is just a gap, like a form that failed to save. Our experiments removed values at random to match that assumption. Real data doesn't always work that way, and it's worth checking before you follow the advice above.

Statisticians describe three kinds of missing data:

- **Missing completely at random.** The gap has nothing to do with the record: a field that sometimes fails to save, or a column dropped from one import. The blank tells you nothing. *This is the only kind we tested.*
- **Missing at random.** The gap depends on something else you can see. For example, every record from one data source lacks a region, while records from other sources have it.
- **Missing not at random.** The gap depends on the missing value itself, or on the person. People with very high or very low incomes often skip the income question. People who don't care much about hobbies leave interests blank. A missing lab result often means the doctor saw no reason to order the test. **Here the blank is information:** two people who both declined to share their income really are more alike than one who declined and one who answered.

This matters because leaving a blank out (rule 1) means two records that are both blank in a field get no credit for that. That's right when the blank is a random gap, but it throws away a real signal when the blank means something. In that case it's better to give the field its own "missing" vector, so that a shared blank counts as agreement. When blanks are random, the same trick backfires: in an earlier experiment, two complete strangers who were each missing half their fields scored 0.50 just from their shared "missing" vectors.

How to decide:

1. **Decide field by field.** Blanks in `income` can be meaningful while blanks in `region` come from a broken import.
2. **Start with how the data was collected.** Optional or sensitive questions, and fields left empty on purpose ("not tested", "not applicable"), are the likeliest to be meaningful. Values lost to errors or pipeline gaps usually aren't.
3. **Look for patterns.** Compare the other fields between records that are blank in a field and records that aren't. If they differ, for example the blanks skew younger or cluster in one data source, the blanks aren't random. (This can show that blanks aren't random, but it can't always tell you *why*, which is why step 2 comes first.)
4. **If you have labelled matches, try both.** Encode the field with blanks left out and with a "missing" vector, and keep whichever finds matches better.
5. **When a field's blanks are meaningful,** give that field its own "missing" vector (never one shared across all fields), and keep leaving blanks out everywhere else.

## Ideas for the "what's possible" post

- **Choose how blanks count, per search.** Cosine and its variants are a choice you make when you write each record. A "blanks don't count" rule works as a reranking step over the top results.
- **One index, many weightings.** Different users, tasks or queries can bring their own field weights at search time.
- **Deduplication without training data.** Weights measured from random pairs, plus normalized vectors, give a matching score that needs no labels.
- **Results you can audit.** Showing how complete each match is makes it clear how much of a score is evidence and how much is absence.

## The five experiments

| # | Question | What we found |
| --- | --- | --- |
| 1 | Does the number of values in a field decide how much that field counts? | Yes, unless each field is scaled first. Interests made up 25% to 50% of a match depending on how many were listed. |
| 2 | How should blanks affect a similarity score? | Plain cosine was best or within a point of best everywhere. Treating blanks as "no penalty" let near-empty records take over the results. |
| 3 | Should rare agreements count for more? | Yes. Weighting fields by how surprising an agreement is, on top of cosine, found noisy duplicates best in every setting. |
| 4 | Does this work with real hypervectors in LanceDB? | Yes. float16 storage and a plain dot product reproduced the exact results, apart from tiny random noise. |
| 5 | Does an approximate index (IVF_PQ) still find the right records? | Yes, if it's trained for cosine and refinement is on: results as relevant as exact search. |

The [report](REPORT.md) has the figures and full numbers for each.

<details>
<summary><b>For researchers: notation, reproducing the results, and files</b></summary>

### Notation

Binding uses $\otimes$ and bundling uses $\oplus$. Without normalization a record is $\bigoplus_{p} \bigoplus_{v \in V_p} h_{\mathrm{role},p} \otimes h_{p,v}$. With per-property normalization it is $\bigoplus_{p} h_{\mathrm{role},p} \otimes (h_p / \lVert h_p \rVert)$, where $h_p = \bigoplus_{v \in V_p} h_{p,v}$. Bundling is computed as an arithmetic sum with no sign threshold; normalization is the separate division by the L2 norm. Experiments 2 and 3 score policies exactly (as if atoms were orthogonal); Experiments 1, 4 and 5 use MAP vectors at D = 2,048.

### Reproduce

From the repository root, after the scale fixture exists in `data/scale/`:

```sh
sh src/missingness/reproduce.sh
```

This regenerates the inputs, runs the tests, the five experiments, the earlier pairwise experiment and the charts. Atoms are regenerated from their seeds with `person.codebook`, identical to those the scale study stores. Experiments 4 and 5 write LanceDB stores of about 2.5 GB and 11 GB to `results/missingness/`; both are gitignored and safe to delete.

### Code

| Module | Role |
| --- | --- |
| `person.py` | Records with missing values, MCAR and duplicate loaders, the MAP-I encoder (omit or token), per-property terms, L2 and majority-sign normalization, exact similarity baselines |
| `policy.py` | Exact per-property similarities, the missing-value policies, Fellegi–Sunter levels and scorers, and the streamed, tie-aware search behind Experiments 2 to 4 |
| `experiment1_weight.py` to `experiment5_index.py` | The five experiments |
| `pairwise.py` | The earlier pairwise experiment: leaving blanks out against null tokens |
| `charts.py` | Figures as PNG and editable SVG |
| `tests/` | Orthogonal-atom exactness, policy denominators, tie-aware ranks against brute force, EM recovery, majority-sign properties, mask rates |

### Saved evidence (`results/missingness/`)

| Files | Contents |
| --- | --- |
| `experiment1_*` | Every A_k / B cosine next to its exact value, how often each predicted ranking holds, and the interests' measured share of same-person matches |
| `experiment2_*` | Per-query precision, own-copy ranks and top-10 coverage mix per policy; summaries with bootstrap intervals; precision along the pivot slope |
| `experiment3_*` | Duplicate ranks per scorer; u, labelled, EM and assumed m with their weights |
| `experiment4_*` | Each LanceDB implementation against its exact policy and hidden truth; L2 against majority-sign partial credit; float16 against float32 |
| `experiment5_summary.csv`, `experiment5_normalizer.csv`, `experiment5_metric_check.csv`, `experiment5_config.json` | Every IVF_PQ setting against exact search and hidden truth; realized-norm against field-count normalization; cosine, L2 and dot indexes compared, plus the MIPS-to-L2 reduction; resolved index settings |
| `experiment5_rabitq.csv` | IVF_RQ (RaBitQ) results from an earlier run, kept for the record; the code is at commit `f194888` |
| `pairwise*`, `retrieval_tokens.csv`, `retrieval_semantic_shift.csv` | Earlier results: leaving blanks out against null tokens, pairwise and in a 30%-missing retrieval check. The retrieval check's code is at commit `450c4e6`. |

### Scope

The fixture is synthetic: every combination of four fields (age band, job, region, three interests), not a sample of real people. Values are removed completely at random, and Experiment 3's value errors are uniform. Results hold for this encoder, fixture and grid; see the report's limits.

</details>
