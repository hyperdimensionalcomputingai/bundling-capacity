# Methodology: when does chance look like a match?

Each record is a hypervector that bundles five facts. A practitioner compares records to find matches, and every comparison of two unrelated records carries a tiny chance of scoring like a match by accident. This study tests a simple probability argument for how those chances add up, and how fast, for two everyday jobs. The results are in the [report](REPORT.md).

## The question

- **Searching for one person** ("people similar to Maya"): one record against the other N − 1. The number of comparisons grows with N.
- **Checking the whole dataset** (deduplication, clustering, a similarity graph): every pair, about N²/2 comparisons. The number grows with N².

It is the birthday paradox. With 23 people, someone sharing *your* birthday is unlikely (about 6%), but *some* two people sharing one is a coin flip (about 50%). The chance per pair is the same; only the number of pairs changes. The study asks whether this argument, with the binomial model for one pair, predicts what happens to real bundled records, and what it says about 2,048, 4,096 and 8,192 dimensions.

## Records and encoder

- **Records.** 40,000 synthetic records with five independent, uniformly drawn properties: region (20 values), education (10), occupation (100), interest cluster (200) and employer (1,000). The populations are the first 4,000, 13,000 and 40,000 records. About 84% of pairs share no property. See [`data/scale`](../../data/scale/README.md).
- **Facts.** Each property has a random ±1 role hypervector, and each value a random ±1 value hypervector. A fact is their element-wise product: $h_{\mathrm{fact},i} = h_{\mathrm{role},i} \otimes h_{\mathrm{value},i}$.
- **Records.** The raw sum of the five facts, $h_{\mathrm{record}} = h_{\mathrm{fact},1} \oplus \cdots \oplus h_{\mathrm{fact},5}$, with no sign step. Every coordinate is −5, −3, −1, 1, 3 or 5, every fact can still be unbound, and a record hypervector stored as int8 takes D bytes.
- **Score.** Cosine similarity: the integer dot product over the product of the two record lengths.

## Theory in four steps

1. **What a match scores.** A pair sharing q of the 5 properties averages about q/5: each shared fact adds D to the dot product, and each record's length is about √(5D).
2. **Where to put the threshold.** $T_q = q/5 - z_{0.999}/\sqrt{D}$, with $z_{0.999} \approx 3.09$, so about 99.9% of pairs sharing q properties score above it. Using 1/√D, the spread of unrelated pairs, slightly overstates the spread of pairs that share facts, so the threshold sits a little low and misses stay under 0.1%.
3. **How often chance crosses it.** Model an unrelated pair's score as $(2H - D)/D$ with $H \sim \operatorname{Binomial}(D, 1/2)$: each coordinate agrees or disagrees like a coin flip. Then $p(T) = P(H \geq \lceil D(1 + T)/2 \rceil)$. The model is exact for ±1 records; for raw sums it has the right mean (0) and spread (1/√D), and the experiment checks whether that is enough.
4. **How chances add up.** With M unrelated comparisons, the expected number of false matches is $M\,p(T)$ and the chance of at least one is about $1 - e^{-M p(T)}$. One search has M ≈ 0.84 N; the whole dataset has M ≈ 0.84 N²/2.

Setting $M\,p(T)$ equal to a risk budget, here 1%, and solving for N gives the number of records that fit: linear in 1/p(T) for one search, and in its square root for the whole dataset.

## The experiment

- **Dimensions.** D = 2,048 (a realistic build), 4,096 and 8,192 (an upper bound at four times the storage).
- **Matches.** One fact (q = 1, threshold 0.132 at D = 2,048) and two facts (q = 2, threshold 0.332).
- **All pairs.** Every record is compared with every other once. Because the populations are nested, one pass gives all three sizes.
- **Trials.** Fresh hypervectors per trial: 40 at D = 2,048, where false matches occur, and 3 at D = 4,096 and 8,192, where theory predicts none.
- **Counted per trial and size.** Unrelated pairs at or above each threshold (whole dataset); records whose own search returned one (one search); pairs sharing at least q properties below the threshold (misses); and the highest unrelated score. The first trial at each dimension also records score histograms for pairs sharing 0, 1 and 2 properties.
- **Sizes.** At D = 2,048 with one-fact matches, theory expects about 0.01, 0.1 and 1 false pair in the whole dataset at 4,000, 13,000 and 40,000 records. These sizes put the effect where it can be measured.

## What the results can and cannot show

The measurements test the four steps where their predictions are large enough to observe. The report then uses the same formula for record counts far beyond what can be built, clearly labelled as formula output. The results describe five independent, uniformly distributed properties and raw-sum bundling; popular values and correlated properties in real data create genuine overlaps and change how many pairs are unrelated. The study does not choose a match threshold for any application.

## Acceptance checks

These are in `tests/test_study.py` and run, with Ruff, before the full study:

- The encoder sums exactly five bound facts with no sign, and unbinding a role recovers its value.
- Shared-property counts are correct on hand-built records.
- Thresholds and binomial tails behave as the formulas say.
- The all-pairs counts (pairs, unrelated pairs, false matches, searches with a false match, misses, highest unrelated score and histograms) match a brute-force computation on a small population at a low dimension, where false matches do occur.

## Implementation

`run.py` regenerates the records, runs every trial, and writes the summaries, figures, manifest and report. `study.py` holds the encoder, the theory and the all-pairs check; `report.py` writes the report from the summary. Hypervectors are regenerated from the trial seed and D (`torch.Generator().manual_seed(seed * 1_000_003 + D)`, drawn with `torchhd.random`), so none are stored. Figures use plotnine.
