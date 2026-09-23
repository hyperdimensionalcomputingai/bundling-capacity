# Questionnaire MAP bundle experiments

## Shared encoder and fixture

- Use the fixed 31-profile synthetic IPIP-50 fixture in `data/qa-encoding`, with the published item order and scoring key. Reverse `-` items before encoding; retain raw responses as ground truth.
- Bind each scored answer's ordered TorchHD level vector to an independent bipolar MAP item key. Add bound facts without sign thresholding. Reuse keys and the five-level codebook across profiles for each dimension and seed.
- Store fixed-size float16 vectors in LanceDB; compute similarities and readouts in float32. Save source hashes and run settings.

## Experiment 1: answer cleanup

Sweep 5, 10, 20, 30, 40, and 50 nested facts; dimensions 512, 2,048, 4,096, and 8,192; and seeds 11, 23, 37, 53, and 71. Unbind each item key, compare with the same five stored answer vectors, and report exact answer accuracy, within-one accuracy, ordinal error, and winner margin. Check selected filtered LanceDB five-way searches against TorchHD cleanup.

## Experiment 2: whole-profile geometry

At the same widths, dimensions, and seeds, compare every pair of complete bundles. Use matched-item mean absolute response difference and mean codebook-level similarity as references. Report pair-ranking correlation, nearest-neighbor agreement, and three-neighbor overlap. A respondent ID alone is not a semantic similarity label.

## Experiment 3: similarity as bundles grow

Fix the dimension at 4,096 and use widths 5, 10, 20, 40, and 50. For each profile and width, create up to 24 unique variants changing exactly 20% of raw answers by one level, with boundary answers moving inward. Repeat with 60% changed as a farther control. Sample changed positions throughout the width; reuse the same variants across MAP seeds. Re-score and encode each variant with the original keys and codebook.

Compare original/variant whole-bundle cosine with the ideal mean matched-level cosine before bundling. Look for drift in the similarity of 20%-changed pairs as width grows, and use 60%-changed pairs to check that more different answer patterns remain farther apart. Average variants within profile and seed before aggregating profiles; at five facts, enumerate unique variants to avoid duplicate observations. Inspect different-profile pairs by their raw response difference as context, without calling every different ID dissimilar.

The key question is whether profiles with mostly alike answers remain close in hyperspace as bundle width grows. The report should explain this question before the chart and distinguish whole-bundle similarity from exact item cleanup.
