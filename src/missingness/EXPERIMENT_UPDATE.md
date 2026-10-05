# Minimal update: normalization under missingness

## Takeaway

How many interests a record happens to list should not decide how much the
interests property matters. In our current encoder, it does. Normalizing each
property before bundling removes that dependence, and can change which person
looks most similar to a query.

## Premise

A person with one recorded interest may simply have a less complete profile than
a person with three. Their similarity to a query should reflect what was
observed, and we should choose deliberately how much interests count relative to
age, job and region.

The current encoder gives each known value one term. Age, job and region each
contribute one term, but three interests contribute three. So the number of
listed interests, in the query and in each candidate, sets how much influence
the interests property has. Cosine divides by each record's overall norm, but
that does not give every property equal influence.

## Question

Does normalizing each property before bundling change (a) how much a record is
penalized for unknown interests, and (b) which candidate is the nearest match?

The comparison is **with versus without property normalization**. Both use
cosine.

## Records

The query has an age band, job, region and three distinct interests. We compare
it against two kinds of candidate:

| Property | Candidate A_k (sparse) | Candidate B (complete) |
| --- | --- | --- |
| Age band | Matches | Matches |
| Job | Matches | Different |
| Region | Matches | Matches |
| Interests | k of the query's 3 interests known, all matching; the rest unknown | All 3 match |

A_k is built for k = 0, 1, 2, 3, so A has no missing interests only at k = 3.
Unknown interests are omitted, not encoded as mismatches. B has no missing
values; it is the reference that A must beat.

Varying k separates the two effects. The A_k curve shows the cost of missing
interests on its own. Comparing it with B shows when that cost changes the
winner.

## Encodings

Use the same role and value hypervectors for every record:

1. **Without property normalization:** bind each known value to its role and
   add all the facts together. Three interests contribute three terms.
2. **With property normalization:** add the known values within each property,
   scale that property's vector to unit length, then bind it to its role and add
   the properties together. Every observed property has the same length.
   A property with no known values contributes nothing.

## Predictions

The ideal algebra, assuming independent, orthogonal facts, predicts:

| Encoding | A_0 | A_1 | A_2 | A_3 | B |
| --- | --- | --- | --- | --- | --- |
| Without normalization | 0.707 | 0.816 | 0.913 | 1.000 | 0.833 |
| With normalization | 0.866 | 0.894 | 0.954 | 1.000 | 0.750 |

What a reader should see:

- **Without normalization**, an incomplete profile loses up to 0.29 (from 1.000
  down to 0.707), and A needs at least 2 of 3 interests to beat B.
- **With normalization**, that penalty shrinks to 0.13 but does not disappear,
  and A beats B at every k.

These are predictions, not measurements. Report the measured scores and whether
the predicted rankings hold. Do not adjust the seed to obtain them. Without
normalization, the gap between A_1 and B is small (about 0.017), so report its
spread across seeds as well as the seed-11 value.

## Explanation for readers

Keep it to three points, each tied to the chart:

1. **Who sets the weight.** Without normalization, the query's three interests
   make up 3 of its 6 terms, so interests carry half its weight. With
   normalization they carry a quarter, the same as each other property.
2. **Partial knowledge earns partial credit.** Normalization does not fill in
   missing values. A record that knows k of the query's 3 interests earns
   √(k/3) of the interests credit: about 0.58 for one, 0.82 for two.
3. **Unknown costs less than wrong.** An unknown property adds nothing to the
   match, but it also makes the record's vector shorter, so cosine partly
   compensates. A mismatched property also adds nothing to the match, but it
   still makes the vector longer. That is why, with normalization, A_0, with no
   interests known, scores 0.866 and beats B (0.750), which matches every
   interest but has the wrong job.

Point 3 is the result readers will question. Present it as the consequence of a
weighting choice: each observed property counts equally, and absence is not
treated as evidence of a mismatch. Do not present it as proof that sparse records
should win.

## Output

One chart: score against k (0 to 3) for A, as one line per encoding, with B
drawn as a horizontal reference line for each encoding. Place it next to the
record table above. This is the central illustration in the missingness report.

## Scope and caveats

- This does not test one strong semantic interest match against several weaker
  ones; our categorical interests either match exactly or differ.
- Existing results remain evidence for the current encoder.
- No million-record run, parameter sweep, text embeddings or fixture
  regeneration. Construct these records directly.
- This document is a plan only.

## Methods note

Use the existing MAP categorical representation, 2,048 dimensions and seed 11
for the worked example. Compute in float32 and score by cosine. Normalize with
the L2 norm, which preserves direction; do not apply a sign threshold. The query
is encoded with the same method as the candidates in each comparison. A_k uses
the first k of the query's interests. Repeat over 100 seeds and report the mean,
standard deviation and the fraction of seeds where each predicted ranking holds.
In a quick check over 500 seeds, B beat A_1 without normalization in 99.4% of
them, and A_1 beat B with normalization in all of them.
