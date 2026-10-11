# Bundling capacity: inside and outside a MAP bundle

A MAP hypervector bundle combines bound facts. This repository tests that bundle from two directions.

Throughout these docs, $h$ denotes a hypervector, $\otimes$ denotes **binding**, and $\oplus$ denotes **bundling**. Binding is implemented as element-wise multiplication in both studies. Bundling is the same in both: an arithmetic sum without a final sign threshold, so every bound fact can still be unbound. The regular plus sign ($+$) is reserved for ordinary arithmetic, such as adding scalar similarity contributions.

| | **Inside the bundle** | **Outside the bundle** |
| --- | --- | --- |
| Question | How many facts can one bundle hold before we can't read a fact back? | How many records fit before chance alone makes two unrelated records look like a match, for one search and for the whole dataset? |
| Pressure | Crowding within one vector: every added fact is noise for the others | Crowding among vectors: every added candidate is another chance to score high by accident |
| Workload | One person's scored answers to 5–50 IPIP questionnaire items | 40,000 synthetic records with five categorical properties: region, education, occupation, interest cluster and employer |
| Encoder | $h_{\mathrm{item}} \otimes h_{\mathrm{answer}}$, then additive bundling ($\oplus$); five ordered answer levels | $h_{\mathrm{role}} \otimes h_{\mathrm{value}}$, then additive bundling ($\oplus$) of five facts; cosine between every pair of records |
| Measures | Answer cleanup, whole-profile geometry, similarity as bundles grow | False matches and missed matches against a match threshold, for one search and for the whole dataset, at D = 2,048, 4,096 and 8,192 |
| Report | [src/qa-encoding/REPORT.md](src/qa-encoding/REPORT.md) | [src/scale/REPORT.md](src/scale/REPORT.md) |
| Code | [`src/qa-encoding`](src/qa-encoding/README.md) | [`src/scale`](src/scale/README.md) · [methodology](src/scale/METHODOLOGY.md) |
| Data | [`data/qa-encoding`](data/qa-encoding/README.md), 31 fixed synthetic profiles | [`data/scale`](data/scale/README.md), one fixed sequence of uniformly drawn records |

The two studies use **different encoders and workloads**, so their numbers should not be compared as if one encoder produced both. Both datasets are synthetic. The original records stay the exact source of truth; bundle readout and similarity are measured properties of each encoder on its data, not general capacity limits of a hyperspace.

Each study has its own locked environment under `src/<study>/`. See each README for the reproduction commands.
