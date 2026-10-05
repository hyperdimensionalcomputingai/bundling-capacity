# Bundling capacity: inside and outside a MAP bundle

A MAP hypervector bundle combines bound facts. This repository tests that bundle from two directions.

Throughout these docs, $h$ denotes a hypervector, $\otimes$ denotes **binding**, and $\oplus$ denotes **bundling**. For these additive MAP encoders, binding is implemented as element-wise multiplication and bundling as an arithmetic sum without a final sign threshold. The regular plus sign ($+$) is reserved for ordinary arithmetic, such as adding scalar similarity contributions.

| | **Inside the bundle** | **Outside the bundle** |
| --- | --- | --- |
| Question | How many facts can one bundle hold before we can't read a fact back? | Once a record is a bundle, can we still find it, and only it, among a million others, even when fields are missing? |
| Pressure | Crowding within one vector: every added fact is noise for the others | Crowding among vectors: every added candidate is another chance to score high by accident, and missing fields change what "alike" means |
| Workload | One person's scored answers to 5–50 IPIP questionnaire items | 920,000 directory-style PERSON records: age band, job, region and three interests |
| Encoder | $h_{\mathrm{item}} \otimes h_{\mathrm{answer}}$, then additive bundling ($\oplus$); five ordered answer levels | $h_{\mathrm{role}} \otimes h_{\mathrm{value}}$, then additive bundling ($\oplus$); ordinal age levels and cosine comparison |
| Measures | Answer cleanup, whole-profile geometry, similarity as bundles grow | Chance similarity among 920,000 candidates across D and N; what a missing field and a null spelling do to one comparison; one 30%-missing retrieval check; IVF_PQ recall loss |
| Report | [src/qa-encoding/REPORT.md](src/qa-encoding/REPORT.md) | [src/scale-missingness/REPORT.md](src/scale-missingness/REPORT.md) |
| Code | [`src/qa-encoding`](src/qa-encoding/README.md) | [`src/scale-missingness`](src/scale-missingness/README.md) · [methodology](src/scale-missingness/METHODOLOGY.md) |
| Data | [`data/qa-encoding`](data/qa-encoding/README.md), 31 fixed synthetic profiles | [`data/scale-missingness`](data/scale-missingness/README.md), controlled factorial fixture |
| Linear | HYP-84 | HYP-83 |

The two studies use **different encoders and workloads**, so their accuracy numbers should not be compared as if one encoder produced both. Both fixtures are synthetic. The original records stay the exact source of truth; bundle readout and similarity are measured properties of each encoder on its fixture, not general capacity limits of a hyperspace.

Each study has its own locked environment under `src/<study>/`. See each README for the reproduction commands.
