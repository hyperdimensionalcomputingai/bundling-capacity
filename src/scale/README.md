# Bundled categorical records at scale

This study looks **outside** one bundle. Each record is a majority-sign bundle of five bound categorical facts. We ask how faithfully exact cosine reports the number of shared properties as the candidate population grows to a million records and D ranges from 512 to 10,000. The companion [`qa-encoding`](../qa-encoding/README.md) study looks **inside** one bundle.

- **[Report](REPORT.md):** results, figures and tables.
- **[Methodology](METHODOLOGY.md):** the question, the theory baseline, the runs and the encoder validation.
- **[Data](../../data/scale/README.md):** the million synthetic records.

Binding uses $\otimes$: a fact is $h_{\mathrm{fact},i} = h_{\mathrm{role},i} \otimes h_{\mathrm{value},i}$. Bundling uses $\oplus$: $h_{\mathrm{record}} = h_{\mathrm{fact},1} \oplus \cdots \oplus h_{\mathrm{fact},5}$. Here binding is element-wise multiplication, and bundling takes the coordinate-wise majority sign of all five facts in one operation.

## What this means for HDC practitioners

These hold for five uniformly drawn categorical properties, majority-sign bundling and exact cosine; the [report](REPORT.md) has the measurements.

1. **Count on about 0.14 cosine per shared property.** For five-fact majority bundles, records sharing zero, one, two and three properties average 0, 0.141, 0.281 and 0.438 at any D. Majority bundling discounts each shared fact compared with an additive bundle (0.2 per fact). Exact enumeration predicts the discount, and the measurements match it to within 0.001.
2. **Choose D for the noise; N doesn't change it.** Individual scores scatter by about 1/√D around those means: 0.044 at D = 512, 0.022 at 2,048, 0.010 at 10,000.
3. **Expect the best chance score to climb slowly with N.** Among about 84 million zero-overlap comparisons, the highest score sat roughly 5.5–6 standard deviations above zero. A hundred times more candidates raised it by 0.035 at D = 512 and 0.010 at D = 10,000.
4. **Use D ≥ 4,096 to keep a million chance scores well below one shared property.** At D = 512 the highest zero-overlap score (0.25) beat the typical one-shared score (0.141). At 2,048 it came within one standard deviation of that score. From 4,096 it stayed at least 3 one-shared standard deviations below it.
5. **Size D on paper first.** A binomial tail with the actual number of comparisons predicted the highest chance score to within 0.008 at every D and N tested.
6. **Expect diminishing returns past 8,192.** Going to 10,000 dimensions narrows the spread by about 10% and lowers the highest chance score from 0.061 to 0.057.

Ideas for a "what's possible" post:
- a one-line sizing rule built from the per-fact mean and the binomial tail;
- how the per-fact step changes with the number of facts;
- what skewed real-world value frequencies would do to the group counts.

None of this sets a match threshold; that stays an application decision.

## Reproduce

From the repository root:

```sh
sh src/scale/reproduce.sh
```

This runs Ruff, the focused tests, then `run.py`. The run regenerates the records, runs the encoder validation, scans 15 (D, seed) passes over a million records, and writes the summaries, figures, table and report. It takes about a minute and a half on a recent laptop CPU and needs no stored hypervectors.

| File | Role |
| --- | --- |
| `run.py` | The one run script: records, encoder validation, scan, summaries, figures, manifest, then the report |
| `study.py` | Encoder (TorchHD random role and value vectors, int8 bound facts, majority sign), the exact theory, and the batched scan with exact integer accumulation |
| `report.py` | Writes `REPORT.md` and `table_n1m.csv` from the saved summaries |
| `tests/test_study.py` | Exact enumeration of the means, the encoder, hand-built k = 0–5 records, duplicates and self-exclusion, a small end-to-end run against brute force, controlled pairs, and the discrete binomial references |

Outputs go to `results/scale/`; the [report](REPORT.md#files) lists them.
