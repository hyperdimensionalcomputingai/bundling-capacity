# Fixed questionnaire fixture

`items.csv` follows the [IPIP 50-item sample questionnaire](https://www.ipip.ori.org/New_IPIP-50-item-scale.htm), including its published order, factor, and key direction. Factor IV is **Emotional Stability**. [IPIP's scoring instructions](https://ipip.ori.org/newScoringInstructions.htm) reverse 1–5 responses on `-` keyed items.

`profiles.csv` and `responses.csv` hold 31 **synthetic** profiles and 1,550 raw item responses. These are fixed experimental inputs, not real survey respondents or clinical records. The generator's historical `hyp84-questionnaire-v1` seed is retained so regenerating the fixture produces the same profiles and raw answers.

Run `python data/qa-encoding/generate.py` to regenerate the two profile files. The experiment keeps raw and scored answers in separate columns in LanceDB. Text statements remain metadata; no text is embedded or fed to the encoder.

The sweep uses nested prefixes of 5, 10, 20, 30, 40, and 50 items. The published item order cycles through five factors, so each prefix has equal factor counts. The profiles stay fixed while the number of facts in a bundle grows.
