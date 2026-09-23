# How many answers fit in a MAP bundle?

This repository measures what happens as 5 to 50 scored questionnaire answers are packed into one additive MAP hypervector. It uses independent vectors for item positions and one shared ordered five-level answer scale.

- [Research-facing report and figures](REPORT.md)
- [Experiment design](QA_ENCODING_EXPERIMENT_PLAN.md)
- [Run and measurement details](src/qa-encoding/README.md)
- [Fixed synthetic questionnaire data](data/qa-encoding/README.md)

The original answer list remains the exact record. Bundle readout and profile similarity are measured properties of this encoder on a fixed synthetic fixture.
