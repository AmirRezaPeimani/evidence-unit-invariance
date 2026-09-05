# Rows Are Not Evidence Units

Code and stored results for *Evidence-Unit Invariance in Weak Temporal
Localization*.

Ordinary learning treats materialized annotation rows as training records.
Quotient learning first compiles those rows into evidence classes, then fits
preprocessing and evaluates the same typed losses on canonical records.
The compiler preserves group, acquisition, source, annotation type, temporal
support, annotation content, and declared lineage distinctions.

Statement identity (`evidence_emission_id`) takes precedence. Session identity
is not evidence identity by default: it is used only with
`session_id_is_evidence_unit=True`. Missing or insufficient lineage retains
rows separately. Conflicting signals or validity masks for a matching key
raise an error. Canonical records exclude storage metadata; a separate
membership mapping links stored rows to evidence classes.

## Contents

- `src/evidence_unit_invariance/`: compiler, typed losses, class weighting,
  materialization operators, synthetic fixtures, and interval matching.
- `tests/` and `studies/`: executable checks and small deterministic examples.
- `results/`: numerical results, predictions, evidence-class membership,
  run configurations, optimizer updates, and held-group schedules.
- `schemas/`: stored-row and canonical-record schemas.
- `scripts/`: figure generation and operational-result summary.

## Installation and use

Use Python 3.10 or newer. From the repository root:

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python -m pytest tests -q
MPLBACKEND=Agg PYTHONPATH=src python scripts/build_figures.py
python scripts/summarize_operational_benchmark.py
```

Figures are written to `figures/`. These commands use the included results and
do not train models or download datasets. See [REPRODUCING.md](REPRODUCING.md)
for the examples, schema checks, and scope of full-experiment reproduction.
Public datasets are not redistributed; see [DATA.md](DATA.md).

## License

This project is licensed under the Apache License 2.0.
See [`LICENSE`](LICENSE) for details.
