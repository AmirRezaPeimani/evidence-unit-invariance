# Reproducing the code checks and figures

Run commands from the repository root with the dependencies in
`requirements.txt` installed.

## Tests and schemas

```bash
PYTHONPATH=src python -m pytest tests -q
python -m json.tool schemas/evidence_unit.schema.json >/dev/null
python -m json.tool schemas/canonical_evidence_record.schema.json >/dev/null
```

Tests cover multiplicity, row ordering, evidence-key distinctions, session
opt-in, row-specific fallback, missing-aware equality, conflicting observations,
canonical records, typed losses, group–acquisition–row weighting, and
IoU-qualified maximum-cardinality matching. JSON Schema validation of example
records is included in the tests.

The Python `payload` field stores annotation content. `evidence_emission_id`
is the statement identifier (`emission_id` in the membership table). The
schema also permits observation references; resolve these to arrays before
constructing an `EvidenceRow`.

The timestamp loss applies the positive logistic loss to every bin in its
declared window. The range implementation adds the positive support-bag loss
and the mean negative-complement loss, with coefficient one for each term.
An empty complement uses the full positive bag loss alone. These definitions
are in `src/evidence_unit_invariance/typed_losses.py`.
The loss helpers use the supplied support and logit bins; they do not derive
valid loss bins from the observation mask. Dataset adapters must supply the
valid-bin representation described in the paper.

## Small examples

```bash
PYTHONPATH=src python studies/lineage_selectivity.py
PYTHONPATH=src python studies/class_normalization_equivalence.py
PYTHONPATH=src python studies/materialization_benchmark.py
PYTHONPATH=src python studies/type_multiplicity_stress.py
```

These examples check the compiler and objective on synthetic arrays; they
are not replacements for the dataset experiments. In particular, identical
statements with distinct emission identities remain separate evidence classes.

## Stored-result reproduction

```bash
MPLBACKEND=Agg PYTHONPATH=src python scripts/build_figures.py
python scripts/summarize_operational_benchmark.py
```

The figure script writes eight vector PDFs to `figures/`. It reads the result
tables and probability trajectories, retains the selected temporal examples,
and writes their display summaries to `figures/values/`. The operational
summary uses three-decimal reporting precision while keeping the small
nonzero probability responses in the source tables.

Core results use held-group means. Operational segment-F1 aggregates are
fold means; operational maximum-probability aggregates are the maximum over
folds, not a fold mean. The figure code reads the stored aggregates directly.
Horizontal fold ranges are descriptive ranges, not confidence intervals.

## Full experiments

This compact repository supplies the compiler, losses, examples, schedules,
and stored results. It does not include the complete dataset-preprocessing
pipelines, TCN/BiGRU training entry points, raw data, or model checkpoints.
Full retraining requires those pipelines and locally obtained datasets under
their original licenses, with the target mappings and preprocessing specified
in the paper. The synthetic arrays used by the tests are small fixtures, not
the full Synthetic benchmark generator.

Dataset settings are in `results/run_configs/`, held-group splits in
`results/fold_schedules/`, and realized updates in `results/optimizer_runs.csv`.
The PPG-DaLiA configuration and schedule are
`results/run_configs/ppg_dalia.json` and
`results/fold_schedules/ppg_dalia.json`. Package checksums verify the included
files; they do not independently timestamp the original experiment protocol.

## File integrity

```bash
shasum -a 256 -c CHECKSUMS.sha256
```

The checksum file covers every packaged file except itself. Generated figures
and local caches are not part of the archive.
