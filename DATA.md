# Data and stored results

The experiments use six public datasets—HAPT, MHEALTH, Daphnet Freezing of
Gait, HARTH, HAR70+, and PPG-DaLiA—and a controlled Synthetic benchmark.
The public datasets and their original licenses must be obtained from their
respective providers. No raw recordings or private source records are included.

The tests and figure generation require no raw-data download. `results/`
contains the core and held-group results, operational fold responses,
type-by-multiplicity outcomes (including non-finite fits), scaler and
training/validation comparisons, interval-matching example, and saved
probability trajectories used in the temporal figure.

`evidence_class_membership.csv.gz` maps materialized rows to evidence classes.
The configuration and schedule directories record dataset settings and held
groups. `split_integrity/split_integrity.csv` records split checks, and
`optimizer_runs.csv` records the realized training budgets.

Dataset participant codes are retained as grouping identifiers. They are not
repository-author identities. Probabilities, reference labels, and summary
statistics are processed experimental outputs, not raw recordings.
