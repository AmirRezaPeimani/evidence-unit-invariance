#!/usr/bin/env python3
"""Verify evidence-class preservation for operational materializations."""

from evidence_unit_invariance.compiler import compile_evidence_units
from evidence_unit_invariance.materialization import (
    deterministic_reorder,
    heterogeneous_metadata_fanout,
    one_to_many_join,
    repeated_export,
)
from evidence_unit_invariance.synthetic import make_row


def main() -> None:
    rows = [
        make_row("range", row_id="range", evidence_emission_id="e_range"),
        make_row(
            "timestamp",
            row_id="timestamp",
            source_id="source_2",
            evidence_emission_id="e_timestamp",
        ),
        make_row(
            "negative",
            row_id="negative",
            source_id="source_3",
            evidence_emission_id="e_negative",
        ),
    ]
    conditions = {
        "normalized": rows,
        "one_to_many_metadata_join": one_to_many_join(rows, {"acquisition_a": 2}),
        "repeated_export": repeated_export(rows, 3),
        "heterogeneous_metadata_fanout": heterogeneous_metadata_fanout(rows, [1, 2, 5]),
        "row_reorder": deterministic_reorder(rows),
    }
    for name, materialized in conditions.items():
        compiled = compile_evidence_units(materialized)
        print(name, len(materialized), len(compiled.units))


if __name__ == "__main__":
    main()
