#!/usr/bin/env python3
"""Reproduce the lineage-selectivity class-count fixture."""

from dataclasses import replace

from evidence_unit_invariance.compiler import compile_evidence_units
from evidence_unit_invariance.synthetic import make_row


def main() -> None:
    repeated = make_row("range", row_id="copy_0", evidence_emission_id="shared")
    controls = [
        make_row("timestamp", row_id="control_1", evidence_emission_id="control_1"),
        make_row("negative", row_id="control_2", evidence_emission_id="control_2"),
    ]
    shared = [replace(repeated, row_id=f"copy_{index}") for index in range(5)]
    independent = [
        replace(repeated, row_id=f"copy_{index}", evidence_emission_id=f"e_{index}")
        for index in range(5)
    ]
    print(
        {
            "serialized_rows": len(shared) + len(controls),
            "shared_emission_classes": len(
                compile_evidence_units([*shared, *controls]).units
            ),
            "independent_emission_classes": len(
                compile_evidence_units([*independent, *controls]).units
            ),
        }
    )


if __name__ == "__main__":
    main()
