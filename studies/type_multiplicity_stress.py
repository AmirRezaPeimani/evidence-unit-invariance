#!/usr/bin/env python3
"""Compile type-by-multiplicity fixtures under one shared emission per row set."""

from dataclasses import replace

from evidence_unit_invariance.compiler import ANNOTATION_TYPES, compile_evidence_units
from evidence_unit_invariance.synthetic import make_row


def main() -> None:
    for annotation_type in sorted(ANNOTATION_TYPES):
        base = make_row(
            annotation_type,
            row_id="copy_0",
            evidence_emission_id=f"emission_{annotation_type}",
        )
        for multiplicity in (2, 5, 10):
            rows = [replace(base, row_id=f"copy_{index}") for index in range(multiplicity)]
            print(annotation_type, multiplicity, len(compile_evidence_units(rows).units))


if __name__ == "__main__":
    main()
