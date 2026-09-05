#!/usr/bin/env python3
"""Check inverse-class weighting against quotienting on one fixed fixture."""

from dataclasses import replace

import torch

from evidence_unit_invariance.class_weighting import inverse_class_multiplicity
from evidence_unit_invariance.compiler import compile_evidence_units
from evidence_unit_invariance.synthetic import make_row
from evidence_unit_invariance.typed_losses import typed_empirical_risk, typed_row_loss


def main() -> None:
    base = make_row("range", row_id="copy_0", evidence_emission_id="shared")
    repeated = [replace(base, row_id=f"copy_{index}") for index in range(5)]
    control = make_row(
        "timestamp",
        row_id="control",
        source_id="source_2",
        evidence_emission_id="control",
    )
    rows = [*repeated, control]
    compiled = compile_evidence_units(rows).units
    weights = inverse_class_multiplicity(rows)
    logits = torch.linspace(-1.0, 1.0, 16, dtype=torch.float64)
    weighted = torch.stack(
        [
            weight * typed_row_loss(logits, row)
            for row, weight in zip(rows, weights, strict=True)
        ]
    ).sum() / len(compiled)
    quotient = typed_empirical_risk(logits, compiled)
    difference = float(torch.abs(weighted - quotient))
    if difference > 1e-12:
        raise AssertionError(f"Objectives differ by {difference}")
    print(f"stored rows: {len(rows)}")
    print(f"evidence classes: {len(compiled)}")
    print(f"absolute objective difference: {difference:.3e}")


if __name__ == "__main__":
    main()
