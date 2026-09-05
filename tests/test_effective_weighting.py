from dataclasses import replace

import numpy as np
import torch

from evidence_unit_invariance.class_weighting import inverse_class_multiplicity
from evidence_unit_invariance.compiler import compile_evidence_units
from evidence_unit_invariance.synthetic import make_row
from evidence_unit_invariance.typed_losses import (
    nested_row_weights,
    typed_empirical_risk,
    typed_row_loss,
)


def test_group_acquisition_row_weights_match_declared_objective():
    first = make_row(
        "range",
        row_id="g1_a1_1",
        group_id="g1",
        acquisition_id="a1",
        evidence_emission_id="e1",
    )
    rows = [
        first,
        replace(first, row_id="g1_a1_2", evidence_emission_id="e2"),
        replace(
            first,
            row_id="g1_a2_1",
            acquisition_id="a2",
            evidence_emission_id="e3",
        ),
        replace(
            first,
            row_id="g2_a3_1",
            group_id="g2",
            acquisition_id="a3",
            evidence_emission_id="e4",
        ),
    ]
    weights = np.asarray(nested_row_weights(rows))
    assert np.allclose(weights, [0.125, 0.125, 0.25, 0.5])
    assert np.isclose(weights.sum(), 1.0)
    assert np.isclose(weights[:3].sum(), 0.5)
    assert np.isclose(weights[3:].sum(), 0.5)


def test_class_normalized_rows_equal_quotient_loss_and_gradient():
    base = make_row("range", row_id="copy_0", evidence_emission_id="shared")
    repeated = [replace(base, row_id=f"copy_{index}") for index in range(5)]
    control = make_row(
        "timestamp",
        row_id="control",
        source_id="source_2",
        evidence_emission_id="control",
    )
    materialized = [*repeated, control]
    quotient = compile_evidence_units(materialized).units

    logits = torch.linspace(
        -1.0, 1.0, 16, dtype=torch.float64, requires_grad=True
    )
    row_weights = inverse_class_multiplicity(materialized)
    weighted = torch.stack(
        [
            weight * typed_row_loss(logits, row)
            for row, weight in zip(materialized, row_weights, strict=True)
        ]
    ).sum() / len(quotient)
    quotient_loss = typed_empirical_risk(logits, quotient)
    weighted_gradient = torch.autograd.grad(weighted, logits, retain_graph=True)[0]
    quotient_gradient = torch.autograd.grad(quotient_loss, logits)[0]

    assert torch.equal(weighted, quotient_loss)
    torch.testing.assert_close(
        weighted_gradient,
        quotient_gradient,
        rtol=0,
        atol=1e-15,
    )
