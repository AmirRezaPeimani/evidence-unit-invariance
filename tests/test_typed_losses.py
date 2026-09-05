from dataclasses import replace

import torch
from torch.nn import functional as F

from evidence_unit_invariance.synthetic import make_row
from evidence_unit_invariance.typed_losses import typed_row_loss


def test_all_native_losses_are_finite_and_differentiable():
    for annotation_type in (
        "negative",
        "positive_bag",
        "timestamp",
        "range",
        "dense_fragment",
    ):
        logits = torch.linspace(
            -1.0, 1.0, 16, dtype=torch.float64, requires_grad=True
        )
        loss = typed_row_loss(logits, make_row(annotation_type))
        gradient = torch.autograd.grad(loss, logits)[0]
        assert torch.isfinite(loss)
        assert torch.isfinite(gradient).all()


def test_timestamp_executes_dense_local_window_surrogate():
    row = make_row("timestamp")
    logits = torch.linspace(-1.0, 1.0, 16, dtype=torch.float64)

    observed = typed_row_loss(logits, row)
    expected = F.softplus(-logits[list(row.support)]).mean()

    assert torch.equal(observed, expected)


def test_range_with_empty_complement_executes_full_bag_loss():
    row = replace(make_row("range"), support=tuple(range(16)))
    logits = torch.linspace(-1.0, 1.0, 16, dtype=torch.float64)

    observed = typed_row_loss(logits, row)
    probabilities = torch.sigmoid(logits)
    bag_probability = 1.0 - torch.prod(1.0 - probabilities)
    expected = -torch.log(bag_probability)

    assert torch.equal(observed, expected)
