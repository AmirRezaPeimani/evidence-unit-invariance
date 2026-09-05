"""Native typed weak losses used after evidence-unit compilation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch
from torch.nn import functional as F

from .compiler import EvidenceRecord


def _indices(row: EvidenceRecord, logits: torch.Tensor) -> torch.Tensor:
    if not row.support:
        raise ValueError("Typed loss requires a nonempty support")
    return torch.as_tensor(row.support, dtype=torch.long, device=logits.device)


def _positive_bag(probabilities: torch.Tensor) -> torch.Tensor:
    probability = 1.0 - torch.prod(1.0 - probabilities)
    return -torch.log(probability.clamp_min(torch.finfo(probabilities.dtype).tiny))


def typed_row_loss(logits: torch.Tensor, row: EvidenceRecord) -> torch.Tensor:
    """Evaluate the executed native loss for one weak evidence record.

    Timestamp statements use a dense local-window surrogate: every bin in the
    declared support is assigned the positive logistic loss. Range statements
    use a positive bag loss on their support plus a negative logistic loss on
    the complement. When that complement is empty, the executed range loss is
    the full positive bag loss with no negative-complement term.
    """

    if logits.ndim != 1:
        raise ValueError("Logits must be one-dimensional")
    support = _indices(row, logits)
    selected = logits[support]
    probabilities = torch.sigmoid(selected)

    if row.annotation_type == "negative":
        return F.softplus(selected).mean()
    if row.annotation_type == "positive_bag":
        return _positive_bag(probabilities)
    if row.annotation_type == "timestamp":
        return F.softplus(-selected).mean()
    if row.annotation_type == "range":
        positive = _positive_bag(probabilities)
        outside_mask = torch.ones(
            logits.shape[0], dtype=torch.bool, device=logits.device
        )
        outside_mask[support] = False
        if not torch.any(outside_mask):
            return positive
        negative = F.softplus(logits[outside_mask]).mean()
        return positive + negative
    if row.annotation_type == "dense_fragment":
        targets = torch.as_tensor(
            row.payload["targets"], dtype=logits.dtype, device=logits.device
        )
        if targets.shape != selected.shape:
            raise ValueError("Dense targets must match the declared support")
        return F.binary_cross_entropy_with_logits(selected, targets)
    raise ValueError(f"Unsupported annotation type: {row.annotation_type}")


def typed_empirical_risk(
    logits: torch.Tensor, rows: Sequence[EvidenceRecord]
) -> torch.Tensor:
    """Average the native typed row losses without changing their definitions."""

    if not rows:
        raise ValueError("Typed risk requires at least one evidence unit")
    return torch.stack([typed_row_loss(logits, row) for row in rows]).mean()


def nested_row_weights(rows: Sequence[EvidenceRecord]) -> tuple[float, ...]:
    """Give equal mass to groups, acquisitions, then rows within acquisitions."""

    if not rows:
        raise ValueError("Nested risk requires at least one evidence unit")
    acquisitions_by_group: dict[str, set[str]] = defaultdict(set)
    rows_by_acquisition: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        acquisitions_by_group[row.group_id].add(row.acquisition_id)
        rows_by_acquisition[(row.group_id, row.acquisition_id)] += 1
    group_count = len(acquisitions_by_group)
    return tuple(
        1.0
        / group_count
        / len(acquisitions_by_group[row.group_id])
        / rows_by_acquisition[(row.group_id, row.acquisition_id)]
        for row in rows
    )


def nested_typed_empirical_risk(
    logits_by_acquisition: Mapping[str, torch.Tensor],
    rows: Sequence[EvidenceRecord],
) -> torch.Tensor:
    """Evaluate the group–acquisition–row normalized typed objective."""

    weights = nested_row_weights(rows)
    losses = []
    for row, weight in zip(rows, weights, strict=True):
        try:
            logits = logits_by_acquisition[row.acquisition_id]
        except KeyError as error:
            raise KeyError(
                f"Missing logits for acquisition {row.acquisition_id!r}"
            ) from error
        losses.append(weight * typed_row_loss(logits, row))
    return torch.stack(losses).sum()
