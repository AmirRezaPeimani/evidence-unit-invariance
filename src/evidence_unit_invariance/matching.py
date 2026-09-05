"""IoU-qualified maximum-cardinality interval matching."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


Interval = tuple[int, int]


def interval_iou(left: Interval, right: Interval) -> float:
    """Return half-open interval intersection over union."""

    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union else 0.0


def qualified_interval_matching(
    references: Sequence[Interval],
    predictions: Sequence[Interval],
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Maximize qualified-pair cardinality, then total IoU.

    The additive cardinality reward exceeds any possible total-IoU
    difference, making the optimization lexicographic.
    """

    if not references or not predictions:
        empty_float = np.asarray([], dtype=float)
        empty_int = np.asarray([], dtype=int)
        return empty_float, empty_int, empty_int

    matrix = np.asarray(
        [
            [interval_iou(reference, prediction) for prediction in predictions]
            for reference in references
        ],
        dtype=float,
    )
    qualified = matrix >= threshold
    cardinality_reward = min(len(references), len(predictions)) + 1.0
    score = np.where(qualified, cardinality_reward + matrix, 0.0)
    ref_index, pred_index = linear_sum_assignment(-score)
    keep = qualified[ref_index, pred_index]
    return matrix[ref_index[keep], pred_index[keep]], ref_index[keep], pred_index[keep]
