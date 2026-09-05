"""Deterministic evidence rows used by the executable example and tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .compiler import EvidenceRow


def base_signal() -> np.ndarray:
    time = np.linspace(0.0, 1.0, 16, dtype=np.float64)
    return np.column_stack((time, np.cos(2.0 * np.pi * time)))


def make_row(
    annotation_type: str,
    *,
    row_id: str = "row_1",
    group_id: str = "group_a",
    acquisition_id: str = "acquisition_a",
    source_id: str = "source_1",
    evidence_emission_id: str | None = "emission_1",
    annotation_session_id: str | None = None,
) -> EvidenceRow:
    support_by_type = {
        "negative": (0, 1, 2, 3),
        "positive_bag": (4, 5, 6, 7),
        "timestamp": (7, 8, 9),
        "range": (6, 7, 8, 9),
        "dense_fragment": (5, 6, 7, 8),
    }
    payload_by_type = {
        "negative": {"state": 0},
        "positive_bag": {"state": 1},
        "timestamp": {"candidate": 8, "tolerance": 1},
        "range": {"state": 1},
        "dense_fragment": {"targets": [0, 1, 1, 0]},
    }
    return EvidenceRow(
        row_id=row_id,
        group_id=group_id,
        acquisition_id=acquisition_id,
        source_id=source_id,
        annotation_type=annotation_type,
        support=support_by_type[annotation_type],
        payload=payload_by_type[annotation_type],
        signal=base_signal(),
        validity_mask=np.ones(16, dtype=bool),
        evidence_emission_id=evidence_emission_id,
        annotation_session_id=annotation_session_id,
    )


def duplicate(row: EvidenceRow, count: int) -> list[EvidenceRow]:
    if count < 1:
        raise ValueError("Count must be positive")
    return [
        replace(row, row_id=f"{row.row_id}_copy_{index + 1}")
        for index in range(count)
    ]


def figure_one_rows() -> list[EvidenceRow]:
    """Return five distinct evidence units matching the mechanism illustration."""

    return [
        make_row("range", row_id="a_s1_range"),
        make_row("timestamp", row_id="a_s1_timestamp"),
        make_row(
            "range",
            row_id="a_s2_range",
            source_id="source_2",
        ),
        make_row(
            "negative",
            row_id="b_s1_negative",
            acquisition_id="acquisition_b",
        ),
        make_row(
            "positive_bag",
            row_id="b_s2_bag",
            acquisition_id="acquisition_b",
            source_id="source_2",
        ),
    ]
