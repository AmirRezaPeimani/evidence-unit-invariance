"""Evidence-preserving row materializations used by the benchmark."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Sequence

from .compiler import EvidenceRow


def _copy(row: EvidenceRow, suffix: str) -> EvidenceRow:
    return replace(row, row_id=f"{row.row_id}::{suffix}")


def repeated_export(rows: Sequence[EvidenceRow], exports: int = 3) -> list[EvidenceRow]:
    """Concatenate repeated exports while preserving emission identity."""

    if exports < 1:
        raise ValueError("exports must be positive")
    return [
        _copy(row, f"export_{export_index}")
        for export_index in range(exports)
        for row in rows
    ]


def one_to_many_join(
    rows: Sequence[EvidenceRow],
    fanout_by_acquisition: dict[str, int],
) -> list[EvidenceRow]:
    """Materialize a one-to-many metadata join."""

    output: list[EvidenceRow] = []
    for row in rows:
        fanout = fanout_by_acquisition.get(row.acquisition_id, 1)
        if fanout < 1:
            raise ValueError("fan-out must be positive")
        output.extend(_copy(row, f"join_{index}") for index in range(fanout))
    return output


def heterogeneous_metadata_fanout(
    rows: Sequence[EvidenceRow],
    fanouts: Iterable[int],
) -> list[EvidenceRow]:
    """Apply deterministic per-row fan-out counts."""

    counts = list(fanouts)
    if len(counts) != len(rows):
        raise ValueError("one fan-out count is required per row")
    output: list[EvidenceRow] = []
    for row, count in zip(rows, counts):
        if count < 1:
            raise ValueError("fan-out must be positive")
        output.extend(_copy(row, f"fanout_{index}") for index in range(count))
    return output


def deterministic_reorder(rows: Sequence[EvidenceRow]) -> list[EvidenceRow]:
    """Reverse a table without changing row content or identity."""

    return list(reversed(rows))
