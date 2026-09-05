"""Inverse evidence-class multiplicity weights."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from .compiler import EvidenceRow


def inverse_class_multiplicity(rows: Sequence[EvidenceRow]) -> tuple[float, ...]:
    """Return row weights that assign unit mass to every evidence class."""

    counts = Counter(row.evidence_key() for row in rows)
    return tuple(1.0 / counts[row.evidence_key()] for row in rows)
