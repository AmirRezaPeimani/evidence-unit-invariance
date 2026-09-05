"""Compiler applied before fitted preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ANNOTATION_TYPES = {
    "negative",
    "positive_bag",
    "timestamp",
    "range",
    "dense_fragment",
}


class CompilerConflict(ValueError):
    """Raised when one evidence key is paired with conflicting observations."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported annotation content value: {type(value)!r}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class EvidenceRow:
    """One serialized weak-evidence row and its observed acquisition payload."""

    row_id: str
    group_id: str
    acquisition_id: str
    source_id: str
    annotation_type: str
    support: tuple[int, ...]
    payload: Mapping[str, Any]
    signal: np.ndarray
    validity_mask: np.ndarray
    evidence_emission_id: str | None = None
    annotation_session_id: str | None = None
    session_id_is_evidence_unit: bool = False

    def __post_init__(self) -> None:
        if self.annotation_type not in ANNOTATION_TYPES:
            raise ValueError(f"Unsupported annotation type: {self.annotation_type}")
        if not isinstance(self.session_id_is_evidence_unit, bool):
            raise TypeError("session_id_is_evidence_unit must be a Boolean")
        if not all(
            isinstance(value, str) and value
            for value in (
                self.row_id,
                self.group_id,
                self.acquisition_id,
                self.source_id,
            )
        ):
            raise ValueError("Row, group, acquisition, and source IDs are required")
        if tuple(sorted(set(self.support))) != self.support:
            raise ValueError("Support indices must be unique and sorted")
        signal = np.asarray(self.signal)
        validity = np.asarray(self.validity_mask, dtype=bool)
        if signal.ndim not in {1, 2}:
            raise ValueError("Signal must have shape [time] or [time, channel]")
        if validity.shape not in {(signal.shape[0],), signal.shape}:
            raise ValueError("Validity mask must be per-time or match signal shape")
        if self.support and max(self.support) >= signal.shape[0]:
            raise ValueError("Support extends beyond the observed signal")
        _canonical_json(self.payload)

    def evidence_key(self) -> tuple[Any, ...]:
        """Return the declared evidence-unit key, excluding storage metadata."""

        if self.evidence_emission_id is not None:
            lineage = ("statement", self.evidence_emission_id)
        elif (
            self.annotation_session_id is not None
            and self.session_id_is_evidence_unit
        ):
            lineage = ("session", self.annotation_session_id)
        else:
            # Row IDs are required storage identifiers and serve as stable,
            # non-collapsing tokens when lineage is missing or insufficient.
            lineage = (
                "row",
                hashlib.sha256(
                    f"row-fallback::{self.row_id}".encode("utf-8")
                ).hexdigest(),
            )
        return (
            self.group_id,
            self.acquisition_id,
            self.source_id,
            self.annotation_type,
            _canonical_json(self.support),
            _canonical_json(self.payload),
            lineage,
        )


@dataclass(frozen=True)
class EvidenceUnit:
    """Canonical evidence record emitted by the compiler.

    Storage-only fields such as ``row_id`` and the session-scope configuration
    flag are intentionally absent. Serialized-row membership remains available
    separately through :class:`Membership`.
    """

    group_id: str
    acquisition_id: str
    source_id: str
    annotation_type: str
    support: tuple[int, ...]
    payload: Mapping[str, Any]
    signal: np.ndarray
    validity_mask: np.ndarray
    lineage_scope: str
    lineage_id: str

    def evidence_key(self) -> tuple[Any, ...]:
        """Return the complete canonical evidence-class key."""

        return (
            self.group_id,
            self.acquisition_id,
            self.source_id,
            self.annotation_type,
            _canonical_json(self.support),
            _canonical_json(self.payload),
            (self.lineage_scope, self.lineage_id),
        )


EvidenceRecord = EvidenceRow | EvidenceUnit


@dataclass(frozen=True)
class Membership:
    row_id: str
    class_id: str
    representative_row_id: str
    class_size: int


@dataclass(frozen=True)
class CompiledEvidence:
    units: tuple[EvidenceUnit, ...]
    membership: tuple[Membership, ...]


@dataclass(frozen=True)
class ScalerState:
    mean: np.ndarray
    standard_deviation: np.ndarray
    valid_count: np.ndarray


def _same_observation(left: EvidenceRow, right: EvidenceRow) -> bool:
    left_signal = np.asarray(left.signal)
    right_signal = np.asarray(right.signal)
    if left_signal.shape != right_signal.shape:
        return False
    left_valid = np.asarray(left.validity_mask, dtype=bool)
    right_valid = np.asarray(right.validity_mask, dtype=bool)
    if not np.array_equal(left_valid, right_valid):
        return False
    left_missing = np.isnan(left_signal)
    right_missing = np.isnan(right_signal)
    if not np.array_equal(left_missing, right_missing):
        return False
    return bool(
        np.array_equal(left_signal[~left_missing], right_signal[~right_missing])
    )


def _class_id(key: tuple[Any, ...]) -> str:
    digest = hashlib.sha256(
        _canonical_json(key).encode("utf-8")
    ).hexdigest()
    return f"eu_{digest[:16]}"


def compile_evidence_units(rows: Sequence[EvidenceRow]) -> CompiledEvidence:
    """Compile serialized rows into canonical, storage-independent records."""

    classes: dict[tuple[Any, ...], list[EvidenceRow]] = {}
    observed_row_ids: set[str] = set()
    for row in rows:
        if row.row_id in observed_row_ids:
            raise ValueError(f"Duplicate storage row ID: {row.row_id!r}")
        observed_row_ids.add(row.row_id)
        classes.setdefault(row.evidence_key(), []).append(row)

    units: list[EvidenceUnit] = []
    membership: list[Membership] = []
    for key in sorted(classes):
        # Row IDs are storage metadata, but sorting them gives each class a
        # stable representative when equivalent input rows are reordered.
        members = sorted(classes[key], key=lambda row: row.row_id)
        representative = members[0]
        for candidate in members[1:]:
            if not _same_observation(representative, candidate):
                raise CompilerConflict(
                    "Matching evidence key has conflicting signal or validity mask"
                )
        lineage_scope, lineage_id = key[-1]
        units.append(
            EvidenceUnit(
                group_id=representative.group_id,
                acquisition_id=representative.acquisition_id,
                source_id=representative.source_id,
                annotation_type=representative.annotation_type,
                support=representative.support,
                payload=_jsonable(representative.payload),
                signal=np.array(representative.signal, copy=True),
                validity_mask=np.array(
                    representative.validity_mask, dtype=bool, copy=True
                ),
                lineage_scope=lineage_scope,
                lineage_id=lineage_id,
            )
        )
        identifier = _class_id(key)
        membership.extend(
            Membership(
                row_id=row.row_id,
                class_id=identifier,
                representative_row_id=representative.row_id,
                class_size=len(members),
            )
            for row in members
        )
    return CompiledEvidence(tuple(units), tuple(membership))


def _as_signal_and_validity(row: EvidenceRecord) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(row.signal, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[:, None]
    validity = np.asarray(row.validity_mask, dtype=bool)
    if validity.ndim == 1:
        validity = np.broadcast_to(validity[:, None], signal.shape)
    return signal, validity & np.isfinite(signal)


def fit_channel_scaler(rows: Iterable[EvidenceRecord]) -> ScalerState:
    """Fit a channelwise scaler over compiled evidence units."""

    materialized = [_as_signal_and_validity(row) for row in rows]
    if not materialized:
        raise ValueError("Cannot fit a scaler without evidence units")
    channel_counts = {signal.shape[1] for signal, _ in materialized}
    if len(channel_counts) != 1:
        raise ValueError("All signals must have the same channel count")
    channels = channel_counts.pop()
    values = [[] for _ in range(channels)]
    for signal, valid in materialized:
        for channel in range(channels):
            values[channel].extend(signal[valid[:, channel], channel].tolist())
    if any(not channel_values for channel_values in values):
        raise ValueError("Every channel needs at least one valid value")
    mean = np.asarray([np.mean(channel) for channel in values], dtype=np.float64)
    standard_deviation = np.asarray(
        [np.std(channel) for channel in values], dtype=np.float64
    )
    standard_deviation[standard_deviation == 0] = 1.0
    valid_count = np.asarray([len(channel) for channel in values], dtype=np.int64)
    return ScalerState(mean, standard_deviation, valid_count)


def compile_before_preprocessing(
    rows: Sequence[EvidenceRow],
) -> tuple[CompiledEvidence, ScalerState]:
    """Compile evidence classes, then fit preprocessing over canonical records."""

    compiled = compile_evidence_units(rows)
    return compiled, fit_channel_scaler(compiled.units)
