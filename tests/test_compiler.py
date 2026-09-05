from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from evidence_unit_invariance.compiler import (
    CompilerConflict,
    EvidenceUnit,
    compile_before_preprocessing,
    compile_evidence_units,
)
from evidence_unit_invariance.synthetic import duplicate, make_row
from evidence_unit_invariance.typed_losses import typed_empirical_risk


ANNOTATION_TYPES = (
    "negative",
    "positive_bag",
    "timestamp",
    "range",
    "dense_fragment",
)
MULTIPLICITIES = (2, 3, 5, 10)


def loss_and_gradient(rows):
    logits = torch.linspace(
        -1.0, 1.0, 16, dtype=torch.float64, requires_grad=True
    )
    loss = typed_empirical_risk(logits, rows)
    gradient = torch.autograd.grad(loss, logits)[0]
    return loss.detach(), gradient.detach()


@pytest.mark.parametrize("annotation_type", ANNOTATION_TYPES)
@pytest.mark.parametrize("multiplicity", MULTIPLICITIES)
def test_multiplicity_invariance(annotation_type, multiplicity):
    row = make_row(annotation_type)
    control_type = "negative" if annotation_type != "negative" else "range"
    control = make_row(
        control_type,
        row_id="control",
        source_id="source_2",
    )
    clean, clean_scaler = compile_before_preprocessing([row, control])
    repeated, repeated_scaler = compile_before_preprocessing(
        [*duplicate(row, multiplicity), control]
    )
    clean_loss, clean_gradient = loss_and_gradient(clean.units)
    repeated_loss, repeated_gradient = loss_and_gradient(repeated.units)

    assert len(clean.units) == len(repeated.units) == 2
    assert torch.equal(clean_loss, repeated_loss)
    assert torch.equal(clean_gradient, repeated_gradient)
    assert np.array_equal(clean_scaler.mean, repeated_scaler.mean)
    assert np.array_equal(
        clean_scaler.standard_deviation,
        repeated_scaler.standard_deviation,
    )


@pytest.mark.parametrize("annotation_type", ANNOTATION_TYPES)
def test_row_reordering_invariance(annotation_type):
    first = make_row(annotation_type, row_id="first")
    repeated = replace(first, row_id="repeated")
    second = make_row(
        "negative" if annotation_type != "negative" else "range",
        row_id="second",
        source_id="source_2",
    )
    forward = compile_evidence_units([first, repeated, second])
    reverse = compile_evidence_units([second, repeated, first])
    assert [row.evidence_key() for row in forward.units] == [
        row.evidence_key() for row in reverse.units
    ]
    assert all(isinstance(unit, EvidenceUnit) for unit in forward.units)
    assert all(not hasattr(unit, "row_id") for unit in forward.units)
    assert torch.equal(
        loss_and_gradient(forward.units)[0],
        loss_and_gradient(reverse.units)[0],
    )
    assert torch.equal(
        loss_and_gradient(forward.units)[1],
        loss_and_gradient(reverse.units)[1],
    )


def test_compiler_emits_canonical_records_without_storage_metadata():
    row = make_row(
        "range",
        row_id="serialized_row",
        evidence_emission_id="declared_statement",
        annotation_session_id="collection_session",
    )

    compiled = compile_evidence_units([row])
    unit = compiled.units[0]

    assert isinstance(unit, EvidenceUnit)
    assert unit.lineage_scope == "statement"
    assert unit.lineage_id == "declared_statement"
    assert not hasattr(unit, "row_id")
    assert not hasattr(unit, "evidence_emission_id")
    assert not hasattr(unit, "annotation_session_id")
    assert not hasattr(unit, "session_id_is_evidence_unit")
    assert compiled.membership[0].row_id == "serialized_row"


def test_declared_semantic_boundaries_are_retained():
    base = make_row("range", row_id="base")
    rows = [
        base,
        replace(base, row_id="group", group_id="group_b"),
        replace(base, row_id="acquisition", acquisition_id="acquisition_b"),
        replace(base, row_id="source", source_id="source_2"),
        replace(base, row_id="type", annotation_type="timestamp"),
        replace(base, row_id="support", support=(7, 8, 9)),
        replace(base, row_id="lineage", evidence_emission_id="emission_2"),
    ]
    compiled = compile_evidence_units(rows)
    assert (len(rows), len(compiled.units)) == (7, 7), (
        "Expected seven distinct evidence classes for baseline, group, "
        "acquisition, source, type, support, and lineage variants"
    )

    same_statement_other_session = replace(
        base,
        row_id="same_statement_other_session",
        annotation_session_id="session_2",
        session_id_is_evidence_unit=True,
    )
    assert len(
        compile_evidence_units([base, same_statement_other_session]).units
    ) == 1


def test_missing_aware_equality_collapses():
    signal = make_row("range").signal.copy()
    signal[3, 0] = np.nan
    first = replace(make_row("range"), row_id="first", signal=signal.copy())
    second = replace(make_row("range"), row_id="second", signal=signal.copy())
    assert len(compile_evidence_units([first, second]).units) == 1


def test_undeclared_matching_rows_are_retained_separately():
    first = make_row(
        "range",
        row_id="ambiguous_first",
        evidence_emission_id=None,
        annotation_session_id=None,
    )
    second = replace(first, row_id="ambiguous_second")
    assert len(compile_evidence_units([first, second]).units) == 2


def test_same_session_independent_statements_retained_by_default():
    first = make_row(
        "range",
        row_id="same_session_statement_1",
        evidence_emission_id=None,
        annotation_session_id="shared_session",
    )
    first = replace(first, session_id_is_evidence_unit=False)
    second = replace(first, row_id="same_session_statement_2")

    compiled = compile_evidence_units([first, second])

    assert len(compiled.units) == 2
    assert len({member.class_id for member in compiled.membership}) == 2


def test_same_session_collapses_when_explicitly_session_scoped():
    first = make_row(
        "range",
        row_id="same_session_statement_1",
        evidence_emission_id=None,
        annotation_session_id="shared_session",
    )
    first = replace(first, session_id_is_evidence_unit=True)
    second = replace(first, row_id="same_session_statement_2")

    compiled = compile_evidence_units([first, second])

    assert len(compiled.units) == 1
    assert len({member.class_id for member in compiled.membership}) == 1
    assert {member.class_size for member in compiled.membership} == {2}


def test_conflicting_signal_rejects_conflicts():
    first = make_row("range", row_id="first")
    changed = first.signal.copy()
    changed[0, 0] += 1.0
    second = replace(first, row_id="second", signal=changed)
    with pytest.raises(CompilerConflict):
        compile_evidence_units([first, second])


def test_conflicting_validity_mask_rejects_conflicts():
    first = make_row("range", row_id="first")
    changed = first.validity_mask.copy()
    changed[0] = False
    second = replace(first, row_id="second", validity_mask=changed)
    with pytest.raises(CompilerConflict):
        compile_evidence_units([first, second])
