from dataclasses import replace

import numpy as np

from evidence_unit_invariance.class_weighting import inverse_class_multiplicity
from evidence_unit_invariance.compiler import CompilerConflict, compile_evidence_units
from evidence_unit_invariance.materialization import (
    deterministic_reorder,
    heterogeneous_metadata_fanout,
    one_to_many_join,
    repeated_export,
)
from evidence_unit_invariance.synthetic import make_row


def test_materializations_preserve_declared_classes():
    rows = [
        make_row("range", row_id="r1", evidence_emission_id="e1"),
        make_row(
            "timestamp",
            row_id="r2",
            source_id="source_2",
            evidence_emission_id="e2",
        ),
    ]
    conditions = [
        rows,
        repeated_export(rows, exports=3),
        one_to_many_join(rows, {"acquisition_a": 2}),
        heterogeneous_metadata_fanout(rows, [2, 5]),
        deterministic_reorder(rows),
    ]
    assert [len(compile_evidence_units(condition).units) for condition in conditions] == [
        2,
        2,
        2,
        2,
        2,
    ]


def test_independent_emissions_are_not_content_deduplicated():
    base = make_row("range", row_id="shared", evidence_emission_id="e1")
    independent = replace(base, row_id="independent", evidence_emission_id="e2")
    assert len(compile_evidence_units([base, independent]).units) == 2


def test_class_weights_assign_unit_mass_per_emission():
    base = make_row("range", row_id="base", evidence_emission_id="e1")
    rows = [replace(base, row_id=f"copy_{index}") for index in range(5)]
    weights = inverse_class_multiplicity(rows)
    assert np.isclose(sum(weights), 1.0)
    assert all(np.isclose(weight, 0.2) for weight in weights)


def test_conflicting_signal_under_shared_identity_is_rejected():
    base = make_row("range", row_id="base", evidence_emission_id="e1")
    signal = base.signal.copy()
    signal[0, 0] += 1.0
    conflict = replace(base, row_id="conflict", signal=signal)
    try:
        compile_evidence_units([base, conflict])
    except CompilerConflict:
        return
    raise AssertionError("A shared identity with conflicting signal must raise an error")
