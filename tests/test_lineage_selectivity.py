from dataclasses import replace

from evidence_unit_invariance.compiler import compile_evidence_units
from evidence_unit_invariance.synthetic import make_row


def test_lineage_selectivity_respects_declared_lineage():
    base = make_row("range", row_id="copy_0", evidence_emission_id="shared")
    shared = [replace(base, row_id=f"copy_{index}") for index in range(5)]
    independent = [
        replace(
            base,
            row_id=f"independent_{index}",
            evidence_emission_id=f"emission_{index}",
        )
        for index in range(5)
    ]
    ambiguous = [
        replace(
            base,
            row_id=f"ambiguous_{index}",
            evidence_emission_id=None,
            annotation_session_id=None,
        )
        for index in range(2)
    ]

    assert len(compile_evidence_units(shared).units) == 1
    assert len(compile_evidence_units(independent).units) == 5
    assert len(compile_evidence_units(ambiguous).units) == 2
