from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import jsonschema

from evidence_unit_invariance.compiler import compile_evidence_units
from evidence_unit_invariance.synthetic import make_row


ROOT = Path(__file__).resolve().parents[1]


def _schema(name: str) -> dict:
    return json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))


def test_serialized_row_schema_has_safe_session_default():
    schema = _schema("evidence_unit.schema.json")
    session_flag = schema["properties"]["session_id_is_evidence_unit"]

    assert session_flag["type"] == "boolean"
    assert session_flag["default"] is False
    assert "session_id_is_evidence_unit" not in schema["required"]


def test_canonical_record_schema_excludes_storage_metadata():
    schema = _schema("canonical_evidence_record.schema.json")
    unit = compile_evidence_units([make_row("range")]).units[0]
    instance = asdict(unit)
    instance["support"] = list(instance["support"])
    instance["signal"] = instance["signal"].tolist()
    instance["validity_mask"] = instance["validity_mask"].tolist()

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(instance=instance, schema=schema)
    assert "row_id" not in schema["properties"]
    assert "annotation_session_id" not in schema["properties"]
    assert "session_id_is_evidence_unit" not in schema["properties"]
