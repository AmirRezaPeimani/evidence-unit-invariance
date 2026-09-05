"""Evidence-unit compilation for weak temporal localization."""

from .compiler import (
    CompiledEvidence,
    CompilerConflict,
    EvidenceRecord,
    EvidenceRow,
    EvidenceUnit,
    ScalerState,
    compile_before_preprocessing,
    compile_evidence_units,
    fit_channel_scaler,
)
from .class_weighting import inverse_class_multiplicity
from .matching import qualified_interval_matching
from .typed_losses import nested_row_weights, nested_typed_empirical_risk

__all__ = [
    "CompiledEvidence",
    "CompilerConflict",
    "EvidenceRecord",
    "EvidenceRow",
    "EvidenceUnit",
    "ScalerState",
    "compile_before_preprocessing",
    "compile_evidence_units",
    "fit_channel_scaler",
    "inverse_class_multiplicity",
    "nested_row_weights",
    "nested_typed_empirical_risk",
    "qualified_interval_matching",
]
