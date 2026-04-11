"""Public surface for ensemble feature workflows."""

from numereng.features.ensemble.contracts import (
    EnsembleBuildRequest,
    EnsembleComponent,
    EnsembleMetric,
    EnsembleNeutralizationMode,
    EnsembleRecord,
    EnsembleResult,
    EnsembleSelectionRequest,
    EnsembleSelectionResult,
    EnsembleSelectionSelectionMode,
    EnsembleSelectionSourceRule,
    EnsembleSelectionVariantName,
)
from numereng.features.ensemble.selection import (
    EnsembleSelectionError,
    EnsembleSelectionExecutionError,
    EnsembleSelectionValidationError,
    select_ensemble,
)
from numereng.features.ensemble.service import (
    EnsembleError,
    EnsembleExecutionError,
    EnsembleNotFoundError,
    EnsembleValidationError,
    build_ensemble,
    get_ensemble_view,
    list_ensembles_view,
)

__all__ = [
    "EnsembleBuildRequest",
    "EnsembleComponent",
    "EnsembleError",
    "EnsembleExecutionError",
    "EnsembleMetric",
    "EnsembleNeutralizationMode",
    "EnsembleNotFoundError",
    "EnsembleRecord",
    "EnsembleResult",
    "EnsembleSelectionError",
    "EnsembleSelectionExecutionError",
    "EnsembleSelectionRequest",
    "EnsembleSelectionResult",
    "EnsembleSelectionSelectionMode",
    "EnsembleSelectionSourceRule",
    "EnsembleSelectionValidationError",
    "EnsembleSelectionVariantName",
    "EnsembleValidationError",
    "build_ensemble",
    "get_ensemble_view",
    "list_ensembles_view",
    "select_ensemble",
]
