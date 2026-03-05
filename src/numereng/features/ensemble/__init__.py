"""Public surface for ensemble feature workflows."""

from numereng.features.ensemble.contracts import (
    EnsembleBuildRequest,
    EnsembleComponent,
    EnsembleMetric,
    EnsembleNeutralizationMode,
    EnsembleRecord,
    EnsembleResult,
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
    "EnsembleValidationError",
    "build_ensemble",
    "get_ensemble_view",
    "list_ensembles_view",
]
