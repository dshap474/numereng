"""Public surface for experiment lifecycle feature services."""

from numereng.features.experiments.contracts import (
    ExperimentPromotionResult,
    ExperimentRecord,
    ExperimentReport,
    ExperimentReportRow,
    ExperimentStatus,
    ExperimentTrainResult,
)
from numereng.features.experiments.service import (
    ExperimentAlreadyExistsError,
    ExperimentError,
    ExperimentNotFoundError,
    ExperimentRunNotFoundError,
    ExperimentValidationError,
    create_experiment,
    get_experiment,
    list_experiments,
    promote_experiment,
    report_experiment,
    train_experiment,
)

__all__ = [
    "ExperimentAlreadyExistsError",
    "ExperimentError",
    "ExperimentNotFoundError",
    "ExperimentPromotionResult",
    "ExperimentRecord",
    "ExperimentReport",
    "ExperimentReportRow",
    "ExperimentRunNotFoundError",
    "ExperimentStatus",
    "ExperimentTrainResult",
    "ExperimentValidationError",
    "create_experiment",
    "get_experiment",
    "list_experiments",
    "promote_experiment",
    "report_experiment",
    "train_experiment",
]
