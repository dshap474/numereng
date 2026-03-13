"""Public surface for experiment lifecycle feature services."""

from numereng.features.experiments.contracts import (
    ExperimentArchiveResult,
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
    archive_experiment,
    create_experiment,
    get_experiment,
    list_experiments,
    promote_experiment,
    report_experiment,
    train_experiment,
    unarchive_experiment,
)

__all__ = [
    "ExperimentAlreadyExistsError",
    "ExperimentArchiveResult",
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
    "archive_experiment",
    "create_experiment",
    "get_experiment",
    "list_experiments",
    "promote_experiment",
    "report_experiment",
    "train_experiment",
    "unarchive_experiment",
]
