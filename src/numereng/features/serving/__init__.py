"""Public serving feature surface."""

from numereng.features.serving.contracts import (
    LiveBuildResult,
    LiveSubmitResult,
    ModelUploadResult,
    PackageDiagnosticsSyncResult,
    PackageEvaluationDataset,
    PackageScoreResult,
    PackageScoreRuntime,
    PackageScoreStage,
    PickleBuildResult,
    ServingBlendRule,
    ServingComponentInspection,
    ServingComponentSpec,
    ServingInspectionResult,
    ServingNeutralizationSpec,
    SubmissionPackageRecord,
)
from numereng.features.serving.evaluation import score_submission_package, sync_submission_package_diagnostics
from numereng.features.serving.repo import ServingPackageNotFoundError, ServingValidationError
from numereng.features.serving.runtime import ServingRuntimeError, ServingUnsupportedConfigError
from numereng.features.serving.service import (
    build_live_submission_package,
    build_submission_pickle,
    create_submission_package,
    inspect_package,
    list_submission_packages,
    submit_live_package,
    upload_submission_pickle,
)

__all__ = [
    "LiveBuildResult",
    "LiveSubmitResult",
    "ModelUploadResult",
    "PackageDiagnosticsSyncResult",
    "PackageEvaluationDataset",
    "PackageScoreResult",
    "PackageScoreRuntime",
    "PackageScoreStage",
    "PickleBuildResult",
    "ServingBlendRule",
    "ServingComponentInspection",
    "ServingComponentSpec",
    "ServingInspectionResult",
    "ServingNeutralizationSpec",
    "ServingPackageNotFoundError",
    "ServingRuntimeError",
    "ServingUnsupportedConfigError",
    "ServingValidationError",
    "SubmissionPackageRecord",
    "build_live_submission_package",
    "build_submission_pickle",
    "create_submission_package",
    "inspect_package",
    "list_submission_packages",
    "score_submission_package",
    "submit_live_package",
    "sync_submission_package_diagnostics",
    "upload_submission_pickle",
]
