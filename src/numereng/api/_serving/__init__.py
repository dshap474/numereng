"""Serving API facade preserving the historical private import path."""

from __future__ import annotations

from numereng.api._serving.live import serve_live_build, serve_live_submit
from numereng.api._serving.packages import serve_package_create, serve_package_inspect, serve_package_list
from numereng.api._serving.pickle import serve_pickle_build, serve_pickle_upload
from numereng.api._serving.scoring import serve_package_score, serve_package_sync_diagnostics
from numereng.features.serving import (
    PackageDiagnosticsSyncResult,
    ServingBlendRule,
    ServingComponentSpec,
    ServingInspectionResult,
    ServingNeutralizationSpec,
    ServingPackageNotFoundError,
    ServingRuntimeError,
    ServingUnsupportedConfigError,
    ServingValidationError,
    SubmissionPackageRecord,
    build_live_submission_package,
    build_submission_pickle,
    create_submission_package,
    inspect_package,
    list_submission_packages,
    score_submission_package,
    submit_live_package,
    sync_submission_package_diagnostics,
    upload_submission_pickle,
)
from numereng.features.submission import (
    SubmissionModelNotFoundError,
    SubmissionModelUploadFileNotFoundError,
    SubmissionModelUploadFormatUnsupportedError,
)

__all__ = [
    "serve_live_build",
    "serve_live_submit",
    "serve_package_create",
    "serve_package_inspect",
    "serve_package_list",
    "serve_package_score",
    "serve_package_sync_diagnostics",
    "serve_pickle_build",
    "serve_pickle_upload",
]
