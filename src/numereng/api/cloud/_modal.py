"""Modal managed training API handlers."""

from __future__ import annotations

from numereng.features.cloud.modal import (
    CloudModalError,
    CloudModalResponse,
    ModalDataSyncRequest,
    ModalDeployRequest,
    ModalTrainCancelRequest,
    ModalTrainLogsRequest,
    ModalTrainPullRequest,
    ModalTrainStatusRequest,
    ModalTrainSubmitRequest,
)
from numereng.platform.errors import PackageError


def cloud_modal_data_sync(request: ModalDataSyncRequest) -> CloudModalResponse:
    """Sync config-required training data files into a Modal Volume."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_modal_service().data_sync(request)
    except CloudModalError as exc:
        raise PackageError(str(exc)) from exc
    except Exception as exc:
        raise PackageError(f"cloud_modal_unexpected_error:{exc}") from exc


def cloud_modal_deploy(request: ModalDeployRequest) -> CloudModalResponse:
    """Deploy a Modal training function backed by ECR."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_modal_service().deploy(request)
    except CloudModalError as exc:
        raise PackageError(str(exc)) from exc
    except Exception as exc:
        raise PackageError(f"cloud_modal_unexpected_error:{exc}") from exc


def cloud_modal_train_submit(request: ModalTrainSubmitRequest) -> CloudModalResponse:
    """Submit managed training call to Modal."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_modal_service().train_submit(request)
    except CloudModalError as exc:
        raise PackageError(str(exc)) from exc
    except Exception as exc:
        raise PackageError(f"cloud_modal_unexpected_error:{exc}") from exc


def cloud_modal_train_status(request: ModalTrainStatusRequest) -> CloudModalResponse:
    """Fetch status for managed Modal training call."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_modal_service().train_status(request)
    except CloudModalError as exc:
        raise PackageError(str(exc)) from exc
    except Exception as exc:
        raise PackageError(f"cloud_modal_unexpected_error:{exc}") from exc


def cloud_modal_train_logs(request: ModalTrainLogsRequest) -> CloudModalResponse:
    """Fetch logs metadata for managed Modal training call."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_modal_service().train_logs(request)
    except CloudModalError as exc:
        raise PackageError(str(exc)) from exc
    except Exception as exc:
        raise PackageError(f"cloud_modal_unexpected_error:{exc}") from exc


def cloud_modal_train_cancel(request: ModalTrainCancelRequest) -> CloudModalResponse:
    """Cancel a managed Modal training call."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_modal_service().train_cancel(request)
    except CloudModalError as exc:
        raise PackageError(str(exc)) from exc
    except Exception as exc:
        raise PackageError(f"cloud_modal_unexpected_error:{exc}") from exc


def cloud_modal_train_pull(request: ModalTrainPullRequest) -> CloudModalResponse:
    """Fetch output metadata for managed Modal training call."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_modal_service().train_pull(request)
    except CloudModalError as exc:
        raise PackageError(str(exc)) from exc
    except Exception as exc:
        raise PackageError(f"cloud_modal_unexpected_error:{exc}") from exc


__all__ = [
    "cloud_modal_data_sync",
    "cloud_modal_deploy",
    "cloud_modal_train_cancel",
    "cloud_modal_train_logs",
    "cloud_modal_train_pull",
    "cloud_modal_train_status",
    "cloud_modal_train_submit",
]
