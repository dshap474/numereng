"""AWS managed training API handlers."""

from __future__ import annotations

from numereng.features.cloud.aws import (
    AwsImageBuildPushRequest,
    AwsTrainCancelRequest,
    AwsTrainExtractRequest,
    AwsTrainLogsRequest,
    AwsTrainPullRequest,
    AwsTrainStatusRequest,
    AwsTrainSubmitRequest,
    CloudAwsError,
    CloudAwsResponse,
)
from numereng.platform.errors import PackageError


def cloud_aws_image_build_push(request: AwsImageBuildPushRequest) -> CloudAwsResponse:
    """Build and push docker image to ECR."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_aws_managed_service().image_build_push(request)
    except CloudAwsError as exc:
        raise PackageError(str(exc)) from exc
    except Exception as exc:
        raise PackageError(f"cloud_aws_unexpected_error:{exc}") from exc


def cloud_aws_train_submit(request: AwsTrainSubmitRequest) -> CloudAwsResponse:
    """Submit managed training job to SageMaker or Batch."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_aws_managed_service().train_submit(request)
    except CloudAwsError as exc:
        raise PackageError(str(exc)) from exc
    except Exception as exc:
        raise PackageError(f"cloud_aws_unexpected_error:{exc}") from exc


def cloud_aws_train_status(request: AwsTrainStatusRequest) -> CloudAwsResponse:
    """Fetch status for managed training job."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_aws_managed_service().train_status(request)
    except CloudAwsError as exc:
        raise PackageError(str(exc)) from exc
    except Exception as exc:
        raise PackageError(f"cloud_aws_unexpected_error:{exc}") from exc


def cloud_aws_train_logs(request: AwsTrainLogsRequest) -> CloudAwsResponse:
    """Fetch CloudWatch logs for managed training job."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_aws_managed_service().train_logs(request)
    except CloudAwsError as exc:
        raise PackageError(str(exc)) from exc
    except Exception as exc:
        raise PackageError(f"cloud_aws_unexpected_error:{exc}") from exc


def cloud_aws_train_cancel(request: AwsTrainCancelRequest) -> CloudAwsResponse:
    """Cancel managed training job."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_aws_managed_service().train_cancel(request)
    except CloudAwsError as exc:
        raise PackageError(str(exc)) from exc
    except Exception as exc:
        raise PackageError(f"cloud_aws_unexpected_error:{exc}") from exc


def cloud_aws_train_pull(request: AwsTrainPullRequest) -> CloudAwsResponse:
    """Pull managed training outputs from S3."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_aws_managed_service().train_pull(request)
    except CloudAwsError as exc:
        raise PackageError(str(exc)) from exc
    except Exception as exc:
        raise PackageError(f"cloud_aws_unexpected_error:{exc}") from exc


def cloud_aws_train_extract(request: AwsTrainExtractRequest) -> CloudAwsResponse:
    """Extract pulled managed training outputs into local run store."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_aws_managed_service().train_extract(request)
    except CloudAwsError as exc:
        raise PackageError(str(exc)) from exc
    except Exception as exc:
        raise PackageError(f"cloud_aws_unexpected_error:{exc}") from exc


__all__ = [
    "cloud_aws_image_build_push",
    "cloud_aws_train_cancel",
    "cloud_aws_train_extract",
    "cloud_aws_train_logs",
    "cloud_aws_train_pull",
    "cloud_aws_train_status",
    "cloud_aws_train_submit",
]
