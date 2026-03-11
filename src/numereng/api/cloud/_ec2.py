"""EC2 cloud API handlers."""

from __future__ import annotations

from numereng.features.cloud.aws import (
    CloudEc2Error,
    CloudEc2Response,
    Ec2ConfigUploadRequest,
    Ec2InitIamRequest,
    Ec2InstallRequest,
    Ec2LogsRequest,
    Ec2PackageBuildUploadRequest,
    Ec2ProvisionRequest,
    Ec2PullRequest,
    Ec2PushRequest,
    Ec2S3CopyRequest,
    Ec2S3ListRequest,
    Ec2S3RemoveRequest,
    Ec2SetupDataRequest,
    Ec2StatusRequest,
    Ec2TerminateRequest,
    Ec2TrainPollRequest,
    Ec2TrainStartRequest,
)
from numereng.platform.errors import PackageError


def cloud_ec2_init_iam(request: Ec2InitIamRequest) -> CloudEc2Response:
    """Initialize IAM role, instance profile, and default security group."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_ec2_service().init_iam(request)
    except CloudEc2Error as exc:
        raise PackageError(str(exc)) from exc


def cloud_ec2_setup_data(request: Ec2SetupDataRequest) -> CloudEc2Response:
    """Sync local dataset cache files to S3 for EC2 training workers."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_ec2_service().setup_data(request)
    except CloudEc2Error as exc:
        raise PackageError(str(exc)) from exc


def cloud_ec2_provision(request: Ec2ProvisionRequest) -> CloudEc2Response:
    """Provision and bootstrap an EC2 worker instance."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_ec2_service().provision(request)
    except CloudEc2Error as exc:
        raise PackageError(str(exc)) from exc


def cloud_ec2_package_build_upload(request: Ec2PackageBuildUploadRequest) -> CloudEc2Response:
    """Build numereng wheel/requirements and upload package assets to S3."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_ec2_service().package_build_upload(request)
    except CloudEc2Error as exc:
        raise PackageError(str(exc)) from exc


def cloud_ec2_config_upload(request: Ec2ConfigUploadRequest) -> CloudEc2Response:
    """Upload a run config file to the S3 run prefix."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_ec2_service().config_upload(request)
    except CloudEc2Error as exc:
        raise PackageError(str(exc)) from exc


def cloud_ec2_push(request: Ec2PushRequest) -> CloudEc2Response:
    """Push package/config/data assets to the provisioned EC2 worker."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_ec2_service().push(request)
    except CloudEc2Error as exc:
        raise PackageError(str(exc)) from exc


def cloud_ec2_install(request: Ec2InstallRequest) -> CloudEc2Response:
    """Install Python runtime dependencies and numereng wheel on remote worker."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_ec2_service().install(request)
    except CloudEc2Error as exc:
        raise PackageError(str(exc)) from exc


def cloud_ec2_train_start(request: Ec2TrainStartRequest) -> CloudEc2Response:
    """Launch remote training process on EC2 worker."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_ec2_service().train_start(request)
    except CloudEc2Error as exc:
        raise PackageError(str(exc)) from exc


def cloud_ec2_train_poll(request: Ec2TrainPollRequest) -> CloudEc2Response:
    """Poll remote training process completion state."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_ec2_service().train_poll(request)
    except CloudEc2Error as exc:
        raise PackageError(str(exc)) from exc


def cloud_ec2_logs(request: Ec2LogsRequest) -> CloudEc2Response:
    """Read remote training log tail via SSM."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_ec2_service().logs(request)
    except CloudEc2Error as exc:
        raise PackageError(str(exc)) from exc


def cloud_ec2_pull(request: Ec2PullRequest) -> CloudEc2Response:
    """Sync run outputs from EC2 -> S3 -> local output directory."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_ec2_service().pull(request)
    except CloudEc2Error as exc:
        raise PackageError(str(exc)) from exc


def cloud_ec2_terminate(request: Ec2TerminateRequest) -> CloudEc2Response:
    """Terminate an EC2 worker instance."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_ec2_service().terminate(request)
    except CloudEc2Error as exc:
        raise PackageError(str(exc)) from exc


def cloud_ec2_status(request: Ec2StatusRequest) -> CloudEc2Response:
    """Get current instance/run status from cloud and optional local state doc."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_ec2_service().status(request)
    except CloudEc2Error as exc:
        raise PackageError(str(exc)) from exc


def cloud_ec2_s3_list(request: Ec2S3ListRequest) -> CloudEc2Response:
    """List S3 keys under a prefix."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_ec2_service().s3_list(request)
    except CloudEc2Error as exc:
        raise PackageError(str(exc)) from exc


def cloud_ec2_s3_copy(request: Ec2S3CopyRequest) -> CloudEc2Response:
    """Copy artifacts between local filesystem and S3 URIs."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_ec2_service().s3_copy(request)
    except CloudEc2Error as exc:
        raise PackageError(str(exc)) from exc


def cloud_ec2_s3_remove(request: Ec2S3RemoveRequest) -> CloudEc2Response:
    """Delete S3 object or prefix recursively."""
    from numereng import api as api_module

    try:
        return api_module._create_cloud_ec2_service().s3_remove(request)
    except CloudEc2Error as exc:
        raise PackageError(str(exc)) from exc


__all__ = [
    "cloud_ec2_config_upload",
    "cloud_ec2_init_iam",
    "cloud_ec2_install",
    "cloud_ec2_logs",
    "cloud_ec2_package_build_upload",
    "cloud_ec2_provision",
    "cloud_ec2_pull",
    "cloud_ec2_push",
    "cloud_ec2_s3_copy",
    "cloud_ec2_s3_list",
    "cloud_ec2_s3_remove",
    "cloud_ec2_setup_data",
    "cloud_ec2_status",
    "cloud_ec2_terminate",
    "cloud_ec2_train_poll",
    "cloud_ec2_train_start",
]
