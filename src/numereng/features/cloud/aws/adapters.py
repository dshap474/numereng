"""Adapter protocols for AWS cloud lifecycle orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(slots=True)
class LaunchInstanceSpec:
    """EC2 launch parameters for a training worker."""

    image_id: str
    instance_type: str
    user_data: str
    run_id: str
    region: str
    iam_role_name: str | None
    security_group: str | None
    bucket: str
    data_version: str
    use_spot: bool
    volume_size_gb: int = 100
    key_name: str | None = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class InstanceStatus:
    """Normalized instance status payload."""

    instance_id: str
    state: str
    instance_type: str
    run_id: str | None = None
    public_ip: str | None = None
    private_ip: str | None = None
    launch_time: str | None = None


@dataclass(slots=True)
class SsmCommandResult:
    """Result payload for SSM command executions."""

    exit_code: int
    stdout: str
    stderr: str


class Ec2Adapter(Protocol):
    """EC2 control-plane adapter."""

    def launch_instance(self, spec: LaunchInstanceSpec) -> str:
        """Launch an instance and return its instance ID."""

    def wait_for_instance(self, instance_id: str, target_state: str, timeout_seconds: int) -> bool:
        """Wait until instance reaches target state."""

    def terminate_instance(self, instance_id: str) -> None:
        """Terminate an instance."""

    def get_instance_status(self, instance_id: str) -> InstanceStatus:
        """Get status for one instance."""

    def list_training_instances(self) -> list[InstanceStatus]:
        """List non-terminated training instances."""

    def get_spot_price(self, instance_type: str) -> float | None:
        """Get latest spot price for an instance type."""

    def resolve_security_group_id(self, security_group: str) -> str | None:
        """Resolve a security-group name to id."""


class S3Adapter(Protocol):
    """S3 data-plane adapter."""

    def ensure_bucket_exists(self, bucket: str, region: str) -> None:
        """Ensure bucket exists."""

    def upload_file(self, local_path: Path, bucket: str, key: str) -> str:
        """Upload one file and return S3 URI."""

    def download_file(self, bucket: str, key: str, local_path: Path) -> Path:
        """Download one file from S3."""

    def list_keys(self, bucket: str, prefix: str) -> list[str]:
        """List object keys under a prefix."""

    def delete_key(self, bucket: str, key: str) -> None:
        """Delete one object key."""

    def delete_prefix(self, bucket: str, prefix: str) -> int:
        """Delete objects under prefix and return count."""

    def copy_object(
        self,
        src_bucket: str,
        src_key: str,
        dst_bucket: str,
        dst_key: str,
    ) -> str:
        """Copy an object and return destination URI."""


class SsmAdapter(Protocol):
    """SSM command execution adapter."""

    def wait_for_ssm(self, instance_id: str, timeout_seconds: int) -> None:
        """Wait for SSM ping online."""

    def run_command(
        self,
        instance_id: str,
        command: str,
        timeout_seconds: int,
    ) -> SsmCommandResult:
        """Run shell command on remote instance."""


class IamAdapter(Protocol):
    """IAM and network setup adapter."""

    def ensure_training_role(self, role_name: str, bucket: str) -> str:
        """Ensure role and policies exist, return role ARN."""

    def ensure_instance_profile(self, role_name: str) -> str:
        """Ensure instance profile exists, return profile ARN."""

    def ensure_security_group(self, region: str, group_name: str) -> str:
        """Ensure security group exists, return group id."""


@dataclass(slots=True)
class SageMakerTrainingSpec:
    """SageMaker training-job creation parameters."""

    job_name: str
    image_uri: str
    role_arn: str
    input_config_uri: str
    output_s3_uri: str
    checkpoint_s3_uri: str | None
    instance_type: str
    instance_count: int
    volume_size_gb: int
    max_runtime_seconds: int
    max_wait_seconds: int | None
    use_spot: bool
    environment: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class SageMakerTrainingStatus:
    """Normalized SageMaker training-job status payload."""

    job_name: str
    job_arn: str | None
    status: str
    secondary_status: str | None = None
    failure_reason: str | None = None
    output_s3_uri: str | None = None
    log_group: str | None = None
    log_stream_prefix: str | None = None


class SageMakerAdapter(Protocol):
    """SageMaker control-plane adapter."""

    def start_training(self, spec: SageMakerTrainingSpec) -> SageMakerTrainingStatus:
        """Create one training job."""

    def describe_training(self, job_name: str) -> SageMakerTrainingStatus:
        """Read one training job status."""

    def stop_training(self, job_name: str) -> None:
        """Stop one training job."""


@dataclass(slots=True)
class BatchJobSpec:
    """AWS Batch job submission parameters."""

    job_name: str
    job_queue: str
    job_definition: str
    parameters: dict[str, str] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class BatchJobStatus:
    """Normalized AWS Batch job status payload."""

    job_id: str
    status: str
    status_reason: str | None = None
    log_stream_name: str | None = None


class BatchAdapter(Protocol):
    """AWS Batch control-plane adapter."""

    def submit_job(self, spec: BatchJobSpec) -> str:
        """Submit one batch job and return id."""

    def describe_job(self, job_id: str) -> BatchJobStatus:
        """Read one batch job status."""

    def cancel_job(self, job_id: str, *, reason: str) -> None:
        """Cancel one batch job."""

    def terminate_job(self, job_id: str, *, reason: str) -> None:
        """Terminate one running batch job."""


@dataclass(slots=True)
class CloudLogLine:
    """Normalized CloudWatch log event payload."""

    timestamp_ms: int
    message: str


class CloudWatchLogsAdapter(Protocol):
    """CloudWatch logs adapter."""

    def list_stream_names(
        self,
        *,
        log_group: str,
        stream_prefix: str,
        limit: int,
    ) -> list[str]:
        """List streams by prefix ordered by recent events."""

    def fetch_log_events(
        self,
        *,
        log_group: str,
        stream_name: str,
        limit: int,
        next_token: str | None = None,
        start_from_head: bool = False,
    ) -> tuple[list[CloudLogLine], str | None]:
        """Fetch one page of log events."""


class EcrAdapter(Protocol):
    """Amazon ECR adapter."""

    def ensure_repository(self, repository_name: str) -> str:
        """Ensure one ECR repository exists and return repository URI."""

    def get_account_id(self) -> str:
        """Return current caller account id."""

    def get_login_password(self) -> str:
        """Return docker login password for ECR."""

    def image_uri(self, repository_name: str, image_tag: str) -> str:
        """Build fully-qualified image URI for repository/tag."""

    def get_image_digest(self, repository_name: str, image_tag: str) -> str | None:
        """Return image digest if tag exists."""


class DockerAdapter(Protocol):
    """Docker CLI adapter."""

    def build_image(
        self,
        *,
        context_dir: Path,
        tag: str,
        dockerfile: Path | None = None,
        build_args: dict[str, str] | None = None,
        platform: str | None = None,
    ) -> None:
        """Build one docker image."""

    def tag_image(self, *, source_tag: str, target_tag: str) -> None:
        """Tag image from source -> target."""

    def push_image(self, *, tag: str) -> None:
        """Push one docker image tag."""

    def login(self, *, registry: str, username: str, password: str) -> None:
        """Docker login against one registry."""


class WheelBuilder(Protocol):
    """Builds wheel/requirements artifacts for remote install."""

    def build_assets(self, output_dir: Path) -> list[Path]:
        """Build package assets in output_dir and return generated paths."""
