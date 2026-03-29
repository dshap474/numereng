"""Typed contracts for the EC2 cloud lifecycle feature."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from numereng.config.training import ensure_json_config_path
from numereng.features.cloud.aws.managed_contracts import CloudRuntimeProfile


class CloudEc2State(BaseModel):
    """Serializable state payload that can be passed between CLI steps."""

    run_id: str | None = None
    external_run_id: str | None = None
    canonical_run_id: str | None = None
    instance_id: str | None = None
    region: str | None = None
    bucket: str | None = None
    instance_type: str | None = None
    is_gpu: bool = False
    runtime_profile: CloudRuntimeProfile = "standard"
    data_version: str | None = None
    training_pid: int | None = None
    status: str = "unknown"
    artifacts: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    last_updated_at: str | None = None

    def touched(self) -> CloudEc2State:
        """Return a copy with a refreshed timestamp."""
        return self.model_copy(
            update={"last_updated_at": datetime.now(tz=UTC).isoformat()},
            deep=True,
        )


class CloudEc2Response(BaseModel):
    """Generic JSON response shape for cloud EC2 operations."""

    action: str
    message: str
    state: CloudEc2State | None = None
    result: dict[str, Any] = Field(default_factory=dict)


class CloudEc2RequestBase(BaseModel):
    """Common request fields shared by step commands."""

    state_path: str | None = None
    store_root: str = ".numereng"

    def state_file(self) -> Path | None:
        if self.state_path is None:
            return None
        return Path(self.state_path)


class Ec2InitIamRequest(BaseModel):
    """Request payload for IAM bootstrap."""

    region: str | None = None
    bucket: str | None = None
    role_name: str | None = None
    security_group_name: str | None = None


class Ec2SetupDataRequest(BaseModel):
    """Request payload for data sync to S3."""

    cache_dir: str | None = None
    data_version: str = "v5.2"
    region: str | None = None
    bucket: str | None = None


class Ec2ProvisionRequest(CloudEc2RequestBase):
    """Request payload for instance provisioning."""

    tier: str = "large"
    run_id: str | None = None
    region: str | None = None
    bucket: str | None = None
    data_version: str = "v5.2"
    use_spot: bool = True


class Ec2PackageBuildUploadRequest(CloudEc2RequestBase):
    """Request payload for wheel/requirements build and upload."""

    run_id: str | None = None
    region: str | None = None
    bucket: str | None = None


class Ec2ConfigUploadRequest(CloudEc2RequestBase):
    """Request payload for config upload."""

    run_id: str | None = None
    config_path: str
    region: str | None = None
    bucket: str | None = None

    @field_validator("config_path")
    @classmethod
    def _validate_config_path(cls, value: str) -> str:
        return ensure_json_config_path(value, field_name="config_path")


class Ec2PushRequest(CloudEc2RequestBase):
    """Request payload for downloading package/config/data to remote instance."""

    run_id: str | None = None
    instance_id: str | None = None
    region: str | None = None
    bucket: str | None = None
    data_version: str | None = None


class Ec2InstallRequest(CloudEc2RequestBase):
    """Request payload for runtime installation on remote instance."""

    run_id: str | None = None
    instance_id: str | None = None
    region: str | None = None
    runtime_profile: CloudRuntimeProfile | None = None

    @field_validator("runtime_profile", mode="before")
    @classmethod
    def _normalize_runtime_profile(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip().lower()
            return stripped or None
        return value


class Ec2TrainStartRequest(CloudEc2RequestBase):
    """Request payload for launching remote training."""

    run_id: str | None = None
    instance_id: str | None = None
    region: str | None = None


class Ec2TrainPollRequest(CloudEc2RequestBase):
    """Request payload for polling remote training status."""

    run_id: str | None = None
    instance_id: str | None = None
    region: str | None = None
    timeout_seconds: int = 7200
    interval_seconds: int = 20


class Ec2LogsRequest(CloudEc2RequestBase):
    """Request payload for tailing remote logs."""

    instance_id: str | None = None
    region: str | None = None
    lines: int = 100
    follow: bool = False


class Ec2PullRequest(CloudEc2RequestBase):
    """Request payload for pulling completed run outputs."""

    run_id: str | None = None
    instance_id: str | None = None
    region: str | None = None
    bucket: str | None = None
    output_dir: str | None = None


class Ec2TerminateRequest(CloudEc2RequestBase):
    """Request payload for instance termination."""

    instance_id: str | None = None
    region: str | None = None


class Ec2StatusRequest(CloudEc2RequestBase):
    """Request payload for run/instance status lookup."""

    run_id: str | None = None
    region: str | None = None


class Ec2S3ListRequest(BaseModel):
    """Request payload for listing S3 objects."""

    prefix: str
    region: str | None = None
    bucket: str | None = None


class Ec2S3CopyRequest(BaseModel):
    """Request payload for copying between local path and S3 URI."""

    src: str
    dst: str
    region: str | None = None
    bucket: str | None = None


class Ec2S3RemoveRequest(BaseModel):
    """Request payload for deleting S3 objects."""

    uri: str
    recursive: bool = False
    region: str | None = None
    bucket: str | None = None
