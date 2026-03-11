"""Typed contracts for managed AWS training lifecycle commands."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from numereng.config.training import ensure_json_config_path, ensure_json_config_uri

CloudRuntimeProfile = Literal["standard", "lgbm-cuda"]


class CloudAwsState(BaseModel):
    """Serializable state payload for managed AWS command chaining."""

    run_id: str | None = None
    backend: Literal["sagemaker", "batch"] | None = None
    region: str | None = None
    bucket: str | None = None
    repository: str | None = None
    image_tag: str | None = None
    image_uri: str | None = None
    runtime_profile: CloudRuntimeProfile = "standard"
    training_job_name: str | None = None
    training_job_arn: str | None = None
    batch_job_id: str | None = None
    status: str = "unknown"
    artifacts: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    last_updated_at: str | None = None

    def touched(self) -> CloudAwsState:
        """Return a copy with refreshed timestamp."""
        return self.model_copy(update={"last_updated_at": datetime.now(tz=UTC).isoformat()}, deep=True)


class CloudAwsResponse(BaseModel):
    """Generic JSON response shape for managed AWS operations."""

    action: str
    message: str
    state: CloudAwsState | None = None
    result: dict[str, Any] = Field(default_factory=dict)


class CloudAwsRequestBase(BaseModel):
    """Common request fields for managed AWS command set."""

    state_path: str | None = None
    store_root: str = ".numereng"

    def state_file(self) -> Path | None:
        if self.state_path is None:
            return None
        return Path(self.state_path)


class AwsImageBuildPushRequest(CloudAwsRequestBase):
    """Request payload for build + push image to ECR."""

    run_id: str | None = None
    region: str | None = None
    bucket: str | None = None
    repository: str | None = None
    image_tag: str | None = None
    context_dir: str = "."
    dockerfile: str | None = None
    runtime_profile: CloudRuntimeProfile = "standard"
    build_args: dict[str, str] = Field(default_factory=dict)
    platform: str | None = None

    @field_validator("runtime_profile", mode="before")
    @classmethod
    def _normalize_runtime_profile(cls, value: object) -> object:
        if value is None:
            return "standard"
        if isinstance(value, str):
            stripped = value.strip().lower()
            if stripped:
                return stripped
        return value


class AwsTrainSubmitRequest(CloudAwsRequestBase):
    """Request payload for managed train submission."""

    run_id: str | None = None
    backend: Literal["sagemaker", "batch"] = "sagemaker"
    region: str | None = None
    bucket: str | None = None
    config_path: str | None = None
    config_s3_uri: str | None = None
    image_uri: str | None = None
    runtime_profile: CloudRuntimeProfile | None = None
    role_arn: str | None = None
    instance_type: str = "ml.m5.2xlarge"
    instance_count: int = Field(default=1, ge=1)
    volume_size_gb: int = Field(default=100, ge=1)
    max_runtime_seconds: int = Field(default=14_400, ge=60)
    max_wait_seconds: int | None = Field(default=None, ge=60)
    use_spot: bool = True
    checkpoint_s3_uri: str | None = None
    output_s3_uri: str | None = None
    batch_job_queue: str | None = None
    batch_job_definition: str | None = None
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator("config_path")
    @classmethod
    def _validate_config_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return ensure_json_config_path(value, field_name="config_path")

    @field_validator("config_s3_uri")
    @classmethod
    def _validate_config_s3_uri(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return ensure_json_config_uri(value, field_name="config_s3_uri")

    @field_validator("runtime_profile", mode="before")
    @classmethod
    def _normalize_runtime_profile(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip().lower()
            return stripped or None
        return value

    @model_validator(mode="after")
    def _validate_payload(self) -> AwsTrainSubmitRequest:
        if self.config_path is None and self.config_s3_uri is None:
            raise ValueError("either config_path or config_s3_uri is required")
        if self.max_wait_seconds is not None and self.max_wait_seconds < self.max_runtime_seconds:
            raise ValueError("max_wait_seconds must be >= max_runtime_seconds")
        return self


class AwsTrainStatusRequest(CloudAwsRequestBase):
    """Request payload for train status lookup."""

    backend: Literal["sagemaker", "batch"] | None = None
    run_id: str | None = None
    training_job_name: str | None = None
    batch_job_id: str | None = None
    region: str | None = None


class AwsTrainLogsRequest(CloudAwsRequestBase):
    """Request payload for managed train logs."""

    backend: Literal["sagemaker", "batch"] | None = None
    run_id: str | None = None
    training_job_name: str | None = None
    batch_job_id: str | None = None
    region: str | None = None
    lines: int = Field(default=100, ge=1, le=5000)
    follow: bool = False


class AwsTrainCancelRequest(CloudAwsRequestBase):
    """Request payload for canceling active train job."""

    backend: Literal["sagemaker", "batch"] | None = None
    run_id: str | None = None
    training_job_name: str | None = None
    batch_job_id: str | None = None
    region: str | None = None


class AwsTrainPullRequest(CloudAwsRequestBase):
    """Request payload for pulling managed outputs from S3."""

    run_id: str | None = None
    region: str | None = None
    bucket: str | None = None
    output_s3_uri: str | None = None
    output_dir: str | None = None


class AwsTrainExtractRequest(CloudAwsRequestBase):
    """Request payload for extracting pulled managed outputs into the run store."""

    run_id: str | None = None
    region: str | None = None
    bucket: str | None = None
    output_dir: str | None = None
