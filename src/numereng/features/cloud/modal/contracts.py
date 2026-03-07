"""Typed contracts for Modal cloud training lifecycle commands."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from numereng.config.training import ensure_json_config_path

ModalTrainingProfile = Literal["simple", "purged_walk_forward", "full_history_refit"]
ModalTrainingEngineMode = Literal["official", "custom", "full_history"]
ModalCallStatus = Literal[
    "unknown",
    "deployed",
    "submitted",
    "running",
    "completed",
    "failed",
    "cancelled",
    "expired",
]

_ECR_IMAGE_URI_RE = re.compile(
    r"^(?P<account_id>\d{12})\.dkr\.ecr\.(?P<region>[a-z0-9-]+)\.amazonaws\.com/"
    r"(?P<repository>[a-z0-9._\-/]+):(?P<tag>[A-Za-z0-9._\-]+)$"
)


class ModalEcrImageRef(BaseModel):
    """Parsed components for a validated ECR image URI."""

    account_id: str
    region: str
    repository: str
    tag: str


def parse_ecr_image_uri(value: str) -> ModalEcrImageRef:
    """Validate and parse an ECR image URI in `<registry>/<repo>:<tag>` form."""
    candidate = value.strip()
    if not candidate or "@" in candidate:
        raise ValueError(
            "invalid ecr_image_uri: expected 123456789012.dkr.ecr.<region>.amazonaws.com/<repository>:<tag>"
        )
    match = _ECR_IMAGE_URI_RE.match(candidate)
    if match is None:
        raise ValueError(
            "invalid ecr_image_uri: expected 123456789012.dkr.ecr.<region>.amazonaws.com/<repository>:<tag>"
        )
    return ModalEcrImageRef(
        account_id=match.group("account_id"),
        region=match.group("region"),
        repository=match.group("repository"),
        tag=match.group("tag"),
    )


class CloudModalState(BaseModel):
    """Serializable state payload for managed Modal command chaining."""

    run_id: str | None = None
    call_id: str | None = None
    app_name: str | None = None
    function_name: str | None = None
    environment_name: str | None = None
    ecr_image_uri: str | None = None
    data_volume_name: str | None = None
    deployment_id: str | None = None
    app_page_url: str | None = None
    app_logs_url: str | None = None
    status: ModalCallStatus = "unknown"
    artifacts: dict[str, str] = Field(default_factory=dict)
    data_manifest: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    last_updated_at: str | None = None

    def touched(self) -> CloudModalState:
        """Return a copy with refreshed timestamp."""
        return self.model_copy(update={"last_updated_at": datetime.now(tz=UTC).isoformat()}, deep=True)


class CloudModalResponse(BaseModel):
    """Generic JSON response shape for managed Modal operations."""

    action: str
    message: str
    state: CloudModalState | None = None
    result: dict[str, Any] = Field(default_factory=dict)


class CloudModalRequestBase(BaseModel):
    """Common request fields shared by Modal command set."""

    state_path: str | None = None

    def state_file(self) -> Path | None:
        if self.state_path is None:
            return None
        return Path(self.state_path)


class ModalTrainSubmitRequest(CloudModalRequestBase):
    """Request payload for managed Modal training submission."""

    config_path: str
    output_dir: str | None = None
    profile: ModalTrainingProfile | None = None
    # Legacy compatibility keys retained for migrating callers.
    engine_mode: ModalTrainingEngineMode | None = None
    window_size_eras: int | None = Field(default=None, ge=1)
    embargo_eras: int | None = Field(default=None, ge=1)
    app_name: str = "numereng-train"
    function_name: str = "train_remote"
    environment_name: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("config_path")
    @classmethod
    def _validate_config_path(cls, value: str) -> str:
        return ensure_json_config_path(value, field_name="config_path")

    @field_validator("profile", mode="before")
    @classmethod
    def _reject_submission_profile(cls, value: object) -> object:
        if value is not None and str(value) == "submission":
            raise ValueError("training profile 'submission' was renamed to 'full_history_refit'")
        return value


class ModalTrainStatusRequest(CloudModalRequestBase):
    """Request payload for managed Modal training status lookup."""

    call_id: str | None = None
    timeout_seconds: float = Field(default=0.0, ge=0)


class ModalTrainLogsRequest(CloudModalRequestBase):
    """Request payload for managed Modal training logs lookup."""

    call_id: str | None = None
    lines: int = Field(default=200, ge=1, le=5000)


class ModalTrainCancelRequest(CloudModalRequestBase):
    """Request payload for canceling one managed Modal training call."""

    call_id: str | None = None


class ModalTrainPullRequest(CloudModalRequestBase):
    """Request payload for retrieving managed Modal training outputs."""

    call_id: str | None = None
    output_dir: str | None = None
    timeout_seconds: float | None = Field(default=0.0, ge=0)


class ModalDeployRequest(CloudModalRequestBase):
    """Request payload for deploying Modal training function from ECR."""

    app_name: str = "numereng-train"
    function_name: str = "train_remote"
    ecr_image_uri: str
    data_volume_name: str | None = None
    environment_name: str | None = None
    aws_profile: str | None = None
    timeout_seconds: int | None = Field(default=None, ge=1)
    gpu: str | None = None
    cpu: float | None = Field(default=None, gt=0)
    memory_mb: int | None = Field(default=None, ge=128)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("ecr_image_uri")
    @classmethod
    def _validate_ecr_image_uri(cls, value: str) -> str:
        parse_ecr_image_uri(value)
        return value.strip()

    @field_validator("data_volume_name")
    @classmethod
    def _validate_data_volume_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("data_volume_name must not be empty when provided")
        return stripped


class ModalDeployResult(BaseModel):
    """Serializable deploy result returned by the Modal adapter."""

    deployed: bool
    deployment_id: str | None = None
    app_page_url: str | None = None
    app_logs_url: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ModalDataSyncFile(BaseModel):
    """One local-to-volume file mapping for Modal data sync."""

    source_path: str
    remote_path: str
    size_bytes: int = Field(default=0, ge=0)


class ModalDataSyncRequest(CloudModalRequestBase):
    """Request payload for syncing required training data files to Modal Volume."""

    config_path: str
    volume_name: str
    create_if_missing: bool = True
    force: bool = False
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("config_path")
    @classmethod
    def _validate_config_path(cls, value: str) -> str:
        return ensure_json_config_path(value, field_name="config_path")

    @field_validator("volume_name")
    @classmethod
    def _validate_volume_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("volume_name must not be empty")
        return stripped


class ModalDataSyncResult(BaseModel):
    """Serializable sync result returned by the Modal adapter."""

    volume_name: str
    uploaded_files: list[str] = Field(default_factory=list)
    file_count: int = Field(default=0, ge=0)
    total_bytes: int = Field(default=0, ge=0)
    create_if_missing: bool = True
    force: bool = False


class ModalRuntimePayload(BaseModel):
    """Serialized remote runtime input passed to Modal worker function."""

    config_text: str
    config_filename: str = "train.json"
    output_dir: str | None = None
    profile: ModalTrainingProfile | None = None
    # Legacy compatibility keys retained for migrating callers.
    engine_mode: ModalTrainingEngineMode | None = None
    window_size_eras: int | None = Field(default=None, ge=1)
    embargo_eras: int | None = Field(default=None, ge=1)

    @field_validator("config_filename")
    @classmethod
    def _validate_config_filename(cls, value: str) -> str:
        return ensure_json_config_path(value, field_name="config_filename")

    @field_validator("profile", mode="before")
    @classmethod
    def _reject_submission_profile(cls, value: object) -> object:
        if value is not None and str(value) == "submission":
            raise ValueError("training profile 'submission' was renamed to 'full_history_refit'")
        return value


class ModalRuntimeResult(BaseModel):
    """Serialized remote runtime result returned by Modal worker function."""

    run_id: str
    predictions_path: str
    results_path: str
    output_dir: str
