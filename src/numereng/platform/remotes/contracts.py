"""Typed contracts for user-local remote monitoring targets."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class RemoteTargetError(ValueError):
    """Remote target configuration could not be loaded or validated."""


class SshRemoteTargetProfile(BaseModel):
    """One SSH-backed numereng store that can be monitored read-only."""

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    kind: Literal["ssh"] = "ssh"
    ssh_config_host: str | None = None
    host: str | None = None
    user_env: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    identity_file_env: str | None = None
    repo_root: str = Field(min_length=1)
    store_root: str = Field(min_length=1)
    runner_cmd: str = Field(default="uv run numereng", min_length=1)
    enabled: bool = True
    tags: list[str] = Field(default_factory=list)
    connect_timeout_seconds: int = Field(default=5, ge=1, le=60)
    command_timeout_seconds: int = Field(default=15, ge=1, le=300)

    @model_validator(mode="after")
    def _validate_host_source(self) -> SshRemoteTargetProfile:
        if bool(self.ssh_config_host) == bool(self.host):
            raise ValueError("exactly one of ssh_config_host or host is required")
        return self


RemoteTargetProfile = SshRemoteTargetProfile

__all__ = [
    "RemoteTargetError",
    "RemoteTargetProfile",
    "SshRemoteTargetProfile",
]
