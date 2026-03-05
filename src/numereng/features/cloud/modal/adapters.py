"""Adapter protocols and errors for Modal cloud lifecycle orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from numereng.features.cloud.modal.contracts import (
    ModalDataSyncFile,
    ModalDataSyncResult,
    ModalDeployResult,
    ModalRuntimePayload,
    ModalRuntimeResult,
)


class ModalAdapterError(Exception):
    """Base adapter-layer error for Modal provider interactions."""


class ModalCallPendingError(ModalAdapterError):
    """Raised when a Modal call has not completed yet."""


class ModalCallCancelledError(ModalAdapterError):
    """Raised when a Modal call was cancelled."""


class ModalCallNotFoundError(ModalAdapterError):
    """Raised when a Modal call id cannot be found or was evicted."""


class ModalRemoteExecutionError(ModalAdapterError):
    """Raised when remote call execution completed with an error."""


class ModalCallHandle(Protocol):
    """Handle for a Modal function call instance."""

    @property
    def object_id(self) -> str:
        """Return provider call identifier."""

    def get(self, *, timeout_seconds: float | None = None) -> ModalRuntimeResult:
        """Get call result, optionally waiting up to `timeout_seconds`."""

    def cancel(self) -> None:
        """Cancel the provider call."""


class ModalTrainingAdapter(Protocol):
    """Provider adapter protocol for Modal training call lifecycle."""

    def submit_training(
        self,
        *,
        app_name: str,
        function_name: str,
        payload: ModalRuntimePayload,
        environment_name: str | None = None,
    ) -> ModalCallHandle:
        """Submit one remote training call."""

    def lookup_call(self, call_id: str) -> ModalCallHandle:
        """Lookup an existing remote training call by id."""

    def sync_data(
        self,
        *,
        volume_name: str,
        files: Sequence[ModalDataSyncFile],
        create_if_missing: bool = True,
        force: bool = False,
    ) -> ModalDataSyncResult:
        """Upload data files into one Modal Volume."""

    def deploy_training(
        self,
        *,
        app_name: str,
        function_name: str,
        ecr_image_uri: str,
        data_volume_name: str | None = None,
        environment_name: str | None = None,
        aws_profile: str | None = None,
        timeout_seconds: int | None = None,
        gpu: str | None = None,
        cpu: float | None = None,
        memory_mb: int | None = None,
    ) -> ModalDeployResult:
        """Deploy or update a remote training function backed by ECR image."""
