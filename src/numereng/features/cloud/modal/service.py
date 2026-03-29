"""Service layer for managed Modal training workflows."""

from __future__ import annotations

import re
from pathlib import Path

from numereng.features.cloud.modal.adapters import (
    ModalAdapterError,
    ModalCallCancelledError,
    ModalCallNotFoundError,
    ModalCallPendingError,
    ModalRemoteExecutionError,
    ModalTrainingAdapter,
)
from numereng.features.cloud.modal.contracts import (
    CloudModalRequestBase,
    CloudModalResponse,
    CloudModalState,
    ModalCallStatus,
    ModalDataSyncRequest,
    ModalDeployRequest,
    ModalRuntimeResult,
    ModalTrainCancelRequest,
    ModalTrainLogsRequest,
    ModalTrainPullRequest,
    ModalTrainStatusRequest,
    ModalTrainSubmitRequest,
)
from numereng.features.cloud.modal.data_sync import MODAL_DATASETS_MOUNT_PATH, resolve_required_data_files
from numereng.features.cloud.modal.modal_adapters import SdkModalTrainingAdapter
from numereng.features.cloud.modal.runtime import build_runtime_payload_from_config
from numereng.features.cloud.modal.state_store import CloudModalStateStore
from numereng.features.store.layout import resolve_cloud_op_state_path, resolve_path, validate_cloud_state_path


class CloudModalError(Exception):
    """Feature-level error for managed Modal cloud orchestration."""


_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]{20,}")
_CLOUD_PROVIDER = "modal"


class CloudModalService:
    """Managed Modal service with pluggable provider adapter."""

    def __init__(
        self,
        *,
        adapter: ModalTrainingAdapter | None = None,
        state_store: CloudModalStateStore | None = None,
    ) -> None:
        self._adapter = adapter or SdkModalTrainingAdapter()
        self._state_store = state_store or CloudModalStateStore()

    def _resolved_state_path(
        self,
        request: CloudModalRequestBase,
        *,
        state: CloudModalState | None = None,
    ) -> Path | None:
        state_path = request.state_file()
        if state_path is not None:
            try:
                return validate_cloud_state_path(
                    target_path=state_path,
                    store_root=request.store_root,
                    error_code="modal_state_path_noncanonical",
                    allow_legacy_cloud=True,
                )
            except ValueError as exc:
                resolved = resolve_path(state_path)
                if resolved.suffix.lower() != ".json":
                    raise CloudModalError(f"modal_state_path_extension_invalid:{resolved}") from exc
                raise CloudModalError(str(exc)) from exc
        op_id = self._default_state_op_id(request, state=state)
        if op_id is None:
            return None
        return resolve_cloud_op_state_path(store_root=request.store_root, provider=_CLOUD_PROVIDER, op_id=op_id)

    def _default_state_op_id(
        self,
        request: CloudModalRequestBase,
        *,
        state: CloudModalState | None = None,
    ) -> str | None:
        call_id = getattr(request, "call_id", None) or (state.call_id if state is not None else None)
        if isinstance(call_id, str) and call_id.strip():
            return call_id
        if hasattr(request, "function_name") and hasattr(request, "app_name"):
            app_name = getattr(request, "app_name", None)
            function_name = getattr(request, "function_name", None)
            if isinstance(app_name, str) and app_name.strip() and isinstance(function_name, str) and function_name.strip():
                return f"deploy-{app_name}-{function_name}"
        if hasattr(request, "volume_name"):
            volume_name = getattr(request, "volume_name", None)
            if isinstance(volume_name, str) and volume_name.strip():
                return f"data-sync-{volume_name}"
        return None

    def _load_state(self, request: CloudModalRequestBase) -> CloudModalState:
        try:
            state_path = self._resolved_state_path(request)
            if state_path is None:
                return CloudModalState()
            loaded = self._state_store.load(state_path)
            if loaded is None:
                return CloudModalState()
            return loaded
        except Exception as exc:
            raise CloudModalError(f"modal_state_load_failed:{exc}") from exc

    def _persist_state(self, request: CloudModalRequestBase, state: CloudModalState) -> CloudModalState:
        touched = state.touched()
        try:
            state_path = self._resolved_state_path(request, state=touched)
            if state_path is not None:
                self._state_store.save(state_path, touched)
            return touched
        except Exception as exc:
            raise CloudModalError(f"modal_state_persist_failed:{exc}") from exc

    def _resolve_call_id(self, explicit: str | None, state: CloudModalState) -> str:
        if explicit is not None and explicit:
            return explicit
        if state.call_id is not None and state.call_id:
            return state.call_id
        raise CloudModalError("missing required value: call_id")

    def deploy(self, request: ModalDeployRequest) -> CloudModalResponse:
        state = self._load_state(request)
        data_volume_name = request.data_volume_name or state.data_volume_name
        try:
            deploy_result = self._adapter.deploy_training(
                app_name=request.app_name,
                function_name=request.function_name,
                ecr_image_uri=request.ecr_image_uri,
                data_volume_name=data_volume_name,
                environment_name=request.environment_name,
                aws_profile=request.aws_profile,
                timeout_seconds=request.timeout_seconds,
                gpu=request.gpu,
                cpu=request.cpu,
                memory_mb=request.memory_mb,
            )
        except ModalAdapterError as exc:
            raise CloudModalError(str(exc)) from exc

        metadata = dict(state.metadata)
        metadata.pop("error", None)
        metadata.update(request.metadata)
        metadata["ecr_image_uri"] = request.ecr_image_uri
        if request.aws_profile:
            metadata["aws_profile"] = request.aws_profile
        if data_volume_name:
            metadata["data_volume_name"] = data_volume_name
        if deploy_result.warnings:
            metadata["deploy_warning_count"] = str(len(deploy_result.warnings))

        next_state = state.model_copy(
            update={
                "run_id": None,
                "call_id": None,
                "app_name": request.app_name,
                "function_name": request.function_name,
                "environment_name": request.environment_name,
                "ecr_image_uri": request.ecr_image_uri,
                "data_volume_name": data_volume_name,
                "deployment_id": deploy_result.deployment_id,
                "app_page_url": deploy_result.app_page_url,
                "app_logs_url": deploy_result.app_logs_url,
                "status": "deployed",
                "artifacts": {},
                "metadata": metadata,
            },
            deep=True,
        )
        persisted = self._persist_state(request, next_state)
        return CloudModalResponse(
            action="cloud.modal.deploy",
            message="modal training function deployed",
            state=persisted,
            result={
                "deployed": deploy_result.deployed,
                "deployment_id": deploy_result.deployment_id,
                "app_name": request.app_name,
                "function_name": request.function_name,
                "ecr_image_uri": request.ecr_image_uri,
                "data_volume_name": data_volume_name,
                "environment_name": request.environment_name,
                "app_page_url": deploy_result.app_page_url,
                "app_logs_url": deploy_result.app_logs_url,
                "warnings": deploy_result.warnings,
                "data_mount_path": MODAL_DATASETS_MOUNT_PATH if data_volume_name else None,
            },
        )

    def data_sync(self, request: ModalDataSyncRequest) -> CloudModalResponse:
        state = self._load_state(request)
        resolved_config_path = Path(request.config_path).expanduser().resolve()
        try:
            data_version, dataset_variant, manifest = resolve_required_data_files(config_path=resolved_config_path)
            sync_result = self._adapter.sync_data(
                volume_name=request.volume_name,
                files=manifest,
                create_if_missing=request.create_if_missing,
                force=request.force,
            )
        except (OSError, ValueError, ModalAdapterError) as exc:
            raise CloudModalError(str(exc)) from exc

        metadata = dict(state.metadata)
        metadata.pop("error", None)
        metadata.update(request.metadata)
        metadata["data_sync_config_path"] = str(resolved_config_path)
        metadata["data_sync_data_version"] = data_version
        metadata["data_sync_dataset_variant"] = dataset_variant
        metadata["data_sync_file_count"] = str(sync_result.file_count)

        next_state = state.model_copy(
            update={
                "data_volume_name": sync_result.volume_name,
                "data_manifest": {item.remote_path: item.source_path for item in manifest},
                "metadata": metadata,
            },
            deep=True,
        )
        persisted = self._persist_state(request, next_state)
        return CloudModalResponse(
            action="cloud.modal.data.sync",
            message="modal data volume synced",
            state=persisted,
            result={
                "volume_name": sync_result.volume_name,
                "data_version": data_version,
                "dataset_variant": dataset_variant,
                "file_count": sync_result.file_count,
                "total_bytes": sync_result.total_bytes,
                "uploaded_files": sync_result.uploaded_files,
                "create_if_missing": sync_result.create_if_missing,
                "force": sync_result.force,
                "mount_path": MODAL_DATASETS_MOUNT_PATH,
                "config_path": str(resolved_config_path),
                "files": [item.model_dump(mode="json") for item in manifest],
            },
        )

    def train_submit(self, request: ModalTrainSubmitRequest) -> CloudModalResponse:
        state = self._load_state(request)
        try:
            payload = build_runtime_payload_from_config(
                config_path=request.config_path,
                output_dir=request.output_dir,
                profile=request.profile,
                engine_mode=request.engine_mode,
                window_size_eras=request.window_size_eras,
                embargo_eras=request.embargo_eras,
            )
            handle = self._adapter.submit_training(
                app_name=request.app_name,
                function_name=request.function_name,
                payload=payload,
                environment_name=request.environment_name,
            )
            call_id = handle.object_id
        except (OSError, ModalAdapterError) as exc:
            raise CloudModalError(str(exc)) from exc

        metadata = dict(state.metadata)
        metadata.pop("error", None)
        metadata.update(request.metadata)
        metadata["config_path"] = str(Path(request.config_path).expanduser())

        next_state = state.model_copy(
            update={
                "run_id": None,
                "call_id": call_id,
                "app_name": request.app_name,
                "function_name": request.function_name,
                "status": "submitted",
                "artifacts": {},
                "metadata": metadata,
            },
            deep=True,
        )
        try:
            persisted = self._persist_state(request, next_state)
        except CloudModalError as exc:
            raise CloudModalError(f"{exc}; modal_call_submitted_call_id:{call_id}") from exc
        return CloudModalResponse(
            action="cloud.modal.train.submit",
            message="modal training call submitted",
            state=persisted,
            result={
                "call_id": call_id,
                "status": persisted.status,
                "app_name": request.app_name,
                "function_name": request.function_name,
            },
        )

    def train_status(self, request: ModalTrainStatusRequest) -> CloudModalResponse:
        state = self._load_state(request)
        call_id = self._resolve_call_id(request.call_id, state)
        status, runtime_result, error_message = self._poll_call(call_id, timeout_seconds=request.timeout_seconds)

        next_state = state.model_copy(update={"call_id": call_id, "status": status}, deep=True)
        result: dict[str, object] = {"call_id": call_id, "status": status}
        if runtime_result is not None:
            next_state = self._apply_runtime_result(next_state, runtime_result)
            result.update(runtime_result.model_dump())
        if error_message is not None:
            sanitized_error = self._sanitize_error_message(error_message)
            metadata = dict(next_state.metadata)
            metadata["error"] = sanitized_error
            next_state = next_state.model_copy(update={"metadata": metadata}, deep=True)
            result["error"] = sanitized_error

        persisted = self._persist_state(request, next_state)
        return CloudModalResponse(
            action="cloud.modal.train.status",
            message=f"modal training status: {status}",
            state=persisted,
            result={key: self._coerce_json_value(value) for key, value in result.items()},
        )

    def train_logs(self, request: ModalTrainLogsRequest) -> CloudModalResponse:
        state = self._load_state(request)
        call_id = self._resolve_call_id(request.call_id, state)
        status, runtime_result, error_message = self._poll_call(call_id, timeout_seconds=0.0)

        next_state = state.model_copy(update={"call_id": call_id, "status": status}, deep=True)
        result: dict[str, object] = {
            "call_id": call_id,
            "status": status,
            "requested_lines": request.lines,
            "log_source_hint": f"view detailed logs in the Modal dashboard for call_id '{call_id}'",
        }
        if runtime_result is not None:
            next_state = self._apply_runtime_result(next_state, runtime_result)
            result["run_id"] = runtime_result.run_id
        if error_message is not None:
            sanitized_error = self._sanitize_error_message(error_message)
            metadata = dict(next_state.metadata)
            metadata["error"] = sanitized_error
            next_state = next_state.model_copy(update={"metadata": metadata}, deep=True)
            result["error"] = sanitized_error

        persisted = self._persist_state(request, next_state)
        return CloudModalResponse(
            action="cloud.modal.train.logs",
            message=f"modal training logs metadata: {status}",
            state=persisted,
            result={key: self._coerce_json_value(value) for key, value in result.items()},
        )

    def train_cancel(self, request: ModalTrainCancelRequest) -> CloudModalResponse:
        state = self._load_state(request)
        call_id = self._resolve_call_id(request.call_id, state)
        try:
            handle = self._adapter.lookup_call(call_id)
            handle.cancel()
        except ModalAdapterError as exc:
            raise CloudModalError(str(exc)) from exc

        next_state = state.model_copy(
            update={
                "call_id": call_id,
                "status": "cancelled",
            },
            deep=True,
        )
        persisted = self._persist_state(request, next_state)
        return CloudModalResponse(
            action="cloud.modal.train.cancel",
            message="modal training call cancelled",
            state=persisted,
            result={
                "call_id": call_id,
                "status": "cancelled",
            },
        )

    def train_pull(self, request: ModalTrainPullRequest) -> CloudModalResponse:
        state = self._load_state(request)
        call_id = self._resolve_call_id(request.call_id, state)
        status, runtime_result, error_message = self._poll_call(call_id, timeout_seconds=request.timeout_seconds)

        next_state = state.model_copy(update={"call_id": call_id, "status": status}, deep=True)
        result: dict[str, object] = {"call_id": call_id, "status": status}

        if runtime_result is not None:
            next_state = self._apply_runtime_result(next_state, runtime_result)
            result.update(runtime_result.model_dump())
            if request.output_dir is not None:
                result["requested_output_dir"] = request.output_dir
        if error_message is not None:
            sanitized_error = self._sanitize_error_message(error_message)
            metadata = dict(next_state.metadata)
            metadata["error"] = sanitized_error
            next_state = next_state.model_copy(update={"metadata": metadata}, deep=True)
            result["error"] = sanitized_error

        persisted = self._persist_state(request, next_state)
        message = "modal training output metadata described"
        if status == "running":
            message = "modal training call still running"
        elif status == "failed":
            message = "modal training call failed"
        elif status == "cancelled":
            message = "modal training call was cancelled"
        elif status == "expired":
            message = "modal training call no longer exists"
        return CloudModalResponse(
            action="cloud.modal.train.pull",
            message=message,
            state=persisted,
            result={key: self._coerce_json_value(value) for key, value in result.items()},
        )

    def _poll_call(
        self,
        call_id: str,
        *,
        timeout_seconds: float | None,
    ) -> tuple[ModalCallStatus, ModalRuntimeResult | None, str | None]:
        try:
            handle = self._adapter.lookup_call(call_id)
            runtime_result = handle.get(timeout_seconds=timeout_seconds)
        except ModalCallPendingError:
            return "running", None, None
        except ModalCallCancelledError:
            return "cancelled", None, None
        except ModalCallNotFoundError:
            return "expired", None, None
        except ModalRemoteExecutionError as exc:
            return "failed", None, str(exc)
        except ModalAdapterError as exc:
            raise CloudModalError(str(exc)) from exc
        return "completed", runtime_result, None

    def _apply_runtime_result(
        self,
        state: CloudModalState,
        runtime_result: ModalRuntimeResult,
    ) -> CloudModalState:
        artifacts = dict(state.artifacts)
        artifacts["predictions_path"] = runtime_result.predictions_path
        artifacts["results_path"] = runtime_result.results_path
        artifacts["output_dir"] = runtime_result.output_dir
        return state.model_copy(
            update={
                "run_id": runtime_result.run_id,
                "status": "completed",
                "artifacts": artifacts,
            },
            deep=True,
        )

    def _coerce_json_value(self, value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        return value

    def _sanitize_error_message(self, message: str) -> str:
        collapsed = " ".join(message.strip().split())
        if not collapsed:
            return "modal_remote_error"
        redacted = _TOKEN_RE.sub("[REDACTED]", collapsed)
        if len(redacted) > 400:
            return redacted[:400] + "...(truncated)"
        return redacted


def _is_canonical_cloud_state_path(path: Path) -> bool:
    parts = path.parts
    for index in range(len(parts) - 2):
        if parts[index] == ".numereng" and parts[index + 1] == "cloud":
            return True
    return False
