"""Modal SDK-backed adapter implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from numereng.features.cloud.modal.adapters import (
    ModalAdapterError,
    ModalCallCancelledError,
    ModalCallHandle,
    ModalCallNotFoundError,
    ModalCallPendingError,
    ModalRemoteExecutionError,
    ModalTrainingAdapter,
)
from numereng.features.cloud.modal.contracts import (
    ModalDataSyncFile,
    ModalDataSyncResult,
    ModalDeployResult,
    ModalRuntimePayload,
    ModalRuntimeResult,
    parse_ecr_image_uri,
)
from numereng.features.cloud.modal.data_sync import MODAL_DATASETS_MOUNT_PATH


def _load_modal_modules() -> tuple[Any, Any, Any]:
    try:
        modal = import_module("modal")
        modal_exception = import_module("modal.exception")
        modal_runner = import_module("modal.runner")
    except Exception as exc:  # pragma: no cover - exercised only when modal is not installed
        raise ModalAdapterError(
            "modal_sdk_missing: install modal and authenticate with `modal setup`"
        ) from exc
    return modal, modal_exception, modal_runner


def _resolve_aws_credentials(*, aws_profile: str | None, region: str) -> dict[str, str]:
    try:
        import boto3
    except Exception as exc:  # pragma: no cover - boto3 is a required runtime dependency
        raise ModalAdapterError("aws_sdk_missing: install boto3") from exc

    try:
        if aws_profile is None:
            session = boto3.Session()
        else:
            session = boto3.Session(profile_name=aws_profile)
    except Exception as exc:
        raise ModalAdapterError(f"aws_profile_resolution_failed:{exc}") from exc

    try:
        credentials = session.get_credentials()
    except Exception as exc:
        raise ModalAdapterError(f"aws_credentials_resolution_failed:{exc}") from exc
    if credentials is None:
        raise ModalAdapterError("aws_credentials_missing")

    frozen = credentials.get_frozen_credentials()
    if not frozen.access_key or not frozen.secret_key:
        raise ModalAdapterError("aws_credentials_incomplete")

    values = {
        "AWS_ACCESS_KEY_ID": frozen.access_key,
        "AWS_SECRET_ACCESS_KEY": frozen.secret_key,
        "AWS_REGION": session.region_name or region,
    }
    if frozen.token:
        values["AWS_SESSION_TOKEN"] = frozen.token
    return values


@dataclass(slots=True)
class _SdkModalCallHandle(ModalCallHandle):
    _call: Any
    _modal_exception: Any

    @property
    def object_id(self) -> str:
        object_id = getattr(self._call, "object_id", None)
        if isinstance(object_id, str) and object_id:
            return object_id
        raise ModalAdapterError("modal_call_id_missing")

    def get(self, *, timeout_seconds: float | None = None) -> ModalRuntimeResult:
        try:
            if timeout_seconds is None:
                raw = self._call.get()
            else:
                raw = self._call.get(timeout=timeout_seconds)
        except BaseException as exc:  # pragma: no branch - explicit normalization path
            if self._is_cancelled(exc):
                raise ModalCallCancelledError("modal_call_cancelled") from exc
            if isinstance(exc, Exception):
                if self._is_timeout(exc):
                    raise ModalCallPendingError("modal_call_pending") from exc
                if self._is_not_found(exc):
                    raise ModalCallNotFoundError("modal_call_not_found") from exc
                raise ModalRemoteExecutionError(str(exc) or "modal_call_failed") from exc
            raise
        return _coerce_runtime_result(raw)

    def cancel(self) -> None:
        try:
            self._call.cancel()
        except Exception as exc:
            raise ModalAdapterError(str(exc) or "modal_call_cancel_failed") from exc

    def _is_timeout(self, exc: BaseException) -> bool:
        timeout_cls = getattr(self._modal_exception, "TimeoutError", None)
        if timeout_cls is not None and isinstance(exc, timeout_cls):
            return True
        return exc.__class__.__name__.endswith("TimeoutError")

    def _is_not_found(self, exc: BaseException) -> bool:
        not_found_cls = getattr(self._modal_exception, "NotFoundError", None)
        if not_found_cls is not None and isinstance(exc, not_found_cls):
            return True
        return "NotFoundError" in exc.__class__.__name__

    def _is_cancelled(self, exc: BaseException) -> bool:
        cancelled_cls = getattr(self._modal_exception, "InputCancellation", None)
        if cancelled_cls is not None and isinstance(exc, cancelled_cls):
            return True
        return "Cancellation" in exc.__class__.__name__


def _coerce_runtime_result(raw: Any) -> ModalRuntimeResult:
    if isinstance(raw, ModalRuntimeResult):
        return raw
    if isinstance(raw, dict):
        return ModalRuntimeResult.model_validate(raw)

    model_dump = getattr(raw, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return ModalRuntimeResult.model_validate(dumped)

    raise ModalAdapterError(f"modal_call_result_invalid:{type(raw).__name__}")


class SdkModalTrainingAdapter(ModalTrainingAdapter):
    """Modal SDK-backed training adapter."""

    def __init__(self) -> None:
        self._modal, self._modal_exception, self._modal_runner = _load_modal_modules()

    def submit_training(
        self,
        *,
        app_name: str,
        function_name: str,
        payload: ModalRuntimePayload,
        environment_name: str | None = None,
    ) -> ModalCallHandle:
        try:
            function = self._modal.Function.from_name(
                app_name,
                function_name,
                environment_name=environment_name,
            )
            call = function.spawn(payload.model_dump(mode="python"))
        except Exception as exc:
            raise ModalAdapterError(str(exc) or "modal_submit_failed") from exc
        return _SdkModalCallHandle(_call=call, _modal_exception=self._modal_exception)

    def lookup_call(self, call_id: str) -> ModalCallHandle:
        if not call_id:
            raise ModalAdapterError("modal_call_id_missing")
        try:
            call = self._modal.FunctionCall.from_id(call_id)
        except Exception as exc:
            if "NotFoundError" in exc.__class__.__name__:
                raise ModalCallNotFoundError("modal_call_not_found") from exc
            raise ModalAdapterError(str(exc) or "modal_call_lookup_failed") from exc
        return _SdkModalCallHandle(_call=call, _modal_exception=self._modal_exception)

    def sync_data(
        self,
        *,
        volume_name: str,
        files: Sequence[ModalDataSyncFile],
        create_if_missing: bool = True,
        force: bool = False,
    ) -> ModalDataSyncResult:
        if not volume_name:
            raise ModalAdapterError("modal_volume_name_missing")
        file_list = list(files)
        if not file_list:
            raise ModalAdapterError("modal_data_sync_files_missing")
        try:
            volume = self._modal.Volume.from_name(
                volume_name,
                create_if_missing=create_if_missing,
            )
            with volume.batch_upload(force=force) as batch:
                for item in file_list:
                    batch.put_file(item.source_path, item.remote_path)
        except Exception as exc:
            raise ModalAdapterError(f"modal_data_sync_failed:{exc}") from exc
        return ModalDataSyncResult(
            volume_name=volume_name,
            uploaded_files=[item.remote_path for item in file_list],
            file_count=len(file_list),
            total_bytes=sum(item.size_bytes for item in file_list),
            create_if_missing=create_if_missing,
            force=force,
        )

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
        ecr_ref = parse_ecr_image_uri(ecr_image_uri)
        credentials = _resolve_aws_credentials(aws_profile=aws_profile, region=ecr_ref.region)

        try:
            aws_secret = self._modal.Secret.from_dict(credentials)
            # The ECR image can carry a default ENTRYPOINT/CMD used for other runtimes
            # (for example SageMaker). Modal functions must own process startup.
            image = self._modal.Image.from_aws_ecr(
                ecr_image_uri,
                secret=aws_secret,
                setup_dockerfile_commands=["ENTRYPOINT []", "CMD []"],
            )
            app = self._modal.App(app_name)
            data_volume = None
            if data_volume_name is not None:
                data_volume = self._modal.Volume.from_name(
                    data_volume_name,
                    create_if_missing=False,
                )
        except Exception as exc:
            raise ModalAdapterError(f"modal_deploy_setup_failed:{exc}") from exc

        decorator_kwargs: dict[str, object] = {
            "image": image,
            "name": function_name,
            # Deploy helper function is defined inside this method; serialized mode
            # allows non-global function registration for Modal app decoration.
            "serialized": True,
        }
        if timeout_seconds is not None:
            decorator_kwargs["timeout"] = timeout_seconds
        if gpu is not None:
            decorator_kwargs["gpu"] = gpu
        if cpu is not None:
            decorator_kwargs["cpu"] = cpu
        if memory_mb is not None:
            decorator_kwargs["memory"] = memory_mb
        if data_volume_name is not None and data_volume is not None:
            decorator_kwargs["volumes"] = {MODAL_DATASETS_MOUNT_PATH: data_volume}

        def _train_remote(payload: dict[str, object]) -> dict[str, str]:
            from numereng.features.cloud.modal.runtime import run_training_payload

            return run_training_payload(payload)

        decorator = app.function(**decorator_kwargs)
        _ = decorator(_train_remote)
        try:
            deployed = self._modal_runner.deploy_app(
                app,
                name=app_name,
                environment_name=environment_name,
            )
        except Exception as exc:
            raise ModalAdapterError(f"modal_deploy_failed:{exc}") from exc

        raw_warnings = getattr(deployed, "warnings", [])
        warnings = [str(item) for item in raw_warnings] if isinstance(raw_warnings, list) else []
        return ModalDeployResult(
            deployed=True,
            deployment_id=str(getattr(deployed, "app_id", "")) or None,
            app_page_url=str(getattr(deployed, "app_page_url", "")) or None,
            app_logs_url=str(getattr(deployed, "app_logs_url", "")) or None,
            warnings=warnings,
        )
