from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from numereng.features.cloud.modal.adapters import (
    ModalAdapterError,
    ModalCallCancelledError,
    ModalCallNotFoundError,
    ModalCallPendingError,
)
from numereng.features.cloud.modal.contracts import ModalDataSyncFile, ModalRuntimeResult
from numereng.features.cloud.modal.data_sync import MODAL_DATASETS_MOUNT_PATH
from numereng.features.cloud.modal.modal_adapters import (
    SdkModalTrainingAdapter,
    _coerce_runtime_result,
    _resolve_aws_credentials,
    _SdkModalCallHandle,
)


class _FakeTimeoutError(Exception):
    pass


class _FakeNotFoundError(Exception):
    pass


class _FakeInputCancellation(Exception):
    pass


class _FakeModalException:
    TimeoutError = _FakeTimeoutError
    NotFoundError = _FakeNotFoundError
    InputCancellation = _FakeInputCancellation


@dataclass(slots=True)
class _FakeCall:
    object_id: str = "fc-test"
    result: object | None = None
    error: BaseException | None = None
    cancelled: bool = False

    def get(self, timeout: float | None = None) -> object:
        _ = timeout
        if self.error is not None:
            raise self.error
        if self.result is None:
            raise RuntimeError("missing result")
        return self.result

    def cancel(self) -> None:
        self.cancelled = True


def test_handle_get_maps_timeout_to_pending() -> None:
    call = _FakeCall(error=_FakeTimeoutError("slow"))
    handle = _SdkModalCallHandle(_call=call, _modal_exception=_FakeModalException)

    with pytest.raises(ModalCallPendingError, match="modal_call_pending"):
        handle.get(timeout_seconds=0.1)


def test_handle_get_maps_not_found_to_call_not_found() -> None:
    call = _FakeCall(error=_FakeNotFoundError("gone"))
    handle = _SdkModalCallHandle(_call=call, _modal_exception=_FakeModalException)

    with pytest.raises(ModalCallNotFoundError, match="modal_call_not_found"):
        handle.get(timeout_seconds=0.1)


def test_handle_get_maps_input_cancellation_to_cancelled() -> None:
    call = _FakeCall(error=_FakeInputCancellation("cancelled"))
    handle = _SdkModalCallHandle(_call=call, _modal_exception=_FakeModalException)

    with pytest.raises(ModalCallCancelledError, match="modal_call_cancelled"):
        handle.get(timeout_seconds=0.1)


def test_handle_cancel_forwards_to_call() -> None:
    call = _FakeCall(result={"run_id": "r", "predictions_path": "p", "results_path": "rj", "output_dir": "o"})
    handle = _SdkModalCallHandle(_call=call, _modal_exception=_FakeModalException)

    handle.cancel()

    assert call.cancelled is True


def test_coerce_runtime_result_accepts_dict() -> None:
    result = _coerce_runtime_result(
        {"run_id": "run-1", "predictions_path": "/tmp/p", "results_path": "/tmp/r", "output_dir": "/tmp"}
    )

    assert isinstance(result, ModalRuntimeResult)
    assert result.run_id == "run-1"


def test_coerce_runtime_result_accepts_model_dump_source() -> None:
    class _Dumpable:
        def model_dump(self) -> dict[str, Any]:
            return {
                "run_id": "run-2",
                "predictions_path": "/tmp/p2",
                "results_path": "/tmp/r2",
                "output_dir": "/tmp2",
            }

    result = _coerce_runtime_result(_Dumpable())

    assert result.run_id == "run-2"


def test_coerce_runtime_result_rejects_unknown_shape() -> None:
    with pytest.raises(ModalAdapterError, match="modal_call_result_invalid"):
        _coerce_runtime_result(object())


def test_resolve_aws_credentials_supports_profile_and_session_token(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Frozen:
        access_key = "AKIAEXAMPLE"
        secret_key = "SECRET"
        token = "SESSION"

    class _Credentials:
        def get_frozen_credentials(self) -> _Frozen:
            return _Frozen()

    class _Session:
        def __init__(self, profile_name: str | None = None) -> None:
            self.profile_name = profile_name
            self.region_name = "us-east-2"

        def get_credentials(self) -> _Credentials:
            return _Credentials()

    class _Boto3:
        Session = _Session

    monkeypatch.setitem(sys.modules, "boto3", _Boto3())

    resolved = _resolve_aws_credentials(aws_profile="default", region="us-east-1")

    assert resolved["AWS_ACCESS_KEY_ID"] == "AKIAEXAMPLE"
    assert resolved["AWS_SECRET_ACCESS_KEY"] == "SECRET"
    assert resolved["AWS_SESSION_TOKEN"] == "SESSION"
    assert resolved["AWS_REGION"] == "us-east-2"


def test_resolve_aws_credentials_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Session:
        region_name = None

        def __init__(self, profile_name: str | None = None) -> None:
            _ = profile_name

        def get_credentials(self) -> None:
            return None

    class _Boto3:
        Session = _Session

    monkeypatch.setitem(sys.modules, "boto3", _Boto3())

    with pytest.raises(ModalAdapterError, match="aws_credentials_missing"):
        _resolve_aws_credentials(aws_profile=None, region="us-east-2")


def test_deploy_training_uses_ecr_image_and_returns_deploy_result(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeSecret:
        @staticmethod
        def from_dict(values: dict[str, str]) -> dict[str, str]:
            captured["secret_values"] = values
            return {"secret": "ok"}

    class _FakeImage:
        @staticmethod
        def from_aws_ecr(tag: str, secret: dict[str, str], **kwargs: object) -> dict[str, str]:
            captured["ecr_tag"] = tag
            captured["secret"] = secret
            captured["image_kwargs"] = kwargs
            return {"image": "ok"}

    class _FakeApp:
        def __init__(self, name: str) -> None:
            captured["app_name"] = name

        def function(self, **kwargs: object) -> Any:
            captured["function_kwargs"] = kwargs

            def _decorator(fn: Any) -> Any:
                captured["function_name"] = fn.__name__
                return fn

            return _decorator

    class _FakeVolume:
        @staticmethod
        def from_name(name: str, create_if_missing: bool = False) -> dict[str, object]:
            captured["volume"] = {"name": name, "create_if_missing": create_if_missing}
            return {"volume": name}

    class _FakeDeployResult:
        app_id = "ap-1"
        app_page_url = "https://modal.test/apps/ap-1"
        app_logs_url = "https://modal.test/logs/ap-1"
        warnings = ["w1"]

    class _FakeRunner:
        @staticmethod
        def deploy_app(app: object, name: str | None = None, environment_name: str | None = None) -> _FakeDeployResult:
            captured["deploy_app"] = {"app": app, "name": name, "environment_name": environment_name}
            return _FakeDeployResult()

    class _FakeModal:
        Secret = _FakeSecret
        Image = _FakeImage
        App = _FakeApp
        Volume = _FakeVolume

    monkeypatch.setattr(
        "numereng.features.cloud.modal.modal_adapters._load_modal_modules",
        lambda: (_FakeModal, _FakeModalException, _FakeRunner),
    )
    monkeypatch.setattr(
        "numereng.features.cloud.modal.modal_adapters._resolve_aws_credentials",
        lambda *, aws_profile, region: {
            "AWS_ACCESS_KEY_ID": "AKIA",
            "AWS_SECRET_ACCESS_KEY": "SECRET",
            "AWS_REGION": region,
        },
    )

    adapter = SdkModalTrainingAdapter()
    result = adapter.deploy_training(
        app_name="numereng-train",
        function_name="train_remote",
        ecr_image_uri="699475917808.dkr.ecr.us-east-2.amazonaws.com/numereng-training:latest",
        environment_name="main",
        aws_profile="default",
        timeout_seconds=600,
        gpu="T4",
        cpu=2.0,
        memory_mb=8192,
        data_volume_name="numereng-v52",
    )

    assert captured["ecr_tag"] == "699475917808.dkr.ecr.us-east-2.amazonaws.com/numereng-training:latest"
    assert captured["image_kwargs"]["setup_dockerfile_commands"] == ["ENTRYPOINT []", "CMD []"]
    assert captured["function_kwargs"]["name"] == "train_remote"
    assert captured["function_kwargs"]["timeout"] == 600
    assert captured["function_kwargs"]["gpu"] == "T4"
    assert captured["function_kwargs"]["cpu"] == 2.0
    assert captured["function_kwargs"]["memory"] == 8192
    assert captured["function_kwargs"]["volumes"] == {MODAL_DATASETS_MOUNT_PATH: {"volume": "numereng-v52"}}
    assert captured["volume"] == {"name": "numereng-v52", "create_if_missing": False}
    assert captured["deploy_app"]["name"] == "numereng-train"
    assert captured["deploy_app"]["environment_name"] == "main"
    assert result.deployed is True
    assert result.deployment_id == "ap-1"


def test_deploy_training_maps_deploy_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeSecret:
        @staticmethod
        def from_dict(values: dict[str, str]) -> dict[str, str]:
            _ = values
            return {"secret": "ok"}

    class _FakeImage:
        @staticmethod
        def from_aws_ecr(tag: str, secret: dict[str, str], **kwargs: object) -> dict[str, str]:
            _ = (tag, secret, kwargs)
            return {"image": "ok"}

    class _FakeApp:
        def __init__(self, name: str) -> None:
            _ = name

        def function(self, **kwargs: object) -> Any:
            _ = kwargs

            def _decorator(fn: Any) -> Any:
                return fn

            return _decorator

    class _FakeRunner:
        @staticmethod
        def deploy_app(app: object, name: str | None = None, environment_name: str | None = None) -> object:
            _ = (app, name, environment_name)
            raise RuntimeError("boom")

    class _FakeModal:
        Secret = _FakeSecret
        Image = _FakeImage
        App = _FakeApp

    monkeypatch.setattr(
        "numereng.features.cloud.modal.modal_adapters._load_modal_modules",
        lambda: (_FakeModal, _FakeModalException, _FakeRunner),
    )
    monkeypatch.setattr(
        "numereng.features.cloud.modal.modal_adapters._resolve_aws_credentials",
        lambda *, aws_profile, region: {
            "AWS_ACCESS_KEY_ID": "AKIA",
            "AWS_SECRET_ACCESS_KEY": "SECRET",
            "AWS_REGION": region,
        },
    )

    adapter = SdkModalTrainingAdapter()
    with pytest.raises(ModalAdapterError, match="modal_deploy_failed"):
        adapter.deploy_training(
            app_name="numereng-train",
            function_name="train_remote",
            ecr_image_uri="699475917808.dkr.ecr.us-east-2.amazonaws.com/numereng-training:latest",
            environment_name=None,
            aws_profile=None,
            timeout_seconds=None,
            gpu=None,
            cpu=None,
            memory_mb=None,
        )


def test_sync_data_uploads_manifest_to_volume(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    local_file = tmp_path / "datasets" / "v5.2" / "features.json"
    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.write_text("{}", encoding="utf-8")

    class _FakeBatchUpload:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def __enter__(self) -> _FakeBatchUpload:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            _ = (exc_type, exc, tb)
            return None

        def put_file(self, source_path: str, remote_path: str) -> None:
            self.calls.append((source_path, remote_path))

    class _FakeVolumeObject:
        def batch_upload(self, force: bool = False) -> _FakeBatchUpload:
            captured["force"] = force
            batch = _FakeBatchUpload()
            captured["batch"] = batch
            return batch

    class _FakeVolume:
        @staticmethod
        def from_name(name: str, create_if_missing: bool = False) -> _FakeVolumeObject:
            captured["volume"] = {"name": name, "create_if_missing": create_if_missing}
            return _FakeVolumeObject()

    class _FakeModal:
        Volume = _FakeVolume

    monkeypatch.setattr(
        "numereng.features.cloud.modal.modal_adapters._load_modal_modules",
        lambda: (_FakeModal, _FakeModalException, object()),
    )

    adapter = SdkModalTrainingAdapter()
    result = adapter.sync_data(
        volume_name="numereng-v52",
        files=[
            ModalDataSyncFile(
                source_path=str(local_file),
                remote_path="v5.2/features.json",
                size_bytes=local_file.stat().st_size,
            )
        ],
        create_if_missing=True,
        force=True,
    )

    batch = captured["batch"]
    assert isinstance(batch, _FakeBatchUpload)
    assert captured["volume"] == {"name": "numereng-v52", "create_if_missing": True}
    assert captured["force"] is True
    assert batch.calls == [(str(local_file), "v5.2/features.json")]
    assert result.volume_name == "numereng-v52"
    assert result.file_count == 1
    assert result.uploaded_files == ["v5.2/features.json"]


def test_sync_data_maps_upload_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    local_file = tmp_path / "datasets" / "v5.2" / "features.json"
    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.write_text("{}", encoding="utf-8")

    class _FakeBatchUpload:
        def __enter__(self) -> _FakeBatchUpload:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            _ = (exc_type, exc, tb)
            return None

        def put_file(self, source_path: str, remote_path: str) -> None:
            _ = (source_path, remote_path)
            raise RuntimeError("boom")

    class _FakeVolumeObject:
        def batch_upload(self, force: bool = False) -> _FakeBatchUpload:
            _ = force
            return _FakeBatchUpload()

    class _FakeVolume:
        @staticmethod
        def from_name(name: str, create_if_missing: bool = False) -> _FakeVolumeObject:
            _ = (name, create_if_missing)
            return _FakeVolumeObject()

    class _FakeModal:
        Volume = _FakeVolume

    monkeypatch.setattr(
        "numereng.features.cloud.modal.modal_adapters._load_modal_modules",
        lambda: (_FakeModal, _FakeModalException, object()),
    )

    adapter = SdkModalTrainingAdapter()
    with pytest.raises(ModalAdapterError, match="modal_data_sync_failed"):
        adapter.sync_data(
            volume_name="numereng-v52",
            files=[
                ModalDataSyncFile(
                    source_path=str(local_file),
                    remote_path="v5.2/features.json",
                    size_bytes=local_file.stat().st_size,
                )
            ],
        )
