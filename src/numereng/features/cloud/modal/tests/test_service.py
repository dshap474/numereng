from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from numereng.features.cloud.modal.adapters import (
    ModalAdapterError,
    ModalCallCancelledError,
    ModalCallNotFoundError,
    ModalCallPendingError,
    ModalRemoteExecutionError,
    ModalTrainingAdapter,
)
from numereng.features.cloud.modal.contracts import (
    CloudModalState,
    ModalDataSyncFile,
    ModalDataSyncRequest,
    ModalDataSyncResult,
    ModalDeployRequest,
    ModalDeployResult,
    ModalRuntimePayload,
    ModalRuntimeResult,
    ModalTrainCancelRequest,
    ModalTrainLogsRequest,
    ModalTrainPullRequest,
    ModalTrainStatusRequest,
    ModalTrainSubmitRequest,
)
from numereng.features.cloud.modal.data_sync import MODAL_DATASETS_MOUNT_PATH
from numereng.features.cloud.modal.service import CloudModalError, CloudModalService
from numereng.features.cloud.modal.state_store import CloudModalStateStore


@dataclass(slots=True)
class _FakeCallHandle:
    object_id: str
    phase: str = "running"
    result: ModalRuntimeResult | None = None
    failed_message: str = "remote training failed"
    cancelled: bool = False

    def get(self, *, timeout_seconds: float | None = None) -> ModalRuntimeResult:
        _ = timeout_seconds
        if self.phase == "running":
            raise ModalCallPendingError("modal_call_pending")
        if self.phase == "cancelled":
            raise ModalCallCancelledError("modal_call_cancelled")
        if self.phase == "expired":
            raise ModalCallNotFoundError("modal_call_not_found")
        if self.phase == "failed":
            raise ModalRemoteExecutionError(self.failed_message)
        if self.result is None:
            raise ModalAdapterError("modal_result_missing")
        return self.result

    def cancel(self) -> None:
        self.cancelled = True
        self.phase = "cancelled"


class _FakeModalTrainingAdapter(ModalTrainingAdapter):
    def __init__(self) -> None:
        self.calls: dict[str, _FakeCallHandle] = {}
        self.submit_payloads: list[tuple[str, str, ModalRuntimePayload, str | None]] = []
        self.sync_payloads: list[dict[str, object]] = []
        self.deploy_payloads: list[dict[str, object]] = []
        self._seq = 0

    def submit_training(
        self,
        *,
        app_name: str,
        function_name: str,
        payload: ModalRuntimePayload,
        environment_name: str | None = None,
    ) -> _FakeCallHandle:
        self._seq += 1
        call_id = f"fc-{self._seq}"
        handle = _FakeCallHandle(object_id=call_id)
        self.calls[call_id] = handle
        self.submit_payloads.append((app_name, function_name, payload, environment_name))
        return handle

    def lookup_call(self, call_id: str) -> _FakeCallHandle:
        try:
            return self.calls[call_id]
        except KeyError as exc:
            raise ModalCallNotFoundError("modal_call_not_found") from exc

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
        self.deploy_payloads.append(
            {
                "app_name": app_name,
                "function_name": function_name,
                "ecr_image_uri": ecr_image_uri,
                "data_volume_name": data_volume_name,
                "environment_name": environment_name,
                "aws_profile": aws_profile,
                "timeout_seconds": timeout_seconds,
                "gpu": gpu,
                "cpu": cpu,
                "memory_mb": memory_mb,
            }
        )
        return ModalDeployResult(
            deployed=True,
            deployment_id="ap-1",
            app_page_url="https://modal.test/app/ap-1",
            app_logs_url="https://modal.test/logs/ap-1",
            warnings=["warning-one"],
        )

    def sync_data(
        self,
        *,
        volume_name: str,
        files: Sequence[ModalDataSyncFile],
        create_if_missing: bool = True,
        force: bool = False,
    ) -> ModalDataSyncResult:
        self.sync_payloads.append(
            {
                "volume_name": volume_name,
                "files": files,
                "create_if_missing": create_if_missing,
                "force": force,
            }
        )
        uploaded_files = []
        total_bytes = 0
        for item in files:
            remote_path = getattr(item, "remote_path", None)
            size_bytes = getattr(item, "size_bytes", 0)
            if isinstance(remote_path, str):
                uploaded_files.append(remote_path)
            if isinstance(size_bytes, int):
                total_bytes += size_bytes
        return ModalDataSyncResult(
            volume_name=volume_name,
            uploaded_files=uploaded_files,
            file_count=len(uploaded_files),
            total_bytes=total_bytes,
            create_if_missing=create_if_missing,
            force=force,
        )


class _FailingStateStore(CloudModalStateStore):
    def __init__(self, *, fail_on_load: bool = False, fail_on_save: bool = False) -> None:
        self._fail_on_load = fail_on_load
        self._fail_on_save = fail_on_save

    def load(self, path: Path) -> CloudModalState | None:
        if self._fail_on_load:
            raise OSError("forced load failure")
        return super().load(path)

    def save(self, path: Path, state: CloudModalState) -> None:
        if self._fail_on_save:
            raise OSError("forced save failure")
        super().save(path, state)


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {"data_version": "v5.2", "dataset_variant": "non_downsampled"},
                "model": {"type": "LGBMRegressor", "params": {}},
                "training": {},
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _write_data_sync_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "train-sync.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {
                    "data_version": "v5.2",
                    "dataset_variant": "non_downsampled",
                    "benchmark_source": {
                        "source": "path",
                        "predictions_path": "v5.2/benchmark.parquet",
                    },
                    "meta_model_data_path": "v5.2/meta_model.parquet",
                },
                "model": {"type": "LGBMRegressor", "params": {}},
                "training": {},
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _state_path(tmp_path: Path, name: str = "state.json") -> Path:
    return tmp_path / ".numereng" / "cloud" / name


def test_train_submit_persists_state(tmp_path: Path) -> None:
    adapter = _FakeModalTrainingAdapter()
    service = CloudModalService(adapter=adapter)
    config_path = _write_config(tmp_path)
    state_path = _state_path(tmp_path)

    response = service.train_submit(
        ModalTrainSubmitRequest(
            config_path=str(config_path),
            app_name="neng-app",
            function_name="train_remote",
            metadata={"owner": "test"},
            state_path=str(state_path),
        )
    )

    assert response.action == "cloud.modal.train.submit"
    assert response.state is not None
    assert response.state.call_id == "fc-1"
    assert response.state.status == "submitted"
    assert response.state.metadata["owner"] == "test"
    assert adapter.submit_payloads
    assert adapter.submit_payloads[0][2].config_filename == "train.json"
    assert state_path.exists()


def test_deploy_persists_state(tmp_path: Path) -> None:
    adapter = _FakeModalTrainingAdapter()
    service = CloudModalService(adapter=adapter)
    state_path = _state_path(tmp_path)

    response = service.deploy(
        ModalDeployRequest(
            app_name="numereng-train",
            function_name="train_remote",
            ecr_image_uri="699475917808.dkr.ecr.us-east-2.amazonaws.com/numereng-training:latest",
            environment_name="main",
            aws_profile="default",
            timeout_seconds=900,
            gpu="T4",
            cpu=2.0,
            memory_mb=8192,
            data_volume_name="numereng-v52",
            metadata={"owner": "daniel"},
            state_path=str(state_path),
        )
    )

    assert response.action == "cloud.modal.deploy"
    assert response.result["deployed"] is True
    assert response.result["deployment_id"] == "ap-1"
    assert response.state is not None
    assert response.state.status == "deployed"
    assert response.state.run_id is None
    assert response.state.call_id is None
    assert response.state.deployment_id == "ap-1"
    assert response.state.artifacts == {}
    assert response.state.ecr_image_uri == "699475917808.dkr.ecr.us-east-2.amazonaws.com/numereng-training:latest"
    assert response.state.data_volume_name == "numereng-v52"
    assert response.state.metadata["owner"] == "daniel"
    assert adapter.deploy_payloads
    assert adapter.deploy_payloads[0]["aws_profile"] == "default"
    assert adapter.deploy_payloads[0]["data_volume_name"] == "numereng-v52"
    assert response.result["data_mount_path"] == MODAL_DATASETS_MOUNT_PATH
    assert state_path.exists()


def test_deploy_uses_synced_volume_from_state(tmp_path: Path) -> None:
    adapter = _FakeModalTrainingAdapter()
    service = CloudModalService(adapter=adapter)
    state_path = _state_path(tmp_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        CloudModalState(data_volume_name="persisted-volume").model_dump_json(),
        encoding="utf-8",
    )

    response = service.deploy(
        ModalDeployRequest(
            ecr_image_uri="699475917808.dkr.ecr.us-east-2.amazonaws.com/numereng-training:latest",
            state_path=str(state_path),
        )
    )

    assert response.state is not None
    assert response.state.data_volume_name == "persisted-volume"
    assert adapter.deploy_payloads[0]["data_volume_name"] == "persisted-volume"


def test_data_sync_uploads_config_required_files_and_persists_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    datasets_root = tmp_path / ".numereng" / "datasets" / "v5.2"
    datasets_root.mkdir(parents=True, exist_ok=True)
    (datasets_root / "train.parquet").write_bytes(b"train")
    (datasets_root / "validation.parquet").write_bytes(b"validation")
    (datasets_root / "benchmark.parquet").write_bytes(b"bench")
    (datasets_root / "meta_model.parquet").write_bytes(b"meta")
    (datasets_root / "features.json").write_text("{}", encoding="utf-8")
    config_path = _write_data_sync_config(tmp_path)

    adapter = _FakeModalTrainingAdapter()
    service = CloudModalService(adapter=adapter)
    state_path = _state_path(tmp_path)
    response = service.data_sync(
        ModalDataSyncRequest(
            config_path=str(config_path),
            volume_name="numereng-v52",
            metadata={"owner": "daniel"},
            state_path=str(state_path),
        )
    )

    assert response.action == "cloud.modal.data.sync"
    assert response.state is not None
    assert response.state.data_volume_name == "numereng-v52"
    assert response.state.metadata["owner"] == "daniel"
    assert response.state.metadata["data_sync_dataset_variant"] == "non_downsampled"
    assert response.state.data_manifest["v5.2/train.parquet"].endswith("/.numereng/datasets/v5.2/train.parquet")
    assert response.state.data_manifest["v5.2/validation.parquet"].endswith(
        "/.numereng/datasets/v5.2/validation.parquet"
    )
    assert response.result["mount_path"] == MODAL_DATASETS_MOUNT_PATH
    assert response.result["dataset_variant"] == "non_downsampled"
    assert response.result["file_count"] == 5
    assert adapter.sync_payloads
    assert adapter.sync_payloads[0]["volume_name"] == "numereng-v52"


def test_data_sync_requires_local_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = _write_data_sync_config(tmp_path)
    service = CloudModalService(adapter=_FakeModalTrainingAdapter())

    with pytest.raises(CloudModalError, match="modal_data_sync_file_not_found"):
        service.data_sync(
            ModalDataSyncRequest(
                config_path=str(config_path),
                volume_name="numereng-v52",
            )
        )


def test_train_submit_clears_stale_artifacts_and_run_id(tmp_path: Path) -> None:
    adapter = _FakeModalTrainingAdapter()
    service = CloudModalService(adapter=adapter)
    config_path = _write_config(tmp_path)
    state_path = _state_path(tmp_path)
    stale_state = CloudModalState(
        run_id="run-stale",
        call_id="fc-stale",
        status="completed",
        artifacts={"predictions_path": "/tmp/stale.parquet"},
        metadata={"error": "old error", "keep": "yes"},
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(stale_state.model_dump_json(), encoding="utf-8")

    response = service.train_submit(
        ModalTrainSubmitRequest(
            config_path=str(config_path),
            state_path=str(state_path),
        )
    )

    assert response.state is not None
    assert response.state.run_id is None
    assert response.state.artifacts == {}
    assert response.state.metadata.get("error") is None
    assert response.state.metadata["keep"] == "yes"


def test_train_status_completed_updates_run_artifacts(tmp_path: Path) -> None:
    adapter = _FakeModalTrainingAdapter()
    service = CloudModalService(adapter=adapter)
    config_path = _write_config(tmp_path)
    state_path = _state_path(tmp_path)

    submit = service.train_submit(ModalTrainSubmitRequest(config_path=str(config_path), state_path=str(state_path)))
    assert submit.state is not None
    call_id = submit.state.call_id
    assert call_id is not None

    adapter.calls[call_id].phase = "completed"
    adapter.calls[call_id].result = ModalRuntimeResult(
        run_id="run-100",
        predictions_path="/tmp/preds.parquet",
        results_path="/tmp/results.json",
        output_dir="/tmp",
    )

    status = service.train_status(ModalTrainStatusRequest(state_path=str(state_path)))

    assert status.result["status"] == "completed"
    assert status.result["run_id"] == "run-100"
    assert status.state is not None
    assert status.state.run_id == "run-100"
    assert status.state.artifacts["predictions_path"] == "/tmp/preds.parquet"


def test_train_pull_returns_running_for_pending_call(tmp_path: Path) -> None:
    adapter = _FakeModalTrainingAdapter()
    service = CloudModalService(adapter=adapter)
    config_path = _write_config(tmp_path)
    state_path = _state_path(tmp_path)

    service.train_submit(ModalTrainSubmitRequest(config_path=str(config_path), state_path=str(state_path)))
    pull = service.train_pull(ModalTrainPullRequest(state_path=str(state_path), timeout_seconds=0))

    assert pull.result["status"] == "running"
    assert "still running" in pull.message


def test_train_logs_contract_is_metadata_only(tmp_path: Path) -> None:
    adapter = _FakeModalTrainingAdapter()
    service = CloudModalService(adapter=adapter)
    config_path = _write_config(tmp_path)
    state_path = _state_path(tmp_path)

    service.train_submit(ModalTrainSubmitRequest(config_path=str(config_path), state_path=str(state_path)))
    logs = service.train_logs(ModalTrainLogsRequest(state_path=str(state_path), lines=333))

    assert logs.result["requested_lines"] == 333
    assert "log_source_hint" in logs.result
    assert "lines" not in logs.result


def test_train_pull_metadata_only_keeps_runtime_output_dir(tmp_path: Path) -> None:
    adapter = _FakeModalTrainingAdapter()
    service = CloudModalService(adapter=adapter)
    config_path = _write_config(tmp_path)
    state_path = _state_path(tmp_path)

    submit = service.train_submit(ModalTrainSubmitRequest(config_path=str(config_path), state_path=str(state_path)))
    assert submit.state is not None
    call_id = submit.state.call_id
    assert call_id is not None

    adapter.calls[call_id].phase = "completed"
    adapter.calls[call_id].result = ModalRuntimeResult(
        run_id="run-200",
        predictions_path="/tmp/preds-200.parquet",
        results_path="/tmp/results-200.json",
        output_dir="/remote/out",
    )

    pull = service.train_pull(
        ModalTrainPullRequest(
            state_path=str(state_path),
            output_dir="/local/override",
            timeout_seconds=1,
        )
    )

    assert pull.state is not None
    assert pull.state.artifacts["output_dir"] == "/remote/out"
    assert pull.result["requested_output_dir"] == "/local/override"
    assert "metadata described" in pull.message


def test_train_cancel_marks_call_cancelled(tmp_path: Path) -> None:
    adapter = _FakeModalTrainingAdapter()
    service = CloudModalService(adapter=adapter)
    config_path = _write_config(tmp_path)
    state_path = _state_path(tmp_path)

    submit = service.train_submit(ModalTrainSubmitRequest(config_path=str(config_path), state_path=str(state_path)))
    assert submit.state is not None
    call_id = submit.state.call_id
    assert call_id is not None

    cancel = service.train_cancel(ModalTrainCancelRequest(state_path=str(state_path)))

    assert cancel.result["status"] == "cancelled"
    assert adapter.calls[call_id].cancelled is True


def test_train_submit_raises_for_missing_config(tmp_path: Path) -> None:
    service = CloudModalService(adapter=_FakeModalTrainingAdapter())

    with pytest.raises(CloudModalError, match="config_path_not_found"):
        service.train_submit(ModalTrainSubmitRequest(config_path=str(tmp_path / "missing.json")))


def test_train_status_corrupt_state_file_translates_error(tmp_path: Path) -> None:
    service = CloudModalService(adapter=_FakeModalTrainingAdapter())
    state_path = _state_path(tmp_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(CloudModalError, match="modal_state_load_failed"):
        service.train_status(ModalTrainStatusRequest(state_path=str(state_path)))


def test_train_submit_persist_failure_surfaces_call_id(tmp_path: Path) -> None:
    service = CloudModalService(
        adapter=_FakeModalTrainingAdapter(),
        state_store=_FailingStateStore(fail_on_save=True),
    )
    config_path = _write_config(tmp_path)

    with pytest.raises(CloudModalError, match="modal_call_submitted_call_id:fc-1"):
        service.train_submit(
            ModalTrainSubmitRequest(
                config_path=str(config_path),
                state_path=str(_state_path(tmp_path)),
            )
        )


def test_train_status_redacts_remote_error_message(tmp_path: Path) -> None:
    adapter = _FakeModalTrainingAdapter()
    service = CloudModalService(adapter=adapter)
    config_path = _write_config(tmp_path)
    state_path = _state_path(tmp_path)

    submit = service.train_submit(ModalTrainSubmitRequest(config_path=str(config_path), state_path=str(state_path)))
    assert submit.state is not None
    call_id = submit.state.call_id
    assert call_id is not None

    adapter.calls[call_id].phase = "failed"
    adapter.calls[call_id].failed_message = "job failed token=SUPERSECRET_TOKEN_ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    status = service.train_status(ModalTrainStatusRequest(state_path=str(state_path)))

    assert status.result["status"] == "failed"
    assert status.state is not None
    assert "[REDACTED]" in str(status.result["error"])
    assert "[REDACTED]" in status.state.metadata["error"]


def test_deploy_translates_adapter_error(tmp_path: Path) -> None:
    class _FailingDeployAdapter(_FakeModalTrainingAdapter):
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
            _ = (
                app_name,
                function_name,
                ecr_image_uri,
                data_volume_name,
                environment_name,
                aws_profile,
                timeout_seconds,
                gpu,
                cpu,
                memory_mb,
            )
            raise ModalAdapterError("modal_deploy_failed")

    service = CloudModalService(adapter=_FailingDeployAdapter())
    with pytest.raises(CloudModalError, match="modal_deploy_failed"):
        service.deploy(
            ModalDeployRequest(
                ecr_image_uri="699475917808.dkr.ecr.us-east-2.amazonaws.com/numereng-training:latest"
            )
        )


def test_state_path_must_be_canonical_cloud_json(tmp_path: Path) -> None:
    service = CloudModalService(adapter=_FakeModalTrainingAdapter())
    invalid_path = tmp_path / "state.json"

    with pytest.raises(CloudModalError, match="modal_state_path_noncanonical"):
        service.train_status(ModalTrainStatusRequest(state_path=str(invalid_path)))

    with pytest.raises(CloudModalError, match="modal_state_path_extension_invalid"):
        service.train_status(ModalTrainStatusRequest(state_path=str(_state_path(tmp_path, "state.txt"))))
