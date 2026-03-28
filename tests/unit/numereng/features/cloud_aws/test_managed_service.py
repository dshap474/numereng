from __future__ import annotations

import io
import json
import sqlite3
import tarfile
from pathlib import Path

import pytest

from numereng.features.cloud.aws.adapters import (
    BatchAdapter,
    BatchJobSpec,
    BatchJobStatus,
    CloudLogLine,
    CloudWatchLogsAdapter,
    DockerAdapter,
    EcrAdapter,
    S3Adapter,
    SageMakerAdapter,
    SageMakerTrainingSpec,
    SageMakerTrainingStatus,
)
from numereng.features.cloud.aws.managed_contracts import (
    AwsImageBuildPushRequest,
    AwsTrainCancelRequest,
    AwsTrainExtractRequest,
    AwsTrainLogsRequest,
    AwsTrainPullRequest,
    AwsTrainStatusRequest,
    AwsTrainSubmitRequest,
)
from numereng.features.cloud.aws.managed_service import CloudAwsError, CloudAwsManagedService


class _FakeEcr(EcrAdapter):
    def __init__(self) -> None:
        self.repository: str | None = None
        self.image_tag: str | None = None

    def ensure_repository(self, repository_name: str) -> str:
        self.repository = repository_name
        return f"123456789012.dkr.ecr.us-east-2.amazonaws.com/{repository_name}"

    def get_account_id(self) -> str:
        return "123456789012"

    def get_login_password(self) -> str:
        return "pw"

    def image_uri(self, repository_name: str, image_tag: str) -> str:
        return f"123456789012.dkr.ecr.us-east-2.amazonaws.com/{repository_name}:{image_tag}"

    def get_image_digest(self, repository_name: str, image_tag: str) -> str | None:
        self.repository = repository_name
        self.image_tag = image_tag
        return "sha256:abc"


class _FakeS3(S3Adapter):
    def __init__(self) -> None:
        self.uploaded: list[tuple[Path, str, str]] = []
        self.downloaded: list[tuple[str, str, Path]] = []
        self.keys: list[str] = []

    def ensure_bucket_exists(self, bucket: str, region: str) -> None:
        _ = (bucket, region)

    def upload_file(self, local_path: Path, bucket: str, key: str) -> str:
        self.uploaded.append((local_path, bucket, key))
        return f"s3://{bucket}/{key}"

    def download_file(self, bucket: str, key: str, local_path: Path) -> Path:
        self.downloaded.append((bucket, key, local_path))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text("payload", encoding="utf-8")
        return local_path

    def list_keys(self, bucket: str, prefix: str) -> list[str]:
        _ = (bucket, prefix)
        return list(self.keys)

    def delete_key(self, bucket: str, key: str) -> None:
        _ = (bucket, key)

    def delete_prefix(self, bucket: str, prefix: str) -> int:
        _ = (bucket, prefix)
        return 0

    def copy_object(self, src_bucket: str, src_key: str, dst_bucket: str, dst_key: str) -> str:
        _ = (src_bucket, src_key, dst_bucket, dst_key)
        return "s3://dst/key"


class _FakeSageMaker(SageMakerAdapter):
    def __init__(self) -> None:
        self.last_spec: SageMakerTrainingSpec | None = None

    def start_training(self, spec: SageMakerTrainingSpec) -> SageMakerTrainingStatus:
        self.last_spec = spec
        return SageMakerTrainingStatus(
            job_name=spec.job_name,
            job_arn=f"arn:aws:sagemaker:us-east-2:123:training-job/{spec.job_name}",
            status="InProgress",
            secondary_status="Starting",
            output_s3_uri=spec.output_s3_uri,
            log_group="/aws/sagemaker/TrainingJobs",
            log_stream_prefix=f"{spec.job_name}/",
        )

    def describe_training(self, job_name: str) -> SageMakerTrainingStatus:
        return SageMakerTrainingStatus(
            job_name=job_name,
            job_arn=f"arn:aws:sagemaker:us-east-2:123:training-job/{job_name}",
            status="Completed",
            secondary_status="Completed",
            output_s3_uri="s3://numereng-artifacts/runs/run-1/managed-output/",
            log_group="/aws/sagemaker/TrainingJobs",
            log_stream_prefix=f"{job_name}/",
        )

    def stop_training(self, job_name: str) -> None:
        _ = job_name


class _FakeBatch(BatchAdapter):
    def __init__(self) -> None:
        self.status = "RUNNING"
        self.cancelled: list[tuple[str, str]] = []
        self.terminated: list[tuple[str, str]] = []

    def submit_job(self, spec: BatchJobSpec) -> str:
        _ = spec
        return "batch-job-1"

    def describe_job(self, job_id: str) -> BatchJobStatus:
        return BatchJobStatus(job_id=job_id, status=self.status, status_reason=None, log_stream_name="stream-a")

    def cancel_job(self, job_id: str, *, reason: str) -> None:
        self.cancelled.append((job_id, reason))

    def terminate_job(self, job_id: str, *, reason: str) -> None:
        self.terminated.append((job_id, reason))


class _FakeLogs(CloudWatchLogsAdapter):
    def list_stream_names(self, *, log_group: str, stream_prefix: str, limit: int) -> list[str]:
        _ = (log_group, stream_prefix, limit)
        return ["stream-a"]

    def fetch_log_events(
        self,
        *,
        log_group: str,
        stream_name: str,
        limit: int,
        next_token: str | None = None,
        start_from_head: bool = False,
    ) -> tuple[list[CloudLogLine], str | None]:
        _ = (log_group, stream_name, limit, next_token, start_from_head)
        return [CloudLogLine(timestamp_ms=1, message="hello")], None


class _FakeDocker(DockerAdapter):
    def __init__(self) -> None:
        self.built: list[str] = []
        self.dockerfiles: list[Path | None] = []
        self.tagged: list[tuple[str, str]] = []
        self.pushed: list[str] = []

    def build_image(
        self,
        *,
        context_dir: Path,
        tag: str,
        dockerfile: Path | None = None,
        build_args: dict[str, str] | None = None,
        platform: str | None = None,
    ) -> None:
        _ = (context_dir, dockerfile, build_args, platform)
        self.built.append(tag)
        self.dockerfiles.append(dockerfile)

    def tag_image(self, *, source_tag: str, target_tag: str) -> None:
        self.tagged.append((source_tag, target_tag))

    def push_image(self, *, tag: str) -> None:
        self.pushed.append(tag)

    def login(self, *, registry: str, username: str, password: str) -> None:
        _ = (registry, username, password)


def _build_service(
    *,
    ecr: _FakeEcr | None = None,
    s3: _FakeS3 | None = None,
    sagemaker: _FakeSageMaker | None = None,
    batch: _FakeBatch | None = None,
    logs: _FakeLogs | None = None,
    docker: _FakeDocker | None = None,
) -> tuple[CloudAwsManagedService, _FakeEcr, _FakeS3, _FakeSageMaker, _FakeBatch, _FakeLogs, _FakeDocker]:
    fake_ecr = ecr or _FakeEcr()
    fake_s3 = s3 or _FakeS3()
    fake_sm = sagemaker or _FakeSageMaker()
    fake_batch = batch or _FakeBatch()
    fake_logs = logs or _FakeLogs()
    fake_docker = docker or _FakeDocker()
    service = CloudAwsManagedService(
        ecr_factory=lambda _region: fake_ecr,
        s3_factory=lambda _region: fake_s3,
        sagemaker_factory=lambda _region: fake_sm,
        batch_factory=lambda _region: fake_batch,
        logs_factory=lambda _region: fake_logs,
        docker_adapter=fake_docker,
    )
    return service, fake_ecr, fake_s3, fake_sm, fake_batch, fake_logs, fake_docker


def _state_path(tmp_path: Path, name: str = "state.json") -> Path:
    return tmp_path / ".numereng" / "cloud" / name


def test_image_build_push_happy_path(tmp_path: Path) -> None:
    service, _ecr, _s3, _sm, _batch, _logs, docker = _build_service()
    docker_dir = tmp_path / "docker"
    docker_dir.mkdir()
    (docker_dir / "Dockerfile.sagemaker").write_text("FROM scratch\n", encoding="utf-8")

    response = service.image_build_push(
        AwsImageBuildPushRequest(
            run_id="run-1",
            context_dir=str(tmp_path),
            repository="numereng-training",
            image_tag="v1",
            store_root=str(tmp_path / ".numereng"),
        )
    )

    assert response.action == "cloud.aws.image.build-push"
    assert response.state is not None
    assert response.state.image_uri is not None
    assert docker.built
    assert docker.pushed


def test_image_build_push_cuda_profile_uses_cuda_dockerfile(tmp_path: Path) -> None:
    service, _ecr, _s3, _sm, _batch, _logs, docker = _build_service()
    docker_dir = tmp_path / "docker"
    docker_dir.mkdir()
    (docker_dir / "Dockerfile.sagemaker").write_text("FROM scratch\n", encoding="utf-8")
    (docker_dir / "Dockerfile.sagemaker-lgbm-cuda").write_text("FROM scratch\n", encoding="utf-8")

    response = service.image_build_push(
        AwsImageBuildPushRequest(
            run_id="run-cuda",
            context_dir=str(tmp_path),
            repository="numereng-training",
            image_tag="cuda",
            runtime_profile="lgbm-cuda",
            store_root=str(tmp_path / ".numereng"),
        )
    )

    assert response.state is not None
    assert response.state.runtime_profile == "lgbm-cuda"
    assert docker.dockerfiles[-1] == tmp_path / "docker" / "Dockerfile.sagemaker-lgbm-cuda"


def test_train_submit_and_status_sagemaker(tmp_path: Path) -> None:
    service, _ecr, _s3, sagemaker, _batch, _logs, _docker = _build_service()
    config_path = tmp_path / "train.json"
    config_path.write_text(
        (
            '{"data": {"data_version": "v5.2", "dataset_variant": "non_downsampled"}, '
            '"model": {"type": "LGBMRegressor", "params": {}}, "training": {}}'
        ),
        encoding="utf-8",
    )

    submit = service.train_submit(
        AwsTrainSubmitRequest(
            run_id="run-1",
            backend="sagemaker",
            config_path=str(config_path),
            image_uri="123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training:v1",
            role_arn="arn:aws:iam::123456789012:role/numereng-sagemaker",
            state_path=str(_state_path(tmp_path)),
            store_root=str(tmp_path / ".numereng"),
        )
    )
    assert submit.action == "cloud.aws.train.submit"
    assert submit.state is not None
    assert submit.state.training_job_name is not None
    assert sagemaker.last_spec is not None

    status = service.train_status(
        AwsTrainStatusRequest(
            state_path=str(_state_path(tmp_path)),
            store_root=str(tmp_path / ".numereng"),
        )
    )
    assert status.action == "cloud.aws.train.status"
    assert status.result["status"] == "Completed"


def test_train_status_preserves_cloud_job_metadata_without_explicit_submit_context(tmp_path: Path) -> None:
    service, _ecr, _s3, _sagemaker, _batch, _logs, _docker = _build_service()
    state_path = _state_path(tmp_path)
    config_dir = tmp_path / ".numereng" / "experiments" / "exp-live" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "train.json"
    config_path.write_text(
        (
            '{"data": {"data_version": "v5.2", "dataset_variant": "non_downsampled"}, '
            '"model": {"type": "LGBMRegressor", "params": {}}, "training": {}}'
        ),
        encoding="utf-8",
    )

    submit = service.train_submit(
        AwsTrainSubmitRequest(
            run_id="run-1",
            backend="sagemaker",
            config_path=str(config_path),
            image_uri="123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training:v1",
            role_arn="arn:aws:iam::123456789012:role/numereng-sagemaker",
            state_path=str(state_path),
            store_root=str(tmp_path / ".numereng"),
        )
    )
    assert submit.state is not None

    status = service.train_status(
        AwsTrainStatusRequest(
            state_path=str(state_path),
            store_root=str(tmp_path / ".numereng"),
        )
    )

    with sqlite3.connect(tmp_path / ".numereng" / "numereng.db") as conn:
        row = conn.execute(
            """
            SELECT metadata_json
            FROM cloud_jobs
            WHERE run_id = ? AND provider = ? AND provider_job_id = ?
            """,
            ("run-1", "sagemaker", status.result["training_job_name"]),
        ).fetchone()

    assert row is not None
    metadata = json.loads(str(row[0]))
    assert metadata["experiment_id"] == "exp-live"
    assert metadata["config_path"] == str(config_path)
    assert metadata["config_id"] == str(config_path)
    assert metadata["config_label"] == "train.json"
    assert metadata["state_path"] == str(state_path.resolve())
    assert metadata["secondary_status"] == "Completed"
    assert metadata["last_progress_percent"] == "100.0"


def test_train_submit_cuda_requires_cuda_runtime_profile(tmp_path: Path) -> None:
    service, _ecr, _s3, _sagemaker, _batch, _logs, _docker = _build_service()
    config_path = tmp_path / "train.json"
    config_path.write_text(
        (
            '{"data": {"data_version": "v5.2", "dataset_variant": "non_downsampled"}, '
            '"model": {"type": "LGBMRegressor", "device": "cuda", "params": {}}, "training": {}}'
        ),
        encoding="utf-8",
    )

    with pytest.raises(CloudAwsError, match="aws_cuda_requires_runtime_profile_lgbm_cuda"):
        service.train_submit(
            AwsTrainSubmitRequest(
                run_id="run-cuda",
                backend="sagemaker",
                config_path=str(config_path),
                image_uri="123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training:v1",
                role_arn="arn:aws:iam::123456789012:role/numereng-sagemaker",
                instance_type="ml.g5.2xlarge",
                store_root=str(tmp_path / ".numereng"),
            )
        )


def test_train_submit_cuda_rejects_cpu_instance_type(tmp_path: Path) -> None:
    service, _ecr, _s3, _sagemaker, _batch, _logs, _docker = _build_service()
    config_path = tmp_path / "train.json"
    config_path.write_text(
        (
            '{"data": {"data_version": "v5.2", "dataset_variant": "non_downsampled"}, '
            '"model": {"type": "LGBMRegressor", "device": "cuda", "params": {}}, "training": {}}'
        ),
        encoding="utf-8",
    )

    with pytest.raises(CloudAwsError, match="aws_cuda_requires_gpu_instance_type:ml.m5.2xlarge"):
        service.train_submit(
            AwsTrainSubmitRequest(
                run_id="run-cuda",
                backend="sagemaker",
                config_path=str(config_path),
                image_uri="123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training:v1",
                runtime_profile="lgbm-cuda",
                role_arn="arn:aws:iam::123456789012:role/numereng-sagemaker",
                instance_type="ml.m5.2xlarge",
                store_root=str(tmp_path / ".numereng"),
            )
        )


def test_train_submit_rejects_cuda_runtime_profile_for_cpu_config(tmp_path: Path) -> None:
    service, _ecr, _s3, _sagemaker, _batch, _logs, _docker = _build_service()
    config_path = tmp_path / "train.json"
    config_path.write_text(
        (
            '{"data": {"data_version": "v5.2", "dataset_variant": "non_downsampled"}, '
            '"model": {"type": "LGBMRegressor", "params": {}}, "training": {}}'
        ),
        encoding="utf-8",
    )

    with pytest.raises(CloudAwsError, match="aws_runtime_profile_requires_cuda_config"):
        service.train_submit(
            AwsTrainSubmitRequest(
                run_id="run-cpu",
                backend="sagemaker",
                config_path=str(config_path),
                image_uri="123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training:v1",
                runtime_profile="lgbm-cuda",
                role_arn="arn:aws:iam::123456789012:role/numereng-sagemaker",
                instance_type="ml.g5.2xlarge",
                store_root=str(tmp_path / ".numereng"),
            )
        )


def test_train_logs_and_pull(tmp_path: Path) -> None:
    service, _ecr, s3, _sm, _batch, _logs, _docker = _build_service()
    state_path = _state_path(tmp_path)

    submit = service.train_submit(
        AwsTrainSubmitRequest(
            run_id="run-1",
            backend="sagemaker",
            config_s3_uri="s3://numereng-artifacts/runs/run-1/config/train.json",
            image_uri="123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training:v1",
            role_arn="arn:aws:iam::123456789012:role/numereng-sagemaker",
            state_path=str(state_path),
            store_root=str(tmp_path / ".numereng"),
        )
    )
    assert submit.state is not None

    logs = service.train_logs(
        AwsTrainLogsRequest(
            state_path=str(state_path),
            store_root=str(tmp_path / ".numereng"),
        )
    )
    assert logs.action == "cloud.aws.train.logs"
    assert logs.result["lines"] == ["hello"]

    s3.keys = [
        "runs/run-1/managed-output/run.json",
        "runs/run-1/managed-output/results.json",
        "runs/run-1/managed-output/metrics.json",
    ]

    pulled = service.train_pull(
        AwsTrainPullRequest(
            run_id="run-1",
            output_s3_uri="s3://numereng-artifacts/runs/run-1/managed-output/",
            output_dir=str(tmp_path / "out"),
            state_path=str(state_path),
            store_root=str(tmp_path / ".numereng"),
        )
    )
    assert pulled.action == "cloud.aws.train.pull"
    assert pulled.result["downloaded_count"] == 3


def test_train_pull_skips_unsafe_keys(tmp_path: Path) -> None:
    service, _ecr, s3, _sm, _batch, _logs, _docker = _build_service()
    s3.keys = [
        "runs/run-1/managed-output/results.json",
        "runs/run-1/managed-output/../../escape.txt",
        "runs/run-1/managed-output/nested/../metrics.json",
    ]

    pulled = service.train_pull(
        AwsTrainPullRequest(
            run_id="run-1",
            output_s3_uri="s3://numereng-artifacts/runs/run-1/managed-output/",
            output_dir=str(tmp_path / "out"),
            store_root=str(tmp_path / ".numereng"),
        )
    )

    assert pulled.result["downloaded_count"] == 1
    assert pulled.result["skipped_unsafe_keys"] == [
        "runs/run-1/managed-output/../../escape.txt",
        "runs/run-1/managed-output/nested/../metrics.json",
    ]


def test_train_pull_rejects_noncanonical_store_output_dir(tmp_path: Path) -> None:
    service, _ecr, _s3, _sm, _batch, _logs, _docker = _build_service()
    store_root = tmp_path / ".numereng"

    with pytest.raises(CloudAwsError, match="aws_train_pull_output_dir_noncanonical"):
        service.train_pull(
            AwsTrainPullRequest(
                run_id="run-1",
                output_s3_uri="s3://numereng-artifacts/runs/run-1/managed-output/",
                output_dir=str(store_root / "smoke_live_check"),
                store_root=str(store_root),
            )
        )


def test_train_pull_defaults_output_dir_to_cloud_pull(
    tmp_path: Path,
) -> None:
    service, _ecr, s3, _sm, _batch, _logs, _docker = _build_service()
    s3.keys = [
        "runs/run-1/managed-output/job-1/output/output.tar.gz",
    ]

    store_root = tmp_path / ".numereng"
    pulled = service.train_pull(
        AwsTrainPullRequest(
            run_id="run-1",
            output_s3_uri="s3://numereng-artifacts/runs/run-1/managed-output/",
            store_root=str(store_root),
        )
    )

    expected_output_dir = (store_root / "cloud" / "run-1" / "pull").resolve()
    expected_download_path = (expected_output_dir / "job-1" / "output" / "output.tar.gz").resolve()

    assert pulled.result["output_dir"] == str(expected_output_dir)
    assert pulled.result["downloaded_files"] == [str(expected_download_path)]
    assert expected_download_path.is_file()
    assert not (store_root / "runs" / "run-1" / "job-1" / "output" / "output.tar.gz").exists()


def test_train_extract_extracts_tarball_and_indexes_inner_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _ecr, s3, _sm, _batch, _logs, _docker = _build_service()
    s3.keys = [
        "runs/run-1/managed-output/job-1/output/output.tar.gz",
    ]

    def _fake_download_file(bucket: str, key: str, local_path: Path) -> Path:
        _ = (bucket, key)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "1",
            "run_id": "inner-run-1",
            "run_hash": "hash",
            "run_type": "training",
            "status": "FINISHED",
            "created_at": "2026-01-01T00:00:00Z",
            "config": {"path": "cfg", "hash": "cfg-hash"},
            "data": {"version": "v5.2", "feature_set": "small", "target_col": "target"},
            "model": {"type": "LGBMRegressor"},
            "training": {"engine": "custom"},
            "artifacts": {"predictions": "artifacts/predictions/preds.parquet"},
        }
        with tarfile.open(local_path, mode="w:gz") as archive:
            members: list[tuple[str, bytes]] = [
                ("runs/inner-run-1/run.json", json.dumps(manifest).encode("utf-8")),
                ("runs/inner-run-1/resolved.json", b"{}"),
                ("runs/inner-run-1/results.json", b"{}"),
                ("runs/inner-run-1/metrics.json", b"{}"),
                ("runs/inner-run-1/artifacts/predictions/preds.parquet", b"parquet"),
            ]
            for name, payload in members:
                info = tarfile.TarInfo(name=name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        return local_path

    monkeypatch.setattr(s3, "download_file", _fake_download_file)

    indexed_run_ids: list[str] = []

    def _fake_index_run(*, store_root: str, run_id: str) -> None:
        _ = store_root
        indexed_run_ids.append(run_id)

    monkeypatch.setattr("numereng.features.cloud.aws.managed_service.index_run", _fake_index_run)

    store_root = tmp_path / ".numereng"
    pulled = service.train_pull(
        AwsTrainPullRequest(
            run_id="run-1",
            output_s3_uri="s3://numereng-artifacts/runs/run-1/managed-output/",
            output_dir=str(tmp_path / "pull"),
            store_root=str(store_root),
        )
    )
    assert pulled.result["downloaded_count"] == 1

    extracted = service.train_extract(
        AwsTrainExtractRequest(
            run_id="run-1",
            output_dir=str(tmp_path / "pull"),
            store_root=str(store_root),
        )
    )

    assert extracted.result["extracted_run_ids"] == ["inner-run-1"]
    assert extracted.result["archive_extract_warnings"] == []
    assert extracted.result["indexed_run_ids"] == ["inner-run-1"]
    assert indexed_run_ids == ["inner-run-1"]
    assert extracted.result["indexed"] is True
    assert extracted.result["index_warning"] is None

    assert (store_root / "runs" / "inner-run-1" / "run.json").is_file()
    assert (store_root / "runs" / "inner-run-1" / "results.json").is_file()
    assert (store_root / "runs" / "inner-run-1" / "metrics.json").is_file()
    assert (store_root / "runs" / "inner-run-1" / "artifacts" / "predictions" / "preds.parquet").is_file()


def test_train_extract_rejects_noncanonical_store_output_dir(tmp_path: Path) -> None:
    service, _ecr, _s3, _sm, _batch, _logs, _docker = _build_service()
    store_root = tmp_path / ".numereng"

    with pytest.raises(CloudAwsError, match="aws_train_pull_output_dir_noncanonical"):
        service.train_extract(
            AwsTrainExtractRequest(
                run_id="run-1",
                output_dir=str(store_root / "smoke_live_check"),
                store_root=str(store_root),
            )
        )


def test_train_cancel_batch_running_uses_terminate(tmp_path: Path) -> None:
    fake_batch = _FakeBatch()
    fake_batch.status = "RUNNING"
    service, _ecr, _s3, _sm, _batch, _logs, _docker = _build_service(batch=fake_batch)

    response = service.train_cancel(
        AwsTrainCancelRequest(
            run_id="run-1",
            backend="batch",
            batch_job_id="batch-job-1",
            state_path=str(_state_path(tmp_path)),
            store_root=str(tmp_path / ".numereng"),
        )
    )

    assert response.result["requested_action"] == "terminate"
    assert fake_batch.terminated == [("batch-job-1", "numereng_user_cancel")]
    assert fake_batch.cancelled == []


def test_train_cancel_batch_pending_uses_cancel(tmp_path: Path) -> None:
    fake_batch = _FakeBatch()
    fake_batch.status = "SUBMITTED"
    service, _ecr, _s3, _sm, _batch, _logs, _docker = _build_service(batch=fake_batch)

    response = service.train_cancel(
        AwsTrainCancelRequest(
            run_id="run-1",
            backend="batch",
            batch_job_id="batch-job-1",
            state_path=str(_state_path(tmp_path)),
            store_root=str(tmp_path / ".numereng"),
        )
    )

    assert response.result["requested_action"] == "cancel"
    assert fake_batch.cancelled == [("batch-job-1", "numereng_user_cancel")]
    assert fake_batch.terminated == []


def test_invalid_state_file_raises_cloud_error(tmp_path: Path) -> None:
    service, _ecr, _s3, _sm, _batch, _logs, _docker = _build_service()
    state_path = _state_path(tmp_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("{bad json", encoding="utf-8")

    with pytest.raises(CloudAwsError, match="invalid state file"):
        service.train_status(
            AwsTrainStatusRequest(
                run_id="run-1",
                state_path=str(state_path),
                store_root=str(tmp_path / ".numereng"),
            )
        )


def test_state_path_must_be_canonical_cloud_json(tmp_path: Path) -> None:
    service, _ecr, _s3, _sm, _batch, _logs, _docker = _build_service()
    store_root = tmp_path / ".numereng"

    with pytest.raises(CloudAwsError, match="aws_state_path_noncanonical"):
        service.train_status(
            AwsTrainStatusRequest(
                run_id="run-1",
                backend="sagemaker",
                state_path=str(tmp_path / "state.json"),
                store_root=str(store_root),
            )
        )

    with pytest.raises(CloudAwsError, match="aws_state_path_extension_invalid"):
        service.train_status(
            AwsTrainStatusRequest(
                run_id="run-1",
                backend="sagemaker",
                state_path=str(_state_path(tmp_path, "state.txt")),
                store_root=str(store_root),
            )
        )


def test_train_extract_skips_non_runs_members_and_preserves_store_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _ecr, s3, _sm, _batch, _logs, _docker = _build_service()
    s3.keys = ["runs/run-1/managed-output/job-1/output/output.tar.gz"]

    def _fake_download_file(bucket: str, key: str, local_path: Path) -> Path:
        _ = (bucket, key)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "1",
            "run_id": "inner-run-1",
            "run_hash": "hash-inner",
            "run_type": "training",
            "status": "FINISHED",
            "created_at": "2026-01-01T00:00:00Z",
            "config": {"path": "cfg", "hash": "cfg-hash"},
            "data": {"version": "v5.2", "feature_set": "small", "target_col": "target"},
            "model": {"type": "LGBMRegressor"},
            "training": {"engine": "custom"},
            "artifacts": {"predictions": "artifacts/predictions/preds.parquet"},
        }
        with tarfile.open(local_path, mode="w:gz") as archive:
            members: list[tuple[str, bytes]] = [
                ("numereng.db", b"do-not-touch"),
                ("runs/inner-run-1/run.json", json.dumps(manifest).encode("utf-8")),
                ("runs/inner-run-1/results.json", b"{}"),
            ]
            for name, payload in members:
                info = tarfile.TarInfo(name=name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        return local_path

    monkeypatch.setattr(s3, "download_file", _fake_download_file)
    monkeypatch.setattr("numereng.features.cloud.aws.managed_service.index_run", lambda *, store_root, run_id: None)

    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True, exist_ok=True)
    store_db = store_root / "numereng.db"
    store_db.write_text("keep-me", encoding="utf-8")

    pulled = service.train_pull(
        AwsTrainPullRequest(
            run_id="run-1",
            output_s3_uri="s3://numereng-artifacts/runs/run-1/managed-output/",
            output_dir=str(tmp_path / "pull"),
            store_root=str(store_root),
        )
    )
    assert pulled.result["downloaded_count"] == 1

    extracted = service.train_extract(
        AwsTrainExtractRequest(
            run_id="run-1",
            output_dir=str(tmp_path / "pull"),
            store_root=str(store_root),
        )
    )

    assert extracted.result["extracted_run_ids"] == ["inner-run-1"]
    assert extracted.result["archive_extract_warnings"] == ["tar_member_outside_runs_skipped:numereng.db"]
    assert store_db.read_text(encoding="utf-8") == "keep-me"


def test_train_extract_rejects_unsafe_tar_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _ecr, s3, _sm, _batch, _logs, _docker = _build_service()
    s3.keys = ["runs/run-1/managed-output/job-1/output/output.tar.gz"]

    def _fake_download_file(bucket: str, key: str, local_path: Path) -> Path:
        _ = (bucket, key)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(local_path, mode="w:gz") as archive:
            payload = b"bad"
            info = tarfile.TarInfo(name="runs/inner-run-1/../../numereng.db")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        return local_path

    monkeypatch.setattr(s3, "download_file", _fake_download_file)

    service.train_pull(
        AwsTrainPullRequest(
            run_id="run-1",
            output_s3_uri="s3://numereng-artifacts/runs/run-1/managed-output/",
            output_dir=str(tmp_path / "pull"),
            store_root=str(tmp_path / ".numereng"),
        )
    )

    with pytest.raises(CloudAwsError, match="tar_member_unsafe"):
        service.train_extract(
            AwsTrainExtractRequest(
                run_id="run-1",
                output_dir=str(tmp_path / "pull"),
                store_root=str(tmp_path / ".numereng"),
            )
        )


def test_train_extract_rejects_run_dir_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _ecr, s3, _sm, _batch, _logs, _docker = _build_service()
    s3.keys = ["runs/run-1/managed-output/job-1/output/output.tar.gz"]

    def _fake_download_file(bucket: str, key: str, local_path: Path) -> Path:
        _ = (bucket, key)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {"run_id": "inner-run-1", "run_hash": "new-hash"}
        with tarfile.open(local_path, mode="w:gz") as archive:
            payload = json.dumps(manifest).encode("utf-8")
            info = tarfile.TarInfo(name="runs/inner-run-1/run.json")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        return local_path

    monkeypatch.setattr(s3, "download_file", _fake_download_file)

    store_root = tmp_path / ".numereng"
    existing_run_dir = store_root / "runs" / "inner-run-1"
    existing_run_dir.mkdir(parents=True, exist_ok=True)
    existing_run_dir.joinpath("run.json").write_text(
        json.dumps({"run_id": "inner-run-1", "run_hash": "old-hash"}),
        encoding="utf-8",
    )

    service.train_pull(
        AwsTrainPullRequest(
            run_id="run-1",
            output_s3_uri="s3://numereng-artifacts/runs/run-1/managed-output/",
            output_dir=str(tmp_path / "pull"),
            store_root=str(store_root),
        )
    )

    with pytest.raises(CloudAwsError, match="run_dir_conflict:inner-run-1"):
        service.train_extract(
            AwsTrainExtractRequest(
                run_id="run-1",
                output_dir=str(tmp_path / "pull"),
                store_root=str(store_root),
            )
        )
