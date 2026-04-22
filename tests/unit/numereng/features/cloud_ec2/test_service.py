from __future__ import annotations

from pathlib import Path

import pytest

from numereng.features.cloud.aws import service as cloud_ec2_service_module
from numereng.features.cloud.aws.adapters import (
    Ec2Adapter,
    IamAdapter,
    InstanceStatus,
    LaunchInstanceSpec,
    S3Adapter,
    SsmAdapter,
    SsmCommandResult,
    WheelBuilder,
)
from numereng.features.cloud.aws.contracts import (
    CloudEc2State,
    Ec2ConfigUploadRequest,
    Ec2InitIamRequest,
    Ec2InstallRequest,
    Ec2LogsRequest,
    Ec2PackageBuildUploadRequest,
    Ec2ProvisionRequest,
    Ec2PullRequest,
    Ec2PushRequest,
    Ec2S3CopyRequest,
    Ec2S3ListRequest,
    Ec2S3RemoveRequest,
    Ec2SetupDataRequest,
    Ec2StatusRequest,
    Ec2TerminateRequest,
    Ec2TrainPollRequest,
    Ec2TrainStartRequest,
)
from numereng.features.cloud.aws.service import CloudEc2Error, CloudEc2Service, _parse_s3_uri


@pytest.fixture(autouse=True)
def _configured_cloud_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cloud_ec2_service_module, "DEFAULT_BUCKET", "example-bucket")
    monkeypatch.setattr(cloud_ec2_service_module, "DEFAULT_IAM_ROLE", "numereng-training-role")
    monkeypatch.setattr(cloud_ec2_service_module, "DEFAULT_SECURITY_GROUP", "numereng-training")
    monkeypatch.setattr(cloud_ec2_service_module, "DEFAULT_CPU_AMI", "ami-cpu123")
    monkeypatch.setattr(cloud_ec2_service_module, "DEFAULT_GPU_AMI", "ami-gpu123")


class _FakeEc2(Ec2Adapter):
    def __init__(self) -> None:
        self.launched_spec: LaunchInstanceSpec | None = None
        self.terminated_ids: list[str] = []
        self.wait_result: bool = True
        self.spot_price: float | None = 0.11
        self.instance_status = InstanceStatus(
            instance_id="i-test123",
            state="running",
            instance_type="r7i.4xlarge",
            run_id="run-1",
        )
        self.training_instances: list[InstanceStatus] = [self.instance_status]

    def launch_instance(self, spec: LaunchInstanceSpec) -> str:
        self.launched_spec = spec
        return "i-test123"

    def wait_for_instance(self, instance_id: str, target_state: str, timeout_seconds: int) -> bool:
        _ = (instance_id, target_state, timeout_seconds)
        return self.wait_result

    def terminate_instance(self, instance_id: str) -> None:
        self.terminated_ids.append(instance_id)

    def get_instance_status(self, instance_id: str) -> InstanceStatus:
        _ = instance_id
        return self.instance_status

    def list_training_instances(self) -> list[InstanceStatus]:
        return self.training_instances

    def get_spot_price(self, instance_type: str) -> float | None:
        _ = instance_type
        return self.spot_price

    def resolve_security_group_id(self, security_group: str) -> str | None:
        _ = security_group
        return "sg-test123"


class _FakeS3(S3Adapter):
    def __init__(self) -> None:
        self.bucket_ensures: list[tuple[str, str]] = []
        self.uploaded: list[tuple[Path, str, str]] = []
        self.downloaded: list[tuple[str, str, Path]] = []
        self.keys: dict[tuple[str, str], list[str]] = {}
        self.deleted_key_calls: list[tuple[str, str]] = []
        self.deleted_prefix_calls: list[tuple[str, str]] = []
        self.copy_calls: list[tuple[str, str, str, str]] = []

    def ensure_bucket_exists(self, bucket: str, region: str) -> None:
        self.bucket_ensures.append((bucket, region))

    def upload_file(self, local_path: Path, bucket: str, key: str) -> str:
        self.uploaded.append((local_path, bucket, key))
        return f"s3://{bucket}/{key}"

    def download_file(self, bucket: str, key: str, local_path: Path) -> Path:
        self.downloaded.append((bucket, key, local_path))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text("payload", encoding="utf-8")
        return local_path

    def list_keys(self, bucket: str, prefix: str) -> list[str]:
        return self.keys.get((bucket, prefix), [])

    def delete_key(self, bucket: str, key: str) -> None:
        self.deleted_key_calls.append((bucket, key))

    def delete_prefix(self, bucket: str, prefix: str) -> int:
        self.deleted_prefix_calls.append((bucket, prefix))
        return 2

    def copy_object(self, src_bucket: str, src_key: str, dst_bucket: str, dst_key: str) -> str:
        self.copy_calls.append((src_bucket, src_key, dst_bucket, dst_key))
        return f"s3://{dst_bucket}/{dst_key}"


class _FakeSsm(SsmAdapter):
    def __init__(self) -> None:
        self.wait_calls: list[tuple[str, int]] = []
        self.command_calls: list[tuple[str, str, int]] = []
        self.responses: list[SsmCommandResult] = []
        self.fail_substrings: dict[str, Exception] = {}

    def wait_for_ssm(self, instance_id: str, timeout_seconds: int) -> None:
        self.wait_calls.append((instance_id, timeout_seconds))

    def run_command(self, instance_id: str, command: str, timeout_seconds: int) -> SsmCommandResult:
        self.command_calls.append((instance_id, command, timeout_seconds))
        for match, exc in self.fail_substrings.items():
            if match in command:
                raise exc
        if self.responses:
            return self.responses.pop(0)
        return SsmCommandResult(exit_code=0, stdout="", stderr="")


class _FakeIam(IamAdapter):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def ensure_training_role(self, role_name: str, bucket: str) -> str:
        self.calls.append(("role", role_name, bucket))
        return "arn:aws:iam::123:role/test"

    def ensure_instance_profile(self, role_name: str) -> str:
        self.calls.append(("profile", role_name, ""))
        return "arn:aws:iam::123:instance-profile/test"

    def ensure_security_group(self, region: str, group_name: str) -> str:
        self.calls.append(("sg", region, group_name))
        return "sg-test123"


class _FakeWheelBuilder(WheelBuilder):
    def build_assets(self, output_dir: Path) -> list[Path]:
        dist_dir = output_dir / "dist"
        dist_dir.mkdir(parents=True, exist_ok=True)
        wheel_path = dist_dir / "numereng-0.1.0-py3-none-any.whl"
        wheel_path.write_text("wheel", encoding="utf-8")
        req_path = output_dir / "requirements.txt"
        req_path.write_text("pydantic==2.8.0\n", encoding="utf-8")
        return [wheel_path, req_path]


def _build_service(
    fake_ec2: _FakeEc2 | None = None,
    fake_s3: _FakeS3 | None = None,
    fake_ssm: _FakeSsm | None = None,
    fake_iam: _FakeIam | None = None,
) -> tuple[CloudEc2Service, _FakeEc2, _FakeS3, _FakeSsm, _FakeIam]:
    ec2 = fake_ec2 or _FakeEc2()
    s3 = fake_s3 or _FakeS3()
    ssm = fake_ssm or _FakeSsm()
    iam = fake_iam or _FakeIam()
    service = CloudEc2Service(
        ec2_factory=lambda region: ec2,
        s3_factory=lambda region: s3,
        ssm_factory=lambda region: ssm,
        iam_adapter=iam,
        wheel_builder=_FakeWheelBuilder(),
    )
    return service, ec2, s3, ssm, iam


def _state_path(tmp_path: Path, name: str = "state.json") -> Path:
    return tmp_path / ".numereng" / "cloud" / name


def test_init_iam_returns_expected_ids() -> None:
    service, _ec2, _s3, _ssm, iam = _build_service()

    response = service.init_iam(Ec2InitIamRequest(region="us-east-2", bucket="bucket-a"))

    assert response.action == "cloud.ec2.init-iam"
    assert response.result["security_group_id"] == "sg-test123"
    assert len(iam.calls) == 3


def test_setup_data_syncs_required_and_optional_files(tmp_path: Path) -> None:
    service, _ec2, s3, _ssm, _iam = _build_service()

    version_dir = tmp_path / "datasets" / "v5.2"
    version_dir.mkdir(parents=True)
    for name in [
        "train.parquet",
        "validation.parquet",
        "features.json",
        "validation_example_preds.parquet",
        "downsampled_full.parquet",
        "downsampled_full_benchmark_models.parquet",
    ]:
        (version_dir / name).write_text("x", encoding="utf-8")

    response = service.setup_data(
        Ec2SetupDataRequest(cache_dir=str(tmp_path / "datasets"), data_version="v5.2", bucket="bucket-a")
    )

    assert response.action == "cloud.ec2.setup-data"
    assert response.result["missing_optional"] == []
    assert len(s3.uploaded) == 6


def test_setup_data_raises_when_required_file_missing(tmp_path: Path) -> None:
    service, _ec2, _s3, _ssm, _iam = _build_service()

    version_dir = tmp_path / "datasets" / "v5.2"
    version_dir.mkdir(parents=True)
    (version_dir / "train.parquet").write_text("x", encoding="utf-8")

    with pytest.raises(CloudEc2Error, match="missing required data file"):
        service.setup_data(Ec2SetupDataRequest(cache_dir=str(tmp_path / "datasets"), data_version="v5.2"))


def test_provision_uses_state_file_context_and_persists(tmp_path: Path) -> None:
    state_path = _state_path(tmp_path, "cloud-state.json")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = CloudEc2State(run_id="run-1", region="us-east-2", bucket="bucket-1")
    state_path.write_text(state.model_dump_json(), encoding="utf-8")

    service, ec2, _s3, ssm, _iam = _build_service()

    response = service.provision(Ec2ProvisionRequest(tier="large", state_path=str(state_path), use_spot=True))

    assert response.state is not None
    assert response.state.instance_id == "i-test123"
    assert response.state.status == "ready"
    assert ec2.launched_spec is not None
    assert ec2.launched_spec.run_id == "run-1"
    assert ssm.wait_calls[0] == ("i-test123", 420)


def test_provision_raises_without_run_id() -> None:
    service, _ec2, _s3, _ssm, _iam = _build_service()

    with pytest.raises(CloudEc2Error, match="missing required value: run_id"):
        service.provision(Ec2ProvisionRequest(tier="large"))


def test_provision_requires_configured_ec2_launch_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    service, _ec2, _s3, _ssm, _iam = _build_service()
    monkeypatch.setattr(cloud_ec2_service_module, "DEFAULT_IAM_ROLE", "")
    monkeypatch.setattr(cloud_ec2_service_module, "DEFAULT_SECURITY_GROUP", "")
    monkeypatch.setattr(cloud_ec2_service_module, "DEFAULT_CPU_AMI", "")
    monkeypatch.setattr(cloud_ec2_service_module, "DEFAULT_GPU_AMI", "")

    with pytest.raises(CloudEc2Error, match="NUMERENG_EC2_AMI_CPU"):
        service.provision(Ec2ProvisionRequest(tier="large", run_id="run-1", bucket="bucket-1"))


def test_package_build_upload_writes_artifact_uris() -> None:
    service, _ec2, s3, _ssm, _iam = _build_service()

    response = service.package_build_upload(
        Ec2PackageBuildUploadRequest(run_id="run-22", region="us-east-2", bucket="bucket-22")
    )

    assert response.state is not None
    assert response.state.status == "package_uploaded"
    assert "requirements.txt" in response.result["uploaded"]
    assert len(s3.uploaded) == 2


def test_config_upload_success_and_missing_file(tmp_path: Path) -> None:
    service, _ec2, _s3, _ssm, _iam = _build_service()
    state_path = _state_path(tmp_path)

    with pytest.raises(CloudEc2Error, match="config path does not exist"):
        service.config_upload(
            Ec2ConfigUploadRequest(
                run_id="run-1",
                config_path=str(tmp_path / "missing.json"),
                state_path=str(state_path),
            )
        )

    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"data": {"data_version": "v5.2"}, "model": {"type": "LGBMRegressor", "params": {}}, "training": {}}',
        encoding="utf-8",
    )

    response = service.config_upload(
        Ec2ConfigUploadRequest(run_id="run-1", config_path=str(config_path), state_path=str(state_path))
    )
    assert response.result["config_uri"] == "s3://example-bucket/runs/run-1/config.json"


def test_push_uses_state_file_for_instance_and_run(tmp_path: Path) -> None:
    state_path = _state_path(tmp_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = CloudEc2State(
        run_id="run-3",
        instance_id="i-3",
        region="us-east-2",
        bucket="bucket-3",
        data_version="v5.2",
    )
    state_path.write_text(state.model_dump_json(), encoding="utf-8")

    service, _ec2, _s3, ssm, _iam = _build_service()

    response = service.push(Ec2PushRequest(state_path=str(state_path)))

    assert response.state is not None
    assert response.state.status == "artifacts_pushed"
    assert len(ssm.command_calls) == 7


def test_install_runs_remote_commands() -> None:
    service, _ec2, _s3, ssm, _iam = _build_service()

    response = service.install(Ec2InstallRequest(run_id="run-4", instance_id="i-4", region="us-east-2"))

    assert response.state is not None
    assert response.state.status == "runtime_installed"
    assert len(ssm.command_calls) == 3


def test_install_cuda_profile_requires_gpu_instance(tmp_path: Path) -> None:
    state_path = _state_path(tmp_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        CloudEc2State(run_id="run-4", instance_id="i-4", region="us-east-2", is_gpu=False).model_dump_json(),
        encoding="utf-8",
    )
    service, _ec2, _s3, _ssm, _iam = _build_service()

    with pytest.raises(CloudEc2Error, match="cloud_ec2_cuda_install_requires_gpu_instance"):
        service.install(
            Ec2InstallRequest(
                run_id="run-4",
                instance_id="i-4",
                region="us-east-2",
                runtime_profile="lgbm-cuda",
                state_path=str(state_path),
            )
        )


def test_install_cuda_profile_runs_lightgbm_cuda_reinstall(tmp_path: Path) -> None:
    state_path = _state_path(tmp_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        CloudEc2State(
            run_id="run-4",
            instance_id="i-4",
            region="us-east-2",
            is_gpu=True,
            runtime_profile="lgbm-cuda",
        ).model_dump_json(),
        encoding="utf-8",
    )
    service, _ec2, _s3, ssm, _iam = _build_service()

    response = service.install(Ec2InstallRequest(run_id="run-4", instance_id="i-4", state_path=str(state_path)))

    assert response.state is not None
    assert response.state.runtime_profile == "lgbm-cuda"
    assert len(ssm.command_calls) == 4
    assert (
        "CMAKE_ARGS='-DUSE_CUDA=ON' uv pip install --force-reinstall --no-binary lightgbm lightgbm"
        in (ssm.command_calls[2][1])
    )


def test_train_start_success_and_invalid_pid() -> None:
    service, _ec2, _s3, ssm, _iam = _build_service()
    ssm.responses.append(SsmCommandResult(exit_code=0, stdout="4321\n", stderr=""))

    response = service.train_start(Ec2TrainStartRequest(run_id="run-5", instance_id="i-5"))
    assert response.result["training_pid"] == 4321

    ssm.responses.append(SsmCommandResult(exit_code=0, stdout="not-a-pid\n", stderr=""))
    with pytest.raises(CloudEc2Error, match="invalid PID"):
        service.train_start(Ec2TrainStartRequest(run_id="run-5", instance_id="i-5"))


def test_train_poll_completed_and_failed_on_terminated_instance(tmp_path: Path) -> None:
    service, ec2, _s3, ssm, _iam = _build_service()

    request = Ec2TrainPollRequest(run_id="run-6", instance_id="i-6", region="us-east-2", timeout_seconds=30)
    state_path = _state_path(tmp_path, "run6-state.json")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        CloudEc2State(run_id="run-6", instance_id="i-6", region="us-east-2", training_pid=1234).model_dump_json(),
        encoding="utf-8",
    )
    request = request.model_copy(update={"state_path": str(state_path)})
    ssm.responses.append(SsmCommandResult(exit_code=0, stdout="0\n", stderr=""))

    response = service.train_poll(request)
    assert response.result["status"] == "completed"

    failing_service, failing_ec2, _s3b, failing_ssm, _iamb = _build_service()
    failing_ssm.fail_substrings["cat /opt/numereng/training.exit_code"] = RuntimeError("ssm down")
    failing_ec2.instance_status = InstanceStatus(
        instance_id="i-7",
        state="terminated",
        instance_type="r7i.4xlarge",
        run_id="run-7",
    )

    poll_request = Ec2TrainPollRequest(run_id="run-7", instance_id="i-7", region="us-east-2", timeout_seconds=1)
    temp_state_path = _state_path(tmp_path, "run7-state.json")
    temp_state_path.parent.mkdir(parents=True, exist_ok=True)
    temp_state_path.write_text(
        CloudEc2State(run_id="run-7", instance_id="i-7", region="us-east-2", training_pid=555).model_dump_json(),
        encoding="utf-8",
    )
    poll_request = poll_request.model_copy(update={"state_path": str(temp_state_path)})

    failed = failing_service.train_poll(poll_request)
    assert failed.result["status"] == "failed"


def test_train_poll_requires_training_pid() -> None:
    service, _ec2, _s3, _ssm, _iam = _build_service()

    with pytest.raises(CloudEc2Error, match="training_pid"):
        service.train_poll(Ec2TrainPollRequest(run_id="run-x", instance_id="i-x", region="us-east-2"))


def test_logs_pull_terminate_and_status_flows(tmp_path: Path) -> None:
    service, ec2, s3, ssm, _iam = _build_service()

    ssm.responses.append(SsmCommandResult(exit_code=0, stdout="tail-output", stderr=""))
    logs = service.logs(Ec2LogsRequest(instance_id="i-8", follow=True))
    assert logs.result["follow"] is True
    assert "follow mode" in logs.message

    s3.keys[("bucket-8", "runs/run-8/")] = ["runs/run-8/a.json", "runs/run-8/b.json"]
    pull = service.pull(Ec2PullRequest(run_id="run-8", instance_id="i-8", bucket="bucket-8", output_dir=str(tmp_path)))
    assert pull.result["downloaded_count"] == 2

    terminate = service.terminate(Ec2TerminateRequest(instance_id="i-8", region="us-east-2"))
    assert terminate.state is not None
    assert terminate.state.status == "terminated"
    assert ec2.terminated_ids == ["i-8"]

    status = service.status(Ec2StatusRequest(run_id="run-1", region="us-east-2"))
    assert "instance" in status.result

    ec2.training_instances = []
    status_none = service.status(Ec2StatusRequest(run_id="run-none", region="us-east-2"))
    assert "no active instance" in status_none.message


def test_pull_rejects_noncanonical_output_dir() -> None:
    service, _ec2, _s3, _ssm, _iam = _build_service()

    with pytest.raises(CloudEc2Error, match="cloud_ec2_pull_output_dir_noncanonical"):
        service.pull(
            Ec2PullRequest(
                run_id="run-8",
                instance_id="i-8",
                bucket="bucket-8",
                output_dir=str((Path(".numereng") / "smoke_live_check").resolve()),
            )
        )


def test_pull_skips_unsafe_keys(tmp_path: Path) -> None:
    service, _ec2, s3, _ssm, _iam = _build_service()
    s3.keys[("bucket-8", "runs/run-8/")] = [
        "runs/run-8/results.json",
        "runs/run-8/../../escape.json",
        "runs/run-8/nested/../metrics.json",
    ]

    pull = service.pull(
        Ec2PullRequest(
            run_id="run-8",
            instance_id="i-8",
            bucket="bucket-8",
            output_dir=str(tmp_path / "downloads"),
        )
    )

    assert pull.result["downloaded_count"] == 1
    assert pull.result["skipped_unsafe_keys"] == [
        "runs/run-8/../../escape.json",
        "runs/run-8/nested/../metrics.json",
    ]


def test_s3_operations_and_edge_cases(tmp_path: Path) -> None:
    service, _ec2, s3, _ssm, _iam = _build_service()

    listed = service.s3_list(Ec2S3ListRequest(prefix="/runs/abc", bucket="bucket-z", region="us-east-2"))
    assert listed.result["prefix"] == "runs/abc"

    copied = service.s3_copy(
        Ec2S3CopyRequest(src="s3://bucket-a/runs/x.json", dst="s3://bucket-b/runs/y.json", region="us-east-2")
    )
    assert copied.result["destination"] == "s3://bucket-b/runs/y.json"

    s3.keys[("bucket-a", "runs/prefix/")] = ["runs/prefix/a.txt", "runs/prefix/b.txt"]
    dst_dir = tmp_path / "downloads"
    dl = service.s3_copy(Ec2S3CopyRequest(src="s3://bucket-a/runs/prefix/", dst=str(dst_dir), region="us-east-2"))
    assert dl.result["count"] == 2

    local_file = tmp_path / "metrics.json"
    local_file.write_text("{}", encoding="utf-8")
    ul = service.s3_copy(Ec2S3CopyRequest(src=str(local_file), dst="s3://bucket-u/runs/", region="us-east-2"))
    assert str(ul.result["destination"]).endswith("/metrics.json")

    with pytest.raises(CloudEc2Error, match="source path does not exist"):
        service.s3_copy(Ec2S3CopyRequest(src=str(tmp_path / "missing.txt"), dst="s3://bucket/miss"))

    with pytest.raises(CloudEc2Error, match="either src or dst must be an s3:// URI"):
        service.s3_copy(Ec2S3CopyRequest(src=str(local_file), dst=str(tmp_path / "out.txt")))

    rm_prefix = service.s3_remove(Ec2S3RemoveRequest(uri="s3://bucket-r/runs/run-1/", recursive=True))
    assert rm_prefix.result["deleted_count"] == 2

    rm_file = service.s3_remove(Ec2S3RemoveRequest(uri="s3://bucket-r/runs/run-2/a.json", recursive=False))
    assert rm_file.result["deleted_count"] == 1

    with pytest.raises(CloudEc2Error, match="must be an s3:// URI"):
        service.s3_remove(Ec2S3RemoveRequest(uri="runs/local/path"))


def test_s3_copy_prefix_download_skips_unsafe_keys(tmp_path: Path) -> None:
    service, _ec2, s3, _ssm, _iam = _build_service()
    s3.keys[("bucket-a", "runs/prefix/")] = [
        "runs/prefix/one.json",
        "runs/prefix/../../escape.txt",
        "runs/prefix/deep/../two.json",
    ]

    response = service.s3_copy(
        Ec2S3CopyRequest(src="s3://bucket-a/runs/prefix/", dst=str(tmp_path / "downloads"), region="us-east-2")
    )

    assert response.result["count"] == 1
    assert response.result["skipped_unsafe_keys"] == [
        "runs/prefix/../../escape.txt",
        "runs/prefix/deep/../two.json",
    ]


def test_state_path_must_be_canonical_cloud_json(tmp_path: Path) -> None:
    service, _ec2, _s3, _ssm, _iam = _build_service()
    invalid_path = tmp_path / "state.json"
    invalid_path.write_text("{}", encoding="utf-8")

    with pytest.raises(CloudEc2Error, match="cloud_ec2_state_path_noncanonical"):
        service.status(Ec2StatusRequest(run_id="run-1", state_path=str(invalid_path)))

    with pytest.raises(CloudEc2Error, match="cloud_ec2_state_path_extension_invalid"):
        service.status(Ec2StatusRequest(run_id="run-1", state_path=str(_state_path(tmp_path, "state.txt"))))


def test_parse_s3_uri_cases() -> None:
    assert _parse_s3_uri("s3://bucket-a/key.json", default_bucket="bucket-default") == ("bucket-a", "key.json")
    assert _parse_s3_uri("s3://bucket-a", default_bucket="bucket-default") == ("bucket-a", "")
    assert _parse_s3_uri("local-path", default_bucket="bucket-default") is None

    with pytest.raises(CloudEc2Error, match="missing bucket"):
        _parse_s3_uri("s3://", default_bucket="bucket-default")
