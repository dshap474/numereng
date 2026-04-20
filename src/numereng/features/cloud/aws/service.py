"""Service layer for modular EC2 cloud lifecycle commands."""

from __future__ import annotations

import base64
import os
import shlex
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from numereng.features.cloud.aws.adapters import (
    Ec2Adapter,
    IamAdapter,
    InstanceStatus,
    LaunchInstanceSpec,
    S3Adapter,
    SsmAdapter,
    WheelBuilder,
)
from numereng.features.cloud.aws.aws_adapters import (
    AwsEc2Adapter,
    AwsIamAdapter,
    AwsS3Adapter,
    AwsSsmAdapter,
    UvWheelBuilder,
)
from numereng.features.cloud.aws.contracts import (
    CloudEc2RequestBase,
    CloudEc2Response,
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
from numereng.features.cloud.aws.managed_contracts import CloudRuntimeProfile
from numereng.features.cloud.aws.state_store import CloudEc2StateStore
from numereng.features.store import StoreError, index_run
from numereng.features.store.layout import (
    ensure_allowed_store_target,
    resolve_cloud_run_state_path,
    resolve_path,
    validate_cloud_state_path,
)
from numereng.platform.run_execution import (
    RUN_EXECUTION_ENV_B64_VAR,
    build_run_execution,
    serialize_run_execution,
    stamp_run_execution,
)


class CloudEc2Error(Exception):
    """Feature-level error for cloud EC2 orchestration."""


DEFAULT_REGION = (os.getenv("NUMERENG_AWS_REGION") or "us-east-2").strip()
DEFAULT_BUCKET = (os.getenv("NUMERENG_S3_BUCKET") or "").strip()
DEFAULT_IAM_ROLE = (os.getenv("NUMERENG_EC2_IAM_ROLE") or "").strip()
DEFAULT_SECURITY_GROUP = (os.getenv("NUMERENG_EC2_SECURITY_GROUP") or "").strip()
DEFAULT_DATA_VERSION = "v5.2"

DEFAULT_CPU_AMI = (os.getenv("NUMERENG_EC2_AMI_CPU") or "").strip()
DEFAULT_GPU_AMI = (os.getenv("NUMERENG_EC2_AMI_GPU") or "").strip()
DEFAULT_RUNTIME_PROFILE: CloudRuntimeProfile = "standard"
CUDA_RUNTIME_PROFILE: CloudRuntimeProfile = "lgbm-cuda"
_CLOUD_PROVIDER = "aws"

CPU_INSTANCE_TYPES: dict[str, str] = {
    "m7i.xlarge": "m7i.xlarge",
    "m7i.2xlarge": "m7i.2xlarge",
    "r7i.2xlarge": "r7i.2xlarge",
    "m7i.4xlarge": "m7i.4xlarge",
    "r7i.4xlarge": "r7i.4xlarge",
    "r7i.8xlarge": "r7i.8xlarge",
    "small": "m7i.xlarge",
    "medium": "r7i.2xlarge",
    "medium-fast": "m7i.4xlarge",
    "large": "r7i.4xlarge",
    "2xlarge": "r7i.8xlarge",
}

GPU_INSTANCE_TYPES: dict[str, str] = {
    "g4dn.xlarge": "g4dn.xlarge",
    "g6.xlarge": "g6.xlarge",
    "g6.2xlarge": "g6.2xlarge",
    "g5.xlarge": "g5.xlarge",
    "g5.2xlarge": "g5.2xlarge",
    "gpu-budget": "g4dn.xlarge",
    "gpu": "g6.2xlarge",
    "gpu-large": "g5.2xlarge",
}


def resolve_instance_type(tier: str) -> str:
    """Resolve alias to concrete instance type."""
    resolved = CPU_INSTANCE_TYPES.get(tier) or GPU_INSTANCE_TYPES.get(tier)
    if resolved is not None:
        return resolved
    choices = sorted(set(CPU_INSTANCE_TYPES.keys()) | set(GPU_INSTANCE_TYPES.keys()))
    raise CloudEc2Error(f"unknown tier '{tier}'. available: {', '.join(choices)}")


def is_gpu_instance(instance_type: str) -> bool:
    """Infer whether an instance type is GPU-capable by family prefix."""
    prefix = instance_type.split(".")[0] if "." in instance_type else instance_type
    return prefix.startswith(("g", "p", "inf", "trn"))


def _bootstrap_user_data(*, data_version: str, gpu: bool) -> str:
    if gpu:
        return f"""#!/bin/bash
set -euo pipefail
exec &>> /var/log/numereng-bootstrap.log
snap install amazon-ssm-agent 2>/dev/null || apt-get install -y amazon-ssm-agent 2>/dev/null || true
systemctl start amazon-ssm-agent 2>/dev/null || snap start amazon-ssm-agent 2>/dev/null || true
apt-get update -y -q
DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
python3.12 python3.12-venv python3-pip curl build-essential cmake ninja-build git
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=\"/root/.local/bin:$PATH\"
mkdir -p /opt/numereng
python3.12 -m venv /opt/numereng/.venv
mkdir -p /opt/numereng/.numereng/datasets/{data_version}
mkdir -p /opt/numereng/.numereng/configs
mkdir -p /opt/numereng/.numereng/runs
touch /opt/numereng/.ready
"""

    return f"""#!/bin/bash
set -euo pipefail
exec &>> /var/log/numereng-bootstrap.log
systemctl start amazon-ssm-agent 2>/dev/null || true
dnf install -y -q python3.12 python3.12-pip python3.12-devel gcc-c++ git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=\"/root/.local/bin:$PATH\"
mkdir -p /opt/numereng
python3.12 -m venv /opt/numereng/.venv
mkdir -p /opt/numereng/.numereng/datasets/{data_version}
mkdir -p /opt/numereng/.numereng/configs
mkdir -p /opt/numereng/.numereng/runs
touch /opt/numereng/.ready
"""


class CloudEc2Service:
    """EC2 cloud lifecycle service with pluggable adapters."""

    def __init__(
        self,
        *,
        ec2_factory: Callable[[str], Ec2Adapter] | None = None,
        s3_factory: Callable[[str], S3Adapter] | None = None,
        ssm_factory: Callable[[str], SsmAdapter] | None = None,
        iam_adapter: IamAdapter | None = None,
        wheel_builder: WheelBuilder | None = None,
        state_store: CloudEc2StateStore | None = None,
    ) -> None:
        self._ec2_factory = ec2_factory or (lambda region: AwsEc2Adapter(region=region))
        self._s3_factory = s3_factory or (lambda region: AwsS3Adapter(region=region))
        self._ssm_factory = ssm_factory or (lambda region: AwsSsmAdapter(region=region))
        self._iam = iam_adapter or AwsIamAdapter()
        self._wheel_builder = wheel_builder or UvWheelBuilder()
        self._state_store = state_store or CloudEc2StateStore()

    def _resolved_state_path(
        self,
        request: CloudEc2RequestBase,
        *,
        state: CloudEc2State | None = None,
    ) -> Path | None:
        state_path = request.state_file()
        if state_path is not None:
            try:
                return validate_cloud_state_path(
                    target_path=state_path,
                    store_root=request.store_root,
                    error_code="cloud_ec2_state_path_noncanonical",
                    allow_legacy_cloud=True,
                )
            except ValueError as exc:
                resolved = resolve_path(state_path)
                if resolved.suffix.lower() != ".json":
                    raise CloudEc2Error(f"cloud_ec2_state_path_extension_invalid:{resolved}") from exc
                raise CloudEc2Error(str(exc)) from exc
        run_id = getattr(request, "run_id", None) or (state.run_id if state is not None else None)
        if run_id is None or not run_id.strip():
            return None
        return resolve_cloud_run_state_path(store_root=request.store_root, provider=_CLOUD_PROVIDER, run_id=run_id)

    def _load_state(self, request: CloudEc2RequestBase) -> CloudEc2State:
        state_path = self._resolved_state_path(request)
        if state_path is None:
            return CloudEc2State()
        loaded = self._state_store.load(state_path)
        if loaded is None:
            return CloudEc2State()
        return loaded

    def _persist_state(self, request: CloudEc2RequestBase, state: CloudEc2State) -> CloudEc2State:
        touched = state.touched()
        state_path = self._resolved_state_path(request, state=touched)
        if state_path is not None:
            self._state_store.save(state_path, touched)
        return touched

    def _cloud_execution(
        self,
        *,
        request: CloudEc2RequestBase,
        state: CloudEc2State,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, object]:
        state_path = self._resolved_state_path(request, state=state)
        merged_metadata = dict(metadata or {})
        return build_run_execution(
            kind="cloud",
            provider=_CLOUD_PROVIDER,
            backend="ec2",
            target_id=state.run_id,
            instance_id=state.instance_id,
            region=state.region,
            state_path=str(state_path) if state_path is not None else None,
            submitted_at=merged_metadata.get("submitted_at"),
            pulled_at=merged_metadata.get("pulled_at"),
            metadata={
                key: value
                for key, value in merged_metadata.items()
                if key not in {"submitted_at", "pulled_at"} and value
            }
            or None,
        )

    def _resolve_region(self, explicit: str | None, state: CloudEc2State | None = None) -> str:
        if explicit is not None and explicit:
            return explicit
        if state is not None and state.region is not None and state.region:
            return state.region
        return DEFAULT_REGION

    def _resolve_bucket(self, explicit: str | None, state: CloudEc2State | None = None) -> str:
        if explicit is not None and explicit:
            return explicit
        if state is not None and state.bucket is not None and state.bucket:
            return state.bucket
        return DEFAULT_BUCKET

    def _resolve_data_version(self, explicit: str | None, state: CloudEc2State | None = None) -> str:
        if explicit is not None and explicit:
            return explicit
        if state is not None and state.data_version is not None and state.data_version:
            return state.data_version
        return DEFAULT_DATA_VERSION

    def _resolve_runtime_profile(
        self,
        explicit: CloudRuntimeProfile | None,
        state: CloudEc2State | None = None,
    ) -> CloudRuntimeProfile:
        if explicit is not None:
            return explicit
        if state is not None:
            return state.runtime_profile
        return DEFAULT_RUNTIME_PROFILE

    def _require(self, value: str | None, field_name: str) -> str:
        if value is None or not value:
            raise CloudEc2Error(f"missing required value: {field_name}")
        return value

    def _ec2(self, region: str) -> Ec2Adapter:
        return self._ec2_factory(region)

    def _s3(self, region: str) -> S3Adapter:
        return self._s3_factory(region)

    def _ssm(self, region: str) -> SsmAdapter:
        return self._ssm_factory(region)

    def init_iam(self, request: Ec2InitIamRequest) -> CloudEc2Response:
        region = request.region or DEFAULT_REGION
        bucket = self._require(request.bucket or DEFAULT_BUCKET, "bucket or NUMERENG_S3_BUCKET")
        role_name = self._require(request.role_name or DEFAULT_IAM_ROLE, "role_name or NUMERENG_EC2_IAM_ROLE")
        security_group_name = self._require(
            request.security_group_name or DEFAULT_SECURITY_GROUP,
            "security_group_name or NUMERENG_EC2_SECURITY_GROUP",
        )

        role_arn = self._iam.ensure_training_role(role_name=role_name, bucket=bucket)
        profile_arn = self._iam.ensure_instance_profile(role_name=role_name)
        security_group_id = self._iam.ensure_security_group(region=region, group_name=security_group_name)

        return CloudEc2Response(
            action="cloud.ec2.init-iam",
            message="IAM and network bootstrap complete",
            result={
                "region": region,
                "bucket": bucket,
                "role_name": role_name,
                "role_arn": role_arn,
                "instance_profile_arn": profile_arn,
                "security_group_name": security_group_name,
                "security_group_id": security_group_id,
            },
        )

    def setup_data(self, request: Ec2SetupDataRequest) -> CloudEc2Response:
        region = request.region or DEFAULT_REGION
        bucket = self._require(request.bucket or DEFAULT_BUCKET, "bucket or NUMERENG_S3_BUCKET")
        data_version = request.data_version or DEFAULT_DATA_VERSION
        cache_root = Path(request.cache_dir or ".numereng/datasets")
        version_dir = cache_root / data_version

        if not version_dir.exists():
            raise CloudEc2Error(f"data version directory does not exist: {version_dir}")

        required_files = [
            "train.parquet",
            "validation.parquet",
            "features.json",
        ]
        optional_files = ["validation_example_preds.parquet"]
        optional_files.extend(
            [
                "downsampled_full.parquet",
                "downsampled_full_benchmark_models.parquet",
            ]
        )

        s3 = self._s3(region)
        s3.ensure_bucket_exists(bucket=bucket, region=region)

        uploaded: dict[str, str] = {}
        for filename in required_files:
            local_path = version_dir / filename
            if not local_path.exists():
                raise CloudEc2Error(f"missing required data file: {local_path}")
            key = f"data/{data_version}/{filename}"
            uploaded[filename] = s3.upload_file(local_path=local_path, bucket=bucket, key=key)

        missing_optional: list[str] = []
        for filename in optional_files:
            local_path = version_dir / filename
            if not local_path.exists():
                missing_optional.append(filename)
                continue
            key = f"data/{data_version}/{filename}"
            uploaded[filename] = s3.upload_file(local_path=local_path, bucket=bucket, key=key)

        return CloudEc2Response(
            action="cloud.ec2.setup-data",
            message="Data synchronized to S3",
            result={
                "region": region,
                "bucket": bucket,
                "data_version": data_version,
                "uploaded": uploaded,
                "missing_optional": missing_optional,
            },
        )

    def provision(self, request: Ec2ProvisionRequest) -> CloudEc2Response:
        state = self._load_state(request)
        run_id = self._require(request.run_id or state.run_id, "run_id")
        region = self._resolve_region(request.region, state)
        bucket = self._require(self._resolve_bucket(request.bucket, state), "bucket or NUMERENG_S3_BUCKET")
        data_version = self._resolve_data_version(request.data_version, state)

        instance_type = resolve_instance_type(request.tier)
        gpu = is_gpu_instance(instance_type)
        ami_id = self._require(
            DEFAULT_GPU_AMI if gpu else DEFAULT_CPU_AMI,
            "NUMERENG_EC2_AMI_GPU" if gpu else "NUMERENG_EC2_AMI_CPU",
        )
        role_name = self._require(DEFAULT_IAM_ROLE, "NUMERENG_EC2_IAM_ROLE")
        security_group_name = self._require(DEFAULT_SECURITY_GROUP, "NUMERENG_EC2_SECURITY_GROUP")

        ec2 = self._ec2(region)
        ssm = self._ssm(region)
        spot_price = ec2.get_spot_price(instance_type) if request.use_spot else None

        launch_spec = LaunchInstanceSpec(
            image_id=ami_id,
            instance_type=instance_type,
            user_data=_bootstrap_user_data(data_version=data_version, gpu=gpu),
            run_id=run_id,
            region=region,
            iam_role_name=role_name,
            security_group=security_group_name,
            bucket=bucket,
            data_version=data_version,
            use_spot=request.use_spot,
            volume_size_gb=150 if gpu else 100,
            tags={"Lifecycle": "interactive"},
        )
        instance_id = ec2.launch_instance(launch_spec)

        if not ec2.wait_for_instance(instance_id, target_state="running", timeout_seconds=180):
            raise CloudEc2Error(f"instance {instance_id} failed to reach running state")

        ssm.wait_for_ssm(instance_id=instance_id, timeout_seconds=420)
        ready = False
        for _ in range(30):
            try:
                ssm.run_command(
                    instance_id=instance_id,
                    command="test -f /opt/numereng/.ready",
                    timeout_seconds=30,
                )
                ready = True
                break
            except Exception:
                time.sleep(10)

        if not ready:
            raise CloudEc2Error(f"instance {instance_id} bootstrap did not finish")

        next_state = state.model_copy(
            update={
                "run_id": run_id,
                "external_run_id": run_id,
                "instance_id": instance_id,
                "region": region,
                "bucket": bucket,
                "instance_type": instance_type,
                "is_gpu": gpu,
                "data_version": data_version,
                "status": "ready",
                "metadata": {
                    **state.metadata,
                    "tier": request.tier,
                    "use_spot": "true" if request.use_spot else "false",
                },
            },
            deep=True,
        )
        next_state = self._persist_state(request, next_state)

        result: dict[str, object] = {
            "instance_id": instance_id,
            "instance_type": instance_type,
            "is_gpu": gpu,
            "run_id": run_id,
            "region": region,
            "bucket": bucket,
            "use_spot": request.use_spot,
        }
        if spot_price is not None:
            result["spot_price"] = spot_price

        return CloudEc2Response(
            action="cloud.ec2.provision",
            message="Instance provisioned and SSM-ready",
            state=next_state,
            result=result,
        )

    def package_build_upload(self, request: Ec2PackageBuildUploadRequest) -> CloudEc2Response:
        state = self._load_state(request)
        run_id = self._require(request.run_id or state.run_id, "run_id")
        region = self._resolve_region(request.region, state)
        bucket = self._require(self._resolve_bucket(request.bucket, state), "bucket or NUMERENG_S3_BUCKET")

        s3 = self._s3(region)
        s3.ensure_bucket_exists(bucket=bucket, region=region)

        uploaded: dict[str, str] = {}
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            artifacts = self._wheel_builder.build_assets(tmp_dir)
            for artifact in artifacts:
                key = f"runs/{run_id}/package/{artifact.name}"
                uploaded[artifact.name] = s3.upload_file(local_path=artifact, bucket=bucket, key=key)

        next_state = state.model_copy(
            update={
                "run_id": run_id,
                "region": region,
                "bucket": bucket,
                "status": "package_uploaded",
                "artifacts": {**state.artifacts, **uploaded},
            },
            deep=True,
        )
        next_state = self._persist_state(request, next_state)

        return CloudEc2Response(
            action="cloud.ec2.package.build-upload",
            message="Package artifacts uploaded",
            state=next_state,
            result={"uploaded": uploaded},
        )

    def config_upload(self, request: Ec2ConfigUploadRequest) -> CloudEc2Response:
        state = self._load_state(request)
        run_id = self._require(request.run_id or state.run_id, "run_id")
        region = self._resolve_region(request.region, state)
        bucket = self._require(self._resolve_bucket(request.bucket, state), "bucket or NUMERENG_S3_BUCKET")
        config_path = Path(request.config_path)
        if not config_path.exists():
            raise CloudEc2Error(f"config path does not exist: {config_path}")

        s3 = self._s3(region)
        key = f"runs/{run_id}/config.json"
        uri = s3.upload_file(local_path=config_path, bucket=bucket, key=key)

        next_state = state.model_copy(
            update={
                "run_id": run_id,
                "region": region,
                "bucket": bucket,
                "status": "config_uploaded",
                "artifacts": {**state.artifacts, "config": uri},
            },
            deep=True,
        )
        next_state = self._persist_state(request, next_state)

        return CloudEc2Response(
            action="cloud.ec2.config.upload",
            message="Run config uploaded",
            state=next_state,
            result={"config_uri": uri},
        )

    def push(self, request: Ec2PushRequest) -> CloudEc2Response:
        state = self._load_state(request)
        run_id = self._require(request.run_id or state.run_id, "run_id")
        instance_id = self._require(request.instance_id or state.instance_id, "instance_id")
        region = self._resolve_region(request.region, state)
        bucket = self._require(self._resolve_bucket(request.bucket, state), "bucket or NUMERENG_S3_BUCKET")
        data_version = self._resolve_data_version(request.data_version, state)

        ssm = self._ssm(region)

        quoted_bucket = shlex.quote(bucket)
        quoted_run_id = shlex.quote(run_id)
        quoted_region = shlex.quote(region)
        quoted_version = shlex.quote(data_version)

        ssm.run_command(
            instance_id=instance_id,
            command="mkdir -p /opt/numereng/package /opt/numereng/.numereng/configs",
            timeout_seconds=60,
        )
        ssm.run_command(
            instance_id=instance_id,
            command=(
                "mkdir -p /opt/numereng/.numereng/datasets/"
                f"{quoted_version} && "
                f"aws s3 sync s3://{quoted_bucket}/runs/{quoted_run_id}/package/ /opt/numereng/package/ "
                f"--region {quoted_region}"
            ),
            timeout_seconds=300,
        )
        ssm.run_command(
            instance_id=instance_id,
            command=(
                f"aws s3 cp s3://{quoted_bucket}/runs/{quoted_run_id}/config.json "
                f"/opt/numereng/.numereng/configs/run_config.json --region {quoted_region}"
            ),
            timeout_seconds=120,
        )

        required_data = ["train.parquet", "validation.parquet", "features.json"]
        optional_data = ["validation_example_preds.parquet"]

        for filename in required_data:
            quoted_file = shlex.quote(filename)
            ssm.run_command(
                instance_id=instance_id,
                command=(
                    f"aws s3 cp s3://{quoted_bucket}/data/{quoted_version}/{quoted_file} "
                    f"/opt/numereng/.numereng/datasets/{quoted_version}/{quoted_file} "
                    f"--region {quoted_region}"
                ),
                timeout_seconds=300,
            )

        for filename in optional_data:
            quoted_file = shlex.quote(filename)
            ssm.run_command(
                instance_id=instance_id,
                command=(
                    f"aws s3 cp s3://{quoted_bucket}/data/{quoted_version}/{quoted_file} "
                    f"/opt/numereng/.numereng/datasets/{quoted_version}/{quoted_file} "
                    f"--region {quoted_region} 2>/dev/null || true"
                ),
                timeout_seconds=300,
            )

        next_state = state.model_copy(
            update={
                "run_id": run_id,
                "instance_id": instance_id,
                "region": region,
                "bucket": bucket,
                "data_version": data_version,
                "status": "artifacts_pushed",
            },
            deep=True,
        )
        next_state = self._persist_state(request, next_state)

        return CloudEc2Response(
            action="cloud.ec2.push",
            message="Package/config/data pushed to instance",
            state=next_state,
            result={"instance_id": instance_id, "run_id": run_id},
        )

    def install(self, request: Ec2InstallRequest) -> CloudEc2Response:
        state = self._load_state(request)
        run_id = self._require(request.run_id or state.run_id, "run_id")
        instance_id = self._require(request.instance_id or state.instance_id, "instance_id")
        region = self._resolve_region(request.region, state)
        runtime_profile = self._resolve_runtime_profile(request.runtime_profile, state)
        if runtime_profile == CUDA_RUNTIME_PROFILE and not state.is_gpu:
            raise CloudEc2Error("cloud_ec2_cuda_install_requires_gpu_instance")

        ssm = self._ssm(region)
        activate = ". /opt/numereng/.venv/bin/activate"

        ssm.run_command(
            instance_id=instance_id,
            command=(
                "export PATH=/root/.local/bin:$PATH && "
                f"{activate} && "
                "uv pip install -r /opt/numereng/package/requirements.txt"
            ),
            timeout_seconds=900,
        )
        ssm.run_command(
            instance_id=instance_id,
            command=(
                "export PATH=/root/.local/bin:$PATH && "
                f"{activate} && "
                "uv pip install --no-deps /opt/numereng/package/numereng-*.whl"
            ),
            timeout_seconds=240,
        )
        if runtime_profile == CUDA_RUNTIME_PROFILE:
            ssm.run_command(
                instance_id=instance_id,
                command=(
                    "export PATH=/root/.local/bin:$PATH && "
                    "test -x /usr/bin/nvidia-smi && /usr/bin/nvidia-smi >/dev/null && "
                    f"{activate} && "
                    "CMAKE_ARGS='-DUSE_CUDA=ON' uv pip install --force-reinstall --no-binary lightgbm lightgbm"
                ),
                timeout_seconds=1800,
            )
        ssm.run_command(
            instance_id=instance_id,
            command=f"{activate} && python -c 'import numereng; print(\"ok\")'",
            timeout_seconds=60,
        )

        next_state = state.model_copy(
            update={
                "run_id": run_id,
                "instance_id": instance_id,
                "region": region,
                "runtime_profile": runtime_profile,
                "status": "runtime_installed",
            },
            deep=True,
        )
        next_state = self._persist_state(request, next_state)

        return CloudEc2Response(
            action="cloud.ec2.install",
            message="Remote runtime installed",
            state=next_state,
            result={"instance_id": instance_id, "run_id": run_id},
        )

    def train_start(self, request: Ec2TrainStartRequest) -> CloudEc2Response:
        state = self._load_state(request)
        run_id = self._require(request.run_id or state.run_id, "run_id")
        instance_id = self._require(request.instance_id or state.instance_id, "instance_id")
        region = self._resolve_region(request.region, state)
        submitted_at = _utc_now_iso()
        execution_payload = serialize_run_execution(
            self._cloud_execution(
                request=request,
                state=state.model_copy(
                    update={
                        "run_id": run_id,
                        "instance_id": instance_id,
                        "region": region,
                    },
                    deep=True,
                ),
                metadata={
                    **state.metadata,
                    "submitted_at": submitted_at,
                    "bucket": state.bucket or "",
                    "runtime_profile": state.runtime_profile,
                },
            )
        )
        encoded_execution = base64.urlsafe_b64encode(execution_payload.encode("utf-8")).decode("ascii")

        ssm = self._ssm(region)
        command = (
            "cd /opt/numereng && "
            ". .venv/bin/activate && "
            "PYTHONUNBUFFERED=1 nohup bash -c '"
            f"{RUN_EXECUTION_ENV_B64_VAR}={encoded_execution} "
            "numereng run train "
            "--config /opt/numereng/.numereng/configs/run_config.json "
            "--output-dir /opt/numereng/.numereng "
            "2>&1; "
            "echo $? > /opt/numereng/training.exit_code"
            "' > /var/log/numereng-training.log 2>&1 & "
            "echo $! > /opt/numereng/training.pid && cat /opt/numereng/training.pid"
        )
        result = ssm.run_command(instance_id=instance_id, command=command, timeout_seconds=120)
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise CloudEc2Error("remote training start returned empty PID output")
        pid_text = lines[-1]
        if not pid_text.isdigit():
            raise CloudEc2Error(f"remote training start returned invalid PID: {pid_text}")
        pid = int(pid_text)

        next_state = state.model_copy(
            update={
                "run_id": run_id,
                "instance_id": instance_id,
                "region": region,
                "training_pid": pid,
                "status": "training_running",
                "metadata": {
                    **state.metadata,
                    "submitted_at": submitted_at,
                },
            },
            deep=True,
        )
        next_state = self._persist_state(request, next_state)

        return CloudEc2Response(
            action="cloud.ec2.train.start",
            message="Remote training started",
            state=next_state,
            result={"training_pid": pid, "instance_id": instance_id, "run_id": run_id},
        )

    def train_poll(self, request: Ec2TrainPollRequest) -> CloudEc2Response:
        state = self._load_state(request)
        run_id = self._require(request.run_id or state.run_id, "run_id")
        instance_id = self._require(request.instance_id or state.instance_id, "instance_id")
        region = self._resolve_region(request.region, state)
        training_pid = state.training_pid
        if training_pid is None:
            raise CloudEc2Error("missing required value: training_pid (run train start first)")

        ssm = self._ssm(region)
        ec2 = self._ec2(region)

        status = "timeout"
        start = time.time()
        while time.time() - start < request.timeout_seconds:
            try:
                poll = ssm.run_command(
                    instance_id=instance_id,
                    command=(
                        f"cat /opt/numereng/training.exit_code 2>/dev/null || "
                        f"(test -d /proc/{training_pid} && echo RUNNING || echo -1)"
                    ),
                    timeout_seconds=60,
                )
                output = poll.stdout.strip()
            except Exception:
                instance = ec2.get_instance_status(instance_id)
                if instance.state in {"terminated", "shutting-down"}:
                    status = "failed"
                    break
                time.sleep(max(1, request.interval_seconds))
                continue

            if output == "RUNNING":
                time.sleep(max(1, request.interval_seconds))
                continue

            exit_code = int(output) if output.lstrip("-").isdigit() else -1
            if exit_code == 0:
                status = "completed"
            else:
                status = "failed"
            break

        next_state = state.model_copy(
            update={
                "run_id": run_id,
                "instance_id": instance_id,
                "region": region,
                "status": f"training_{status}",
            },
            deep=True,
        )
        next_state = self._persist_state(request, next_state)

        message = "Remote training completed" if status == "completed" else "Remote training not completed successfully"
        return CloudEc2Response(
            action="cloud.ec2.train.poll",
            message=message,
            state=next_state,
            result={
                "run_id": run_id,
                "instance_id": instance_id,
                "training_pid": training_pid,
                "status": status,
                "timeout_seconds": request.timeout_seconds,
            },
        )

    def logs(self, request: Ec2LogsRequest) -> CloudEc2Response:
        state = self._load_state(request)
        instance_id = self._require(request.instance_id or state.instance_id, "instance_id")
        region = self._resolve_region(request.region, state)
        ssm = self._ssm(region)

        command = (
            f"tail -n {int(request.lines)} /var/log/numereng-training.log "
            f"|| tail -n {int(request.lines)} /var/log/numerai-training.log || true"
        )
        result = ssm.run_command(instance_id=instance_id, command=command, timeout_seconds=60)

        message = "Fetched remote log tail"
        if request.follow:
            message = "Fetched remote log tail (follow mode currently returns a snapshot)"

        return CloudEc2Response(
            action="cloud.ec2.logs",
            message=message,
            state=state if state.instance_id is not None else None,
            result={
                "instance_id": instance_id,
                "lines": request.lines,
                "follow": request.follow,
                "log": result.stdout,
            },
        )

    def pull(self, request: Ec2PullRequest) -> CloudEc2Response:
        state = self._load_state(request)
        run_id = self._require(request.run_id or state.run_id, "run_id")
        instance_id = self._require(request.instance_id or state.instance_id, "instance_id")
        region = self._resolve_region(request.region, state)
        bucket = self._require(self._resolve_bucket(request.bucket, state), "bucket or NUMERENG_S3_BUCKET")
        output_root = Path(request.output_dir or request.store_root).expanduser().resolve()
        try:
            ensure_allowed_store_target(
                target_path=output_root,
                store_root=request.store_root,
                allowed_prefixes=("runs",),
                allow_store_root=True,
                error_code="cloud_ec2_pull_output_dir_noncanonical",
            )
        except ValueError as exc:
            raise CloudEc2Error(str(exc)) from exc

        ssm = self._ssm(region)
        s3 = self._s3(region)

        quoted_run_id = shlex.quote(run_id)
        quoted_bucket = shlex.quote(bucket)
        quoted_region = shlex.quote(region)

        ssm.run_command(
            instance_id=instance_id,
            command=(
                "cd /opt/numereng && "
                f"aws s3 sync .numereng/runs/{quoted_run_id} s3://{quoted_bucket}/runs/{quoted_run_id}/ "
                f"--exclude 'run.json' --region {quoted_region}"
            ),
            timeout_seconds=900,
        )
        ssm.run_command(
            instance_id=instance_id,
            command=(
                "cd /opt/numereng && "
                f"[ -f .numereng/runs/{quoted_run_id}/run.json ] && "
                f"aws s3 cp .numereng/runs/{quoted_run_id}/run.json s3://{quoted_bucket}/runs/{quoted_run_id}/run.json "
                f"--region {quoted_region} || true"
            ),
            timeout_seconds=120,
        )

        prefix = f"runs/{run_id}/"
        keys = s3.list_keys(bucket=bucket, prefix=prefix)
        downloaded: list[str] = []
        skipped_unsafe_keys: list[str] = []
        for key in keys:
            relative = key[len(prefix) :] if key.startswith(prefix) else Path(key).name
            local_path = _resolve_safe_relative_path(base_dir=output_root, relative=relative)
            if local_path is None:
                skipped_unsafe_keys.append(key)
                continue
            s3.download_file(bucket=bucket, key=key, local_path=local_path)
            downloaded.append(str(local_path))

        next_state = state.model_copy(
            update={
                "run_id": run_id,
                "canonical_run_id": run_id,
                "instance_id": instance_id,
                "region": region,
                "bucket": bucket,
                "status": "results_pulled",
                "metadata": {
                    **state.metadata,
                    "pulled_at": _utc_now_iso(),
                },
            },
            deep=True,
        )
        next_state = self._persist_state(request, next_state)

        manifest_candidates = (
            output_root / "run.json",
            output_root / "runs" / run_id / "run.json",
        )
        manifest_path = next((candidate for candidate in manifest_candidates if candidate.is_file()), None)
        if manifest_path is not None:
            try:
                stamp_run_execution(
                    manifest_path=manifest_path,
                    execution=self._cloud_execution(
                        request=request,
                        state=next_state,
                        metadata={
                            **next_state.metadata,
                            "bucket": bucket,
                            "runtime_profile": next_state.runtime_profile,
                        },
                    ),
                )
            except (OSError, ValueError) as exc:
                next_state = self._persist_state(
                    request,
                    next_state.model_copy(
                        update={
                            "metadata": {
                                **next_state.metadata,
                                "execution_stamp_error": str(exc),
                            }
                        },
                        deep=True,
                    ),
                )
        canonical_manifest = (Path(request.store_root).expanduser().resolve() / "runs" / run_id / "run.json").resolve()
        if canonical_manifest.is_file():
            try:
                index_run(store_root=request.store_root, run_id=run_id)
            except StoreError:
                pass

        return CloudEc2Response(
            action="cloud.ec2.pull",
            message="Run artifacts synchronized to local output",
            state=next_state,
            result={
                "run_id": run_id,
                "output_root": str(output_root),
                "downloaded_count": len(downloaded),
                "skipped_unsafe_keys": skipped_unsafe_keys,
            },
        )

    def terminate(self, request: Ec2TerminateRequest) -> CloudEc2Response:
        state = self._load_state(request)
        instance_id = self._require(request.instance_id or state.instance_id, "instance_id")
        region = self._resolve_region(request.region, state)

        ec2 = self._ec2(region)
        ec2.terminate_instance(instance_id)

        next_state = state.model_copy(
            update={
                "instance_id": instance_id,
                "region": region,
                "status": "terminated",
            },
            deep=True,
        )
        next_state = self._persist_state(request, next_state)

        return CloudEc2Response(
            action="cloud.ec2.terminate",
            message="Instance termination requested",
            state=next_state,
            result={"instance_id": instance_id, "region": region},
        )

    def status(self, request: Ec2StatusRequest) -> CloudEc2Response:
        state = self._load_state(request)
        region = self._resolve_region(request.region, state)
        run_id = request.run_id or state.run_id

        ec2 = self._ec2(region)

        instance_status: InstanceStatus | None = None
        if run_id is not None:
            for candidate in ec2.list_training_instances():
                if candidate.run_id == run_id:
                    instance_status = candidate
                    break
        elif state.instance_id is not None:
            instance_status = ec2.get_instance_status(state.instance_id)

        result: dict[str, object] = {
            "region": region,
            "run_id": run_id,
            "state_document": state.model_dump(),
        }
        if instance_status is not None:
            result["instance"] = {
                "instance_id": instance_status.instance_id,
                "state": instance_status.state,
                "instance_type": instance_status.instance_type,
                "run_id": instance_status.run_id,
                "public_ip": instance_status.public_ip,
                "private_ip": instance_status.private_ip,
                "launch_time": instance_status.launch_time,
            }

        message = "Status fetched"
        if instance_status is None:
            message = "Status fetched (no active instance found)"

        return CloudEc2Response(
            action="cloud.ec2.status",
            message=message,
            state=state if state.run_id is not None or state.instance_id is not None else None,
            result=result,
        )

    def s3_list(self, request: Ec2S3ListRequest) -> CloudEc2Response:
        region = request.region or DEFAULT_REGION
        bucket = self._require(request.bucket or DEFAULT_BUCKET, "bucket or NUMERENG_S3_BUCKET")
        prefix = request.prefix.lstrip("/")
        keys = self._s3(region).list_keys(bucket=bucket, prefix=prefix)
        return CloudEc2Response(
            action="cloud.ec2.s3.ls",
            message="S3 objects listed",
            result={"bucket": bucket, "region": region, "prefix": prefix, "keys": keys},
        )

    def s3_copy(self, request: Ec2S3CopyRequest) -> CloudEc2Response:
        region = request.region or DEFAULT_REGION
        default_bucket = request.bucket or DEFAULT_BUCKET
        s3 = self._s3(region)

        src_s3 = _parse_s3_uri(request.src, default_bucket=default_bucket)
        dst_s3 = _parse_s3_uri(request.dst, default_bucket=default_bucket)

        if src_s3 is not None and dst_s3 is not None:
            uri = s3.copy_object(
                src_bucket=src_s3[0],
                src_key=src_s3[1],
                dst_bucket=dst_s3[0],
                dst_key=dst_s3[1],
            )
            return CloudEc2Response(
                action="cloud.ec2.s3.cp",
                message="Copied object within S3",
                result={"destination": uri},
            )

        if src_s3 is not None and dst_s3 is None:
            src_bucket, src_key = src_s3
            dst_path = Path(request.dst)

            if request.src.endswith("/") or src_key.endswith("/"):
                keys = s3.list_keys(bucket=src_bucket, prefix=src_key)
                downloaded: list[str] = []
                skipped_unsafe_keys: list[str] = []
                for key in keys:
                    rel = key[len(src_key) :] if key.startswith(src_key) else Path(key).name
                    local_path = _resolve_safe_relative_path(base_dir=dst_path, relative=rel)
                    if local_path is None:
                        skipped_unsafe_keys.append(key)
                        continue
                    s3.download_file(bucket=src_bucket, key=key, local_path=local_path)
                    downloaded.append(str(local_path))
                return CloudEc2Response(
                    action="cloud.ec2.s3.cp",
                    message="Downloaded S3 prefix",
                    result={
                        "downloaded": downloaded,
                        "count": len(downloaded),
                        "skipped_unsafe_keys": skipped_unsafe_keys,
                    },
                )

            local_target = dst_path
            if dst_path.exists() and dst_path.is_dir():
                local_target = dst_path / Path(src_key).name
            s3.download_file(bucket=src_bucket, key=src_key, local_path=local_target)
            return CloudEc2Response(
                action="cloud.ec2.s3.cp",
                message="Downloaded S3 object",
                result={"downloaded": str(local_target)},
            )

        if src_s3 is None and dst_s3 is not None:
            src_path = Path(request.src)
            if not src_path.exists():
                raise CloudEc2Error(f"source path does not exist: {src_path}")
            if not src_path.is_file():
                raise CloudEc2Error("source path must be a file for local->S3 copy")

            dst_bucket, dst_key = dst_s3
            final_key = dst_key
            if final_key.endswith("/"):
                final_key = f"{final_key}{src_path.name}"
            uri = s3.upload_file(local_path=src_path, bucket=dst_bucket, key=final_key)
            return CloudEc2Response(
                action="cloud.ec2.s3.cp",
                message="Uploaded local file to S3",
                result={"destination": uri},
            )

        raise CloudEc2Error("either src or dst must be an s3:// URI")

    def s3_remove(self, request: Ec2S3RemoveRequest) -> CloudEc2Response:
        region = request.region or DEFAULT_REGION
        default_bucket = request.bucket or DEFAULT_BUCKET
        parsed = _parse_s3_uri(request.uri, default_bucket=default_bucket)
        if parsed is None:
            raise CloudEc2Error("--uri must be an s3:// URI")
        bucket, key = parsed

        s3 = self._s3(region)
        deleted_count = 0
        if request.recursive or key.endswith("/"):
            deleted_count = s3.delete_prefix(bucket=bucket, prefix=key)
        else:
            s3.delete_key(bucket=bucket, key=key)
            deleted_count = 1

        return CloudEc2Response(
            action="cloud.ec2.s3.rm",
            message="S3 objects deleted",
            result={"bucket": bucket, "key": key, "deleted_count": deleted_count},
        )


def _parse_s3_uri(value: str, *, default_bucket: str) -> tuple[str, str] | None:
    value_stripped = value.strip()
    if not value_stripped.startswith("s3://"):
        return None

    body = value_stripped[len("s3://") :]
    if not body:
        raise CloudEc2Error("invalid s3 URI: missing bucket")

    if "/" not in body:
        return body, ""

    bucket, key = body.split("/", 1)
    if not bucket:
        bucket = default_bucket
    return bucket, key


def _resolve_safe_relative_path(*, base_dir: Path, relative: str) -> Path | None:
    stripped = relative.lstrip("/")
    if not stripped:
        return None
    relative_path = Path(stripped)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return None
    root = base_dir.resolve()
    destination = (root / relative_path).resolve()
    try:
        destination.relative_to(root)
    except ValueError:
        return None
    return destination


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _is_canonical_cloud_state_path(path: Path) -> bool:
    parts = path.parts
    for index in range(len(parts) - 2):
        if parts[index] == ".numereng" and parts[index + 1] == "cloud":
            return True
    return False
