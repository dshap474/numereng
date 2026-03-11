"""Service layer for managed AWS training workflows (ECR + SageMaker/Batch)."""

from __future__ import annotations

import json
import os
import re
import shutil
import tarfile
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

from numereng.config.training import TrainingConfigLoaderError, load_training_config_json
from numereng.features.cloud.aws.adapters import (
    BatchAdapter,
    BatchJobSpec,
    CloudWatchLogsAdapter,
    DockerAdapter,
    EcrAdapter,
    S3Adapter,
    SageMakerAdapter,
    SageMakerTrainingSpec,
)
from numereng.features.cloud.aws.aws_adapters import (
    AwsBatchAdapter,
    AwsCloudWatchLogsAdapter,
    AwsEcrAdapter,
    AwsS3Adapter,
    AwsSageMakerAdapter,
    SubprocessDockerAdapter,
)
from numereng.features.cloud.aws.managed_contracts import (
    AwsImageBuildPushRequest,
    AwsTrainCancelRequest,
    AwsTrainExtractRequest,
    AwsTrainLogsRequest,
    AwsTrainPullRequest,
    AwsTrainStatusRequest,
    AwsTrainSubmitRequest,
    CloudAwsRequestBase,
    CloudAwsResponse,
    CloudAwsState,
    CloudRuntimeProfile,
)
from numereng.features.cloud.aws.managed_state_store import CloudAwsStateStore
from numereng.features.store import StoreCloudJobUpsert, StoreError, index_run, upsert_cloud_job
from numereng.features.store.layout import ensure_allowed_store_target
from numereng.features.training.service import resolve_model_config


class CloudAwsError(Exception):
    """Feature-level error for managed AWS cloud orchestration."""


DEFAULT_REGION = os.getenv("NUMERENG_AWS_REGION", "us-east-2") or "us-east-2"
DEFAULT_BUCKET = os.getenv("NUMERENG_S3_BUCKET", "numereng-artifacts") or "numereng-artifacts"
DEFAULT_ECR_REPOSITORY = (
    os.getenv("NUMERENG_AWS_ECR_REPOSITORY", "numereng-training") or "numereng-training"
)
DEFAULT_SAGEMAKER_ROLE_ARN = os.getenv("NUMERENG_AWS_SAGEMAKER_ROLE_ARN", "")
DEFAULT_BATCH_JOB_QUEUE = os.getenv("NUMERENG_AWS_BATCH_JOB_QUEUE", "")
DEFAULT_BATCH_JOB_DEFINITION = os.getenv("NUMERENG_AWS_BATCH_JOB_DEFINITION", "")
DEFAULT_INSTANCE_TYPE = os.getenv("NUMERENG_AWS_SAGEMAKER_INSTANCE_TYPE", "ml.m5.2xlarge") or "ml.m5.2xlarge"
DEFAULT_RUNTIME_PROFILE: CloudRuntimeProfile = "standard"
CUDA_RUNTIME_PROFILE: CloudRuntimeProfile = "lgbm-cuda"
STANDARD_SAGEMAKER_DOCKERFILE = "docker/Dockerfile.sagemaker"
CUDA_SAGEMAKER_DOCKERFILE = "docker/Dockerfile.sagemaker-lgbm-cuda"
_DEFAULT_JOB_NAME_PREFIX = "numereng"
_SAFE_TOKEN = re.compile(r"[^a-zA-Z0-9-]+")
_SAFE_RUN_ID = re.compile(r"^[\w\-.]+$")


class CloudAwsManagedService:
    """Managed AWS service with pluggable adapters."""

    def __init__(
        self,
        *,
        ecr_factory: Callable[[str], EcrAdapter] | None = None,
        s3_factory: Callable[[str], S3Adapter] | None = None,
        sagemaker_factory: Callable[[str], SageMakerAdapter] | None = None,
        batch_factory: Callable[[str], BatchAdapter] | None = None,
        logs_factory: Callable[[str], CloudWatchLogsAdapter] | None = None,
        docker_adapter: DockerAdapter | None = None,
        state_store: CloudAwsStateStore | None = None,
    ) -> None:
        self._ecr_factory = ecr_factory or (lambda region: AwsEcrAdapter(region=region))
        self._s3_factory = s3_factory or (lambda region: AwsS3Adapter(region=region))
        self._sagemaker_factory = sagemaker_factory or (lambda region: AwsSageMakerAdapter(region=region))
        self._batch_factory = batch_factory or (lambda region: AwsBatchAdapter(region=region))
        self._logs_factory = logs_factory or (lambda region: AwsCloudWatchLogsAdapter(region=region))
        self._docker = docker_adapter or SubprocessDockerAdapter()
        self._state_store = state_store or CloudAwsStateStore()

    def _validated_state_path(self, request: CloudAwsRequestBase) -> Path | None:
        state_path = request.state_file()
        if state_path is None:
            return None
        resolved = state_path.expanduser().resolve()
        root = Path(request.store_root).expanduser().resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise CloudAwsError(f"aws_state_path_noncanonical:{resolved}") from exc
        if not relative.parts or relative.parts[0] != "cloud":
            raise CloudAwsError(f"aws_state_path_noncanonical:{resolved}")
        if resolved.suffix.lower() != ".json":
            raise CloudAwsError(f"aws_state_path_extension_invalid:{resolved}")
        return resolved

    def _load_state(self, request: CloudAwsRequestBase) -> CloudAwsState:
        state_path = self._validated_state_path(request)
        if state_path is None:
            return CloudAwsState()
        try:
            loaded = self._state_store.load(state_path)
        except Exception as exc:
            raise CloudAwsError(f"invalid state file: {state_path}") from exc
        if loaded is None:
            return CloudAwsState()
        return loaded

    def _persist_state(self, request: CloudAwsRequestBase, state: CloudAwsState) -> CloudAwsState:
        touched = state.touched()
        state_path = self._validated_state_path(request)
        if state_path is not None:
            self._state_store.save(state_path, touched)
        return touched

    def _resolve_region(self, explicit: str | None, state: CloudAwsState | None = None) -> str:
        if explicit is not None and explicit:
            return explicit
        if state is not None and state.region is not None and state.region:
            return state.region
        return DEFAULT_REGION

    def _resolve_bucket(self, explicit: str | None, state: CloudAwsState | None = None) -> str:
        if explicit is not None and explicit:
            return explicit
        if state is not None and state.bucket is not None and state.bucket:
            return state.bucket
        return DEFAULT_BUCKET

    def _resolve_runtime_profile(
        self,
        explicit: CloudRuntimeProfile | None,
        state: CloudAwsState | None = None,
    ) -> CloudRuntimeProfile:
        if explicit is not None:
            return explicit
        if state is not None:
            return state.runtime_profile
        return DEFAULT_RUNTIME_PROFILE

    def _default_dockerfile_for_profile(self, context_dir: Path, runtime_profile: CloudRuntimeProfile) -> Path | None:
        dockerfile_name = (
            CUDA_SAGEMAKER_DOCKERFILE if runtime_profile == CUDA_RUNTIME_PROFILE else STANDARD_SAGEMAKER_DOCKERFILE
        )
        dockerfile_path = context_dir / dockerfile_name
        if not dockerfile_path.is_file():
            raise CloudAwsError(f"dockerfile path does not exist: {dockerfile_path}")
        return dockerfile_path

    def _resolve_config_device(self, request: AwsTrainSubmitRequest) -> str | None:
        if request.config_path is None:
            return None
        try:
            payload = load_training_config_json(Path(request.config_path))
        except TrainingConfigLoaderError as exc:
            raise CloudAwsError(str(exc)) from exc
        model_payload = payload.get("model")
        if not isinstance(model_payload, dict):
            raise CloudAwsError("training_config_model_missing")
        model_type, model_params = resolve_model_config(model_payload)
        if model_type != "LGBMRegressor":
            return None
        device_type = model_params.get("device_type")
        if not isinstance(device_type, str):
            return None
        return device_type

    def _validate_runtime_profile_for_submit(
        self,
        *,
        backend: Literal["sagemaker", "batch"],
        runtime_profile: CloudRuntimeProfile,
        instance_type: str,
        config_device: str | None,
    ) -> None:
        if config_device != "cuda" and runtime_profile == CUDA_RUNTIME_PROFILE:
            raise CloudAwsError("aws_runtime_profile_requires_cuda_config")
        if config_device == "cuda":
            if backend != "sagemaker":
                raise CloudAwsError("aws_cuda_requires_sagemaker_backend")
            if runtime_profile != CUDA_RUNTIME_PROFILE:
                raise CloudAwsError("aws_cuda_requires_runtime_profile_lgbm_cuda")
            if not instance_type.startswith(("ml.g", "ml.p")):
                raise CloudAwsError(f"aws_cuda_requires_gpu_instance_type:{instance_type}")

    def _require(self, value: str | None, field_name: str) -> str:
        if value is None or not value:
            raise CloudAwsError(f"missing required value: {field_name}")
        return value

    def _ecr(self, region: str) -> EcrAdapter:
        return self._ecr_factory(region)

    def _s3(self, region: str) -> S3Adapter:
        return self._s3_factory(region)

    def _sagemaker(self, region: str) -> SageMakerAdapter:
        return self._sagemaker_factory(region)

    def _batch(self, region: str) -> BatchAdapter:
        return self._batch_factory(region)

    def _logs(self, region: str) -> CloudWatchLogsAdapter:
        return self._logs_factory(region)

    def image_build_push(self, request: AwsImageBuildPushRequest) -> CloudAwsResponse:
        state = self._load_state(request)
        region = self._resolve_region(request.region, state)
        bucket = self._resolve_bucket(request.bucket, state)
        run_id = request.run_id or state.run_id or self._generated_run_id()
        repository = request.repository or state.repository or DEFAULT_ECR_REPOSITORY
        image_tag = request.image_tag or state.image_tag or _slug_token(run_id)
        runtime_profile = self._resolve_runtime_profile(request.runtime_profile, state)

        context_dir = Path(request.context_dir).expanduser().resolve()
        if not context_dir.is_dir():
            raise CloudAwsError(f"docker build context does not exist: {context_dir}")

        dockerfile_path = self._default_dockerfile_for_profile(context_dir, runtime_profile)
        if request.dockerfile is not None:
            dockerfile_path = Path(request.dockerfile).expanduser().resolve()
            if not dockerfile_path.is_file():
                raise CloudAwsError(f"dockerfile path does not exist: {dockerfile_path}")

        ecr = self._ecr(region)
        repository_uri = ecr.ensure_repository(repository)
        image_uri = f"{repository_uri}:{image_tag}"
        local_tag = f"{repository}:{image_tag}"

        self._docker.build_image(
            context_dir=context_dir,
            tag=local_tag,
            dockerfile=dockerfile_path,
            build_args=request.build_args,
            platform=request.platform,
        )
        self._docker.tag_image(source_tag=local_tag, target_tag=image_uri)

        registry = repository_uri.split("/", 1)[0]
        self._docker.login(registry=registry, username="AWS", password=ecr.get_login_password())
        self._docker.push_image(tag=image_uri)

        digest = ecr.get_image_digest(repository, image_tag)

        next_artifacts = dict(state.artifacts)
        next_artifacts["image_uri"] = image_uri

        next_state = state.model_copy(
            update={
                "run_id": run_id,
                "region": region,
                "bucket": bucket,
                "repository": repository,
                "image_tag": image_tag,
                "image_uri": image_uri,
                "runtime_profile": runtime_profile,
                "status": "image_pushed",
                "artifacts": next_artifacts,
            },
            deep=True,
        )
        persisted = self._persist_state(request, next_state)

        self._record_cloud_job(
            request=request,
            run_id=run_id,
            provider="ecr",
            backend="image",
            provider_job_id=image_tag,
            status="IMAGE_PUSHED",
            region=region,
            image_uri=image_uri,
            output_s3_uri=None,
            error_message=None,
        )

        return CloudAwsResponse(
            action="cloud.aws.image.build-push",
            message="docker image built and pushed to ECR",
            state=persisted,
            result={
                "run_id": run_id,
                "region": region,
                "bucket": bucket,
                "repository": repository,
                "image_tag": image_tag,
                "image_uri": image_uri,
                "runtime_profile": runtime_profile,
                "image_digest": digest,
            },
        )

    def train_submit(self, request: AwsTrainSubmitRequest) -> CloudAwsResponse:
        state = self._load_state(request)
        backend = request.backend
        region = self._resolve_region(request.region, state)
        bucket = self._resolve_bucket(request.bucket, state)
        run_id = request.run_id or state.run_id
        run_id = self._require(run_id, "run_id")
        runtime_profile = self._resolve_runtime_profile(request.runtime_profile, state)
        config_device = self._resolve_config_device(request)
        instance_type = request.instance_type or DEFAULT_INSTANCE_TYPE
        self._validate_runtime_profile_for_submit(
            backend=backend,
            runtime_profile=runtime_profile,
            instance_type=instance_type,
            config_device=config_device,
        )

        config_uri = self._resolve_config_uri(request=request, run_id=run_id, region=region, bucket=bucket)
        image_uri = request.image_uri or state.image_uri
        image_uri = self._require(image_uri, "image_uri")

        output_s3_uri = (
            request.output_s3_uri
            or state.artifacts.get("output_s3_uri")
            or f"s3://{bucket}/runs/{run_id}/managed-output/"
        )
        checkpoint_s3_uri = request.checkpoint_s3_uri
        if checkpoint_s3_uri is None and request.use_spot:
            checkpoint_s3_uri = f"s3://{bucket}/runs/{run_id}/checkpoints/"

        if backend == "sagemaker":
            role_arn = request.role_arn or DEFAULT_SAGEMAKER_ROLE_ARN
            role_arn = self._require(role_arn, "role_arn")
            training_job_name = self._job_name(run_id, prefix="neng-sm")

            max_wait_seconds = request.max_wait_seconds
            if max_wait_seconds is None and request.use_spot:
                max_wait_seconds = request.max_runtime_seconds * 3

            env = {
                "NUMERENG_RUN_ID": run_id,
                "NUMERENG_CONFIG_S3_URI": config_uri,
                "NUMERENG_OUTPUT_S3_URI": output_s3_uri,
                **request.env,
            }

            spec = SageMakerTrainingSpec(
                job_name=training_job_name,
                image_uri=image_uri,
                role_arn=role_arn,
                input_config_uri=config_uri,
                output_s3_uri=output_s3_uri,
                checkpoint_s3_uri=checkpoint_s3_uri,
                instance_type=instance_type,
                instance_count=request.instance_count,
                volume_size_gb=request.volume_size_gb,
                max_runtime_seconds=request.max_runtime_seconds,
                max_wait_seconds=max_wait_seconds,
                use_spot=request.use_spot,
                environment=env,
                tags={"Project": "numereng", "RunId": run_id},
            )
            training_status = self._sagemaker(region).start_training(spec)

            next_artifacts = dict(state.artifacts)
            next_artifacts["config_s3_uri"] = config_uri
            next_artifacts["output_s3_uri"] = output_s3_uri
            if checkpoint_s3_uri is not None:
                next_artifacts["checkpoint_s3_uri"] = checkpoint_s3_uri

            next_state = state.model_copy(
                update={
                    "run_id": run_id,
                    "backend": "sagemaker",
                    "region": region,
                    "bucket": bucket,
                    "image_uri": image_uri,
                    "runtime_profile": runtime_profile,
                    "training_job_name": training_status.job_name,
                    "training_job_arn": training_status.job_arn,
                    "status": training_status.status,
                    "artifacts": next_artifacts,
                },
                deep=True,
            )
            persisted = self._persist_state(request, next_state)

            self._record_cloud_job(
                request=request,
                run_id=run_id,
                provider="sagemaker",
                backend="sagemaker",
                provider_job_id=training_status.job_name,
                status=training_status.status,
                region=region,
                image_uri=image_uri,
                output_s3_uri=output_s3_uri,
                error_message=training_status.failure_reason,
                started_at=_utc_now_iso(),
            )

            return CloudAwsResponse(
                action="cloud.aws.train.submit",
                message="managed training submitted to SageMaker",
                state=persisted,
                result={
                    "backend": "sagemaker",
                    "run_id": run_id,
                    "training_job_name": training_status.job_name,
                    "training_job_arn": training_status.job_arn,
                    "status": training_status.status,
                    "config_s3_uri": config_uri,
                    "output_s3_uri": output_s3_uri,
                    "checkpoint_s3_uri": checkpoint_s3_uri,
                    "image_uri": image_uri,
                    "runtime_profile": runtime_profile,
                },
            )

        if backend == "batch":
            batch_job_queue = request.batch_job_queue or DEFAULT_BATCH_JOB_QUEUE
            batch_job_definition = request.batch_job_definition or DEFAULT_BATCH_JOB_DEFINITION
            if not batch_job_queue:
                raise CloudAwsError("missing required value: batch_job_queue")
            if not batch_job_definition:
                raise CloudAwsError("missing required value: batch_job_definition")

            job_name = self._job_name(run_id, prefix="neng-batch")
            parameters = {
                "run_id": run_id,
                "config_s3_uri": config_uri,
                "image_uri": image_uri,
                "output_s3_uri": output_s3_uri,
                "region": region,
                "bucket": bucket,
            }
            if checkpoint_s3_uri is not None:
                parameters["checkpoint_s3_uri"] = checkpoint_s3_uri

            batch_job_id = self._batch(region).submit_job(
                BatchJobSpec(
                    job_name=job_name,
                    job_queue=batch_job_queue,
                    job_definition=batch_job_definition,
                    parameters=parameters,
                    environment=request.env,
                    tags={"Project": "numereng", "RunId": run_id},
                )
            )

            next_artifacts = dict(state.artifacts)
            next_artifacts["config_s3_uri"] = config_uri
            next_artifacts["output_s3_uri"] = output_s3_uri
            if checkpoint_s3_uri is not None:
                next_artifacts["checkpoint_s3_uri"] = checkpoint_s3_uri

            next_state = state.model_copy(
                update={
                    "run_id": run_id,
                    "backend": "batch",
                    "region": region,
                    "bucket": bucket,
                    "image_uri": image_uri,
                    "runtime_profile": runtime_profile,
                    "batch_job_id": batch_job_id,
                    "status": "SUBMITTED",
                    "artifacts": next_artifacts,
                },
                deep=True,
            )
            persisted = self._persist_state(request, next_state)

            self._record_cloud_job(
                request=request,
                run_id=run_id,
                provider="batch",
                backend="batch",
                provider_job_id=batch_job_id,
                status="SUBMITTED",
                region=region,
                image_uri=image_uri,
                output_s3_uri=output_s3_uri,
                error_message=None,
                started_at=_utc_now_iso(),
            )

            return CloudAwsResponse(
                action="cloud.aws.train.submit",
                message="managed training submitted to AWS Batch",
                state=persisted,
                result={
                    "backend": "batch",
                    "run_id": run_id,
                    "batch_job_id": batch_job_id,
                    "status": "SUBMITTED",
                    "config_s3_uri": config_uri,
                    "output_s3_uri": output_s3_uri,
                    "checkpoint_s3_uri": checkpoint_s3_uri,
                    "image_uri": image_uri,
                    "runtime_profile": runtime_profile,
                },
            )

        raise CloudAwsError(f"unsupported backend: {backend}")

    def train_status(self, request: AwsTrainStatusRequest) -> CloudAwsResponse:
        state = self._load_state(request)
        backend_value = request.backend or state.backend
        backend = cast(Literal["sagemaker", "batch"], self._require(backend_value, "backend"))
        run_id = request.run_id or state.run_id
        run_id = self._require(run_id, "run_id")
        region = self._resolve_region(request.region, state)

        if backend == "sagemaker":
            training_job_name = request.training_job_name or state.training_job_name
            training_job_name = self._require(training_job_name, "training_job_name")
            training_status = self._sagemaker(region).describe_training(training_job_name)

            next_artifacts = dict(state.artifacts)
            if training_status.output_s3_uri is not None:
                next_artifacts["output_s3_uri"] = training_status.output_s3_uri

            next_metadata = dict(state.metadata)
            if training_status.failure_reason is not None:
                next_metadata["failure_reason"] = training_status.failure_reason

            next_state = state.model_copy(
                update={
                    "run_id": run_id,
                    "backend": "sagemaker",
                    "region": region,
                    "training_job_name": training_status.job_name,
                    "training_job_arn": training_status.job_arn,
                    "status": training_status.status,
                    "artifacts": next_artifacts,
                    "metadata": next_metadata,
                },
                deep=True,
            )
            persisted = self._persist_state(request, next_state)

            self._record_cloud_job(
                request=request,
                run_id=run_id,
                provider="sagemaker",
                backend="sagemaker",
                provider_job_id=training_status.job_name,
                status=training_status.status,
                region=region,
                image_uri=state.image_uri,
                output_s3_uri=training_status.output_s3_uri,
                error_message=training_status.failure_reason,
                finished_at=_finished_timestamp(training_status.status),
            )

            return CloudAwsResponse(
                action="cloud.aws.train.status",
                message="fetched SageMaker training status",
                state=persisted,
                result={
                    "backend": "sagemaker",
                    "run_id": run_id,
                    "training_job_name": training_status.job_name,
                    "training_job_arn": training_status.job_arn,
                    "status": training_status.status,
                    "secondary_status": training_status.secondary_status,
                    "failure_reason": training_status.failure_reason,
                    "output_s3_uri": training_status.output_s3_uri,
                },
            )

        if backend == "batch":
            batch_job_id = request.batch_job_id or state.batch_job_id
            batch_job_id = self._require(batch_job_id, "batch_job_id")
            batch_status = self._batch(region).describe_job(batch_job_id)

            next_metadata = dict(state.metadata)
            if batch_status.status_reason is not None:
                next_metadata["status_reason"] = batch_status.status_reason
            if batch_status.log_stream_name is not None:
                next_metadata["log_stream_name"] = batch_status.log_stream_name

            next_state = state.model_copy(
                update={
                    "run_id": run_id,
                    "backend": "batch",
                    "region": region,
                    "batch_job_id": batch_status.job_id,
                    "status": batch_status.status,
                    "metadata": next_metadata,
                },
                deep=True,
            )
            persisted = self._persist_state(request, next_state)

            self._record_cloud_job(
                request=request,
                run_id=run_id,
                provider="batch",
                backend="batch",
                provider_job_id=batch_status.job_id,
                status=batch_status.status,
                region=region,
                image_uri=state.image_uri,
                output_s3_uri=state.artifacts.get("output_s3_uri"),
                error_message=batch_status.status_reason,
                finished_at=_finished_timestamp(batch_status.status),
            )

            return CloudAwsResponse(
                action="cloud.aws.train.status",
                message="fetched AWS Batch job status",
                state=persisted,
                result={
                    "backend": "batch",
                    "run_id": run_id,
                    "batch_job_id": batch_status.job_id,
                    "status": batch_status.status,
                    "status_reason": batch_status.status_reason,
                    "log_stream_name": batch_status.log_stream_name,
                },
            )

        raise CloudAwsError(f"unsupported backend: {backend}")

    def train_logs(self, request: AwsTrainLogsRequest) -> CloudAwsResponse:
        state = self._load_state(request)
        backend_value = request.backend or state.backend
        backend = cast(Literal["sagemaker", "batch"], self._require(backend_value, "backend"))
        run_id = request.run_id or state.run_id
        run_id = self._require(run_id, "run_id")
        region = self._resolve_region(request.region, state)

        if backend == "sagemaker":
            training_job_name = request.training_job_name or state.training_job_name
            training_job_name = self._require(training_job_name, "training_job_name")

            training_status = self._sagemaker(region).describe_training(training_job_name)
            log_group = training_status.log_group or "/aws/sagemaker/TrainingJobs"
            stream_prefix = training_status.log_stream_prefix or f"{training_job_name}/"
            stream_names = self._logs(region).list_stream_names(
                log_group=log_group,
                stream_prefix=stream_prefix,
                limit=5,
            )
            if not stream_names:
                return CloudAwsResponse(
                    action="cloud.aws.train.logs",
                    message="no training log streams found yet",
                    state=state,
                    result={"backend": "sagemaker", "run_id": run_id, "lines": []},
                )

            stream_name = stream_names[0]
            events, next_token = self._logs(region).fetch_log_events(
                log_group=log_group,
                stream_name=stream_name,
                limit=request.lines,
                next_token=None,
                start_from_head=False,
            )
            lines = [event.message for event in events][-request.lines :]
            return CloudAwsResponse(
                action="cloud.aws.train.logs",
                message="fetched SageMaker log events",
                state=state,
                result={
                    "backend": "sagemaker",
                    "run_id": run_id,
                    "training_job_name": training_job_name,
                    "log_group": log_group,
                    "log_stream": stream_name,
                    "follow": request.follow,
                    "next_token": next_token,
                    "lines": lines,
                },
            )

        if backend == "batch":
            batch_job_id = request.batch_job_id or state.batch_job_id
            batch_job_id = self._require(batch_job_id, "batch_job_id")
            batch_status = self._batch(region).describe_job(batch_job_id)
            log_stream = batch_status.log_stream_name
            if log_stream is None:
                return CloudAwsResponse(
                    action="cloud.aws.train.logs",
                    message="batch job does not have a cloudwatch log stream yet",
                    state=state,
                    result={"backend": "batch", "run_id": run_id, "lines": []},
                )

            events, next_token = self._logs(region).fetch_log_events(
                log_group="/aws/batch/job",
                stream_name=log_stream,
                limit=request.lines,
                next_token=None,
                start_from_head=False,
            )
            lines = [event.message for event in events][-request.lines :]
            return CloudAwsResponse(
                action="cloud.aws.train.logs",
                message="fetched AWS Batch log events",
                state=state,
                result={
                    "backend": "batch",
                    "run_id": run_id,
                    "batch_job_id": batch_job_id,
                    "log_group": "/aws/batch/job",
                    "log_stream": log_stream,
                    "follow": request.follow,
                    "next_token": next_token,
                    "lines": lines,
                },
            )

        raise CloudAwsError(f"unsupported backend: {backend}")

    def train_cancel(self, request: AwsTrainCancelRequest) -> CloudAwsResponse:
        state = self._load_state(request)
        backend_value = request.backend or state.backend
        backend = cast(Literal["sagemaker", "batch"], self._require(backend_value, "backend"))
        run_id = request.run_id or state.run_id
        run_id = self._require(run_id, "run_id")
        region = self._resolve_region(request.region, state)

        if backend == "sagemaker":
            training_job_name = request.training_job_name or state.training_job_name
            training_job_name = self._require(training_job_name, "training_job_name")
            self._sagemaker(region).stop_training(training_job_name)
            next_state = state.model_copy(
                update={
                    "run_id": run_id,
                    "backend": "sagemaker",
                    "status": "Stopping",
                    "training_job_name": training_job_name,
                },
                deep=True,
            )
            persisted = self._persist_state(request, next_state)

            self._record_cloud_job(
                request=request,
                run_id=run_id,
                provider="sagemaker",
                backend="sagemaker",
                provider_job_id=training_job_name,
                status="Stopping",
                region=region,
                image_uri=state.image_uri,
                output_s3_uri=state.artifacts.get("output_s3_uri"),
                error_message=None,
            )

            return CloudAwsResponse(
                action="cloud.aws.train.cancel",
                message="requested SageMaker training stop",
                state=persisted,
                result={
                    "backend": "sagemaker",
                    "run_id": run_id,
                    "training_job_name": training_job_name,
                    "status": "Stopping",
                },
            )

        if backend == "batch":
            batch_job_id = request.batch_job_id or state.batch_job_id
            batch_job_id = self._require(batch_job_id, "batch_job_id")
            batch = self._batch(region)
            batch_status = batch.describe_job(batch_job_id)
            running_states = {"RUNNING", "STARTING"}
            queued_states = {"SUBMITTED", "PENDING", "RUNNABLE"}
            requested_action = "none"
            next_status = batch_status.status
            if batch_status.status in running_states:
                batch.terminate_job(batch_job_id, reason="numereng_user_cancel")
                requested_action = "terminate"
                next_status = "FAILED"
            elif batch_status.status in queued_states:
                batch.cancel_job(batch_job_id, reason="numereng_user_cancel")
                requested_action = "cancel"
                next_status = "FAILED"

            next_state = state.model_copy(
                update={
                    "run_id": run_id,
                    "backend": "batch",
                    "status": next_status,
                    "batch_job_id": batch_job_id,
                },
                deep=True,
            )
            persisted = self._persist_state(request, next_state)

            self._record_cloud_job(
                request=request,
                run_id=run_id,
                provider="batch",
                backend="batch",
                provider_job_id=batch_job_id,
                status=next_status,
                region=region,
                image_uri=state.image_uri,
                output_s3_uri=state.artifacts.get("output_s3_uri"),
                error_message=(
                    "cancelled"
                    if requested_action == "cancel"
                    else ("terminated" if requested_action == "terminate" else batch_status.status_reason)
                ),
            )

            return CloudAwsResponse(
                action="cloud.aws.train.cancel",
                message="requested AWS Batch job cancel/terminate",
                state=persisted,
                result={
                    "backend": "batch",
                    "run_id": run_id,
                    "batch_job_id": batch_job_id,
                    "status": next_status,
                    "requested_action": requested_action,
                },
            )

        raise CloudAwsError(f"unsupported backend: {backend}")

    def train_pull(self, request: AwsTrainPullRequest) -> CloudAwsResponse:
        state = self._load_state(request)
        run_id = request.run_id or state.run_id
        run_id = self._require(run_id, "run_id")
        region = self._resolve_region(request.region, state)
        bucket = self._resolve_bucket(request.bucket, state)

        output_s3_uri = request.output_s3_uri or state.artifacts.get("output_s3_uri")
        if output_s3_uri is None and state.training_job_name:
            status = self._sagemaker(region).describe_training(state.training_job_name)
            output_s3_uri = status.output_s3_uri

        if output_s3_uri is None:
            output_s3_uri = f"s3://{bucket}/runs/{run_id}/managed-output/"

        parsed = _parse_s3_uri(output_s3_uri, default_bucket=bucket)
        if parsed is None:
            raise CloudAwsError(f"invalid S3 URI: {output_s3_uri}")
        src_bucket, src_prefix = parsed

        if not src_prefix:
            prefix = ""
        elif src_prefix.endswith("/"):
            prefix = src_prefix
        else:
            prefix = f"{src_prefix}/"
        keys = self._s3(region).list_keys(src_bucket, prefix)

        store_root_path = Path(request.store_root).expanduser().resolve()
        output_root = self._resolve_pull_output_dir(
            output_dir=request.output_dir,
            run_id=run_id,
            store_root=store_root_path,
        )

        downloaded_files: list[str] = []
        skipped_unsafe_keys: list[str] = []
        for key in keys:
            relative = key[len(prefix) :] if key.startswith(prefix) else Path(key).name
            relative = relative.lstrip("/")
            if not relative:
                continue
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                skipped_unsafe_keys.append(key)
                continue
            local_path = (output_root / relative_path).resolve()
            try:
                local_path.relative_to(output_root)
            except ValueError:
                skipped_unsafe_keys.append(key)
                continue
            self._s3(region).download_file(src_bucket, key, local_path)
            downloaded_files.append(str(local_path))

        next_artifacts = dict(state.artifacts)
        next_artifacts["output_s3_uri"] = output_s3_uri

        next_state = state.model_copy(
            update={
                "run_id": run_id,
                "region": region,
                "bucket": bucket,
                "status": "artifacts_pulled",
                "artifacts": next_artifacts,
            },
            deep=True,
        )
        persisted = self._persist_state(request, next_state)

        provider = "batch" if state.backend == "batch" else "sagemaker"
        provider_job_id = state.batch_job_id if provider == "batch" else state.training_job_name
        if provider_job_id is not None:
            self._record_cloud_job(
                request=request,
                run_id=run_id,
                provider=provider,
                backend=state.backend or provider,
                provider_job_id=provider_job_id,
                status="ARTIFACTS_PULLED",
                region=region,
                image_uri=state.image_uri,
                output_s3_uri=output_s3_uri,
                error_message=None,
            )

        return CloudAwsResponse(
            action="cloud.aws.train.pull",
            message="pulled managed training outputs from S3",
            state=persisted,
            result={
                "run_id": run_id,
                "output_s3_uri": output_s3_uri,
                "output_dir": str(output_root),
                "downloaded_count": len(downloaded_files),
                "downloaded_files": downloaded_files,
                "skipped_unsafe_keys": skipped_unsafe_keys,
            },
        )

    def train_extract(self, request: AwsTrainExtractRequest) -> CloudAwsResponse:
        state = self._load_state(request)
        run_id = request.run_id or state.run_id
        run_id = self._require(run_id, "run_id")
        region = self._resolve_region(request.region, state)
        bucket = self._resolve_bucket(request.bucket, state)

        store_root_path = Path(request.store_root).expanduser().resolve()
        output_root = self._resolve_pull_output_dir(
            output_dir=request.output_dir,
            run_id=run_id,
            store_root=store_root_path,
            create=False,
        )

        downloaded_files = [path.resolve() for path in output_root.rglob("*") if path.is_file()]

        extracted_run_ids: list[str] = []
        archive_extract_warnings: list[str] = []
        for downloaded_path in downloaded_files:
            try:
                run_ids, warnings = _extract_run_tarball(
                    archive_path=downloaded_path,
                    store_root=store_root_path,
                )
            except CloudAwsError:
                raise
            except (tarfile.TarError, OSError) as exc:
                archive_extract_warnings.append(f"tar_extract_failed:{downloaded_path}:{exc}")
                continue
            extracted_run_ids.extend(run_ids)
            archive_extract_warnings.extend(warnings)

        extracted_run_ids = _dedupe_preserve_order(extracted_run_ids)
        index_targets = extracted_run_ids
        indexed_run_ids: list[str] = []
        index_warnings: list[str] = []
        for index_target in index_targets:
            try:
                index_run(store_root=request.store_root, run_id=index_target)
                indexed_run_ids.append(index_target)
            except StoreError as exc:
                index_warnings.append(str(exc))
        indexed = bool(indexed_run_ids)
        index_warning = "; ".join(index_warnings) if index_warnings else None

        next_metadata = dict(state.metadata)
        if index_warning is not None:
            next_metadata["index_warning"] = index_warning
        else:
            next_metadata.pop("index_warning", None)
        if archive_extract_warnings:
            next_metadata["extract_warning"] = "; ".join(archive_extract_warnings)
        else:
            next_metadata.pop("extract_warning", None)

        next_state = state.model_copy(
            update={
                "run_id": run_id,
                "region": region,
                "bucket": bucket,
                "status": "artifacts_extracted",
                "metadata": next_metadata,
            },
            deep=True,
        )
        persisted = self._persist_state(request, next_state)

        output_s3_uri = state.artifacts.get("output_s3_uri")
        provider = "batch" if state.backend == "batch" else "sagemaker"
        provider_job_id = state.batch_job_id if provider == "batch" else state.training_job_name
        if provider_job_id is not None:
            self._record_cloud_job(
                request=request,
                run_id=run_id,
                provider=provider,
                backend=state.backend or provider,
                provider_job_id=provider_job_id,
                status="ARTIFACTS_EXTRACTED",
                region=region,
                image_uri=state.image_uri,
                output_s3_uri=output_s3_uri,
                error_message=index_warning,
            )

        return CloudAwsResponse(
            action="cloud.aws.train.extract",
            message="extracted managed training outputs into local run store",
            state=persisted,
            result={
                "run_id": run_id,
                "output_dir": str(output_root),
                "downloaded_count": len(downloaded_files),
                "downloaded_files": [str(path) for path in downloaded_files],
                "extracted_run_ids": extracted_run_ids,
                "archive_extract_warnings": archive_extract_warnings,
                "indexed": indexed,
                "indexed_run_ids": indexed_run_ids,
                "index_warning": index_warning,
            },
        )

    def _resolve_pull_output_dir(
        self,
        *,
        output_dir: str | None,
        run_id: str,
        store_root: Path,
        create: bool = True,
    ) -> Path:
        resolved = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else (store_root / "cloud" / run_id / "pull").resolve()
        )
        try:
            ensure_allowed_store_target(
                target_path=resolved,
                store_root=store_root,
                allowed_prefixes=("cloud",),
                allow_store_root=False,
                error_code="aws_train_pull_output_dir_noncanonical",
            )
        except ValueError as exc:
            raise CloudAwsError(str(exc)) from exc
        if create:
            resolved.mkdir(parents=True, exist_ok=True)
        elif not resolved.exists():
            raise CloudAwsError(f"pull output directory does not exist: {resolved}")
        return resolved

    def _resolve_config_uri(
        self,
        *,
        request: AwsTrainSubmitRequest,
        run_id: str,
        region: str,
        bucket: str,
    ) -> str:
        if request.config_s3_uri is not None:
            return request.config_s3_uri

        config_path = request.config_path
        config_path = self._require(config_path, "config_path")
        if config_path.startswith("s3://"):
            return config_path

        path = Path(config_path).expanduser().resolve()
        if not path.is_file():
            raise CloudAwsError(f"config path does not exist: {path}")

        key = f"runs/{run_id}/config/{path.name}"
        return self._s3(region).upload_file(path, bucket, key)

    def _record_cloud_job(
        self,
        *,
        request: CloudAwsRequestBase,
        run_id: str,
        provider: str,
        backend: str,
        provider_job_id: str,
        status: str,
        region: str | None,
        image_uri: str | None,
        output_s3_uri: str | None,
        error_message: str | None,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> None:
        metadata_payload = {
            "provider": provider,
            "backend": backend,
            "status": status,
        }
        try:
            upsert_cloud_job(
                store_root=request.store_root,
                job=StoreCloudJobUpsert(
                    run_id=run_id,
                    provider=provider,
                    backend=backend,
                    provider_job_id=provider_job_id,
                    status=status,
                    region=region,
                    image_uri=image_uri,
                    output_s3_uri=output_s3_uri,
                    metadata_json=json.dumps(metadata_payload, sort_keys=True, ensure_ascii=True),
                    error_message=error_message,
                    started_at=started_at,
                    finished_at=finished_at,
                ),
            )
        except StoreError:
            return

    def _generated_run_id(self) -> str:
        return f"aws-{int(time.time())}"

    def _job_name(self, run_id: str, *, prefix: str) -> str:
        ts = int(time.time())
        safe_run_id = _slug_token(run_id)
        base = f"{prefix}-{safe_run_id}-{ts}"
        return base[:63]


def _parse_s3_uri(uri: str, *, default_bucket: str) -> tuple[str, str] | None:
    if not uri.startswith("s3://"):
        return None
    remainder = uri[5:]
    if not remainder:
        raise CloudAwsError("invalid S3 URI: missing bucket")
    if "/" not in remainder:
        return remainder, ""
    bucket, key = remainder.split("/", 1)
    if not bucket:
        bucket = default_bucket
    return bucket, key


def _slug_token(value: str) -> str:
    normalized = _SAFE_TOKEN.sub("-", value.strip())
    normalized = normalized.strip("-").lower()
    return normalized or "run"


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _finished_timestamp(status: str) -> str | None:
    terminal_states = {
        "Completed",
        "Failed",
        "Stopped",
        "SUCCEEDED",
        "FAILED",
    }
    if status in terminal_states:
        return _utc_now_iso()
    return None


def _extract_run_tarball(*, archive_path: Path, store_root: Path) -> tuple[list[str], list[str]]:
    if not _is_tarball_path(archive_path):
        return [], []

    extracted_run_ids: list[str] = []
    warnings: list[str] = []
    seen_run_ids: set[str] = set()
    root = store_root.resolve()
    cloud_tmp_root = (root / "cloud").resolve()
    cloud_tmp_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="aws-extract-", dir=str(cloud_tmp_root)) as tmp_dir_name:
        tmp_root = Path(tmp_dir_name).resolve()

        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive.getmembers():
                member_name = member.name
                if not member_name:
                    continue
                member_path = Path(member_name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise CloudAwsError(f"tar_member_unsafe:{member_name}")

                if member.islnk() or member.issym():
                    raise CloudAwsError(f"tar_member_link_unsafe:{member_name}")

                if not member_path.parts:
                    continue
                if member_path.parts[0] != "runs":
                    warnings.append(f"tar_member_outside_runs_skipped:{member_name}")
                    continue
                if len(member_path.parts) == 1:
                    continue

                run_id = member_path.parts[1]
                if not run_id or not _SAFE_RUN_ID.match(run_id):
                    raise CloudAwsError(f"tar_member_invalid_run_id:{member_name}")

                destination = (tmp_root / member_path).resolve()
                try:
                    destination.relative_to(tmp_root)
                except ValueError as exc:
                    raise CloudAwsError(f"tar_member_unsafe:{member_name}") from exc

                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    extracted_file = archive.extractfile(member)
                    if extracted_file is None:
                        raise CloudAwsError(f"tar_member_read_failed:{member_name}")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with extracted_file:
                        with destination.open("wb") as handle:
                            shutil.copyfileobj(extracted_file, handle)
                else:
                    warnings.append(f"tar_member_unsupported_skipped:{member_name}")
                    continue

                if run_id not in seen_run_ids:
                    seen_run_ids.add(run_id)
                    extracted_run_ids.append(run_id)

        promoted_run_ids: list[str] = []
        for run_id in extracted_run_ids:
            staged_run_dir = (tmp_root / "runs" / run_id).resolve()
            if not staged_run_dir.is_dir():
                continue

            staged_manifest = _load_run_manifest(run_dir=staged_run_dir, run_id=run_id)
            staged_run_hash = _require_run_hash(manifest=staged_manifest, run_id=run_id, label="staged")

            destination_run_dir = (root / "runs" / run_id).resolve()
            if destination_run_dir.exists():
                existing_manifest = _load_run_manifest(run_dir=destination_run_dir, run_id=run_id)
                existing_run_hash = _optional_run_hash(existing_manifest)
                if existing_run_hash is not None and existing_run_hash == staged_run_hash:
                    warnings.append(f"run_dir_exists_same_hash_skipped:{run_id}")
                    continue
                raise CloudAwsError(f"run_dir_conflict:{run_id}")

            destination_run_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staged_run_dir), str(destination_run_dir))
            promoted_run_ids.append(run_id)

        return promoted_run_ids, warnings


def _is_tarball_path(path: Path) -> bool:
    lower_name = path.name.lower()
    return lower_name.endswith(".tar.gz") or lower_name.endswith(".tgz")


def _load_run_manifest(*, run_dir: Path, run_id: str) -> dict[str, object]:
    manifest_path = run_dir / "run.json"
    if not manifest_path.is_file():
        raise CloudAwsError(f"run_manifest_missing:{run_id}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CloudAwsError(f"run_manifest_invalid_json:{run_id}") from exc
    if not isinstance(payload, dict):
        raise CloudAwsError(f"run_manifest_invalid_payload:{run_id}")
    manifest_run_id = payload.get("run_id")
    if isinstance(manifest_run_id, str) and manifest_run_id and manifest_run_id != run_id:
        raise CloudAwsError(f"run_manifest_run_id_mismatch:{run_id}:{manifest_run_id}")
    return payload


def _require_run_hash(*, manifest: dict[str, object], run_id: str, label: str) -> str:
    run_hash = _optional_run_hash(manifest)
    if run_hash is None:
        raise CloudAwsError(f"run_manifest_run_hash_missing:{label}:{run_id}")
    return run_hash


def _optional_run_hash(manifest: dict[str, object]) -> str | None:
    run_hash = manifest.get("run_hash")
    if not isinstance(run_hash, str):
        return None
    stripped = run_hash.strip()
    if not stripped:
        return None
    return stripped


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
