"""AWS-backed adapter implementations for EC2 cloud lifecycle."""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from numereng.features.cloud.aws.adapters import (
    BatchAdapter,
    BatchJobSpec,
    BatchJobStatus,
    CloudLogLine,
    CloudWatchLogsAdapter,
    DockerAdapter,
    Ec2Adapter,
    EcrAdapter,
    IamAdapter,
    InstanceStatus,
    LaunchInstanceSpec,
    S3Adapter,
    SageMakerAdapter,
    SageMakerTrainingSpec,
    SageMakerTrainingStatus,
    SsmAdapter,
    SsmCommandResult,
    WheelBuilder,
)

logger = logging.getLogger(__name__)


def _import_boto3() -> Any:
    try:
        import boto3
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime deps
        raise RuntimeError("boto3 is required for cloud EC2 operations") from exc
    return boto3


def _parse_tag_dict(tags: list[dict[str, str]] | None) -> dict[str, str]:
    if tags is None:
        return {}
    out: dict[str, str] = {}
    for entry in tags:
        key = entry.get("Key")
        value = entry.get("Value")
        if key is None or value is None:
            continue
        out[str(key)] = str(value)
    return out


class AwsEc2Adapter(Ec2Adapter):
    """EC2 adapter using boto3."""

    def __init__(self, region: str, *, ec2_client: Any | None = None) -> None:
        self._region = region
        self._ec2: Any | None = ec2_client

    @property
    def ec2(self) -> Any:
        if self._ec2 is None:
            boto3 = _import_boto3()
            self._ec2 = boto3.client("ec2", region_name=self._region)
        return self._ec2

    def launch_instance(self, spec: LaunchInstanceSpec) -> str:
        user_data = spec.user_data
        security_group_id: str | None = None
        if spec.security_group is not None:
            security_group_id = self.resolve_security_group_id(spec.security_group)
            if security_group_id is None:
                raise RuntimeError(f"unable to resolve security group: {spec.security_group}")

        params: dict[str, Any] = {
            "ImageId": spec.image_id,
            "InstanceType": spec.instance_type,
            "MinCount": 1,
            "MaxCount": 1,
            "UserData": user_data,
            "InstanceInitiatedShutdownBehavior": "terminate",
            "BlockDeviceMappings": [
                {
                    "DeviceName": "/dev/xvda",
                    "Ebs": {
                        "VolumeSize": spec.volume_size_gb,
                        "VolumeType": "gp3",
                        "DeleteOnTermination": True,
                    },
                }
            ],
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": f"numereng-{spec.run_id}"},
                        {"Key": "Project", "Value": "numereng"},
                        {"Key": "RunId", "Value": spec.run_id},
                        *[{"Key": key, "Value": value} for key, value in spec.tags.items()],
                    ],
                }
            ],
        }

        if spec.iam_role_name is not None:
            params["IamInstanceProfile"] = {"Name": spec.iam_role_name}

        if security_group_id is not None:
            params["SecurityGroupIds"] = [security_group_id]

        if spec.key_name is not None:
            params["KeyName"] = spec.key_name

        if spec.use_spot:
            params["InstanceMarketOptions"] = {
                "MarketType": "spot",
                "SpotOptions": {
                    "SpotInstanceType": "one-time",
                    "InstanceInterruptionBehavior": "terminate",
                },
            }

        response = self.ec2.run_instances(**params)
        instances = response.get("Instances") or []
        if not instances:
            raise RuntimeError("EC2 run_instances returned no instances")
        instance_id = instances[0].get("InstanceId")
        if not isinstance(instance_id, str) or not instance_id:
            raise RuntimeError("EC2 run_instances returned invalid InstanceId")
        return instance_id

    def wait_for_instance(self, instance_id: str, target_state: str, timeout_seconds: int) -> bool:
        start = time.time()
        while time.time() - start < timeout_seconds:
            status = self.get_instance_status(instance_id)
            if status.state == target_state:
                return True
            if status.state in {"terminated", "shutting-down"}:
                return False
            time.sleep(5)
        return False

    def terminate_instance(self, instance_id: str) -> None:
        self.ec2.terminate_instances(InstanceIds=[instance_id])

    def get_instance_status(self, instance_id: str) -> InstanceStatus:
        response = self.ec2.describe_instances(InstanceIds=[instance_id])
        reservations = response.get("Reservations") or []
        if not reservations:
            raise RuntimeError(f"instance not found: {instance_id}")
        instances = reservations[0].get("Instances") or []
        if not instances:
            raise RuntimeError(f"instance not found: {instance_id}")
        instance = instances[0]
        tags = _parse_tag_dict(instance.get("Tags"))
        launch_time = instance.get("LaunchTime")
        launch_time_str = str(launch_time) if launch_time is not None else None
        return InstanceStatus(
            instance_id=str(instance.get("InstanceId") or instance_id),
            state=str((instance.get("State") or {}).get("Name") or "unknown"),
            instance_type=str(instance.get("InstanceType") or "unknown"),
            run_id=tags.get("RunId"),
            public_ip=(str(instance.get("PublicIpAddress")) if instance.get("PublicIpAddress") else None),
            private_ip=(str(instance.get("PrivateIpAddress")) if instance.get("PrivateIpAddress") else None),
            launch_time=launch_time_str,
        )

    def list_training_instances(self) -> list[InstanceStatus]:
        response = self.ec2.describe_instances(
            Filters=[
                {
                    "Name": "instance-state-name",
                    "Values": ["pending", "running", "stopping", "stopped"],
                },
                {"Name": "tag:Project", "Values": ["numereng"]},
            ]
        )
        out: list[InstanceStatus] = []
        reservations = response.get("Reservations") or []
        for reservation in reservations:
            instances = reservation.get("Instances") or []
            for instance in instances:
                tags = _parse_tag_dict(instance.get("Tags"))
                launch_time = instance.get("LaunchTime")
                launch_time_str = str(launch_time) if launch_time is not None else None
                out.append(
                    InstanceStatus(
                        instance_id=str(instance.get("InstanceId") or ""),
                        state=str((instance.get("State") or {}).get("Name") or "unknown"),
                        instance_type=str(instance.get("InstanceType") or "unknown"),
                        run_id=tags.get("RunId"),
                        public_ip=(
                            str(instance.get("PublicIpAddress"))
                            if instance.get("PublicIpAddress")
                            else None
                        ),
                        private_ip=(
                            str(instance.get("PrivateIpAddress"))
                            if instance.get("PrivateIpAddress")
                            else None
                        ),
                        launch_time=launch_time_str,
                    )
                )
        return out

    def get_spot_price(self, instance_type: str) -> float | None:
        response = self.ec2.describe_spot_price_history(
            InstanceTypes=[instance_type],
            ProductDescriptions=["Linux/UNIX"],
            MaxResults=1,
        )
        history = response.get("SpotPriceHistory") or []
        if not history:
            return None
        raw_price = history[0].get("SpotPrice")
        if raw_price is None:
            return None
        try:
            return float(raw_price)
        except (TypeError, ValueError):
            return None

    def resolve_security_group_id(self, security_group: str) -> str | None:
        if security_group.startswith("sg-"):
            return security_group
        response = self.ec2.describe_security_groups(
            Filters=[
                {
                    "Name": "group-name",
                    "Values": [security_group],
                }
            ]
        )
        groups = response.get("SecurityGroups") or []
        if not groups:
            return None
        group_id = groups[0].get("GroupId")
        if group_id is None:
            return None
        return str(group_id)


class AwsS3Adapter(S3Adapter):
    """S3 adapter using boto3."""

    def __init__(self, region: str, *, s3_client: Any | None = None) -> None:
        self._region = region
        self._s3: Any | None = s3_client

    @property
    def s3(self) -> Any:
        if self._s3 is None:
            boto3 = _import_boto3()
            self._s3 = boto3.client("s3", region_name=self._region)
        return self._s3

    def ensure_bucket_exists(self, bucket: str, region: str) -> None:
        try:
            self.s3.head_bucket(Bucket=bucket)
            return
        except Exception:
            pass

        if region == "us-east-1":
            self.s3.create_bucket(Bucket=bucket)
            return

        self.s3.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": region},
        )

    def upload_file(self, local_path: Path, bucket: str, key: str) -> str:
        self.s3.upload_file(str(local_path), bucket, key)
        return f"s3://{bucket}/{key}"

    def download_file(self, bucket: str, key: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(bucket, key, str(local_path))
        return local_path

    def list_keys(self, bucket: str, prefix: str) -> list[str]:
        keys: list[str] = []
        continuation: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
            if continuation is not None:
                kwargs["ContinuationToken"] = continuation
            response = self.s3.list_objects_v2(**kwargs)
            for obj in response.get("Contents") or []:
                key = obj.get("Key")
                if isinstance(key, str):
                    keys.append(key)
            if not response.get("IsTruncated"):
                break
            continuation = response.get("NextContinuationToken")
            if not isinstance(continuation, str) or not continuation:
                break
        return keys

    def delete_key(self, bucket: str, key: str) -> None:
        self.s3.delete_object(Bucket=bucket, Key=key)

    def delete_prefix(self, bucket: str, prefix: str) -> int:
        keys = self.list_keys(bucket=bucket, prefix=prefix)
        if not keys:
            return 0

        deleted = 0
        for start in range(0, len(keys), 1000):
            chunk = keys[start : start + 1000]
            payload = {"Objects": [{"Key": key} for key in chunk], "Quiet": True}
            self.s3.delete_objects(Bucket=bucket, Delete=payload)
            deleted += len(chunk)
        return deleted

    def copy_object(self, src_bucket: str, src_key: str, dst_bucket: str, dst_key: str) -> str:
        source = {"Bucket": src_bucket, "Key": src_key}
        self.s3.copy_object(CopySource=source, Bucket=dst_bucket, Key=dst_key)
        return f"s3://{dst_bucket}/{dst_key}"


class AwsSsmAdapter(SsmAdapter):
    """SSM adapter using boto3."""

    def __init__(self, region: str, *, ssm_client: Any | None = None) -> None:
        self._region = region
        self._ssm: Any | None = ssm_client

    @property
    def ssm(self) -> Any:
        if self._ssm is None:
            boto3 = _import_boto3()
            self._ssm = boto3.client("ssm", region_name=self._region)
        return self._ssm

    def wait_for_ssm(self, instance_id: str, timeout_seconds: int) -> None:
        start = time.time()
        while time.time() - start < timeout_seconds:
            response = self.ssm.describe_instance_information(
                Filters=[{"Key": "InstanceIds", "Values": [instance_id]}]
            )
            instances = response.get("InstanceInformationList") or []
            if instances and instances[0].get("PingStatus") == "Online":
                return
            time.sleep(10)
        raise RuntimeError(f"instance {instance_id} did not become SSM-online within timeout")

    def run_command(self, instance_id: str, command: str, timeout_seconds: int) -> SsmCommandResult:
        response = self.ssm.send_command(
            InstanceIds=[instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [command]},
            TimeoutSeconds=min(timeout_seconds, 3600),
        )
        command_id = response["Command"]["CommandId"]

        start = time.time()
        invocation: dict[str, Any] | None = None
        while time.time() - start < timeout_seconds:
            try:
                invocation = self.ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
            except self.ssm.exceptions.InvocationDoesNotExist:
                time.sleep(2)
                continue

            status = invocation.get("Status")
            if status in {"Success", "Failed", "Cancelled", "TimedOut"}:
                break
            time.sleep(2)

        if invocation is None:
            raise RuntimeError("SSM command invocation unavailable")

        status = str(invocation.get("Status") or "Unknown")
        raw_response_code = invocation.get("ResponseCode")
        exit_code = int(raw_response_code) if raw_response_code is not None else -1
        stdout = str(invocation.get("StandardOutputContent") or "")
        stderr = str(invocation.get("StandardErrorContent") or "")

        if status != "Success":
            raise RuntimeError(
                f"ssm_command_failed status={status} exit_code={exit_code} stderr={stderr[:500]}"
            )

        return SsmCommandResult(exit_code=exit_code, stdout=stdout, stderr=stderr)


class AwsIamAdapter(IamAdapter):
    """IAM/network bootstrap adapter using boto3."""

    _trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    def __init__(
        self,
        *,
        iam_client: Any | None = None,
        ec2_client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._iam: Any | None = iam_client
        self._ec2_client_factory = ec2_client_factory

    @property
    def iam(self) -> Any:
        if self._iam is None:
            boto3 = _import_boto3()
            self._iam = boto3.client("iam")
        return self._iam

    def ensure_training_role(self, role_name: str, bucket: str) -> str:
        try:
            role = self.iam.get_role(RoleName=role_name)["Role"]
            role_arn = str(role["Arn"])
        except Exception:
            role = self.iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(self._trust_policy),
                Description="IAM role for numereng EC2 training",
                Tags=[{"Key": "Project", "Value": "numereng"}],
            )["Role"]
            role_arn = str(role["Arn"])

        policy_doc = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "S3Access",
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:GetBucketLocation"],
                    "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"],
                }
            ],
        }

        self.iam.put_role_policy(
            RoleName=role_name,
            PolicyName=f"{role_name}-policy",
            PolicyDocument=json.dumps(policy_doc),
        )

        self.iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
        )

        return role_arn

    def ensure_instance_profile(self, role_name: str) -> str:
        try:
            profile = self.iam.get_instance_profile(InstanceProfileName=role_name)["InstanceProfile"]
        except Exception:
            profile = self.iam.create_instance_profile(
                InstanceProfileName=role_name,
                Tags=[{"Key": "Project", "Value": "numereng"}],
            )["InstanceProfile"]

        try:
            self.iam.add_role_to_instance_profile(InstanceProfileName=role_name, RoleName=role_name)
        except Exception:
            pass

        return str(profile["Arn"])

    def ensure_security_group(self, region: str, group_name: str) -> str:
        if self._ec2_client_factory is not None:
            ec2 = self._ec2_client_factory(region)
        else:
            boto3 = _import_boto3()
            ec2 = boto3.client("ec2", region_name=region)

        response = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [group_name]}])
        groups = response.get("SecurityGroups") or []
        if groups:
            group_id = groups[0].get("GroupId")
            if isinstance(group_id, str) and group_id:
                return group_id

        vpcs = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}]).get("Vpcs") or []
        if not vpcs:
            raise RuntimeError("no default VPC found; configure security group manually")
        vpc_id = vpcs[0].get("VpcId")
        if not isinstance(vpc_id, str) or not vpc_id:
            raise RuntimeError("unable to resolve default VPC id")

        created = ec2.create_security_group(
            GroupName=group_name,
            Description="numereng training security group",
            VpcId=vpc_id,
            TagSpecifications=[
                {
                    "ResourceType": "security-group",
                    "Tags": [
                        {"Key": "Name", "Value": group_name},
                        {"Key": "Project", "Value": "numereng"},
                    ],
                }
            ],
        )
        group_id = created.get("GroupId")
        if not isinstance(group_id, str) or not group_id:
            raise RuntimeError("failed to create security group")

        try:
            ec2.authorize_security_group_ingress(
                GroupId=group_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 22,
                        "ToPort": 22,
                        "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH access"}],
                    }
                ],
            )
        except Exception:
            pass

        return group_id


class UvWheelBuilder(WheelBuilder):
    """Build numereng wheel + locked requirements via uv."""

    def __init__(self) -> None:
        self._repo_root = self._find_repo_root()

    def _find_repo_root(self) -> Path:
        start = Path(__file__).resolve()
        for candidate in [start, *start.parents]:
            if (candidate / "uv.lock").exists() and (candidate / "pyproject.toml").exists():
                return candidate
        raise RuntimeError("unable to locate repo root for wheel build")

    def build_assets(self, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        dist_dir = output_dir / "dist"
        dist_dir.mkdir(parents=True, exist_ok=True)
        requirements_path = output_dir / "requirements.txt"

        subprocess.run(
            [
                "uv",
                "build",
                "--package",
                "numereng",
                "--wheel",
                "-o",
                str(dist_dir),
                "--no-build-logs",
            ],
            cwd=self._repo_root,
            check=True,
        )
        subprocess.run(
            [
                "uv",
                "export",
                "--format",
                "requirements.txt",
                "--package",
                "numereng",
                "--locked",
                "--no-dev",
                "--no-emit-workspace",
                "--no-emit-project",
                "--no-hashes",
                "--output-file",
                str(requirements_path),
            ],
            cwd=self._repo_root,
            check=True,
        )

        artifacts: list[Path] = [requirements_path]
        artifacts.extend(sorted(dist_dir.glob("*.whl")))
        if not any(path.name.endswith(".whl") for path in artifacts):
            raise RuntimeError("wheel build produced no .whl artifacts")
        return artifacts


class AwsEcrAdapter(EcrAdapter):
    """ECR adapter using boto3."""

    def __init__(
        self,
        region: str,
        *,
        ecr_client: Any | None = None,
        sts_client: Any | None = None,
    ) -> None:
        self._region = region
        self._ecr: Any | None = ecr_client
        self._sts: Any | None = sts_client

    @property
    def ecr(self) -> Any:
        if self._ecr is None:
            boto3 = _import_boto3()
            self._ecr = boto3.client("ecr", region_name=self._region)
        return self._ecr

    @property
    def sts(self) -> Any:
        if self._sts is None:
            boto3 = _import_boto3()
            self._sts = boto3.client("sts", region_name=self._region)
        return self._sts

    def ensure_repository(self, repository_name: str) -> str:
        try:
            response = self.ecr.describe_repositories(repositoryNames=[repository_name])
            repositories = response.get("repositories") or []
        except Exception:
            response = self.ecr.create_repository(
                repositoryName=repository_name,
                imageTagMutability="MUTABLE",
                imageScanningConfiguration={"scanOnPush": False},
                tags=[{"Key": "Project", "Value": "numereng"}],
            )
            repositories = [response.get("repository")]

        if not repositories:
            raise RuntimeError(f"unable to ensure ECR repository: {repository_name}")

        repository_uri = repositories[0].get("repositoryUri")
        if not isinstance(repository_uri, str) or not repository_uri:
            raise RuntimeError(f"ECR repository URI unavailable: {repository_name}")
        return repository_uri

    def get_account_id(self) -> str:
        payload = self.sts.get_caller_identity()
        account = payload.get("Account")
        if not isinstance(account, str) or not account:
            raise RuntimeError("unable to resolve caller account id")
        return account

    def get_login_password(self) -> str:
        response = self.ecr.get_authorization_token()
        auth_data = response.get("authorizationData") or []
        if not auth_data:
            raise RuntimeError("missing ECR authorization data")
        encoded = auth_data[0].get("authorizationToken")
        if not isinstance(encoded, str) or not encoded:
            raise RuntimeError("missing ECR authorization token")
        decoded = base64.b64decode(encoded).decode("utf-8")
        if ":" not in decoded:
            raise RuntimeError("invalid ECR authorization token payload")
        _username, password = decoded.split(":", 1)
        if not password:
            raise RuntimeError("invalid ECR authorization token password")
        return password

    def image_uri(self, repository_name: str, image_tag: str) -> str:
        account_id = self.get_account_id()
        return f"{account_id}.dkr.ecr.{self._region}.amazonaws.com/{repository_name}:{image_tag}"

    def get_image_digest(self, repository_name: str, image_tag: str) -> str | None:
        try:
            response = self.ecr.describe_images(
                repositoryName=repository_name,
                imageIds=[{"imageTag": image_tag}],
            )
        except Exception:
            return None

        details = response.get("imageDetails") or []
        if not details:
            return None
        digest = details[0].get("imageDigest")
        if not isinstance(digest, str) or not digest:
            return None
        return digest


class AwsSageMakerAdapter(SageMakerAdapter):
    """SageMaker adapter using boto3."""

    def __init__(self, region: str, *, sagemaker_client: Any | None = None) -> None:
        self._region = region
        self._sagemaker: Any | None = sagemaker_client

    @property
    def sagemaker(self) -> Any:
        if self._sagemaker is None:
            boto3 = _import_boto3()
            self._sagemaker = boto3.client("sagemaker", region_name=self._region)
        return self._sagemaker

    def start_training(self, spec: SageMakerTrainingSpec) -> SageMakerTrainingStatus:
        stopping: dict[str, int] = {"MaxRuntimeInSeconds": spec.max_runtime_seconds}
        if spec.max_wait_seconds is not None:
            stopping["MaxWaitTimeInSeconds"] = spec.max_wait_seconds

        request: dict[str, Any] = {
            "TrainingJobName": spec.job_name,
            "AlgorithmSpecification": {
                "TrainingImage": spec.image_uri,
                "TrainingInputMode": "File",
            },
            "RoleArn": spec.role_arn,
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": spec.input_config_uri,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                }
            ],
            "OutputDataConfig": {"S3OutputPath": spec.output_s3_uri},
            "ResourceConfig": {
                "InstanceType": spec.instance_type,
                "InstanceCount": spec.instance_count,
                "VolumeSizeInGB": spec.volume_size_gb,
            },
            "StoppingCondition": stopping,
            "EnableManagedSpotTraining": spec.use_spot,
            "Environment": spec.environment,
            "Tags": [{"Key": key, "Value": value} for key, value in spec.tags.items()],
        }
        if spec.checkpoint_s3_uri is not None:
            request["CheckpointConfig"] = {
                "S3Uri": spec.checkpoint_s3_uri,
                "LocalPath": "/opt/ml/checkpoints",
            }

        self.sagemaker.create_training_job(**request)
        return self.describe_training(spec.job_name)

    def describe_training(self, job_name: str) -> SageMakerTrainingStatus:
        response = self.sagemaker.describe_training_job(TrainingJobName=job_name)
        arn = response.get("TrainingJobArn")
        status = response.get("TrainingJobStatus")
        secondary = response.get("SecondaryStatus")
        failure_reason = response.get("FailureReason")
        output_config = response.get("OutputDataConfig") or {}
        output_uri = output_config.get("S3OutputPath")
        if not isinstance(output_uri, str):
            output_uri = None

        return SageMakerTrainingStatus(
            job_name=job_name,
            job_arn=(str(arn) if isinstance(arn, str) else None),
            status=str(status or "Unknown"),
            secondary_status=(str(secondary) if isinstance(secondary, str) else None),
            failure_reason=(str(failure_reason) if isinstance(failure_reason, str) else None),
            output_s3_uri=output_uri,
            log_group="/aws/sagemaker/TrainingJobs",
            log_stream_prefix=f"{job_name}/",
        )

    def stop_training(self, job_name: str) -> None:
        self.sagemaker.stop_training_job(TrainingJobName=job_name)


class AwsBatchAdapter(BatchAdapter):
    """AWS Batch adapter using boto3."""

    def __init__(self, region: str, *, batch_client: Any | None = None) -> None:
        self._region = region
        self._batch: Any | None = batch_client

    @property
    def batch(self) -> Any:
        if self._batch is None:
            boto3 = _import_boto3()
            self._batch = boto3.client("batch", region_name=self._region)
        return self._batch

    def submit_job(self, spec: BatchJobSpec) -> str:
        request: dict[str, Any] = {
            "jobName": spec.job_name,
            "jobQueue": spec.job_queue,
            "jobDefinition": spec.job_definition,
            "parameters": spec.parameters,
            "tags": spec.tags,
        }
        if spec.environment:
            request["containerOverrides"] = {
                "environment": [{"name": key, "value": value} for key, value in spec.environment.items()]
            }

        response = self.batch.submit_job(**request)
        job_id = response.get("jobId")
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError("AWS Batch submit_job returned no job id")
        return job_id

    def describe_job(self, job_id: str) -> BatchJobStatus:
        response = self.batch.describe_jobs(jobs=[job_id])
        jobs = response.get("jobs") or []
        if not jobs:
            raise RuntimeError(f"AWS Batch job not found: {job_id}")
        job = jobs[0]
        container = job.get("container") or {}
        status_reason = job.get("statusReason")
        log_stream_name = container.get("logStreamName")
        return BatchJobStatus(
            job_id=job_id,
            status=str(job.get("status") or "UNKNOWN"),
            status_reason=(str(status_reason) if isinstance(status_reason, str) else None),
            log_stream_name=(str(log_stream_name) if isinstance(log_stream_name, str) else None),
        )

    def cancel_job(self, job_id: str, *, reason: str) -> None:
        self.batch.cancel_job(jobId=job_id, reason=reason)

    def terminate_job(self, job_id: str, *, reason: str) -> None:
        self.batch.terminate_job(jobId=job_id, reason=reason)


class AwsCloudWatchLogsAdapter(CloudWatchLogsAdapter):
    """CloudWatch Logs adapter using boto3."""

    def __init__(self, region: str, *, logs_client: Any | None = None) -> None:
        self._region = region
        self._logs: Any | None = logs_client

    @property
    def logs(self) -> Any:
        if self._logs is None:
            boto3 = _import_boto3()
            self._logs = boto3.client("logs", region_name=self._region)
        return self._logs

    def list_stream_names(
        self,
        *,
        log_group: str,
        stream_prefix: str,
        limit: int,
    ) -> list[str]:
        request: dict[str, Any] = {
            "logGroupName": log_group,
            "limit": max(1, limit),
        }
        if stream_prefix:
            # CloudWatch does not allow orderBy=LastEventTime when using a name prefix.
            request["logStreamNamePrefix"] = stream_prefix
        else:
            request["orderBy"] = "LastEventTime"
            request["descending"] = True

        response = self.logs.describe_log_streams(**request)
        streams = list(response.get("logStreams") or [])
        if stream_prefix:
            streams.sort(key=_last_event_timestamp, reverse=True)

        names: list[str] = []
        for stream in streams:
            stream_name = stream.get("logStreamName")
            if isinstance(stream_name, str) and stream_name:
                names.append(stream_name)
            if len(names) >= limit:
                break
        return names

    def fetch_log_events(
        self,
        *,
        log_group: str,
        stream_name: str,
        limit: int,
        next_token: str | None = None,
        start_from_head: bool = False,
    ) -> tuple[list[CloudLogLine], str | None]:
        request: dict[str, Any] = {
            "logGroupName": log_group,
            "logStreamName": stream_name,
            "limit": limit,
            "startFromHead": start_from_head,
        }
        if next_token is not None:
            request["nextToken"] = next_token

        response = self.logs.get_log_events(**request)
        events: list[CloudLogLine] = []
        for event in response.get("events") or []:
            timestamp = event.get("timestamp")
            message = event.get("message")
            if not isinstance(message, str):
                continue
            events.append(CloudLogLine(timestamp_ms=int(timestamp or 0), message=message))
        token = response.get("nextForwardToken")
        return events, (str(token) if isinstance(token, str) else None)


class SubprocessDockerAdapter(DockerAdapter):
    """Docker adapter backed by local docker CLI subprocess calls."""

    def build_image(
        self,
        *,
        context_dir: Path,
        tag: str,
        dockerfile: Path | None = None,
        build_args: dict[str, str] | None = None,
        platform: str | None = None,
    ) -> None:
        command: list[str] = ["docker", "build", "-t", tag]
        if dockerfile is not None:
            command.extend(["-f", str(dockerfile)])
        if platform is not None:
            command.extend(["--platform", platform])
        if build_args is not None:
            for key, value in sorted(build_args.items()):
                command.extend(["--build-arg", f"{key}={value}"])
        command.append(str(context_dir))
        subprocess.run(command, check=True)

    def tag_image(self, *, source_tag: str, target_tag: str) -> None:
        subprocess.run(["docker", "tag", source_tag, target_tag], check=True)

    def push_image(self, *, tag: str) -> None:
        subprocess.run(["docker", "push", tag], check=True)

    def login(self, *, registry: str, username: str, password: str) -> None:
        process = subprocess.run(
            ["docker", "login", "--username", username, "--password-stdin", registry],
            input=password,
            text=True,
            capture_output=True,
            check=False,
        )
        if process.returncode != 0:
            stderr = process.stderr.strip()
            raise RuntimeError(f"docker login failed: {stderr}")


def build_assets_with_tempdir(builder: WheelBuilder) -> tuple[list[Path], tempfile.TemporaryDirectory[str]]:
    """Build assets into a temporary directory and return both paths and handle.

    Caller must keep the returned tempdir object alive until artifacts are consumed.
    """

    tmpdir = tempfile.TemporaryDirectory()
    paths = builder.build_assets(Path(tmpdir.name))
    return paths, tmpdir


def _last_event_timestamp(stream: dict[str, Any]) -> int:
    raw = stream.get("lastEventTimestamp")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    return 0
