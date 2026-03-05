from __future__ import annotations

import base64
from datetime import UTC, datetime

import pytest
from botocore.stub import Stubber

from numereng.features.cloud.aws.adapters import BatchJobSpec, SageMakerTrainingSpec, SageMakerTrainingStatus
from numereng.features.cloud.aws.aws_adapters import (
    AwsBatchAdapter,
    AwsCloudWatchLogsAdapter,
    AwsEcrAdapter,
    AwsSageMakerAdapter,
)


def test_ecr_ensure_repository_existing(ecr_client: object, ecr_stubber: Stubber, sts_client: object) -> None:
    adapter = AwsEcrAdapter(region="us-east-2", ecr_client=ecr_client, sts_client=sts_client)
    ecr_stubber.add_response(
        "describe_repositories",
        {
            "repositories": [
                {
                    "repositoryName": "numereng-training",
                    "repositoryUri": "123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training",
                }
            ]
        },
        {"repositoryNames": ["numereng-training"]},
    )

    uri = adapter.ensure_repository("numereng-training")
    assert uri == "123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training"


def test_ecr_ensure_repository_create_on_miss(ecr_client: object, ecr_stubber: Stubber, sts_client: object) -> None:
    adapter = AwsEcrAdapter(region="us-east-2", ecr_client=ecr_client, sts_client=sts_client)
    ecr_stubber.add_client_error(
        "describe_repositories",
        service_error_code="RepositoryNotFoundException",
        http_status_code=404,
    )
    ecr_stubber.add_response(
        "create_repository",
        {
            "repository": {
                "repositoryName": "numereng-training",
                "repositoryUri": "123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training",
            }
        },
        {
            "repositoryName": "numereng-training",
            "imageTagMutability": "MUTABLE",
            "imageScanningConfiguration": {"scanOnPush": False},
            "tags": [{"Key": "Project", "Value": "numereng"}],
        },
    )

    uri = adapter.ensure_repository("numereng-training")
    assert uri == "123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training"


def test_ecr_get_login_password_decodes_token(
    ecr_client: object,
    ecr_stubber: Stubber,
    sts_client: object,
) -> None:
    adapter = AwsEcrAdapter(region="us-east-2", ecr_client=ecr_client, sts_client=sts_client)
    token = base64.b64encode(b"AWS:secret-password").decode("ascii")
    ecr_stubber.add_response(
        "get_authorization_token",
        {"authorizationData": [{"authorizationToken": token}]},
    )

    assert adapter.get_login_password() == "secret-password"


def test_ecr_get_image_digest_returns_none_on_error(
    ecr_client: object,
    ecr_stubber: Stubber,
    sts_client: object,
) -> None:
    adapter = AwsEcrAdapter(region="us-east-2", ecr_client=ecr_client, sts_client=sts_client)
    ecr_stubber.add_client_error(
        "describe_images",
        service_error_code="ImageNotFoundException",
        http_status_code=404,
    )

    digest = adapter.get_image_digest("numereng-training", "v1")
    assert digest is None


def test_sagemaker_start_training_builds_expected_request(
    monkeypatch: pytest.MonkeyPatch,
    sagemaker_client: object,
    sagemaker_stubber: Stubber,
) -> None:
    adapter = AwsSageMakerAdapter(region="us-east-2", sagemaker_client=sagemaker_client)
    spec = SageMakerTrainingSpec(
        job_name="neng-sm-run-1-123",
        image_uri="123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training:v1",
        role_arn="arn:aws:iam::123456789012:role/numereng-sagemaker",
        input_config_uri="s3://numereng-artifacts/runs/run-1/config/train.json",
        output_s3_uri="s3://numereng-artifacts/runs/run-1/managed-output/",
        checkpoint_s3_uri="s3://numereng-artifacts/runs/run-1/checkpoints/",
        instance_type="ml.m5.2xlarge",
        instance_count=1,
        volume_size_gb=100,
        max_runtime_seconds=14_400,
        max_wait_seconds=43_200,
        use_spot=True,
        environment={"NUMERENG_RUN_ID": "run-1"},
        tags={"Project": "numereng", "RunId": "run-1"},
    )
    expected_request = {
        "TrainingJobName": "neng-sm-run-1-123",
        "AlgorithmSpecification": {
            "TrainingImage": "123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training:v1",
            "TrainingInputMode": "File",
        },
        "RoleArn": "arn:aws:iam::123456789012:role/numereng-sagemaker",
        "InputDataConfig": [
            {
                "ChannelName": "training",
                "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": "s3://numereng-artifacts/runs/run-1/config/train.json",
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
            }
        ],
        "OutputDataConfig": {"S3OutputPath": "s3://numereng-artifacts/runs/run-1/managed-output/"},
        "ResourceConfig": {
            "InstanceType": "ml.m5.2xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 100,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 14_400,
            "MaxWaitTimeInSeconds": 43_200,
        },
        "EnableManagedSpotTraining": True,
        "Environment": {"NUMERENG_RUN_ID": "run-1"},
        "Tags": [
            {"Key": "Project", "Value": "numereng"},
            {"Key": "RunId", "Value": "run-1"},
        ],
        "CheckpointConfig": {
            "S3Uri": "s3://numereng-artifacts/runs/run-1/checkpoints/",
            "LocalPath": "/opt/ml/checkpoints",
        },
    }
    sagemaker_stubber.add_response(
        "create_training_job",
        {"TrainingJobArn": "arn:aws:sagemaker:us-east-2:123456789012:training-job/neng-sm-run-1-123"},
        expected_request,
    )
    monkeypatch.setattr(
        adapter,
        "describe_training",
        lambda _job_name: SageMakerTrainingStatus(
            job_name="neng-sm-run-1-123",
            job_arn="arn:aws:sagemaker:us-east-2:123456789012:training-job/neng-sm-run-1-123",
            status="InProgress",
        ),
    )

    status = adapter.start_training(spec)
    assert status.status == "InProgress"


def test_sagemaker_describe_training_maps_response(sagemaker_client: object, sagemaker_stubber: Stubber) -> None:
    adapter = AwsSageMakerAdapter(region="us-east-2", sagemaker_client=sagemaker_client)
    sagemaker_stubber.add_response(
        "describe_training_job",
        {
            "TrainingJobName": "job-1",
            "TrainingJobArn": "arn:aws:sagemaker:us-east-2:123456789012:training-job/job-1",
            "ModelArtifacts": {"S3ModelArtifacts": "s3://bucket/path/model.tar.gz"},
            "TrainingJobStatus": "Completed",
            "SecondaryStatus": "Completed",
            "StoppingCondition": {"MaxRuntimeInSeconds": 60},
            "CreationTime": datetime.now(tz=UTC),
            "OutputDataConfig": {"S3OutputPath": "s3://bucket/path/output/"},
            "FailureReason": "none",
        },
        {"TrainingJobName": "job-1"},
    )

    status = adapter.describe_training("job-1")
    assert status.job_name == "job-1"
    assert status.status == "Completed"
    assert status.output_s3_uri == "s3://bucket/path/output/"


def test_sagemaker_stop_training_calls_api(sagemaker_client: object, sagemaker_stubber: Stubber) -> None:
    adapter = AwsSageMakerAdapter(region="us-east-2", sagemaker_client=sagemaker_client)
    sagemaker_stubber.add_response("stop_training_job", {}, {"TrainingJobName": "job-1"})

    adapter.stop_training("job-1")


def test_batch_submit_and_describe(batch_client: object, batch_stubber: Stubber) -> None:
    adapter = AwsBatchAdapter(region="us-east-2", batch_client=batch_client)
    spec = BatchJobSpec(
        job_name="neng-batch-run-1-123",
        job_queue="queue-a",
        job_definition="jobdef-a",
        parameters={"run_id": "run-1"},
        environment={"NUMERENG_RUN_ID": "run-1"},
        tags={"Project": "numereng", "RunId": "run-1"},
    )
    batch_stubber.add_response(
        "submit_job",
        {"jobName": "neng-batch-run-1-123", "jobId": "batch-job-1"},
        {
            "jobName": "neng-batch-run-1-123",
            "jobQueue": "queue-a",
            "jobDefinition": "jobdef-a",
            "parameters": {"run_id": "run-1"},
            "tags": {"Project": "numereng", "RunId": "run-1"},
            "containerOverrides": {"environment": [{"name": "NUMERENG_RUN_ID", "value": "run-1"}]},
        },
    )
    batch_stubber.add_response(
        "describe_jobs",
        {
            "jobs": [
                {
                    "jobName": "neng-batch-run-1-123",
                    "jobId": "batch-job-1",
                    "jobQueue": "queue-a",
                    "status": "RUNNING",
                    "startedAt": 1,
                    "jobDefinition": "jobdef-a",
                    "container": {"logStreamName": "batch/log/stream"},
                }
            ]
        },
        {"jobs": ["batch-job-1"]},
    )

    job_id = adapter.submit_job(spec)
    status = adapter.describe_job(job_id)
    assert job_id == "batch-job-1"
    assert status.status == "RUNNING"
    assert status.log_stream_name == "batch/log/stream"


def test_batch_cancel_and_terminate(batch_client: object, batch_stubber: Stubber) -> None:
    adapter = AwsBatchAdapter(region="us-east-2", batch_client=batch_client)
    batch_stubber.add_response(
        "cancel_job",
        {},
        {"jobId": "batch-job-1", "reason": "test-cancel"},
    )
    batch_stubber.add_response(
        "terminate_job",
        {},
        {"jobId": "batch-job-1", "reason": "test-terminate"},
    )

    adapter.cancel_job("batch-job-1", reason="test-cancel")
    adapter.terminate_job("batch-job-1", reason="test-terminate")


def test_logs_list_stream_names_with_prefix_avoids_order_by(logs_client: object, logs_stubber: Stubber) -> None:
    adapter = AwsCloudWatchLogsAdapter(region="us-east-2", logs_client=logs_client)
    logs_stubber.add_response(
        "describe_log_streams",
        {
            "logStreams": [
                {"logStreamName": "job/a", "lastEventTimestamp": 100},
                {"logStreamName": "job/b", "lastEventTimestamp": 200},
                {"logStreamName": "job/c", "lastEventTimestamp": 50},
            ]
        },
        {"logGroupName": "/aws/sagemaker/TrainingJobs", "logStreamNamePrefix": "job/", "limit": 2},
    )

    names = adapter.list_stream_names(log_group="/aws/sagemaker/TrainingJobs", stream_prefix="job/", limit=2)
    assert names == ["job/b", "job/a"]


def test_logs_list_stream_names_without_prefix_uses_last_event_order(
    logs_client: object,
    logs_stubber: Stubber,
) -> None:
    adapter = AwsCloudWatchLogsAdapter(region="us-east-2", logs_client=logs_client)
    logs_stubber.add_response(
        "describe_log_streams",
        {"logStreams": [{"logStreamName": "a"}, {"logStreamName": "b"}]},
        {
            "logGroupName": "/aws/batch/job",
            "limit": 2,
            "orderBy": "LastEventTime",
            "descending": True,
        },
    )

    names = adapter.list_stream_names(log_group="/aws/batch/job", stream_prefix="", limit=2)
    assert names == ["a", "b"]


def test_logs_fetch_log_events(logs_client: object, logs_stubber: Stubber) -> None:
    adapter = AwsCloudWatchLogsAdapter(region="us-east-2", logs_client=logs_client)
    logs_stubber.add_response(
        "get_log_events",
        {
            "events": [{"timestamp": 123, "message": "line-1"}],
            "nextForwardToken": "token-2",
        },
        {
            "logGroupName": "/aws/batch/job",
            "logStreamName": "job/stream",
            "limit": 50,
            "startFromHead": False,
            "nextToken": "token-1",
        },
    )

    events, token = adapter.fetch_log_events(
        log_group="/aws/batch/job",
        stream_name="job/stream",
        limit=50,
        next_token="token-1",
        start_from_head=False,
    )
    assert len(events) == 1
    assert events[0].message == "line-1"
    assert token == "token-2"
