from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from botocore.stub import ANY, Stubber

from numereng.features.cloud.aws.aws_adapters import AwsIamAdapter, AwsS3Adapter, AwsSsmAdapter


def test_s3_ensure_bucket_exists_when_bucket_present(s3_client: object, s3_stubber: Stubber) -> None:
    adapter = AwsS3Adapter(region="us-east-2", s3_client=s3_client)
    s3_stubber.add_response("head_bucket", {}, {"Bucket": "bucket-a"})

    adapter.ensure_bucket_exists(bucket="bucket-a", region="us-east-2")


def test_s3_ensure_bucket_exists_creates_us_east_1(s3_client: object, s3_stubber: Stubber) -> None:
    adapter = AwsS3Adapter(region="us-east-1", s3_client=s3_client)
    s3_stubber.add_client_error("head_bucket", service_error_code="404", http_status_code=404)
    s3_stubber.add_response("create_bucket", {}, {"Bucket": "bucket-b"})

    adapter.ensure_bucket_exists(bucket="bucket-b", region="us-east-1")


def test_s3_ensure_bucket_exists_creates_regional_bucket(s3_client: object, s3_stubber: Stubber) -> None:
    adapter = AwsS3Adapter(region="us-east-2", s3_client=s3_client)
    s3_stubber.add_client_error("head_bucket", service_error_code="404", http_status_code=404)
    s3_stubber.add_response(
        "create_bucket",
        {},
        {"Bucket": "bucket-c", "CreateBucketConfiguration": {"LocationConstraint": "us-east-2"}},
    )

    adapter.ensure_bucket_exists(bucket="bucket-c", region="us-east-2")


def test_s3_upload_and_download_file_use_client_methods(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    s3_client: object,
) -> None:
    adapter = AwsS3Adapter(region="us-east-2", s3_client=s3_client)

    uploaded: dict[str, str] = {}

    def fake_upload_file(local: str, bucket: str, key: str) -> None:
        uploaded["local"] = local
        uploaded["bucket"] = bucket
        uploaded["key"] = key

    def fake_download_file(bucket: str, key: str, local: str) -> None:
        uploaded["download_bucket"] = bucket
        uploaded["download_key"] = key
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        Path(local).write_text("ok", encoding="utf-8")

    monkeypatch.setattr(adapter.s3, "upload_file", fake_upload_file)
    monkeypatch.setattr(adapter.s3, "download_file", fake_download_file)

    local_file = tmp_path / "input.txt"
    local_file.write_text("hello", encoding="utf-8")

    uri = adapter.upload_file(local_path=local_file, bucket="bucket-d", key="runs/run-1/input.txt")
    assert uri == "s3://bucket-d/runs/run-1/input.txt"
    assert uploaded["bucket"] == "bucket-d"

    output_path = tmp_path / "out" / "result.txt"
    downloaded = adapter.download_file(bucket="bucket-d", key="runs/run-1/result.txt", local_path=output_path)
    assert downloaded == output_path
    assert output_path.read_text(encoding="utf-8") == "ok"


def test_s3_list_keys_handles_pagination(s3_client: object, s3_stubber: Stubber) -> None:
    adapter = AwsS3Adapter(region="us-east-2", s3_client=s3_client)

    s3_stubber.add_response(
        "list_objects_v2",
        {
            "IsTruncated": True,
            "NextContinuationToken": "token-1",
            "Contents": [{"Key": "runs/run-1/a.json"}],
            "Name": "bucket-e",
            "Prefix": "runs/run-1/",
            "KeyCount": 1,
            "MaxKeys": 1000,
        },
        {"Bucket": "bucket-e", "Prefix": "runs/run-1/"},
    )
    s3_stubber.add_response(
        "list_objects_v2",
        {
            "IsTruncated": False,
            "Contents": [{"Key": "runs/run-1/b.json"}],
            "Name": "bucket-e",
            "Prefix": "runs/run-1/",
            "KeyCount": 1,
            "MaxKeys": 1000,
        },
        {"Bucket": "bucket-e", "Prefix": "runs/run-1/", "ContinuationToken": "token-1"},
    )

    keys = adapter.list_keys(bucket="bucket-e", prefix="runs/run-1/")
    assert keys == ["runs/run-1/a.json", "runs/run-1/b.json"]


def test_s3_delete_key_and_prefix(
    monkeypatch: pytest.MonkeyPatch,
    s3_client: object,
    s3_stubber: Stubber,
) -> None:
    adapter = AwsS3Adapter(region="us-east-2", s3_client=s3_client)

    s3_stubber.add_response("delete_object", {}, {"Bucket": "bucket-f", "Key": "runs/run-1/a.json"})
    adapter.delete_key(bucket="bucket-f", key="runs/run-1/a.json")

    keys = ["runs/run-1/a.json", "runs/run-1/b.json"]

    def fake_list_keys(bucket: str, prefix: str) -> list[str]:
        _ = (bucket, prefix)
        return keys

    monkeypatch.setattr(adapter, "list_keys", fake_list_keys)

    s3_stubber.add_response(
        "delete_objects",
        {"Deleted": [{"Key": "runs/run-1/a.json"}, {"Key": "runs/run-1/b.json"}]},
        {
            "Bucket": "bucket-f",
            "Delete": {
                "Objects": [{"Key": "runs/run-1/a.json"}, {"Key": "runs/run-1/b.json"}],
                "Quiet": True,
            },
        },
    )

    deleted = adapter.delete_prefix(bucket="bucket-f", prefix="runs/run-1/")
    assert deleted == 2


def test_s3_copy_object(s3_client: object, s3_stubber: Stubber) -> None:
    adapter = AwsS3Adapter(region="us-east-2", s3_client=s3_client)
    s3_stubber.add_response(
        "copy_object",
        {
            "CopyObjectResult": {
                "ETag": '"abc"',
                "LastModified": datetime.now(tz=UTC),
            }
        },
        {
            "CopySource": {"Bucket": "src-b", "Key": "a.bin"},
            "Bucket": "dst-b",
            "Key": "b.bin",
        },
    )

    uri = adapter.copy_object("src-b", "a.bin", "dst-b", "b.bin")
    assert uri == "s3://dst-b/b.bin"


def test_ssm_wait_for_ssm_success_after_retry(
    monkeypatch: pytest.MonkeyPatch,
    ssm_client: object,
    ssm_stubber: Stubber,
) -> None:
    adapter = AwsSsmAdapter(region="us-east-2", ssm_client=ssm_client)

    ssm_stubber.add_response(
        "describe_instance_information",
        {"InstanceInformationList": [{"InstanceId": "i-1", "PingStatus": "Offline"}]},
        {"Filters": [{"Key": "InstanceIds", "Values": ["i-1"]}]},
    )
    ssm_stubber.add_response(
        "describe_instance_information",
        {"InstanceInformationList": [{"InstanceId": "i-1", "PingStatus": "Online"}]},
        {"Filters": [{"Key": "InstanceIds", "Values": ["i-1"]}]},
    )
    monkeypatch.setattr("numereng.features.cloud.aws.aws_adapters.time.sleep", lambda _: None)

    adapter.wait_for_ssm("i-1", timeout_seconds=60)


def test_ssm_wait_for_ssm_timeout(monkeypatch: pytest.MonkeyPatch, ssm_client: object) -> None:
    adapter = AwsSsmAdapter(region="us-east-2", ssm_client=ssm_client)

    monkeypatch.setattr(
        adapter.ssm,
        "describe_instance_information",
        lambda Filters: {"InstanceInformationList": []},
    )

    timeline = iter([0.0, 0.1, 11.0])
    monkeypatch.setattr("numereng.features.cloud.aws.aws_adapters.time.time", lambda: next(timeline))
    monkeypatch.setattr("numereng.features.cloud.aws.aws_adapters.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="did not become SSM-online"):
        adapter.wait_for_ssm("i-2", timeout_seconds=10)


def test_ssm_run_command_success(
    monkeypatch: pytest.MonkeyPatch,
    ssm_client: object,
    ssm_stubber: Stubber,
) -> None:
    adapter = AwsSsmAdapter(region="us-east-2", ssm_client=ssm_client)

    ssm_stubber.add_response(
        "send_command",
        {"Command": {"CommandId": "cccccccc-cccc-cccc-cccc-cccccccccccc"}},
        {
            "InstanceIds": ["i-3"],
            "DocumentName": "AWS-RunShellScript",
            "Parameters": {"commands": ["echo ok"]},
            "TimeoutSeconds": 60,
        },
    )
    ssm_stubber.add_response(
        "get_command_invocation",
        {
            "CommandId": "cccccccc-cccc-cccc-cccc-cccccccccccc",
            "InstanceId": "i-3",
            "Status": "InProgress",
            "ResponseCode": -1,
            "StandardOutputContent": "",
            "StandardErrorContent": "",
        },
        {"CommandId": "cccccccc-cccc-cccc-cccc-cccccccccccc", "InstanceId": "i-3"},
    )
    ssm_stubber.add_response(
        "get_command_invocation",
        {
            "CommandId": "cccccccc-cccc-cccc-cccc-cccccccccccc",
            "InstanceId": "i-3",
            "Status": "Success",
            "ResponseCode": 0,
            "StandardOutputContent": "done\n",
            "StandardErrorContent": "",
        },
        {"CommandId": "cccccccc-cccc-cccc-cccc-cccccccccccc", "InstanceId": "i-3"},
    )
    monkeypatch.setattr("numereng.features.cloud.aws.aws_adapters.time.sleep", lambda _: None)

    result = adapter.run_command("i-3", "echo ok", timeout_seconds=60)
    assert result.exit_code == 0
    assert result.stdout == "done\n"


def test_ssm_run_command_failed_status_raises(
    monkeypatch: pytest.MonkeyPatch,
    ssm_client: object,
    ssm_stubber: Stubber,
) -> None:
    adapter = AwsSsmAdapter(region="us-east-2", ssm_client=ssm_client)

    ssm_stubber.add_response(
        "send_command",
        {"Command": {"CommandId": "dddddddd-dddd-dddd-dddd-dddddddddddd"}},
        {
            "InstanceIds": ["i-4"],
            "DocumentName": "AWS-RunShellScript",
            "Parameters": {"commands": ["exit 2"]},
            "TimeoutSeconds": 30,
        },
    )
    ssm_stubber.add_response(
        "get_command_invocation",
        {
            "CommandId": "dddddddd-dddd-dddd-dddd-dddddddddddd",
            "InstanceId": "i-4",
            "Status": "Failed",
            "ResponseCode": 2,
            "StandardOutputContent": "",
            "StandardErrorContent": "boom",
        },
        {"CommandId": "dddddddd-dddd-dddd-dddd-dddddddddddd", "InstanceId": "i-4"},
    )
    monkeypatch.setattr("numereng.features.cloud.aws.aws_adapters.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="ssm_command_failed"):
        adapter.run_command("i-4", "exit 2", timeout_seconds=30)


def test_iam_ensure_training_role_existing(iam_client: object, iam_stubber: Stubber) -> None:
    adapter = AwsIamAdapter(iam_client=iam_client)

    iam_stubber.add_response(
        "get_role",
        {
            "Role": {
                "Path": "/",
                "RoleName": "numereng-training-role",
                "RoleId": "AROAXYZ123456789",
                "Arn": "arn:aws:iam::123:role/numereng-training-role",
                "CreateDate": datetime.now(tz=UTC),
                "AssumeRolePolicyDocument": "{}",
                "MaxSessionDuration": 3600,
            }
        },
        {"RoleName": "numereng-training-role"},
    )
    iam_stubber.add_response(
        "put_role_policy",
        {},
        {"RoleName": "numereng-training-role", "PolicyName": "numereng-training-role-policy", "PolicyDocument": ANY},
    )
    iam_stubber.add_response(
        "attach_role_policy",
        {},
        {
            "RoleName": "numereng-training-role",
            "PolicyArn": "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
        },
    )

    arn = adapter.ensure_training_role("numereng-training-role", "bucket-z")
    assert arn == "arn:aws:iam::123:role/numereng-training-role"


def test_iam_ensure_training_role_create_when_missing(iam_client: object, iam_stubber: Stubber) -> None:
    adapter = AwsIamAdapter(iam_client=iam_client)

    iam_stubber.add_client_error("get_role", service_error_code="NoSuchEntity", http_status_code=404)
    iam_stubber.add_response(
        "create_role",
        {
            "Role": {
                "Path": "/",
                "RoleName": "new-role",
                "RoleId": "AROANEW123456789",
                "Arn": "arn:aws:iam::123:role/new-role",
                "CreateDate": datetime.now(tz=UTC),
                "AssumeRolePolicyDocument": "{}",
                "MaxSessionDuration": 3600,
            }
        },
        {
            "RoleName": "new-role",
            "AssumeRolePolicyDocument": ANY,
            "Description": "IAM role for numereng EC2 training",
            "Tags": [{"Key": "Project", "Value": "numereng"}],
        },
    )
    iam_stubber.add_response(
        "put_role_policy",
        {},
        {"RoleName": "new-role", "PolicyName": "new-role-policy", "PolicyDocument": ANY},
    )
    iam_stubber.add_response(
        "attach_role_policy",
        {},
        {
            "RoleName": "new-role",
            "PolicyArn": "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
        },
    )

    arn = adapter.ensure_training_role("new-role", "bucket-new")
    assert arn == "arn:aws:iam::123:role/new-role"


def test_iam_ensure_instance_profile_paths(iam_client: object, iam_stubber: Stubber) -> None:
    adapter = AwsIamAdapter(iam_client=iam_client)

    iam_stubber.add_response(
        "get_instance_profile",
        {
            "InstanceProfile": {
                "Path": "/",
                "InstanceProfileName": "prof-a",
                "InstanceProfileId": "AIPROF1234567890",
                "Arn": "arn:aws:iam::123:instance-profile/prof-a",
                "CreateDate": datetime.now(tz=UTC),
                "Roles": [],
            }
        },
        {"InstanceProfileName": "prof-a"},
    )
    iam_stubber.add_response(
        "add_role_to_instance_profile",
        {},
        {"InstanceProfileName": "prof-a", "RoleName": "prof-a"},
    )

    assert adapter.ensure_instance_profile("prof-a") == "arn:aws:iam::123:instance-profile/prof-a"


def test_iam_ensure_security_group_existing(ec2_client: object, ec2_stubber: Stubber) -> None:
    adapter = AwsIamAdapter(ec2_client_factory=lambda region: ec2_client)

    ec2_stubber.add_response(
        "describe_security_groups",
        {"SecurityGroups": [{"GroupId": "sg-existing"}]},
        {"Filters": [{"Name": "group-name", "Values": ["numereng-training"]}]},
    )

    group_id = adapter.ensure_security_group(region="us-east-2", group_name="numereng-training")
    assert group_id == "sg-existing"


def test_iam_ensure_security_group_create_when_missing(ec2_client: object, ec2_stubber: Stubber) -> None:
    adapter = AwsIamAdapter(ec2_client_factory=lambda region: ec2_client)

    ec2_stubber.add_response(
        "describe_security_groups",
        {"SecurityGroups": []},
        {"Filters": [{"Name": "group-name", "Values": ["numereng-training"]}]},
    )
    ec2_stubber.add_response(
        "describe_vpcs",
        {"Vpcs": [{"VpcId": "vpc-123"}]},
        {"Filters": [{"Name": "is-default", "Values": ["true"]}]},
    )
    ec2_stubber.add_response(
        "create_security_group",
        {"GroupId": "sg-created"},
        {
            "GroupName": "numereng-training",
            "Description": "numereng training security group",
            "VpcId": "vpc-123",
            "TagSpecifications": [
                {
                    "ResourceType": "security-group",
                    "Tags": [
                        {"Key": "Name", "Value": "numereng-training"},
                        {"Key": "Project", "Value": "numereng"},
                    ],
                }
            ],
        },
    )
    ec2_stubber.add_response(
        "authorize_security_group_ingress",
        {},
        {
            "GroupId": "sg-created",
            "IpPermissions": [
                {
                    "IpProtocol": "tcp",
                    "FromPort": 22,
                    "ToPort": 22,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH access"}],
                }
            ],
        },
    )

    group_id = adapter.ensure_security_group(region="us-east-2", group_name="numereng-training")
    assert group_id == "sg-created"


def test_iam_ensure_security_group_raises_without_default_vpc(ec2_client: object, ec2_stubber: Stubber) -> None:
    adapter = AwsIamAdapter(ec2_client_factory=lambda region: ec2_client)

    ec2_stubber.add_response(
        "describe_security_groups",
        {"SecurityGroups": []},
        {"Filters": [{"Name": "group-name", "Values": ["numereng-training"]}]},
    )
    ec2_stubber.add_response(
        "describe_vpcs",
        {"Vpcs": []},
        {"Filters": [{"Name": "is-default", "Values": ["true"]}]},
    )

    with pytest.raises(RuntimeError, match="no default VPC"):
        adapter.ensure_security_group(region="us-east-2", group_name="numereng-training")
