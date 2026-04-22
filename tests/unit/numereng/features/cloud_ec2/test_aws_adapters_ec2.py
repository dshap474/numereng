from __future__ import annotations

from typing import Any

import pytest
from botocore.stub import Stubber

from numereng.features.cloud.aws.adapters import InstanceStatus, LaunchInstanceSpec
from numereng.features.cloud.aws.aws_adapters import AwsEc2Adapter

from .stub_payloads import (
    describe_instances_response,
    ec2_instance_payload,
)


def _launch_spec(*, use_spot: bool = True, security_group: str = "numereng-training") -> LaunchInstanceSpec:
    return LaunchInstanceSpec(
        image_id="ami-abc123",
        instance_type="r7i.4xlarge",
        user_data="#!/bin/bash\necho hi\n",
        run_id="run-1",
        region="us-east-2",
        iam_role_name="numereng-training-role",
        security_group=security_group,
        bucket="example-bucket",
        data_version="v5.2",
        use_spot=use_spot,
        volume_size_gb=120,
        tags={"Env": "test"},
    )


def test_launch_instance_spot_with_security_group_lookup(ec2_client: object, ec2_stubber: Stubber) -> None:
    adapter = AwsEc2Adapter(region="us-east-2", ec2_client=ec2_client)
    spec = _launch_spec(use_spot=True, security_group="numereng-training")

    expected_sg_params = {"Filters": [{"Name": "group-name", "Values": ["numereng-training"]}]}
    ec2_stubber.add_response(
        "describe_security_groups",
        {"SecurityGroups": [{"GroupId": "sg-12345"}]},
        expected_sg_params,
    )

    expected_run_params: dict[str, Any] = {
        "ImageId": "ami-abc123",
        "InstanceType": "r7i.4xlarge",
        "MinCount": 1,
        "MaxCount": 1,
        "UserData": spec.user_data,
        "InstanceInitiatedShutdownBehavior": "terminate",
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/xvda",
                "Ebs": {
                    "VolumeSize": 120,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        ],
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": "numereng-run-1"},
                    {"Key": "Project", "Value": "numereng"},
                    {"Key": "RunId", "Value": "run-1"},
                    {"Key": "Env", "Value": "test"},
                ],
            }
        ],
        "IamInstanceProfile": {"Name": "numereng-training-role"},
        "SecurityGroupIds": ["sg-12345"],
        "InstanceMarketOptions": {
            "MarketType": "spot",
            "SpotOptions": {
                "SpotInstanceType": "one-time",
                "InstanceInterruptionBehavior": "terminate",
            },
        },
    }
    ec2_stubber.add_response(
        "run_instances",
        {
            "Instances": [
                {
                    "InstanceId": "i-abc123",
                    "ImageId": "ami-abc123",
                    "State": {"Code": 0, "Name": "pending"},
                }
            ]
        },
        expected_run_params,
    )

    instance_id = adapter.launch_instance(spec)
    assert instance_id == "i-abc123"


def test_launch_instance_on_demand_uses_sg_id_directly(ec2_client: object, ec2_stubber: Stubber) -> None:
    adapter = AwsEc2Adapter(region="us-east-2", ec2_client=ec2_client)
    spec = _launch_spec(use_spot=False, security_group="sg-direct")

    expected_run_params: dict[str, Any] = {
        "ImageId": "ami-abc123",
        "InstanceType": "r7i.4xlarge",
        "MinCount": 1,
        "MaxCount": 1,
        "UserData": spec.user_data,
        "InstanceInitiatedShutdownBehavior": "terminate",
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/xvda",
                "Ebs": {
                    "VolumeSize": 120,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        ],
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": "numereng-run-1"},
                    {"Key": "Project", "Value": "numereng"},
                    {"Key": "RunId", "Value": "run-1"},
                    {"Key": "Env", "Value": "test"},
                ],
            }
        ],
        "IamInstanceProfile": {"Name": "numereng-training-role"},
        "SecurityGroupIds": ["sg-direct"],
    }
    ec2_stubber.add_response(
        "run_instances",
        {
            "Instances": [
                {
                    "InstanceId": "i-ondemand",
                    "ImageId": "ami-abc123",
                    "State": {"Code": 0, "Name": "pending"},
                }
            ]
        },
        expected_run_params,
    )

    instance_id = adapter.launch_instance(spec)
    assert instance_id == "i-ondemand"


def test_launch_instance_raises_for_invalid_instance_id(ec2_client: object, ec2_stubber: Stubber) -> None:
    adapter = AwsEc2Adapter(region="us-east-2", ec2_client=ec2_client)
    spec = _launch_spec(security_group="sg-direct")

    ec2_stubber.add_response(
        "run_instances",
        {
            "Instances": [
                {
                    "ImageId": "ami-abc123",
                    "State": {"Code": 0, "Name": "pending"},
                }
            ]
        },
        {
            "ImageId": "ami-abc123",
            "InstanceType": "r7i.4xlarge",
            "MinCount": 1,
            "MaxCount": 1,
            "UserData": spec.user_data,
            "InstanceInitiatedShutdownBehavior": "terminate",
            "BlockDeviceMappings": [
                {
                    "DeviceName": "/dev/xvda",
                    "Ebs": {
                        "VolumeSize": 120,
                        "VolumeType": "gp3",
                        "DeleteOnTermination": True,
                    },
                }
            ],
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": "numereng-run-1"},
                        {"Key": "Project", "Value": "numereng"},
                        {"Key": "RunId", "Value": "run-1"},
                        {"Key": "Env", "Value": "test"},
                    ],
                }
            ],
            "IamInstanceProfile": {"Name": "numereng-training-role"},
            "SecurityGroupIds": ["sg-direct"],
            "InstanceMarketOptions": {
                "MarketType": "spot",
                "SpotOptions": {
                    "SpotInstanceType": "one-time",
                    "InstanceInterruptionBehavior": "terminate",
                },
            },
        },
    )

    with pytest.raises(RuntimeError, match="invalid InstanceId"):
        adapter.launch_instance(spec)


def test_wait_for_instance_success(monkeypatch: pytest.MonkeyPatch, ec2_client: object) -> None:
    adapter = AwsEc2Adapter(region="us-east-2", ec2_client=ec2_client)
    states = iter(
        [
            InstanceStatus(instance_id="i-1", state="pending", instance_type="r7i.4xlarge"),
            InstanceStatus(instance_id="i-1", state="running", instance_type="r7i.4xlarge"),
        ]
    )

    monkeypatch.setattr(adapter, "get_instance_status", lambda _: next(states))
    monkeypatch.setattr("numereng.features.cloud.aws.aws_adapters.time.sleep", lambda _: None)

    assert adapter.wait_for_instance("i-1", target_state="running", timeout_seconds=30)


def test_wait_for_instance_false_on_terminated(monkeypatch: pytest.MonkeyPatch, ec2_client: object) -> None:
    adapter = AwsEc2Adapter(region="us-east-2", ec2_client=ec2_client)
    monkeypatch.setattr(
        adapter,
        "get_instance_status",
        lambda _: InstanceStatus(instance_id="i-1", state="terminated", instance_type="r7i.4xlarge"),
    )
    assert adapter.wait_for_instance("i-1", target_state="running", timeout_seconds=30) is False


def test_get_instance_status_parses_instance_payload(ec2_client: object, ec2_stubber: Stubber) -> None:
    adapter = AwsEc2Adapter(region="us-east-2", ec2_client=ec2_client)

    ec2_stubber.add_response(
        "describe_instances",
        describe_instances_response(
            instances=[
                ec2_instance_payload(
                    instance_id="i-xyz",
                    state="running",
                    instance_type="r7i.4xlarge",
                    run_id="run-55",
                )
            ]
        ),
        {"InstanceIds": ["i-xyz"]},
    )

    status = adapter.get_instance_status("i-xyz")
    assert status.instance_id == "i-xyz"
    assert status.state == "running"
    assert status.instance_type == "r7i.4xlarge"
    assert status.run_id == "run-55"


def test_get_instance_status_raises_when_not_found(ec2_client: object, ec2_stubber: Stubber) -> None:
    adapter = AwsEc2Adapter(region="us-east-2", ec2_client=ec2_client)
    ec2_stubber.add_response("describe_instances", {"Reservations": []}, {"InstanceIds": ["i-missing"]})

    with pytest.raises(RuntimeError, match="instance not found"):
        adapter.get_instance_status("i-missing")


def test_list_training_instances_uses_expected_filters(ec2_client: object, ec2_stubber: Stubber) -> None:
    adapter = AwsEc2Adapter(region="us-east-2", ec2_client=ec2_client)

    ec2_stubber.add_response(
        "describe_instances",
        describe_instances_response(
            instances=[
                ec2_instance_payload(
                    instance_id="i-list-1",
                    state="running",
                    instance_type="r7i.2xlarge",
                    run_id="run-list",
                )
            ]
        ),
        {
            "Filters": [
                {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]},
                {"Name": "tag:Project", "Values": ["numereng"]},
            ]
        },
    )

    listed = adapter.list_training_instances()
    assert len(listed) == 1
    assert listed[0].instance_id == "i-list-1"
    assert listed[0].run_id == "run-list"


def test_get_spot_price_returns_float(ec2_client: object, ec2_stubber: Stubber) -> None:
    adapter = AwsEc2Adapter(region="us-east-2", ec2_client=ec2_client)
    ec2_stubber.add_response(
        "describe_spot_price_history",
        {
            "SpotPriceHistory": [
                {
                    "AvailabilityZone": "us-east-2a",
                    "InstanceType": "r7i.4xlarge",
                    "ProductDescription": "Linux/UNIX",
                    "SpotPrice": "0.1023",
                    "Timestamp": "2026-02-20T00:00:00Z",
                }
            ]
        },
        {
            "InstanceTypes": ["r7i.4xlarge"],
            "ProductDescriptions": ["Linux/UNIX"],
            "MaxResults": 1,
        },
    )

    assert adapter.get_spot_price("r7i.4xlarge") == pytest.approx(0.1023)


def test_get_spot_price_handles_missing_or_invalid_values(ec2_client: object, ec2_stubber: Stubber) -> None:
    adapter = AwsEc2Adapter(region="us-east-2", ec2_client=ec2_client)
    ec2_stubber.add_response(
        "describe_spot_price_history",
        {"SpotPriceHistory": []},
        {
            "InstanceTypes": ["r7i.2xlarge"],
            "ProductDescriptions": ["Linux/UNIX"],
            "MaxResults": 1,
        },
    )
    ec2_stubber.add_response(
        "describe_spot_price_history",
        {
            "SpotPriceHistory": [
                {
                    "AvailabilityZone": "us-east-2a",
                    "InstanceType": "r7i.2xlarge",
                    "ProductDescription": "Linux/UNIX",
                    "SpotPrice": "not-a-number",
                    "Timestamp": "2026-02-20T00:00:00Z",
                }
            ]
        },
        {
            "InstanceTypes": ["r7i.2xlarge"],
            "ProductDescriptions": ["Linux/UNIX"],
            "MaxResults": 1,
        },
    )

    assert adapter.get_spot_price("r7i.2xlarge") is None
    assert adapter.get_spot_price("r7i.2xlarge") is None


def test_resolve_security_group_id_by_name_and_missing(ec2_client: object, ec2_stubber: Stubber) -> None:
    adapter = AwsEc2Adapter(region="us-east-2", ec2_client=ec2_client)

    assert adapter.resolve_security_group_id("sg-direct") == "sg-direct"

    ec2_stubber.add_response(
        "describe_security_groups",
        {"SecurityGroups": [{"GroupId": "sg-resolved"}]},
        {"Filters": [{"Name": "group-name", "Values": ["named-sg"]}]},
    )
    ec2_stubber.add_response(
        "describe_security_groups",
        {"SecurityGroups": []},
        {"Filters": [{"Name": "group-name", "Values": ["missing-sg"]}]},
    )

    assert adapter.resolve_security_group_id("named-sg") == "sg-resolved"
    assert adapter.resolve_security_group_id("missing-sg") is None


def test_terminate_instance_calls_ec2_api(ec2_client: object, ec2_stubber: Stubber) -> None:
    adapter = AwsEc2Adapter(region="us-east-2", ec2_client=ec2_client)
    ec2_stubber.add_response(
        "terminate_instances",
        {
            "TerminatingInstances": [
                {
                    "InstanceId": "i-kill",
                    "CurrentState": {"Code": 32, "Name": "shutting-down"},
                    "PreviousState": {"Code": 16, "Name": "running"},
                }
            ]
        },
        {"InstanceIds": ["i-kill"]},
    )

    adapter.terminate_instance("i-kill")
