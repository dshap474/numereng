from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def ec2_instance_payload(
    *,
    instance_id: str,
    state: str,
    instance_type: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    tags: list[dict[str, str]] = [{"Key": "Project", "Value": "numereng"}]
    if run_id is not None:
        tags.append({"Key": "RunId", "Value": run_id})
    return {
        "InstanceId": instance_id,
        "State": {"Name": state},
        "InstanceType": instance_type,
        "LaunchTime": datetime(2026, 2, 20, 12, 0, 0, tzinfo=UTC),
        "Tags": tags,
        "PrivateIpAddress": "10.0.0.12",
        "PublicIpAddress": "3.10.1.20",
    }


def describe_instances_response(*, instances: list[dict[str, Any]]) -> dict[str, Any]:
    return {"Reservations": [{"Instances": instances}]}


def ssm_invocation(*, status: str, response_code: int, stdout: str = "", stderr: str = "") -> dict[str, Any]:
    return {
        "CommandId": "cmd-123",
        "InstanceId": "i-123",
        "Status": status,
        "ResponseCode": response_code,
        "StandardOutputContent": stdout,
        "StandardErrorContent": stderr,
    }
