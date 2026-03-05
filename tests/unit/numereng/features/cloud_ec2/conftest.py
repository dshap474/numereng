from __future__ import annotations

import boto3
import pytest
from botocore.stub import Stubber

_TEST_REGION = "us-east-2"


@pytest.fixture(autouse=True)
def _aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", _TEST_REGION)
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


@pytest.fixture
def ec2_client() -> object:
    return boto3.client("ec2", region_name=_TEST_REGION)


@pytest.fixture
def s3_client() -> object:
    return boto3.client("s3", region_name=_TEST_REGION)


@pytest.fixture
def ssm_client() -> object:
    return boto3.client("ssm", region_name=_TEST_REGION)


@pytest.fixture
def iam_client() -> object:
    return boto3.client("iam", region_name=_TEST_REGION)


@pytest.fixture
def ec2_stubber(ec2_client: object) -> Stubber:
    stubber = Stubber(ec2_client)
    stubber.activate()
    try:
        yield stubber
    finally:
        stubber.deactivate()


@pytest.fixture
def s3_stubber(s3_client: object) -> Stubber:
    stubber = Stubber(s3_client)
    stubber.activate()
    try:
        yield stubber
    finally:
        stubber.deactivate()


@pytest.fixture
def ssm_stubber(ssm_client: object) -> Stubber:
    stubber = Stubber(ssm_client)
    stubber.activate()
    try:
        yield stubber
    finally:
        stubber.deactivate()


@pytest.fixture
def iam_stubber(iam_client: object) -> Stubber:
    stubber = Stubber(iam_client)
    stubber.activate()
    try:
        yield stubber
    finally:
        stubber.deactivate()
