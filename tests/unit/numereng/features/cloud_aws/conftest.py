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
def ecr_client() -> object:
    return boto3.client("ecr", region_name=_TEST_REGION)


@pytest.fixture
def sts_client() -> object:
    return boto3.client("sts", region_name=_TEST_REGION)


@pytest.fixture
def sagemaker_client() -> object:
    return boto3.client("sagemaker", region_name=_TEST_REGION)


@pytest.fixture
def batch_client() -> object:
    return boto3.client("batch", region_name=_TEST_REGION)


@pytest.fixture
def logs_client() -> object:
    return boto3.client("logs", region_name=_TEST_REGION)


@pytest.fixture
def ecr_stubber(ecr_client: object) -> Stubber:
    stubber = Stubber(ecr_client)
    stubber.activate()
    try:
        yield stubber
    finally:
        stubber.deactivate()


@pytest.fixture
def sts_stubber(sts_client: object) -> Stubber:
    stubber = Stubber(sts_client)
    stubber.activate()
    try:
        yield stubber
    finally:
        stubber.deactivate()


@pytest.fixture
def sagemaker_stubber(sagemaker_client: object) -> Stubber:
    stubber = Stubber(sagemaker_client)
    stubber.activate()
    try:
        yield stubber
    finally:
        stubber.deactivate()


@pytest.fixture
def batch_stubber(batch_client: object) -> Stubber:
    stubber = Stubber(batch_client)
    stubber.activate()
    try:
        yield stubber
    finally:
        stubber.deactivate()


@pytest.fixture
def logs_stubber(logs_client: object) -> Stubber:
    stubber = Stubber(logs_client)
    stubber.activate()
    try:
        yield stubber
    finally:
        stubber.deactivate()
