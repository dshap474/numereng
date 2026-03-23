from __future__ import annotations

import pytest
from pydantic import ValidationError

from numereng.features.cloud.modal.contracts import (
    ModalDataSyncRequest,
    ModalDeployRequest,
    ModalRuntimePayload,
    ModalTrainSubmitRequest,
    parse_ecr_image_uri,
)


def test_parse_ecr_image_uri_accepts_expected_shape() -> None:
    ref = parse_ecr_image_uri("123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training:latest")

    assert ref.account_id == "123456789012"
    assert ref.region == "us-east-2"
    assert ref.repository == "numereng-training"
    assert ref.tag == "latest"


def test_parse_ecr_image_uri_rejects_digest_form() -> None:
    with pytest.raises(ValueError, match="invalid ecr_image_uri"):
        parse_ecr_image_uri("123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training@sha256:deadbeef")


def test_modal_deploy_request_rejects_invalid_ecr_uri() -> None:
    with pytest.raises(ValidationError, match="invalid ecr_image_uri"):
        ModalDeployRequest(ecr_image_uri="not-a-valid-uri")


def test_modal_deploy_request_allows_runtime_options() -> None:
    request = ModalDeployRequest(
        ecr_image_uri="123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training:latest",
        timeout_seconds=600,
        gpu="T4",
        cpu=2.0,
        memory_mb=4096,
    )

    assert request.timeout_seconds == 600
    assert request.gpu == "T4"
    assert request.cpu == 2.0
    assert request.memory_mb == 4096


def test_modal_deploy_request_allows_data_volume_name() -> None:
    request = ModalDeployRequest(
        ecr_image_uri="123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training:latest",
        data_volume_name="numereng-v52",
    )

    assert request.data_volume_name == "numereng-v52"


def test_modal_data_sync_request_requires_config_and_volume() -> None:
    with pytest.raises(ValidationError, match="config_path must not be empty"):
        ModalDataSyncRequest(config_path=" ", volume_name="numereng-v52")

    with pytest.raises(ValidationError, match="volume_name must not be empty"):
        ModalDataSyncRequest(config_path="train.json", volume_name=" ")


def test_modal_requests_require_json_config_paths() -> None:
    with pytest.raises(ValidationError, match="config_path must reference a .json file"):
        ModalDataSyncRequest(config_path="train.yaml", volume_name="numereng-v52")

    with pytest.raises(ValidationError, match="config_path must reference a .json file"):
        ModalTrainSubmitRequest(config_path="train.yaml")


def test_modal_runtime_payload_requires_json_filename() -> None:
    with pytest.raises(ValidationError, match="config_filename must reference a .json file"):
        ModalRuntimePayload(config_text="{}", config_filename="train.yaml")
