from __future__ import annotations

from pathlib import Path

import pytest

from numereng.features.cloud.aws.managed_contracts import CloudAwsState
from numereng.features.cloud.aws.managed_state_store import CloudAwsStateStore


def test_managed_state_store_roundtrip(tmp_path: Path) -> None:
    store = CloudAwsStateStore()
    state_path = tmp_path / "managed-state.json"
    state = CloudAwsState(
        run_id="run-1",
        backend="sagemaker",
        region="us-east-2",
        bucket="example-bucket",
        status="InProgress",
        artifacts={"output_s3_uri": "s3://example-bucket/runs/run-1/managed-output/"},
    )

    store.save(state_path, state)
    loaded = store.load(state_path)

    assert loaded is not None
    assert loaded.run_id == "run-1"
    assert loaded.backend == "sagemaker"
    assert loaded.status == "InProgress"


def test_managed_state_store_load_missing_returns_none(tmp_path: Path) -> None:
    store = CloudAwsStateStore()
    assert store.load(tmp_path / "missing.json") is None


def test_managed_state_store_invalid_json_is_normalized(tmp_path: Path) -> None:
    store = CloudAwsStateStore()
    state_path = tmp_path / "broken.json"
    state_path.write_text("{bad json", encoding="utf-8")

    with pytest.raises(ValueError, match="state_document_invalid_json"):
        store.load(state_path)


def test_managed_state_store_invalid_schema_is_normalized(tmp_path: Path) -> None:
    store = CloudAwsStateStore()
    state_path = tmp_path / "broken-schema.json"
    state_path.write_text('{"backend":"not-valid"}', encoding="utf-8")

    with pytest.raises(ValueError, match="state_document_invalid_schema"):
        store.load(state_path)
