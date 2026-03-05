from __future__ import annotations

from pathlib import Path

from numereng.features.cloud.aws.contracts import CloudEc2State
from numereng.features.cloud.aws.state_store import CloudEc2StateStore


def test_state_store_roundtrip(tmp_path: Path) -> None:
    store = CloudEc2StateStore()
    state_path = tmp_path / "state.json"

    state = CloudEc2State(
        run_id="run-1",
        instance_id="i-123",
        region="us-east-2",
        bucket="bucket-1",
        status="ready",
        artifacts={"config": "s3://bucket-1/runs/run-1/config.json"},
    )
    store.save(state_path, state)

    loaded = store.load(state_path)
    assert loaded is not None
    assert loaded.run_id == "run-1"
    assert loaded.instance_id == "i-123"
    assert loaded.status == "ready"
    assert loaded.artifacts["config"].endswith("config.json")


def test_state_store_load_missing_returns_none(tmp_path: Path) -> None:
    store = CloudEc2StateStore()
    assert store.load(tmp_path / "does-not-exist.json") is None
