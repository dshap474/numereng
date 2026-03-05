from __future__ import annotations

from pathlib import Path

import pytest

from numereng.features.cloud.modal.contracts import CloudModalState
from numereng.features.cloud.modal.state_store import CloudModalStateStore


def test_load_missing_returns_none(tmp_path: Path) -> None:
    store = CloudModalStateStore()
    loaded = store.load(tmp_path / "missing-state.json")
    assert loaded is None


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    store = CloudModalStateStore()
    path = tmp_path / "state" / "modal-state.json"
    state = CloudModalState(
        run_id="run-1",
        call_id="fc-1",
        app_name="numereng-train",
        function_name="train_remote",
        status="submitted",
        artifacts={"predictions_path": "/tmp/preds.parquet"},
        metadata={"note": "ok"},
    )

    store.save(path, state)
    loaded = store.load(path)

    assert loaded is not None
    assert loaded.call_id == "fc-1"
    assert loaded.artifacts["predictions_path"] == "/tmp/preds.parquet"
    assert path.exists()


def test_load_raises_for_invalid_payload_shape(tmp_path: Path) -> None:
    store = CloudModalStateStore()
    path = tmp_path / "state.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid state document"):
        store.load(path)


def test_load_raises_for_invalid_json(tmp_path: Path) -> None:
    store = CloudModalStateStore()
    path = tmp_path / "state.json"
    path.write_text("{broken", encoding="utf-8")

    with pytest.raises(ValueError):
        store.load(path)
