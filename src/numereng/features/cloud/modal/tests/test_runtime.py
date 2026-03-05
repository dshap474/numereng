from __future__ import annotations

import json
from pathlib import Path

import pytest

from numereng.features.cloud.modal import runtime
from numereng.features.cloud.modal.contracts import ModalRuntimePayload
from numereng.features.telemetry import get_launch_metadata
from numereng.features.training.models import TrainingRunResult


def test_build_runtime_payload_from_config_reads_source_file(tmp_path: Path) -> None:
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {"data_version": "v5.2", "dataset_variant": "non_downsampled"},
                "model": {"type": "LGBMRegressor", "params": {}},
                "training": {},
            }
        ),
        encoding="utf-8",
    )

    payload = runtime.build_runtime_payload_from_config(
        config_path=config_path,
        output_dir=str(tmp_path / "output"),
        engine_mode="custom",
        window_size_eras=144,
        embargo_eras=9,
    )

    assert payload.config_filename == "train.json"
    assert "data_version" in payload.config_text
    assert payload.engine_mode == "custom"
    assert payload.window_size_eras == 144
    assert payload.embargo_eras == 9


def test_run_training_payload_invokes_training_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_training(
        *,
        config_path: str | Path,
        output_dir: str | Path | None,
        engine_mode: str | None,
        window_size_eras: int | None,
        embargo_eras: int | None,
    ) -> TrainingRunResult:
        launch_metadata = get_launch_metadata()
        assert launch_metadata is not None
        assert launch_metadata.source == "cloud.modal.runtime"
        assert launch_metadata.operation_type == "run"
        assert launch_metadata.job_type == "run"
        captured["config_path"] = str(config_path)
        captured["output_dir"] = str(output_dir) if output_dir is not None else None
        captured["engine_mode"] = engine_mode
        captured["window_size_eras"] = window_size_eras
        captured["embargo_eras"] = embargo_eras
        return TrainingRunResult(
            run_id="run-42",
            predictions_path=tmp_path / "preds.parquet",
            results_path=tmp_path / "results.json",
        )

    monkeypatch.setattr(runtime, "_load_run_training", lambda: fake_run_training)
    payload = ModalRuntimePayload(
        config_text=json.dumps(
            {
                "data": {"data_version": "v5.2", "dataset_variant": "non_downsampled"},
                "model": {"type": "LGBMRegressor", "params": {}},
                "training": {},
            }
        ),
        config_filename="train.json",
        output_dir=str(tmp_path / "remote-out"),
        engine_mode="official",
    )

    result = runtime.run_training_payload(payload)

    assert result["run_id"] == "run-42"
    assert result["predictions_path"].endswith("preds.parquet")
    assert result["results_path"].endswith("results.json")
    assert captured["engine_mode"] == "official"
    assert captured["window_size_eras"] is None
    assert captured["embargo_eras"] is None


def test_run_training_payload_rejects_parent_traversal_filename() -> None:
    payload = ModalRuntimePayload(
        config_text='{"x": 1}',
        config_filename="../escape.json",
    )

    with pytest.raises(ValueError, match="runtime_config_filename_invalid"):
        runtime.run_training_payload(payload)


def test_run_training_payload_rejects_absolute_filename() -> None:
    payload = ModalRuntimePayload(
        config_text='{"x": 1}',
        config_filename="/tmp/escape.json",
    )

    with pytest.raises(ValueError, match="runtime_config_filename_invalid"):
        runtime.run_training_payload(payload)


def test_run_training_payload_rejects_nested_filename() -> None:
    payload = ModalRuntimePayload(
        config_text='{"x": 1}',
        config_filename="nested/train.json",
    )

    with pytest.raises(ValueError, match="runtime_config_filename_invalid"):
        runtime.run_training_payload(payload)
