from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import pytest

from numereng.features.training.mlflow_tracking import maybe_log_training_run


def test_maybe_log_training_run_disabled_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NUMERENG_MLFLOW_ENABLED", raising=False)
    result = maybe_log_training_run(
        run_id="run-1",
        config={"a": 1},
        metrics_payload={"corr": 0.1},
        artifacts={},
        output_root=tmp_path,
    )
    assert result is None


def test_maybe_log_training_run_enabled_without_mlflow_returns_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NUMERENG_MLFLOW_ENABLED", "1")
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)

    result = maybe_log_training_run(
        run_id="run-1",
        config={"a": 1},
        metrics_payload={"corr": 0.1},
        artifacts={},
        output_root=tmp_path,
    )
    assert result == "mlflow_not_installed"


def test_maybe_log_training_run_enabled_logs_to_mlflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRun:
        def __enter__(self) -> _FakeRun:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> Literal[False]:
            _ = (exc_type, exc, tb)
            return False

    class _FakeMlflow:
        def __init__(self) -> None:
            self.params: dict[str, str] = {}
            self.metrics: dict[str, float] = {}
            self.artifacts: list[str] = []
            self.tracking_uri: str | None = None
            self.experiment: str | None = None

        def set_tracking_uri(self, value: str) -> None:
            self.tracking_uri = value

        def set_experiment(self, value: str) -> None:
            self.experiment = value

        def start_run(self, run_name: str) -> _FakeRun:
            _ = run_name
            return _FakeRun()

        def log_param(self, key: str, value: str) -> None:
            self.params[key] = value

        def log_metric(self, key: str, value: float) -> None:
            self.metrics[key] = value

        def log_artifact(self, path: str) -> None:
            self.artifacts.append(path)

    fake_mlflow = _FakeMlflow()
    monkeypatch.setenv("NUMERENG_MLFLOW_ENABLED", "true")
    monkeypatch.setenv("NUMERENG_MLFLOW_TRACKING_URI", "http://mlflow.local")
    monkeypatch.setenv("NUMERENG_MLFLOW_EXPERIMENT", "numereng-exp")
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    artifact_path = tmp_path / "metrics.json"
    artifact_path.write_text("{}", encoding="utf-8")

    result = maybe_log_training_run(
        run_id="run-2",
        config={"training": {"engine": {"mode": "official"}}},
        metrics_payload={"corr": {"mean": 0.123}},
        artifacts={"metrics": "metrics.json"},
        output_root=tmp_path,
    )

    assert result is None
    assert fake_mlflow.tracking_uri == "http://mlflow.local"
    assert fake_mlflow.experiment == "numereng-exp"
    assert fake_mlflow.params["training.engine.mode"] == "official"
    assert fake_mlflow.metrics["corr.mean"] == 0.123
    assert str(artifact_path) in fake_mlflow.artifacts
