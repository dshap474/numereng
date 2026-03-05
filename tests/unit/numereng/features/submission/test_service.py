from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from numereng.features.submission.service import (
    SubmissionModelNotFoundError,
    SubmissionPredictionsFileNotFoundError,
    SubmissionPredictionsReadError,
    SubmissionRunIdInvalidError,
    SubmissionRunNotFoundError,
    SubmissionRunPredictionsNotFoundError,
    SubmissionRunPredictionsNotLiveEligibleError,
    SubmissionRunPredictionsPathUnsafeError,
    submit_predictions_file,
    submit_run_predictions,
)


class _FakeSubmissionClient:
    def __init__(self, models: dict[str, str], submission_id: str = "submission-1") -> None:
        self._models = models
        self._submission_id = submission_id
        self.upload_calls: list[tuple[str, str]] = []

    def get_models(self) -> dict[str, str]:
        return dict(self._models)

    def upload_predictions(self, *, file_path: str, model_id: str) -> str:
        self.upload_calls.append((file_path, model_id))
        return self._submission_id


def _write_predictions_file(
    path: Path,
    *,
    include_target: bool = False,
    include_cv_fold: bool = False,
) -> None:
    payload: dict[str, list[object]] = {
        "id": ["id_1", "id_2"],
        "prediction": [0.1, 0.2],
    }
    if include_target:
        payload["target"] = [0.3, 0.4]
    if include_cv_fold:
        payload["cv_fold"] = [0, 1]
    frame = pd.DataFrame(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def test_submit_predictions_file_success(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.csv"
    predictions_path.write_text("id,prediction\n1,0.1\n", encoding="utf-8")
    client = _FakeSubmissionClient(models={"main": "model-1"})

    result = submit_predictions_file(
        predictions_path=predictions_path,
        model_name="main",
        client=client,
    )

    expected_path = str(predictions_path.resolve())
    assert result.submission_id == "submission-1"
    assert result.model_id == "model-1"
    assert result.model_name == "main"
    assert str(result.predictions_path) == expected_path
    assert client.upload_calls == [(expected_path, "model-1")]


def test_submit_predictions_file_missing_file(tmp_path: Path) -> None:
    client = _FakeSubmissionClient(models={"main": "model-1"})

    with pytest.raises(SubmissionPredictionsFileNotFoundError):
        submit_predictions_file(
            predictions_path=tmp_path / "missing.csv",
            model_name="main",
            client=client,
        )


def test_submit_predictions_file_missing_model(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.csv"
    predictions_path.write_text("id,prediction\n1,0.1\n", encoding="utf-8")
    client = _FakeSubmissionClient(models={"other": "model-2"})

    with pytest.raises(SubmissionModelNotFoundError):
        submit_predictions_file(
            predictions_path=predictions_path,
            model_name="main",
            client=client,
        )


def test_submit_run_predictions_resolves_val_predictions(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    predictions_path = store_root / "runs" / "run-1" / "artifacts" / "predictions" / "val_predictions.parquet"
    _write_predictions_file(predictions_path)
    client = _FakeSubmissionClient(models={"main": "model-1"})

    result = submit_run_predictions(
        run_id="run-1",
        model_name="main",
        store_root=store_root,
        client=client,
    )

    expected_path = str(predictions_path.resolve())
    assert result.run_id == "run-1"
    assert str(result.predictions_path) == expected_path
    assert client.upload_calls == [(expected_path, "model-1")]


def test_submit_run_predictions_prefers_manifest_predictions_path(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-1"
    predictions_path = run_dir / "artifacts" / "predictions" / "custom_name.parquet"
    _write_predictions_file(predictions_path)
    (run_dir / "run.json").write_text(
        json.dumps({"artifacts": {"predictions": "artifacts/predictions/custom_name.parquet"}}),
        encoding="utf-8",
    )
    client = _FakeSubmissionClient(models={"main": "model-1"})

    result = submit_run_predictions(
        run_id="run-1",
        model_name="main",
        store_root=store_root,
        client=client,
    )

    expected_path = str(predictions_path.resolve())
    assert result.run_id == "run-1"
    assert str(result.predictions_path) == expected_path
    assert client.upload_calls == [(expected_path, "model-1")]


def test_submit_run_predictions_resolves_single_generic_predictions_file(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    predictions_path = store_root / "runs" / "run-1" / "artifacts" / "predictions" / "engine_run.parquet"
    _write_predictions_file(predictions_path)
    client = _FakeSubmissionClient(models={"main": "model-1"})

    result = submit_run_predictions(
        run_id="run-1",
        model_name="main",
        store_root=store_root,
        client=client,
    )

    expected_path = str(predictions_path.resolve())
    assert result.run_id == "run-1"
    assert str(result.predictions_path) == expected_path
    assert client.upload_calls == [(expected_path, "model-1")]


def test_submit_run_predictions_resolves_uppercase_generic_predictions_file(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    predictions_path = store_root / "runs" / "run-1" / "artifacts" / "predictions" / "engine_run.CSV"
    _write_predictions_file(predictions_path)
    client = _FakeSubmissionClient(models={"main": "model-1"})

    result = submit_run_predictions(
        run_id="run-1",
        model_name="main",
        store_root=store_root,
        client=client,
    )

    expected_path = str(predictions_path.resolve())
    assert result.run_id == "run-1"
    assert str(result.predictions_path) == expected_path
    assert client.upload_calls == [(expected_path, "model-1")]


def test_submit_run_predictions_missing_run(tmp_path: Path) -> None:
    client = _FakeSubmissionClient(models={"main": "model-1"})

    with pytest.raises(SubmissionRunNotFoundError):
        submit_run_predictions(
            run_id="missing",
            model_name="main",
            store_root=tmp_path / ".numereng",
            client=client,
        )


def test_submit_run_predictions_rejects_invalid_run_id(tmp_path: Path) -> None:
    client = _FakeSubmissionClient(models={"main": "model-1"})

    with pytest.raises(SubmissionRunIdInvalidError):
        submit_run_predictions(
            run_id="../run-1",
            model_name="main",
            store_root=tmp_path / ".numereng",
            client=client,
        )


def test_submit_run_predictions_rejects_manifest_path_escape(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-1"
    outside_path = tmp_path / "outside.parquet"
    _write_predictions_file(outside_path)
    (run_dir / "artifacts" / "predictions").mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(
        json.dumps({"artifacts": {"predictions": "../../../outside.parquet"}}),
        encoding="utf-8",
    )
    client = _FakeSubmissionClient(models={"main": "model-1"})

    with pytest.raises(SubmissionRunPredictionsPathUnsafeError):
        submit_run_predictions(
            run_id="run-1",
            model_name="main",
            store_root=store_root,
            client=client,
        )


def test_submit_run_predictions_missing_predictions(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    (store_root / "runs" / "run-1" / "artifacts" / "predictions").mkdir(parents=True)
    client = _FakeSubmissionClient(models={"main": "model-1"})

    with pytest.raises(SubmissionRunPredictionsNotFoundError):
        submit_run_predictions(
            run_id="run-1",
            model_name="main",
            store_root=store_root,
            client=client,
        )


def test_submit_predictions_file_rejects_non_live_columns(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    _write_predictions_file(predictions_path, include_target=True)
    client = _FakeSubmissionClient(models={"main": "model-1"})

    with pytest.raises(SubmissionRunPredictionsNotLiveEligibleError):
        submit_predictions_file(
            predictions_path=predictions_path,
            model_name="main",
            client=client,
        )


def test_submit_predictions_file_allows_non_live_columns_with_override(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    _write_predictions_file(predictions_path, include_target=True)
    client = _FakeSubmissionClient(models={"main": "model-1"})

    result = submit_predictions_file(
        predictions_path=predictions_path,
        model_name="main",
        client=client,
        allow_non_live_artifact=True,
    )

    expected_path = str(predictions_path.resolve())
    assert str(result.predictions_path) == expected_path
    assert client.upload_calls == [(expected_path, "model-1")]


def test_submit_predictions_file_surfaces_read_failure(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    predictions_path.write_text("not parquet", encoding="utf-8")
    client = _FakeSubmissionClient(models={"main": "model-1"})

    with pytest.raises(SubmissionPredictionsReadError):
        submit_predictions_file(
            predictions_path=predictions_path,
            model_name="main",
            client=client,
        )


def test_submit_run_predictions_rejects_non_live_columns(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    predictions_path = store_root / "runs" / "run-1" / "artifacts" / "predictions" / "predictions.parquet"
    _write_predictions_file(predictions_path, include_target=True, include_cv_fold=True)
    client = _FakeSubmissionClient(models={"main": "model-1"})

    with pytest.raises(SubmissionRunPredictionsNotLiveEligibleError):
        submit_run_predictions(
            run_id="run-1",
            model_name="main",
            store_root=store_root,
            client=client,
        )


def test_submit_run_predictions_allows_non_live_columns_with_override(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    predictions_path = store_root / "runs" / "run-1" / "artifacts" / "predictions" / "predictions.parquet"
    _write_predictions_file(predictions_path, include_target=True)
    client = _FakeSubmissionClient(models={"main": "model-1"})

    result = submit_run_predictions(
        run_id="run-1",
        model_name="main",
        store_root=store_root,
        client=client,
        allow_non_live_artifact=True,
    )

    expected_path = str(predictions_path.resolve())
    assert result.run_id == "run-1"
    assert str(result.predictions_path) == expected_path
    assert client.upload_calls == [(expected_path, "model-1")]
