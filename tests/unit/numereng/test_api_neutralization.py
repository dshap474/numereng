from __future__ import annotations

from pathlib import Path

import pytest

import numereng.api as api_module
from numereng.features.feature_neutralization import (
    NeutralizationResult,
    NeutralizationValidationError,
)
from numereng.platform.errors import PackageError


def _result(*, run_id: str | None) -> NeutralizationResult:
    return NeutralizationResult(
        source_path=Path("/tmp/source.parquet"),
        output_path=Path("/tmp/output.neutralized.parquet"),
        run_id=run_id,
        neutralizer_path=Path("/tmp/neutralizers.parquet"),
        neutralizer_cols=("feature_a", "feature_b"),
        proportion=0.5,
        mode="era",
        rank_output=True,
        source_rows=100,
        neutralizer_rows=100,
        matched_rows=100,
    )


def test_neutralize_apply_by_predictions_delegates_and_returns_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_neutralize_prediction_artifact(*, request: object) -> NeutralizationResult:
        assert getattr(request, "predictions_path") == Path("predictions.parquet")
        return _result(run_id=None)

    monkeypatch.setattr(api_module, "neutralize_prediction_artifact", fake_neutralize_prediction_artifact)

    response = api_module.neutralize_apply(
        api_module.NeutralizeRequest(
            predictions_path="predictions.parquet",
            neutralizer_path="neutralizers.parquet",
        )
    )

    assert response.output_path == "/tmp/output.neutralized.parquet"
    assert response.run_id is None


def test_neutralize_apply_preserves_explicit_empty_neutralizer_cols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_neutralize_prediction_artifact(*, request: object) -> NeutralizationResult:
        assert getattr(request, "neutralizer_cols") == ()
        return _result(run_id=None)

    monkeypatch.setattr(api_module, "neutralize_prediction_artifact", fake_neutralize_prediction_artifact)

    response = api_module.neutralize_apply(
        api_module.NeutralizeRequest(
            predictions_path="predictions.parquet",
            neutralizer_path="neutralizers.parquet",
            neutralizer_cols=[],
        )
    )

    assert response.run_id is None


def test_neutralize_apply_by_run_delegates_and_returns_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_neutralize_run_prediction_artifact(**kwargs: object) -> NeutralizationResult:
        assert kwargs["run_id"] == "run-1"
        return _result(run_id="run-1")

    monkeypatch.setattr(api_module, "neutralize_run_prediction_artifact", fake_neutralize_run_prediction_artifact)

    response = api_module.neutralize_apply(
        api_module.NeutralizeRequest(
            run_id="run-1",
            neutralizer_path="neutralizers.parquet",
        )
    )

    assert response.run_id == "run-1"


def test_neutralize_apply_translates_feature_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_neutralize_prediction_artifact(*, request: object) -> NeutralizationResult:
        _ = request
        raise NeutralizationValidationError("neutralization_failed")

    monkeypatch.setattr(api_module, "neutralize_prediction_artifact", fake_neutralize_prediction_artifact)

    with pytest.raises(PackageError, match="neutralization_failed"):
        api_module.neutralize_apply(
            api_module.NeutralizeRequest(
                predictions_path="predictions.parquet",
                neutralizer_path="neutralizers.parquet",
            )
        )
