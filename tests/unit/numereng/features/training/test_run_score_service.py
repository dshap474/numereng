from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

import numereng.features.scoring.run_service as run_score_module
from numereng.features.scoring.models import PostTrainingScoringResult, ResolvedScoringPolicy
from numereng.features.training.errors import TrainingError


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _summary_frame(mean: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "mean": mean,
                "std": 0.2,
                "sharpe": 0.5,
                "max_drawdown": 0.1,
                "avg_corr_with_benchmark": 0.01,
            }
        ],
        index=["prediction"],
    )


def _scoring_summaries() -> dict[str, pd.DataFrame]:
    return {
        "corr": _summary_frame(0.1),
        "fnc": _summary_frame(0.09),
        "feature_exposure": _summary_frame(0.12),
        "max_feature_exposure": _summary_frame(0.22),
        "mmc": _summary_frame(0.03),
        "cwmm": _summary_frame(0.2),
        "bmc": _summary_frame(0.01),
        "bmc_last_200_eras": _summary_frame(0.02),
    }


def test_metrics_payload_from_summaries_excludes_payout_fields() -> None:
    payload = run_score_module._metrics_payload_from_summaries(_scoring_summaries())
    assert "payout_estimate" not in payload
    assert "payout_estimate_mean" not in payload
    assert isinstance(payload["feature_exposure"], dict)
    assert isinstance(payload["max_feature_exposure"], dict)
    assert cast(dict[str, object], payload["feature_exposure"])["mean"] == pytest.approx(0.12)
    assert cast(dict[str, object], payload["max_feature_exposure"])["mean"] == pytest.approx(0.22)


def test_score_run_updates_run_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    run_id = "run-123"
    run_dir = store_root / "runs" / run_id
    predictions_path = run_dir / "artifacts" / "predictions" / "pred.parquet"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text("placeholder", encoding="utf-8")

    _write_json(
        run_dir / "run.json",
        {
            "run_id": run_id,
            "status": "FINISHED",
            "data": {"version": "v5.2", "feature_set": "medium", "target_col": "target"},
            "artifacts": {
                "predictions": "artifacts/predictions/pred.parquet",
                "resolved_config": "resolved.json",
                "results": "results.json",
            },
            "training": {
                "scoring": {"mode": "materialized", "era_chunk_size": 64, "effective_backend": "failed"}
            },
        },
    )
    _write_json(
        run_dir / "resolved.json",
        {
            "data": {
                "data_version": "v5.2",
                "dataset_variant": "non_downsampled",
                "dataset_scope": "train_plus_validation",
                "feature_set": "medium",
                "target_col": "target",
                "benchmark_model": "v52_lgbm_ender20",
                "meta_model_col": "numerai_meta_model",
                "loading": {"scoring_mode": "materialized", "era_chunk_size": 64},
            },
            "output": {"predictions_name": "pred"},
        },
    )
    _write_json(
        run_dir / "results.json",
        {
            "metrics": {"status": "failed", "reason": "old"},
            "output": {"output_dir": str(run_dir), "predictions_file": "artifacts/predictions/pred.parquet"},
            "training": {"scoring": {"mode": "materialized", "era_chunk_size": 64, "effective_backend": "failed"}},
        },
    )

    monkeypatch.setattr(run_score_module, "create_training_data_client", lambda: object())
    monkeypatch.setattr(
        run_score_module,
        "run_post_training_scoring",
        lambda **kwargs: PostTrainingScoringResult(
            summaries=_scoring_summaries(),
            score_provenance={"execution": {"effective_scoring_mode": "materialized"}},
            effective_scoring_backend="materialized",
            policy=ResolvedScoringPolicy(
                fnc_feature_set="fncv3_features",
                fnc_target_policy="scoring_target",
                benchmark_min_overlap_ratio=0.0,
                include_feature_neutral_metrics=True,
            ),
        ),
    )
    indexed_runs: list[str] = []
    monkeypatch.setattr(
        run_score_module,
        "index_run",
        lambda *, store_root, run_id: indexed_runs.append(run_id),
    )

    result = run_score_module.score_run(run_id=run_id, store_root=store_root)

    assert result.run_id == run_id
    assert result.predictions_path == predictions_path.resolve()
    assert result.metrics_path == run_dir / "metrics.json"
    assert result.score_provenance_path == run_dir / "score_provenance.json"
    assert indexed_runs == [run_id]

    saved_results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert saved_results["metrics"]["corr"]["mean"] == 0.1
    assert "payout_estimate_mean" not in saved_results["metrics"]
    assert saved_results["output"]["score_provenance_file"] == "score_provenance.json"
    assert saved_results["training"]["scoring"]["effective_backend"] == "materialized"
    assert saved_results["training"]["scoring"]["policy"]["fnc_target_policy"] == "scoring_target"
    assert saved_results["training"]["scoring"]["policy"]["include_feature_neutral_metrics"] is True

    saved_manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert saved_manifest["artifacts"]["score_provenance"] == "score_provenance.json"
    assert saved_manifest["metrics_summary"]["corr"]["mean"] == 0.1
    assert saved_manifest["training"]["scoring"]["policy"]["include_feature_neutral_metrics"] is True


def test_score_run_requires_existing_run(tmp_path: Path) -> None:
    with pytest.raises(TrainingError, match="training_score_run_not_found:run-missing"):
        run_score_module.score_run(run_id="run-missing", store_root=tmp_path / "store")
