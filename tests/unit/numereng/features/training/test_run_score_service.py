from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd
import pyarrow.parquet as pq
import pytest

import numereng.features.scoring.run_service as run_score_module
from numereng.features.scoring.models import PostTrainingScoringResult, ResolvedScoringPolicy, ScoringArtifactBundle
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
            "training": {"scoring": {"requested_stage": "all", "refreshed_stages": []}},
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
                "benchmark_source": {"source": "active"},
                "meta_model_col": "numerai_meta_model",
            },
            "output": {"predictions_name": "pred"},
        },
    )
    _write_json(
        run_dir / "results.json",
        {
            "metrics": {"status": "failed", "reason": "old"},
            "output": {"output_dir": str(run_dir), "predictions_file": "artifacts/predictions/pred.parquet"},
            "training": {"scoring": {"requested_stage": "all", "refreshed_stages": []}},
        },
    )

    monkeypatch.setattr(run_score_module, "create_training_data_client", lambda: object())
    monkeypatch.setattr(
        run_score_module,
        "run_scoring",
        lambda **kwargs: PostTrainingScoringResult(
            summaries=_scoring_summaries(),
            score_provenance={
                "stages": {
                    "emitted": ["run_metric_series", "post_training_core_summary"],
                    "omissions": {"post_training_full": "feature_diagnostics_unavailable"},
                }
            },
            policy=ResolvedScoringPolicy(
                fnc_feature_set="fncv3_features",
                fnc_target_policy="scoring_target",
                benchmark_min_overlap_ratio=0.0,
            ),
            artifacts=ScoringArtifactBundle(
                series_frames={},
                manifest={},
                stage_frames={
                    "run_metric_series": pd.DataFrame(
                        [
                            {
                                "run_id": run_id,
                                "config_hash": "abc123",
                                "seed": None,
                                "target_col": "target",
                                "payout_target_col": "target_ender_20",
                                "prediction_col": "prediction",
                                "era": "era1",
                                "metric_key": "corr_native",
                                "series_type": "per_era",
                                "value": 0.12,
                            }
                        ]
                    ),
                    "post_training_core_summary": pd.DataFrame(
                        [
                            {
                                "run_id": run_id,
                                "config_hash": "abc123",
                                "seed": None,
                                "target_col": "target",
                                "payout_target_col": "target_ender_20",
                                "prediction_col": "prediction",
                                "corr_native_mean": 0.1,
                                "corr_native_std": 0.2,
                                "corr_native_sharpe": 0.5,
                                "corr_native_max_drawdown": 0.1,
                            }
                        ]
                    ),
                },
            ),
            requested_stage="all",
            refreshed_stages=("run_metric_series", "post_training_core"),
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
    assert saved_results["training"]["scoring"]["status"] == "succeeded"
    assert saved_results["training"]["scoring"]["requested_stage"] == "all"
    assert saved_results["training"]["scoring"]["refreshed_stages"] == ["run_metric_series", "post_training_core"]
    assert saved_results["training"]["scoring"]["emitted_stage_files"] == [
        "run_metric_series",
        "post_training_core_summary",
    ]
    assert saved_results["training"]["scoring"]["omissions"]["post_training_full"] == "feature_diagnostics_unavailable"

    saved_manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert saved_manifest["artifacts"]["score_provenance"] == "score_provenance.json"
    assert saved_manifest["artifacts"]["scoring_manifest"] == "artifacts/scoring/manifest.json"
    assert saved_manifest["metrics_summary"]["corr"]["mean"] == 0.1
    assert saved_manifest["training"]["scoring"]["status"] == "succeeded"
    assert saved_manifest["training"]["scoring"]["requested_stage"] == "all"
    assert (run_dir / "artifacts" / "scoring" / "run_metric_series.parquet").is_file()
    assert (run_dir / "artifacts" / "scoring" / "post_training_core_summary.parquet").is_file()
    scoring_manifest = json.loads((run_dir / "artifacts" / "scoring" / "manifest.json").read_text(encoding="utf-8"))
    assert scoring_manifest["requested_stage"] == "all"
    assert scoring_manifest["current_canonical_stages"] == ["run_metric_series", "post_training_core"]
    run_metric_parquet = pq.ParquetFile(run_dir / "artifacts" / "scoring" / "run_metric_series.parquet")
    summary_parquet = pq.ParquetFile(run_dir / "artifacts" / "scoring" / "post_training_core_summary.parquet")
    assert run_metric_parquet.metadata.row_group(0).column(0).compression == "ZSTD"
    assert summary_parquet.metadata.row_group(0).column(0).compression == "ZSTD"


def test_score_run_requires_existing_run(tmp_path: Path) -> None:
    with pytest.raises(TrainingError, match="training_score_run_not_found:run-missing"):
        run_score_module.score_run(run_id="run-missing", store_root=tmp_path / "store")


def test_score_run_partial_stage_preserves_existing_unselected_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    run_id = "run-123"
    run_dir = store_root / "runs" / run_id
    predictions_path = run_dir / "artifacts" / "predictions" / "pred.parquet"
    scoring_dir = run_dir / "artifacts" / "scoring"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text("placeholder", encoding="utf-8")
    scoring_dir.mkdir(parents=True, exist_ok=True)
    existing_stage = scoring_dir / "post_training_core_summary.parquet"
    pd.DataFrame(
        [
            {
                "run_id": run_id,
                "config_hash": "abc123",
                "seed": None,
                "target_col": "target",
                "payout_target_col": "target_ender_20",
                "prediction_col": "prediction",
                "corr_native_mean": 0.2,
                "corr_native_std": 0.1,
                "corr_native_sharpe": 0.4,
                "corr_native_max_drawdown": 0.2,
            }
        ]
    ).to_parquet(existing_stage, index=False)

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
                "benchmark_source": {"source": "active"},
                "meta_model_col": "numerai_meta_model",
            },
            "output": {"predictions_name": "pred"},
        },
    )
    _write_json(run_dir / "results.json", {"output": {}, "training": {}})

    monkeypatch.setattr(run_score_module, "create_training_data_client", lambda: object())
    monkeypatch.setattr(
        run_score_module,
        "run_scoring",
        lambda **kwargs: PostTrainingScoringResult(
            summaries=_scoring_summaries(),
            score_provenance={"stages": {"emitted": ["run_metric_series"], "omissions": {}}},
            policy=ResolvedScoringPolicy(
                fnc_feature_set="fncv3_features",
                fnc_target_policy="scoring_target",
                benchmark_min_overlap_ratio=0.0,
            ),
            artifacts=ScoringArtifactBundle(
                series_frames={},
                manifest={},
                stage_frames={
                    "run_metric_series": pd.DataFrame(
                        [
                            {
                                "run_id": run_id,
                                "config_hash": "abc123",
                                "seed": None,
                                "target_col": "target",
                                "payout_target_col": "target_ender_20",
                                "prediction_col": "prediction",
                                "era": "era1",
                                "metric_key": "corr_native",
                                "series_type": "per_era",
                                "value": 0.12,
                            }
                        ]
                    ),
                },
            ),
            requested_stage="run_metric_series",
            refreshed_stages=("run_metric_series",),
        ),
    )
    monkeypatch.setattr(run_score_module, "index_run", lambda **kwargs: None)

    run_score_module.score_run(run_id=run_id, store_root=store_root, stage="run_metric_series")

    assert (scoring_dir / "run_metric_series.parquet").is_file()
    assert existing_stage.is_file()
    scoring_manifest = json.loads((scoring_dir / "manifest.json").read_text(encoding="utf-8"))
    assert scoring_manifest["requested_stage"] == "run_metric_series"
    assert scoring_manifest["current_canonical_stages"] == ["run_metric_series", "post_training_core"]
    assert scoring_manifest["refreshed_canonical_stages"] == ["run_metric_series"]


def test_score_run_stage_limited_provenance_matches_persisted_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
                "benchmark_source": {"source": "active"},
                "meta_model_col": "numerai_meta_model",
            },
            "output": {"predictions_name": "pred"},
        },
    )
    _write_json(run_dir / "results.json", {"output": {}, "training": {}})

    monkeypatch.setattr(run_score_module, "create_training_data_client", lambda: object())
    monkeypatch.setattr(
        run_score_module,
        "run_scoring",
        lambda **kwargs: PostTrainingScoringResult(
            summaries=_scoring_summaries(),
            score_provenance={
                "stages": {
                    "emitted": ["post_training_core_summary"],
                    "omissions": {"post_training_full": "not_requested"},
                },
                "artifacts": {
                    "charts": ["run_metric_series"],
                    "stage_files": [
                        "post_fold_per_era",
                        "post_fold_snapshots",
                        "post_training_core_summary",
                        "run_metric_series",
                    ],
                },
            },
            policy=ResolvedScoringPolicy(
                fnc_feature_set="fncv3_features",
                fnc_target_policy="scoring_target",
                benchmark_min_overlap_ratio=0.0,
            ),
            artifacts=ScoringArtifactBundle(
                series_frames={},
                manifest={},
                stage_frames={
                    "run_metric_series": pd.DataFrame(
                        [
                            {
                                "run_id": run_id,
                                "config_hash": "abc123",
                                "seed": None,
                                "target_col": "target",
                                "payout_target_col": "target_ender_20",
                                "prediction_col": "prediction",
                                "era": "era1",
                                "metric_key": "corr_native",
                                "series_type": "per_era",
                                "value": 0.12,
                            }
                        ]
                    ),
                    "post_training_core_summary": pd.DataFrame(
                        [
                            {
                                "run_id": run_id,
                                "config_hash": "abc123",
                                "seed": None,
                                "target_col": "target",
                                "payout_target_col": "target_ender_20",
                                "prediction_col": "prediction",
                                "corr_native_mean": 0.1,
                                "corr_native_std": 0.2,
                                "corr_native_sharpe": 0.5,
                                "corr_native_max_drawdown": 0.1,
                            }
                        ]
                    ),
                },
            ),
            requested_stage="post_training_core",
            refreshed_stages=("run_metric_series", "post_training_core"),
        ),
    )
    monkeypatch.setattr(run_score_module, "index_run", lambda **kwargs: None)

    run_score_module.score_run(run_id=run_id, store_root=store_root, stage="post_training_core")

    saved_provenance = json.loads((run_dir / "score_provenance.json").read_text(encoding="utf-8"))
    artifacts_block = cast(dict[str, object], saved_provenance["artifacts"])
    assert artifacts_block["charts"] == ["run_metric_series"]
    assert artifacts_block["stage_files"] == ["post_training_core_summary"]
    assert artifacts_block["requested_stage"] == "post_training_core"
    assert artifacts_block["refreshed_canonical_stages"] == ["run_metric_series", "post_training_core"]
    saved_results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert saved_results["training"]["scoring"]["emitted_stage_files"] == [
        "run_metric_series",
        "post_training_core_summary",
    ]
    assert saved_results["training"]["scoring"]["omissions"] == {"post_training_full": "not_requested"}
    assert (run_dir / "artifacts" / "scoring" / "run_metric_series.parquet").is_file()
    scoring_manifest = json.loads((run_dir / "artifacts" / "scoring" / "manifest.json").read_text(encoding="utf-8"))
    assert scoring_manifest["current_canonical_stages"] == ["run_metric_series", "post_training_core"]
