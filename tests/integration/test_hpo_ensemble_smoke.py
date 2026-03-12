from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import numereng.api as api_module
import numereng.features.hpo.service as hpo_service_module
from numereng.features.feature_neutralization import NeutralizationResult
from numereng.features.training import TrainingRunResult


def _write_base_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "data": {
                    "data_version": "v5.2",
        "dataset_variant": "non_downsampled",
                    "feature_set": "small",
                    "target_col": "target_ender_20",
                    "era_col": "era",
                    "id_col": "id",
                },
                "model": {
                    "type": "LGBMRegressor",
                    "params": {
                        "learning_rate": 0.01,
                        "num_leaves": 64,
                    },
                },
                "training": {
                    "engine": {
                        "mode": "custom",
                        "window_size_eras": 156,
                        "embargo_eras": 8,
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_run_predictions(store_root: Path, run_id: str, frame: pd.DataFrame) -> None:
    run_dir = store_root / "runs" / run_id
    predictions_dir = run_dir / "artifacts" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = predictions_dir / "predictions.csv"
    frame.to_csv(predictions_path, index=False)

    manifest = {
        "run_id": run_id,
        "artifacts": {
            "predictions": "artifacts/predictions/predictions.csv",
        },
    }
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _write_neutralizers(path: Path) -> None:
    pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["0001", "0001", "0002", "0002"],
            "feature_1": [0.1, 0.2, 0.3, 0.4],
            "feature_2": [0.9, 0.8, 0.7, 0.6],
        }
    ).to_parquet(path, index=False)


@pytest.mark.integration
def test_hpo_api_roundtrip_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    config_path = tmp_path / "base.json"
    _write_base_config(config_path)

    metric_values = [0.11, 0.13]
    run_index = {"value": 0}

    def fake_run_training(
        *,
        config_path: str | Path,
        output_dir: str | Path | None = None,
        client: object | None = None,
        engine_mode: str | None = None,
        window_size_eras: int | None = None,
        embargo_eras: int | None = None,
        experiment_id: str | None = None,
    ) -> TrainingRunResult:
        _ = (config_path, output_dir, client, engine_mode, window_size_eras, embargo_eras, experiment_id)
        idx = run_index["value"]
        run_index["value"] += 1
        run_id = f"run-{idx}"
        results_path = tmp_path / f"results_{idx}.json"
        results_path.write_text(
            json.dumps({"metrics": {"bmc_last_200_eras": {"mean": metric_values[idx]}}}),
            encoding="utf-8",
        )
        predictions_path = tmp_path / f"preds_{idx}.parquet"
        predictions_path.write_bytes(b"PAR1")
        return TrainingRunResult(
            run_id=run_id,
            predictions_path=predictions_path,
            results_path=results_path,
        )

    def fake_run_optuna_study(
        *,
        direction: str,
        n_trials: int,
        sampler: str,
        seed: int | None,
        specs: tuple[object, ...],
        objective_callback: Callable[[int, dict[str, Any]], float],
    ) -> tuple[int | None, float | None]:
        _ = (direction, sampler, seed, specs)
        _ = objective_callback(0, {"model.params.learning_rate": 0.01})
        value_1 = objective_callback(1, {"model.params.learning_rate": 0.02})
        assert n_trials == 2
        return 1, value_1

    monkeypatch.setattr(hpo_service_module, "run_training", fake_run_training)
    monkeypatch.setattr(hpo_service_module, "run_optuna_study", fake_run_optuna_study)
    monkeypatch.setattr(hpo_service_module, "index_run", lambda **kwargs: None)

    created = api_module.hpo_create(
        api_module.HpoStudyCreateRequest(
            study_name="integration-smoke",
            config_path=str(config_path),
            experiment_id="exp-smoke",
            n_trials=2,
            search_space={
                "model.params.learning_rate": {
                    "type": "float",
                    "low": 0.001,
                    "high": 0.1,
                }
            },
            store_root=str(store_root),
        )
    )
    assert created.status == "completed"
    assert created.best_run_id == "run-1"
    assert created.storage_path is not None
    assert Path(created.storage_path, "trials_live.csv").is_file()

    listed = api_module.hpo_list(
        api_module.HpoStudyListRequest(
            experiment_id="exp-smoke",
            store_root=str(store_root),
        )
    )
    assert len(listed.studies) == 1
    assert listed.studies[0].study_id == created.study_id

    loaded = api_module.hpo_get(
        api_module.HpoStudyGetRequest(
            study_id=created.study_id,
            store_root=str(store_root),
        )
    )
    assert loaded.study_id == created.study_id
    assert loaded.best_value == pytest.approx(0.13)

    trials = api_module.hpo_trials(
        api_module.HpoStudyTrialsRequest(
            study_id=created.study_id,
            store_root=str(store_root),
        )
    )
    assert len(trials.trials) == 2
    assert trials.trials[0].trial_number == 0
    assert trials.trials[1].trial_number == 1


@pytest.mark.integration
def test_hpo_api_roundtrip_with_neutralization_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    config_path = tmp_path / "base.json"
    neutralizer_path = tmp_path / "neutralizers.parquet"
    _write_base_config(config_path)
    _write_neutralizers(neutralizer_path)

    run_index = {"value": 0}

    def fake_run_training(
        *,
        config_path: str | Path,
        output_dir: str | Path | None = None,
        client: object | None = None,
        engine_mode: str | None = None,
        window_size_eras: int | None = None,
        embargo_eras: int | None = None,
        experiment_id: str | None = None,
    ) -> TrainingRunResult:
        _ = (config_path, output_dir, client, engine_mode, window_size_eras, embargo_eras, experiment_id)
        idx = run_index["value"]
        run_index["value"] += 1
        run_id = f"run-{idx}"
        results_path = tmp_path / f"results_{idx}.json"
        results_path.write_text(
            json.dumps({"metrics": {"bmc_last_200_eras": {"mean": 0.1 + (0.01 * idx)}}}),
            encoding="utf-8",
        )
        predictions_path = tmp_path / f"preds_{idx}.parquet"
        pd.DataFrame(
            {
                "era": ["0001", "0001", "0002", "0002"],
                "id": ["a", "b", "c", "d"],
                "target_ender_20": [0.1, 0.9, 0.2, 0.8],
                "prediction": [0.2, 0.7, 0.3, 0.6],
            }
        ).to_parquet(predictions_path, index=False)
        return TrainingRunResult(
            run_id=run_id,
            predictions_path=predictions_path,
            results_path=results_path,
        )

    def fake_run_optuna_study(
        *,
        direction: str,
        n_trials: int,
        sampler: str,
        seed: int | None,
        specs: tuple[object, ...],
        objective_callback: Callable[[int, dict[str, Any]], float],
    ) -> tuple[int | None, float | None]:
        _ = (direction, sampler, seed, specs)
        value_0 = objective_callback(0, {"model.params.learning_rate": 0.01})
        value_1 = objective_callback(1, {"model.params.learning_rate": 0.02})
        assert n_trials == 2
        return 1, max(value_0, value_1)

    def fake_neutralize_predictions_file(*, request: object, run_id: str | None = None) -> NeutralizationResult:
        output_path = tmp_path / f"{run_id or 'file'}.neutralized.parquet"
        pd.DataFrame(
            {
                "era": ["0001", "0001", "0002", "0002"],
                "id": ["a", "b", "c", "d"],
                "target_ender_20": [0.1, 0.9, 0.2, 0.8],
                "prediction": [0.5, 0.6, 0.4, 0.3],
            }
        ).to_parquet(output_path, index=False)
        return NeutralizationResult(
            source_path=Path(getattr(request, "predictions_path")),
            output_path=output_path,
            run_id=run_id,
            neutralizer_path=Path(getattr(request, "neutralizer_path")),
            neutralizer_cols=("feature_1", "feature_2"),
            proportion=float(getattr(request, "proportion")),
            mode="era",
            rank_output=bool(getattr(request, "rank_output")),
            source_rows=4,
            neutralizer_rows=4,
            matched_rows=4,
        )

    def fake_extract_metric_value_from_predictions(
        *,
        predictions_path: Path,
        trial_config_path: Path,
        metric: str,
        scoring_client: object,
    ) -> float:
        _ = (predictions_path, trial_config_path, metric, scoring_client)
        return 0.2

    monkeypatch.setattr(hpo_service_module, "run_training", fake_run_training)
    monkeypatch.setattr(hpo_service_module, "run_optuna_study", fake_run_optuna_study)
    monkeypatch.setattr(hpo_service_module, "index_run", lambda **kwargs: None)
    monkeypatch.setattr(hpo_service_module, "neutralize_predictions_file", fake_neutralize_predictions_file)
    monkeypatch.setattr(
        hpo_service_module,
        "_extract_metric_value_from_predictions",
        fake_extract_metric_value_from_predictions,
    )

    created = api_module.hpo_create(
        api_module.HpoStudyCreateRequest(
            study_name="integration-smoke-neut",
            config_path=str(config_path),
            experiment_id="exp-smoke",
            n_trials=2,
            neutralize=True,
            neutralizer_path=str(neutralizer_path),
            search_space={
                "model.params.learning_rate": {
                    "type": "float",
                    "low": 0.001,
                    "high": 0.1,
                }
            },
            store_root=str(store_root),
        )
    )
    assert created.status == "completed"
    assert created.best_run_id == "run-1"


@pytest.mark.integration
def test_ensemble_api_roundtrip_smoke(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"

    run_a = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["0001", "0001", "0002", "0002"],
            "target_ender_20": [0.1, 0.9, 0.2, 0.8],
            "prediction": [0.2, 0.7, 0.3, 0.6],
        }
    )
    run_b = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["0001", "0001", "0002", "0002"],
            "target_ender_20": [0.1, 0.9, 0.2, 0.8],
            "prediction": [0.3, 0.6, 0.25, 0.75],
        }
    )
    _write_run_predictions(store_root, "run-a", run_a)
    _write_run_predictions(store_root, "run-b", run_b)

    created = api_module.ensemble_build(
        api_module.EnsembleBuildRequest(
            run_ids=["run-a", "run-b"],
            experiment_id="exp-smoke",
            metric="corr_sharpe",
            target="target_ender_20",
            store_root=str(store_root),
        )
    )
    assert created.status == "completed"
    assert len(created.components) == 2
    assert created.artifacts_path is not None

    artifacts_path = Path(created.artifacts_path)
    assert (artifacts_path / "predictions.parquet").is_file()
    assert (artifacts_path / "correlation_matrix.csv").is_file()
    assert (artifacts_path / "metrics.json").is_file()
    assert (artifacts_path / "weights.csv").is_file()
    assert (artifacts_path / "component_metrics.csv").is_file()
    assert (artifacts_path / "era_metrics.csv").is_file()
    assert (artifacts_path / "regime_metrics.csv").is_file()
    assert (artifacts_path / "lineage.json").is_file()

    listed = api_module.ensemble_list(
        api_module.EnsembleListRequest(
            experiment_id="exp-smoke",
            store_root=str(store_root),
        )
    )
    assert len(listed.ensembles) == 1
    assert listed.ensembles[0].ensemble_id == created.ensemble_id

    loaded = api_module.ensemble_get(
        api_module.EnsembleGetRequest(
            ensemble_id=created.ensemble_id,
            store_root=str(store_root),
        )
    )
    assert loaded.ensemble_id == created.ensemble_id
    assert len(loaded.components) == 2


@pytest.mark.integration
def test_ensemble_api_roundtrip_with_neutralization_smoke(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    neutralizer_path = tmp_path / "neutralizers.parquet"
    _write_neutralizers(neutralizer_path)

    run_a = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["0001", "0001", "0002", "0002"],
            "target_ender_20": [0.1, 0.9, 0.2, 0.8],
            "prediction": [0.2, 0.7, 0.3, 0.6],
        }
    )
    run_b = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["0001", "0001", "0002", "0002"],
            "target_ender_20": [0.1, 0.9, 0.2, 0.8],
            "prediction": [0.3, 0.6, 0.25, 0.75],
        }
    )
    _write_run_predictions(store_root, "run-a", run_a)
    _write_run_predictions(store_root, "run-b", run_b)

    created = api_module.ensemble_build(
        api_module.EnsembleBuildRequest(
            run_ids=["run-a", "run-b"],
            neutralize_members=True,
            neutralize_final=True,
            neutralizer_path=str(neutralizer_path),
            neutralization_mode="era",
            neutralization_proportion=0.5,
            store_root=str(store_root),
        )
    )

    assert created.status == "completed"
    neutralization_config = created.config.get("neutralization")
    assert isinstance(neutralization_config, dict)
    assert neutralization_config.get("members_enabled") is True
    assert neutralization_config.get("final_enabled") is True
    assert created.artifacts_path is not None

    artifacts_path = Path(created.artifacts_path)
    assert (artifacts_path / "predictions.parquet").is_file()
    assert (artifacts_path / "predictions_pre_neutralization.parquet").is_file()
