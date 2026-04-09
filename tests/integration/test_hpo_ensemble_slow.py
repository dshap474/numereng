from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import numereng.api as api_module
import numereng.features.hpo.service as hpo_service_module
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

    predictions_path = predictions_dir / "predictions.parquet"
    frame.to_parquet(predictions_path, index=False)

    manifest = {
        "run_id": run_id,
        "artifacts": {
            "predictions": "artifacts/predictions/predictions.parquet",
        },
    }
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


@pytest.mark.integration
@pytest.mark.slow
def test_hpo_slow_high_trial_count_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    config_path = tmp_path / "base.json"
    _write_base_config(config_path)

    n_trials = 25
    metric_values = [0.05 + (idx * 0.002) for idx in range(n_trials)]
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
        best_trial = None
        best_value = None
        for idx in range(n_trials):
            value = objective_callback(idx, {"model.params.learning_rate": 0.01 + (idx * 0.001)})
            if best_value is None or value > best_value:
                best_value = value
                best_trial = idx
        return best_trial, best_value

    monkeypatch.setattr(hpo_service_module, "run_training", fake_run_training)
    monkeypatch.setattr(hpo_service_module, "run_optuna_study", fake_run_optuna_study)
    monkeypatch.setattr(hpo_service_module, "index_run", lambda **kwargs: None)

    created = api_module.hpo_create(
        api_module.HpoStudyCreateRequest(
            study_name="slow-integration",
            config_path=str(config_path),
            experiment_id="exp-slow",
            n_trials=n_trials,
            search_space={
                "model.params.learning_rate": {
                    "type": "float",
                    "low": 0.001,
                    "high": 0.2,
                }
            },
            workspace_root=str(store_root.parent),
        )
    )
    assert created.status == "completed"
    assert created.best_trial_number == n_trials - 1
    assert created.best_value == pytest.approx(metric_values[-1])
    assert created.storage_path is not None

    trials = api_module.hpo_trials(
        api_module.HpoStudyTrialsRequest(
            study_id=created.study_id,
            workspace_root=str(store_root.parent),
        )
    )
    assert len(trials.trials) == n_trials
    assert trials.trials[-1].trial_number == n_trials - 1

    parquet_path = Path(created.storage_path) / "trials_live.parquet"
    assert parquet_path.is_file()
    frame = pd.read_parquet(parquet_path)
    assert len(frame.index) == n_trials


@pytest.mark.integration
@pytest.mark.slow
def test_ensemble_slow_large_matrix_optimize_smoke(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    eras = [f"{index:04d}" for index in range(1, 21)]

    rows: list[dict[str, object]] = []
    for era_idx, era in enumerate(eras):
        for item_idx in range(25):
            base = float(item_idx) / 25.0
            target = base + (float(era_idx) * 0.01)
            rows.append(
                {
                    "era": era,
                    "id": f"{era}-{item_idx:03d}",
                    "target_ender_20": target,
                }
            )
    base_frame = pd.DataFrame(rows)

    for run_idx in range(4):
        frame = base_frame.copy()
        frame["prediction"] = frame["target_ender_20"] + (run_idx * 0.005) + (frame.index % 7) * 0.0001
        _write_run_predictions(store_root, f"run-{run_idx}", frame)

    created = api_module.ensemble_build(
        api_module.EnsembleBuildRequest(
            run_ids=[f"run-{idx}" for idx in range(4)],
            experiment_id="exp-slow",
            metric="corr_sharpe",
            target="target_ender_20",
            optimize_weights=True,
            workspace_root=str(store_root.parent),
        )
    )
    assert created.status == "completed"
    assert len(created.components) == 4
    assert created.artifacts_path is not None

    weights = [component.weight for component in created.components]
    assert all(weight >= 0.0 for weight in weights)
    assert sum(weights) == pytest.approx(1.0)

    artifacts_path = Path(created.artifacts_path)
    assert (artifacts_path / "predictions.parquet").is_file()
    assert (artifacts_path / "correlation_matrix.parquet").is_file()
    assert (artifacts_path / "metrics.json").is_file()
    assert (artifacts_path / "weights.parquet").is_file()
    assert (artifacts_path / "component_metrics.parquet").is_file()
    assert (artifacts_path / "era_metrics.parquet").is_file()
    assert (artifacts_path / "regime_metrics.parquet").is_file()
    assert (artifacts_path / "lineage.json").is_file()

    loaded = api_module.ensemble_get(
        api_module.EnsembleGetRequest(
            ensemble_id=created.ensemble_id,
            workspace_root=str(store_root.parent),
        )
    )
    assert loaded.ensemble_id == created.ensemble_id
    assert len(loaded.components) == 4
