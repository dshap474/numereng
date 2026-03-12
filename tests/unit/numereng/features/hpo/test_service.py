from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import numereng.features.hpo.service as service_module
from numereng.features.hpo import HpoStudyCreateRequest
from numereng.features.hpo.runner_optuna import HpoOptunaError
from numereng.features.store import StoreError
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


def test_create_study_runs_trials_and_persists_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        _ = (output_dir, client, engine_mode, window_size_eras, embargo_eras, experiment_id)
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
        objective_callback: Any,
    ) -> tuple[int | None, float | None]:
        _ = (direction, sampler, seed, specs)
        assert n_trials == 2
        value_0 = objective_callback(0, {"model.params.learning_rate": 0.01})
        value_1 = objective_callback(1, {"model.params.learning_rate": 0.02})
        assert value_1 > value_0
        return 1, value_1

    monkeypatch.setattr(service_module, "run_training", fake_run_training)
    monkeypatch.setattr(service_module, "run_optuna_study", fake_run_optuna_study)
    indexed_calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        service_module,
        "index_run",
        lambda *, store_root, run_id: indexed_calls.append((Path(store_root), run_id)),
    )

    result = service_module.create_study(
        store_root=store_root,
        request=HpoStudyCreateRequest(
            study_name="lgbm-search",
            experiment_id="2026-02-22_test-exp",
            config_path=config_path,
            n_trials=2,
            search_space={
                "model.params.learning_rate": {
                    "type": "float",
                    "low": 0.001,
                    "high": 0.1,
                    "log": True,
                }
            },
        ),
    )

    assert result.status == "completed"
    assert result.best_trial_number == 1
    assert result.best_value == pytest.approx(0.13)
    assert result.best_run_id == "run-1"
    assert len(result.trials) == 2

    loaded = service_module.get_study_view(store_root=store_root, study_id=result.study_id)
    assert loaded.best_run_id == "run-1"
    assert loaded.best_value == pytest.approx(0.13)

    trial_rows = service_module.get_study_trials_view(store_root=store_root, study_id=result.study_id)
    assert len(trial_rows) == 2
    assert trial_rows[0].trial_number == 0
    assert trial_rows[1].trial_number == 1

    assert (result.storage_path / "trials_live.csv").is_file()
    assert indexed_calls == [(store_root.resolve(), "run-0"), (store_root.resolve(), "run-1")]


def test_get_study_raises_for_missing_id(tmp_path: Path) -> None:
    with pytest.raises(service_module.HpoNotFoundError, match="hpo_study_not_found"):
        service_module.get_study_view(store_root=tmp_path / ".numereng", study_id="missing-study")


def test_create_study_rejects_invalid_study_name(tmp_path: Path) -> None:
    config_path = tmp_path / "base.json"
    _write_base_config(config_path)

    with pytest.raises(service_module.HpoValidationError, match="hpo_study_name_invalid"):
        service_module.create_study(
            store_root=tmp_path / ".numereng",
            request=HpoStudyCreateRequest(
                study_name="   ",
                config_path=config_path,
            ),
        )


def test_create_study_rejects_invalid_experiment_id(tmp_path: Path) -> None:
    config_path = tmp_path / "base.json"
    _write_base_config(config_path)

    with pytest.raises(service_module.HpoValidationError, match="hpo_experiment_id_invalid"):
        service_module.create_study(
            store_root=tmp_path / ".numereng",
            request=HpoStudyCreateRequest(
                study_name="study-a",
                config_path=config_path,
                experiment_id="invalid/id",
            ),
        )


def test_create_study_rejects_missing_config_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"
    with pytest.raises(service_module.HpoValidationError, match="hpo_config_not_found"):
        service_module.create_study(
            store_root=tmp_path / ".numereng",
            request=HpoStudyCreateRequest(
                study_name="study-a",
                config_path=missing_path,
            ),
        )


def test_create_study_fails_fast_on_invalid_neutralizer_path(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    config_path = tmp_path / "base.json"
    _write_base_config(config_path)

    with pytest.raises(service_module.HpoValidationError, match="neutralization_neutralizer_file_not_found"):
        service_module.create_study(
            store_root=store_root,
            request=HpoStudyCreateRequest(
                study_name="bad-neutralizer",
                experiment_id="exp-1",
                config_path=config_path,
                neutralize=True,
                neutralizer_path=tmp_path / "missing_neutralizers.parquet",
            ),
        )


def test_create_study_marks_failed_when_all_trials_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    config_path = tmp_path / "base.json"
    _write_base_config(config_path)

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
        raise RuntimeError("boom")

    def fake_run_optuna_study(
        *,
        direction: str,
        n_trials: int,
        sampler: str,
        seed: int | None,
        specs: tuple[object, ...],
        objective_callback: Any,
    ) -> tuple[int | None, float | None]:
        _ = (direction, sampler, seed, specs)
        for trial_number in range(n_trials):
            try:
                objective_callback(trial_number, {"model.params.learning_rate": 0.01 + float(trial_number) * 0.01})
            except Exception:
                continue
        return None, None

    monkeypatch.setattr(service_module, "run_training", fake_run_training)
    monkeypatch.setattr(service_module, "run_optuna_study", fake_run_optuna_study)
    monkeypatch.setattr(service_module, "index_run", lambda **kwargs: None)

    result = service_module.create_study(
        store_root=store_root,
        request=HpoStudyCreateRequest(
            study_name="all-fail",
            experiment_id="exp-1",
            config_path=config_path,
            n_trials=2,
            search_space={
                "model.params.learning_rate": {
                    "type": "float",
                    "low": 0.001,
                    "high": 0.1,
                }
            },
        ),
    )

    assert result.status == "failed"
    assert result.best_trial_number is None
    assert result.best_value is None
    assert result.best_run_id is None
    assert len(result.trials) == 2
    assert all(trial.status == "failed" for trial in result.trials)

    loaded = service_module.get_study_view(store_root=store_root, study_id=result.study_id)
    assert loaded.status == "failed"
    assert loaded.error_message == "hpo_trials_all_failed"


def test_create_study_uses_minimize_fallback_when_optuna_returns_no_best(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"
    config_path = tmp_path / "base.json"
    _write_base_config(config_path)

    metric_values = [0.5, 0.2]
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
        return TrainingRunResult(run_id=run_id, predictions_path=predictions_path, results_path=results_path)

    def fake_run_optuna_study(
        *,
        direction: str,
        n_trials: int,
        sampler: str,
        seed: int | None,
        specs: tuple[object, ...],
        objective_callback: Any,
    ) -> tuple[int | None, float | None]:
        _ = (direction, sampler, seed, specs)
        for trial_number in range(n_trials):
            objective_callback(trial_number, {"model.params.learning_rate": 0.01 + float(trial_number) * 0.01})
        return None, None

    monkeypatch.setattr(service_module, "run_training", fake_run_training)
    monkeypatch.setattr(service_module, "run_optuna_study", fake_run_optuna_study)
    monkeypatch.setattr(service_module, "index_run", lambda **kwargs: None)

    result = service_module.create_study(
        store_root=store_root,
        request=HpoStudyCreateRequest(
            study_name="minimize-fallback",
            experiment_id="exp-1",
            config_path=config_path,
            n_trials=2,
            direction="minimize",
            search_space={
                "model.params.learning_rate": {
                    "type": "float",
                    "low": 0.001,
                    "high": 0.1,
                }
            },
        ),
    )

    assert result.status == "completed"
    assert result.best_trial_number == 1
    assert result.best_value == pytest.approx(0.2)
    assert result.best_run_id == "run-1"


def test_create_study_translates_optuna_dependency_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    config_path = tmp_path / "base.json"
    _write_base_config(config_path)

    def fake_run_optuna_study(
        *,
        direction: str,
        n_trials: int,
        sampler: str,
        seed: int | None,
        specs: tuple[object, ...],
        objective_callback: Any,
    ) -> tuple[int | None, float | None]:
        _ = (direction, n_trials, sampler, seed, specs, objective_callback)
        raise HpoOptunaError("hpo_dependency_missing_optuna")

    monkeypatch.setattr(service_module, "run_optuna_study", fake_run_optuna_study)

    with pytest.raises(service_module.HpoDependencyError, match="hpo_dependency_missing_optuna"):
        service_module.create_study(
            store_root=store_root,
            request=HpoStudyCreateRequest(
                study_name="dep-missing",
                experiment_id="exp-1",
                config_path=config_path,
            ),
        )

    studies = service_module.list_studies_view(store_root=store_root, experiment_id="exp-1")
    assert len(studies) == 1
    assert studies[0].status == "failed"
    assert studies[0].error_message == "hpo_dependency_missing_optuna"


def test_extract_metric_value_raises_for_missing_metric(tmp_path: Path) -> None:
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps({"metrics": {"corr_mean": 0.12}}), encoding="utf-8")

    with pytest.raises(service_module.HpoExecutionError, match="hpo_metric_not_found"):
        service_module._extract_metric_value(results_path=results_path, metric="bmc_last_200_eras.mean")


def test_extract_metric_value_raises_for_invalid_json(tmp_path: Path) -> None:
    results_path = tmp_path / "results.json"
    results_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(service_module.HpoExecutionError, match="hpo_results_invalid"):
        service_module._extract_metric_value(results_path=results_path, metric="bmc_last_200_eras.mean")


def test_extract_metric_value_from_predictions_surfaces_scoring_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "base.json"
    _write_base_config(config_path)
    predictions_path = tmp_path / "preds.parquet"
    predictions_path.write_bytes(b"PAR1")

    def _raise_scoring_error(**kwargs: object) -> object:
        _ = kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr(service_module, "run_post_training_scoring", _raise_scoring_error)

    class _ScoringClient:
        def download_dataset(
            self,
            filename: str,
            *,
            dest_path: str | None = None,
            round_num: int | None = None,
        ) -> str:
            _ = (filename, round_num)
            return dest_path or filename

    with pytest.raises(service_module.HpoExecutionError, match="hpo_neutralized_metric_compute_failed:boom"):
        service_module._extract_metric_value_from_predictions(
            predictions_path=predictions_path,
            trial_config_path=config_path,
            metric="metrics.bmc_last_200_eras.mean",
            scoring_client=_ScoringClient(),
        )


def test_create_study_marks_trial_failed_when_indexing_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"
    config_path = tmp_path / "base.json"
    _write_base_config(config_path)

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
        _ = (config_path, client, engine_mode, window_size_eras, embargo_eras, experiment_id)
        assert output_dir is not None
        assert Path(output_dir) == store_root.resolve()
        results_path = tmp_path / "results_0.json"
        results_path.write_text(json.dumps({"metrics": {"bmc_last_200_eras": {"mean": 0.11}}}), encoding="utf-8")
        predictions_path = tmp_path / "preds_0.parquet"
        predictions_path.write_bytes(b"PAR1")
        return TrainingRunResult(run_id="run-0", predictions_path=predictions_path, results_path=results_path)

    def fake_run_optuna_study(
        *,
        direction: str,
        n_trials: int,
        sampler: str,
        seed: int | None,
        specs: tuple[object, ...],
        objective_callback: Any,
    ) -> tuple[int | None, float | None]:
        _ = (direction, n_trials, sampler, seed, specs)
        try:
            objective_callback(0, {"model.params.learning_rate": 0.01})
        except Exception:
            pass
        return None, None

    monkeypatch.setattr(service_module, "run_training", fake_run_training)
    monkeypatch.setattr(service_module, "run_optuna_study", fake_run_optuna_study)
    monkeypatch.setattr(
        service_module,
        "index_run",
        lambda **kwargs: (_ for _ in ()).throw(StoreError("store_failed")),
    )

    result = service_module.create_study(
        store_root=store_root,
        request=HpoStudyCreateRequest(
            study_name="index-fail",
            experiment_id="exp-1",
            config_path=config_path,
            n_trials=1,
            search_space={
                "model.params.learning_rate": {
                    "type": "float",
                    "low": 0.001,
                    "high": 0.1,
                }
            },
        ),
    )

    assert result.status == "failed"
    assert result.trials[0].status == "failed"
    assert result.trials[0].error_message == "hpo_trial_index_failed:0"
    assert result.trials[0].run_id == "run-0"
