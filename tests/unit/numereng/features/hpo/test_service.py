from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

import numereng.features.hpo.service as service_module
from numereng.features.hpo import (
    HpoObjectiveSpec,
    HpoPlateauSpec,
    HpoSamplerSpec,
    HpoStoppingSpec,
    HpoStudyCreateRequest,
)
from numereng.features.hpo.runner_optuna import HpoOptunaError, HpoOptunaStudyResult
from numereng.features.store import StoreError
from numereng.features.training import TrainingRunPreview, TrainingRunResult


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
                        "profile": "purged_walk_forward",
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_post_fold_snapshots(store_root: Path, *, run_id: str, objective_value: float) -> None:
    import pandas as pd

    scoring_dir = store_root / "runs" / run_id / "artifacts" / "scoring"
    scoring_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "run_id": run_id,
                "config_hash": "cfg",
                "seed": None,
                "target_col": "target_ender_20",
                "payout_target_col": "target_ender_20",
                "cv_fold": 0,
                "corr_native_fold_mean": 0.0,
                "corr_ender20_fold_mean": 0.0,
                "bmc_fold_mean": objective_value / 2.25,
            }
        ]
    ).to_parquet(scoring_dir / "post_fold_snapshots.parquet", index=False)


def _study_request(
    *,
    study_id: str,
    config_path: Path,
    experiment_id: str | None = "exp-1",
    max_trials: int = 2,
    objective: HpoObjectiveSpec | None = None,
    stopping: HpoStoppingSpec | None = None,
    search_space: dict[str, dict[str, Any]] | None = None,
) -> HpoStudyCreateRequest:
    return HpoStudyCreateRequest(
        study_id=study_id,
        study_name=study_id,
        config_path=config_path,
        experiment_id=experiment_id,
        objective=objective or HpoObjectiveSpec(),
        search_space=search_space
        or {
            "model.params.learning_rate": {
                "type": "float",
                "low": 0.001,
                "high": 0.1,
                "log": True,
            }
        },
        sampler=HpoSamplerSpec(kind="tpe"),
        stopping=stopping or HpoStoppingSpec(max_trials=max_trials, plateau=HpoPlateauSpec(enabled=False)),
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
        _write_post_fold_snapshots(store_root, run_id=run_id, objective_value=metric_values[idx])
        return TrainingRunResult(run_id=run_id, predictions_path=predictions_path, results_path=results_path)

    def fake_run_optuna_study(
        *,
        study_id: str,
        storage_path: Path,
        direction: str,
        sampler: object,
        stopping: HpoStoppingSpec,
        specs: tuple[object, ...],
        objective_callback: Any,
        summary_callback: Any | None = None,
    ) -> HpoOptunaStudyResult:
        _ = (study_id, storage_path, direction, sampler, specs, summary_callback)
        assert stopping.max_trials == 2
        value_0 = objective_callback(0, {"model.params.learning_rate": 0.01})
        value_1 = objective_callback(1, {"model.params.learning_rate": 0.02})
        assert value_1 > value_0
        return HpoOptunaStudyResult(
            best_trial_number=1,
            best_value=value_1,
            attempted_trials=2,
            completed_trials=2,
            failed_trials=0,
            stop_reason="max_trials_reached",
        )

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
        request=_study_request(study_id="study-a", config_path=config_path),
    )

    assert result.status == "completed"
    assert result.best_trial_number == 1
    assert result.best_value == pytest.approx(0.13)
    assert result.best_run_id == "run-1"
    assert result.attempted_trials == 2
    assert result.completed_trials == 2
    assert result.failed_trials == 0
    assert result.stop_reason == "max_trials_reached"
    assert len(result.trials) == 2
    assert (result.storage_path / "study_spec.json").is_file()
    assert (result.storage_path / "study_summary.json").is_file()
    assert (result.storage_path / "optuna_journal.log").parent == result.storage_path

    loaded = service_module.get_study_view(store_root=store_root, study_id=result.study_id)
    assert loaded.best_run_id == "run-1"
    assert loaded.best_value == pytest.approx(0.13)
    assert loaded.completed_trials == 2
    assert loaded.stop_reason == "max_trials_reached"

    trial_rows = service_module.get_study_trials_view(store_root=store_root, study_id=result.study_id)
    assert len(trial_rows) == 2
    assert trial_rows[0].trial_number == 0
    assert trial_rows[1].trial_number == 1

    trials_path = result.storage_path / "trials_live.parquet"
    assert trials_path.is_file()
    assert pq.ParquetFile(trials_path).metadata.row_group(0).column(0).compression == "ZSTD"
    assert indexed_calls == [(store_root.resolve(), "run-0"), (store_root.resolve(), "run-1")]


def test_create_study_rejects_invalid_experiment_id(tmp_path: Path) -> None:
    config_path = tmp_path / "base.json"
    _write_base_config(config_path)

    with pytest.raises(service_module.HpoValidationError, match="hpo_experiment_id_invalid"):
        service_module.create_study(
            store_root=tmp_path / ".numereng",
            request=_study_request(
                study_id="study-a",
                config_path=config_path,
                experiment_id="invalid/id",
            ),
        )


def test_create_study_rejects_spec_mismatch_on_resume(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps({"metrics": {"bmc_last_200_eras": {"mean": 0.11}}}), encoding="utf-8")
        predictions_path = tmp_path / "preds.parquet"
        predictions_path.write_bytes(b"PAR1")
        _write_post_fold_snapshots(store_root, run_id="run-0", objective_value=0.11)
        return TrainingRunResult(run_id="run-0", predictions_path=predictions_path, results_path=results_path)

    def fake_run_optuna_study(**kwargs: Any) -> HpoOptunaStudyResult:
        kwargs["objective_callback"](0, {"model.params.learning_rate": 0.01})
        return HpoOptunaStudyResult(
            best_trial_number=0,
            best_value=0.11,
            attempted_trials=1,
            completed_trials=1,
            failed_trials=0,
            stop_reason="max_trials_reached",
        )

    monkeypatch.setattr(service_module, "run_training", fake_run_training)
    monkeypatch.setattr(service_module, "run_optuna_study", fake_run_optuna_study)
    monkeypatch.setattr(service_module, "index_run", lambda **kwargs: None)

    service_module.create_study(
        store_root=store_root,
        request=_study_request(study_id="study-a", config_path=config_path, max_trials=1),
    )

    with pytest.raises(service_module.HpoValidationError, match="hpo_study_spec_mismatch:study-a"):
        service_module.create_study(
            store_root=store_root,
            request=_study_request(
                study_id="study-a",
                config_path=config_path,
                max_trials=2,
                search_space={
                    "model.params.learning_rate": {
                        "type": "float",
                        "low": 0.002,
                        "high": 0.2,
                    }
                },
            ),
        )


def test_create_study_resumes_existing_study_when_budget_expands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"
    config_path = tmp_path / "base.json"
    _write_base_config(config_path)
    call_state = {"count": 0}

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
        idx = call_state["count"]
        run_id = f"run-{idx}"
        results_path = tmp_path / f"results_{idx}.json"
        results_path.write_text(
            json.dumps({"metrics": {"bmc_last_200_eras": {"mean": 0.10 + (0.01 * idx)}}}),
            encoding="utf-8",
        )
        predictions_path = tmp_path / f"preds_{idx}.parquet"
        predictions_path.write_bytes(b"PAR1")
        _write_post_fold_snapshots(store_root, run_id=run_id, objective_value=0.10 + (0.01 * idx))
        return TrainingRunResult(run_id=run_id, predictions_path=predictions_path, results_path=results_path)

    def fake_run_optuna_study(**kwargs: Any) -> HpoOptunaStudyResult:
        callback = kwargs["objective_callback"]
        if call_state["count"] == 0:
            callback(0, {"model.params.learning_rate": 0.01})
            call_state["count"] = 1
            return HpoOptunaStudyResult(
                best_trial_number=0,
                best_value=0.10,
                attempted_trials=1,
                completed_trials=1,
                failed_trials=0,
                stop_reason="max_trials_reached",
            )
        callback(1, {"model.params.learning_rate": 0.02})
        call_state["count"] = 2
        return HpoOptunaStudyResult(
            best_trial_number=1,
            best_value=0.11,
            attempted_trials=2,
            completed_trials=2,
            failed_trials=0,
            stop_reason="max_trials_reached",
        )

    monkeypatch.setattr(service_module, "run_training", fake_run_training)
    monkeypatch.setattr(service_module, "run_optuna_study", fake_run_optuna_study)
    monkeypatch.setattr(service_module, "index_run", lambda **kwargs: None)

    first = service_module.create_study(
        store_root=store_root,
        request=_study_request(study_id="study-resume", config_path=config_path, max_trials=1),
    )
    second = service_module.create_study(
        store_root=store_root,
        request=_study_request(study_id="study-resume", config_path=config_path, max_trials=2),
    )

    assert first.completed_trials == 1
    assert len(first.trials) == 1
    assert second.completed_trials == 2
    assert len(second.trials) == 2
    assert second.best_trial_number == 1
    assert second.best_run_id == "run-1"


def test_create_study_marks_failed_when_all_trials_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    config_path = tmp_path / "base.json"
    _write_base_config(config_path)

    def fake_run_training(**kwargs: Any) -> TrainingRunResult:
        _ = kwargs
        raise RuntimeError("boom")

    def fake_run_optuna_study(**kwargs: Any) -> HpoOptunaStudyResult:
        callback = kwargs["objective_callback"]
        for trial_number in range(2):
            try:
                callback(trial_number, {"model.params.learning_rate": 0.01 + (0.01 * trial_number)})
            except Exception:
                continue
        return HpoOptunaStudyResult(
            best_trial_number=None,
            best_value=None,
            attempted_trials=2,
            completed_trials=0,
            failed_trials=2,
            stop_reason="all_trials_failed",
        )

    monkeypatch.setattr(service_module, "run_training", fake_run_training)
    monkeypatch.setattr(service_module, "run_optuna_study", fake_run_optuna_study)
    monkeypatch.setattr(service_module, "index_run", lambda **kwargs: None)

    result = service_module.create_study(
        store_root=store_root,
        request=_study_request(study_id="study-fail", config_path=config_path, max_trials=2),
    )

    assert result.status == "failed"
    assert result.stop_reason == "all_trials_failed"
    assert result.error_message == "hpo_trials_all_failed"
    assert len(result.trials) == 2
    assert all(trial.status == "failed" for trial in result.trials)


def test_create_study_translates_optuna_dependency_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    config_path = tmp_path / "base.json"
    _write_base_config(config_path)

    monkeypatch.setattr(
        service_module,
        "run_optuna_study",
        lambda **kwargs: (_ for _ in ()).throw(HpoOptunaError("hpo_dependency_missing_optuna")),
    )

    with pytest.raises(service_module.HpoDependencyError, match="hpo_dependency_missing_optuna"):
        service_module.create_study(
            store_root=store_root,
            request=_study_request(study_id="study-dep", config_path=config_path, max_trials=1),
        )

    studies = service_module.list_studies_view(store_root=store_root, experiment_id="exp-1")
    assert len(studies) == 1
    assert studies[0].status == "failed"
    assert studies[0].error_message == "hpo_dependency_missing_optuna"


def test_extract_post_fold_objective_value_accepts_legacy_bmc_column(tmp_path: Path) -> None:
    import pandas as pd

    store_root = tmp_path / ".numereng"
    scoring_dir = store_root / "runs" / "run-legacy" / "artifacts" / "scoring"
    scoring_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "run_id": "run-legacy",
                "config_hash": "cfg",
                "seed": None,
                "target_col": "target_ender_20",
                "payout_target_col": "target_ender_20",
                "cv_fold": 0,
                "corr_native_fold_mean": 0.0,
                "corr_ender20_fold_mean": 0.0,
                "bmc_ender20_fold_mean": 0.45 / 2.25,
            }
        ]
    ).to_parquet(scoring_dir / "post_fold_snapshots.parquet", index=False)

    value = service_module._extract_post_fold_objective_value(store_root=store_root, run_id="run-legacy")

    assert value == pytest.approx(0.45)


def test_extract_post_fold_objective_value_accepts_generic_snapshot_columns(tmp_path: Path) -> None:
    import pandas as pd

    store_root = tmp_path / ".numereng"
    scoring_dir = store_root / "runs" / "run-generic" / "artifacts" / "scoring"
    scoring_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "run_id": "run-generic",
                "config_hash": "cfg",
                "seed": None,
                "target_col": "target",
                "payout_target_col": "target",
                "cv_fold": 0,
                "corr_native_fold_mean": 0.2,
                "bmc_fold_mean": 0.1,
            }
        ]
    ).to_parquet(scoring_dir / "post_fold_snapshots.parquet", index=False)

    value = service_module._extract_post_fold_objective_value(store_root=store_root, run_id="run-generic")

    assert value == pytest.approx((0.25 * 0.2) + (2.25 * 0.1))


def test_extract_post_fold_objective_value_falls_back_to_results_when_snapshot_missing(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "corr": {"mean": 0.2},
                    "bmc_last_200_eras": {"mean": 0.1},
                }
            }
        ),
        encoding="utf-8",
    )

    value = service_module._extract_post_fold_objective_value(
        store_root=store_root,
        run_id="run-results",
        results_path=results_path,
    )

    assert value == pytest.approx((0.25 * 0.2) + (2.25 * 0.1))


def test_create_study_marks_trial_failed_when_indexing_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"
    config_path = tmp_path / "base.json"
    _write_base_config(config_path)

    def fake_run_training(**kwargs: Any) -> TrainingRunResult:
        _ = kwargs
        results_path = tmp_path / "results_0.json"
        results_path.write_text(json.dumps({"metrics": {"bmc_last_200_eras": {"mean": 0.11}}}), encoding="utf-8")
        predictions_path = tmp_path / "preds_0.parquet"
        predictions_path.write_bytes(b"PAR1")
        return TrainingRunResult(run_id="run-0", predictions_path=predictions_path, results_path=results_path)

    def fake_run_optuna_study(**kwargs: Any) -> HpoOptunaStudyResult:
        try:
            kwargs["objective_callback"](0, {"model.params.learning_rate": 0.01})
        except Exception:
            pass
        return HpoOptunaStudyResult(
            best_trial_number=None,
            best_value=None,
            attempted_trials=1,
            completed_trials=0,
            failed_trials=1,
            stop_reason="all_trials_failed",
        )

    monkeypatch.setattr(service_module, "run_training", fake_run_training)
    monkeypatch.setattr(service_module, "run_optuna_study", fake_run_optuna_study)
    monkeypatch.setattr(
        service_module,
        "index_run",
        lambda **kwargs: (_ for _ in ()).throw(StoreError("store_failed")),
    )

    result = service_module.create_study(
        store_root=store_root,
        request=_study_request(study_id="study-index-fail", config_path=config_path, max_trials=1),
    )

    assert result.status == "failed"
    assert result.trials[0].status == "failed"
    assert result.trials[0].error_message == "hpo_trial_index_failed:0"
    assert result.trials[0].run_id == "run-0"


def test_create_study_reuses_completed_trial_for_identical_params(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"
    config_path = tmp_path / "base.json"
    _write_base_config(config_path)
    calls = {"run_training": 0}

    def fake_run_training(**kwargs: Any) -> TrainingRunResult:
        _ = kwargs
        calls["run_training"] += 1
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps({"metrics": {"bmc_last_200_eras": {"mean": 0.11}}}), encoding="utf-8")
        predictions_path = tmp_path / "preds.parquet"
        predictions_path.write_bytes(b"PAR1")
        _write_post_fold_snapshots(store_root, run_id="run-0", objective_value=0.11)
        return TrainingRunResult(run_id="run-0", predictions_path=predictions_path, results_path=results_path)

    def fake_run_optuna_study(**kwargs: Any) -> HpoOptunaStudyResult:
        callback = kwargs["objective_callback"]
        first_value = callback(0, {"model.params.learning_rate": 0.01})
        second_value = callback(1, {"model.params.learning_rate": 0.01})
        assert second_value == pytest.approx(first_value)
        return HpoOptunaStudyResult(
            best_trial_number=0,
            best_value=first_value,
            attempted_trials=2,
            completed_trials=2,
            failed_trials=0,
            stop_reason="max_trials_reached",
        )

    monkeypatch.setattr(service_module, "run_training", fake_run_training)
    monkeypatch.setattr(service_module, "run_optuna_study", fake_run_optuna_study)
    monkeypatch.setattr(service_module, "index_run", lambda **kwargs: None)

    result = service_module.create_study(
        store_root=store_root,
        request=_study_request(study_id="study-dup-same", config_path=config_path, max_trials=2),
    )

    assert calls["run_training"] == 1
    assert len(result.trials) == 2
    assert result.trials[0].run_id == "run-0"
    assert result.trials[1].run_id == "run-0"
    assert result.trials[1].value == pytest.approx(result.trials[0].value or 0.0)


def test_create_study_reuses_existing_finished_deterministic_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"
    config_path = tmp_path / "base.json"
    _write_base_config(config_path)

    run_id = "run-existing"
    run_dir = store_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps({"status": "FINISHED"}), encoding="utf-8")
    _write_post_fold_snapshots(store_root, run_id=run_id, objective_value=0.14)

    preview = TrainingRunPreview(
        run_id=run_id,
        run_hash="hash-existing",
        config_hash="cfg-existing",
        run_dir=run_dir,
        predictions_path=run_dir / "predictions.parquet",
        results_path=run_dir / "results.json",
        scoring_dir=run_dir / "artifacts" / "scoring",
        run_manifest_path=run_dir / "run.json",
    )

    indexed_calls: list[str] = []

    def fake_run_optuna_study(**kwargs: Any) -> HpoOptunaStudyResult:
        value = kwargs["objective_callback"](0, {"model.params.learning_rate": 0.01})
        return HpoOptunaStudyResult(
            best_trial_number=0,
            best_value=value,
            attempted_trials=1,
            completed_trials=1,
            failed_trials=0,
            stop_reason="max_trials_reached",
        )

    monkeypatch.setattr(service_module, "preview_training_run", lambda **kwargs: preview)
    monkeypatch.setattr(
        service_module,
        "run_training",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("run_training should not be called")),
    )
    monkeypatch.setattr(service_module, "run_optuna_study", fake_run_optuna_study)
    monkeypatch.setattr(service_module, "index_run", lambda **kwargs: indexed_calls.append(kwargs["run_id"]))

    result = service_module.create_study(
        store_root=store_root,
        request=_study_request(study_id="study-reuse-existing", config_path=config_path, max_trials=1),
    )

    assert result.status == "completed"
    assert result.best_run_id == run_id
    assert result.trials[0].run_id == run_id
    assert indexed_calls == [run_id]


def test_create_study_fails_loudly_for_non_reusable_existing_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"
    config_path = tmp_path / "base.json"
    _write_base_config(config_path)

    run_id = "run-stuck"
    run_dir = store_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps({"status": "RUNNING"}), encoding="utf-8")

    preview = TrainingRunPreview(
        run_id=run_id,
        run_hash="hash-stuck",
        config_hash="cfg-stuck",
        run_dir=run_dir,
        predictions_path=run_dir / "predictions.parquet",
        results_path=run_dir / "results.json",
        scoring_dir=run_dir / "artifacts" / "scoring",
        run_manifest_path=run_dir / "run.json",
    )

    def fake_run_optuna_study(**kwargs: Any) -> HpoOptunaStudyResult:
        try:
            kwargs["objective_callback"](0, {"model.params.learning_rate": 0.01})
        except Exception:
            pass
        return HpoOptunaStudyResult(
            best_trial_number=None,
            best_value=None,
            attempted_trials=1,
            completed_trials=0,
            failed_trials=1,
            stop_reason="all_trials_failed",
        )

    monkeypatch.setattr(service_module, "preview_training_run", lambda **kwargs: preview)
    monkeypatch.setattr(
        service_module,
        "run_training",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("run_training should not be called")),
    )
    monkeypatch.setattr(service_module, "run_optuna_study", fake_run_optuna_study)
    monkeypatch.setattr(service_module, "index_run", lambda **kwargs: None)

    result = service_module.create_study(
        store_root=store_root,
        request=_study_request(study_id="study-reuse-stuck", config_path=config_path, max_trials=1),
    )

    assert result.status == "failed"
    assert result.trials[0].status == "failed"
    assert result.trials[0].error_message == (
        "hpo_trial_existing_run_not_reusable:run-stuck:status=running:reset_required"
    )
    assert run_dir.is_dir()
    assert (run_dir / "run.json").is_file()
