from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

import numereng.features.hpo.repo as repo_module
from numereng.features.store import (
    StoreHpoStudyUpsert,
    StoreHpoTrialUpsert,
    init_store_db,
    upsert_hpo_study,
    upsert_hpo_trial,
)


def _insert_raw_study_row(
    *,
    store_root: Path,
    study_id: str,
    status: str = "completed",
    stop_reason: str | None = None,
) -> None:
    init_result = init_store_db(store_root=store_root)
    stamp = datetime.now(UTC).isoformat()
    with sqlite3.connect(init_result.db_path) as conn:
        conn.execute(
            """
            INSERT INTO hpo_studies (
                study_id, experiment_id, study_name, status, metric, direction, n_trials, sampler,
                seed, best_trial_number, best_value, best_run_id, config_json, attempted_trials,
                completed_trials, failed_trials, stop_reason, storage_path, error_message, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                study_id,
                "exp-1",
                "study-a",
                status,
                "bmc_last_200_eras.mean",
                "maximize",
                2,
                "tpe",
                1337,
                1,
                0.12,
                "run-1",
                '{"study_id":"study-a","study_name":"study-a","config_path":"configs/base.json","experiment_id":"exp-1","objective":{"metric":"bmc_last_200_eras.mean","direction":"maximize","neutralization":{"enabled":false,"neutralizer_path":null,"proportion":0.5,"mode":"era","neutralizer_cols":null,"rank_output":true}},"search_space":{"model.params.learning_rate":{"type":"float","low":0.001,"high":0.1,"step":null,"log":false,"choices":null}},"sampler":{"kind":"tpe","seed":1337,"n_startup_trials":10,"multivariate":true,"group":false},"stopping":{"max_trials":2,"max_completed_trials":null,"timeout_seconds":null,"plateau":{"enabled":false,"min_completed_trials":15,"patience_completed_trials":10,"min_improvement_abs":0.00025}}}',  # noqa: E501
                2,
                1,
                1,
                stop_reason,
                str(store_root.parent / "experiments" / "exp-1" / "hpo" / study_id),
                None,
                stamp,
                stamp,
            ),
        )
        conn.commit()


def test_get_study_raises_for_invalid_persisted_status(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    _insert_raw_study_row(store_root=store_root, study_id="study-invalid-status", status="unknown")

    with pytest.raises(ValueError, match="hpo_study_status_invalid"):
        repo_module.get_study(store_root=store_root, study_id="study-invalid-status")


def test_list_studies_raises_for_invalid_stop_reason(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    _insert_raw_study_row(store_root=store_root, study_id="study-invalid-stop-reason", stop_reason="mystery")

    with pytest.raises(ValueError, match="hpo_stop_reason_invalid"):
        repo_module.list_studies(store_root=store_root)


def test_list_trials_raises_for_invalid_trial_status(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    upsert_hpo_study(
        store_root=store_root,
        study=StoreHpoStudyUpsert(
            study_id="study-valid",
            experiment_id="exp-1",
            study_name="study-valid",
            status="completed",
            metric="bmc_last_200_eras.mean",
            direction="maximize",
            n_trials=2,
            sampler="tpe",
            seed=1337,
            best_trial_number=0,
            best_value=0.1,
            best_run_id="run-1",
            config_json="{}",
            attempted_trials=1,
            completed_trials=1,
            failed_trials=0,
            storage_path=str(store_root.parent / "experiments" / "exp-1" / "hpo" / "study-valid"),
        ),
    )
    upsert_hpo_trial(
        store_root=store_root,
        trial=StoreHpoTrialUpsert(
            study_id="study-valid",
            trial_number=0,
            status="mystery-status",
            value=0.1,
            run_id="run-1",
            config_path=str(store_root / "tmp.json"),
            params_json="{}",
            error_message=None,
            started_at="2026-02-22T00:00:00+00:00",
            finished_at="2026-02-22T00:01:00+00:00",
        ),
    )

    with pytest.raises(ValueError, match="hpo_trial_status_invalid"):
        repo_module.list_trials(store_root=store_root, study_id="study-valid")
