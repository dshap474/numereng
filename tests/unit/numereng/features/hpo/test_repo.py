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
    direction: str = "maximize",
    sampler: str = "tpe",
) -> None:
    init_result = init_store_db(store_root=store_root)
    stamp = datetime.now(UTC).isoformat()
    with sqlite3.connect(init_result.db_path) as conn:
        conn.execute(
            """
            INSERT INTO hpo_studies (
                study_id, experiment_id, study_name, status, metric, direction, n_trials, sampler,
                seed, best_trial_number, best_value, best_run_id, config_json, storage_path,
                error_message, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                study_id,
                "exp-1",
                "study-a",
                status,
                "bmc_last_200_eras.mean",
                direction,
                2,
                sampler,
                1337,
                1,
                0.12,
                "run-1",
                "{}",
                str(store_root / "experiments" / "exp-1" / "hpo" / study_id),
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


def test_list_studies_raises_for_invalid_persisted_direction(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    _insert_raw_study_row(store_root=store_root, study_id="study-invalid-direction", direction="sideways")

    with pytest.raises(ValueError, match="hpo_direction_invalid"):
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
            storage_path=str(store_root / "experiments" / "exp-1" / "hpo" / "study-valid"),
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
