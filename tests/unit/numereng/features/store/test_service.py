from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, cast

import pytest

import numereng.features.store.service as store_service
from numereng.features.store import (
    StoreCloudJobUpsert,
    StoreEnsembleComponentUpsert,
    StoreEnsembleMetricUpsert,
    StoreEnsembleUpsert,
    StoreError,
    StoreHpoStudyUpsert,
    StoreHpoTrialUpsert,
    backfill_run_execution,
    doctor_store,
    get_ensemble,
    get_experiment,
    get_hpo_study,
    index_run,
    init_store_db,
    list_ensemble_components,
    list_ensemble_metrics,
    list_ensembles,
    list_experiments,
    list_hpo_studies,
    list_hpo_trials,
    materialize_viz_artifacts,
    rebuild_run_index,
    replace_ensemble_components,
    replace_ensemble_metrics,
    upsert_cloud_job,
    upsert_ensemble,
    upsert_experiment,
    upsert_hpo_study,
    upsert_hpo_trial,
)


def _write_run_manifest(run_dir: Path, *, status: str = "FINISHED") -> None:
    payload = {
        "run_id": run_dir.name,
        "run_hash": f"hash-{run_dir.name}",
        "run_type": "training",
        "status": status,
        "created_at": "2026-02-21T00:00:00+00:00",
        "finished_at": "2026-02-21T00:05:00+00:00" if status == "FINISHED" else None,
        "config": {"hash": "cfg-hash"},
        "artifacts": {
            "resolved_config": "resolved.json",
            "metrics": "metrics.json",
            "results": "results.json",
            "predictions": "artifacts/predictions/preds.parquet",
        },
    }
    (run_dir / "run.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def _write_tmp_remote_config(store_root: Path, *, name: str) -> Path:
    config_path = store_root / "tmp" / "remote-configs" / name
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{}", encoding="utf-8")
    return config_path.resolve()


def _set_path_age_days(path: Path, *, days_old: int) -> None:
    stale_timestamp = store_service.datetime.now(store_service.UTC).timestamp() - (days_old * 24 * 60 * 60)
    os.utime(path, (stale_timestamp, stale_timestamp))


def _insert_run_lifecycle(store_root: Path, *, run_id: str, config_path: Path, status: str) -> None:
    now = "2026-03-28T00:00:00+00:00"
    with sqlite3.connect(store_root / "numereng.db") as conn:
        conn.execute(
            """
            INSERT INTO run_lifecycles (
                run_id,
                run_hash,
                job_id,
                logical_run_id,
                attempt_id,
                attempt_no,
                source,
                operation_type,
                job_type,
                status,
                config_id,
                config_source,
                config_path,
                config_sha256,
                run_dir,
                runtime_path,
                completed_stages_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                f"hash-{run_id}",
                f"job-{run_id}",
                f"logical-{run_id}",
                f"attempt-{run_id}",
                1,
                "cli.experiment.train",
                "train",
                "local_train",
                status,
                f"config-{run_id}",
                "explicit_path",
                str(config_path),
                f"sha-{run_id}",
                str(store_root / "runs" / run_id),
                str(store_root / "runs" / run_id / "runtime.json"),
                "[]",
                now,
                now,
            ),
        )
        conn.commit()


def test_init_store_db_creates_required_tables(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"

    result = init_store_db(store_root=store_root)

    assert result.created is True
    assert result.db_path.exists()
    assert (store_root / "tmp").is_dir()

    with sqlite3.connect(result.db_path) as conn:
        tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()}

    assert "schema_migrations" in tables
    assert "runs" in tables
    assert "metrics" in tables
    assert "run_artifacts" in tables
    assert "run_jobs" in tables
    assert "run_job_events" in tables
    assert "run_job_logs" in tables
    assert "run_job_samples" in tables
    assert "run_lifecycles" in tables
    assert "logical_runs" in tables
    assert "run_attempts" in tables
    assert "cloud_jobs" in tables
    assert "experiments" in tables
    assert "hpo_studies" in tables
    assert "hpo_trials" in tables
    assert "ensembles" in tables
    assert "ensemble_components" in tables
    assert "ensemble_metrics" in tables


def test_upsert_hpo_study_and_trials_roundtrip(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    upsert_hpo_study(
        store_root=store_root,
        study=StoreHpoStudyUpsert(
            study_id="study-1",
            experiment_id="exp-1",
            study_name="Test Study",
            status="running",
            metric="bmc_last_200_eras.mean",
            direction="maximize",
            n_trials=3,
            sampler="tpe",
            seed=42,
            config_json='{"search_space":{}}',
            storage_path=str(store_root.parent / "experiments" / "exp-1" / "hpo" / "study-1"),
        ),
    )
    upsert_hpo_trial(
        store_root=store_root,
        trial=StoreHpoTrialUpsert(
            study_id="study-1",
            trial_number=0,
            status="completed",
            value=0.12,
            run_id="run-1",
            params_json='{"model.params.learning_rate":0.01}',
            config_path="configs/trial_0000.json",
            started_at="2026-02-22T00:00:00+00:00",
            finished_at="2026-02-22T00:10:00+00:00",
        ),
    )
    upsert_hpo_trial(
        store_root=store_root,
        trial=StoreHpoTrialUpsert(
            study_id="study-1",
            trial_number=1,
            status="failed",
            value=None,
            run_id=None,
            params_json='{"model.params.learning_rate":0.02}',
            config_path="configs/trial_0001.json",
            error_message="boom",
        ),
    )

    study = get_hpo_study(store_root=store_root, study_id="study-1")
    assert study is not None
    assert study.study_name == "Test Study"
    assert study.direction == "maximize"

    studies = list_hpo_studies(store_root=store_root, experiment_id="exp-1")
    assert len(studies) == 1
    assert studies[0].study_id == "study-1"

    trials = list_hpo_trials(store_root=store_root, study_id="study-1")
    assert len(trials) == 2
    assert trials[0].trial_number == 0
    assert trials[0].run_id == "run-1"
    assert trials[1].status == "failed"
    assert trials[1].error_message == "boom"


def test_upsert_ensemble_roundtrip(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    upsert_ensemble(
        store_root=store_root,
        ensemble=StoreEnsembleUpsert(
            ensemble_id="ens-1",
            experiment_id="exp-1",
            name="Blend 1",
            method="rank_avg",
            target="target_ender_20",
            metric="corr_sharpe",
            status="completed",
            config_json='{"run_ids":["run-a","run-b"]}',
            artifacts_path=str(store_root.parent / "experiments" / "exp-1" / "ensembles" / "ens-1"),
        ),
    )
    replace_ensemble_components(
        store_root=store_root,
        ensemble_id="ens-1",
        components=(
            StoreEnsembleComponentUpsert(ensemble_id="ens-1", run_id="run-a", weight=0.6, rank=0),
            StoreEnsembleComponentUpsert(ensemble_id="ens-1", run_id="run-b", weight=0.4, rank=1),
        ),
    )
    replace_ensemble_metrics(
        store_root=store_root,
        ensemble_id="ens-1",
        metrics=(
            StoreEnsembleMetricUpsert(ensemble_id="ens-1", name="corr_mean", value=0.11),
            StoreEnsembleMetricUpsert(ensemble_id="ens-1", name="corr_sharpe", value=1.2),
        ),
    )

    ensemble = get_ensemble(store_root=store_root, ensemble_id="ens-1")
    assert ensemble is not None
    assert ensemble.name == "Blend 1"

    listed = list_ensembles(store_root=store_root, experiment_id="exp-1")
    assert len(listed) == 1
    assert listed[0].ensemble_id == "ens-1"

    components = list_ensemble_components(store_root=store_root, ensemble_id="ens-1")
    assert len(components) == 2
    assert components[0].run_id == "run-a"
    assert components[0].weight == 0.6

    metrics = list_ensemble_metrics(store_root=store_root, ensemble_id="ens-1")
    assert len(metrics) == 2
    values = {item.name: item.value for item in metrics}
    assert values["corr_mean"] == 0.11
    assert values["corr_sharpe"] == 1.2


def test_index_run_upserts_run_metrics_and_artifacts(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-1"
    (run_dir / "artifacts" / "predictions").mkdir(parents=True)

    _write_run_manifest(run_dir)
    (run_dir / "resolved.json").write_text('{\n  "foo": "bar"\n}\n')
    (run_dir / "results.json").write_text("{}")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "corr": {"mean": 0.12, "sharpe": 1.5},
                "mmc_mean": 0.03,
                "non_numeric": {"note": "kept_as_json"},
            },
            indent=2,
            sort_keys=True,
        )
    )
    (run_dir / "artifacts" / "predictions" / "preds.parquet").write_bytes(b"PAR1")

    result = index_run(store_root=store_root, run_id="run-1")

    assert result.run_id == "run-1"
    assert result.status == "FINISHED"
    assert result.metrics_indexed >= 4
    assert result.artifacts_indexed >= 5

    with sqlite3.connect(store_root / "numereng.db") as conn:
        run_row = conn.execute("SELECT status, run_path FROM runs WHERE run_id = ?", ("run-1",)).fetchone()
        assert run_row is not None
        assert run_row[0] == "FINISHED"
        assert str(run_dir.resolve()) == str(run_row[1])

        metric_row = conn.execute(
            "SELECT value FROM metrics WHERE run_id = ? AND name = ?",
            ("run-1", "corr.mean"),
        ).fetchone()
        assert metric_row is not None
        assert metric_row[0] == 0.12

        metric_json_row = conn.execute(
            "SELECT value_json FROM metrics WHERE run_id = ? AND name = ?",
            ("run-1", "corr"),
        ).fetchone()
        assert metric_json_row is not None
        assert isinstance(metric_json_row[0], str)

        artifact_row = conn.execute(
            "SELECT exists_flag, relative_path FROM run_artifacts WHERE run_id = ? AND kind = ?",
            ("run-1", "predictions"),
        ).fetchone()
        assert artifact_row is not None
        assert artifact_row[0] == 1
        assert artifact_row[1] == "artifacts/predictions/preds.parquet"


def test_rebuild_run_index_captures_per_run_failures(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"

    ok_run = store_root / "runs" / "run-ok"
    (ok_run / "artifacts" / "predictions").mkdir(parents=True)
    _write_run_manifest(ok_run)
    (ok_run / "resolved.json").write_text("{}")
    (ok_run / "results.json").write_text("{}")
    (ok_run / "metrics.json").write_text("{}")
    (ok_run / "artifacts" / "predictions" / "preds.parquet").write_bytes(b"PAR1")

    bad_run = store_root / "runs" / "run-bad"
    bad_run.mkdir(parents=True)
    (bad_run / "run.json").write_text("{")

    result = rebuild_run_index(store_root=store_root)

    assert result.scanned_runs == 2
    assert result.indexed_runs == 1
    assert result.failed_runs == 1
    assert result.failures[0].run_id == "run-bad"


def test_doctor_reports_missing_db_and_missing_finished_outputs(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    (store_root / "runs" / "run-1").mkdir(parents=True)

    missing_db = doctor_store(store_root=store_root)
    assert missing_db.ok is False
    assert "store_db_missing" in missing_db.issues

    run_dir = store_root / "runs" / "run-1"
    _write_run_manifest(run_dir, status="FINISHED")

    init_store_db(store_root=store_root)
    index_run(store_root=store_root, run_id="run-1")

    report = doctor_store(store_root=store_root)

    assert report.ok is False
    assert report.stats["finished_runs"] == 1
    assert report.stats["finished_missing_resolved"] == 1
    assert report.stats["finished_missing_results"] == 1
    assert report.stats["finished_missing_metrics"] == 1


def test_doctor_fix_strays_deletes_targeted_dirs(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    init_store_db(store_root=store_root)

    legacy_logs = store_root / "logs"
    legacy_logs.mkdir(parents=True, exist_ok=True)
    (legacy_logs / "exp.launch.log").write_text("", encoding="utf-8")

    modal_smoke_data = store_root / "modal_smoke_data" / "v5.2"
    modal_smoke_data.mkdir(parents=True, exist_ok=True)
    (modal_smoke_data / "smoke_validation.parquet").write_text("payload", encoding="utf-8")

    smoke_live_check = store_root / "smoke_live_check" / "runs" / "run-1"
    smoke_live_check.mkdir(parents=True, exist_ok=True)
    (smoke_live_check / "run.json").write_text("{}", encoding="utf-8")

    report = doctor_store(store_root=store_root, fix_strays=True)

    assert report.stray_cleanup_applied is True
    assert str(store_root / "logs") in report.deleted_paths
    assert str(store_root / "modal_smoke_data") in report.deleted_paths
    assert str(store_root / "smoke_live_check") in report.deleted_paths
    assert report.missing_paths == ()
    assert not (store_root / "logs").exists()
    assert not (store_root / "modal_smoke_data").exists()
    assert not (store_root / "smoke_live_check").exists()


def test_doctor_fix_strays_deletes_old_unreferenced_tmp_remote_config(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    init_store_db(store_root=store_root)
    config_path = _write_tmp_remote_config(store_root, name="stale.json")
    _set_path_age_days(config_path, days_old=31)

    report = doctor_store(store_root=store_root, fix_strays=True)

    assert report.stray_cleanup_applied is True
    assert str(config_path) in report.deleted_paths
    assert not config_path.exists()
    assert report.stats["tmp_remote_configs_scanned"] == 1
    assert report.stats["tmp_remote_configs_deleted"] == 1
    assert report.stats["tmp_remote_configs_kept_recent"] == 0
    assert report.stats["tmp_remote_configs_skipped_active"] == 0
    assert report.stats["tmp_remote_configs_cleanup_skipped"] == 0


def test_doctor_fix_strays_keeps_recent_tmp_remote_config(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    init_store_db(store_root=store_root)
    config_path = _write_tmp_remote_config(store_root, name="recent.json")

    report = doctor_store(store_root=store_root, fix_strays=True)

    assert str(config_path) not in report.deleted_paths
    assert config_path.exists()
    assert report.stats["tmp_remote_configs_scanned"] == 1
    assert report.stats["tmp_remote_configs_deleted"] == 0
    assert report.stats["tmp_remote_configs_kept_recent"] == 1
    assert report.stats["tmp_remote_configs_skipped_active"] == 0
    assert report.stats["tmp_remote_configs_cleanup_skipped"] == 0


def test_doctor_fix_strays_keeps_old_active_tmp_remote_config(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    init_store_db(store_root=store_root)
    config_path = _write_tmp_remote_config(store_root, name="active.json")
    _set_path_age_days(config_path, days_old=31)
    _insert_run_lifecycle(store_root, run_id="run-active", config_path=config_path, status="running")

    report = doctor_store(store_root=store_root, fix_strays=True)

    assert str(config_path) not in report.deleted_paths
    assert config_path.exists()
    assert report.stats["tmp_remote_configs_scanned"] == 1
    assert report.stats["tmp_remote_configs_deleted"] == 0
    assert report.stats["tmp_remote_configs_kept_recent"] == 0
    assert report.stats["tmp_remote_configs_skipped_active"] == 1
    assert report.stats["tmp_remote_configs_cleanup_skipped"] == 0


def test_doctor_fix_strays_skips_tmp_remote_config_cleanup_when_db_missing(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    config_path = _write_tmp_remote_config(store_root, name="missing-db.json")
    _set_path_age_days(config_path, days_old=31)

    report = doctor_store(store_root=store_root, fix_strays=True)

    assert "store_db_missing" in report.issues
    assert config_path.exists()
    assert report.stats["tmp_remote_configs_scanned"] == 0
    assert report.stats["tmp_remote_configs_deleted"] == 0
    assert report.stats["tmp_remote_configs_kept_recent"] == 0
    assert report.stats["tmp_remote_configs_skipped_active"] == 0
    assert report.stats["tmp_remote_configs_cleanup_skipped"] == 1


def test_doctor_fix_strays_skips_tmp_remote_config_cleanup_when_db_unreadable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"
    init_store_db(store_root=store_root)
    config_path = _write_tmp_remote_config(store_root, name="unreadable-db.json")
    _set_path_age_days(config_path, days_old=31)
    monkeypatch.setattr(store_service, "_connect_read_only", lambda _db_path: (_ for _ in ()).throw(StoreError("boom")))

    report = doctor_store(store_root=store_root, fix_strays=True)

    assert "store_db_unreadable" in report.issues
    assert config_path.exists()
    assert report.stats["tmp_remote_configs_scanned"] == 0
    assert report.stats["tmp_remote_configs_deleted"] == 0
    assert report.stats["tmp_remote_configs_kept_recent"] == 0
    assert report.stats["tmp_remote_configs_skipped_active"] == 0
    assert report.stats["tmp_remote_configs_cleanup_skipped"] == 1


def test_materialize_viz_artifacts_creates_scoring_manifest_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-1"
    predictions_dir = run_dir / "artifacts" / "predictions"
    predictions_dir.mkdir(parents=True)
    predictions_path = predictions_dir / "preds.parquet"
    predictions_path.write_bytes(b"PAR1")

    _write_run_manifest(run_dir)
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    manifest["experiment_id"] = "exp-1"
    manifest["data"] = {"target_col": "target", "era_col": "era"}
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "resolved.json").write_text("{}")
    (run_dir / "results.json").write_text("{}")
    (run_dir / "metrics.json").write_text("{}")

    def _fake_score_run_for_materialize(**kwargs: object) -> None:
        run_payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        scoring_dir = run_dir / "artifacts" / "scoring"
        scoring_dir.mkdir(parents=True, exist_ok=True)
        (scoring_dir / "run_metric_series.parquet").write_bytes(b"PAR1")
        (scoring_dir / "manifest.json").write_text("{}", encoding="utf-8")
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    **run_payload,
                    "artifacts": {
                        **cast(dict[str, object], run_payload["artifacts"]),
                        "scoring_manifest": "artifacts/scoring/manifest.json",
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(store_service, "_score_run_for_materialize", _fake_score_run_for_materialize)

    first = materialize_viz_artifacts(
        store_root=store_root,
        kind="per-era-corr",
        experiment_id="exp-1",
    )
    second = materialize_viz_artifacts(
        store_root=store_root,
        kind="per-era-corr",
        run_id="run-1",
    )

    assert first.scoped_run_count == 1
    assert first.created_count == 1
    assert first.skipped_count == 0
    assert first.failed_count == 0
    assert second.created_count == 0
    assert second.skipped_count == 1
    assert (run_dir / "artifacts" / "scoring" / "run_metric_series.parquet").is_file()

    saved_manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    artifacts = cast(dict[str, object], saved_manifest["artifacts"])
    assert artifacts["scoring_manifest"] == "artifacts/scoring/manifest.json"


def test_materialize_viz_artifacts_requires_exactly_one_scope(tmp_path: Path) -> None:
    with pytest.raises(StoreError, match="store_viz_artifact_scope_invalid"):
        materialize_viz_artifacts(
            store_root=tmp_path / ".numereng",
            kind="per-era-corr",
        )


def test_materialize_viz_artifacts_missing_run_id_raises(tmp_path: Path) -> None:
    with pytest.raises(StoreError, match="store_run_not_found:run-missing"):
        materialize_viz_artifacts(
            store_root=tmp_path / ".numereng",
            kind="per-era-corr",
            run_id="run-missing",
        )


def test_materialize_viz_artifacts_experiment_scope_skips_unreadable_unrelated_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"
    target_run_dir = store_root / "runs" / "run-target"
    target_predictions_dir = target_run_dir / "artifacts" / "predictions"
    target_predictions_dir.mkdir(parents=True)
    (target_predictions_dir / "preds.parquet").write_bytes(b"PAR1")

    _write_run_manifest(target_run_dir)
    target_manifest = json.loads((target_run_dir / "run.json").read_text(encoding="utf-8"))
    target_manifest["experiment_id"] = "exp-1"
    target_manifest["data"] = {"target_col": "target", "era_col": "era"}
    (target_run_dir / "run.json").write_text(json.dumps(target_manifest, indent=2, sort_keys=True), encoding="utf-8")
    (target_run_dir / "resolved.json").write_text("{}")
    (target_run_dir / "results.json").write_text("{}")
    (target_run_dir / "metrics.json").write_text("{}")

    unrelated_run_dir = store_root / "runs" / "run-bad"
    unrelated_run_dir.mkdir(parents=True)
    (unrelated_run_dir / "run.json").write_text("{bad json", encoding="utf-8")

    monkeypatch.setattr(
        store_service,
        "_score_run_for_materialize",
        lambda **kwargs: (target_run_dir / "artifacts" / "scoring").mkdir(parents=True, exist_ok=True),
    )

    result = materialize_viz_artifacts(
        store_root=store_root,
        kind="per-era-corr",
        experiment_id="exp-1",
    )

    assert result.scoped_run_count == 1
    assert result.created_count == 1
    assert result.skipped_count == 0
    assert result.failed_count == 0


def test_upsert_cloud_job_writes_and_updates_rows(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    upsert_cloud_job(
        store_root=store_root,
        job=StoreCloudJobUpsert(
            run_id="run-1",
            provider="sagemaker",
            backend="sagemaker",
            provider_job_id="job-1",
            status="InProgress",
            region="us-east-2",
            image_uri="123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng:v1",
            output_s3_uri="s3://bucket/runs/run-1/output/",
        ),
    )
    upsert_cloud_job(
        store_root=store_root,
        job=StoreCloudJobUpsert(
            run_id="run-1",
            provider="sagemaker",
            backend="sagemaker",
            provider_job_id="job-1",
            status="Completed",
            region="us-east-2",
            image_uri="123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng:v1",
            output_s3_uri="s3://bucket/runs/run-1/output/",
            finished_at="2026-02-22T00:00:00+00:00",
        ),
    )

    with sqlite3.connect(store_root / "numereng.db") as conn:
        row = conn.execute(
            """
            SELECT status, output_s3_uri, finished_at
            FROM cloud_jobs
            WHERE run_id = ? AND provider = ? AND provider_job_id = ?
            """,
            ("run-1", "sagemaker", "job-1"),
        ).fetchone()

    assert row is not None
    assert row[0] == "Completed"
    assert row[1] == "s3://bucket/runs/run-1/output/"
    assert row[2] == "2026-02-22T00:00:00+00:00"


def test_upsert_cloud_job_keeps_distinct_rows_per_provider(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    upsert_cloud_job(
        store_root=store_root,
        job=StoreCloudJobUpsert(
            run_id="run-1",
            provider="sagemaker",
            backend="sagemaker",
            provider_job_id="job-1",
            status="InProgress",
        ),
    )
    upsert_cloud_job(
        store_root=store_root,
        job=StoreCloudJobUpsert(
            run_id="run-1",
            provider="batch",
            backend="batch",
            provider_job_id="job-1",
            status="SUBMITTED",
        ),
    )

    with sqlite3.connect(store_root / "numereng.db") as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM cloud_jobs WHERE run_id = ? AND provider_job_id = ?",
            ("run-1", "job-1"),
        ).fetchone()

    assert count is not None
    assert count[0] == 2


def test_upsert_cloud_job_merges_metadata_json(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    upsert_cloud_job(
        store_root=store_root,
        job=StoreCloudJobUpsert(
            run_id="run-1",
            provider="sagemaker",
            backend="sagemaker",
            provider_job_id="job-1",
            status="InProgress",
            metadata_json=json.dumps(
                {
                    "experiment_id": "exp-1",
                    "config_path": "experiments/exp-1/configs/base.json",
                    "config_label": "base.json",
                }
            ),
        ),
    )
    upsert_cloud_job(
        store_root=store_root,
        job=StoreCloudJobUpsert(
            run_id="run-1",
            provider="sagemaker",
            backend="sagemaker",
            provider_job_id="job-1",
            status="Completed",
            metadata_json=json.dumps({"status": "Completed", "backend": "sagemaker"}),
        ),
    )

    with sqlite3.connect(store_root / "numereng.db") as conn:
        row = conn.execute(
            """
            SELECT metadata_json
            FROM cloud_jobs
            WHERE run_id = ? AND provider = ? AND provider_job_id = ?
            """,
            ("run-1", "sagemaker", "job-1"),
        ).fetchone()

    assert row is not None
    metadata = json.loads(str(row[0]))
    assert metadata["experiment_id"] == "exp-1"
    assert metadata["config_path"] == "experiments/exp-1/configs/base.json"
    assert metadata["config_label"] == "base.json"
    assert metadata["status"] == "Completed"


def test_init_store_db_migrates_cloud_jobs_legacy_primary_key(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)
    db_path = store_root / "numereng.db"

    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (name TEXT PRIMARY KEY, applied_at TEXT NOT NULL)")
        conn.execute(
            """
            CREATE TABLE cloud_jobs (
                run_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                backend TEXT NOT NULL,
                provider_job_id TEXT NOT NULL,
                status TEXT NOT NULL,
                region TEXT,
                image_uri TEXT,
                output_s3_uri TEXT,
                metadata_json TEXT,
                error_message TEXT,
                started_at TEXT,
                finished_at TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, provider_job_id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO cloud_jobs (
                run_id,
                provider,
                backend,
                provider_job_id,
                status,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("run-legacy", "sagemaker", "sagemaker", "job-1", "Completed", "2026-02-22T00:00:00+00:00"),
        )
        conn.commit()

    init_store_db(store_root=store_root)

    with sqlite3.connect(db_path) as conn:
        pk_info = conn.execute("PRAGMA table_info(cloud_jobs)").fetchall()
        pk_columns = [str(row[1]) for row in sorted(pk_info, key=lambda row: int(row[5])) if int(row[5]) > 0]
        row = conn.execute(
            "SELECT provider, provider_job_id, status FROM cloud_jobs WHERE run_id = ?",
            ("run-legacy",),
        ).fetchone()

    assert pk_columns == ["run_id", "provider", "provider_job_id"]
    assert row is not None
    assert row[0] == "sagemaker"
    assert row[1] == "job-1"
    assert row[2] == "Completed"


def test_index_run_skips_artifacts_outside_run_directory(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    _write_run_manifest(run_dir)
    (run_dir / "resolved.json").write_text("{}", encoding="utf-8")
    (run_dir / "results.json").write_text("{}", encoding="utf-8")
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    outside_artifact = tmp_path / "outside.json"
    outside_artifact.write_text("{}", encoding="utf-8")

    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    payload["artifacts"]["outside"] = str(outside_artifact)
    (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")

    index_run(store_root=store_root, run_id="run-1")

    with sqlite3.connect(store_root / "numereng.db") as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM run_artifacts WHERE run_id = ? AND kind = ?",
            ("run-1", "outside"),
        ).fetchone()

    assert row is not None
    assert row[0] == 0


def test_init_store_db_quarantines_corrupt_sidecars_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)
    db_path = store_root / "numereng.db"
    db_path.write_bytes(b"")
    wal_path = Path(f"{db_path}-wal")
    shm_path = Path(f"{db_path}-shm")
    wal_path.write_bytes(b"corrupt-wal")
    shm_path.write_bytes(b"corrupt-shm")

    original_connect = sqlite3.connect
    call_count = 0

    def flaky_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:  # noqa: ANN401
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise sqlite3.DatabaseError("database disk image is malformed")
        return original_connect(*args, **kwargs)  # type: ignore[no-any-return]

    monkeypatch.setattr("numereng.features.store.service.sqlite3.connect", flaky_connect)

    init_store_db(store_root=store_root)

    assert call_count >= 2
    quarantined_wal = list(store_root.glob("numereng.db-wal.corrupt.*"))
    quarantined_shm = list(store_root.glob("numereng.db-shm.corrupt.*"))
    assert len(quarantined_wal) == 1
    assert len(quarantined_shm) == 1
    assert quarantined_wal[0].read_bytes() == b"corrupt-wal"
    assert quarantined_shm[0].read_bytes() == b"corrupt-shm"


def test_index_run_wraps_sqlite_errors_as_store_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    _write_run_manifest(run_dir)
    (run_dir / "resolved.json").write_text("{}", encoding="utf-8")
    (run_dir / "results.json").write_text("{}", encoding="utf-8")
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")

    def boom(*args: object, **kwargs: object) -> object:
        raise sqlite3.DatabaseError("database disk image is malformed")

    monkeypatch.setattr(store_service, "_index_run_with_connection", boom)

    with pytest.raises(StoreError, match="store_db_corrupt_error:index_run:"):
        index_run(store_root=store_root, run_id="run-1")


def test_rebuild_run_index_captures_per_run_sqlite_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / ".numereng"

    for run_id in ("run-ok", "run-bad"):
        run_dir = store_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_run_manifest(run_dir)
        (run_dir / "resolved.json").write_text("{}", encoding="utf-8")
        (run_dir / "results.json").write_text("{}", encoding="utf-8")
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")

    def fake_index_with_failure(conn: sqlite3.Connection, *, store_root: Path, run_id: str) -> object:
        if run_id == "run-bad":
            raise sqlite3.DatabaseError("database disk image is malformed")
        return store_service.StoreIndexResult(
            run_id=run_id,
            status="FINISHED",
            metrics_indexed=0,
            artifacts_indexed=0,
            run_path=store_root / "runs" / run_id,
            warnings=(),
        )

    monkeypatch.setattr(store_service, "_index_run_with_connection", fake_index_with_failure)

    result = rebuild_run_index(store_root=store_root)

    assert result.scanned_runs == 2
    assert result.indexed_runs == 1
    assert result.failed_runs == 1
    assert result.failures[0].run_id == "run-bad"
    assert result.failures[0].error.startswith("store_db_corrupt_error:rebuild_run_index:run-bad:")


def test_upsert_and_list_experiments(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"

    upsert_experiment(
        store_root=store_root,
        experiment_id="2026-02-22_test-exp",
        name="Test Experiment",
        status="active",
        created_at="2026-02-22T00:00:00+00:00",
        updated_at="2026-02-22T00:10:00+00:00",
        metadata={"tags": ["quick"], "champion_run_id": "run-2"},
    )

    record = get_experiment(store_root=store_root, experiment_id="2026-02-22_test-exp")
    assert record is not None
    assert record.name == "Test Experiment"
    assert record.status == "active"

    rows = list_experiments(store_root=store_root, status="active")
    assert len(rows) == 1
    assert rows[0].experiment_id == "2026-02-22_test-exp"


def test_backfill_run_execution_and_doctor_report_missing_and_legacy_cloud(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (store_root / "cloud").mkdir(parents=True, exist_ok=True)
    _write_run_manifest(run_dir)
    (run_dir / "resolved.json").write_text("{}", encoding="utf-8")
    (run_dir / "results.json").write_text("{}", encoding="utf-8")
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    legacy_state_path = store_root / "cloud" / "run-1.json"
    legacy_state_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "backend": "sagemaker",
                "training_job_name": "job-1",
                "region": "us-east-2",
                "image_uri": "123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng:v1",
                "artifacts": {"output_s3_uri": "s3://bucket/runs/run-1/managed-output/"},
                "metadata": {"submitted_at": "2026-03-28T00:00:00+00:00"},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    doctor_before = doctor_store(store_root=store_root)
    assert "run_execution_missing:1" in doctor_before.issues
    assert "legacy_cloud_state_present:1" in doctor_before.issues

    result = backfill_run_execution(store_root=store_root, all_runs=True)
    assert result.updated_runs == 1
    assert result.updated_run_ids == ("run-1",)

    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["execution"]["kind"] == "cloud"
    assert manifest["execution"]["provider"] == "aws"
    assert manifest["execution"]["backend"] == "sagemaker"
    assert manifest["execution"]["provider_job_id"] == "job-1"
    assert manifest["execution"]["metadata"]["inferred"] is True
    assert "legacy_cloud_state" in manifest["execution"]["metadata"]["inferred_from"]

    doctor_after = doctor_store(store_root=store_root)
    assert "run_execution_missing:1" not in doctor_after.issues
    assert "legacy_cloud_state_present:1" in doctor_after.issues
