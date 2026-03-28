from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from numereng_viz.contracts import capabilities_payload
from numereng_viz.monitor_snapshot import RemoteSnapshotCoordinator, build_monitor_snapshot
from numereng_viz.routes import create_router
from numereng_viz.services import VizService
from numereng_viz.store_adapter import (
    VizStoreAdapter,
    VizStoreConfig,
    _normalize_round_metrics,
    repository_root,
    resolve_store_root,
)

from numereng.features.remote_ops import service as remote_service
from numereng.features.store import StoreCloudJobUpsert, init_store_db, upsert_cloud_job, upsert_experiment
from numereng.platform.remotes.contracts import SshRemoteTargetProfile


def test_resolve_store_root_prefers_explicit(tmp_path: Path) -> None:
    explicit = tmp_path / "store"
    explicit.mkdir()

    resolved = resolve_store_root(explicit)

    assert resolved == explicit.resolve()


def test_repository_root_resolves_project_root() -> None:
    assert repository_root() == Path(__file__).resolve().parents[4]


def test_capabilities_payload_read_only_flag() -> None:
    assert capabilities_payload() == {
        "read_only": True,
        "write_controls": False,
    }


def test_list_experiments_fallback_without_db(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    (store_root / "experiments" / "exp-a").mkdir(parents=True)

    adapter = VizStoreAdapter(VizStoreConfig(store_root=store_root, repo_root=tmp_path))

    payload = adapter.list_experiments()

    assert len(payload) == 1
    assert payload[0]["experiment_id"] == "exp-a"


def test_list_experiment_configs_discovers_json_configs(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    config_dir = store_root / "experiments" / "exp-json" / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "base.json"
    config_path.write_text(
        """
        {
          "run_id": "run-abc",
          "model": {"type": "LGBMRegressor"},
          "data": {"target": "target_20", "feature_set": "small"}
        }
        """.strip(),
        encoding="utf-8",
    )

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path,
        )
    )

    payload = adapter.list_experiment_configs("exp-json", runnable_only=True)

    assert payload["total"] == 1
    assert payload["items"][0]["config_id"].endswith("base.json")
    assert payload["items"][0]["summary"]["model_type"] == "LGBMRegressor"


def test_list_experiments_hides_archived_but_archived_detail_paths_still_resolve(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    live_dir = store_root / "experiments" / "exp-live"
    archived_dir = store_root / "experiments" / "_archive" / "exp-archived"
    (live_dir / "configs").mkdir(parents=True)
    (archived_dir / "configs").mkdir(parents=True)
    (live_dir / "experiment.json").write_text('{"experiment_id":"exp-live","name":"Live","status":"active","runs":[]}')
    (archived_dir / "experiment.json").write_text(
        '{"experiment_id":"exp-archived","name":"Archived","status":"archived","runs":[]}'
    )
    (archived_dir / "EXPERIMENT.md").write_text("# archived\n")
    (archived_dir / "configs" / "base.json").write_text(
        '{"model":{"type":"LGBMRegressor"},"data":{"target":"target_20","feature_set":"small"}}'
    )
    (archived_dir / "results").mkdir()
    (archived_dir / "results" / "r1_model_metrics.json").write_text('{"bmc":{"mean":0.1}}')

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path,
        )
    )

    listing = adapter.list_experiments()
    assert [item["experiment_id"] for item in listing] == ["exp-live"]

    archived = adapter.get_experiment("exp-archived")
    assert archived is not None
    assert archived["status"] == "archived"

    archived_configs = adapter.list_experiment_configs("exp-archived", runnable_only=True)
    assert archived_configs["total"] == 1
    assert archived_configs["items"][0]["summary"]["model_type"] == "LGBMRegressor"

    all_configs = adapter.list_all_configs(limit=500, offset=0)
    assert all_configs["total"] == 0

    archived_doc = adapter.get_experiment_doc("exp-archived", "EXPERIMENT.md")
    assert archived_doc["exists"] is True

    archived_round_results = adapter.list_experiment_round_results("exp-archived")
    assert len(archived_round_results) == 1


def test_normalize_round_metrics_adds_payout_aliases_without_overwriting_native_metrics() -> None:
    payload = _normalize_round_metrics(
        {
            "corr": {"mean": 0.11},
            "mmc": {"mean": 0.01},
            "corr_ender20": {"mean": 0.21},
            "mmc_ender20": {"mean": 0.02},
        }
    )

    assert payload["corr_mean"] == 0.11
    assert payload["mmc_mean"] == 0.02
    assert payload["corr_payout_mean"] == 0.21
    assert payload["mmc_payout_mean"] == 0.02


def test_normalize_round_metrics_omits_payout_aliases_when_payout_metrics_are_missing() -> None:
    payload = _normalize_round_metrics(
        {
            "corr": {"mean": 0.11},
            "mmc": {"mean": 0.01},
        }
    )

    assert payload["corr_mean"] == 0.11
    assert payload["mmc_mean"] == 0.01
    assert "corr_payout_mean" not in payload
    assert payload["mmc_payout_mean"] == 0.01


def test_normalize_round_metrics_promotes_corr_with_benchmark_alias() -> None:
    payload = _normalize_round_metrics(
        {
            "bmc": {"mean": 0.03, "avg_corr_with_benchmark": 0.12},
            "bmc_last_200_eras": {"mean": 0.05},
        }
    )

    assert payload["bmc_mean"] == 0.03
    assert payload["bmc_last_200_eras_mean"] == 0.05
    assert payload["corr_with_benchmark"] == 0.12


def test_get_run_metrics_uses_shared_scalar_metric_normalization(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "status": "FINISHED",
                "created_at": "2026-02-22T00:00:00+00:00",
                "data": {"target_col": "target_20"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "corr": {"mean": 0.11, "sharpe": 1.2, "max_drawdown": -0.03},
                "mmc": {"mean": 0.02},
                "cwmm": {"mean": 0.03},
                "fnc": {"mean": 0.04},
                "bmc": {"mean": 0.05},
                "bmc_last_200_eras": {"mean": 0.06},
                "feature_exposure": {"mean": 0.07},
                "max_feature_exposure": {"mean": 0.08},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "score_provenance.json").write_text(
        json.dumps({"joins": {"predictions_rows": 8, "meta_overlap_rows": 2}}),
        encoding="utf-8",
    )

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path,
        )
    )

    payload = adapter.get_run_metrics("run-1")

    assert payload is not None
    assert payload["bmc_last_200_eras_mean"] == 0.06
    assert payload["bmc_mean"] == 0.05
    assert payload["corr_sharpe"] == 1.2
    assert payload["corr_mean"] == 0.11
    assert payload["mmc_mean"] == 0.02
    assert payload["cwmm_mean"] == 0.03
    assert payload["fnc_mean"] == 0.04
    assert payload["feature_exposure_mean"] == 0.07
    assert payload["max_feature_exposure"] == 0.08
    assert payload["max_drawdown"] == -0.03
    assert payload["mmc_coverage_ratio_rows"] == 0.25


def test_get_scoring_dashboard_normalizes_legacy_contribution_keys(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-1"
    scoring_dir = run_dir / "artifacts" / "scoring"
    scoring_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "status": "FINISHED",
                "data": {"target_col": "target_agnes_60"},
            }
        ),
        encoding="utf-8",
    )
    (scoring_dir / "manifest.json").write_text(json.dumps({"stages": {"omissions": {}}}), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "config_hash": "cfg",
                "seed": None,
                "target_col": "target_agnes_60",
                "payout_target_col": "target_ender_20",
                "prediction_col": "prediction",
                "era": "era1",
                "metric_key": "bmc_ender20",
                "series_type": "per_era",
                "value": 0.1,
            },
            {
                "run_id": "run-1",
                "config_hash": "cfg",
                "seed": None,
                "target_col": "target_agnes_60",
                "payout_target_col": "target_ender_20",
                "prediction_col": "prediction",
                "era": "era1",
                "metric_key": "mmc_ender20",
                "series_type": "per_era",
                "value": 0.2,
            },
            {
                "run_id": "run-1",
                "config_hash": "cfg",
                "seed": None,
                "target_col": "target_agnes_60",
                "payout_target_col": "target_ender_20",
                "prediction_col": "prediction",
                "era": "era1",
                "metric_key": "bmc_native",
                "series_type": "per_era",
                "value": 0.3,
            },
            {
                "run_id": "run-1",
                "config_hash": "cfg",
                "seed": None,
                "target_col": "target_agnes_60",
                "payout_target_col": "target_ender_20",
                "prediction_col": "prediction",
                "era": "era1",
                "metric_key": "baseline_corr_ender20",
                "series_type": "per_era",
                "value": 0.4,
            },
            {
                "run_id": "run-1",
                "config_hash": "cfg",
                "seed": None,
                "target_col": "target_agnes_60",
                "payout_target_col": "target_ender_20",
                "prediction_col": "prediction",
                "era": "era1",
                "metric_key": "corr_delta_vs_baseline_ender20",
                "series_type": "per_era",
                "value": 0.5,
            },
            {
                "run_id": "run-1",
                "config_hash": "cfg",
                "seed": None,
                "target_col": "target_agnes_60",
                "payout_target_col": "target_ender_20",
                "prediction_col": "prediction",
                "era": "era1",
                "metric_key": "corr_delta_vs_baseline_native",
                "series_type": "per_era",
                "value": 0.6,
            },
        ]
    ).to_parquet(scoring_dir / "run_metric_series.parquet", index=False)
    pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "config_hash": "cfg",
                "seed": None,
                "target_col": "target_agnes_60",
                "payout_target_col": "target_ender_20",
                "prediction_col": "prediction",
                "bmc_ender20_mean": 0.11,
                "bmc_ender20_std": 0.21,
                "bmc_last_200_eras_ender20_mean": 0.31,
                "bmc_last_200_eras_ender20_std": 0.41,
                "mmc_ender20_mean": 0.12,
                "mmc_ender20_std": 0.22,
                "corr_delta_vs_baseline_ender20_mean": 0.13,
                "corr_delta_vs_baseline_ender20_std": 0.23,
            }
        ]
    ).to_parquet(scoring_dir / "post_training_summary.parquet", index=False)
    pd.DataFrame(
        [
            {
                "cv_fold": 0,
                "bmc_ender20_fold_mean": 0.14,
            }
        ]
    ).to_parquet(scoring_dir / "post_fold_snapshots.parquet", index=False)

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path,
        )
    )

    payload = adapter.get_scoring_dashboard("run-1")

    assert payload is not None
    assert payload["meta"]["available_metric_keys"] == ["bmc", "corr_delta_vs_baseline", "mmc"]
    series_metric_keys = sorted({str(row["metric_key"]) for row in payload["series"]})
    assert series_metric_keys == ["bmc", "corr_delta_vs_baseline", "mmc"]
    summary = payload["summary"]
    assert isinstance(summary, dict)
    assert summary["bmc_mean"] == pytest.approx(0.11)
    assert summary["bmc_last_200_eras_mean"] == pytest.approx(0.31)
    assert summary["mmc_mean"] == pytest.approx(0.12)
    assert summary["corr_delta_vs_baseline_mean"] == pytest.approx(0.13)
    fold_rows = payload["fold_snapshots"]
    assert isinstance(fold_rows, list)
    assert fold_rows[0]["bmc_fold_mean"] == pytest.approx(0.14)


def _make_readonly_client(*, repo_root: Path, store_root: Path) -> TestClient:
    return _make_client(repo_root=repo_root, store_root=store_root)


def _make_client(*, repo_root: Path, store_root: Path) -> TestClient:
    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=repo_root,
        )
    )
    service = VizService(adapter)
    service.remote_snapshots.fetch_snapshots = lambda: []
    app = FastAPI()
    app.state.viz_service = service
    app.include_router(create_router(service))
    return TestClient(app)


def _seed_experiments_overview_store(tmp_path: Path) -> Path:
    store_root = tmp_path / ".numereng"
    init_result = init_store_db(store_root=store_root)

    experiments = [
        {
            "experiment_id": "exp-live-alert",
            "name": "Live Alert",
            "status": "active",
            "created_at": "2026-03-20T00:00:00+00:00",
            "updated_at": "2026-03-25T10:10:30+00:00",
            "metadata": {"tags": ["alert", "monitor"]},
        },
        {
            "experiment_id": "exp-live",
            "name": "Live Queue",
            "status": "active",
            "created_at": "2026-03-21T00:00:00+00:00",
            "updated_at": "2026-03-25T10:11:30+00:00",
            "metadata": {"tags": ["live"]},
        },
        {
            "experiment_id": "exp-stale",
            "name": "Recovered Stale",
            "status": "active",
            "created_at": "2026-03-22T00:00:00+00:00",
            "updated_at": "2026-03-25T10:12:00+00:00",
            "metadata": {"tags": ["reconcile"]},
        },
        {
            "experiment_id": "exp-canceled",
            "name": "Canceled Work",
            "status": "complete",
            "created_at": "2026-03-23T00:00:00+00:00",
            "updated_at": "2026-03-25T10:13:00+00:00",
            "metadata": {"tags": ["cancel"]},
        },
        {
            "experiment_id": "exp-done",
            "name": "Completed Work",
            "status": "complete",
            "created_at": "2026-03-24T00:00:00+00:00",
            "updated_at": "2026-03-25T10:14:00+00:00",
            "metadata": {"tags": ["done"]},
        },
    ]
    for item in experiments:
        upsert_experiment(store_root=store_root, **item)

    run_rows: list[dict[str, object]] = []
    run_jobs: list[dict[str, object]] = []
    lifecycles: list[dict[str, object]] = []

    def add_run(
        *,
        run_id: str,
        experiment_id: str,
        status: str,
        created_at: str,
        finished_at: str | None,
        config_id: str,
    ) -> None:
        run_dir = store_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        run_rows.append(
            {
                "run_id": run_id,
                "run_hash": f"hash-{run_id}",
                "status": status,
                "run_type": "training",
                "created_at": created_at,
                "finished_at": finished_at,
                "config_hash": f"cfg-{run_id}",
                "experiment_id": experiment_id,
                "run_path": str(run_dir),
                "manifest_json": json.dumps({"run_id": run_id, "status": status, "config_id": config_id}),
                "manifest_mtime_ns": 1,
                "ingested_at": created_at,
            }
        )

    def add_job(
        *,
        job_id: str,
        run_id: str,
        experiment_id: str,
        logical_run_id: str,
        config_id: str,
        status: str,
        created_at: str,
        updated_at: str,
        started_at: str | None = None,
        finished_at: str | None = None,
        cancel_requested: int = 0,
        cancel_requested_at: str | None = None,
        terminal_reason: str | None = None,
        error_json: dict[str, object] | None = None,
    ) -> None:
        run_dir = store_root / "runs" / run_id
        run_jobs.append(
            {
                "job_id": job_id,
                "batch_id": f"batch-{experiment_id}",
                "experiment_id": experiment_id,
                "logical_run_id": logical_run_id,
                "operation_type": "run",
                "attempt_no": 1,
                "attempt_id": f"attempt-{job_id}",
                "config_id": config_id,
                "config_source": "store",
                "config_path": config_id,
                "config_sha256": f"sha-{config_id}",
                "request_json": "{}",
                "job_type": "run",
                "status": status,
                "queue_name": "default",
                "priority": 0,
                "created_at": created_at,
                "queued_at": created_at,
                "started_at": started_at,
                "finished_at": finished_at,
                "updated_at": updated_at,
                "worker_id": "worker-local",
                "pid": 1234,
                "exit_code": None,
                "signal": None,
                "backend": "local",
                "tier": None,
                "budget": None,
                "timeout_seconds": None,
                "canonical_run_id": run_id,
                "external_run_id": None,
                "run_dir": str(run_dir),
                "cancel_requested": cancel_requested,
                "cancel_requested_at": cancel_requested_at,
                "terminal_reason": terminal_reason,
                "terminal_detail_json": json.dumps({"terminal_reason": terminal_reason}) if terminal_reason else None,
                "error_json": json.dumps(error_json) if error_json else None,
            }
        )

    def add_lifecycle(
        *,
        run_id: str,
        job_id: str,
        logical_run_id: str,
        experiment_id: str,
        config_id: str,
        status: str,
        created_at: str,
        updated_at: str,
        current_stage: str | None,
        progress_percent: float | None,
        progress_label: str | None,
        progress_current: int | None,
        progress_total: int | None,
        started_at: str | None = None,
        finished_at: str | None = None,
        last_heartbeat_at: str | None = None,
        cancel_requested: int = 0,
        cancel_requested_at: str | None = None,
        terminal_reason: str | None = None,
        reconciled: int = 0,
    ) -> None:
        run_dir = store_root / "runs" / run_id
        lifecycles.append(
            {
                "run_id": run_id,
                "run_hash": f"hash-{run_id}",
                "config_hash": f"cfg-{run_id}",
                "job_id": job_id,
                "logical_run_id": logical_run_id,
                "attempt_id": f"attempt-{job_id}",
                "attempt_no": 1,
                "source": "cli.run.train",
                "operation_type": "run",
                "job_type": "run",
                "status": status,
                "experiment_id": experiment_id,
                "config_id": config_id,
                "config_source": "store",
                "config_path": config_id,
                "config_sha256": f"sha-{config_id}",
                "run_dir": str(run_dir),
                "runtime_path": str(run_dir / "runtime.json"),
                "backend": "local",
                "worker_id": "worker-local",
                "pid": 1234,
                "host": "localhost",
                "current_stage": current_stage,
                "completed_stages_json": json.dumps(["initializing", "load_data"] if progress_current else []),
                "progress_percent": progress_percent,
                "progress_label": progress_label,
                "progress_current": progress_current,
                "progress_total": progress_total,
                "cancel_requested": cancel_requested,
                "cancel_requested_at": cancel_requested_at,
                "created_at": created_at,
                "queued_at": created_at,
                "started_at": started_at,
                "last_heartbeat_at": last_heartbeat_at,
                "updated_at": updated_at,
                "finished_at": finished_at,
                "terminal_reason": terminal_reason,
                "terminal_detail_json": json.dumps({"reason": terminal_reason}) if terminal_reason else json.dumps({}),
                "latest_metrics_json": json.dumps({"corr_mean": 0.11})
                if progress_percent is not None
                else json.dumps({}),
                "latest_sample_json": json.dumps({"process_rss_gb": 0.42})
                if progress_percent is not None
                else json.dumps({}),
                "reconciled": reconciled,
            }
        )

    add_run(
        run_id="run-live-alert-active",
        experiment_id="exp-live-alert",
        status="RUNNING",
        created_at="2026-03-25T10:00:00+00:00",
        finished_at=None,
        config_id="configs/alert_active.json",
    )
    add_job(
        job_id="job-live-alert-active",
        run_id="run-live-alert-active",
        experiment_id="exp-live-alert",
        logical_run_id="logical-live-alert-active",
        config_id="configs/alert_active.json",
        status="running",
        created_at="2026-03-25T10:00:00+00:00",
        started_at="2026-03-25T10:00:02+00:00",
        updated_at="2026-03-25T10:10:30+00:00",
    )
    add_lifecycle(
        run_id="run-live-alert-active",
        job_id="job-live-alert-active",
        logical_run_id="logical-live-alert-active",
        experiment_id="exp-live-alert",
        config_id="configs/alert_active.json",
        status="running",
        created_at="2026-03-25T10:00:00+00:00",
        started_at="2026-03-25T10:00:02+00:00",
        last_heartbeat_at="2026-03-25T10:10:30+00:00",
        updated_at="2026-03-25T10:10:30+00:00",
        current_stage="train_model",
        progress_percent=72.0,
        progress_label="Fold 5 of 7",
        progress_current=5,
        progress_total=7,
    )

    add_run(
        run_id="run-live-alert-failed",
        experiment_id="exp-live-alert",
        status="FAILED",
        created_at="2026-03-25T09:40:00+00:00",
        finished_at="2026-03-25T10:09:59+00:00",
        config_id="configs/alert_failed.json",
    )
    add_job(
        job_id="job-live-alert-failed",
        run_id="run-live-alert-failed",
        experiment_id="exp-live-alert",
        logical_run_id="logical-live-alert-failed",
        config_id="configs/alert_failed.json",
        status="failed",
        created_at="2026-03-25T09:40:00+00:00",
        started_at="2026-03-25T09:40:02+00:00",
        finished_at="2026-03-25T10:09:59+00:00",
        updated_at="2026-03-25T10:09:59+00:00",
        terminal_reason="training_run_failed",
        error_json={"message": "boom"},
    )
    add_lifecycle(
        run_id="run-live-alert-failed",
        job_id="job-live-alert-failed",
        logical_run_id="logical-live-alert-failed",
        experiment_id="exp-live-alert",
        config_id="configs/alert_failed.json",
        status="failed",
        created_at="2026-03-25T09:40:00+00:00",
        started_at="2026-03-25T09:40:02+00:00",
        finished_at="2026-03-25T10:09:59+00:00",
        updated_at="2026-03-25T10:09:59+00:00",
        current_stage="train_model",
        progress_percent=44.0,
        progress_label="Fold 3 of 7",
        progress_current=3,
        progress_total=7,
        terminal_reason="training_run_failed",
    )

    add_run(
        run_id="run-live-1",
        experiment_id="exp-live",
        status="RUNNING",
        created_at="2026-03-25T10:01:00+00:00",
        finished_at=None,
        config_id="configs/live_fold.json",
    )
    add_job(
        job_id="job-live-1",
        run_id="run-live-1",
        experiment_id="exp-live",
        logical_run_id="logical-live-1",
        config_id="configs/live_fold.json",
        status="running",
        created_at="2026-03-25T10:01:00+00:00",
        started_at="2026-03-25T10:01:01+00:00",
        updated_at="2026-03-25T10:11:30+00:00",
    )
    add_lifecycle(
        run_id="run-live-1",
        job_id="job-live-1",
        logical_run_id="logical-live-1",
        experiment_id="exp-live",
        config_id="configs/live_fold.json",
        status="running",
        created_at="2026-03-25T10:01:00+00:00",
        started_at="2026-03-25T10:01:01+00:00",
        last_heartbeat_at="2026-03-25T10:11:30+00:00",
        updated_at="2026-03-25T10:11:30+00:00",
        current_stage="train_model",
        progress_percent=38.0,
        progress_label="Fold 3 of 7",
        progress_current=3,
        progress_total=7,
    )

    add_run(
        run_id="run-live-2",
        experiment_id="exp-live",
        status="RUNNING",
        created_at="2026-03-25T10:05:00+00:00",
        finished_at=None,
        config_id="configs/live_queued.json",
    )
    add_job(
        job_id="job-live-2",
        run_id="run-live-2",
        experiment_id="exp-live",
        logical_run_id="logical-live-2",
        config_id="configs/live_queued.json",
        status="queued",
        created_at="2026-03-25T10:05:00+00:00",
        updated_at="2026-03-25T10:11:00+00:00",
    )
    add_lifecycle(
        run_id="run-live-2",
        job_id="job-live-2",
        logical_run_id="logical-live-2",
        experiment_id="exp-live",
        config_id="configs/live_queued.json",
        status="queued",
        created_at="2026-03-25T10:05:00+00:00",
        updated_at="2026-03-25T10:11:00+00:00",
        current_stage="queued",
        progress_percent=0.0,
        progress_label="Queued for worker",
        progress_current=0,
        progress_total=1,
    )

    add_run(
        run_id="run-stale",
        experiment_id="exp-stale",
        status="STALE",
        created_at="2026-03-25T09:20:00+00:00",
        finished_at="2026-03-25T10:12:00+00:00",
        config_id="configs/stale.json",
    )
    add_job(
        job_id="job-stale",
        run_id="run-stale",
        experiment_id="exp-stale",
        logical_run_id="logical-stale",
        config_id="configs/stale.json",
        status="stale",
        created_at="2026-03-25T09:20:00+00:00",
        started_at="2026-03-25T09:20:02+00:00",
        finished_at="2026-03-25T10:12:00+00:00",
        updated_at="2026-03-25T10:12:00+00:00",
        terminal_reason="reconciled_stale",
    )
    add_lifecycle(
        run_id="run-stale",
        job_id="job-stale",
        logical_run_id="logical-stale",
        experiment_id="exp-stale",
        config_id="configs/stale.json",
        status="stale",
        created_at="2026-03-25T09:20:00+00:00",
        started_at="2026-03-25T09:20:02+00:00",
        finished_at="2026-03-25T10:12:00+00:00",
        updated_at="2026-03-25T10:12:00+00:00",
        current_stage="train_model",
        progress_percent=61.0,
        progress_label="Fold 5 of 8",
        progress_current=5,
        progress_total=8,
        terminal_reason="reconciled_stale",
        reconciled=1,
    )

    add_run(
        run_id="run-canceled",
        experiment_id="exp-canceled",
        status="CANCELED",
        created_at="2026-03-25T09:30:00+00:00",
        finished_at="2026-03-25T10:13:00+00:00",
        config_id="configs/canceled.json",
    )
    add_job(
        job_id="job-canceled",
        run_id="run-canceled",
        experiment_id="exp-canceled",
        logical_run_id="logical-canceled",
        config_id="configs/canceled.json",
        status="canceled",
        created_at="2026-03-25T09:30:00+00:00",
        started_at="2026-03-25T09:30:01+00:00",
        finished_at="2026-03-25T10:13:00+00:00",
        updated_at="2026-03-25T10:13:00+00:00",
        cancel_requested=1,
        cancel_requested_at="2026-03-25T10:12:30+00:00",
        terminal_reason="cancel_requested",
    )
    add_lifecycle(
        run_id="run-canceled",
        job_id="job-canceled",
        logical_run_id="logical-canceled",
        experiment_id="exp-canceled",
        config_id="configs/canceled.json",
        status="canceled",
        created_at="2026-03-25T09:30:00+00:00",
        started_at="2026-03-25T09:30:01+00:00",
        finished_at="2026-03-25T10:13:00+00:00",
        updated_at="2026-03-25T10:13:00+00:00",
        current_stage="train_model",
        progress_percent=22.0,
        progress_label="Fold 2 of 7",
        progress_current=2,
        progress_total=7,
        cancel_requested=1,
        cancel_requested_at="2026-03-25T10:12:30+00:00",
        terminal_reason="cancel_requested",
    )

    add_run(
        run_id="run-done",
        experiment_id="exp-done",
        status="FINISHED",
        created_at="2026-03-25T09:45:00+00:00",
        finished_at="2026-03-25T10:14:00+00:00",
        config_id="configs/done.json",
    )
    add_job(
        job_id="job-done",
        run_id="run-done",
        experiment_id="exp-done",
        logical_run_id="logical-done",
        config_id="configs/done.json",
        status="completed",
        created_at="2026-03-25T09:45:00+00:00",
        started_at="2026-03-25T09:45:01+00:00",
        finished_at="2026-03-25T10:14:00+00:00",
        updated_at="2026-03-25T10:14:00+00:00",
        terminal_reason="success",
    )
    add_lifecycle(
        run_id="run-done",
        job_id="job-done",
        logical_run_id="logical-done",
        experiment_id="exp-done",
        config_id="configs/done.json",
        status="completed",
        created_at="2026-03-25T09:45:00+00:00",
        started_at="2026-03-25T09:45:01+00:00",
        finished_at="2026-03-25T10:14:00+00:00",
        updated_at="2026-03-25T10:14:00+00:00",
        current_stage="finalize_manifest",
        progress_percent=100.0,
        progress_label="Finalized",
        progress_current=1,
        progress_total=1,
        terminal_reason="success",
    )

    add_run(
        run_id="run-ghost-active",
        experiment_id="exp-done",
        status="RUNNING",
        created_at="2026-03-25T09:50:00+00:00",
        finished_at=None,
        config_id="configs/ghost_active.json",
    )
    add_job(
        job_id="job-ghost-active",
        run_id="run-ghost-active",
        experiment_id="exp-done",
        logical_run_id="logical-ghost-active",
        config_id="configs/ghost_active.json",
        status="running",
        created_at="2026-03-25T09:50:00+00:00",
        started_at="2026-03-25T09:50:01+00:00",
        updated_at="2026-03-25T10:14:30+00:00",
    )

    with sqlite3.connect(init_result.db_path) as conn:
        conn.executemany(
            """
            INSERT INTO runs (
                run_id,
                run_hash,
                status,
                run_type,
                created_at,
                finished_at,
                config_hash,
                experiment_id,
                run_path,
                manifest_json,
                manifest_mtime_ns,
                ingested_at
            ) VALUES (
                :run_id,
                :run_hash,
                :status,
                :run_type,
                :created_at,
                :finished_at,
                :config_hash,
                :experiment_id,
                :run_path,
                :manifest_json,
                :manifest_mtime_ns,
                :ingested_at
            )
            """,
            run_rows,
        )
        conn.executemany(
            """
            INSERT INTO run_jobs (
                job_id,
                batch_id,
                experiment_id,
                logical_run_id,
                operation_type,
                attempt_no,
                attempt_id,
                config_id,
                config_source,
                config_path,
                config_sha256,
                request_json,
                job_type,
                status,
                queue_name,
                priority,
                created_at,
                queued_at,
                started_at,
                finished_at,
                updated_at,
                worker_id,
                pid,
                exit_code,
                signal,
                backend,
                tier,
                budget,
                timeout_seconds,
                canonical_run_id,
                external_run_id,
                run_dir,
                cancel_requested,
                cancel_requested_at,
                terminal_reason,
                terminal_detail_json,
                error_json
            ) VALUES (
                :job_id,
                :batch_id,
                :experiment_id,
                :logical_run_id,
                :operation_type,
                :attempt_no,
                :attempt_id,
                :config_id,
                :config_source,
                :config_path,
                :config_sha256,
                :request_json,
                :job_type,
                :status,
                :queue_name,
                :priority,
                :created_at,
                :queued_at,
                :started_at,
                :finished_at,
                :updated_at,
                :worker_id,
                :pid,
                :exit_code,
                :signal,
                :backend,
                :tier,
                :budget,
                :timeout_seconds,
                :canonical_run_id,
                :external_run_id,
                :run_dir,
                :cancel_requested,
                :cancel_requested_at,
                :terminal_reason,
                :terminal_detail_json,
                :error_json
            )
            """,
            run_jobs,
        )
        conn.executemany(
            """
            INSERT INTO run_lifecycles (
                run_id,
                run_hash,
                config_hash,
                job_id,
                logical_run_id,
                attempt_id,
                attempt_no,
                source,
                operation_type,
                job_type,
                status,
                experiment_id,
                config_id,
                config_source,
                config_path,
                config_sha256,
                run_dir,
                runtime_path,
                backend,
                worker_id,
                pid,
                host,
                current_stage,
                completed_stages_json,
                progress_percent,
                progress_label,
                progress_current,
                progress_total,
                cancel_requested,
                cancel_requested_at,
                created_at,
                queued_at,
                started_at,
                last_heartbeat_at,
                updated_at,
                finished_at,
                terminal_reason,
                terminal_detail_json,
                latest_metrics_json,
                latest_sample_json,
                reconciled
            ) VALUES (
                :run_id,
                :run_hash,
                :config_hash,
                :job_id,
                :logical_run_id,
                :attempt_id,
                :attempt_no,
                :source,
                :operation_type,
                :job_type,
                :status,
                :experiment_id,
                :config_id,
                :config_source,
                :config_path,
                :config_sha256,
                :run_dir,
                :runtime_path,
                :backend,
                :worker_id,
                :pid,
                :host,
                :current_stage,
                :completed_stages_json,
                :progress_percent,
                :progress_label,
                :progress_current,
                :progress_total,
                :cancel_requested,
                :cancel_requested_at,
                :created_at,
                :queued_at,
                :started_at,
                :last_heartbeat_at,
                :updated_at,
                :finished_at,
                :terminal_reason,
                :terminal_detail_json,
                :latest_metrics_json,
                :latest_sample_json,
                :reconciled
            )
            """,
            lifecycles,
        )
        conn.commit()

    return store_root


def test_get_experiments_overview_summarizes_live_and_terminal_state(tmp_path: Path) -> None:
    store_root = _seed_experiments_overview_store(tmp_path)
    adapter = VizStoreAdapter(VizStoreConfig(store_root=store_root, repo_root=tmp_path))

    payload = adapter.get_experiments_overview()

    assert payload["summary"] == {
        "total_experiments": 5,
        "active_experiments": 3,
        "completed_experiments": 2,
        "live_experiments": 2,
        "live_runs": 3,
        "queued_runs": 1,
        "attention_count": 3,
    }
    assert [item["experiment_id"] for item in payload["experiments"]] == [
        "exp-live-alert",
        "exp-live",
        "exp-stale",
        "exp-canceled",
        "exp-done",
    ]
    assert payload["experiments"][0]["attention_state"] == "failed"
    assert payload["experiments"][1]["has_live"] is True
    assert payload["experiments"][2]["attention_state"] == "stale"
    assert payload["experiments"][3]["attention_state"] == "canceled"
    assert payload["experiments"][4]["attention_state"] == "none"
    assert payload["experiments"][4]["has_live"] is False

    assert [item["experiment_id"] for item in payload["live_experiments"]] == [
        "exp-live-alert",
        "exp-live",
    ]
    assert payload["live_experiments"][0]["attention_state"] == "failed"
    assert payload["live_experiments"][0]["aggregate_progress_percent"] == pytest.approx(72.0)
    assert payload["live_experiments"][1]["aggregate_progress_percent"] == pytest.approx(19.0)
    assert payload["live_experiments"][1]["aggregate_progress_percent"] <= 100.0
    assert payload["live_experiments"][1]["queued_run_count"] == 1
    assert [run["status"] for run in payload["live_experiments"][1]["runs"]] == ["running", "queued"]
    assert payload["live_experiments"][1]["runs"][0]["progress_label"] == "Fold 3 of 7"
    assert all(item["experiment_id"] != "exp-done" for item in payload["live_experiments"])

    assert [item["status"] for item in payload["recent_activity"]] == [
        "completed",
        "canceled",
        "stale",
        "failed",
    ]
    assert payload["recent_activity"][1]["terminal_reason"] == "cancel_requested"
    assert payload["recent_activity"][2]["terminal_reason"] == "reconciled_stale"
    assert payload["recent_activity"][3]["experiment_name"] == "Live Alert"


def test_build_monitor_snapshot_ignores_active_looking_jobs_without_live_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = _seed_experiments_overview_store(tmp_path)
    init_result = init_store_db(store_root=store_root)
    monkeypatch.setattr("numereng_viz.monitor_snapshot.reconcile_run_lifecycles", lambda **_: None)
    with sqlite3.connect(init_result.db_path) as conn:
        conn.execute(
            """
            INSERT INTO run_jobs (
                job_id,
                batch_id,
                experiment_id,
                logical_run_id,
                operation_type,
                attempt_no,
                attempt_id,
                config_id,
                config_source,
                config_path,
                config_sha256,
                request_json,
                job_type,
                status,
                queue_name,
                priority,
                created_at,
                queued_at,
                started_at,
                finished_at,
                updated_at,
                worker_id,
                pid,
                exit_code,
                signal,
                backend,
                tier,
                budget,
                timeout_seconds,
                canonical_run_id,
                external_run_id,
                run_dir,
                cancel_requested,
                cancel_requested_at,
                terminal_reason,
                terminal_detail_json,
                error_json
            ) VALUES (
                'job-ghost',
                'batch-ghost',
                'exp-done',
                'logical-ghost',
                'run',
                1,
                'attempt-ghost',
                'ghost.json',
                'config',
                '.numereng/experiments/exp-done/configs/ghost.json',
                'sha256',
                '{}',
                'run',
                'running',
                'default',
                0,
                '2026-03-25T10:12:00+00:00',
                '2026-03-25T10:12:00+00:00',
                '2026-03-25T10:12:01+00:00',
                NULL,
                '2026-03-25T10:12:02+00:00',
                'worker-ghost',
                9999,
                NULL,
                NULL,
                'local',
                NULL,
                NULL,
                NULL,
                'run-ghost',
                NULL,
                '.numereng/runs/run-ghost',
                0,
                NULL,
                NULL,
                '{}',
                NULL
            )
            """
        )
        conn.commit()

    payload = build_monitor_snapshot(store_root=store_root, refresh_cloud=False)

    assert payload["summary"]["live_runs"] == 3
    assert all(item["experiment_id"] != "exp-done" for item in payload["live_experiments"])


def test_remote_snapshot_coordinator_uses_one_cycle_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    target = SshRemoteTargetProfile.model_validate(
        {
            "id": "remote-pc",
            "label": "Remote PC",
            "kind": "ssh",
            "ssh_config_host": "remote-pc",
            "repo_root": "/srv/numereng",
            "store_root": "/srv/numereng/.numereng",
        }
    )
    monkeypatch.setattr("numereng_viz.monitor_snapshot.load_remote_targets", lambda: [target])

    class _SuccessResult:
        returncode = 0
        stdout = (
            json.dumps(
                {
                    "generated_at": "2026-03-25T00:00:00+00:00",
                    "source": {
                        "kind": "local",
                        "id": "local",
                        "label": "Local store",
                        "host": "remote",
                        "store_root": "/srv/numereng/.numereng",
                        "state": "live",
                    },
                    "summary": {
                        "total_experiments": 0,
                        "active_experiments": 0,
                        "completed_experiments": 0,
                        "live_experiments": 1,
                        "live_runs": 1,
                        "queued_runs": 0,
                        "attention_count": 0,
                    },
                    "experiments": [],
                    "live_experiments": [
                        {
                            "experiment_id": "remote-exp",
                            "name": "Remote Experiment",
                            "status": "active",
                            "tags": [],
                            "live_run_count": 1,
                            "queued_run_count": 0,
                            "attention_state": "none",
                            "latest_activity_at": "2026-03-25T00:00:00+00:00",
                            "aggregate_progress_percent": 25.0,
                            "runs": [
                                {
                                    "run_id": "run-1",
                                    "config_label": "base.json",
                                    "status": "running",
                                    "current_stage": "train_model",
                                    "progress_percent": 25.0,
                                    "progress_label": "Fold 1 of 4",
                                    "updated_at": "2026-03-25T00:00:00+00:00",
                                    "terminal_reason": None,
                                }
                            ],
                        }
                    ],
                    "recent_activity": [],
                }
            )
            + "\n"
        )
        stderr = ""

    class _FailureResult:
        returncode = 1
        stdout = ""
        stderr = "ssh failed"

    coordinator = RemoteSnapshotCoordinator(cache_ttl_seconds=60.0)
    monkeypatch.setattr("numereng_viz.monitor_snapshot.subprocess.run", lambda *args, **kwargs: _SuccessResult())
    first = coordinator.fetch_snapshots()
    assert first[0]["source"]["state"] == "live"

    monkeypatch.setattr("numereng_viz.monitor_snapshot.subprocess.run", lambda *args, **kwargs: _FailureResult())
    second = coordinator.fetch_snapshots()
    assert second[0]["source"]["state"] == "cached"


def test_remote_snapshot_coordinator_builds_posix_ssh_command() -> None:
    target = SshRemoteTargetProfile.model_validate(
        {
            "id": "remote-posix",
            "label": "Remote Posix",
            "kind": "ssh",
            "ssh_config_host": "remote-posix",
            "shell": "posix",
            "repo_root": "/srv/numereng",
            "store_root": "/srv/numereng/.numereng",
        }
    )

    coordinator = RemoteSnapshotCoordinator()
    command = coordinator._ssh_command(target)

    assert command[-1] == (
        "cd /srv/numereng && uv run numereng monitor snapshot --store-root /srv/numereng/.numereng --json"
    )


def test_remote_snapshot_coordinator_builds_powershell_ssh_command() -> None:
    target = SshRemoteTargetProfile.model_validate(
        {
            "id": "remote-pc",
            "label": "Remote PC",
            "kind": "ssh",
            "ssh_config_host": "pc",
            "shell": "powershell",
            "repo_root": r"C:\Users\<you>\remote-access\numereng",
            "store_root": r"C:\Users\<you>\remote-access\numereng\.numereng",
        }
    )

    coordinator = RemoteSnapshotCoordinator()
    command = coordinator._ssh_command(target)

    assert command[-1] == (
        'powershell -NoProfile -Command "'
        r"Set-Location 'C:\Users\<you>\remote-access\numereng'; "
        r"uv run numereng monitor snapshot --store-root 'C:\Users\<you>\remote-access\numereng\.numereng' --json"
        '"'
    )


def test_remote_snapshot_coordinator_emits_unavailable_source_with_bootstrap_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = SshRemoteTargetProfile.model_validate(
        {
            "id": "remote-pc",
            "label": "Remote PC",
            "kind": "ssh",
            "ssh_config_host": "pc",
            "shell": "powershell",
            "repo_root": r"C:\Users\<you>\remote-access\numereng",
            "store_root": r"C:\Users\<you>\remote-access\numereng\.numereng",
        }
    )

    class _FailureResult:
        returncode = 1
        stdout = ""
        stderr = "ssh failed"

    monkeypatch.setattr("numereng_viz.monitor_snapshot.load_remote_targets", lambda: [target])
    monkeypatch.setattr("numereng_viz.monitor_snapshot.subprocess.run", lambda *args, **kwargs: _FailureResult())
    monkeypatch.setattr(
        "numereng_viz.monitor_snapshot.load_viz_bootstrap_state",
        lambda **_: remote_service.RemoteVizBootstrapResult(
            store_root=tmp_path / ".numereng",
            state_path=tmp_path / ".numereng" / "remote_ops" / "bootstrap" / "viz.json",
            bootstrapped_at="2026-03-28T00:00:00+00:00",
            ready_count=0,
            degraded_count=1,
            targets=(
                remote_service.RemoteVizBootstrapTargetResult(
                    target=remote_service.RemoteTargetRecord(
                        id="remote-pc",
                        label="Remote PC",
                        kind="ssh",
                        shell="powershell",
                        repo_root=r"C:\Users\<you>\remote-access\numereng",
                        store_root=r"C:\Users\<you>\remote-access\numereng\.numereng",
                        runner_cmd="uv run numereng",
                        python_cmd="uv run python",
                        tags=("pc",),
                    ),
                    bootstrap_status="degraded",
                    last_bootstrap_at="2026-03-28T00:00:00+00:00",
                    last_bootstrap_error="monitor_snapshot_failed",
                    repo_synced=False,
                    repo_sync_skipped=True,
                    doctor_ok=False,
                    issues=("monitor_snapshot_failed",),
                ),
            ),
        ),
    )

    coordinator = RemoteSnapshotCoordinator(store_root=tmp_path / ".numereng")
    snapshots = coordinator.fetch_snapshots()

    assert snapshots[0]["source"]["state"] == "unavailable"
    assert snapshots[0]["source"]["bootstrap_status"] == "degraded"
    assert snapshots[0]["source"]["last_bootstrap_error"] == "monitor_snapshot_failed"
    assert snapshots[0]["live_experiments"] == []


def test_build_monitor_snapshot_repairs_cloud_experiment_context_from_state_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / ".numereng"
    init_store_db(store_root=store_root)
    upsert_experiment(
        store_root=store_root,
        experiment_id="exp-live",
        name="Live Experiment",
        status="active",
        created_at="2026-03-27T10:00:00+00:00",
        updated_at="2026-03-27T10:00:00+00:00",
        metadata={"tags": ["live"]},
    )
    config_path = store_root / "experiments" / "exp-live" / "configs" / "train.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{}", encoding="utf-8")
    upsert_cloud_job(
        store_root=store_root,
        job=StoreCloudJobUpsert(
            run_id="run-cloud-1",
            provider="sagemaker",
            backend="sagemaker",
            provider_job_id="job-cloud-1",
            status="InProgress",
            region="us-east-2",
            metadata_json=json.dumps({"provider": "sagemaker", "backend": "sagemaker", "status": "InProgress"}),
        ),
    )
    cloud_dir = store_root / "cloud"
    cloud_dir.mkdir(parents=True, exist_ok=True)
    (cloud_dir / "run-cloud-1.json").write_text(
        json.dumps(
            {
                "run_id": "run-cloud-1",
                "backend": "sagemaker",
                "region": "us-east-2",
                "training_job_name": "job-cloud-1",
                "status": "InProgress",
                "metadata": {
                    "experiment_id": "exp-live",
                    "config_path": str(config_path),
                    "config_id": str(config_path),
                    "config_label": "train.json",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("numereng_viz.monitor_snapshot.reconcile_run_lifecycles", lambda **_: None)

    class _FakeCloudService:
        def train_status(self, request):
            _ = request
            return type(
                "_Response",
                (),
                {
                    "state": type(
                        "_State",
                        (),
                        {
                            "last_updated_at": "2026-03-27T10:05:00+00:00",
                            "metadata": {
                                "experiment_id": "exp-live",
                                "config_path": str(config_path),
                                "config_id": str(config_path),
                                "config_label": "train.json",
                            },
                        },
                    )(),
                    "result": {
                        "status": "InProgress",
                        "secondary_status": "Training",
                        "failure_reason": None,
                    },
                },
            )()

    monkeypatch.setattr("numereng_viz.monitor_snapshot.CloudAwsManagedService", lambda: _FakeCloudService())

    payload = build_monitor_snapshot(store_root=store_root, refresh_cloud=True)

    assert payload["summary"]["live_runs"] == 1
    assert payload["live_experiments"][0]["experiment_id"] == "exp-live"
    assert payload["live_experiments"][0]["runs"][0]["config_label"] == "train.json"
    assert payload["live_experiments"][0]["runs"][0]["provider_run_id"] == "job-cloud-1"


def test_build_monitor_snapshot_refreshes_only_active_cloud_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / ".numereng"
    init_store_db(store_root=store_root)
    upsert_experiment(
        store_root=store_root,
        experiment_id="exp-live",
        name="Live Experiment",
        status="active",
        created_at="2026-03-27T10:00:00+00:00",
        updated_at="2026-03-27T10:00:00+00:00",
        metadata={"tags": ["live"]},
    )
    upsert_cloud_job(
        store_root=store_root,
        job=StoreCloudJobUpsert(
            run_id="run-cloud-live",
            provider="sagemaker",
            backend="sagemaker",
            provider_job_id="job-cloud-live",
            status="InProgress",
            region="us-east-2",
            metadata_json=json.dumps({"experiment_id": "exp-live", "config_label": "live.json"}),
        ),
    )
    upsert_cloud_job(
        store_root=store_root,
        job=StoreCloudJobUpsert(
            run_id="run-cloud-done",
            provider="sagemaker",
            backend="sagemaker",
            provider_job_id="job-cloud-done",
            status="Completed",
            region="us-east-2",
            metadata_json=json.dumps({"experiment_id": "exp-live", "config_label": "done.json"}),
        ),
    )
    monkeypatch.setattr("numereng_viz.monitor_snapshot.reconcile_run_lifecycles", lambda **_: None)
    calls: list[str] = []

    class _FakeCloudService:
        def train_status(self, request):
            calls.append(request.run_id or "")
            return type(
                "_Response",
                (),
                {
                    "state": type("_State", (), {"last_updated_at": "2026-03-27T10:05:00+00:00", "metadata": {}})(),
                    "result": {
                        "status": "InProgress",
                        "secondary_status": "Training",
                        "failure_reason": None,
                    },
                },
            )()

    monkeypatch.setattr("numereng_viz.monitor_snapshot.CloudAwsManagedService", lambda: _FakeCloudService())

    payload = build_monitor_snapshot(store_root=store_root, refresh_cloud=True)

    assert calls == ["run-cloud-live"]
    assert payload["summary"]["live_runs"] == 1
    assert payload["live_experiments"][0]["runs"][0]["run_id"] == "run-cloud-live"
    assert payload["recent_activity"][0]["run_id"] == "run-cloud-done"
    assert payload["recent_activity"][0]["status"] == "completed"
    assert payload["recent_activity"][0]["progress_percent"] == 100.0
    assert payload["recent_activity"][0]["progress_mode"] == "estimated"


def test_build_monitor_snapshot_marks_local_lifecycle_progress_as_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = _seed_experiments_overview_store(tmp_path)
    monkeypatch.setattr("numereng_viz.monitor_snapshot.reconcile_run_lifecycles", lambda **_: None)

    payload = build_monitor_snapshot(store_root=store_root, refresh_cloud=False)

    live_experiment = next(item for item in payload["live_experiments"] if item["experiment_id"] == "exp-live")
    assert live_experiment["runs"][0]["progress_mode"] == "exact"
    assert live_experiment["runs"][0]["progress_percent"] == pytest.approx(38.0)
    assert live_experiment["runs"][1]["progress_mode"] == "exact"
    assert live_experiment["runs"][1]["progress_percent"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("secondary_status", "expected_progress"),
    [
        ("Starting", 8.0),
        ("Downloading", 22.0),
        ("Training", 68.0),
        ("Uploading", 92.0),
    ],
)
def test_build_monitor_snapshot_estimates_sagemaker_progress_from_secondary_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    secondary_status: str,
    expected_progress: float,
) -> None:
    store_root = tmp_path / ".numereng"
    init_store_db(store_root=store_root)
    upsert_experiment(
        store_root=store_root,
        experiment_id="exp-live",
        name="Live Experiment",
        status="active",
        created_at="2026-03-27T10:00:00+00:00",
        updated_at="2026-03-27T10:00:00+00:00",
        metadata={"tags": ["live"]},
    )
    upsert_cloud_job(
        store_root=store_root,
        job=StoreCloudJobUpsert(
            run_id="run-cloud-live",
            provider="sagemaker",
            backend="sagemaker",
            provider_job_id="job-cloud-live",
            status="InProgress",
            region="us-east-2",
            metadata_json=json.dumps({"experiment_id": "exp-live", "config_label": "live.json"}),
        ),
    )
    monkeypatch.setattr("numereng_viz.monitor_snapshot.reconcile_run_lifecycles", lambda **_: None)

    class _FakeCloudService:
        def train_status(self, request):
            _ = request
            return type(
                "_Response",
                (),
                {
                    "state": type("_State", (), {"last_updated_at": "2026-03-27T10:05:00+00:00", "metadata": {}})(),
                    "result": {
                        "status": "InProgress",
                        "secondary_status": secondary_status,
                        "failure_reason": None,
                    },
                },
            )()

    monkeypatch.setattr("numereng_viz.monitor_snapshot.CloudAwsManagedService", lambda: _FakeCloudService())

    payload = build_monitor_snapshot(store_root=store_root, refresh_cloud=True)

    run = payload["live_experiments"][0]["runs"][0]
    assert run["progress_percent"] == expected_progress
    assert run["progress_mode"] == "estimated"
    assert run["progress_label"] == secondary_status


def test_build_monitor_snapshot_preserves_terminal_cloud_progress_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / ".numereng"
    init_store_db(store_root=store_root)
    upsert_experiment(
        store_root=store_root,
        experiment_id="exp-live",
        name="Live Experiment",
        status="active",
        created_at="2026-03-27T10:00:00+00:00",
        updated_at="2026-03-27T10:00:00+00:00",
        metadata={"tags": ["live"]},
    )
    upsert_cloud_job(
        store_root=store_root,
        job=StoreCloudJobUpsert(
            run_id="run-cloud-failed",
            provider="sagemaker",
            backend="sagemaker",
            provider_job_id="job-cloud-failed",
            status="Failed",
            region="us-east-2",
            metadata_json=json.dumps(
                {
                    "experiment_id": "exp-live",
                    "config_label": "failed.json",
                    "last_progress_percent": "68.0",
                }
            ),
        ),
    )
    monkeypatch.setattr("numereng_viz.monitor_snapshot.reconcile_run_lifecycles", lambda **_: None)

    payload = build_monitor_snapshot(store_root=store_root, refresh_cloud=True)

    activity = payload["recent_activity"][0]
    assert activity["run_id"] == "run-cloud-failed"
    assert activity["status"] == "failed"
    assert activity["progress_percent"] == 68.0
    assert activity["progress_mode"] == "estimated"


def test_experiments_overview_route_merges_remote_snapshots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store_root = _seed_experiments_overview_store(tmp_path)
    adapter = VizStoreAdapter(VizStoreConfig(store_root=store_root, repo_root=tmp_path))
    service = VizService(adapter)
    monkeypatch.setattr("numereng_viz.services.reconcile_run_lifecycles", lambda **_: None)
    monkeypatch.setattr("numereng_viz.monitor_snapshot.reconcile_run_lifecycles", lambda **_: None)
    service.remote_snapshots.fetch_snapshots = lambda: [
        {
            "source": {
                "kind": "ssh",
                "id": "remote-pc",
                "label": "Remote PC",
                "host": "remote-pc",
                "store_root": "/srv/numereng/.numereng",
                "state": "live",
            },
            "summary": {
                "total_experiments": 1,
                "active_experiments": 1,
                "completed_experiments": 0,
                "live_experiments": 1,
                "live_runs": 1,
                "queued_runs": 0,
                "attention_count": 0,
            },
            "experiments": [
                {
                    "experiment_id": "remote-exp",
                    "name": "Remote Experiment",
                    "status": "active",
                    "created_at": "2026-03-25T00:00:00+00:00",
                    "updated_at": "2026-03-25T00:00:00+00:00",
                    "run_count": 1,
                    "tags": ["remote"],
                    "has_live": True,
                    "live_run_count": 1,
                    "attention_state": "none",
                    "latest_activity_at": "2026-03-25T00:00:00+00:00",
                    "source_kind": "ssh",
                    "source_id": "remote-pc",
                    "source_label": "Remote PC",
                    "detail_href": None,
                }
            ],
            "live_experiments": [
                {
                    "experiment_id": "remote-exp",
                    "name": "Remote Experiment",
                    "status": "active",
                    "tags": ["remote"],
                    "live_run_count": 1,
                    "queued_run_count": 0,
                    "attention_state": "none",
                    "latest_activity_at": "2026-03-25T00:00:00+00:00",
                    "aggregate_progress_percent": 25.0,
                    "source_kind": "ssh",
                    "source_id": "remote-pc",
                    "source_label": "Remote PC",
                    "detail_href": None,
                    "runs": [
                        {
                            "run_id": "remote-run-1",
                            "config_label": "base.json",
                            "status": "running",
                            "current_stage": "train_model",
                            "progress_percent": 25.0,
                            "progress_label": "Fold 1 of 4",
                            "updated_at": "2026-03-25T00:00:00+00:00",
                            "terminal_reason": None,
                            "source_kind": "ssh",
                            "source_id": "remote-pc",
                            "source_label": "Remote PC",
                            "backend": "local",
                            "provider_run_id": None,
                            "detail_href": None,
                        }
                    ],
                }
            ],
            "recent_activity": [],
        }
    ]
    app = FastAPI()
    app.include_router(create_router(service))
    client = TestClient(app)

    response = client.get("/api/experiments/overview")

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["live_experiments"] == 3
    assert any(item["experiment_id"] == "remote-exp" for item in payload["experiments"])
    remote_live = next(item for item in payload["live_experiments"] if item["experiment_id"] == "remote-exp")
    assert remote_live["source_kind"] == "ssh"
    assert remote_live["runs"][0]["detail_href"] is None


def test_experiments_overview_route_returns_mission_control_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = _seed_experiments_overview_store(tmp_path)
    monkeypatch.setattr(
        "numereng_viz.services.reconcile_run_lifecycles",
        lambda **_: None,
    )
    monkeypatch.setattr(
        "numereng_viz.monitor_snapshot.reconcile_run_lifecycles",
        lambda **_: None,
    )

    client = _make_client(repo_root=tmp_path, store_root=store_root)
    response = client.get("/api/experiments/overview")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    payload = response.json()
    assert payload["summary"]["live_experiments"] == 2
    assert payload["experiments"][0]["experiment_id"] == "exp-live-alert"
    assert payload["live_experiments"][1]["runs"][1]["status"] == "queued"
    assert payload["recent_activity"][0]["status"] == "completed"


def test_numereng_docs_tree_uses_docs_numereng_root_only(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)

    docs_root = tmp_path / "docs"
    docs_root.mkdir(parents=True)
    (docs_root / "CUSTOM_MODELS.md").write_text("# Root-only doc\n")
    (docs_root / "SUMMARY.md").write_text("## Root\n* [Custom Models](CUSTOM_MODELS.md)\n")

    numereng_docs = docs_root / "numereng"
    numereng_docs.mkdir(parents=True)
    (numereng_docs / "README.md").write_text("# Numereng\n")
    (numereng_docs / "SUMMARY.md").write_text("## Getting Started\n* [Overview](README.md)\n")

    adapter = VizStoreAdapter(VizStoreConfig(store_root=store_root, repo_root=tmp_path))

    tree = adapter.get_doc_tree("numereng")
    assert tree["sections"][0]["heading"] == "Getting Started"
    assert tree["sections"][0]["items"][0]["path"] == "README.md"
    assert adapter.get_doc_content("numereng", "README.md")["exists"] is True
    assert adapter.get_doc_content("numereng", "CUSTOM_MODELS.md")["exists"] is False


def test_summary_tree_normalizes_relative_paths_and_blocks_escape(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)

    docs_numereng = tmp_path / "docs" / "numereng"
    docs_numereng.mkdir(parents=True)
    (docs_numereng / "README.md").write_text("# Overview\n")
    (docs_numereng / "reference").mkdir(parents=True)
    (docs_numereng / "reference" / "signals-+-quantconnect.md").write_text("# Signals\n")
    (docs_numereng / "SUMMARY.md").write_text(
        "\n".join(
            [
                "# Guide",
                "## Reference",
                "* [Overview](./guides/../README.md)",
                "* [Signals + QuantConnect](reference/signals-+-quantconnect.md)",
                "* [Escape](../outside.md)",
            ]
        )
        + "\n"
    )

    adapter = VizStoreAdapter(VizStoreConfig(store_root=store_root, repo_root=tmp_path))
    tree = adapter.get_doc_tree("numereng")
    items = tree["sections"][0]["items"]
    paths = [item["path"] for item in items]

    assert "README.md" in paths
    assert "reference/signals-+-quantconnect.md" in paths
    assert "../outside.md" not in paths
    assert adapter.get_doc_content("numereng", "reference/signals-+-quantconnect.md")["exists"] is True


def test_numerai_docs_tree_appends_generated_forum_archive(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)

    numerai_docs = tmp_path / "docs" / "numerai"
    numerai_docs.mkdir(parents=True)
    (numerai_docs / "SUMMARY.md").write_text("## Root\n* [Overview](README.md)\n")
    (numerai_docs / "README.md").write_text("# Numerai\n")
    forum_dir = numerai_docs / "forum"
    forum_dir.mkdir(parents=True)
    (forum_dir / "INDEX.md").write_text("# Forum Archive\n")
    (forum_dir / "posts" / "2020" / "03").mkdir(parents=True)
    (forum_dir / "posts" / "2020" / "03" / "INDEX.md").write_text("# 2020/03\n")
    (forum_dir / "posts" / "2020" / "INDEX.md").write_text("# 2020\n")

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path,
        )
    )

    tree = adapter.get_doc_tree("numerai")
    sections = tree["sections"]
    forum_section = next(section for section in sections if section["heading"] == "Forum Archive")
    forum_item = forum_section["items"][0]

    assert forum_item["title"] == "2020"
    assert forum_item["path"] == "forum/posts/2020/INDEX.md"
    assert forum_item["children"][0]["title"] == "2020/03"
    assert forum_item["children"][0]["path"] == "forum/posts/2020/03/INDEX.md"
    assert adapter.get_doc_content("numerai", "forum/INDEX.md")["exists"] is True


def test_docs_asset_endpoint_serves_domain_and_shared_assets(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)

    numerai_docs = tmp_path / "docs" / "numerai"
    (numerai_docs / ".gitbook" / "assets").mkdir(parents=True)
    (numerai_docs / ".gitbook" / "assets" / "domain.png").write_bytes(b"domain-image")

    shared_assets = tmp_path / "docs" / "assets"
    shared_assets.mkdir(parents=True)
    (shared_assets / "shared.png").write_bytes(b"shared-image")

    client = _make_readonly_client(repo_root=tmp_path, store_root=store_root)

    domain_resp = client.get("/api/docs/numerai/asset", params={"path": ".gitbook/assets/domain.png"})
    assert domain_resp.status_code == 200
    assert domain_resp.content == b"domain-image"

    shared_resp = client.get("/api/docs/numerai/asset", params={"path": ".gitbook/assets/shared.png"})
    assert shared_resp.status_code == 200
    assert shared_resp.content == b"shared-image"

    missing_resp = client.get("/api/docs/numerai/asset", params={"path": ".gitbook/assets/missing.png"})
    assert missing_resp.status_code == 404

    traversal_resp = client.get("/api/docs/numerai/asset", params={"path": "../secret.png"})
    assert traversal_resp.status_code == 400


def test_get_resolved_config_prefers_json_when_present(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-1"
    run_dir.mkdir(parents=True)

    (run_dir / "resolved.json").write_text('{"foo": "bar"}')
    (run_dir / "resolved.yaml").write_text("foo: legacy")

    adapter = VizStoreAdapter(VizStoreConfig(store_root=store_root, repo_root=tmp_path))

    payload = adapter.get_resolved_config("run-1")

    assert payload == {"yaml": '{"foo": "bar"}'}


def test_get_metrics_for_runs_normalizes_store_aliases(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)
    db_path = store_root / "numereng.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE metrics (run_id TEXT, name TEXT, value REAL, value_json TEXT)")
        conn.execute(
            "INSERT INTO metrics (run_id, name, value, value_json) VALUES (?, ?, ?, ?)",
            ("run-1", "corr.mean", 0.10, None),
        )
        conn.execute(
            "INSERT INTO metrics (run_id, name, value, value_json) VALUES (?, ?, ?, ?)",
            ("run-1", "corr.sharpe", 1.25, None),
        )
        conn.execute(
            "INSERT INTO metrics (run_id, name, value, value_json) VALUES (?, ?, ?, ?)",
            ("run-1", "fnc.mean", 0.04, None),
        )
        conn.execute(
            "INSERT INTO metrics (run_id, name, value, value_json) VALUES (?, ?, ?, ?)",
            ("run-1", "mmc.mean", 0.02, None),
        )
        conn.execute(
            "INSERT INTO metrics (run_id, name, value, value_json) VALUES (?, ?, ?, ?)",
            ("run-1", "bmc.mean", 0.03, None),
        )
        conn.commit()
    finally:
        conn.close()

    adapter = VizStoreAdapter(VizStoreConfig(store_root=store_root, repo_root=tmp_path))

    payload = adapter.get_metrics_for_runs(
        ["run-1"],
        ["corr_mean", "corr_sharpe", "fnc_mean", "mmc_mean", "bmc_mean"],
    )

    assert payload["run-1"]["corr_mean"] == pytest.approx(0.10)
    assert payload["run-1"]["corr_sharpe"] == pytest.approx(1.25)
    assert payload["run-1"]["fnc_mean"] == pytest.approx(0.04)
    assert payload["run-1"]["mmc_mean"] == pytest.approx(0.02)
    assert payload["run-1"]["bmc_mean"] == pytest.approx(0.03)


def test_get_run_metrics_normalizes_nested_metrics_from_filesystem(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        """
        {
          "corr": {"mean": 0.12, "sharpe": 0.8, "std": 0.2, "max_drawdown": 0.9},
          "fnc": {"mean": 0.06, "std": 0.03, "sharpe": 2.0},
          "feature_exposure": {"mean": 0.16, "std": 0.05, "sharpe": 3.2},
          "max_feature_exposure": {"mean": 0.31, "std": 0.07, "sharpe": 4.4},
          "mmc": {"mean": 0.01, "sharpe": 0.2},
          "bmc": {"mean": 0.04}
        }
        """.strip(),
        encoding="utf-8",
    )

    adapter = VizStoreAdapter(VizStoreConfig(store_root=store_root, repo_root=tmp_path))
    payload = adapter.get_run_metrics("run-1")

    assert payload is not None
    assert payload["corr_mean"] == pytest.approx(0.12)
    assert payload["corr_sharpe"] == pytest.approx(0.8)
    assert payload["fnc_mean"] == pytest.approx(0.06)
    assert payload["fnc_sharpe"] == pytest.approx(2.0)
    assert payload["mmc_mean"] == pytest.approx(0.01)
    assert payload["bmc_mean"] == pytest.approx(0.04)
    assert payload["feature_exposure_mean"] == pytest.approx(0.16)
    assert payload["feature_exposure_sharpe"] == pytest.approx(3.2)
    assert payload["max_feature_exposure"] == pytest.approx(0.31)
    assert payload["max_drawdown"] == pytest.approx(0.9)


def test_get_run_metrics_normalizes_nested_value_json_from_sqlite(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)
    db_path = store_root / "numereng.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE metrics (run_id TEXT, name TEXT, value REAL, value_json TEXT)")
        conn.execute(
            "INSERT INTO metrics (run_id, name, value, value_json) VALUES (?, ?, ?, ?)",
            ("run-1", "corr", None, '{"mean": 0.11, "sharpe": 0.9, "max_drawdown": 0.7}'),
        )
        conn.execute(
            "INSERT INTO metrics (run_id, name, value, value_json) VALUES (?, ?, ?, ?)",
            ("run-1", "fnc", None, '{"mean": 0.05, "std": 0.02, "sharpe": 2.5}'),
        )
        conn.execute(
            "INSERT INTO metrics (run_id, name, value, value_json) VALUES (?, ?, ?, ?)",
            ("run-1", "feature_exposure", None, '{"mean": 0.14, "std": 0.03, "sharpe": 4.5}'),
        )
        conn.execute(
            "INSERT INTO metrics (run_id, name, value, value_json) VALUES (?, ?, ?, ?)",
            ("run-1", "max_feature_exposure", None, '{"mean": 0.27, "std": 0.05, "sharpe": 5.4}'),
        )
        conn.execute(
            "INSERT INTO metrics (run_id, name, value, value_json) VALUES (?, ?, ?, ?)",
            ("run-1", "mmc", None, '{"mean": 0.01}'),
        )
        conn.execute(
            "INSERT INTO metrics (run_id, name, value, value_json) VALUES (?, ?, ?, ?)",
            ("run-1", "bmc", None, '{"mean": 0.02}'),
        )
        conn.commit()
    finally:
        conn.close()

    adapter = VizStoreAdapter(VizStoreConfig(store_root=store_root, repo_root=tmp_path))

    payload = adapter.get_run_metrics("run-1")

    assert payload is not None
    assert payload["corr_mean"] == pytest.approx(0.11)
    assert payload["corr_sharpe"] == pytest.approx(0.9)
    assert payload["fnc_mean"] == pytest.approx(0.05)
    assert payload["fnc_sharpe"] == pytest.approx(2.5)
    assert payload["mmc_mean"] == pytest.approx(0.01)
    assert payload["bmc_mean"] == pytest.approx(0.02)
    assert payload["feature_exposure_mean"] == pytest.approx(0.14)
    assert payload["feature_exposure_sharpe"] == pytest.approx(4.5)
    assert payload["max_feature_exposure"] == pytest.approx(0.27)
    assert payload["max_drawdown"] == pytest.approx(0.7)


def test_list_experiment_runs_filesystem_uses_target_col(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        """
        {
          "run_id": "run-1",
          "experiment_id": "exp-1",
          "created_at": "2026-02-22T00:00:00+00:00",
          "status": "FINISHED",
          "model": {"type": "LGBMRegressor"},
          "data": {"target_col": "target_ender_20", "feature_set": "small"}
        }
        """.strip(),
        encoding="utf-8",
    )
    (run_dir / "resolved.json").write_text(
        """
        {
          "model": {
            "type": "LGBMRegressor",
            "params": {"random_state": 43}
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    adapter = VizStoreAdapter(VizStoreConfig(store_root=store_root, repo_root=tmp_path))

    payload = adapter.list_experiment_runs("exp-1")
    assert len(payload) == 1
    assert payload[0]["target_train"] == "target_ender_20"
    assert payload[0]["target_col"] == "target_ender_20"
    assert payload[0]["target"] == "target_ender_20"
    assert payload[0]["seed"] == 43


def test_list_experiment_runs_includes_corr_with_benchmark_metric(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        """
        {
          "run_id": "run-1",
          "experiment_id": "exp-1",
          "created_at": "2026-02-22T00:00:00+00:00",
          "status": "FINISHED",
          "model": {"type": "LGBMRegressor"},
          "data": {"target_col": "target_ender_20", "feature_set": "small"}
        }
        """.strip(),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        """
        {
          "bmc": {"mean": 0.04, "avg_corr_with_benchmark": 0.12},
          "bmc_last_200_eras": {"mean": 0.06}
        }
        """.strip(),
        encoding="utf-8",
    )

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path,
        )
    )

    payload = adapter.list_experiment_runs("exp-1")

    assert len(payload) == 1
    assert payload[0]["metrics"]["corr_with_benchmark"] == pytest.approx(0.12)
    assert payload[0]["metrics"]["bmc_last_200_eras_mean"] == pytest.approx(0.06)


def test_list_experiment_round_results_includes_corr_with_benchmark_metric(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    exp_dir = store_root / "experiments" / "exp-1"
    (exp_dir / "configs").mkdir(parents=True)
    (exp_dir / "results").mkdir(parents=True)
    (exp_dir / "experiment.json").write_text(
        '{"experiment_id":"exp-1","name":"Experiment 1","status":"active","runs":[]}',
        encoding="utf-8",
    )
    (exp_dir / "configs" / "r1_001_base.json").write_text(
        '{"model":{"type":"LGBMRegressor"},"data":{"target":"target_ender_20","feature_set":"small"}}',
        encoding="utf-8",
    )
    (exp_dir / "results" / "r1_base_metrics.json").write_text(
        """
        {
          "bmc": {"mean": 0.04, "avg_corr_with_benchmark": 0.09},
          "bmc_last_200_eras": {"mean": 0.07}
        }
        """.strip(),
        encoding="utf-8",
    )

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path,
        )
    )

    payload = adapter.list_experiment_round_results("exp-1")

    assert len(payload) == 1
    assert payload[0]["metrics"]["corr_with_benchmark"] == pytest.approx(0.09)
    assert payload[0]["metrics"]["bmc_last_200_eras_mean"] == pytest.approx(0.07)


def test_get_run_metrics_enriches_mmc_coverage_ratio_from_provenance(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-1"
    run_dir.mkdir(parents=True)

    (run_dir / "run.json").write_text(
        '{"run_id": "run-1", "data": {"target_col": "target_ender_20"}}',
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text('{"corr": {"mean": 0.12}, "mmc": {"mean": 0.01}}', encoding="utf-8")
    (run_dir / "score_provenance.json").write_text(
        """
        {
          "joins": {
            "predictions_rows": 100,
            "meta_overlap_rows": 80
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    adapter = VizStoreAdapter(VizStoreConfig(store_root=store_root, repo_root=tmp_path))

    payload = adapter.get_run_metrics("run-1")
    assert payload is not None
    assert payload["mmc_coverage_ratio_rows"] == pytest.approx(0.8)


def test_per_era_fallback_derives_corr_rows(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-1"
    predictions_dir = run_dir / "artifacts" / "predictions"
    predictions_dir.mkdir(parents=True)

    predictions_path = predictions_dir / "preds.parquet"
    meta_model_path = store_root / "datasets" / "v5.2" / "meta_model.parquet"
    meta_model_path.parent.mkdir(parents=True)

    predictions = pd.DataFrame(
        {
            "id": [f"id-{idx}" for idx in range(10)],
            "era": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "target_ender_20": [0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, 0.25, 0.0],
            "prediction": [0.1, 0.2, 0.3, 0.45, 0.6, 0.6, 0.45, 0.3, 0.2, 0.1],
        }
    )
    predictions.to_parquet(predictions_path, index=False)

    meta = pd.DataFrame(
        {
            "id": [f"id-{idx}" for idx in range(10)],
            "era": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "numerai_meta_model": [0.2, 0.1, 0.4, 0.35, 0.5, 0.5, 0.25, 0.4, 0.15, 0.3],
        }
    )
    meta.set_index("id").to_parquet(meta_model_path, index=True)

    (run_dir / "run.json").write_text(
        """
        {
          "run_id": "run-1",
          "experiment_id": "exp-1",
          "data": {
            "target_col": "target_ender_20",
            "version": "v5.2"
          },
          "artifacts": {
            "predictions": "artifacts/predictions/preds.parquet",
            "score_provenance": "score_provenance.json"
          }
        }
        """.strip(),
        encoding="utf-8",
    )
    (run_dir / "score_provenance.json").write_text(
        f"""
        {{
          "columns": {{
            "prediction_cols": ["prediction"],
            "target_col": "target_ender_20",
            "id_col": "id",
            "era_col": "era",
            "meta_model_col": "numerai_meta_model"
          }},
          "sources": {{
            "predictions": {{"path": "{predictions_path}"}},
            "meta_model": {{"path": "{meta_model_path}"}}
          }},
          "joins": {{
            "predictions_rows": 10,
            "meta_overlap_rows": 10
          }}
        }}
        """.strip(),
        encoding="utf-8",
    )

    adapter = VizStoreAdapter(VizStoreConfig(store_root=store_root, repo_root=tmp_path))

    result = adapter.get_per_era_corr_result("run-1")
    assert result.payload is not None
    assert [row["era"] for row in result.payload] == [1, 2]
    assert all(float(row["corr"]) > 0.99 for row in result.payload)
    assert result.materialize_ms >= 0
    assert result.wrote_artifact is False


def test_run_diagnostics_sources_route_returns_payload(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-1"
    run_dir.mkdir(parents=True)

    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "artifacts": {
                    "score_provenance": "score_provenance.json",
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "score_provenance.json").write_text(
        json.dumps(
            {
                "columns": {"target_col": "target_ender_20"},
                "joins": {"predictions_rows": 100},
                "sources": {},
            }
        ),
        encoding="utf-8",
    )

    client = _make_client(repo_root=tmp_path, store_root=store_root)
    response = client.get("/api/runs/run-1/diagnostics-sources")

    assert response.status_code == 200
    payload = response.json()
    assert payload["columns"]["target_col"] == "target_ender_20"
    assert payload["joins"]["predictions_rows"] == 100


def test_run_lifecycle_route_returns_runtime_snapshot_fallback(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "runtime.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "run_id": "run-1",
                "run_hash": "hash-run-1",
                "config_hash": "cfg-hash",
                "job_id": "job-1",
                "logical_run_id": "logical-1",
                "attempt_id": "attempt-1",
                "attempt_no": 1,
                "source": "cli.run.train",
                "operation_type": "run",
                "job_type": "run",
                "status": "running",
                "config": {
                    "id": "configs/base.json",
                    "source": "store",
                    "path": "configs/base.json",
                    "sha256": "sha",
                },
                "runtime": {
                    "run_dir": str(run_dir),
                    "runtime_path": str(run_dir / "runtime.json"),
                    "backend": "local",
                    "worker_id": "local",
                    "pid": 1234,
                    "host": "localhost",
                    "current_stage": "train_model",
                    "completed_stages": ["initializing", "load_data"],
                    "progress_percent": 42.5,
                    "progress_label": "Fold 2 of 4",
                    "progress_current": 1,
                    "progress_total": 4,
                    "cancel_requested": False,
                    "cancel_requested_at": None,
                    "created_at": "2026-03-24T00:00:00+00:00",
                    "queued_at": "2026-03-24T00:00:00+00:00",
                    "started_at": "2026-03-24T00:00:01+00:00",
                    "last_heartbeat_at": "2026-03-24T00:00:02+00:00",
                    "updated_at": "2026-03-24T00:00:02+00:00",
                    "finished_at": None,
                    "terminal_reason": None,
                    "terminal_detail": {},
                    "latest_metrics": {"corr_mean": 0.1},
                    "latest_sample": {"process_rss_gb": 0.4},
                    "reconciled": False,
                },
            }
        ),
        encoding="utf-8",
    )

    client = _make_client(repo_root=tmp_path, store_root=store_root)
    response = client.get("/api/runs/run-1/lifecycle")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "run-1"
    assert payload["current_stage"] == "train_model"
    assert payload["progress_percent"] == 42.5
    assert payload["latest_metrics"]["corr_mean"] == 0.1


@pytest.mark.parametrize(
    ("method", "path", "body", "expected_status"),
    [
        ("POST", "/api/run-jobs", {"config_ids": ["configs/base.json"]}, 405),
        ("POST", "/api/run-jobs/job-1/cancel", {}, 404),
        ("POST", "/api/run-jobs/job-1/retry", {}, 404),
        (
            "POST",
            "/api/hpo-jobs",
            {"study_name": "study-a", "config_path": "configs/base.json", "n_trials": 2},
            404,
        ),
        (
            "POST",
            "/api/ensemble-jobs",
            {"run_ids": ["run-a", "run-b"], "method": "rank_avg", "metric": "corr_sharpe"},
            404,
        ),
        ("PUT", "/api/experiments/exp-1/docs/EXPERIMENT.md", {"content": "hello"}, 405),
        ("PUT", "/api/runs/run-1/docs/RUN.md", {"content": "hello"}, 405),
        ("PUT", "/api/notes/content?path=foo.md", {"content": "hello"}, 405),
    ],
)
def test_removed_write_routes_are_unavailable(
    method: str,
    path: str,
    body: dict[str, object],
    expected_status: int,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)
    client = _make_client(repo_root=tmp_path, store_root=store_root)

    response = client.request(method, path, json=body)
    assert response.status_code == expected_status


@pytest.mark.parametrize(
    "path",
    [
        "/api/run-jobs/bad$id/events",
        "/api/run-jobs/bad$id/logs",
        "/api/run-jobs/bad$id/samples",
        "/api/run-jobs/bad$id/stream",
    ],
)
def test_run_job_read_endpoints_invalid_job_id_returns_400(path: str, tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)
    client = _make_client(repo_root=tmp_path, store_root=store_root)

    response = client.get(path)
    assert response.status_code == 400
    assert "Invalid job_id" in response.json()["detail"]


def test_run_job_stream_stops_when_job_disappears() -> None:
    class _StubService:
        def __init__(self) -> None:
            self._calls = 0

        def get_run_job(self, job_id: str) -> dict[str, object] | None:
            self._calls += 1
            if self._calls == 1:
                return {"job_id": job_id, "status": "running"}
            return None

        def list_run_job_events(
            self,
            job_id: str,
            *,
            after_id: int | None,
            limit: int,
        ) -> list[dict[str, object]]:
            _ = job_id
            _ = after_id
            _ = limit
            return []

        def list_run_job_logs(
            self,
            job_id: str,
            *,
            after_id: int | None,
            limit: int,
            stream: str,
        ) -> list[dict[str, object]]:
            _ = job_id
            _ = after_id
            _ = limit
            _ = stream
            return []

        def list_run_job_samples(
            self,
            job_id: str,
            *,
            after_id: int | None,
            limit: int,
        ) -> list[dict[str, object]]:
            _ = job_id
            _ = after_id
            _ = limit
            return []

    service = cast(VizService, _StubService())
    app = FastAPI()
    app.include_router(create_router(service))
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/api/run-jobs/job-1/stream")
    assert response.status_code == 200
    assert response.text == ""


def test_notes_content_accepts_plus_in_path(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    notes_root = store_root / "notes"
    notes_root.mkdir(parents=True)
    target = notes_root / "signals-+-qc.md"
    target.write_text("# Signals\n", encoding="utf-8")

    client = _make_client(repo_root=tmp_path, store_root=store_root)

    tree_response = client.get("/api/notes/tree")
    assert tree_response.status_code == 200
    tree_payload = tree_response.json()
    assert tree_payload["sections"][0]["items"][0]["path"] == "signals-+-qc.md"

    content_response = client.get("/api/notes/content", params={"path": "signals-+-qc.md"})
    assert content_response.status_code == 200
    assert content_response.json() == {"content": "# Signals\n", "exists": True}


def test_get_notes_tree_missing_root_returns_empty_without_creating_directory(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)
    notes_root = store_root / "notes"
    assert not notes_root.exists()

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path,
        )
    )

    payload = adapter.get_notes_tree()
    assert payload == {"sections": []}
    assert not notes_root.exists()


def test_get_ensemble_endpoints_invalid_id_returns_400(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)
    client = _make_readonly_client(repo_root=tmp_path, store_root=store_root)

    response = client.get("/api/ensembles/bad$id")
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid ensemble_id: bad$id"

    response = client.get("/api/ensembles/bad$id/correlations")
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid ensemble_id: bad$id"

    response = client.get("/api/ensembles/bad$id/artifacts")
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid ensemble_id: bad$id"


def test_get_ensemble_artifacts_endpoint_returns_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)
    client = _make_readonly_client(repo_root=tmp_path, store_root=store_root)
    app = cast(FastAPI, client.app)
    service = cast(VizService, app.state.viz_service)

    monkeypatch.setattr(service, "get_ensemble", lambda ensemble_id: {"ensemble_id": ensemble_id})
    monkeypatch.setattr(
        service,
        "get_ensemble_artifacts",
        lambda ensemble_id: {
            "weights": [{"run_id": "run-a", "weight": 0.5, "rank": 0}],
            "component_metrics": [],
            "era_metrics": [],
            "regime_metrics": [],
            "lineage": {"ensemble_id": ensemble_id},
            "bootstrap_metrics": None,
            "heavy_component_predictions_available": False,
            "available_files": ["weights.parquet", "lineage.json"],
        },
    )

    response = client.get("/api/ensembles/ens-1/artifacts")
    assert response.status_code == 200
    payload = response.json()
    assert payload["weights"][0]["run_id"] == "run-a"
    assert payload["lineage"]["ensemble_id"] == "ens-1"


def test_adapter_rejects_ensemble_artifacts_path_outside_store(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)
    db_path = store_root / "numereng.db"
    outside = tmp_path / "outside-artifacts"
    outside.mkdir(parents=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE ensembles (
                ensemble_id TEXT PRIMARY KEY,
                experiment_id TEXT,
                name TEXT,
                method TEXT,
                target TEXT,
                metric TEXT,
                status TEXT,
                config_json TEXT,
                artifacts_path TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO ensembles (
                ensemble_id, experiment_id, name, method, target, metric, status,
                config_json, artifacts_path, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "ens-1",
                "exp-1",
                "ens-1",
                "rank_avg",
                "target_ender_20",
                "corr_sharpe",
                "completed",
                "{}",
                str(outside),
                "2026-02-22T00:00:00+00:00",
                "2026-02-22T00:00:00+00:00",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    adapter = VizStoreAdapter(VizStoreConfig(store_root=store_root, repo_root=tmp_path))

    assert adapter.get_ensemble_artifacts("ens-1") is None
