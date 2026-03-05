from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

import numereng.features.ensemble.repo as repo_module
from numereng.features.store import init_store_db


def _insert_raw_ensemble_row(
    *,
    store_root: Path,
    ensemble_id: str,
    method: str = "rank_avg",
    status: str = "completed",
) -> None:
    init_result = init_store_db(store_root=store_root)
    stamp = datetime.now(UTC).isoformat()
    with sqlite3.connect(init_result.db_path) as conn:
        conn.execute(
            """
            INSERT INTO ensembles (
                ensemble_id, experiment_id, name, method, target, metric, status, config_json,
                artifacts_path, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ensemble_id,
                "exp-1",
                "blend-a",
                method,
                "target_ender_20",
                "corr20v2_sharpe",
                status,
                "{}",
                str(store_root / "experiments" / "exp-1" / "ensembles" / ensemble_id),
                stamp,
                stamp,
            ),
        )
        conn.commit()


def test_get_ensemble_record_rejects_invalid_persisted_method(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    _insert_raw_ensemble_row(store_root=store_root, ensemble_id="ens-invalid-method", method="forward")

    with pytest.raises(ValueError, match="ensemble_method_invalid"):
        repo_module.get_ensemble_record(store_root=store_root, ensemble_id="ens-invalid-method")


def test_list_ensemble_records_rejects_invalid_persisted_status(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    _insert_raw_ensemble_row(store_root=store_root, ensemble_id="ens-invalid-status", status="stale")

    with pytest.raises(ValueError, match="ensemble_status_invalid"):
        repo_module.list_ensemble_records(store_root=store_root)
