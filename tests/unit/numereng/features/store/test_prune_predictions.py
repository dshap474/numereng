from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from numereng.features.store import doctor_store, index_run, prune_predictions


def _write_run(
    store_root: Path,
    run_id: str,
    *,
    status: str = "FINISHED",
    experiment_id: str | None = None,
    write_predictions: bool = True,
) -> Path:
    run_dir = store_root / "runs" / run_id
    predictions_dir = run_dir / "artifacts" / "predictions"
    scoring_dir = run_dir / "artifacts" / "scoring"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    scoring_dir.mkdir(parents=True, exist_ok=True)
    if not write_predictions:
        predictions_dir.rmdir()
    else:
        (predictions_dir / "preds.parquet").write_bytes(b"PAR1")
    (scoring_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "run.log").write_text("log", encoding="utf-8")
    (run_dir / "resolved.json").write_text("{}", encoding="utf-8")
    (run_dir / "results.json").write_text("{}", encoding="utf-8")
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "score_provenance.json").write_text("{}", encoding="utf-8")

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "run_hash": f"hash-{run_id}",
        "run_type": "training",
        "status": status,
        "created_at": "2026-02-21T00:00:00+00:00",
        "finished_at": "2026-02-21T00:05:00+00:00" if status == "FINISHED" else None,
        "config": {"hash": f"cfg-{run_id}"},
        "execution": {"kind": "local", "provider": "local", "backend": "local"},
        "artifacts": {
            "resolved_config": "resolved.json",
            "metrics": "metrics.json",
            "results": "results.json",
            "predictions": "artifacts/predictions/preds.parquet",
            "scoring_manifest": "artifacts/scoring/manifest.json",
        },
    }
    if experiment_id is not None:
        manifest["experiment_id"] = experiment_id
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    index_run(store_root=store_root, run_id=run_id)
    return run_dir


def _write_experiment_manifest(
    store_root: Path,
    experiment_id: str,
    *,
    runs: list[str],
    champion_run_id: str | None = None,
    archived: bool = False,
) -> None:
    root = store_root / "experiments" / ("_archive" if archived else "") / experiment_id
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1",
        "experiment_id": experiment_id,
        "name": experiment_id,
        "status": "archived" if archived else "active",
        "created_at": "2026-02-21T00:00:00+00:00",
        "updated_at": "2026-02-21T00:00:00+00:00",
        "champion_run_id": champion_run_id,
        "runs": runs,
    }
    (root / "experiment.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_submission_package(store_root: Path, experiment_id: str, package_id: str, run_id: str) -> None:
    package_dir = store_root / "experiments" / experiment_id / "submission_packages" / package_id
    package_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "package_id": package_id,
        "experiment_id": experiment_id,
        "components": [
            {
                "component_id": "component-1",
                "run_id": run_id,
                "weight": 1.0,
            }
        ],
    }
    (package_dir / "package.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_prune_predictions_dry_run_reports_bytes_without_deleting(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = _write_run(store_root, "run-dry")

    result = prune_predictions(store_root=store_root, run_ids=("run-dry",))

    assert result.dry_run is True
    assert result.candidate_count == 1
    assert result.pruned_count == 1
    assert result.excluded_count == 0
    assert result.reclaimable_bytes == 4
    assert result.reclaimed_bytes == 0
    assert result.pruned[0].run_id == "run-dry"
    assert (run_dir / "artifacts" / "predictions" / "preds.parquet").is_file()


def test_prune_predictions_apply_removes_only_predictions_and_reindexes(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = _write_run(store_root, "run-apply")

    result = prune_predictions(store_root=store_root, run_ids=("run-apply",), apply=True)

    assert result.dry_run is False
    assert result.pruned_count == 1
    assert result.reclaimable_bytes == 4
    assert result.reclaimed_bytes == 4
    assert not (run_dir / "artifacts" / "predictions").exists()
    assert (run_dir / "run.json").is_file()
    assert (run_dir / "results.json").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "score_provenance.json").is_file()
    assert (run_dir / "run.log").is_file()
    assert (run_dir / "artifacts" / "scoring" / "manifest.json").is_file()

    with sqlite3.connect(store_root / "numereng.db") as conn:
        row = conn.execute(
            "SELECT exists_flag, size_bytes FROM run_artifacts WHERE run_id = ? AND kind = ?",
            ("run-apply", "predictions"),
        ).fetchone()
    assert row == (0, None)
    assert doctor_store(store_root=store_root).ok is True


def test_prune_predictions_excludes_archived_experiment_champion(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = _write_run(store_root, "run-champ")
    _write_experiment_manifest(
        store_root,
        "exp-archived",
        runs=["run-champ"],
        champion_run_id="run-champ",
        archived=True,
    )

    result = prune_predictions(store_root=store_root, all_runs=True, apply=True)

    assert result.pruned_count == 0
    assert result.excluded_count == 1
    assert result.excluded[0].reason == "champion_run:exp-archived"
    assert (run_dir / "artifacts" / "predictions" / "preds.parquet").is_file()


def test_prune_predictions_excludes_submission_package_component_run(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = _write_run(store_root, "run-package")
    _write_submission_package(store_root, "exp-1", "package-1", "run-package")

    result = prune_predictions(store_root=store_root, all_runs=True, apply=True)

    assert result.pruned_count == 0
    assert result.excluded_count == 1
    assert (
        result.excluded[0].reason == "submission_package:experiments/exp-1/submission_packages/package-1/package.json"
    )
    assert (run_dir / "artifacts" / "predictions" / "preds.parquet").is_file()


def test_prune_predictions_excludes_non_finished_run(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = _write_run(store_root, "run-failed", status="FAILED")

    result = prune_predictions(store_root=store_root, all_runs=True, apply=True)

    assert result.pruned_count == 0
    assert result.excluded_count == 1
    assert result.excluded[0].reason == "status_not_finished:FAILED"
    assert (run_dir / "artifacts" / "predictions" / "preds.parquet").is_file()


def test_prune_predictions_excludes_indexed_run_without_predictions_dir(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    _write_run(store_root, "run-empty", write_predictions=False)

    result = prune_predictions(store_root=store_root, all_runs=True)

    assert result.pruned_count == 0
    assert result.excluded_count == 1
    assert result.excluded[0].reason == "predictions_dir_missing"


def test_prune_predictions_reports_explicit_unindexed_run(tmp_path: Path) -> None:
    result = prune_predictions(store_root=tmp_path / ".numereng", run_ids=("run-missing",))

    assert result.candidate_count == 1
    assert result.pruned_count == 0
    assert result.excluded_count == 1
    assert result.excluded[0].run_id == "run-missing"
    assert result.excluded[0].reason == "not_indexed"


def test_prune_predictions_experiment_scope_uses_manifest_and_reports_unindexed(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    _write_run(store_root, "run-indexed", experiment_id="exp-1")
    _write_experiment_manifest(store_root, "exp-1", runs=["run-indexed", "run-unindexed"])

    result = prune_predictions(store_root=store_root, experiment_id="exp-1")

    assert result.candidate_count == 2
    assert [item.run_id for item in result.pruned] == ["run-indexed"]
    assert result.excluded[0].run_id == "run-unindexed"
    assert result.excluded[0].reason == "not_indexed"
