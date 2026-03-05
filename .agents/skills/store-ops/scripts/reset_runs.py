#!/usr/bin/env python3
"""Reset one or more run IDs from numereng store with guardrails."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from numereng.features.experiments import service as experiment_service

_SAFE_ID = re.compile(r"^[\w\-.]+$")


@dataclass(frozen=True)
class StorePaths:
    store_root: Path
    db_path: Path
    runs_dir: Path
    experiments_dir: Path


@dataclass
class Scope:
    target_run_ids: list[str]
    run_dirs_present: list[str]
    manifest_references: dict[str, list[str]]
    active_writers: list[str]
    job_ids: list[str]
    logical_run_ids: list[str]
    study_ids: list[str]
    ensemble_ids: list[str]
    db_counts: dict[str, dict[str, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset one or more run IDs safely.")
    parser.add_argument(
        "--store-root",
        default=".numereng",
        help="Store root path (default: .numereng)",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        required=True,
        help="Target run ID (repeatable)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform destructive mutation. If omitted, outputs dry-run summary only.",
    )
    parser.add_argument(
        "--allow-active-writers",
        action="store_true",
        help="Allow mutation even when active writer processes are detected.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def ensure_safe_run_id(run_id: str) -> str:
    candidate = run_id.strip()
    if not candidate or not _SAFE_ID.match(candidate):
        raise ValueError(f"invalid_run_id:{run_id}")
    return candidate


def ensure_paths(store_root_raw: str) -> StorePaths:
    store_root = Path(store_root_raw).expanduser().resolve()
    return StorePaths(
        store_root=store_root,
        db_path=store_root / "numereng.db",
        runs_dir=store_root / "runs",
        experiments_dir=store_root / "experiments",
    )


def load_json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def normalize_run_ids(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    values: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def detect_active_writers() -> list[str]:
    try:
        output = subprocess.run(
            ["ps", "aux"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except Exception:
        return []

    pattern = re.compile(
        r"numereng\s+(run\s+train|experiment\s+train|cloud\s+aws\s+train\s+submit|cloud\s+modal\s+train\s+submit)",
        re.IGNORECASE,
    )
    return [line.strip() for line in output.splitlines() if pattern.search(line)]


def select_values(cur: sqlite3.Cursor, query: str, params: list[str]) -> list[str]:
    rows = cur.execute(query, params).fetchall()
    values: list[str] = []
    for row in rows:
        value = row[0]
        if isinstance(value, str) and value:
            values.append(value)
    return values


def count_rows(cur: sqlite3.Cursor, table: str, where_clause: str, params: list[str]) -> int:
    query = f"SELECT COUNT(*) FROM {table} WHERE {where_clause}"  # noqa: S608
    return int(cur.execute(query, params).fetchone()[0])


def gather_manifest_references(paths: StorePaths, target_run_ids: set[str]) -> dict[str, list[str]]:
    references: dict[str, list[str]] = {}
    if not target_run_ids or not paths.experiments_dir.is_dir():
        return references

    for manifest_path in sorted(paths.experiments_dir.glob("*/experiment.json")):
        payload = load_json_mapping(manifest_path)
        run_ids = normalize_run_ids(payload.get("runs"))
        matches = sorted(run_id for run_id in run_ids if run_id in target_run_ids)
        if matches:
            references[manifest_path.parent.name] = matches
    return references


def gather_scope(paths: StorePaths, target_run_ids: list[str]) -> Scope:
    unique_run_ids = sorted(set(target_run_ids))
    target_run_id_set = set(unique_run_ids)
    run_dirs_present = sorted(run_id for run_id in unique_run_ids if (paths.runs_dir / run_id).is_dir())
    manifest_references = gather_manifest_references(paths, target_run_id_set)
    active_writers = detect_active_writers()

    db_counts: dict[str, dict[str, int]] = {}
    job_ids: list[str] = []
    logical_run_ids: list[str] = []
    study_ids: list[str] = []
    ensemble_ids: list[str] = []

    if not paths.db_path.is_file() or not unique_run_ids:
        return Scope(
            target_run_ids=unique_run_ids,
            run_dirs_present=run_dirs_present,
            manifest_references=manifest_references,
            active_writers=active_writers,
            job_ids=job_ids,
            logical_run_ids=logical_run_ids,
            study_ids=study_ids,
            ensemble_ids=ensemble_ids,
            db_counts=db_counts,
        )

    conn = sqlite3.connect(paths.db_path)
    cur = conn.cursor()

    placeholders_runs = ",".join("?" for _ in unique_run_ids)

    job_ids_set = set(
        select_values(
            cur,
            f"SELECT DISTINCT job_id FROM run_jobs WHERE canonical_run_id IN ({placeholders_runs})",
            unique_run_ids,
        )
    )
    job_ids_set.update(
        select_values(
            cur,
            f"SELECT DISTINCT job_id FROM run_attempts WHERE canonical_run_id IN ({placeholders_runs})",
            unique_run_ids,
        )
    )
    job_ids = sorted(job_ids_set)

    if job_ids:
        placeholders_jobs = ",".join("?" for _ in job_ids)
        logical_ids_set = set(
            select_values(
                cur,
                f"SELECT DISTINCT logical_run_id FROM run_jobs WHERE job_id IN ({placeholders_jobs})",
                job_ids,
            )
        )
        logical_ids_set.update(
            select_values(
                cur,
                f"SELECT DISTINCT logical_run_id FROM run_attempts WHERE job_id IN ({placeholders_jobs})",
                job_ids,
            )
        )
        logical_run_ids = sorted(logical_ids_set)

    study_ids_set = set(
        select_values(
            cur,
            f"SELECT DISTINCT study_id FROM hpo_trials WHERE run_id IN ({placeholders_runs})",
            unique_run_ids,
        )
    )
    study_ids_set.update(
        select_values(
            cur,
            f"SELECT DISTINCT study_id FROM hpo_studies WHERE best_run_id IN ({placeholders_runs})",
            unique_run_ids,
        )
    )
    study_ids = sorted(study_ids_set)

    ensemble_ids = sorted(
        set(
            select_values(
                cur,
                f"SELECT DISTINCT ensemble_id FROM ensemble_components WHERE run_id IN ({placeholders_runs})",
                unique_run_ids,
            )
        )
    )

    db_counts["by_run"] = {
        "runs": count_rows(cur, "runs", f"run_id IN ({placeholders_runs})", unique_run_ids),
        "metrics": count_rows(cur, "metrics", f"run_id IN ({placeholders_runs})", unique_run_ids),
        "run_artifacts": count_rows(cur, "run_artifacts", f"run_id IN ({placeholders_runs})", unique_run_ids),
        "run_jobs_canonical": count_rows(
            cur,
            "run_jobs",
            f"canonical_run_id IN ({placeholders_runs})",
            unique_run_ids,
        ),
        "run_attempts_canonical": count_rows(
            cur,
            "run_attempts",
            f"canonical_run_id IN ({placeholders_runs})",
            unique_run_ids,
        ),
        "cloud_jobs": count_rows(cur, "cloud_jobs", f"run_id IN ({placeholders_runs})", unique_run_ids),
        "hpo_trials_run": count_rows(cur, "hpo_trials", f"run_id IN ({placeholders_runs})", unique_run_ids),
        "hpo_studies_best_run": count_rows(
            cur,
            "hpo_studies",
            f"best_run_id IN ({placeholders_runs})",
            unique_run_ids,
        ),
        "ensemble_components_run": count_rows(
            cur,
            "ensemble_components",
            f"run_id IN ({placeholders_runs})",
            unique_run_ids,
        ),
    }

    if job_ids:
        placeholders_jobs = ",".join("?" for _ in job_ids)
        db_counts["by_job"] = {
            "run_jobs": count_rows(cur, "run_jobs", f"job_id IN ({placeholders_jobs})", job_ids),
            "run_attempts": count_rows(cur, "run_attempts", f"job_id IN ({placeholders_jobs})", job_ids),
            "run_job_logs": count_rows(cur, "run_job_logs", f"job_id IN ({placeholders_jobs})", job_ids),
            "run_job_events": count_rows(cur, "run_job_events", f"job_id IN ({placeholders_jobs})", job_ids),
            "run_job_samples": count_rows(cur, "run_job_samples", f"job_id IN ({placeholders_jobs})", job_ids),
        }

    if logical_run_ids:
        placeholders_logical = ",".join("?" for _ in logical_run_ids)
        db_counts["by_logical_run"] = {
            "logical_runs": count_rows(
                cur,
                "logical_runs",
                f"logical_run_id IN ({placeholders_logical})",
                logical_run_ids,
            ),
            "run_jobs": count_rows(cur, "run_jobs", f"logical_run_id IN ({placeholders_logical})", logical_run_ids),
            "run_attempts": count_rows(
                cur,
                "run_attempts",
                f"logical_run_id IN ({placeholders_logical})",
                logical_run_ids,
            ),
        }

    if study_ids:
        placeholders_studies = ",".join("?" for _ in study_ids)
        db_counts["by_study"] = {
            "hpo_studies": count_rows(cur, "hpo_studies", f"study_id IN ({placeholders_studies})", study_ids),
            "hpo_trials": count_rows(cur, "hpo_trials", f"study_id IN ({placeholders_studies})", study_ids),
        }

    if ensemble_ids:
        placeholders_ensembles = ",".join("?" for _ in ensemble_ids)
        db_counts["by_ensemble"] = {
            "ensembles": count_rows(cur, "ensembles", f"ensemble_id IN ({placeholders_ensembles})", ensemble_ids),
            "ensemble_components": count_rows(
                cur,
                "ensemble_components",
                f"ensemble_id IN ({placeholders_ensembles})",
                ensemble_ids,
            ),
            "ensemble_metrics": count_rows(
                cur,
                "ensemble_metrics",
                f"ensemble_id IN ({placeholders_ensembles})",
                ensemble_ids,
            ),
        }

    conn.close()

    return Scope(
        target_run_ids=unique_run_ids,
        run_dirs_present=run_dirs_present,
        manifest_references=manifest_references,
        active_writers=active_writers,
        job_ids=job_ids,
        logical_run_ids=logical_run_ids,
        study_ids=study_ids,
        ensemble_ids=ensemble_ids,
        db_counts=db_counts,
    )


def exec_delete(cur: sqlite3.Cursor, query: str, args: list[str]) -> int:
    cur.execute(query, args)
    return int(cur.rowcount)


def mutate(paths: StorePaths, scope: Scope) -> dict[str, Any]:
    target_run_set = set(scope.target_run_ids)

    deleted_run_dirs: list[str] = []
    for run_id in scope.target_run_ids:
        run_dir = paths.runs_dir / run_id
        if run_dir.is_dir():
            shutil.rmtree(run_dir)
            deleted_run_dirs.append(run_id)

    table_deletes: dict[str, int] = {}
    if paths.db_path.is_file():
        conn = sqlite3.connect(paths.db_path)
        cur = conn.cursor()
        conn.execute("BEGIN")
        try:
            if scope.job_ids:
                placeholders_jobs = ",".join("?" for _ in scope.job_ids)
                table_deletes["run_job_logs"] = exec_delete(
                    cur,
                    f"DELETE FROM run_job_logs WHERE job_id IN ({placeholders_jobs})",
                    scope.job_ids,
                )
                table_deletes["run_job_events"] = exec_delete(
                    cur,
                    f"DELETE FROM run_job_events WHERE job_id IN ({placeholders_jobs})",
                    scope.job_ids,
                )
                table_deletes["run_job_samples"] = exec_delete(
                    cur,
                    f"DELETE FROM run_job_samples WHERE job_id IN ({placeholders_jobs})",
                    scope.job_ids,
                )
                table_deletes["run_attempts(job)"] = exec_delete(
                    cur,
                    f"DELETE FROM run_attempts WHERE job_id IN ({placeholders_jobs})",
                    scope.job_ids,
                )
                table_deletes["run_jobs(job)"] = exec_delete(
                    cur,
                    f"DELETE FROM run_jobs WHERE job_id IN ({placeholders_jobs})",
                    scope.job_ids,
                )

            if scope.target_run_ids:
                placeholders_runs = ",".join("?" for _ in scope.target_run_ids)
                table_deletes["run_attempts(canonical)"] = exec_delete(
                    cur,
                    f"DELETE FROM run_attempts WHERE canonical_run_id IN ({placeholders_runs})",
                    scope.target_run_ids,
                )
                table_deletes["run_jobs(canonical)"] = exec_delete(
                    cur,
                    f"DELETE FROM run_jobs WHERE canonical_run_id IN ({placeholders_runs})",
                    scope.target_run_ids,
                )
                table_deletes["metrics"] = exec_delete(
                    cur,
                    f"DELETE FROM metrics WHERE run_id IN ({placeholders_runs})",
                    scope.target_run_ids,
                )
                table_deletes["run_artifacts"] = exec_delete(
                    cur,
                    f"DELETE FROM run_artifacts WHERE run_id IN ({placeholders_runs})",
                    scope.target_run_ids,
                )
                table_deletes["cloud_jobs"] = exec_delete(
                    cur,
                    f"DELETE FROM cloud_jobs WHERE run_id IN ({placeholders_runs})",
                    scope.target_run_ids,
                )
                table_deletes["hpo_trials(run)"] = exec_delete(
                    cur,
                    f"DELETE FROM hpo_trials WHERE run_id IN ({placeholders_runs})",
                    scope.target_run_ids,
                )
                table_deletes["ensemble_components(run)"] = exec_delete(
                    cur,
                    f"DELETE FROM ensemble_components WHERE run_id IN ({placeholders_runs})",
                    scope.target_run_ids,
                )
                cur.execute(
                    (
                        f"UPDATE hpo_studies "
                        "SET best_run_id = NULL, best_trial_number = NULL, best_value = NULL, updated_at = ? "
                        f"WHERE best_run_id IN ({placeholders_runs})"
                    ),
                    [utc_now_iso(), *scope.target_run_ids],
                )
                table_deletes["hpo_studies(best_run)"] = int(cur.rowcount)
                table_deletes["runs"] = exec_delete(
                    cur,
                    f"DELETE FROM runs WHERE run_id IN ({placeholders_runs})",
                    scope.target_run_ids,
                )

            if scope.logical_run_ids:
                placeholders_logical = ",".join("?" for _ in scope.logical_run_ids)
                orphan_rows = cur.execute(
                    (
                        "SELECT lr.logical_run_id "
                        "FROM logical_runs lr "
                        f"WHERE lr.logical_run_id IN ({placeholders_logical}) "
                        "AND NOT EXISTS (SELECT 1 FROM run_jobs rj WHERE rj.logical_run_id = lr.logical_run_id) "
                        "AND NOT EXISTS (SELECT 1 FROM run_attempts ra WHERE ra.logical_run_id = lr.logical_run_id)"
                    ),
                    scope.logical_run_ids,
                ).fetchall()
                orphan_ids = [row[0] for row in orphan_rows if isinstance(row[0], str)]
                if orphan_ids:
                    placeholders_orphan = ",".join("?" for _ in orphan_ids)
                    table_deletes["logical_runs(orphaned)"] = exec_delete(
                        cur,
                        f"DELETE FROM logical_runs WHERE logical_run_id IN ({placeholders_orphan})",
                        orphan_ids,
                    )

            conn.commit()
        except Exception:
            conn.rollback()
            conn.close()
            raise
        conn.close()

    now = utc_now_iso()
    updated_manifests = 0
    champion_cleared = 0
    drafted_manifests = 0
    updated_experiment_ids: list[str] = []

    if paths.experiments_dir.is_dir():
        for manifest_path in sorted(paths.experiments_dir.glob("*/experiment.json")):
            manifest = load_json_mapping(manifest_path)
            if not manifest:
                continue

            run_ids = normalize_run_ids(manifest.get("runs"))
            retained_run_ids = [run_id for run_id in run_ids if run_id not in target_run_set]
            champion_run_id = manifest.get("champion_run_id")
            champion_hit = isinstance(champion_run_id, str) and champion_run_id in target_run_set

            if retained_run_ids == run_ids and not champion_hit:
                continue

            if champion_hit:
                manifest["champion_run_id"] = None
                champion_cleared += 1

            manifest["runs"] = retained_run_ids
            if not retained_run_ids and manifest.get("status") != "draft":
                manifest["status"] = "draft"
                drafted_manifests += 1
            manifest["updated_at"] = now

            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            experiment_service._index_experiment_manifest(paths.store_root, manifest)
            updated_manifests += 1
            updated_experiment_ids.append(manifest_path.parent.name)

    return {
        "deleted_run_dirs": len(deleted_run_dirs),
        "deleted_run_ids": deleted_run_dirs,
        "table_deletes": table_deletes,
        "updated_manifests": updated_manifests,
        "updated_experiment_ids": sorted(updated_experiment_ids),
        "champion_cleared": champion_cleared,
        "drafted_manifests": drafted_manifests,
    }


def make_output(
    *,
    paths: StorePaths,
    scope: Scope,
    execute: bool,
    allow_active_writers: bool,
    mutation_result: dict[str, Any] | None,
) -> dict[str, Any]:
    blocked_reasons: list[str] = []
    if scope.active_writers and not allow_active_writers:
        blocked_reasons.append("active_writers_detected")

    return {
        "store_root": str(paths.store_root),
        "db_path": str(paths.db_path),
        "mode": "execute" if execute else "dry_run",
        "target_run_ids": scope.target_run_ids,
        "run_dirs_present": scope.run_dirs_present,
        "manifest_references": scope.manifest_references,
        "related_ids": {
            "job_ids": scope.job_ids,
            "logical_run_ids": scope.logical_run_ids,
            "study_ids": scope.study_ids,
            "ensemble_ids": scope.ensemble_ids,
        },
        "db_counts": scope.db_counts,
        "active_writers": scope.active_writers,
        "blocked": bool(blocked_reasons),
        "blocked_reasons": blocked_reasons,
        "allow_flags": {
            "allow_active_writers": allow_active_writers,
        },
        "mutation_result": mutation_result,
    }


def main() -> int:
    args = parse_args()
    paths = ensure_paths(args.store_root)

    try:
        run_ids = sorted({ensure_safe_run_id(value) for value in args.run_id})
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    scope = gather_scope(paths, run_ids)
    blocked = bool(scope.active_writers and not args.allow_active_writers)
    mutation_result: dict[str, Any] | None = None

    if args.execute:
        if blocked:
            output = make_output(
                paths=paths,
                scope=scope,
                execute=True,
                allow_active_writers=args.allow_active_writers,
                mutation_result=None,
            )
            print(json.dumps(output, indent=2 if args.pretty else None, sort_keys=True))
            return 1
        mutation_result = mutate(paths, scope)
        scope = gather_scope(paths, run_ids)

    output = make_output(
        paths=paths,
        scope=scope,
        execute=args.execute,
        allow_active_writers=args.allow_active_writers,
        mutation_result=mutation_result,
    )
    print(json.dumps(output, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
