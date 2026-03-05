#!/usr/bin/env python3
"""Reset experiment-linked run data in numereng store with guardrails."""

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

PRESERVE_DESIGN_FILES = "preserve_design_files"
PRESERVE_ONLY_CONFIGS = "preserve_only_configs"
HARD_WIPE_EXPERIMENT_DIRS = "hard_wipe_experiment_dirs"


@dataclass(frozen=True)
class StorePaths:
    store_root: Path
    db_path: Path
    runs_dir: Path
    experiments_dir: Path


@dataclass
class Scope:
    experiment_ids: list[str]
    manifest_runs: dict[str, list[str]]
    target_run_ids: list[str]
    job_ids: list[str]
    logical_run_ids: list[str]
    study_ids: list[str]
    ensemble_ids: list[str]
    overlaps_manifest: dict[str, list[str]]
    overlaps_db_runs: list[dict[str, str | None]]
    active_writers: list[str]
    db_counts: dict[str, dict[str, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset experiment-linked run data safely.")
    parser.add_argument(
        "--store-root",
        default=".numereng",
        help="Store root path (default: .numereng)",
    )
    parser.add_argument(
        "--experiment-id",
        action="append",
        required=True,
        help="Target experiment ID (repeatable)",
    )
    parser.add_argument(
        "--preserve-policy",
        default=PRESERVE_DESIGN_FILES,
        choices=[PRESERVE_DESIGN_FILES, PRESERVE_ONLY_CONFIGS, HARD_WIPE_EXPERIMENT_DIRS],
        help="Reset policy for experiment directories.",
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
        "--allow-overlap",
        action="store_true",
        help="Allow mutation even when target run IDs overlap with non-target experiments.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


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


def ensure_paths(store_root_raw: str) -> StorePaths:
    store_root = Path(store_root_raw).expanduser().resolve()
    return StorePaths(
        store_root=store_root,
        db_path=store_root / "numereng.db",
        runs_dir=store_root / "runs",
        experiments_dir=store_root / "experiments",
    )


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


def gather_manifest_runs(paths: StorePaths, experiment_ids: list[str]) -> dict[str, list[str]]:
    manifest_runs: dict[str, list[str]] = {}
    for experiment_id in experiment_ids:
        manifest_path = paths.experiments_dir / experiment_id / "experiment.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing_manifest:{manifest_path}")
        payload = load_json_mapping(manifest_path)
        run_ids_raw = payload.get("runs")
        run_ids: list[str] = []
        if isinstance(run_ids_raw, list):
            for value in run_ids_raw:
                if isinstance(value, str) and value.strip():
                    run_ids.append(value.strip())
        manifest_runs[experiment_id] = sorted(set(run_ids))
    return manifest_runs


def gather_runs_from_filesystem(paths: StorePaths, target_experiment_ids: set[str]) -> list[str]:
    if not paths.runs_dir.is_dir() or not target_experiment_ids:
        return []
    run_ids: list[str] = []
    for run_dir in sorted(paths.runs_dir.iterdir(), key=lambda item: item.name):
        if not run_dir.is_dir():
            continue
        payload = load_json_mapping(run_dir / "run.json")
        if payload.get("experiment_id") in target_experiment_ids:
            run_ids.append(run_dir.name)
    return sorted(set(run_ids))


def gather_manifest_overlaps(
    paths: StorePaths, target_experiment_ids: set[str], target_run_ids: set[str]
) -> dict[str, list[str]]:
    overlaps: dict[str, list[str]] = {}
    if not target_run_ids or not paths.experiments_dir.is_dir():
        return overlaps

    for manifest_path in sorted(paths.experiments_dir.glob("*/experiment.json")):
        experiment_id = manifest_path.parent.name
        if experiment_id in target_experiment_ids:
            continue
        payload = load_json_mapping(manifest_path)
        runs = payload.get("runs")
        if not isinstance(runs, list):
            continue
        intersection = sorted({value for value in runs if isinstance(value, str) and value in target_run_ids})
        if intersection:
            overlaps[experiment_id] = intersection
    return overlaps


def gather_scope(paths: StorePaths, experiment_ids: list[str]) -> Scope:
    if not paths.db_path.is_file():
        raise FileNotFoundError(f"missing_db:{paths.db_path}")

    manifest_runs = gather_manifest_runs(paths, experiment_ids)
    active_writers = detect_active_writers()

    conn = sqlite3.connect(paths.db_path)
    cur = conn.cursor()

    target_run_ids: set[str] = set()
    for values in manifest_runs.values():
        target_run_ids.update(values)

    placeholders_exp = ",".join("?" for _ in experiment_ids)

    db_runs = select_values(
        cur,
        f"SELECT run_id FROM runs WHERE experiment_id IN ({placeholders_exp})",
        experiment_ids,
    )
    target_run_ids.update(db_runs)

    job_canonical_runs = select_values(
        cur,
        (
            f"SELECT canonical_run_id FROM run_jobs "
            f"WHERE experiment_id IN ({placeholders_exp}) "
            "AND canonical_run_id IS NOT NULL AND canonical_run_id != ''"
        ),
        experiment_ids,
    )
    target_run_ids.update(job_canonical_runs)

    fs_runs = gather_runs_from_filesystem(paths, set(experiment_ids))
    target_run_ids.update(fs_runs)

    target_run_ids_list = sorted(target_run_ids)

    # Find related IDs.
    clause_parts: list[str] = []
    clause_params: list[str] = []
    clause_parts.append(f"experiment_id IN ({placeholders_exp})")
    clause_params.extend(experiment_ids)
    if target_run_ids_list:
        placeholders_runs = ",".join("?" for _ in target_run_ids_list)
        clause_parts.append(f"canonical_run_id IN ({placeholders_runs})")
        clause_params.extend(target_run_ids_list)
    where_clause = " OR ".join(clause_parts)

    job_ids = sorted(
        set(select_values(cur, f"SELECT DISTINCT job_id FROM run_jobs WHERE {where_clause}", clause_params))
    )
    logical_run_ids = sorted(
        set(
            select_values(
                cur,
                f"SELECT DISTINCT logical_run_id FROM logical_runs WHERE experiment_id IN ({placeholders_exp})",
                experiment_ids,
            )
        )
    )
    study_ids = sorted(
        set(
            select_values(
                cur,
                f"SELECT DISTINCT study_id FROM hpo_studies WHERE experiment_id IN ({placeholders_exp})",
                experiment_ids,
            )
        )
    )
    ensemble_ids = sorted(
        set(
            select_values(
                cur,
                f"SELECT DISTINCT ensemble_id FROM ensembles WHERE experiment_id IN ({placeholders_exp})",
                experiment_ids,
            )
        )
    )

    overlaps_manifest = gather_manifest_overlaps(paths, set(experiment_ids), set(target_run_ids_list))

    overlaps_db_runs: list[dict[str, str | None]] = []
    if target_run_ids_list:
        placeholders_runs = ",".join("?" for _ in target_run_ids_list)
        query = (
            f"SELECT run_id, experiment_id FROM runs "
            f"WHERE run_id IN ({placeholders_runs}) "
            f"AND COALESCE(experiment_id, '') NOT IN ({placeholders_exp})"
        )
        rows = cur.execute(query, target_run_ids_list + experiment_ids).fetchall()
        overlaps_db_runs = [
            {
                "run_id": row[0] if isinstance(row[0], str) else None,
                "experiment_id": row[1] if isinstance(row[1], str) else None,
            }
            for row in rows
        ]

    db_counts: dict[str, dict[str, int]] = {
        "by_experiment": {
            "experiments": count_rows(cur, "experiments", f"experiment_id IN ({placeholders_exp})", experiment_ids),
            "runs": count_rows(cur, "runs", f"experiment_id IN ({placeholders_exp})", experiment_ids),
            "run_jobs": count_rows(cur, "run_jobs", f"experiment_id IN ({placeholders_exp})", experiment_ids),
            "logical_runs": count_rows(cur, "logical_runs", f"experiment_id IN ({placeholders_exp})", experiment_ids),
            "hpo_studies": count_rows(cur, "hpo_studies", f"experiment_id IN ({placeholders_exp})", experiment_ids),
            "ensembles": count_rows(cur, "ensembles", f"experiment_id IN ({placeholders_exp})", experiment_ids),
        }
    }

    if target_run_ids_list:
        placeholders_runs = ",".join("?" for _ in target_run_ids_list)
        db_counts["by_run"] = {
            "metrics": count_rows(cur, "metrics", f"run_id IN ({placeholders_runs})", target_run_ids_list),
            "run_artifacts": count_rows(cur, "run_artifacts", f"run_id IN ({placeholders_runs})", target_run_ids_list),
            "run_attempts_canonical": count_rows(
                cur,
                "run_attempts",
                f"canonical_run_id IN ({placeholders_runs})",
                target_run_ids_list,
            ),
            "run_jobs_canonical": count_rows(
                cur,
                "run_jobs",
                f"canonical_run_id IN ({placeholders_runs})",
                target_run_ids_list,
            ),
            "hpo_trials_run": count_rows(cur, "hpo_trials", f"run_id IN ({placeholders_runs})", target_run_ids_list),
            "ensemble_components_run": count_rows(
                cur,
                "ensemble_components",
                f"run_id IN ({placeholders_runs})",
                target_run_ids_list,
            ),
        }

    if job_ids:
        placeholders_jobs = ",".join("?" for _ in job_ids)
        db_counts["by_job"] = {
            "run_job_logs": count_rows(cur, "run_job_logs", f"job_id IN ({placeholders_jobs})", job_ids),
            "run_job_events": count_rows(cur, "run_job_events", f"job_id IN ({placeholders_jobs})", job_ids),
            "run_job_samples": count_rows(cur, "run_job_samples", f"job_id IN ({placeholders_jobs})", job_ids),
        }

    if study_ids:
        placeholders_studies = ",".join("?" for _ in study_ids)
        db_counts["by_study"] = {
            "hpo_trials": count_rows(cur, "hpo_trials", f"study_id IN ({placeholders_studies})", study_ids),
        }

    if ensemble_ids:
        placeholders_ensembles = ",".join("?" for _ in ensemble_ids)
        db_counts["by_ensemble"] = {
            "ensemble_metrics": count_rows(
                cur,
                "ensemble_metrics",
                f"ensemble_id IN ({placeholders_ensembles})",
                ensemble_ids,
            ),
            "ensemble_components": count_rows(
                cur,
                "ensemble_components",
                f"ensemble_id IN ({placeholders_ensembles})",
                ensemble_ids,
            ),
        }

    conn.close()

    return Scope(
        experiment_ids=experiment_ids,
        manifest_runs=manifest_runs,
        target_run_ids=target_run_ids_list,
        job_ids=job_ids,
        logical_run_ids=logical_run_ids,
        study_ids=study_ids,
        ensemble_ids=ensemble_ids,
        overlaps_manifest=overlaps_manifest,
        overlaps_db_runs=overlaps_db_runs,
        active_writers=active_writers,
        db_counts=db_counts,
    )


def exec_delete(cur: sqlite3.Cursor, query: str, args: list[str]) -> int:
    cur.execute(query, args)
    return int(cur.rowcount)


def render_experiment_doc(manifest: dict[str, Any]) -> str:
    hypothesis = manifest.get("hypothesis") if isinstance(manifest.get("hypothesis"), str) else "n/a"
    tags_raw = manifest.get("tags")
    tags: list[str] = []
    if isinstance(tags_raw, list):
        tags = [item for item in tags_raw if isinstance(item, str) and item.strip()]
    return "\n".join(
        [
            f"# {manifest.get('experiment_id')}",
            "",
            "## Summary",
            f"- name: {manifest.get('name')}",
            f"- hypothesis: {hypothesis}",
            f"- status: {manifest.get('status')}",
            f"- tags: {', '.join(tags) if tags else 'none'}",
            "",
            "## Notes",
            "- Track experiment findings and next actions in this file.",
            "",
        ]
    )


def mutate(paths: StorePaths, scope: Scope, preserve_policy: str) -> dict[str, Any]:
    conn = sqlite3.connect(paths.db_path)
    cur = conn.cursor()

    table_deletes: dict[str, int] = {}

    # Remove run directories first so rebuild/index cannot rehydrate deleted runs.
    deleted_run_dirs = 0
    for run_id in scope.target_run_ids:
        run_dir = paths.runs_dir / run_id
        if run_dir.is_dir():
            shutil.rmtree(run_dir)
            deleted_run_dirs += 1

    placeholders_exp = ",".join("?" for _ in scope.experiment_ids)

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

        if scope.study_ids:
            placeholders_studies = ",".join("?" for _ in scope.study_ids)
            table_deletes["hpo_trials(study_id)"] = exec_delete(
                cur,
                f"DELETE FROM hpo_trials WHERE study_id IN ({placeholders_studies})",
                scope.study_ids,
            )

        if scope.ensemble_ids:
            placeholders_ensembles = ",".join("?" for _ in scope.ensemble_ids)
            table_deletes["ensemble_metrics"] = exec_delete(
                cur,
                f"DELETE FROM ensemble_metrics WHERE ensemble_id IN ({placeholders_ensembles})",
                scope.ensemble_ids,
            )
            table_deletes["ensemble_components(ensemble)"] = exec_delete(
                cur,
                f"DELETE FROM ensemble_components WHERE ensemble_id IN ({placeholders_ensembles})",
                scope.ensemble_ids,
            )

        if scope.target_run_ids:
            placeholders_runs = ",".join("?" for _ in scope.target_run_ids)
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
            table_deletes["run_attempts(canonical)"] = exec_delete(
                cur,
                f"DELETE FROM run_attempts WHERE canonical_run_id IN ({placeholders_runs})",
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

        if scope.logical_run_ids:
            placeholders_logical = ",".join("?" for _ in scope.logical_run_ids)
            table_deletes["run_attempts(logical)"] = exec_delete(
                cur,
                f"DELETE FROM run_attempts WHERE logical_run_id IN ({placeholders_logical})",
                scope.logical_run_ids,
            )

        if scope.job_ids:
            placeholders_jobs = ",".join("?" for _ in scope.job_ids)
            table_deletes["run_jobs(job_id)"] = exec_delete(
                cur,
                f"DELETE FROM run_jobs WHERE job_id IN ({placeholders_jobs})",
                scope.job_ids,
            )

        table_deletes["run_jobs(experiment)"] = exec_delete(
            cur,
            f"DELETE FROM run_jobs WHERE experiment_id IN ({placeholders_exp})",
            scope.experiment_ids,
        )

        table_deletes["logical_runs"] = exec_delete(
            cur,
            f"DELETE FROM logical_runs WHERE experiment_id IN ({placeholders_exp})",
            scope.experiment_ids,
        )
        table_deletes["hpo_studies"] = exec_delete(
            cur,
            f"DELETE FROM hpo_studies WHERE experiment_id IN ({placeholders_exp})",
            scope.experiment_ids,
        )
        table_deletes["ensembles"] = exec_delete(
            cur,
            f"DELETE FROM ensembles WHERE experiment_id IN ({placeholders_exp})",
            scope.experiment_ids,
        )

        if scope.target_run_ids:
            placeholders_runs = ",".join("?" for _ in scope.target_run_ids)
            table_deletes["runs(run_id)"] = exec_delete(
                cur,
                f"DELETE FROM runs WHERE run_id IN ({placeholders_runs})",
                scope.target_run_ids,
            )

        table_deletes["runs(experiment)"] = exec_delete(
            cur,
            f"DELETE FROM runs WHERE experiment_id IN ({placeholders_exp})",
            scope.experiment_ids,
        )

        if preserve_policy == HARD_WIPE_EXPERIMENT_DIRS:
            table_deletes["experiments"] = exec_delete(
                cur,
                f"DELETE FROM experiments WHERE experiment_id IN ({placeholders_exp})",
                scope.experiment_ids,
            )

        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise

    now = utc_now_iso()

    if preserve_policy == HARD_WIPE_EXPERIMENT_DIRS:
        deleted_experiment_dirs = 0
        for experiment_id in scope.experiment_ids:
            exp_dir = paths.experiments_dir / experiment_id
            if exp_dir.is_dir():
                shutil.rmtree(exp_dir)
                deleted_experiment_dirs += 1
        conn.close()
        return {
            "deleted_run_dirs": deleted_run_dirs,
            "deleted_experiment_dirs": deleted_experiment_dirs,
            "table_deletes": table_deletes,
            "updated_manifests": 0,
            "preserve_policy": preserve_policy,
        }

    updated_manifests = 0
    rewritten_docs = 0
    for experiment_id in scope.experiment_ids:
        manifest_path = paths.experiments_dir / experiment_id / "experiment.json"
        manifest = load_json_mapping(manifest_path)
        if not manifest:
            continue

        manifest["status"] = "draft"
        manifest["runs"] = []
        manifest["champion_run_id"] = None
        manifest["updated_at"] = now
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        updated_manifests += 1

        if preserve_policy == PRESERVE_ONLY_CONFIGS:
            doc_path = manifest_path.parent / "EXPERIMENT.md"
            doc_path.write_text(render_experiment_doc(manifest), encoding="utf-8")
            rewritten_docs += 1

        experiment_service._index_experiment_manifest(paths.store_root, manifest)

    conn.close()

    return {
        "deleted_run_dirs": deleted_run_dirs,
        "deleted_experiment_dirs": 0,
        "table_deletes": table_deletes,
        "updated_manifests": updated_manifests,
        "rewritten_experiment_docs": rewritten_docs,
        "preserve_policy": preserve_policy,
    }


def make_output(
    *,
    paths: StorePaths,
    scope: Scope,
    preserve_policy: str,
    execute: bool,
    allow_active_writers: bool,
    allow_overlap: bool,
    mutation_result: dict[str, Any] | None,
) -> dict[str, Any]:
    blocked_reasons: list[str] = []
    if scope.active_writers and not allow_active_writers:
        blocked_reasons.append("active_writers_detected")
    if (scope.overlaps_manifest or scope.overlaps_db_runs) and not allow_overlap:
        blocked_reasons.append("run_overlap_detected")

    return {
        "store_root": str(paths.store_root),
        "db_path": str(paths.db_path),
        "mode": "execute" if execute else "dry_run",
        "preserve_policy": preserve_policy,
        "target_experiment_ids": scope.experiment_ids,
        "manifest_runs": scope.manifest_runs,
        "target_run_ids": scope.target_run_ids,
        "related_ids": {
            "job_ids": scope.job_ids,
            "logical_run_ids": scope.logical_run_ids,
            "study_ids": scope.study_ids,
            "ensemble_ids": scope.ensemble_ids,
        },
        "db_counts": scope.db_counts,
        "active_writers": scope.active_writers,
        "overlaps": {
            "manifest": scope.overlaps_manifest,
            "db_runs": scope.overlaps_db_runs,
        },
        "blocked": bool(blocked_reasons),
        "blocked_reasons": blocked_reasons,
        "allow_flags": {
            "allow_active_writers": allow_active_writers,
            "allow_overlap": allow_overlap,
        },
        "mutation_result": mutation_result,
    }


def main() -> int:
    args = parse_args()
    paths = ensure_paths(args.store_root)
    experiment_ids = sorted({value.strip() for value in args.experiment_id if value.strip()})
    if not experiment_ids:
        raise SystemExit("no_target_experiment_ids")

    scope = gather_scope(paths, experiment_ids)

    blocked = False
    if scope.active_writers and not args.allow_active_writers:
        blocked = True
    if (scope.overlaps_manifest or scope.overlaps_db_runs) and not args.allow_overlap:
        blocked = True

    mutation_result: dict[str, Any] | None = None

    if args.execute:
        if blocked:
            output = make_output(
                paths=paths,
                scope=scope,
                preserve_policy=args.preserve_policy,
                execute=True,
                allow_active_writers=args.allow_active_writers,
                allow_overlap=args.allow_overlap,
                mutation_result=None,
            )
            dump = json.dumps(output, indent=2 if args.pretty else None, sort_keys=True)
            print(dump)
            return 1

        mutation_result = mutate(paths, scope, args.preserve_policy)
        # Refresh scope after mutation for post-state counts.
        scope = gather_scope(paths, experiment_ids)

    output = make_output(
        paths=paths,
        scope=scope,
        preserve_policy=args.preserve_policy,
        execute=args.execute,
        allow_active_writers=args.allow_active_writers,
        allow_overlap=args.allow_overlap,
        mutation_result=mutation_result,
    )
    dump = json.dumps(output, indent=2 if args.pretty else None, sort_keys=True)
    print(dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
