#!/usr/bin/env python3
"""Read-only impact analysis for numereng store operations."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StorePaths:
    store_root: Path
    db_path: Path
    runs_dir: Path
    experiments_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect store-impact summary without mutation.")
    parser.add_argument(
        "--store-root",
        default=".numereng",
        help="Store root path (default: .numereng)",
    )
    parser.add_argument(
        "--experiment-id",
        action="append",
        default=[],
        help="Target experiment ID (repeatable)",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="Explicit target run ID (repeatable)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser.parse_args()


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
    matches: list[str] = []
    for line in output.splitlines():
        if pattern.search(line):
            matches.append(line.strip())
    return matches


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


def ensure_paths(store_root_raw: str) -> StorePaths:
    store_root = Path(store_root_raw).expanduser().resolve()
    return StorePaths(
        store_root=store_root,
        db_path=store_root / "numereng.db",
        runs_dir=store_root / "runs",
        experiments_dir=store_root / "experiments",
    )


def count_rows(cur: sqlite3.Cursor, table: str, where_clause: str, params: list[str]) -> int:
    query = f"SELECT COUNT(*) FROM {table} WHERE {where_clause}"  # noqa: S608
    return int(cur.execute(query, params).fetchone()[0])


def select_values(cur: sqlite3.Cursor, query: str, params: list[str]) -> list[str]:
    rows = cur.execute(query, params).fetchall()
    values: list[str] = []
    for row in rows:
        value = row[0]
        if isinstance(value, str) and value:
            values.append(value)
    return values


def gather_manifest_runs(paths: StorePaths, experiment_ids: list[str]) -> dict[str, list[str]]:
    manifest_runs: dict[str, list[str]] = {}
    for experiment_id in sorted(set(experiment_ids)):
        manifest_path = paths.experiments_dir / experiment_id / "experiment.json"
        payload = load_json_mapping(manifest_path)
        runs_raw = payload.get("runs")
        run_ids: list[str] = []
        if isinstance(runs_raw, list):
            for value in runs_raw:
                if isinstance(value, str) and value.strip():
                    run_ids.append(value.strip())
        manifest_runs[experiment_id] = sorted(set(run_ids))
    return manifest_runs


def gather_db_runs_for_experiments(cur: sqlite3.Cursor, experiment_ids: list[str]) -> list[str]:
    if not experiment_ids:
        return []
    placeholders = ",".join("?" for _ in experiment_ids)
    query = f"SELECT run_id FROM runs WHERE experiment_id IN ({placeholders})"  # noqa: S608
    return sorted(set(select_values(cur, query, experiment_ids)))


def gather_job_canonical_runs(cur: sqlite3.Cursor, experiment_ids: list[str]) -> list[str]:
    if not experiment_ids:
        return []
    placeholders = ",".join("?" for _ in experiment_ids)
    query = (
        f"SELECT canonical_run_id FROM run_jobs "
        f"WHERE experiment_id IN ({placeholders}) "
        "AND canonical_run_id IS NOT NULL AND canonical_run_id != ''"
    )
    return sorted(set(select_values(cur, query, experiment_ids)))


def gather_runs_from_filesystem(paths: StorePaths, experiment_ids: set[str]) -> list[str]:
    if not paths.runs_dir.is_dir() or not experiment_ids:
        return []

    run_ids: list[str] = []
    for run_dir in sorted(paths.runs_dir.iterdir(), key=lambda item: item.name):
        if not run_dir.is_dir():
            continue
        payload = load_json_mapping(run_dir / "run.json")
        if payload.get("experiment_id") in experiment_ids:
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


def main() -> int:
    args = parse_args()
    paths = ensure_paths(args.store_root)

    experiment_ids = sorted({value.strip() for value in args.experiment_id if value.strip()})
    explicit_run_ids = sorted({value.strip() for value in args.run_id if value.strip()})
    active_writers = detect_active_writers()

    summary: dict[str, Any] = {
        "store_root": str(paths.store_root),
        "db_path": str(paths.db_path),
        "db_exists": paths.db_path.is_file(),
        "target_experiment_ids": experiment_ids,
        "explicit_run_ids": explicit_run_ids,
        "active_writers": active_writers,
        "manifest_runs": {},
        "derived_run_ids": {},
        "target_run_ids": [],
        "db_counts": {},
        "related_ids": {},
        "overlaps": {},
    }

    manifest_runs = gather_manifest_runs(paths, experiment_ids)
    summary["manifest_runs"] = manifest_runs

    target_run_ids: set[str] = set(explicit_run_ids)
    for values in manifest_runs.values():
        target_run_ids.update(values)

    if not paths.db_path.is_file():
        summary["derived_run_ids"] = {
            "from_experiment_runs_table": [],
            "from_experiment_run_jobs": [],
            "from_run_manifests": gather_runs_from_filesystem(paths, set(experiment_ids)),
        }
        target_run_ids.update(summary["derived_run_ids"]["from_run_manifests"])
        summary["target_run_ids"] = sorted(target_run_ids)
        dump = json.dumps(summary, indent=2 if args.pretty else None, sort_keys=True)
        print(dump)
        return 0

    conn = sqlite3.connect(paths.db_path)
    cur = conn.cursor()

    runs_from_experiments = gather_db_runs_for_experiments(cur, experiment_ids)
    runs_from_jobs = gather_job_canonical_runs(cur, experiment_ids)
    runs_from_filesystem = gather_runs_from_filesystem(paths, set(experiment_ids))

    target_run_ids.update(runs_from_experiments)
    target_run_ids.update(runs_from_jobs)
    target_run_ids.update(runs_from_filesystem)

    summary["derived_run_ids"] = {
        "from_experiment_runs_table": runs_from_experiments,
        "from_experiment_run_jobs": runs_from_jobs,
        "from_run_manifests": runs_from_filesystem,
    }
    summary["target_run_ids"] = sorted(target_run_ids)

    # DB counts by experiment scope
    db_counts: dict[str, Any] = {}
    if experiment_ids:
        placeholders_exp = ",".join("?" for _ in experiment_ids)
        db_counts["by_experiment"] = {
            "experiments": count_rows(cur, "experiments", f"experiment_id IN ({placeholders_exp})", experiment_ids),
            "runs": count_rows(cur, "runs", f"experiment_id IN ({placeholders_exp})", experiment_ids),
            "run_jobs": count_rows(cur, "run_jobs", f"experiment_id IN ({placeholders_exp})", experiment_ids),
            "logical_runs": count_rows(cur, "logical_runs", f"experiment_id IN ({placeholders_exp})", experiment_ids),
            "hpo_studies": count_rows(cur, "hpo_studies", f"experiment_id IN ({placeholders_exp})", experiment_ids),
            "ensembles": count_rows(cur, "ensembles", f"experiment_id IN ({placeholders_exp})", experiment_ids),
        }

    target_runs_list = sorted(target_run_ids)
    if target_runs_list:
        placeholders_runs = ",".join("?" for _ in target_runs_list)
        db_counts["by_run"] = {
            "metrics": count_rows(cur, "metrics", f"run_id IN ({placeholders_runs})", target_runs_list),
            "run_artifacts": count_rows(cur, "run_artifacts", f"run_id IN ({placeholders_runs})", target_runs_list),
            "run_attempts_canonical": count_rows(
                cur,
                "run_attempts",
                f"canonical_run_id IN ({placeholders_runs})",
                target_runs_list,
            ),
            "run_jobs_canonical": count_rows(
                cur,
                "run_jobs",
                f"canonical_run_id IN ({placeholders_runs})",
                target_runs_list,
            ),
            "hpo_trials_run": count_rows(cur, "hpo_trials", f"run_id IN ({placeholders_runs})", target_runs_list),
            "ensemble_components_run": count_rows(
                cur,
                "ensemble_components",
                f"run_id IN ({placeholders_runs})",
                target_runs_list,
            ),
        }

    # Related IDs for job/logical/ensemble/hpo scopes
    related: dict[str, Any] = {
        "job_ids": [],
        "logical_run_ids": [],
        "study_ids": [],
        "ensemble_ids": [],
    }

    if experiment_ids or target_runs_list:
        clause_parts: list[str] = []
        params: list[str] = []
        if experiment_ids:
            placeholders_exp = ",".join("?" for _ in experiment_ids)
            clause_parts.append(f"experiment_id IN ({placeholders_exp})")
            params.extend(experiment_ids)
        if target_runs_list:
            placeholders_runs = ",".join("?" for _ in target_runs_list)
            clause_parts.append(f"canonical_run_id IN ({placeholders_runs})")
            params.extend(target_runs_list)

        where_clause = " OR ".join(clause_parts)
        job_ids = select_values(cur, f"SELECT DISTINCT job_id FROM run_jobs WHERE {where_clause}", params)
        related["job_ids"] = sorted(set(job_ids))

    if experiment_ids:
        placeholders_exp = ",".join("?" for _ in experiment_ids)
        related["logical_run_ids"] = sorted(
            set(
                select_values(
                    cur,
                    f"SELECT DISTINCT logical_run_id FROM logical_runs WHERE experiment_id IN ({placeholders_exp})",
                    experiment_ids,
                )
            )
        )
        related["study_ids"] = sorted(
            set(
                select_values(
                    cur,
                    f"SELECT DISTINCT study_id FROM hpo_studies WHERE experiment_id IN ({placeholders_exp})",
                    experiment_ids,
                )
            )
        )
        related["ensemble_ids"] = sorted(
            set(
                select_values(
                    cur,
                    f"SELECT DISTINCT ensemble_id FROM ensembles WHERE experiment_id IN ({placeholders_exp})",
                    experiment_ids,
                )
            )
        )

    if related["job_ids"]:
        placeholders_jobs = ",".join("?" for _ in related["job_ids"])
        db_counts["by_job"] = {
            "run_job_logs": count_rows(cur, "run_job_logs", f"job_id IN ({placeholders_jobs})", related["job_ids"]),
            "run_job_events": count_rows(cur, "run_job_events", f"job_id IN ({placeholders_jobs})", related["job_ids"]),
            "run_job_samples": count_rows(
                cur, "run_job_samples", f"job_id IN ({placeholders_jobs})", related["job_ids"]
            ),
        }

    if related["study_ids"]:
        placeholders_studies = ",".join("?" for _ in related["study_ids"])
        db_counts["by_study"] = {
            "hpo_trials": count_rows(cur, "hpo_trials", f"study_id IN ({placeholders_studies})", related["study_ids"]),
        }

    if related["ensemble_ids"]:
        placeholders_ensembles = ",".join("?" for _ in related["ensemble_ids"])
        db_counts["by_ensemble"] = {
            "ensemble_metrics": count_rows(
                cur,
                "ensemble_metrics",
                f"ensemble_id IN ({placeholders_ensembles})",
                related["ensemble_ids"],
            ),
            "ensemble_components": count_rows(
                cur,
                "ensemble_components",
                f"ensemble_id IN ({placeholders_ensembles})",
                related["ensemble_ids"],
            ),
        }

    overlaps: dict[str, Any] = {
        "manifest": gather_manifest_overlaps(paths, set(experiment_ids), set(target_runs_list)),
        "db_runs": [],
    }

    if experiment_ids and target_runs_list:
        placeholders_runs = ",".join("?" for _ in target_runs_list)
        placeholders_exp = ",".join("?" for _ in experiment_ids)
        query = (
            f"SELECT run_id, experiment_id FROM runs "
            f"WHERE run_id IN ({placeholders_runs}) "
            f"AND COALESCE(experiment_id, '') NOT IN ({placeholders_exp})"
        )
        rows = cur.execute(query, target_runs_list + experiment_ids).fetchall()
        overlaps["db_runs"] = [{"run_id": row[0], "experiment_id": row[1]} for row in rows if isinstance(row[0], str)]

    conn.close()

    summary["db_counts"] = db_counts
    summary["related_ids"] = related
    summary["overlaps"] = overlaps

    dump = json.dumps(summary, indent=2 if args.pretty else None, sort_keys=True)
    print(dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
