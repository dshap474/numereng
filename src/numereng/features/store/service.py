"""SQLite-backed store bootstrap, run indexing, and diagnostics helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from numereng.features.store.layout import (
    CANONICAL_STORE_TOP_LEVEL_DIRS,
    CANONICAL_STORE_TOP_LEVEL_FILES,
    targeted_stray_paths,
)

_SAFE_ID = re.compile(r"^[\w\-.]+$")
_SCHEMA_MIGRATIONS = (
    "2026_02_store_index_v1",
    "2026_02_store_index_v2_cloud_jobs",
    "2026_02_store_index_v3_experiments",
    "2026_02_store_index_v4_cloud_jobs_pk_provider",
    "2026_02_store_index_v5_hpo_ensemble",
    "2026_02_store_index_v6_run_ops_telemetry",
)
_SCHEMA_MIGRATION_NAME = _SCHEMA_MIGRATIONS[-1]

_REQUIRED_TABLES = (
    "schema_migrations",
    "runs",
    "metrics",
    "run_artifacts",
    "run_jobs",
    "run_job_events",
    "run_job_logs",
    "run_job_samples",
    "logical_runs",
    "run_attempts",
    "cloud_jobs",
    "experiments",
    "hpo_studies",
    "hpo_trials",
    "ensembles",
    "ensemble_components",
    "ensemble_metrics",
)

_CANONICAL_ARTIFACTS = (
    ("manifest", "run.json"),
    ("resolved_config", "resolved.json"),
    ("resolved_config_legacy", "resolved.yaml"),
    ("results", "results.json"),
    ("metrics", "metrics.json"),
)

_SQLITE_CORRUPTION_TOKENS = (
    "database disk image is malformed",
    "malformed",
    "file is not a database",
    "database corrupted",
)

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        name TEXT PRIMARY KEY,
        applied_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        run_hash TEXT NOT NULL,
        status TEXT NOT NULL,
        run_type TEXT NOT NULL,
        created_at TEXT NOT NULL,
        finished_at TEXT,
        config_hash TEXT,
        experiment_id TEXT,
        run_path TEXT NOT NULL,
        manifest_json TEXT NOT NULL,
        manifest_mtime_ns INTEGER NOT NULL,
        ingested_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS metrics (
        run_id TEXT NOT NULL,
        name TEXT NOT NULL,
        value REAL,
        value_json TEXT,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (run_id, name)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS run_artifacts (
        run_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        relative_path TEXT NOT NULL,
        absolute_path TEXT NOT NULL,
        exists_flag INTEGER NOT NULL,
        size_bytes INTEGER,
        sha256 TEXT,
        mtime_ns INTEGER,
        PRIMARY KEY (run_id, kind, relative_path)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS run_jobs (
        job_id TEXT PRIMARY KEY,
        batch_id TEXT NOT NULL,
        experiment_id TEXT,
        logical_run_id TEXT,
        operation_type TEXT,
        attempt_no INTEGER NOT NULL DEFAULT 1,
        attempt_id TEXT,
        config_id TEXT NOT NULL,
        config_source TEXT NOT NULL,
        config_path TEXT NOT NULL,
        config_sha256 TEXT NOT NULL,
        request_json TEXT NOT NULL,
        job_type TEXT NOT NULL,
        status TEXT NOT NULL,
        queue_name TEXT NOT NULL,
        priority INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        queued_at TEXT,
        started_at TEXT,
        finished_at TEXT,
        updated_at TEXT NOT NULL,
        worker_id TEXT,
        pid INTEGER,
        exit_code INTEGER,
        signal INTEGER,
        backend TEXT,
        tier TEXT,
        budget REAL,
        timeout_seconds INTEGER,
        canonical_run_id TEXT,
        external_run_id TEXT,
        run_dir TEXT,
        cancel_requested INTEGER NOT NULL DEFAULT 0,
        error_json TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS run_job_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT NOT NULL,
        sequence INTEGER NOT NULL,
        event_type TEXT NOT NULL,
        source TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS run_job_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT NOT NULL,
        line_no INTEGER NOT NULL,
        stream TEXT NOT NULL,
        line TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS run_job_samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT NOT NULL,
        cpu_percent REAL,
        rss_gb REAL,
        ram_available_gb REAL,
        gpu_percent REAL,
        gpu_mem_gb REAL,
        process_cpu_percent REAL,
        process_rss_gb REAL,
        host_cpu_percent REAL,
        host_ram_available_gb REAL,
        host_ram_used_gb REAL,
        host_gpu_percent REAL,
        host_gpu_mem_used_gb REAL,
        scope TEXT,
        status TEXT,
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS logical_runs (
        logical_run_id TEXT PRIMARY KEY,
        experiment_id TEXT,
        operation_type TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        metadata_json TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS run_attempts (
        attempt_id TEXT PRIMARY KEY,
        logical_run_id TEXT NOT NULL,
        job_id TEXT NOT NULL,
        attempt_no INTEGER NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT,
        updated_at TEXT NOT NULL,
        worker_id TEXT,
        pid INTEGER,
        exit_code INTEGER,
        signal INTEGER,
        error_json TEXT,
        canonical_run_id TEXT,
        external_run_id TEXT,
        run_dir TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS cloud_jobs (
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
        PRIMARY KEY (run_id, provider, provider_job_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        metadata_json TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS hpo_studies (
        study_id TEXT PRIMARY KEY,
        experiment_id TEXT,
        study_name TEXT NOT NULL,
        status TEXT NOT NULL,
        metric TEXT NOT NULL,
        direction TEXT NOT NULL,
        n_trials INTEGER NOT NULL,
        sampler TEXT NOT NULL,
        seed INTEGER,
        best_trial_number INTEGER,
        best_value REAL,
        best_run_id TEXT,
        config_json TEXT,
        storage_path TEXT,
        error_message TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS hpo_trials (
        study_id TEXT NOT NULL,
        trial_number INTEGER NOT NULL,
        status TEXT NOT NULL,
        value REAL,
        run_id TEXT,
        config_path TEXT,
        params_json TEXT,
        error_message TEXT,
        started_at TEXT,
        finished_at TEXT,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (study_id, trial_number)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS ensembles (
        ensemble_id TEXT PRIMARY KEY,
        experiment_id TEXT,
        name TEXT NOT NULL,
        method TEXT NOT NULL,
        target TEXT NOT NULL,
        metric TEXT NOT NULL,
        status TEXT NOT NULL,
        config_json TEXT,
        artifacts_path TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS ensemble_components (
        ensemble_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        weight REAL NOT NULL,
        rank INTEGER NOT NULL,
        PRIMARY KEY (ensemble_id, run_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS ensemble_metrics (
        ensemble_id TEXT NOT NULL,
        name TEXT NOT NULL,
        value REAL,
        PRIMARY KEY (ensemble_id, name)
    );
    """,
)

_INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS idx_runs_status_created ON runs(status, created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_runs_experiment_created ON runs(experiment_id, created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id);",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_run_kind ON run_artifacts(run_id, kind);",
    "CREATE INDEX IF NOT EXISTS idx_run_jobs_experiment_created ON run_jobs(experiment_id, created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_run_jobs_status_created ON run_jobs(status, created_at DESC);",
    (
        "CREATE INDEX IF NOT EXISTS idx_run_jobs_logical_attempt "
        "ON run_jobs(logical_run_id, attempt_no DESC, created_at DESC);"
    ),
    "CREATE INDEX IF NOT EXISTS idx_run_jobs_batch_created ON run_jobs(batch_id, created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_run_jobs_canonical_finished ON run_jobs(canonical_run_id, finished_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_run_job_events_job_id_id ON run_job_events(job_id, id ASC);",
    "CREATE INDEX IF NOT EXISTS idx_run_job_logs_job_id_id ON run_job_logs(job_id, id ASC);",
    "CREATE INDEX IF NOT EXISTS idx_run_job_samples_job_id_id ON run_job_samples(job_id, id ASC);",
    "CREATE INDEX IF NOT EXISTS idx_logical_runs_exp_updated ON logical_runs(experiment_id, updated_at DESC);",
    (
        "CREATE INDEX IF NOT EXISTS idx_run_attempts_logical_attempt "
        "ON run_attempts(logical_run_id, attempt_no DESC, created_at DESC);"
    ),
    "CREATE INDEX IF NOT EXISTS idx_run_attempts_job_created ON run_attempts(job_id, created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_cloud_jobs_run_status ON cloud_jobs(run_id, status);",
    "CREATE INDEX IF NOT EXISTS idx_cloud_jobs_provider ON cloud_jobs(provider, backend);",
    "CREATE INDEX IF NOT EXISTS idx_experiments_status_updated ON experiments(status, updated_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_hpo_studies_exp_updated ON hpo_studies(experiment_id, updated_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_hpo_studies_status_updated ON hpo_studies(status, updated_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_hpo_trials_study_trial ON hpo_trials(study_id, trial_number ASC);",
    "CREATE INDEX IF NOT EXISTS idx_ensembles_exp_created ON ensembles(experiment_id, created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_ensemble_components_ensemble_rank ON ensemble_components(ensemble_id, rank ASC);",
    "CREATE INDEX IF NOT EXISTS idx_ensemble_metrics_ensemble_name ON ensemble_metrics(ensemble_id, name);",
)


class StoreError(Exception):
    """Base exception for store lifecycle and indexing failures."""


class StoreRunNotFoundError(StoreError):
    """Raised when the target run directory does not exist."""


class StoreRunManifestNotFoundError(StoreError):
    """Raised when a run directory does not include `run.json`."""


class StoreRunManifestInvalidError(StoreError):
    """Raised when `run.json` exists but is not valid JSON mapping."""


@dataclass(frozen=True)
class StoreInitResult:
    """Result payload for store DB bootstrap."""

    store_root: Path
    db_path: Path
    created: bool
    schema_migration: str


@dataclass(frozen=True)
class StoreIndexResult:
    """Result payload for one indexed run."""

    run_id: str
    status: str
    metrics_indexed: int
    artifacts_indexed: int
    run_path: Path
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class StoreRebuildFailure:
    """One run-level rebuild failure."""

    run_id: str
    error: str


@dataclass(frozen=True)
class StoreRebuildResult:
    """Result payload for full rebuild over `runs/*`."""

    store_root: Path
    db_path: Path
    scanned_runs: int
    indexed_runs: int
    failed_runs: int
    failures: tuple[StoreRebuildFailure, ...]


@dataclass(frozen=True)
class StoreDoctorResult:
    """Result payload for store consistency checks."""

    store_root: Path
    db_path: Path
    ok: bool
    issues: tuple[str, ...]
    stats: dict[str, int]
    stray_cleanup_applied: bool = False
    deleted_paths: tuple[str, ...] = ()
    missing_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class StoreMaterializeVizArtifactsFailure:
    """One run-level viz artifact materialization failure."""

    run_id: str
    error: str


@dataclass(frozen=True)
class StoreMaterializeVizArtifactsResult:
    """Result payload for persisted visualization artifact backfills."""

    store_root: Path
    kind: str
    scoped_run_count: int
    created_count: int
    skipped_count: int
    failed_count: int
    failures: tuple[StoreMaterializeVizArtifactsFailure, ...]


@dataclass(frozen=True)
class StoreCloudJobUpsert:
    """Payload for inserting/updating one cloud job metadata row."""

    run_id: str
    provider: str
    backend: str
    provider_job_id: str
    status: str
    region: str | None = None
    image_uri: str | None = None
    output_s3_uri: str | None = None
    metadata_json: str | None = None
    error_message: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


@dataclass(frozen=True)
class StoreExperimentRecord:
    """One experiment row indexed in store DB."""

    experiment_id: str
    name: str
    status: str
    created_at: str
    updated_at: str
    metadata_json: str


@dataclass(frozen=True)
class StoreHpoStudyUpsert:
    """Payload for inserting/updating one HPO study row."""

    study_id: str
    experiment_id: str | None
    study_name: str
    status: str
    metric: str
    direction: str
    n_trials: int
    sampler: str
    seed: int | None = None
    best_trial_number: int | None = None
    best_value: float | None = None
    best_run_id: str | None = None
    config_json: str | None = None
    storage_path: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class StoreHpoStudyRecord:
    """One HPO study row loaded from store DB."""

    study_id: str
    experiment_id: str | None
    study_name: str
    status: str
    metric: str
    direction: str
    n_trials: int
    sampler: str
    seed: int | None
    best_trial_number: int | None
    best_value: float | None
    best_run_id: str | None
    config_json: str | None
    storage_path: str | None
    error_message: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class StoreHpoTrialUpsert:
    """Payload for inserting/updating one HPO trial row."""

    study_id: str
    trial_number: int
    status: str
    value: float | None = None
    run_id: str | None = None
    config_path: str | None = None
    params_json: str | None = None
    error_message: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


@dataclass(frozen=True)
class StoreHpoTrialRecord:
    """One HPO trial row loaded from store DB."""

    study_id: str
    trial_number: int
    status: str
    value: float | None
    run_id: str | None
    config_path: str | None
    params_json: str | None
    error_message: str | None
    started_at: str | None
    finished_at: str | None
    updated_at: str


@dataclass(frozen=True)
class StoreEnsembleUpsert:
    """Payload for inserting/updating one ensemble row."""

    ensemble_id: str
    experiment_id: str | None
    name: str
    method: str
    target: str
    metric: str
    status: str
    config_json: str | None = None
    artifacts_path: str | None = None


@dataclass(frozen=True)
class StoreEnsembleRecord:
    """One ensemble row loaded from store DB."""

    ensemble_id: str
    experiment_id: str | None
    name: str
    method: str
    target: str
    metric: str
    status: str
    config_json: str | None
    artifacts_path: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class StoreEnsembleComponentUpsert:
    """Payload for inserting/updating one ensemble component row."""

    ensemble_id: str
    run_id: str
    weight: float
    rank: int


@dataclass(frozen=True)
class StoreEnsembleComponentRecord:
    """One ensemble component row loaded from store DB."""

    ensemble_id: str
    run_id: str
    weight: float
    rank: int


@dataclass(frozen=True)
class StoreEnsembleMetricUpsert:
    """Payload for inserting/updating one ensemble metric row."""

    ensemble_id: str
    name: str
    value: float | None


@dataclass(frozen=True)
class StoreEnsembleMetricRecord:
    """One ensemble metric row loaded from store DB."""

    ensemble_id: str
    name: str
    value: float | None


def resolve_store_root(store_root: str | Path = ".numereng") -> Path:
    """Resolve one store root path."""

    return Path(store_root).expanduser().resolve()


def init_store_db(*, store_root: str | Path = ".numereng") -> StoreInitResult:
    """Create store root + DB schema if missing."""

    resolved_store_root = resolve_store_root(store_root)
    resolved_store_root.mkdir(parents=True, exist_ok=True)
    (resolved_store_root / "runs").mkdir(parents=True, exist_ok=True)

    db_path = resolved_store_root / "numereng.db"
    created = not db_path.exists()

    try:
        with _connect_rw(db_path) as conn:
            _init_schema(conn)
    except sqlite3.Error as exc:
        raise _wrap_sqlite_error(operation="init_store_db", db_path=db_path, exc=exc) from exc

    return StoreInitResult(
        store_root=resolved_store_root,
        db_path=db_path,
        created=created,
        schema_migration=_SCHEMA_MIGRATION_NAME,
    )


def index_run(*, store_root: str | Path = ".numereng", run_id: str) -> StoreIndexResult:
    """Index one run from filesystem artifacts into SQLite."""

    safe_run_id = _ensure_safe_run_id(run_id)
    init_result = init_store_db(store_root=store_root)

    try:
        with _connect_rw(init_result.db_path) as conn:
            _init_schema(conn)
            return _index_run_with_connection(
                conn,
                store_root=init_result.store_root,
                run_id=safe_run_id,
            )
    except sqlite3.Error as exc:
        raise _wrap_sqlite_error(operation="index_run", db_path=init_result.db_path, exc=exc) from exc


def rebuild_run_index(*, store_root: str | Path = ".numereng") -> StoreRebuildResult:
    """Re-index every run directory under `runs/*` into SQLite."""

    init_result = init_store_db(store_root=store_root)
    runs_dir = init_result.store_root / "runs"

    scanned_runs = 0
    indexed_runs = 0
    failures: list[StoreRebuildFailure] = []

    if not runs_dir.is_dir():
        return StoreRebuildResult(
            store_root=init_result.store_root,
            db_path=init_result.db_path,
            scanned_runs=0,
            indexed_runs=0,
            failed_runs=0,
            failures=(),
        )

    try:
        with _connect_rw(init_result.db_path) as conn:
            _init_schema(conn)
            for run_dir in sorted(runs_dir.iterdir(), key=lambda item: item.name):
                if not run_dir.is_dir():
                    continue
                scanned_runs += 1
                run_id = run_dir.name
                try:
                    _ensure_safe_run_id(run_id)
                    _index_run_with_connection(conn, store_root=init_result.store_root, run_id=run_id)
                except StoreError as exc:
                    failures.append(StoreRebuildFailure(run_id=run_id, error=str(exc)))
                    continue
                except sqlite3.Error as exc:
                    wrapped = _wrap_sqlite_error(
                        operation=f"rebuild_run_index:{run_id}",
                        db_path=init_result.db_path,
                        exc=exc,
                    )
                    failures.append(StoreRebuildFailure(run_id=run_id, error=str(wrapped)))
                    continue
                indexed_runs += 1
    except sqlite3.Error as exc:
        raise _wrap_sqlite_error(operation="rebuild_run_index", db_path=init_result.db_path, exc=exc) from exc

    return StoreRebuildResult(
        store_root=init_result.store_root,
        db_path=init_result.db_path,
        scanned_runs=scanned_runs,
        indexed_runs=indexed_runs,
        failed_runs=len(failures),
        failures=tuple(failures),
    )


def doctor_store(
    *,
    store_root: str | Path = ".numereng",
    fix_strays: bool = False,
) -> StoreDoctorResult:
    """Validate store shape and DB/filesystem consistency."""

    resolved_store_root = resolve_store_root(store_root)
    db_path = resolved_store_root / "numereng.db"
    runs_dir = resolved_store_root / "runs"

    issues: list[str] = []
    stats: dict[str, int] = {}
    deleted_paths: list[str] = []
    missing_paths: list[str] = []

    stats["canonical_top_level_dirs"] = len(CANONICAL_STORE_TOP_LEVEL_DIRS)
    stats["canonical_top_level_files"] = len(CANONICAL_STORE_TOP_LEVEL_FILES)

    if fix_strays:
        for stray_path in targeted_stray_paths(store_root=resolved_store_root):
            if not stray_path.exists():
                missing_paths.append(str(stray_path))
                continue
            if not stray_path.is_dir():
                issues.append(f"targeted_stray_not_dir:{stray_path}")
                continue
            try:
                shutil.rmtree(stray_path)
            except OSError as exc:
                issues.append(f"targeted_stray_delete_failed:{stray_path}:{exc}")
                continue
            deleted_paths.append(str(stray_path))

    stats["targeted_strays_deleted"] = len(deleted_paths)
    stats["targeted_strays_missing"] = len(missing_paths)

    filesystem_runs = 0
    run_dirs: list[Path] = []
    if runs_dir.is_dir():
        run_dirs = [path for path in sorted(runs_dir.iterdir(), key=lambda item: item.name) if path.is_dir()]
        filesystem_runs = len(run_dirs)
    stats["filesystem_runs"] = filesystem_runs

    if not db_path.exists():
        issues.append("store_db_missing")
        return StoreDoctorResult(
            store_root=resolved_store_root,
            db_path=db_path,
            ok=False,
            issues=tuple(issues),
            stats=stats,
            stray_cleanup_applied=fix_strays,
            deleted_paths=tuple(deleted_paths),
            missing_paths=tuple(missing_paths),
        )

    try:
        conn = _connect_read_only(db_path)
    except (StoreError, sqlite3.Error):
        issues.append("store_db_unreadable")
        return StoreDoctorResult(
            store_root=resolved_store_root,
            db_path=db_path,
            ok=False,
            issues=tuple(issues),
            stats=stats,
            stray_cleanup_applied=fix_strays,
            deleted_paths=tuple(deleted_paths),
            missing_paths=tuple(missing_paths),
        )

    with conn:
        missing_tables = [name for name in _REQUIRED_TABLES if not _table_exists(conn, name)]
        if missing_tables:
            issues.append("store_db_missing_tables:" + ",".join(missing_tables))

        if _table_exists(conn, "runs"):
            indexed_runs = _count_query(conn, "SELECT COUNT(*) FROM runs")
            stats["indexed_runs"] = indexed_runs
            if indexed_runs != filesystem_runs:
                issues.append("run_count_mismatch")

        if _table_exists(conn, "metrics"):
            stats["indexed_metrics"] = _count_query(conn, "SELECT COUNT(*) FROM metrics")

        if _table_exists(conn, "run_artifacts"):
            stats["indexed_artifacts"] = _count_query(conn, "SELECT COUNT(*) FROM run_artifacts")
        if _table_exists(conn, "cloud_jobs"):
            stats["indexed_cloud_jobs"] = _count_query(conn, "SELECT COUNT(*) FROM cloud_jobs")

    finished_runs = 0
    finished_missing_resolved = 0
    finished_missing_results = 0
    finished_missing_metrics = 0

    for run_dir in run_dirs:
        manifest_path = run_dir / "run.json"
        if not manifest_path.is_file():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue

        status = payload.get("status")
        if status != "FINISHED":
            continue

        finished_runs += 1
        has_resolved = (run_dir / "resolved.json").is_file() or (run_dir / "resolved.yaml").is_file()
        if not has_resolved:
            finished_missing_resolved += 1
        if not (run_dir / "results.json").is_file():
            finished_missing_results += 1
        if not (run_dir / "metrics.json").is_file():
            finished_missing_metrics += 1

    stats["finished_runs"] = finished_runs
    stats["finished_missing_resolved"] = finished_missing_resolved
    stats["finished_missing_results"] = finished_missing_results
    stats["finished_missing_metrics"] = finished_missing_metrics

    if finished_missing_resolved:
        issues.append(f"finished_runs_missing_resolved:{finished_missing_resolved}")
    if finished_missing_results:
        issues.append(f"finished_runs_missing_results:{finished_missing_results}")
    if finished_missing_metrics:
        issues.append(f"finished_runs_missing_metrics:{finished_missing_metrics}")

    return StoreDoctorResult(
        store_root=resolved_store_root,
        db_path=db_path,
        ok=not issues,
        issues=tuple(issues),
        stats=stats,
        stray_cleanup_applied=fix_strays,
        deleted_paths=tuple(deleted_paths),
        missing_paths=tuple(missing_paths),
    )


def materialize_viz_artifacts(
    *,
    store_root: str | Path = ".numereng",
    kind: str,
    run_id: str | None = None,
    experiment_id: str | None = None,
    all_runs: bool = False,
) -> StoreMaterializeVizArtifactsResult:
    """Materialize persisted viz artifacts for existing runs."""

    resolved_kind = kind.strip()
    if resolved_kind != "per-era-corr":
        raise StoreError(f"store_viz_artifact_kind_unsupported:{kind}")

    scope_flags = int(run_id is not None) + int(experiment_id is not None) + int(all_runs)
    if scope_flags != 1:
        raise StoreError("store_viz_artifact_scope_invalid")

    resolved_store_root = resolve_store_root(store_root)
    runs_dir = resolved_store_root / "runs"
    if not runs_dir.is_dir():
        if run_id is not None:
            raise StoreError(f"store_run_not_found:{_ensure_safe_run_id(run_id)}")
        return StoreMaterializeVizArtifactsResult(
            store_root=resolved_store_root,
            kind=resolved_kind,
            scoped_run_count=0,
            created_count=0,
            skipped_count=0,
            failed_count=0,
            failures=(),
        )

    target_run_id = _ensure_safe_run_id(run_id) if run_id is not None else None
    target_experiment_id = _ensure_safe_experiment_id(experiment_id) if experiment_id is not None else None
    selected_runs: list[str] = []
    failures: list[StoreMaterializeVizArtifactsFailure] = []
    created_count = 0
    skipped_count = 0

    for run_dir in sorted(runs_dir.iterdir(), key=lambda item: item.name):
        if not run_dir.is_dir():
            continue
        candidate_run_id = run_dir.name
        try:
            _ensure_safe_run_id(candidate_run_id)
        except StoreError:
            continue
        if target_run_id is not None and candidate_run_id != target_run_id:
            continue
        if target_experiment_id is not None:
            try:
                manifest = _load_run_manifest_for_materialize(run_dir)
            except StoreError:
                continue
            if _as_nonempty_str(manifest.get("experiment_id")) != target_experiment_id:
                continue
        selected_runs.append(candidate_run_id)

    if target_run_id is not None and not selected_runs:
        raise StoreError(f"store_run_not_found:{target_run_id}")

    for candidate_run_id in selected_runs:
        run_dir = runs_dir / candidate_run_id
        try:
            created = _materialize_run_per_era_corr(run_dir)
        except StoreError as exc:
            failures.append(StoreMaterializeVizArtifactsFailure(run_id=candidate_run_id, error=str(exc)))
            continue
        if created:
            created_count += 1
        else:
            skipped_count += 1

    return StoreMaterializeVizArtifactsResult(
        store_root=resolved_store_root,
        kind=resolved_kind,
        scoped_run_count=len(selected_runs),
        created_count=created_count,
        skipped_count=skipped_count,
        failed_count=len(failures),
        failures=tuple(failures),
    )


def upsert_cloud_job(
    *,
    store_root: str | Path = ".numereng",
    job: StoreCloudJobUpsert,
) -> None:
    """Insert or update one cloud-job metadata row."""

    run_id = _ensure_safe_run_id(job.run_id)
    if not job.provider.strip():
        raise StoreError("store_cloud_job_provider_invalid")
    if not job.backend.strip():
        raise StoreError("store_cloud_job_backend_invalid")
    if not job.provider_job_id.strip():
        raise StoreError("store_cloud_job_provider_job_id_invalid")
    if not job.status.strip():
        raise StoreError("store_cloud_job_status_invalid")

    init_result = init_store_db(store_root=store_root)
    updated_at = _utc_now_iso()
    try:
        with _connect_rw(init_result.db_path) as conn:
            _init_schema(conn)
            conn.execute(
                """
                INSERT INTO cloud_jobs (
                    run_id,
                    provider,
                    backend,
                    provider_job_id,
                    status,
                    region,
                    image_uri,
                    output_s3_uri,
                    metadata_json,
                    error_message,
                    started_at,
                    finished_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, provider, provider_job_id) DO UPDATE SET
                    provider = excluded.provider,
                    backend = excluded.backend,
                    status = excluded.status,
                    region = excluded.region,
                    image_uri = excluded.image_uri,
                    output_s3_uri = excluded.output_s3_uri,
                    metadata_json = excluded.metadata_json,
                    error_message = excluded.error_message,
                    started_at = excluded.started_at,
                    finished_at = excluded.finished_at,
                    updated_at = excluded.updated_at
                """,
                (
                    run_id,
                    job.provider,
                    job.backend,
                    job.provider_job_id,
                    job.status,
                    job.region,
                    job.image_uri,
                    job.output_s3_uri,
                    job.metadata_json,
                    job.error_message,
                    job.started_at,
                    job.finished_at,
                    updated_at,
                ),
            )
            conn.commit()
    except sqlite3.Error as exc:
        raise _wrap_sqlite_error(operation="upsert_cloud_job", db_path=init_result.db_path, exc=exc) from exc


def upsert_experiment(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str,
    name: str,
    status: str,
    created_at: str,
    updated_at: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Insert or update one experiment metadata row."""

    safe_experiment_id = _ensure_safe_experiment_id(experiment_id)
    if not name.strip():
        raise StoreError("store_experiment_name_invalid")
    if not status.strip():
        raise StoreError("store_experiment_status_invalid")
    if not created_at.strip():
        raise StoreError("store_experiment_created_at_invalid")
    if not updated_at.strip():
        raise StoreError("store_experiment_updated_at_invalid")

    metadata_json = _safe_json_dumps(metadata or {})
    init_result = init_store_db(store_root=store_root)
    with _connect_rw(init_result.db_path) as conn:
        _init_schema(conn)
        conn.execute(
            """
            INSERT INTO experiments (
                experiment_id,
                name,
                status,
                created_at,
                updated_at,
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(experiment_id) DO UPDATE SET
                name = excluded.name,
                status = excluded.status,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at,
                metadata_json = excluded.metadata_json
            """,
            (
                safe_experiment_id,
                name,
                status,
                created_at,
                updated_at,
                metadata_json,
            ),
        )
        conn.commit()


def get_experiment(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str,
) -> StoreExperimentRecord | None:
    """Load one experiment metadata row from store DB."""

    safe_experiment_id = _ensure_safe_experiment_id(experiment_id)
    init_result = init_store_db(store_root=store_root)

    with _connect_rw(init_result.db_path) as conn:
        _init_schema(conn)
        row = conn.execute(
            """
            SELECT experiment_id, name, status, created_at, updated_at, metadata_json
            FROM experiments
            WHERE experiment_id = ?
            """,
            (safe_experiment_id,),
        ).fetchone()

    if row is None:
        return None

    return StoreExperimentRecord(
        experiment_id=str(row["experiment_id"]),
        name=str(row["name"]),
        status=str(row["status"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        metadata_json=str(row["metadata_json"]),
    )


def list_experiments(
    *,
    store_root: str | Path = ".numereng",
    status: str | None = None,
) -> tuple[StoreExperimentRecord, ...]:
    """List experiment metadata rows ordered by newest update first."""

    init_result = init_store_db(store_root=store_root)

    sql = """
        SELECT experiment_id, name, status, created_at, updated_at, metadata_json
        FROM experiments
    """
    params: tuple[str, ...] = ()
    if status is not None:
        sql += " WHERE status = ?"
        params = (status,)
    sql += " ORDER BY updated_at DESC"

    with _connect_rw(init_result.db_path) as conn:
        _init_schema(conn)
        rows = conn.execute(sql, params).fetchall()

    return tuple(
        StoreExperimentRecord(
            experiment_id=str(row["experiment_id"]),
            name=str(row["name"]),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            metadata_json=str(row["metadata_json"]),
        )
        for row in rows
    )


def upsert_hpo_study(
    *,
    store_root: str | Path = ".numereng",
    study: StoreHpoStudyUpsert,
) -> None:
    """Insert or update one HPO study row."""

    safe_study_id = _ensure_safe_study_id(study.study_id)
    safe_experiment_id = _ensure_safe_experiment_id(study.experiment_id) if study.experiment_id else None
    if not study.study_name.strip():
        raise StoreError("store_hpo_study_name_invalid")
    if not study.status.strip():
        raise StoreError("store_hpo_study_status_invalid")
    if not study.metric.strip():
        raise StoreError("store_hpo_study_metric_invalid")
    if study.direction not in {"maximize", "minimize"}:
        raise StoreError("store_hpo_study_direction_invalid")
    if study.n_trials < 1:
        raise StoreError("store_hpo_study_n_trials_invalid")
    if not study.sampler.strip():
        raise StoreError("store_hpo_study_sampler_invalid")
    safe_best_run_id = _ensure_safe_run_id(study.best_run_id) if study.best_run_id else None

    init_result = init_store_db(store_root=store_root)
    now = _utc_now_iso()
    with _connect_rw(init_result.db_path) as conn:
        _init_schema(conn)
        conn.execute(
            """
            INSERT INTO hpo_studies (
                study_id,
                experiment_id,
                study_name,
                status,
                metric,
                direction,
                n_trials,
                sampler,
                seed,
                best_trial_number,
                best_value,
                best_run_id,
                config_json,
                storage_path,
                error_message,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(study_id) DO UPDATE SET
                experiment_id = excluded.experiment_id,
                study_name = excluded.study_name,
                status = excluded.status,
                metric = excluded.metric,
                direction = excluded.direction,
                n_trials = excluded.n_trials,
                sampler = excluded.sampler,
                seed = excluded.seed,
                best_trial_number = excluded.best_trial_number,
                best_value = excluded.best_value,
                best_run_id = excluded.best_run_id,
                config_json = excluded.config_json,
                storage_path = excluded.storage_path,
                error_message = excluded.error_message,
                updated_at = excluded.updated_at
            """,
            (
                safe_study_id,
                safe_experiment_id,
                study.study_name,
                study.status,
                study.metric,
                study.direction,
                study.n_trials,
                study.sampler,
                study.seed,
                study.best_trial_number,
                study.best_value,
                safe_best_run_id,
                study.config_json,
                study.storage_path,
                study.error_message,
                now,
                now,
            ),
        )
        conn.commit()


def get_hpo_study(
    *,
    store_root: str | Path = ".numereng",
    study_id: str,
) -> StoreHpoStudyRecord | None:
    """Load one HPO study row from store DB."""

    safe_study_id = _ensure_safe_study_id(study_id)
    init_result = init_store_db(store_root=store_root)

    with _connect_rw(init_result.db_path) as conn:
        _init_schema(conn)
        row = conn.execute(
            """
            SELECT
                study_id,
                experiment_id,
                study_name,
                status,
                metric,
                direction,
                n_trials,
                sampler,
                seed,
                best_trial_number,
                best_value,
                best_run_id,
                config_json,
                storage_path,
                error_message,
                created_at,
                updated_at
            FROM hpo_studies
            WHERE study_id = ?
            """,
            (safe_study_id,),
        ).fetchone()

    if row is None:
        return None

    return StoreHpoStudyRecord(
        study_id=str(row["study_id"]),
        experiment_id=_as_nonempty_str(row["experiment_id"]),
        study_name=str(row["study_name"]),
        status=str(row["status"]),
        metric=str(row["metric"]),
        direction=str(row["direction"]),
        n_trials=int(row["n_trials"]),
        sampler=str(row["sampler"]),
        seed=int(row["seed"]) if row["seed"] is not None else None,
        best_trial_number=int(row["best_trial_number"]) if row["best_trial_number"] is not None else None,
        best_value=float(row["best_value"]) if row["best_value"] is not None else None,
        best_run_id=_as_nonempty_str(row["best_run_id"]),
        config_json=_as_nonempty_str(row["config_json"]),
        storage_path=_as_nonempty_str(row["storage_path"]),
        error_message=_as_nonempty_str(row["error_message"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def list_hpo_studies(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[StoreHpoStudyRecord, ...]:
    """List HPO studies ordered by newest update first."""

    safe_experiment_id = _ensure_safe_experiment_id(experiment_id) if experiment_id else None
    if limit < 1:
        raise StoreError("store_hpo_list_limit_invalid")
    if offset < 0:
        raise StoreError("store_hpo_list_offset_invalid")

    init_result = init_store_db(store_root=store_root)
    clauses: list[str] = []
    params: list[Any] = []
    if safe_experiment_id is not None:
        clauses.append("experiment_id = ?")
        params.append(safe_experiment_id)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

    sql = (
        """
        SELECT
            study_id,
            experiment_id,
            study_name,
            status,
            metric,
            direction,
            n_trials,
            sampler,
            seed,
            best_trial_number,
            best_value,
            best_run_id,
            config_json,
            storage_path,
            error_message,
            created_at,
            updated_at
        FROM hpo_studies
        """
        + where
        + " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
    )

    with _connect_rw(init_result.db_path) as conn:
        _init_schema(conn)
        rows = conn.execute(sql, tuple(params + [limit, offset])).fetchall()

    return tuple(
        StoreHpoStudyRecord(
            study_id=str(row["study_id"]),
            experiment_id=_as_nonempty_str(row["experiment_id"]),
            study_name=str(row["study_name"]),
            status=str(row["status"]),
            metric=str(row["metric"]),
            direction=str(row["direction"]),
            n_trials=int(row["n_trials"]),
            sampler=str(row["sampler"]),
            seed=int(row["seed"]) if row["seed"] is not None else None,
            best_trial_number=int(row["best_trial_number"]) if row["best_trial_number"] is not None else None,
            best_value=float(row["best_value"]) if row["best_value"] is not None else None,
            best_run_id=_as_nonempty_str(row["best_run_id"]),
            config_json=_as_nonempty_str(row["config_json"]),
            storage_path=_as_nonempty_str(row["storage_path"]),
            error_message=_as_nonempty_str(row["error_message"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
        for row in rows
    )


def upsert_hpo_trial(
    *,
    store_root: str | Path = ".numereng",
    trial: StoreHpoTrialUpsert,
) -> None:
    """Insert or update one HPO trial row."""

    safe_study_id = _ensure_safe_study_id(trial.study_id)
    if trial.trial_number < 0:
        raise StoreError("store_hpo_trial_number_invalid")
    if not trial.status.strip():
        raise StoreError("store_hpo_trial_status_invalid")
    safe_run_id = _ensure_safe_run_id(trial.run_id) if trial.run_id else None

    init_result = init_store_db(store_root=store_root)
    updated_at = _utc_now_iso()
    with _connect_rw(init_result.db_path) as conn:
        _init_schema(conn)
        conn.execute(
            """
            INSERT INTO hpo_trials (
                study_id,
                trial_number,
                status,
                value,
                run_id,
                config_path,
                params_json,
                error_message,
                started_at,
                finished_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(study_id, trial_number) DO UPDATE SET
                status = excluded.status,
                value = excluded.value,
                run_id = excluded.run_id,
                config_path = excluded.config_path,
                params_json = excluded.params_json,
                error_message = excluded.error_message,
                started_at = excluded.started_at,
                finished_at = excluded.finished_at,
                updated_at = excluded.updated_at
            """,
            (
                safe_study_id,
                trial.trial_number,
                trial.status,
                trial.value,
                safe_run_id,
                trial.config_path,
                trial.params_json,
                trial.error_message,
                trial.started_at,
                trial.finished_at,
                updated_at,
            ),
        )
        conn.commit()


def list_hpo_trials(
    *,
    store_root: str | Path = ".numereng",
    study_id: str,
) -> tuple[StoreHpoTrialRecord, ...]:
    """List all trials for one study ordered by trial number."""

    safe_study_id = _ensure_safe_study_id(study_id)
    init_result = init_store_db(store_root=store_root)

    with _connect_rw(init_result.db_path) as conn:
        _init_schema(conn)
        rows = conn.execute(
            """
            SELECT
                study_id,
                trial_number,
                status,
                value,
                run_id,
                config_path,
                params_json,
                error_message,
                started_at,
                finished_at,
                updated_at
            FROM hpo_trials
            WHERE study_id = ?
            ORDER BY trial_number ASC
            """,
            (safe_study_id,),
        ).fetchall()

    return tuple(
        StoreHpoTrialRecord(
            study_id=str(row["study_id"]),
            trial_number=int(row["trial_number"]),
            status=str(row["status"]),
            value=float(row["value"]) if row["value"] is not None else None,
            run_id=_as_nonempty_str(row["run_id"]),
            config_path=_as_nonempty_str(row["config_path"]),
            params_json=_as_nonempty_str(row["params_json"]),
            error_message=_as_nonempty_str(row["error_message"]),
            started_at=_as_nonempty_str(row["started_at"]),
            finished_at=_as_nonempty_str(row["finished_at"]),
            updated_at=str(row["updated_at"]),
        )
        for row in rows
    )


def upsert_ensemble(
    *,
    store_root: str | Path = ".numereng",
    ensemble: StoreEnsembleUpsert,
) -> None:
    """Insert or update one ensemble row."""

    safe_ensemble_id = _ensure_safe_ensemble_id(ensemble.ensemble_id)
    safe_experiment_id = _ensure_safe_experiment_id(ensemble.experiment_id) if ensemble.experiment_id else None
    if not ensemble.name.strip():
        raise StoreError("store_ensemble_name_invalid")
    if not ensemble.method.strip():
        raise StoreError("store_ensemble_method_invalid")
    if not ensemble.target.strip():
        raise StoreError("store_ensemble_target_invalid")
    if not ensemble.metric.strip():
        raise StoreError("store_ensemble_metric_invalid")
    if not ensemble.status.strip():
        raise StoreError("store_ensemble_status_invalid")

    init_result = init_store_db(store_root=store_root)
    now = _utc_now_iso()
    with _connect_rw(init_result.db_path) as conn:
        _init_schema(conn)
        conn.execute(
            """
            INSERT INTO ensembles (
                ensemble_id,
                experiment_id,
                name,
                method,
                target,
                metric,
                status,
                config_json,
                artifacts_path,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ensemble_id) DO UPDATE SET
                experiment_id = excluded.experiment_id,
                name = excluded.name,
                method = excluded.method,
                target = excluded.target,
                metric = excluded.metric,
                status = excluded.status,
                config_json = excluded.config_json,
                artifacts_path = excluded.artifacts_path,
                updated_at = excluded.updated_at
            """,
            (
                safe_ensemble_id,
                safe_experiment_id,
                ensemble.name,
                ensemble.method,
                ensemble.target,
                ensemble.metric,
                ensemble.status,
                ensemble.config_json,
                ensemble.artifacts_path,
                now,
                now,
            ),
        )
        conn.commit()


def get_ensemble(
    *,
    store_root: str | Path = ".numereng",
    ensemble_id: str,
) -> StoreEnsembleRecord | None:
    """Load one ensemble row by ID."""

    safe_ensemble_id = _ensure_safe_ensemble_id(ensemble_id)
    init_result = init_store_db(store_root=store_root)

    with _connect_rw(init_result.db_path) as conn:
        _init_schema(conn)
        row = conn.execute(
            """
            SELECT
                ensemble_id,
                experiment_id,
                name,
                method,
                target,
                metric,
                status,
                config_json,
                artifacts_path,
                created_at,
                updated_at
            FROM ensembles
            WHERE ensemble_id = ?
            """,
            (safe_ensemble_id,),
        ).fetchone()

    if row is None:
        return None

    return StoreEnsembleRecord(
        ensemble_id=str(row["ensemble_id"]),
        experiment_id=_as_nonempty_str(row["experiment_id"]),
        name=str(row["name"]),
        method=str(row["method"]),
        target=str(row["target"]),
        metric=str(row["metric"]),
        status=str(row["status"]),
        config_json=_as_nonempty_str(row["config_json"]),
        artifacts_path=_as_nonempty_str(row["artifacts_path"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def list_ensembles(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[StoreEnsembleRecord, ...]:
    """List ensembles ordered by newest creation time."""

    safe_experiment_id = _ensure_safe_experiment_id(experiment_id) if experiment_id else None
    if limit < 1:
        raise StoreError("store_ensemble_list_limit_invalid")
    if offset < 0:
        raise StoreError("store_ensemble_list_offset_invalid")

    init_result = init_store_db(store_root=store_root)
    clauses: list[str] = []
    params: list[Any] = []
    if safe_experiment_id is not None:
        clauses.append("experiment_id = ?")
        params.append(safe_experiment_id)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = (
        """
        SELECT
            ensemble_id,
            experiment_id,
            name,
            method,
            target,
            metric,
            status,
            config_json,
            artifacts_path,
            created_at,
            updated_at
        FROM ensembles
        """
        + where
        + " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    )

    with _connect_rw(init_result.db_path) as conn:
        _init_schema(conn)
        rows = conn.execute(sql, tuple(params + [limit, offset])).fetchall()

    return tuple(
        StoreEnsembleRecord(
            ensemble_id=str(row["ensemble_id"]),
            experiment_id=_as_nonempty_str(row["experiment_id"]),
            name=str(row["name"]),
            method=str(row["method"]),
            target=str(row["target"]),
            metric=str(row["metric"]),
            status=str(row["status"]),
            config_json=_as_nonempty_str(row["config_json"]),
            artifacts_path=_as_nonempty_str(row["artifacts_path"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
        for row in rows
    )


def replace_ensemble_components(
    *,
    store_root: str | Path = ".numereng",
    ensemble_id: str,
    components: tuple[StoreEnsembleComponentUpsert, ...],
) -> None:
    """Replace component rows for one ensemble."""

    safe_ensemble_id = _ensure_safe_ensemble_id(ensemble_id)
    init_result = init_store_db(store_root=store_root)

    with _connect_rw(init_result.db_path) as conn:
        _init_schema(conn)
        conn.execute("DELETE FROM ensemble_components WHERE ensemble_id = ?", (safe_ensemble_id,))
        if components:
            conn.executemany(
                """
                INSERT INTO ensemble_components (
                    ensemble_id,
                    run_id,
                    weight,
                    rank
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(ensemble_id, run_id) DO UPDATE SET
                    weight = excluded.weight,
                    rank = excluded.rank
                """,
                [
                    (
                        safe_ensemble_id,
                        _ensure_safe_run_id(component.run_id),
                        float(component.weight),
                        int(component.rank),
                    )
                    for component in components
                ],
            )
        conn.commit()


def list_ensemble_components(
    *,
    store_root: str | Path = ".numereng",
    ensemble_id: str,
) -> tuple[StoreEnsembleComponentRecord, ...]:
    """List components for one ensemble ordered by rank."""

    safe_ensemble_id = _ensure_safe_ensemble_id(ensemble_id)
    init_result = init_store_db(store_root=store_root)

    with _connect_rw(init_result.db_path) as conn:
        _init_schema(conn)
        rows = conn.execute(
            """
            SELECT ensemble_id, run_id, weight, rank
            FROM ensemble_components
            WHERE ensemble_id = ?
            ORDER BY rank ASC
            """,
            (safe_ensemble_id,),
        ).fetchall()

    return tuple(
        StoreEnsembleComponentRecord(
            ensemble_id=str(row["ensemble_id"]),
            run_id=str(row["run_id"]),
            weight=float(row["weight"]),
            rank=int(row["rank"]),
        )
        for row in rows
    )


def replace_ensemble_metrics(
    *,
    store_root: str | Path = ".numereng",
    ensemble_id: str,
    metrics: tuple[StoreEnsembleMetricUpsert, ...],
) -> None:
    """Replace metric rows for one ensemble."""

    safe_ensemble_id = _ensure_safe_ensemble_id(ensemble_id)
    init_result = init_store_db(store_root=store_root)

    with _connect_rw(init_result.db_path) as conn:
        _init_schema(conn)
        conn.execute("DELETE FROM ensemble_metrics WHERE ensemble_id = ?", (safe_ensemble_id,))
        if metrics:
            conn.executemany(
                """
                INSERT INTO ensemble_metrics (
                    ensemble_id,
                    name,
                    value
                ) VALUES (?, ?, ?)
                ON CONFLICT(ensemble_id, name) DO UPDATE SET
                    value = excluded.value
                """,
                [
                    (
                        safe_ensemble_id,
                        metric.name,
                        metric.value,
                    )
                    for metric in metrics
                ],
            )
        conn.commit()


def list_ensemble_metrics(
    *,
    store_root: str | Path = ".numereng",
    ensemble_id: str,
) -> tuple[StoreEnsembleMetricRecord, ...]:
    """List metric rows for one ensemble."""

    safe_ensemble_id = _ensure_safe_ensemble_id(ensemble_id)
    init_result = init_store_db(store_root=store_root)

    with _connect_rw(init_result.db_path) as conn:
        _init_schema(conn)
        rows = conn.execute(
            """
            SELECT ensemble_id, name, value
            FROM ensemble_metrics
            WHERE ensemble_id = ?
            ORDER BY name ASC
            """,
            (safe_ensemble_id,),
        ).fetchall()

    return tuple(
        StoreEnsembleMetricRecord(
            ensemble_id=str(row["ensemble_id"]),
            name=str(row["name"]),
            value=float(row["value"]) if row["value"] is not None else None,
        )
        for row in rows
    )


def _index_run_with_connection(conn: sqlite3.Connection, *, store_root: Path, run_id: str) -> StoreIndexResult:
    run_dir = (store_root / "runs" / run_id).resolve()
    if not run_dir.is_dir():
        raise StoreRunNotFoundError(f"store_run_not_found:{run_id}")

    manifest_path = run_dir / "run.json"
    if not manifest_path.is_file():
        raise StoreRunManifestNotFoundError(f"store_run_manifest_not_found:{run_id}")

    manifest = _load_manifest(manifest_path=manifest_path, run_id=run_id)
    metrics_payload = _load_metrics_payload(run_dir)

    warnings: list[str] = []
    manifest_run_id = manifest.get("run_id")
    if isinstance(manifest_run_id, str) and manifest_run_id and manifest_run_id != run_id:
        warnings.append(f"manifest_run_id_mismatch:{manifest_run_id}")

    status = _as_nonempty_str(manifest.get("status")) or "UNKNOWN"
    run_hash = _as_nonempty_str(manifest.get("run_hash")) or ""
    run_type = _as_nonempty_str(manifest.get("run_type")) or "training"
    created_at = _as_nonempty_str(manifest.get("created_at")) or _iso_from_ts(manifest_path.stat().st_mtime)
    finished_at = _as_nonempty_str(manifest.get("finished_at"))
    config_hash = _extract_config_hash(manifest)
    experiment_id = _extract_experiment_id(manifest)
    manifest_json = _canonical_json(manifest)
    manifest_mtime_ns = manifest_path.stat().st_mtime_ns
    ingested_at = _utc_now_iso()

    metric_rows = _build_metric_rows(metrics_payload)
    artifact_rows = _build_artifact_rows(run_dir=run_dir, manifest=manifest)

    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
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
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                run_hash = excluded.run_hash,
                status = excluded.status,
                run_type = excluded.run_type,
                created_at = excluded.created_at,
                finished_at = excluded.finished_at,
                config_hash = excluded.config_hash,
                experiment_id = excluded.experiment_id,
                run_path = excluded.run_path,
                manifest_json = excluded.manifest_json,
                manifest_mtime_ns = excluded.manifest_mtime_ns,
                ingested_at = excluded.ingested_at
            """,
            (
                run_id,
                run_hash,
                status,
                run_type,
                created_at,
                finished_at,
                config_hash,
                experiment_id,
                str(run_dir),
                manifest_json,
                manifest_mtime_ns,
                ingested_at,
            ),
        )

        conn.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
        if metric_rows:
            conn.executemany(
                """
                INSERT INTO metrics (run_id, name, value, value_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id, name) DO UPDATE SET
                    value = excluded.value,
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                [
                    (
                        run_id,
                        metric_name,
                        metric_value,
                        metric_json,
                        ingested_at,
                    )
                    for metric_name, metric_value, metric_json in metric_rows
                ],
            )

        conn.execute("DELETE FROM run_artifacts WHERE run_id = ?", (run_id,))
        if artifact_rows:
            conn.executemany(
                """
                INSERT INTO run_artifacts (
                    run_id,
                    kind,
                    relative_path,
                    absolute_path,
                    exists_flag,
                    size_bytes,
                    sha256,
                    mtime_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, kind, relative_path) DO UPDATE SET
                    absolute_path = excluded.absolute_path,
                    exists_flag = excluded.exists_flag,
                    size_bytes = excluded.size_bytes,
                    sha256 = excluded.sha256,
                    mtime_ns = excluded.mtime_ns
                """,
                [
                    (
                        run_id,
                        kind,
                        relative_path,
                        absolute_path,
                        exists_flag,
                        size_bytes,
                        sha256,
                        mtime_ns,
                    )
                    for kind, relative_path, absolute_path, exists_flag, size_bytes, sha256, mtime_ns in artifact_rows
                ],
            )

        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return StoreIndexResult(
        run_id=run_id,
        status=status,
        metrics_indexed=len(metric_rows),
        artifacts_indexed=len(artifact_rows),
        run_path=run_dir,
        warnings=tuple(warnings),
    )


def _build_metric_rows(metrics_payload: dict[str, Any]) -> list[tuple[str, float | None, str | None]]:
    collected: dict[str, tuple[float | None, str | None]] = {}

    for key, value in metrics_payload.items():
        _collect_metric_rows(collected, metric_name=str(key), value=value)

    return [(name, payload[0], payload[1]) for name, payload in sorted(collected.items())]


def _collect_metric_rows(
    collected: dict[str, tuple[float | None, str | None]],
    *,
    metric_name: str,
    value: Any,
) -> None:
    normalized_name = metric_name.strip()
    if not normalized_name:
        return

    scalar = _coerce_finite_float(value)
    if scalar is not None:
        collected[normalized_name] = (scalar, None)
        return

    if value is None:
        collected[normalized_name] = (None, None)
        return

    if isinstance(value, dict):
        collected[normalized_name] = (None, _safe_json_dumps(value))
        for child_key, child_value in value.items():
            child_name = f"{normalized_name}.{child_key}"
            _collect_metric_rows(collected, metric_name=child_name, value=child_value)
        return

    collected[normalized_name] = (None, _safe_json_dumps(value))


def _build_artifact_rows(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
) -> list[tuple[str, str, str, int, int | None, str | None, int | None]]:
    candidates: list[tuple[str, str]] = list(_CANONICAL_ARTIFACTS)

    artifacts_payload = manifest.get("artifacts")
    if isinstance(artifacts_payload, dict):
        for kind, reference in artifacts_payload.items():
            if not isinstance(reference, str):
                continue
            if not reference.strip():
                continue
            candidates.append((str(kind), reference))

    predictions_dir = run_dir / "artifacts" / "predictions"
    if predictions_dir.is_dir():
        for artifact in sorted(predictions_dir.iterdir(), key=lambda path: path.name):
            if artifact.is_file() and artifact.suffix.lower() in {".parquet", ".csv"}:
                try:
                    relative = artifact.resolve().relative_to(run_dir.resolve()).as_posix()
                except ValueError:
                    relative = artifact.name
                candidates.append(("predictions_discovered", relative))

    deduped: dict[tuple[str, str], tuple[str, str, str, int, int | None, str | None, int | None]] = {}
    for kind, reference in candidates:
        resolved = _resolve_artifact_reference(run_dir=run_dir, reference=reference)
        if resolved is None:
            continue

        relative_path, absolute_path = resolved
        exists_flag = 1 if absolute_path.is_file() else 0

        size_bytes: int | None = None
        mtime_ns: int | None = None
        sha256: str | None = None
        if exists_flag:
            try:
                stat = absolute_path.stat()
            except OSError:
                stat = None
            if stat is not None:
                size_bytes = stat.st_size
                mtime_ns = stat.st_mtime_ns
                sha256 = _sha256_file(absolute_path)

        deduped[(kind, relative_path)] = (
            kind,
            relative_path,
            str(absolute_path),
            exists_flag,
            size_bytes,
            sha256,
            mtime_ns,
        )

    return list(deduped.values())


def _resolve_artifact_reference(*, run_dir: Path, reference: str) -> tuple[str, Path] | None:
    raw = reference.strip()
    if not raw:
        return None

    run_root = run_dir.resolve()
    try:
        candidate = Path(raw).expanduser()
    except Exception:
        return None

    if candidate.is_absolute():
        absolute_path = candidate.resolve()
    else:
        absolute_path = (run_root / candidate).resolve()

    if absolute_path.is_dir():
        return None

    try:
        relative = absolute_path.relative_to(run_root).as_posix()
    except ValueError:
        return None

    return relative, absolute_path


def _load_manifest(*, manifest_path: Path, run_id: str) -> dict[str, Any]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise StoreRunManifestInvalidError(f"store_run_manifest_read_failed:{run_id}") from exc
    except json.JSONDecodeError as exc:
        raise StoreRunManifestInvalidError(f"store_run_manifest_invalid_json:{run_id}") from exc

    if not isinstance(payload, dict):
        raise StoreRunManifestInvalidError(f"store_run_manifest_not_object:{run_id}")

    return payload


def _load_metrics_payload(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        return {}

    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    return payload if isinstance(payload, dict) else {}


def _extract_config_hash(manifest: dict[str, Any]) -> str | None:
    config_hash = manifest.get("config_hash")
    if isinstance(config_hash, str) and config_hash:
        return config_hash

    config_payload = manifest.get("config")
    if isinstance(config_payload, dict):
        nested_hash = config_payload.get("hash")
        if isinstance(nested_hash, str) and nested_hash:
            return nested_hash

    return None


def _extract_experiment_id(manifest: dict[str, Any]) -> str | None:
    experiment_id = manifest.get("experiment_id")
    if isinstance(experiment_id, str) and experiment_id:
        return experiment_id

    lineage = manifest.get("lineage")
    if isinstance(lineage, dict):
        lineage_experiment_id = lineage.get("experiment_id")
        if isinstance(lineage_experiment_id, str) and lineage_experiment_id:
            return lineage_experiment_id

    return None


def _connect_rw(db_path: Path) -> sqlite3.Connection:
    return _connect_sqlite(
        db_path=db_path,
        read_only=False,
        timeout_seconds=5.0,
        busy_timeout_ms=5000,
        operation="connect_rw",
    )


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    return _connect_sqlite(
        db_path=db_path,
        read_only=True,
        timeout_seconds=2.0,
        busy_timeout_ms=2000,
        operation="connect_read_only",
    )


def _connect_sqlite(
    *,
    db_path: Path,
    read_only: bool,
    timeout_seconds: float,
    busy_timeout_ms: int,
    operation: str,
) -> sqlite3.Connection:
    recovery_attempted = False
    quarantined_sidecars: tuple[Path, ...] = ()

    while True:
        try:
            if read_only:
                conn = sqlite3.connect(
                    f"file:{db_path}?mode=ro",
                    uri=True,
                    check_same_thread=False,
                    timeout=timeout_seconds,
                )
            else:
                conn = sqlite3.connect(
                    db_path,
                    check_same_thread=False,
                    timeout=timeout_seconds,
                )
            conn.row_factory = sqlite3.Row
            conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms};")
            if not read_only:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=FULL;")
                conn.execute("PRAGMA wal_autocheckpoint=1000;")
            return conn
        except sqlite3.Error as exc:
            if not recovery_attempted and _should_quarantine_sidecars(db_path=db_path, exc=exc):
                recovery_attempted = True
                try:
                    quarantined_sidecars = _quarantine_sqlite_sidecars(db_path=db_path)
                except OSError as quarantine_exc:
                    raise StoreError(
                        f"store_db_sidecar_quarantine_failed:{db_path}:{quarantine_exc}"
                    ) from quarantine_exc
                continue
            raise _wrap_sqlite_error(
                operation=operation,
                db_path=db_path,
                exc=exc,
                quarantined_sidecars=quarantined_sidecars,
            ) from exc


def _should_quarantine_sidecars(*, db_path: Path, exc: sqlite3.Error) -> bool:
    if not _is_corruption_error(exc):
        return False
    return any(sidecar.exists() for sidecar in _sqlite_sidecar_paths(db_path))


def _sqlite_sidecar_paths(db_path: Path) -> tuple[Path, Path]:
    return (Path(f"{db_path}-wal"), Path(f"{db_path}-shm"))


def _quarantine_sqlite_sidecars(*, db_path: Path) -> tuple[Path, ...]:
    suffix = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    quarantined: list[Path] = []
    for sidecar in _sqlite_sidecar_paths(db_path):
        if not sidecar.exists():
            continue
        target = _sidecar_quarantine_target(sidecar=sidecar, suffix=suffix)
        sidecar.rename(target)
        quarantined.append(target)
    return tuple(quarantined)


def _sidecar_quarantine_target(*, sidecar: Path, suffix: str) -> Path:
    candidate = sidecar.with_name(f"{sidecar.name}.corrupt.{suffix}")
    counter = 1
    while candidate.exists():
        candidate = sidecar.with_name(f"{sidecar.name}.corrupt.{suffix}.{counter}")
        counter += 1
    return candidate


def _is_corruption_error(exc: sqlite3.Error) -> bool:
    error_text = str(exc).lower()
    return any(token in error_text for token in _SQLITE_CORRUPTION_TOKENS)


def _wrap_sqlite_error(
    *,
    operation: str,
    db_path: Path,
    exc: sqlite3.Error,
    quarantined_sidecars: tuple[Path, ...] = (),
) -> StoreError:
    kind = "corrupt" if _is_corruption_error(exc) else "sqlite"
    detail = str(exc).strip() or exc.__class__.__name__
    error_code = f"store_db_{kind}_error:{operation}:{db_path}:{detail}"
    if quarantined_sidecars:
        sidecars = ",".join(path.name for path in quarantined_sidecars)
        error_code = f"{error_code}:quarantined={sidecars}"
    return StoreError(error_code)


def _init_schema(conn: sqlite3.Connection) -> None:
    for statement in _SCHEMA_STATEMENTS:
        conn.execute(statement)

    _migrate_cloud_jobs_primary_key(conn)

    for statement in _INDEX_STATEMENTS:
        conn.execute(statement)

    for migration_name in _SCHEMA_MIGRATIONS:
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (name, applied_at) VALUES (?, ?)",
            (migration_name, _utc_now_iso()),
        )
    conn.commit()


def _migrate_cloud_jobs_primary_key(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "cloud_jobs"):
        return

    pk_columns = _cloud_jobs_primary_key_columns(conn)
    if pk_columns == ("run_id", "provider", "provider_job_id"):
        return

    conn.execute(
        """
        CREATE TABLE cloud_jobs_new (
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
            PRIMARY KEY (run_id, provider, provider_job_id)
        );
        """
    )
    conn.execute(
        """
        INSERT INTO cloud_jobs_new (
            run_id,
            provider,
            backend,
            provider_job_id,
            status,
            region,
            image_uri,
            output_s3_uri,
            metadata_json,
            error_message,
            started_at,
            finished_at,
            updated_at
        )
        SELECT
            run_id,
            provider,
            backend,
            provider_job_id,
            status,
            region,
            image_uri,
            output_s3_uri,
            metadata_json,
            error_message,
            started_at,
            finished_at,
            updated_at
        FROM cloud_jobs;
        """
    )
    conn.execute("DROP TABLE cloud_jobs;")
    conn.execute("ALTER TABLE cloud_jobs_new RENAME TO cloud_jobs;")


def _cloud_jobs_primary_key_columns(conn: sqlite3.Connection) -> tuple[str, ...]:
    rows = conn.execute("PRAGMA table_info(cloud_jobs)").fetchall()
    numbered: list[tuple[int, str]] = []
    for row in rows:
        raw_number = row["pk"] if isinstance(row, sqlite3.Row) else row[5]
        raw_name = row["name"] if isinstance(row, sqlite3.Row) else row[1]
        if not isinstance(raw_number, int) or raw_number <= 0:
            continue
        if not isinstance(raw_name, str):
            continue
        numbered.append((raw_number, raw_name))
    numbered.sort(key=lambda item: item[0])
    return tuple(name for _, name in numbered)


def _count_query(conn: sqlite3.Connection, sql: str) -> int:
    row = conn.execute(sql).fetchone()
    if row is None:
        return 0
    value = row[0]
    return int(value) if isinstance(value, (int, float)) else 0


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _build_primary_per_era_corr_frame(**kwargs: Any):
    from numereng.features.scoring.artifacts import build_primary_per_era_corr_frame

    return build_primary_per_era_corr_frame(**kwargs)


def _persist_primary_per_era_corr_artifacts(frame: Any, *, predictions_dir: Path):
    from numereng.features.scoring.artifacts import persist_primary_per_era_corr_artifacts

    return persist_primary_per_era_corr_artifacts(frame, predictions_dir=predictions_dir)


def _load_run_manifest_for_materialize(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "run.json"
    if not manifest_path.is_file():
        raise StoreError(f"store_run_manifest_missing:{run_dir.name}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StoreError(f"store_run_manifest_invalid:{run_dir.name}") from exc
    if not isinstance(payload, dict):
        raise StoreError(f"store_run_manifest_invalid:{run_dir.name}")
    return payload


def _materialize_run_per_era_corr(run_dir: Path) -> bool:
    run_id = _ensure_safe_run_id(run_dir.name)
    manifest = _load_run_manifest_for_materialize(run_dir)
    artifacts = manifest.get("artifacts")
    artifacts_map = dict(artifacts) if isinstance(artifacts, dict) else {}
    predictions_ref = artifacts_map.get("predictions")
    if not isinstance(predictions_ref, str) or not predictions_ref.strip():
        raise StoreError(f"store_viz_artifact_predictions_missing:{run_id}")

    predictions_path = (run_dir / predictions_ref).resolve()
    if not predictions_path.is_file():
        raise StoreError(f"store_viz_artifact_predictions_not_found:{run_id}")

    data = manifest.get("data")
    data_map = data if isinstance(data, dict) else {}
    target_col = _as_nonempty_str(data_map.get("target_col"))
    if target_col is None:
        raise StoreError(f"store_viz_artifact_target_missing:{run_id}")
    era_col = _as_nonempty_str(data_map.get("era_col")) or "era"

    predictions_dir = run_dir / "artifacts" / "predictions"
    parquet_path = predictions_dir / "val_per_era_corr20v2.parquet"
    csv_path = predictions_dir / "val_per_era_corr20v2.csv"
    if parquet_path.is_file() and csv_path.is_file():
        return False

    frame = _build_primary_per_era_corr_frame(
        predictions_path=predictions_path,
        target_col=target_col,
        era_col=era_col,
    )
    if frame is None or frame.empty:
        raise StoreError(f"store_viz_artifact_rows_missing:{run_id}")

    paths = _persist_primary_per_era_corr_artifacts(frame, predictions_dir=predictions_dir)
    artifacts_map["per_era_corr"] = str(paths.parquet_path.relative_to(run_dir))
    artifacts_map["per_era_corr_csv"] = str(paths.csv_path.relative_to(run_dir))
    manifest["artifacts"] = artifacts_map

    manifest_path = run_dir / "run.json"
    tmp_path = manifest_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    tmp_path.replace(manifest_path)

    try:
        index_run(store_root=run_dir.parent.parent, run_id=run_id)
    except StoreError as exc:
        raise StoreError(f"store_viz_artifact_index_failed:{run_id}") from exc
    return True


def _ensure_safe_run_id(run_id: str) -> str:
    if not run_id or not _SAFE_ID.match(run_id):
        raise StoreError(f"store_run_id_invalid:{run_id}")
    return run_id


def _ensure_safe_experiment_id(experiment_id: str) -> str:
    if not experiment_id or not _SAFE_ID.match(experiment_id):
        raise StoreError(f"store_experiment_id_invalid:{experiment_id}")
    return experiment_id


def _ensure_safe_study_id(study_id: str) -> str:
    if not study_id or not _SAFE_ID.match(study_id):
        raise StoreError(f"store_study_id_invalid:{study_id}")
    return study_id


def _ensure_safe_ensemble_id(ensemble_id: str) -> str:
    if not ensemble_id or not _SAFE_ID.match(ensemble_id):
        raise StoreError(f"store_ensemble_id_invalid:{ensemble_id}")
    return ensemble_id


def _as_nonempty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _coerce_finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=True)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _iso_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


__all__ = [
    "StoreCloudJobUpsert",
    "StoreDoctorResult",
    "StoreEnsembleComponentRecord",
    "StoreEnsembleComponentUpsert",
    "StoreEnsembleMetricRecord",
    "StoreEnsembleMetricUpsert",
    "StoreEnsembleRecord",
    "StoreEnsembleUpsert",
    "StoreError",
    "StoreExperimentRecord",
    "StoreHpoStudyRecord",
    "StoreHpoStudyUpsert",
    "StoreHpoTrialRecord",
    "StoreHpoTrialUpsert",
    "StoreIndexResult",
    "StoreInitResult",
    "StoreMaterializeVizArtifactsFailure",
    "StoreMaterializeVizArtifactsResult",
    "StoreRebuildFailure",
    "StoreRebuildResult",
    "StoreRunManifestInvalidError",
    "StoreRunManifestNotFoundError",
    "StoreRunNotFoundError",
    "doctor_store",
    "get_ensemble",
    "get_experiment",
    "get_hpo_study",
    "index_run",
    "init_store_db",
    "list_ensemble_components",
    "list_ensemble_metrics",
    "list_ensembles",
    "list_experiments",
    "list_hpo_studies",
    "list_hpo_trials",
    "materialize_viz_artifacts",
    "rebuild_run_index",
    "replace_ensemble_components",
    "replace_ensemble_metrics",
    "resolve_store_root",
    "upsert_ensemble",
    "upsert_experiment",
    "upsert_hpo_study",
    "upsert_hpo_trial",
    "upsert_cloud_job",
]
