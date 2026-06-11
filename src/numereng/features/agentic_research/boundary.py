"""Decision -> config materialization: reject-never-edit allowlist, caps, dedup, reuse guard."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from numereng.config.training import TrainingConfig, load_training_config_json
from numereng.features.agentic_research import memory
from numereng.features.agentic_research.types import (
    ALLOWED_PATHS_METADATA_KEY,
    RUN_PLAN_FIELDS,
    SCORING_STAGE,
    VALUE_CAPS_METADATA_KEY,
    AgenticResearchDuplicateCandidate,
    AgenticResearchValidationError,
    ResearchDecision,
)
from numereng.features.experiments import ExperimentRecord, ExperimentTrainResult
from numereng.features.training.errors import TrainingError
from numereng.features.training.run_store import compute_config_hash

# Mutable substrate (invariant 1). data.target_horizon is intentionally absent: it is
# derived from the target_col suffix and a mismatch is REJECTED (never controller-edited).
ALLOWED_CHANGE_PATHS = (
    "data.feature_set",
    "data.target_col",
    "data.scoring_targets",
    "preprocessing.nan_missing_all_twos",
    "preprocessing.missing_value",
    "model.type",
    "model.module_path",
    "model.device",
    "model.params.*",
    "model.x_groups",
    "model.data_needed",
    "model.target_transform.*",
    "training.engine.profile",
    "training.engine.window_size_eras",
    "training.engine.embargo_eras",
    "training.resources.parallel_folds",
    "training.resources.max_threads_per_worker",
    "output.predictions_name",
)
# Paths that reach into the frozen evaluator (invariant 2). No allowed-path override may
# touch the scoring protocol; an attempt is rejected at session init.
_SCORING_FROZEN_PREFIXES = ("scoring", "validation", "evaluation")
_TRAIN_RUN_DIR_NOT_FRESH_PREFIX = "training_run_dir_not_fresh:"


def materialize_config(
    *,
    experiment: ExperimentRecord,
    round_label: str,
    decision: ResearchDecision,
) -> Path:
    """Reject-never-edit: validate the decision against the parent, write the child config.

    The decision payload is never mutated. Out-of-cap, disallowed-path, horizon-mismatch,
    schema-invalid, or duplicate-with-recorded-run cases raise a stable token. An orphan
    config (hash exists, no recorded run) is overwritten and run."""
    config_dir = experiment.manifest_path.parent / "configs"
    parent_path = config_dir / str(decision.parent_config)
    if not parent_path.is_file():
        raise AgenticResearchValidationError(f"agentic_research_parent_config_not_found:{decision.parent_config}")
    payload = load_training_config_json(parent_path)
    allowed_paths = program_allowed_paths(experiment)
    value_caps = program_value_caps(experiment)
    for change in decision.changes:
        if not _matches_any_path(change.path, allowed_paths):
            raise AgenticResearchValidationError(f"agentic_research_change_path_not_allowed:{change.path}")
        if change.path in value_caps and change.value is not None:
            if isinstance(change.value, bool) or not isinstance(change.value, (int, float)):
                raise AgenticResearchValidationError(f"agentic_research_change_value_not_numeric:{change.path}")
            lo, hi = value_caps[change.path]
            if not (lo <= float(change.value) <= hi):
                raise AgenticResearchValidationError(f"agentic_research_change_value_out_of_cap:{change.path}")
    for change in decision.changes:
        _assign_dotted(payload, change.path.split("."), deepcopy(change.value))
    _assert_horizon_matches_target(payload)
    try:
        validated = TrainingConfig.model_validate(payload).model_dump(mode="python", exclude_none=True)
    except Exception as exc:
        raise AgenticResearchValidationError(f"agentic_research_config_schema_invalid:{exc}") from exc
    candidate_hash = compute_config_hash(validated)
    existing = existing_config_hashes(config_dir)
    if candidate_hash in existing:
        colliding = existing[candidate_hash]
        if config_hash_has_recorded_run(experiment, colliding):
            raise AgenticResearchDuplicateCandidate(f"agentic_research_candidate_duplicate:{candidate_hash[:12]}")
        # Crash orphan: a config with this hash exists but no recorded run. Adopt the
        # round's own filename (overwriting any prior orphan there) and run it.
    path = config_dir / _round_config_filename(round_label)
    memory.write_json(path, validated)
    return path


def baseline_config(experiment: ExperimentRecord, round_label: str) -> Path:
    """Copy the seed config verbatim for a baseline round."""
    config_dir = experiment.manifest_path.parent / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    parent_path = _first_config_path(experiment)
    payload = load_training_config_json(parent_path)
    path = config_dir / _round_config_filename(round_label)
    memory.write_json(path, payload)
    return path


def reuse_finished_run_on_hash_collision(
    *, root: Path, experiment: ExperimentRecord, exc: TrainingError, index_run: Callable[..., object]
) -> ExperimentTrainResult | None:
    """Reuse an existing FINISHED run dir only when it is owned by this experiment.

    Same-experiment retries reuse the existing run as an idempotent recovery path;
    a cross-experiment collision hard-fails with a stable token (stale evidence guard)."""
    msg = str(exc)
    if not msg.startswith(_TRAIN_RUN_DIR_NOT_FRESH_PREFIX):
        return None
    parts = msg.split(":")
    if len(parts) < 3:
        return None
    run_id = parts[1]
    run_dir = root / "runs" / run_id
    run_json_path = run_dir / "run.json"
    if not run_json_path.is_file():
        return None
    try:
        run_manifest = json.loads(run_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if run_manifest.get("status") != "FINISHED":
        return None
    source_experiment_id = run_manifest.get("experiment_id")
    if source_experiment_id != experiment.experiment_id:
        raise AgenticResearchValidationError(
            "agentic_research_stale_run_reuse_blocked:"
            f"{run_id}:source_experiment={source_experiment_id or 'unknown'}:"
            f"current_experiment={experiment.experiment_id}"
        )
    predictions_name = None
    output_block = run_manifest.get("output")
    if isinstance(output_block, dict):
        predictions_name = output_block.get("predictions_name")
    if not isinstance(predictions_name, str) or not predictions_name:
        predictions_name = "predictions"
    predictions_path = run_dir / "artifacts" / "predictions" / f"{predictions_name}.parquet"
    link_reused_run_to_experiment(experiment=experiment, run_id=run_id)
    try:
        index_run(store_root=root, run_id=run_id)
    except Exception:
        pass
    return ExperimentTrainResult(
        experiment_id=experiment.experiment_id,
        run_id=run_id,
        predictions_path=predictions_path,
        results_path=run_dir / "results.json",
    )


def link_reused_run_to_experiment(*, experiment: ExperimentRecord, run_id: str) -> None:
    """Append run_id to the experiment manifest's runs list if not already present."""
    manifest_path = experiment.manifest_path
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        runs = []
    if run_id in runs:
        return
    runs.append(run_id)
    manifest["runs"] = runs
    if manifest.get("status") == "draft":
        manifest["status"] = "active"
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    memory.write_json(manifest_path, manifest)


def record_round_config_in_run_plan(*, experiment: ExperimentRecord, round_label: str, config_path: Path) -> None:
    """Add a run_plan.csv row pinning the round's config and frozen scoring stage."""
    path = experiment.manifest_path.parent / "run_plan.csv"
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = [{key: value or "" for key, value in row.items()} for row in reader]
    for field in RUN_PLAN_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    config_stem = config_path.stem
    for row in rows:
        if row.get("round", "").strip() == round_label and Path(row.get("config_path", "")).stem == config_stem:
            return
    rows.append(
        {
            **{field: "" for field in fieldnames},
            "plan_index": str(_next_run_plan_index(rows)),
            "round": round_label,
            "config_path": _relative_to_experiment(experiment, config_path),
            "score_stage_default": SCORING_STAGE,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def assert_scoring_paths_frozen(experiment: ExperimentRecord) -> None:
    """Reject an allowed-path override that reaches into the frozen evaluator (invariant 2)."""
    raw = experiment.metadata.get(ALLOWED_PATHS_METADATA_KEY)
    if not isinstance(raw, list):
        return
    for item in raw:
        if not isinstance(item, str):
            continue
        head = item.strip().split(".")[0]
        if head in _SCORING_FROZEN_PREFIXES:
            raise AgenticResearchValidationError(f"agentic_research_scoring_path_frozen:{item.strip()}")


def program_allowed_paths(experiment: ExperimentRecord) -> tuple[str, ...]:
    """Resolve the allowlist, narrowing to a metadata override if present and valid."""
    raw = experiment.metadata.get(ALLOWED_PATHS_METADATA_KEY)
    if not isinstance(raw, list):
        return ALLOWED_CHANGE_PATHS
    narrowed: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        candidate = item.strip()
        if candidate and _matches_any_path(candidate, ALLOWED_CHANGE_PATHS):
            narrowed.append(candidate)
    return tuple(narrowed) if narrowed else ALLOWED_CHANGE_PATHS


def program_value_caps(experiment: ExperimentRecord) -> dict[str, tuple[float, float]]:
    """Parse flat value caps from experiment metadata."""
    raw = experiment.metadata.get(VALUE_CAPS_METADATA_KEY)
    if not isinstance(raw, dict):
        return {}
    caps: dict[str, tuple[float, float]] = {}
    for key, bounds in raw.items():
        if not isinstance(key, str) or not key.strip():
            continue
        if not isinstance(bounds, list) or len(bounds) != 2:
            continue
        lo, hi = bounds
        if isinstance(lo, bool) or isinstance(hi, bool):
            continue
        if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
            continue
        caps[key.strip()] = (float(lo), float(hi))
    return caps


def existing_config_hashes(config_dir: Path) -> dict[str, str]:
    """Map each on-disk config hash to its filename (latest wins on collision)."""
    hashes: dict[str, str] = {}
    for path in sorted(config_dir.glob("*.json")):
        try:
            payload = load_training_config_json(path)
        except Exception:
            continue
        hashes[compute_config_hash(payload)] = path.name
    return hashes


def config_hash_has_recorded_run(experiment: ExperimentRecord, config_name: str) -> bool:
    """Dedup-vs-orphan: True only if the journal recorded a run for this config file."""
    return memory.journal_has_recorded_run(experiment, config_name)


def _assert_horizon_matches_target(payload: dict[str, object]) -> None:
    data = payload.get("data")
    if not isinstance(data, dict):
        return
    horizon = data.get("target_horizon")
    target = data.get("target_col")
    if horizon is None or not isinstance(target, str):
        return
    expected = "20d" if target.endswith("_20") else "60d" if target.endswith("_60") else None
    if expected is not None and horizon != expected:
        raise AgenticResearchValidationError(
            f"agentic_research_horizon_target_mismatch:{target}:{horizon}:expected_{expected}"
        )


def _assign_dotted(payload: dict[str, object], parts: list[str], value: object) -> None:
    cursor: dict[str, object] = payload
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            raise AgenticResearchValidationError(f"agentic_research_change_target_not_mapping:{'.'.join(parts)}")
        cursor = cast(dict[str, object], child)
    cursor[parts[-1]] = value


def _matches_any_path(path: str, allowed: tuple[str, ...]) -> bool:
    for entry in allowed:
        if entry.endswith(".*") and path.startswith(entry[:-1]):
            return True
        if path == entry:
            return True
    return False


def _round_config_filename(round_label: str) -> str:
    suffix = round_label.removeprefix("r")
    return f"config_{suffix}.json" if suffix.isdigit() else f"{round_label}_config.json"


def _relative_to_experiment(experiment: ExperimentRecord, path: Path) -> str:
    try:
        return str(path.relative_to(experiment.manifest_path.parent))
    except ValueError:
        return str(path)


def _next_run_plan_index(rows: list[dict[str, str]]) -> int:
    indexes: list[int] = []
    for row in rows:
        try:
            indexes.append(int(row.get("plan_index", "")))
        except ValueError:
            continue
    return max(indexes, default=len(rows)) + 1


def _first_config_path(experiment: ExperimentRecord) -> Path:
    config_dir = experiment.manifest_path.parent / "configs"
    configs = sorted(config_dir.glob("*.json"))
    if not configs:
        raise AgenticResearchValidationError(f"agentic_research_config_missing:{experiment.experiment_id}")
    return configs[0]
