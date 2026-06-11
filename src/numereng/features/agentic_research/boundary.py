"""Decision-to-config boundary: allowlist, caps, dedup, and reuse guard."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from numereng.config.training import TrainingConfig, load_training_config_json
from numereng.features.agentic_research import memory
from numereng.features.agentic_research import types as ar_types
from numereng.features.experiments import ExperimentRecord, ExperimentTrainResult
from numereng.features.training.errors import TrainingError
from numereng.features.training.run_store import compute_config_hash

ALLOWED_CHANGE_PATHS = tuple(
    "data.feature_set data.target_col data.scoring_targets "
    "preprocessing.nan_missing_all_twos preprocessing.missing_value "
    "model.type model.module_path model.device model.params.* model.x_groups "
    "model.data_needed model.target_transform.* "
    "training.engine.profile training.engine.window_size_eras training.engine.embargo_eras "
    "training.resources.parallel_folds training.resources.max_threads_per_worker output.predictions_name".split()
)
_SCORING_FROZEN_PREFIXES = ("scoring", "validation", "evaluation")
_TRAIN_RUN_DIR_NOT_FRESH_PREFIX = "training_run_dir_not_fresh:"


def materialize_config(*, experiment: ExperimentRecord, round_label: str, decision: ar_types.ResearchDecision) -> Path:
    config_dir = experiment.manifest_path.parent / "configs"
    parent_path = config_dir / str(decision.parent_config)
    if not parent_path.is_file():
        raise ar_types.AgenticResearchValidationError(
            f"agentic_research_parent_config_not_found:{decision.parent_config}"
        )
    payload = load_training_config_json(parent_path)
    allowed_paths = program_allowed_paths(experiment)
    value_caps = program_value_caps(experiment)
    for change in decision.changes:
        if not _matches_any_path(change.path, allowed_paths):
            raise ar_types.AgenticResearchValidationError(f"agentic_research_change_path_not_allowed:{change.path}")
        if change.path in value_caps and change.value is not None:
            if isinstance(change.value, bool) or not isinstance(change.value, (int, float)):
                raise ar_types.AgenticResearchValidationError(
                    f"agentic_research_change_value_not_numeric:{change.path}"
                )
            lo, hi = value_caps[change.path]
            if not lo <= float(change.value) <= hi:
                raise ar_types.AgenticResearchValidationError(f"agentic_research_change_value_out_of_cap:{change.path}")
        _assign_dotted(payload, change.path.split("."), deepcopy(change.value))
    _assert_horizon_matches_target(payload)
    try:
        validated = TrainingConfig.model_validate(payload).model_dump(mode="python", exclude_none=True)
    except Exception as exc:
        raise ar_types.AgenticResearchValidationError(f"agentic_research_config_schema_invalid:{exc}") from exc
    candidate_hash = compute_config_hash(validated)
    existing = existing_config_hashes(config_dir)
    if candidate_hash in existing and config_hash_has_recorded_run(experiment, existing[candidate_hash]):
        raise ar_types.AgenticResearchDuplicateCandidate(f"agentic_research_candidate_duplicate:{candidate_hash[:12]}")
    path = config_dir / _round_config_filename(round_label)
    memory.write_json(path, validated)
    return path


def baseline_config(experiment: ExperimentRecord, round_label: str) -> Path:
    config_dir = experiment.manifest_path.parent / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    configs = sorted(config_dir.glob("*.json"))
    if not configs:
        raise ar_types.AgenticResearchValidationError(f"agentic_research_config_missing:{experiment.experiment_id}")
    path = config_dir / _round_config_filename(round_label)
    memory.write_json(path, load_training_config_json(configs[0]))
    return path


def reuse_finished_run_on_hash_collision(
    *, root: Path, experiment: ExperimentRecord, exc: TrainingError, index_run: Callable[..., object]
) -> ExperimentTrainResult | None:
    msg = str(exc)
    if not msg.startswith(_TRAIN_RUN_DIR_NOT_FRESH_PREFIX):
        return None
    parts = msg.split(":")
    if len(parts) < 3:
        return None
    run_id = parts[1]
    run_dir = root / "runs" / run_id
    try:
        run_manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if run_manifest.get("status") != "FINISHED":
        return None
    source_experiment_id = run_manifest.get("experiment_id")
    if source_experiment_id != experiment.experiment_id:
        raise ar_types.AgenticResearchValidationError(
            "agentic_research_stale_run_reuse_blocked:"
            f"{run_id}:source_experiment={source_experiment_id or 'unknown'}:"
            f"current_experiment={experiment.experiment_id}"
        )
    output = run_manifest.get("output")
    predictions_name = output.get("predictions_name") if isinstance(output, dict) else None
    if not isinstance(predictions_name, str) or not predictions_name:
        predictions_name = "predictions"
    link_reused_run_to_experiment(experiment=experiment, run_id=run_id)
    try:
        index_run(store_root=root, run_id=run_id)
    except Exception:
        pass
    return ExperimentTrainResult(
        experiment_id=experiment.experiment_id,
        run_id=run_id,
        predictions_path=run_dir / "artifacts" / "predictions" / f"{predictions_name}.parquet",
        results_path=run_dir / "results.json",
    )


def link_reused_run_to_experiment(*, experiment: ExperimentRecord, run_id: str) -> None:
    try:
        manifest = json.loads(experiment.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    runs = manifest.get("runs") if isinstance(manifest.get("runs"), list) else []
    if run_id in runs:
        return
    runs.append(run_id)
    manifest["runs"] = runs
    if manifest.get("status") == "draft":
        manifest["status"] = "active"
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    memory.write_json(experiment.manifest_path, manifest)


def record_round_config_in_run_plan(*, experiment: ExperimentRecord, round_label: str, config_path: Path) -> None:
    path = experiment.manifest_path.parent / "run_plan.csv"
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = [{key: value or "" for key, value in row.items()} for row in reader]
    fieldnames.extend(field for field in ar_types.RUN_PLAN_FIELDS if field not in fieldnames)
    if any(
        row.get("round", "").strip() == round_label and Path(row.get("config_path", "")).stem == config_path.stem
        for row in rows
    ):
        return
    rows.append(
        {
            **{field: "" for field in fieldnames},
            "plan_index": str(
                max(
                    (int(row.get("plan_index", "")) for row in rows if row.get("plan_index", "").isdigit()),
                    default=len(rows),
                )
                + 1
            ),
            "round": round_label,
            "config_path": _relative_to_experiment(experiment, config_path),
            "score_stage_default": ar_types.SCORING_STAGE,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def assert_scoring_paths_frozen(experiment: ExperimentRecord) -> None:
    raw = experiment.metadata.get(ar_types.ALLOWED_PATHS_METADATA_KEY)
    if not isinstance(raw, list):
        return
    for item in raw:
        if isinstance(item, str) and item.strip().split(".")[0] in _SCORING_FROZEN_PREFIXES:
            raise ar_types.AgenticResearchValidationError(f"agentic_research_scoring_path_frozen:{item.strip()}")


def program_allowed_paths(experiment: ExperimentRecord) -> tuple[str, ...]:
    raw = experiment.metadata.get(ar_types.ALLOWED_PATHS_METADATA_KEY)
    if not isinstance(raw, list):
        return ALLOWED_CHANGE_PATHS
    narrowed = tuple(
        item.strip()
        for item in raw
        if isinstance(item, str) and item.strip() and _matches_any_path(item.strip(), ALLOWED_CHANGE_PATHS)
    )
    return narrowed or ALLOWED_CHANGE_PATHS


def program_value_caps(experiment: ExperimentRecord) -> dict[str, tuple[float, float]]:
    raw = experiment.metadata.get(ar_types.VALUE_CAPS_METADATA_KEY)
    if not isinstance(raw, dict):
        return {}
    caps: dict[str, tuple[float, float]] = {}
    for key, bounds in raw.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(bounds, list) or len(bounds) != 2:
            continue
        lo, hi = bounds
        if (
            not isinstance(lo, bool)
            and not isinstance(hi, bool)
            and isinstance(lo, (int, float))
            and isinstance(hi, (int, float))
        ):
            caps[key.strip()] = (float(lo), float(hi))
    return caps


def existing_config_hashes(config_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(config_dir.glob("*.json")):
        try:
            hashes[compute_config_hash(load_training_config_json(path))] = path.name
        except Exception:
            continue
    return hashes


def config_hash_has_recorded_run(experiment: ExperimentRecord, config_name: str) -> bool:
    return memory.journal_has_recorded_run(experiment, config_name)


def _assert_horizon_matches_target(payload: dict[str, object]) -> None:
    data = payload.get("data")
    if not isinstance(data, dict):
        return
    horizon, target = data.get("target_horizon"), data.get("target_col")
    if horizon is None or not isinstance(target, str):
        return
    expected = "20d" if target.endswith("_20") else "60d" if target.endswith("_60") else None
    if expected is not None and horizon != expected:
        raise ar_types.AgenticResearchValidationError(
            f"agentic_research_horizon_target_mismatch:{target}:{horizon}:expected_{expected}"
        )


def _assign_dotted(payload: dict[str, object], parts: list[str], value: object) -> None:
    cursor = payload
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            raise ar_types.AgenticResearchValidationError(
                f"agentic_research_change_target_not_mapping:{'.'.join(parts)}"
            )
        cursor = child
    cursor[parts[-1]] = value


def _matches_any_path(path: str, allowed: tuple[str, ...]) -> bool:
    return any(path == entry or entry.endswith(".*") and path.startswith(entry[:-1]) for entry in allowed)


def _round_config_filename(round_label: str) -> str:
    suffix = round_label.removeprefix("r")
    return f"config_{suffix}.json" if suffix.isdigit() else f"{round_label}_config.json"


def _relative_to_experiment(experiment: ExperimentRecord, path: Path) -> str:
    try:
        return str(path.relative_to(experiment.manifest_path.parent))
    except ValueError:
        return str(path)
