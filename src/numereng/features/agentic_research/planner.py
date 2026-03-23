"""Planning helpers for the agentic research supervisor."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Final

from numereng.config.training import TrainingConfig
from numereng.features.agentic_research.contracts import (
    CodexConfigPayload,
    CodexDecision,
    ResearchBestRun,
    ResearchPathState,
    ResearchPhaseState,
    ResearchProgramState,
    ResearchRoundState,
)
from numereng.features.agentic_research.state import utc_now_iso
from numereng.features.agentic_research.strategy import (
    ResearchStrategyDefinition,
    get_phase_definition,
    next_phase_definition,
)
from numereng.features.experiments import ExperimentRecord, ExperimentReport, ExperimentReportRow

_FILENAME_SLUG_RE = re.compile(r"[^a-z0-9]+")
_WILDCARD_SENTINEL: Final[str] = "*"
_ALLOWED_OVERRIDE_PATHS: Final[tuple[str, ...]] = (
    "data.feature_set",
    "data.target_col",
    "data.scoring_targets",
    "data.target_horizon",
    "data.loading.era_chunk_size",
    "data.loading.include_feature_neutral_metrics",
    "preprocessing.nan_missing_all_twos",
    "preprocessing.missing_value",
    "model.type",
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
    "output.results_name",
)
_FALLBACK_BASE_CONFIG: Final[dict[str, object]] = {
    "data": {
        "data_version": "v5.2",
        "dataset_variant": "non_downsampled",
        "dataset_scope": "train_plus_validation",
        "feature_set": "small",
        "target_col": "target_ender_20",
        "target_horizon": "20d",
        "era_col": "era",
        "id_col": "id",
        "benchmark_source": {
            "source": "active",
            "predictions_path": None,
            "pred_col": "prediction",
            "name": None,
        },
        "meta_model_data_path": None,
        "meta_model_col": "numerai_meta_model",
        "embargo_eras": None,
        "baselines_dir": None,
        "loading": {
            "mode": "materialized",
            "scoring_mode": "materialized",
            "era_chunk_size": 64,
            "include_feature_neutral_metrics": True,
        },
    },
    "preprocessing": {
        "nan_missing_all_twos": False,
        "missing_value": 2.0,
    },
    "model": {
        "type": "LGBMRegressor",
        "params": {
            "n_estimators": 10,
            "learning_rate": 0.01,
            "max_depth": 6,
            "num_leaves": 64,
            "colsample_bytree": 0.1,
            "random_state": 42,
        },
        "x_groups": ["features"],
        "data_needed": None,
        "module_path": None,
        "target_transform": None,
        "benchmark": None,
        "baseline": None,
    },
    "training": {
        "engine": {
            "profile": "purged_walk_forward",
            "mode": None,
            "window_size_eras": None,
            "embargo_eras": None,
        },
        "resources": {
            "parallel_folds": 1,
            "parallel_backend": "joblib",
            "memmap_enabled": True,
            "max_threads_per_worker": "default",
            "sklearn_working_memory_mib": None,
        },
        "cache": {
            "mode": "deterministic",
            "cache_fold_specs": True,
            "cache_features": True,
            "cache_labels": True,
            "cache_fold_matrices": False,
        },
    },
    "output": {
        "output_dir": None,
        "baselines_dir": None,
        "predictions_name": "val_predictions_scout",
        "results_name": None,
    },
}


def next_round_label(round_number: int) -> str:
    """Return the canonical round label for one numeric round."""
    return f"r{round_number}"


def build_round_state(
    *,
    round_number: int,
    experiment_id: str,
    path_id: str,
    phase_id: str | None = None,
) -> ResearchRoundState:
    """Create one fresh round state."""
    now = utc_now_iso()
    return ResearchRoundState(
        round_number=round_number,
        round_label=next_round_label(round_number),
        experiment_id=experiment_id,
        path_id=path_id,
        status="planning",
        next_config_index=0,
        phase_id=phase_id,
        started_at=now,
        updated_at=now,
    )


def render_prompt(
    *,
    program: ResearchProgramState,
    active_path: ResearchPathState,
    experiment: ExperimentRecord,
    report: ExperimentReport | None,
    recent_round_summaries: list[dict[str, object]],
    forced_action: str | None,
    strategy: ResearchStrategyDefinition,
    current_phase: ResearchPhaseState | None,
    base_config_snapshot: dict[str, object],
    base_config_source: str,
    valid_config_examples: list[dict[str, object]],
) -> str:
    """Render the Codex planning prompt from compact structured context."""
    template = strategy.prompt_path.read_text(encoding="utf-8")
    report_rows: list[dict[str, object]] = []
    if report is not None:
        report_rows = [_row_to_dict(row) for row in report.rows[:10]]
    current_phase_id = current_phase.phase_id if current_phase is not None else None
    current_phase_definition = get_phase_definition(strategy, current_phase_id)
    next_phase = next_phase_definition(strategy, current_phase_id)
    context = {
        "strategy": {
            "id": strategy.strategy_id,
            "title": strategy.title,
            "description": strategy.description,
            "phase_aware": strategy.phase_aware,
        },
        "root_experiment_id": program.root_experiment_id,
        "active_experiment_id": program.active_experiment_id,
        "active_path_id": program.active_path_id,
        "active_path_hypothesis": active_path.hypothesis,
        "active_path_generation": active_path.generation,
        "active_path_plateau_streak": active_path.plateau_streak,
        "active_path_scale_confirmation_used": active_path.scale_confirmation_used,
        "active_path_needs_scale_confirmation": active_path.needs_scale_confirmation,
        "improvement_threshold": program.improvement_threshold,
        "next_round_label": next_round_label(program.next_round_number),
        "forced_action": forced_action or "none",
        "root_experiment_name": experiment.name,
        "root_experiment_hypothesis": experiment.hypothesis,
        "best_overall": _best_to_dict(program.best_overall),
        "metric_policy": {
            "primary": "bmc_last_200_eras.mean",
            "tie_break": "bmc.mean",
            "sanity_checks": ["corr.mean", "mmc.mean", "cwmm.mean"],
        },
        "current_report_metric": report.metric if report is not None else "bmc_last_200_eras.mean",
        "current_report_rows": report_rows,
        "recent_round_summaries": recent_round_summaries[-3:],
        "current_phase": _phase_to_dict(current_phase),
        "current_phase_summary": (
            {
                "phase_id": current_phase_definition.phase_id,
                "title": current_phase_definition.title,
                "summary": current_phase_definition.summary,
                "gate": current_phase_definition.gate,
            }
            if current_phase_definition is not None
            else None
        ),
        "next_phase_summary": (
            {
                "phase_id": next_phase.phase_id,
                "title": next_phase.title,
                "summary": next_phase.summary,
                "gate": next_phase.gate,
            }
            if next_phase is not None
            else None
        ),
        "phase_sequence": [
            {
                "phase_id": phase.phase_id,
                "title": phase.title,
                "summary": phase.summary,
                "gate": phase.gate,
            }
            for phase in strategy.phases
        ],
        "base_config_source": base_config_source,
        "base_config_snapshot": base_config_snapshot,
        "allowed_override_paths": list(_ALLOWED_OVERRIDE_PATHS),
        "config_filename_rules": {
            "agent_must_provide": "short slug filename ending in .json",
            "python_will_add_round_prefix": True,
            "python_will_normalize_slug": True,
        },
        "valid_config_examples": [_prompt_ready_example(item) for item in valid_config_examples],
    }
    return template.replace("$CONTEXT_JSON", json.dumps(context, indent=2, sort_keys=True))


def validate_decision(
    decision: CodexDecision,
    *,
    strategy: ResearchStrategyDefinition,
    current_phase: ResearchPhaseState | None,
) -> None:
    """Validate one Codex planning decision against supervisor rules."""
    if decision.next_action not in {"continue", "scale", "pivot", "stop"}:
        raise ValueError("agentic_research_codex_action_invalid")
    if strategy.phase_aware:
        if decision.phase_action not in {"stay", "advance", "complete"}:
            raise ValueError("agentic_research_codex_phase_action_invalid")
        if decision.phase_action in {"advance", "complete"} and not (
            decision.phase_transition_rationale and decision.phase_transition_rationale.strip()
        ):
            raise ValueError("agentic_research_codex_phase_transition_rationale_missing")
        if (
            decision.phase_action == "advance"
            and next_phase_definition(
                strategy,
                current_phase.phase_id if current_phase is not None else None,
            )
            is None
        ):
            raise ValueError("agentic_research_codex_phase_advance_invalid")
        if decision.phase_action == "complete" and decision.next_action != "stop":
            raise ValueError("agentic_research_codex_phase_complete_requires_stop")
        if decision.next_action == "stop" and decision.phase_action not in {"complete", "stay"}:
            raise ValueError("agentic_research_codex_phase_stop_invalid")
    elif decision.phase_action is not None:
        raise ValueError("agentic_research_codex_phase_action_unexpected")
    if decision.next_action == "stop":
        if decision.configs:
            raise ValueError("agentic_research_codex_stop_with_configs_invalid")
        return
    if len(decision.configs) < 4 or len(decision.configs) > 5:
        raise ValueError("agentic_research_codex_config_count_invalid")
    seen: set[str] = set()
    for item in decision.configs:
        if not item.filename.strip():
            raise ValueError("agentic_research_codex_config_filename_missing")
        if item.filename in seen:
            raise ValueError("agentic_research_codex_config_filename_duplicate")
        seen.add(item.filename)
        if not isinstance(item.overrides, dict):
            raise ValueError("agentic_research_codex_config_overrides_invalid")


def materialize_configs(
    *,
    round_label: str,
    config_dir: Path,
    base_config: dict[str, object],
    configs: list[CodexConfigPayload],
) -> list[str]:
    """Validate and write Codex-generated configs into the experiment config dir."""
    config_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for item in configs:
        filename = normalize_config_filename(round_label=round_label, filename=item.filename)
        materialized_payload = materialize_config_payload(base_config=base_config, overrides=item.overrides)
        validated = TrainingConfig.model_validate(materialized_payload)
        path = config_dir / filename
        rendered = json.dumps(
            validated.model_dump(mode="python", exclude_none=True),
            indent=2,
            sort_keys=True,
        )
        path.write_text(rendered, encoding="utf-8")
        written.append(filename)
    return written


def materialize_config_payload(
    *,
    base_config: dict[str, object],
    overrides: dict[str, object],
) -> dict[str, object]:
    """Apply one override payload onto a validated base config snapshot."""
    payload = deepcopy(base_config)
    if not isinstance(payload, dict):
        raise ValueError("agentic_research_base_config_invalid")
    _apply_overrides(target=payload, overrides=overrides, path=())
    validated = TrainingConfig.model_validate(payload)
    return validated.model_dump(mode="python", exclude_none=True)


def normalize_config_filename(*, round_label: str, filename: str) -> str:
    """Normalize one Codex-proposed filename into the canonical experiment format."""
    stem = Path(filename).stem.lower()
    slug = _FILENAME_SLUG_RE.sub("-", stem).strip("-") or "config"
    if not slug.startswith(f"{round_label}-") and not slug.startswith(f"{round_label}_"):
        slug = f"{round_label}-{slug}"
    slug = slug.replace("-", "_", 1)
    return f"{slug}.json"


def select_reference_configs(*, config_dirs: list[Path]) -> tuple[dict[str, object], str, list[dict[str, object]]]:
    """Load the authoritative base config snapshot and compact valid examples."""
    valid_snapshots: list[tuple[str, dict[str, object]]] = []
    for config_dir in config_dirs:
        if not config_dir.is_dir():
            continue
        for config_path in sorted(config_dir.glob("*.json")):
            try:
                parsed = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(parsed, dict):
                continue
            try:
                validated = TrainingConfig.model_validate(parsed)
            except Exception:
                continue
            valid_snapshots.append(
                (
                    str(config_path),
                    validated.model_dump(mode="python", exclude_none=True),
                )
            )
            if len(valid_snapshots) >= 2:
                break
        if len(valid_snapshots) >= 2:
            break
    if not valid_snapshots:
        fallback = TrainingConfig.model_validate(_FALLBACK_BASE_CONFIG).model_dump(mode="python", exclude_none=True)
        example_overrides = _fallback_example_overrides(fallback)
        variant = materialize_config_payload(base_config=fallback, overrides=example_overrides)
        return (
            fallback,
            "built_in_fallback",
            [
                {
                    "filename": "reference_base.json",
                    "rationale": "Fallback validated base config.",
                    "overrides": {},
                    "materialized_config": fallback,
                },
                {
                    "filename": "reference_variant.json",
                    "rationale": "Fallback single-variable variant.",
                    "overrides": example_overrides,
                    "materialized_config": variant,
                },
            ],
        )
    base_source, base_snapshot = valid_snapshots[0]
    examples = [
        {
            "filename": Path(path).name,
            "rationale": "Validated reference config already present in the experiment lineage.",
            "overrides": {},
            "materialized_config": snapshot,
        }
        for path, snapshot in valid_snapshots
    ]
    if len(examples) == 1:
        example_overrides = _fallback_example_overrides(base_snapshot)
        examples.append(
            {
                "filename": "reference_variant.json",
                "rationale": "Validated synthetic single-variable variant built from the base snapshot.",
                "overrides": example_overrides,
                "materialized_config": materialize_config_payload(
                    base_config=base_snapshot,
                    overrides=example_overrides,
                ),
            }
        )
    return base_snapshot, base_source, examples[:2]


def round_summary_from_report(
    *,
    report: ExperimentReport,
    run_ids: list[str],
    round_state: ResearchRoundState,
) -> dict[str, object]:
    """Build a compact persisted round summary from the experiment report."""
    rows = [_row_to_dict(row) for row in report.rows if row.run_id in set(run_ids)]
    return {
        "round_label": round_state.round_label,
        "experiment_id": round_state.experiment_id,
        "path_id": round_state.path_id,
        "decision_action": round_state.decision_action,
        "experiment_question": round_state.experiment_question,
        "winner_criteria": round_state.winner_criteria,
        "decision_rationale": round_state.decision_rationale,
        "phase_id": round_state.phase_id,
        "phase_action": round_state.phase_action,
        "phase_transition_rationale": round_state.phase_transition_rationale,
        "run_ids": list(run_ids),
        "rows": rows,
        "best_row": rows[0] if rows else None,
    }


def select_best_row(rows: list[ExperimentReportRow]) -> ExperimentReportRow | None:
    """Select the best row using the primary metric and tie-break metric."""
    best: ExperimentReportRow | None = None
    for row in rows:
        if row.bmc_last_200_eras_mean is None:
            continue
        if best is None:
            best = row
            continue
        if row.bmc_last_200_eras_mean > (best.bmc_last_200_eras_mean or float("-inf")):
            best = row
            continue
        if row.bmc_last_200_eras_mean == best.bmc_last_200_eras_mean and (row.bmc_mean or float("-inf")) > (
            best.bmc_mean or float("-inf")
        ):
            best = row
    return best


def update_best_overall(
    *,
    current_best: ResearchBestRun,
    round_best: ExperimentReportRow | None,
    experiment_id: str,
    threshold: float,
) -> tuple[ResearchBestRun, bool]:
    """Update the program-level best run tracker and return whether it improved."""
    if round_best is None or round_best.bmc_last_200_eras_mean is None:
        return current_best, False
    prior_value = current_best.bmc_last_200_eras_mean
    improved = prior_value is None or (round_best.bmc_last_200_eras_mean - prior_value) >= threshold
    if not improved:
        return current_best, False
    return (
        ResearchBestRun(
            experiment_id=experiment_id,
            run_id=round_best.run_id,
            bmc_last_200_eras_mean=round_best.bmc_last_200_eras_mean,
            bmc_mean=round_best.bmc_mean,
            corr_mean=round_best.corr_mean,
            mmc_mean=round_best.mmc_mean,
            cwmm_mean=round_best.cwmm_mean,
            updated_at=utc_now_iso(),
        ),
        True,
    )


def update_path_after_round(
    *,
    path: ResearchPathState,
    round_best: ExperimentReportRow | None,
    improved: bool,
) -> ResearchPathState:
    """Return the updated path state after one completed round."""
    now = utc_now_iso()
    if improved:
        return replace(
            path,
            rounds_completed=path.rounds_completed + 1,
            plateau_streak=0,
            needs_scale_confirmation=False,
            best_run_id=round_best.run_id if round_best is not None else path.best_run_id,
            updated_at=now,
        )
    new_streak = path.plateau_streak + 1
    needs_scale_confirmation = path.needs_scale_confirmation
    if new_streak >= 2 and not path.scale_confirmation_used:
        needs_scale_confirmation = True
    return replace(
        path,
        rounds_completed=path.rounds_completed + 1,
        plateau_streak=new_streak,
        needs_scale_confirmation=needs_scale_confirmation,
        updated_at=now,
    )


def force_action_for_path(path: ResearchPathState) -> str | None:
    """Return the forced next action implied by path plateau rules."""
    if path.needs_scale_confirmation:
        return "scale"
    return None


def should_pivot_path(path: ResearchPathState) -> bool:
    """Return whether the next planning step should pivot into a new child experiment."""
    return path.plateau_streak >= 2 and path.scale_confirmation_used and not path.needs_scale_confirmation


def child_experiment_id(*, root_experiment_id: str, generation: int, path_slug: str) -> str:
    """Build the canonical child experiment id for one new path."""
    slug = normalize_path_slug(path_slug)
    return f"{root_experiment_id}--p{generation:02d}-{slug}"


def normalize_path_slug(value: str) -> str:
    """Normalize one Codex-proposed path slug to the experiment id contract."""
    lowered = value.strip().lower()
    normalized = _FILENAME_SLUG_RE.sub("-", lowered).strip("-")
    return normalized or "new-path"


def _apply_overrides(
    *,
    target: dict[str, object],
    overrides: dict[str, object],
    path: tuple[str, ...],
) -> None:
    for key, value in overrides.items():
        if not isinstance(key, str) or not key:
            raise ValueError("agentic_research_override_key_invalid")
        candidate_path = (*path, key)
        if isinstance(value, dict):
            existing = target.get(key)
            if existing is None:
                existing = {}
            if not isinstance(existing, dict):
                raise ValueError(f"agentic_research_override_target_not_mapping:{'.'.join(candidate_path)}")
            target[key] = deepcopy(existing)
            _apply_overrides(target=target[key], overrides=value, path=candidate_path)
            continue
        dotted = ".".join(candidate_path)
        if not _path_allowed(candidate_path):
            raise ValueError(f"agentic_research_override_path_not_allowed:{dotted}")
        target[key] = value


def _path_allowed(path: tuple[str, ...]) -> bool:
    for allowed in _ALLOWED_OVERRIDE_PATHS:
        allowed_tokens = tuple(allowed.split("."))
        if len(path) != len(allowed_tokens):
            continue
        matches = all(
            token == _WILDCARD_SENTINEL or token == current for token, current in zip(allowed_tokens, path, strict=True)
        )
        if matches:
            return True
    return False


def _fallback_example_overrides(base_config: dict[str, object]) -> dict[str, object]:
    learning_rate = base_config.get("model", {}) if isinstance(base_config.get("model"), dict) else {}
    params = learning_rate.get("params", {}) if isinstance(learning_rate, dict) else {}
    value = params.get("learning_rate")
    if isinstance(value, (int, float)) and value > 0:
        return {"model": {"params": {"learning_rate": round(float(value) * 0.5, 6)}}}
    return {"data": {"feature_set": "medium"}}


def _prompt_ready_example(example: dict[str, object]) -> dict[str, object]:
    prepared = dict(example)
    overrides = prepared.pop("overrides", {})
    prepared["override_items"] = _override_items(overrides if isinstance(overrides, dict) else {})
    return prepared


def _override_items(overrides: dict[str, object]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    _collect_override_items(items=items, overrides=overrides, path=())
    return items


def _collect_override_items(
    *,
    items: list[dict[str, str]],
    overrides: dict[str, object],
    path: tuple[str, ...],
) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict):
            _collect_override_items(items=items, overrides=value, path=(*path, key))
            continue
        items.append(
            {
                "path": ".".join((*path, key)),
                "value_json": json.dumps(value, sort_keys=True),
            }
        )


def _row_to_dict(row: ExperimentReportRow) -> dict[str, object]:
    return {
        "run_id": row.run_id,
        "metric_value": row.metric_value,
        "corr_mean": row.corr_mean,
        "mmc_mean": row.mmc_mean,
        "cwmm_mean": row.cwmm_mean,
        "bmc_mean": row.bmc_mean,
        "bmc_last_200_eras_mean": row.bmc_last_200_eras_mean,
        "is_champion": row.is_champion,
    }


def _best_to_dict(best: ResearchBestRun) -> dict[str, object]:
    return {
        "experiment_id": best.experiment_id,
        "run_id": best.run_id,
        "bmc_last_200_eras_mean": best.bmc_last_200_eras_mean,
        "bmc_mean": best.bmc_mean,
        "corr_mean": best.corr_mean,
        "mmc_mean": best.mmc_mean,
        "cwmm_mean": best.cwmm_mean,
        "updated_at": best.updated_at,
    }


def _phase_to_dict(phase: ResearchPhaseState | None) -> dict[str, object] | None:
    if phase is None:
        return None
    return {
        "phase_id": phase.phase_id,
        "phase_title": phase.phase_title,
        "status": phase.status,
        "round_count": phase.round_count,
        "transition_rationale": phase.transition_rationale,
        "started_at": phase.started_at,
        "updated_at": phase.updated_at,
    }
