"""Planning and path-management helpers for agentic research."""

from __future__ import annotations

import json
import re
import tempfile
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Final

from numereng.config.training import TrainingConfig
from numereng.features.agentic_research.utils.llm import (
    AgenticResearchCodexError,
    AgenticResearchOpenRouterError,
    run_codex_planner,
    run_codex_raw_planner,
    run_openrouter_planner,
    run_openrouter_raw_planner,
)
from numereng.features.agentic_research.utils.mutation import (
    MaterializedMutationConfig,
    ParentConfigSelection,
    materialize_mutation_config,
    parse_mutation_response,
    render_mutation_prompt,
    select_parent_config,
    write_materialized_config,
)
from numereng.features.agentic_research.utils.programs import (
    get_phase_definition,
    next_phase_definition,
    render_prompt_template,
    render_validation_feedback_block,
)
from numereng.features.agentic_research.utils.store import (
    append_planner_trace,
    planner_trace_payload,
    recent_round_summaries,
    save_round_bundle,
    session_program_path,
    upsert_agentic_research_metadata,
    utc_now_iso,
)
from numereng.features.agentic_research.utils.types import (
    CodexConfigPayload,
    CodexDecision,
    CodexPlannerExecution,
    MutationPlannerExecution,
    RawPlannerExecution,
    ResearchBestRun,
    ResearchLineageLink,
    ResearchLineageState,
    ResearchPathState,
    ResearchPhaseState,
    ResearchProgramDefinition,
    ResearchProgramRoundPolicy,
    ResearchProgramState,
    ResearchRoundState,
)
from numereng.features.experiments import (
    ExperimentError,
    ExperimentRecord,
    ExperimentReport,
    ExperimentReportRow,
    create_experiment,
    get_experiment,
    report_experiment,
)
from numereng.platform.clients.openrouter import active_model_source

_FILENAME_SLUG_RE = re.compile(r"[^a-z0-9]+")
_WILDCARD_SENTINEL: Final[str] = "*"
_DEFAULT_ROUND_POLICY = ResearchProgramRoundPolicy(
    plateau_non_improving_rounds=2,
    require_scale_confirmation=True,
    scale_confirmation_rounds=1,
)
_FALLBACK_ALLOWED_PATHS = (
    "data.feature_set",
    "data.target_col",
    "data.scoring_targets",
    "data.target_horizon",
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
_REPORT_METRIC_FIELDS = {
    "bmc_last_200_eras.mean": "bmc_last_200_eras_mean",
    "bmc.mean": "bmc_mean",
    "corr.mean": "corr_mean",
    "mmc.mean": "mmc_mean",
    "cwmm.mean": "cwmm_mean",
}
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


def pivot_to_child_path(
    *,
    root: Path,
    state: ResearchProgramState,
    lineage: ResearchLineageState,
    active_path: ResearchPathState,
    pivot_reason: str | None = None,
    child_hypothesis: str | None = None,
    child_slug: str | None = None,
    source_round: str | None = None,
) -> tuple[ResearchProgramState, ResearchLineageState]:
    generation = state.total_paths_created
    reason = pivot_reason or "plateau"
    hypothesis = child_hypothesis or active_path.hypothesis
    slug = child_slug or child_hypothesis or f"path-{generation}"
    path_id = f"p{generation:02d}"
    now = utc_now_iso()
    child_id = child_experiment_id(
        root_experiment_id=state.root_experiment_id,
        generation=generation,
        path_slug=slug,
    )
    child = create_experiment(
        store_root=root,
        experiment_id=child_id,
        name=f"Agentic Research Path {generation}: {slug}",
        hypothesis=hypothesis,
        tags=["agentic-research", f"path:{generation:02d}"],
    )
    upsert_agentic_research_metadata(
        root=root,
        experiment=child,
        metadata_update={
            "root_experiment_id": state.root_experiment_id,
            "program_experiment_id": state.program_experiment_id,
            "program_id": state.program_id,
            "program_title": state.program_title,
            "program_source": state.program_source,
            "parent_experiment_id": active_path.experiment_id,
            "path_id": path_id,
            "pivot_reason": reason,
            "source_round": source_round,
            "generation": generation,
        },
    )
    retired_path = replace(
        active_path,
        status="pivoted",
        pivot_reason=reason,
        updated_at=now,
    )
    child_path = ResearchPathState(
        path_id=path_id,
        experiment_id=child.experiment_id,
        parent_experiment_id=active_path.experiment_id,
        generation=generation,
        hypothesis=hypothesis,
        status="active",
        pivot_reason=reason,
        source_round=source_round,
        rounds_completed=0,
        plateau_streak=0,
        scale_confirmation_used=False,
        needs_scale_confirmation=False,
        best_run_id=None,
        created_at=now,
        updated_at=now,
    )
    new_paths = [retired_path if item.path_id == retired_path.path_id else item for item in state.paths]
    new_paths.append(child_path)
    updated_state = replace(
        state,
        active_path_id=child_path.path_id,
        active_experiment_id=child.experiment_id,
        total_paths_created=state.total_paths_created + 1,
        paths=new_paths,
        status="running",
        current_round=None,
        last_checkpoint="path_pivoted",
        updated_at=now,
    )
    updated_lineage = replace(
        lineage,
        active_path_id=child_path.path_id,
        links=[
            *lineage.links,
            ResearchLineageLink(
                path_id=child_path.path_id,
                experiment_id=child.experiment_id,
                parent_experiment_id=active_path.experiment_id,
                generation=generation,
                source_round=source_round,
                pivot_reason=reason,
                created_at=now,
            ),
        ],
    )
    return updated_state, updated_lineage


def _stop_failed_planning_round(
    *,
    auto_dir: Path,
    round_artifact_dir: Path,
    round_state: Any,
    program_state: ResearchProgramState,
    session_source_path: Path,
    prompt_text: str,
    executions: list[Any],
    error: str,
    lineage: ResearchLineageState,
) -> tuple[ResearchProgramState, ResearchLineageState, str]:
    planner_payload = _persist_planner_trace(
        round_state=round_state,
        program_state=program_state,
        round_artifact_dir=round_artifact_dir,
        session_source_path=session_source_path,
        prompt_text=prompt_text,
        executions=executions,
        status="failed",
        error=error,
    )
    save_round_bundle(
        round_artifact_dir=round_artifact_dir,
        round_state=round_state,
        program_state=program_state,
        session_source_path=session_source_path,
        planner_payload=planner_payload,
    )
    return stop_program(program_state, reason="codex_planning_failed"), lineage, "stop"


def _persist_planner_trace(
    *,
    round_state: Any,
    program_state: ResearchProgramState,
    round_artifact_dir: Path,
    session_source_path: Path,
    prompt_text: str,
    executions: list[Any],
    status: str,
    error: str | None = None,
) -> dict[str, object]:
    payload = planner_trace_payload(
        round_state=round_state,
        program_state=program_state,
        round_artifact_dir=round_artifact_dir,
        session_source_path=session_source_path,
        prompt_text=prompt_text,
        executions=executions,
        status=status,
        error=error,
    )
    append_planner_trace(round_artifact_dir=round_artifact_dir, payload=payload)
    return payload


def _planning_inputs(
    *,
    root: Path,
    auto_dir: Path,
    state: ResearchProgramState,
    definition: ResearchProgramDefinition,
) -> tuple[Path, ExperimentReport | None, list[dict[str, object]]]:
    return (
        session_program_path(auto_dir),
        safe_report(
            root=root,
            experiment_id=state.active_experiment_id,
            primary_metric=definition.metric_policy.primary,
        ),
        recent_round_summaries(auto_dir=auto_dir),
    )


def _pivot_round_to_child_path(
    *,
    root: Path,
    state: ResearchProgramState,
    lineage: ResearchLineageState,
    active_path: ResearchPathState,
    round_state: ResearchRoundState,
    pivot_reason: str,
    child_hypothesis: str,
    child_slug: str,
) -> tuple[ResearchProgramState, ResearchLineageState, ResearchPathState, ResearchRoundState]:
    state, lineage = pivot_to_child_path(
        root=root,
        state=state,
        lineage=lineage,
        active_path=active_path,
        pivot_reason=pivot_reason,
        child_hypothesis=child_hypothesis,
        child_slug=child_slug,
        source_round=round_state.round_label,
    )
    return state, lineage, require_active_path(state), _round_state_for_active_path(round_state, state=state)


def _run_structured_planner_attempts(
    *,
    state: ResearchProgramState,
    definition: ResearchProgramDefinition,
    active_path: ResearchPathState,
    experiment: ExperimentRecord,
    report: ExperimentReport | None,
    recent_rounds: list[dict[str, object]],
    forced_action: str | None,
    base_config_snapshot: dict[str, object],
    base_config_source: str,
    valid_config_examples: list[dict[str, object]],
    round_artifact_dir: Path,
) -> tuple[str, list[CodexPlannerExecution], CodexDecision | None, str | None]:
    validation_feedback: str | None = None
    executions: list[CodexPlannerExecution] = []
    prompt = ""
    for _attempt in range(2):
        prompt = render_prompt(
            program=state,
            definition=definition,
            active_path=active_path,
            experiment=experiment,
            report=report,
            recent_round_summaries=recent_rounds,
            forced_action=forced_action,
            current_phase=state.current_phase,
            base_config_snapshot=base_config_snapshot,
            base_config_source=base_config_source,
            valid_config_examples=valid_config_examples,
            validation_feedback=validation_feedback,
        )
        try:
            execution = run_planner(
                prompt=prompt,
                codex_command=state.codex_command,
                definition=definition,
                artifact_dir=round_artifact_dir,
            )
            executions.append(execution)
            validate_decision(execution.decision, definition=definition, current_phase=state.current_phase)
            return prompt, executions, execution.decision, None
        except (AgenticResearchCodexError, AgenticResearchOpenRouterError, ValueError, json.JSONDecodeError) as exc:
            validation_feedback = str(exc)
    return prompt, executions, None, validation_feedback


def _run_mutation_planner_attempts(
    *,
    root: Path,
    state: ResearchProgramState,
    definition: ResearchProgramDefinition,
    parent: ParentConfigSelection,
    recent_rounds: list[dict[str, object]],
    round_state: Any,
    round_artifact_dir: Path,
    comparison_dirs: list[Path],
) -> tuple[str, list[Any], str | None, MaterializedMutationConfig | None, str | None]:
    validation_feedback: str | None = None
    executions: list[Any] = []
    prompt = ""
    for _attempt in range(2):
        raw_execution: RawPlannerExecution | None = None
        prompt = render_mutation_prompt(
            definition=definition,
            parent=parent,
            recent_round_summaries=recent_rounds,
            validation_feedback=validation_feedback,
        )
        try:
            raw_execution = run_mutation_planner(
                prompt=prompt,
                codex_command=state.codex_command,
                artifact_dir=round_artifact_dir,
            )
            proposal = parse_mutation_response(raw_execution.raw_response_text, definition=definition)
            candidate = materialize_mutation_config(
                round_label=round_state.round_label,
                config_dir=experiment_config_dir(root=root, experiment_id=round_state.experiment_id),
                parent=parent,
                proposal=proposal,
                comparison_dirs=comparison_dirs,
            )
            executions.append(
                MutationPlannerExecution(
                    proposal=proposal,
                    attempts=raw_execution.attempts,
                    stdout_jsonl=raw_execution.stdout_jsonl,
                    stderr_text=raw_execution.stderr_text,
                    last_message=asdict(proposal),
                    raw_response_text=raw_execution.raw_response_text,
                )
            )
            return prompt, executions, proposal.rationale, candidate, None
        except (AgenticResearchCodexError, AgenticResearchOpenRouterError, ValueError, json.JSONDecodeError) as exc:
            if raw_execution is not None:
                executions.append(raw_execution)
            validation_feedback = str(exc)
    return prompt, executions, None, None, validation_feedback


def plan_structured_round(
    *,
    root: Path,
    state: ResearchProgramState,
    lineage: ResearchLineageState,
    auto_dir: Path,
    definition: ResearchProgramDefinition,
    experiment: ExperimentRecord,
    active_path: ResearchPathState,
    round_state: Any,
    round_artifact_dir: Path,
) -> tuple[ResearchProgramState, ResearchLineageState, str | None]:
    session_source_path, report, recent_rounds = _planning_inputs(
        root=root,
        auto_dir=auto_dir,
        state=state,
        definition=definition,
    )
    forced_action = force_action_for_path(active_path, round_policy=definition.round_policy)
    base_config_snapshot, base_config_source, valid_config_examples = select_reference_configs(
        config_dirs=planner_reference_config_dirs(root=root, state=state, active_path=active_path)
    )
    prompt, executions, decision, error = _run_structured_planner_attempts(
        state=state,
        definition=definition,
        active_path=active_path,
        experiment=experiment,
        report=report,
        recent_rounds=recent_rounds,
        forced_action=forced_action,
        base_config_snapshot=base_config_snapshot,
        base_config_source=base_config_source,
        valid_config_examples=valid_config_examples,
        round_artifact_dir=round_artifact_dir,
    )
    if decision is None:
        return _stop_failed_planning_round(
            auto_dir=auto_dir,
            round_artifact_dir=round_artifact_dir,
            round_state=round_state,
            program_state=state,
            session_source_path=session_source_path,
            prompt_text=prompt,
            executions=executions,
            error=error or "agentic_research_structured_planning_failed",
            lineage=lineage,
        )
    planner_payload = _persist_planner_trace(
        round_state=round_state,
        program_state=state,
        round_artifact_dir=round_artifact_dir,
        session_source_path=session_source_path,
        prompt_text=prompt,
        executions=executions,
        status="succeeded",
    )
    state = apply_program_phase_action(state=state, definition=definition, decision=decision)
    if forced_action == "scale" and decision.next_action != "scale":
        decision = replace(decision, next_action="scale")
    if decision.next_action == "stop":
        return stop_program(state, reason="codex_requested_stop"), lineage, "stop"
    if decision.next_action == "pivot":
        state, lineage, active_path, round_state = _pivot_round_to_child_path(
            root=root,
            state=state,
            lineage=lineage,
            active_path=active_path,
            pivot_reason=decision.decision_rationale,
            child_hypothesis=decision.path_hypothesis,
            child_slug=decision.path_slug,
            round_state=round_state,
        )
    config_dir = experiment_config_dir(root=root, experiment_id=round_state.experiment_id)
    filenames = materialize_configs(
        round_label=round_state.round_label,
        config_dir=config_dir,
        base_config=base_config_snapshot,
        configs=decision.configs,
        allowed_paths=definition.config_policy.allowed_paths,
    )
    if force_action_for_path(active_path, round_policy=definition.round_policy) == "scale":
        state = _mark_scale_confirmation_used(state, active_path=active_path)
    round_state = replace(
        round_state,
        status="planned",
        config_filenames=filenames,
        decision_action=decision.next_action,
        experiment_question=decision.experiment_question,
        winner_criteria=decision.winner_criteria,
        decision_rationale=decision.decision_rationale,
        decision_path_hypothesis=decision.path_hypothesis,
        decision_path_slug=decision.path_slug,
        phase_id=state.current_phase.phase_id if state.current_phase is not None else None,
        phase_action=decision.phase_action,
        phase_transition_rationale=decision.phase_transition_rationale,
        updated_at=utc_now_iso(),
    )
    return _finish_planned_round(
        round_artifact_dir=round_artifact_dir,
        round_state=round_state,
        program_state=state,
        session_source_path=session_source_path,
        planner_payload=planner_payload,
        lineage=lineage,
    )


def plan_mutation_round(
    *,
    root: Path,
    state: ResearchProgramState,
    lineage: ResearchLineageState,
    auto_dir: Path,
    definition: ResearchProgramDefinition,
    experiment: ExperimentRecord,
    active_path: ResearchPathState,
    round_state: Any,
    round_artifact_dir: Path,
) -> tuple[ResearchProgramState, ResearchLineageState, str | None]:
    session_source_path, report, recent_rounds = _planning_inputs(
        root=root,
        auto_dir=auto_dir,
        state=state,
        definition=definition,
    )
    comparison_dirs = planner_reference_config_dirs(root=root, state=state, active_path=active_path)
    try:
        parent = select_parent_config(
            root=root,
            experiment=experiment,
            report=report,
            config_dirs=comparison_dirs,
        )
    except ValueError as exc:
        return _stop_failed_planning_round(
            auto_dir=auto_dir,
            round_artifact_dir=round_artifact_dir,
            round_state=round_state,
            program_state=state,
            session_source_path=session_source_path,
            prompt_text="",
            executions=[],
            error=str(exc),
            lineage=lineage,
        )
    round_state = replace(
        round_state,
        parent_config_filename=parent.config_filename,
        parent_run_id=parent.run_id,
        parent_selection_reason=parent.selection_reason,
        updated_at=utc_now_iso(),
    )
    prompt, executions, rationale, candidate, error = _run_mutation_planner_attempts(
        root=root,
        state=state,
        definition=definition,
        parent=parent,
        recent_rounds=recent_rounds,
        round_state=round_state,
        round_artifact_dir=round_artifact_dir,
        comparison_dirs=comparison_dirs,
    )
    if rationale is None or candidate is None:
        return _stop_failed_planning_round(
            auto_dir=auto_dir,
            round_artifact_dir=round_artifact_dir,
            round_state=round_state,
            program_state=state,
            session_source_path=session_source_path,
            prompt_text=prompt,
            executions=executions,
            error=error or "agentic_research_mutation_planning_failed",
            lineage=lineage,
        )
    planner_payload = _persist_planner_trace(
        round_state=round_state,
        program_state=state,
        round_artifact_dir=round_artifact_dir,
        session_source_path=session_source_path,
        prompt_text=prompt,
        executions=executions,
        status="succeeded",
    )
    action = _next_mutation_action(active_path, round_policy=definition.round_policy)
    if action == "pivot":
        state, lineage, active_path, round_state = _pivot_round_to_child_path(
            root=root,
            state=state,
            lineage=lineage,
            active_path=active_path,
            pivot_reason=f"Deterministic pivot after plateau before {round_state.round_label}.",
            child_hypothesis=active_path.hypothesis,
            child_slug=f"{active_path.path_id}-{round_state.round_label}",
            round_state=round_state,
        )
    config_dir = experiment_config_dir(root=root, experiment_id=round_state.experiment_id)
    write_materialized_config(config_dir=config_dir, candidate=candidate)
    if action == "scale":
        state = _mark_scale_confirmation_used(state, active_path=active_path)
    round_state = replace(
        round_state,
        status="planned",
        config_filenames=[candidate.filename],
        decision_action=action,
        experiment_question=_mutation_experiment_question(parent),
        winner_criteria=_mutation_winner_criteria(definition),
        decision_rationale=rationale,
        change_set=candidate.change_set,
        llm_rationale=rationale,
        updated_at=utc_now_iso(),
    )
    return _finish_planned_round(
        round_artifact_dir=round_artifact_dir,
        round_state=round_state,
        program_state=state,
        session_source_path=session_source_path,
        planner_payload=planner_payload,
        lineage=lineage,
    )


def seed_best_run(*, root: Path, experiment: ExperimentRecord, primary_metric: str) -> ResearchBestRun:
    if not experiment.runs:
        return ResearchBestRun()
    try:
        report = report_experiment(
            store_root=root,
            experiment_id=experiment.experiment_id,
            metric=primary_metric,
            limit=1,
        )
    except ExperimentError:
        return ResearchBestRun()
    if not report.rows:
        return ResearchBestRun()
    return _best_run_from_row(experiment_id=experiment.experiment_id, row=report.rows[0])


def safe_report(*, root: Path, experiment_id: str, primary_metric: str) -> Any | None:
    try:
        return report_experiment(store_root=root, experiment_id=experiment_id, metric=primary_metric, limit=10)
    except ExperimentError:
        return None


def planner_reference_config_dirs(
    *,
    root: Path,
    state: ResearchProgramState,
    active_path: ResearchPathState,
) -> list[Path]:
    config_dirs: list[Path] = []
    seen_experiment_ids: set[str] = set()
    paths_by_experiment_id = {path.experiment_id: path for path in state.paths}
    current_path: ResearchPathState | None = active_path
    while current_path is not None:
        experiment_id = current_path.experiment_id
        if experiment_id not in seen_experiment_ids:
            config_dirs.append(experiment_config_dir(root=root, experiment_id=experiment_id))
            seen_experiment_ids.add(experiment_id)
        parent_experiment_id = current_path.parent_experiment_id
        if parent_experiment_id is None:
            break
        if parent_experiment_id in seen_experiment_ids:
            break
        parent_path = paths_by_experiment_id.get(parent_experiment_id)
        if parent_path is None:
            config_dirs.append(experiment_config_dir(root=root, experiment_id=parent_experiment_id))
            seen_experiment_ids.add(parent_experiment_id)
            break
        current_path = parent_path
    if state.program_experiment_id not in seen_experiment_ids:
        config_dirs.append(experiment_config_dir(root=root, experiment_id=state.program_experiment_id))
    return config_dirs


def replace_path_state(state: ResearchProgramState, updated_path: ResearchPathState) -> ResearchProgramState:
    new_paths = [updated_path if item.path_id == updated_path.path_id else item for item in state.paths]
    return replace(state, paths=new_paths, updated_at=utc_now_iso())


def _round_state_for_active_path(round_state: ResearchRoundState, *, state: ResearchProgramState) -> ResearchRoundState:
    return replace(
        round_state,
        experiment_id=state.active_experiment_id,
        path_id=state.active_path_id,
        updated_at=utc_now_iso(),
    )


def _mark_scale_confirmation_used(
    state: ResearchProgramState,
    *,
    active_path: ResearchPathState,
) -> ResearchProgramState:
    return replace_path_state(
        state,
        replace(
            active_path,
            scale_confirmation_used=True,
            needs_scale_confirmation=False,
            scale_confirmation_rounds_completed=active_path.scale_confirmation_rounds_completed,
            updated_at=utc_now_iso(),
        ),
    )


def _finish_planned_round(
    *,
    round_artifact_dir: Path,
    round_state: ResearchRoundState,
    program_state: ResearchProgramState,
    session_source_path: Path,
    planner_payload: dict[str, object],
    lineage: ResearchLineageState,
) -> tuple[ResearchProgramState, ResearchLineageState, None]:
    save_round_bundle(
        round_artifact_dir=round_artifact_dir,
        round_state=round_state,
        program_state=program_state,
        session_source_path=session_source_path,
        planner_payload=planner_payload,
    )
    return (
        replace(
            program_state,
            current_round=round_state,
            last_checkpoint="round_planned",
            updated_at=utc_now_iso(),
        ),
        lineage,
        None,
    )


def require_active_path(state: ResearchProgramState) -> ResearchPathState:
    for item in state.paths:
        if item.path_id == state.active_path_id:
            return item
    raise ValueError("agentic_research_active_path_missing")


def stop_program(state: ResearchProgramState, *, reason: str) -> ResearchProgramState:
    return replace(
        state,
        status="stopped",
        stop_reason=reason,
        last_checkpoint="stopped",
        updated_at=utc_now_iso(),
    )


def experiment_config_dir(*, root: Path, experiment_id: str) -> Path:
    record = get_experiment(store_root=root, experiment_id=experiment_id)
    return record.manifest_path.parent / "configs"


def apply_program_phase_action(
    *,
    state: ResearchProgramState,
    definition: ResearchProgramDefinition,
    decision: Any,
) -> ResearchProgramState:
    if not definition.phases:
        return state
    current_phase = state.current_phase
    if current_phase is None:
        raise ValueError("agentic_research_phase_state_missing")
    phase_action = decision.phase_action or "stay"
    if phase_action == "stay":
        return state
    if phase_action == "complete":
        return replace(
            state,
            current_phase=replace(
                current_phase,
                status="completed",
                transition_rationale=decision.phase_transition_rationale,
                updated_at=utc_now_iso(),
            ),
            updated_at=utc_now_iso(),
        )
    next_phase = next_phase_definition(definition, current_phase.phase_id)
    if next_phase is None:
        raise ValueError("agentic_research_phase_advance_invalid")
    now = utc_now_iso()
    return replace(
        state,
        current_phase=ResearchPhaseState(
            phase_id=next_phase.phase_id,
            phase_title=next_phase.title,
            status="active",
            round_count=0,
            transition_rationale=decision.phase_transition_rationale,
            started_at=now,
            updated_at=now,
        ),
        updated_at=now,
    )


def increment_phase_round_count(current_phase: ResearchPhaseState | None) -> ResearchPhaseState | None:
    if current_phase is None or current_phase.status != "active":
        return current_phase
    return replace(
        current_phase,
        round_count=current_phase.round_count + 1,
        updated_at=utc_now_iso(),
    )

def planner_schema(definition: ResearchProgramDefinition) -> dict[str, object]:
    properties: dict[str, object] = {
        "experiment_question": {"type": "string"},
        "winner_criteria": {"type": "string"},
        "decision_rationale": {"type": "string"},
        "next_action": {
            "type": "string",
            "enum": ["continue", "scale", "pivot", "stop"],
        },
        "path_hypothesis": {"type": "string"},
        "path_slug": {"type": "string"},
        "configs": {
            "type": "array",
            "minItems": 0,
            "maxItems": definition.config_policy.max_candidate_configs,
            "items": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "rationale": {"type": "string"},
                    "overrides": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "value_json": {"type": "string"},
                            },
                            "required": ["path", "value_json"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["filename", "rationale", "overrides"],
                "additionalProperties": False,
            },
        },
    }
    required = [
        "experiment_question",
        "winner_criteria",
        "decision_rationale",
        "next_action",
        "path_hypothesis",
        "path_slug",
        "configs",
    ]
    if definition.phases:
        properties["phase_action"] = {
            "type": "string",
            "enum": ["stay", "advance", "complete"],
        }
        properties["phase_transition_rationale"] = {"type": "string"}
        required.extend(["phase_action", "phase_transition_rationale"])
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def run_planner(
    *,
    prompt: str,
    codex_command: list[str],
    definition: ResearchProgramDefinition,
    artifact_dir: Path,
) -> CodexPlannerExecution:
    schema = planner_schema(definition)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=artifact_dir,
        prefix=".planner_schema_",
        suffix=".json",
        delete=False,
    ) as handle:
        schema_path = Path(handle.name)
    try:
        schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")
        source = active_model_source()
        if source == "openrouter":
            return run_openrouter_planner(
                prompt=prompt,
                schema_path=schema_path,
                artifact_dir=artifact_dir,
            )
        return run_codex_planner(
            prompt=prompt,
            command=codex_command,
            schema_path=schema_path,
            artifact_dir=artifact_dir,
        )
    finally:
        try:
            if schema_path.exists():
                schema_path.unlink()
        except OSError:
            pass


def run_mutation_planner(
    *,
    prompt: str,
    codex_command: list[str],
    artifact_dir: Path,
) -> RawPlannerExecution:
    if active_model_source() == "openrouter":
        return run_openrouter_raw_planner(prompt=prompt)
    return run_codex_raw_planner(prompt=prompt, command=codex_command, artifact_dir=artifact_dir)


def _next_mutation_action(active_path: ResearchPathState, *, round_policy: Any) -> str:
    if should_pivot_path(active_path, round_policy=round_policy):
        return "pivot"
    if force_action_for_path(active_path, round_policy=round_policy) == "scale":
        return "scale"
    return "continue"


def _mutation_experiment_question(parent: ParentConfigSelection) -> str:
    return f"What is the next targeted mutation after `{parent.config_filename}`?"


def _mutation_winner_criteria(definition: ResearchProgramDefinition) -> str:
    sanity_checks = ", ".join(definition.metric_policy.sanity_checks) or "no additional sanity checks"
    return (
        f"Improve {definition.metric_policy.primary}, use {definition.metric_policy.tie_break} as the tie-break, "
        f"and sanity-check {sanity_checks}."
    )



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


def _phase_summary(phase: Any | None) -> dict[str, object] | None:
    if phase is None:
        return None
    return {
        "phase_id": phase.phase_id,
        "title": phase.title,
        "summary": phase.summary,
        "gate": phase.gate,
    }


def render_prompt(
    *,
    program: ResearchProgramState,
    definition: ResearchProgramDefinition,
    active_path: ResearchPathState,
    experiment: ExperimentRecord,
    report: ExperimentReport | None,
    recent_round_summaries: list[dict[str, object]],
    forced_action: str | None,
    current_phase: ResearchPhaseState | None,
    base_config_snapshot: dict[str, object],
    base_config_source: str,
    valid_config_examples: list[dict[str, object]],
    validation_feedback: str | None = None,
) -> str:
    """Render the Codex planning prompt from compact structured context."""
    report_rows = [_row_to_dict(row) for row in report.rows[:10]] if report is not None else []
    current_phase_id = current_phase.phase_id if current_phase is not None else None
    current_phase_definition = get_phase_definition(definition, current_phase_id)
    next_phase = next_phase_definition(definition, current_phase_id)
    config_policy = {
        **asdict(definition.config_policy),
        "allowed_paths": list(definition.config_policy.allowed_paths),
    }
    context = {
        "program": {
            "id": definition.program_id,
            "title": definition.title,
            "description": definition.description,
            "source": definition.source,
            "planner_contract": definition.planner_contract,
            "phase_aware": bool(definition.phases),
        },
        "root_experiment_id": program.root_experiment_id,
        "active_experiment_id": program.active_experiment_id,
        "active_path_id": program.active_path_id,
        "active_path_hypothesis": active_path.hypothesis,
        "active_path_generation": active_path.generation,
        "active_path_plateau_streak": active_path.plateau_streak,
        "active_path_scale_confirmation_used": active_path.scale_confirmation_used,
        "active_path_needs_scale_confirmation": active_path.needs_scale_confirmation,
        "active_path_scale_confirmation_rounds_completed": active_path.scale_confirmation_rounds_completed,
        "improvement_threshold": program.improvement_threshold,
        "next_round_label": next_round_label(program.next_round_number),
        "forced_action": forced_action or "none",
        "root_experiment_name": experiment.name,
        "root_experiment_hypothesis": experiment.hypothesis,
        "best_overall": _best_to_dict(program.best_overall),
        "metric_policy": asdict(definition.metric_policy),
        "round_policy": asdict(definition.round_policy),
        "current_report_metric": report.metric if report is not None else definition.metric_policy.primary,
        "current_report_rows": report_rows,
        "recent_round_summaries": recent_round_summaries[-3:],
        "current_phase": _phase_to_dict(current_phase),
        "current_phase_summary": _phase_summary(current_phase_definition),
        "next_phase_summary": _phase_summary(next_phase),
        "phase_sequence": [_phase_summary(phase) for phase in definition.phases],
        "base_config_source": base_config_source,
        "base_config_snapshot": base_config_snapshot,
        "allowed_override_paths": list(definition.config_policy.allowed_paths),
        "config_policy": config_policy,
        "config_filename_rules": {
            "agent_must_provide": "short slug filename ending in .json",
            "python_will_add_round_prefix": True,
            "python_will_normalize_slug": True,
        },
        "valid_config_examples": [_prompt_ready_example(item) for item in valid_config_examples],
    }
    return render_prompt_template(
        definition.prompt_template,
        {
            "CONTEXT_JSON": json.dumps(context, indent=2, sort_keys=True),
            "VALIDATION_FEEDBACK_BLOCK": render_validation_feedback_block(validation_feedback),
        },
        source=definition.source_path or definition.program_id,
    )


def validate_decision(
    decision: CodexDecision,
    *,
    definition: ResearchProgramDefinition,
    current_phase: ResearchPhaseState | None,
) -> None:
    """Validate one Codex planning decision against supervisor rules."""
    if decision.next_action not in {"continue", "scale", "pivot", "stop"}:
        raise ValueError("agentic_research_codex_action_invalid")
    if definition.phases:
        if decision.phase_action not in {"stay", "advance", "complete"}:
            raise ValueError("agentic_research_codex_phase_action_invalid")
        if decision.phase_action in {"advance", "complete"} and not (
            decision.phase_transition_rationale and decision.phase_transition_rationale.strip()
        ):
            raise ValueError("agentic_research_codex_phase_transition_rationale_missing")
        if (
            decision.phase_action == "advance"
            and next_phase_definition(
                definition,
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
    min_configs = definition.config_policy.min_candidate_configs or 0
    max_configs = definition.config_policy.max_candidate_configs
    if len(decision.configs) < min_configs or len(decision.configs) > max_configs:
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
    allowed_paths: tuple[str, ...] | None = None,
) -> list[str]:
    """Validate and write Codex-generated configs into the experiment config dir."""
    config_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for item in configs:
        filename = normalize_config_filename(round_label=round_label, filename=item.filename)
        materialized_payload = materialize_config_payload(
            base_config=base_config,
            overrides=item.overrides,
            allowed_paths=allowed_paths,
        )
        path = config_dir / filename
        path.write_text(json.dumps(materialized_payload, indent=2, sort_keys=True), encoding="utf-8")
        written.append(filename)
    return written


def materialize_config_payload(
    *,
    base_config: dict[str, object],
    overrides: dict[str, object],
    allowed_paths: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Apply one override payload onto a validated base config snapshot."""
    payload = deepcopy(base_config)
    if not isinstance(payload, dict):
        raise ValueError("agentic_research_base_config_invalid")
    _apply_overrides(
        target=payload,
        overrides=overrides,
        path=(),
        allowed_paths=allowed_paths or _FALLBACK_ALLOWED_PATHS,
    )
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
            snapshot = _validated_config_snapshot(config_path)
            if snapshot is None:
                continue
            valid_snapshots.append((str(config_path), snapshot))
            if len(valid_snapshots) >= 2:
                break
        if len(valid_snapshots) >= 2:
            break
    if not valid_snapshots:
        fallback = TrainingConfig.model_validate(_FALLBACK_BASE_CONFIG).model_dump(mode="python", exclude_none=True)
        example_overrides = _fallback_example_overrides(fallback)
        variant = materialize_config_payload(
            base_config=fallback,
            overrides=example_overrides,
            allowed_paths=_FALLBACK_ALLOWED_PATHS,
        )
        return (
            fallback,
            "built_in_fallback",
            [
                _reference_config_example(
                    filename="reference_base.json",
                    rationale="Fallback validated base config.",
                    overrides={},
                    materialized_config=fallback,
                ),
                _reference_config_example(
                    filename="reference_variant.json",
                    rationale="Fallback single-variable variant.",
                    overrides=example_overrides,
                    materialized_config=variant,
                ),
            ],
        )
    base_source, base_snapshot = valid_snapshots[0]
    examples = [
        _reference_config_example(
            filename=Path(path).name,
            rationale="Validated reference config already present in the experiment lineage.",
            overrides={},
            materialized_config=snapshot,
        )
        for path, snapshot in valid_snapshots
    ]
    if len(examples) == 1:
        example_overrides = _fallback_example_overrides(base_snapshot)
        examples.append(
            _reference_config_example(
                filename="reference_variant.json",
                rationale="Validated synthetic single-variable variant built from the base snapshot.",
                overrides=example_overrides,
                materialized_config=materialize_config_payload(
                    base_config=base_snapshot,
                    overrides=example_overrides,
                    allowed_paths=_FALLBACK_ALLOWED_PATHS,
                ),
            )
        )
    return base_snapshot, base_source, examples[:2]


def _reference_config_example(
    *,
    filename: str,
    rationale: str,
    overrides: dict[str, object],
    materialized_config: dict[str, object],
) -> dict[str, object]:
    return {
        "filename": filename,
        "rationale": rationale,
        "overrides": overrides,
        "materialized_config": materialized_config,
    }


def _validated_config_snapshot(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        validated = TrainingConfig.model_validate(payload)
    except Exception:
        return None
    return validated.model_dump(mode="python", exclude_none=True)


def round_summary_from_report(
    *,
    report: ExperimentReport,
    run_ids: list[str],
    round_state: ResearchRoundState,
    primary_metric: str,
    tie_break_metric: str,
) -> dict[str, object]:
    """Build a compact persisted round summary from the experiment report."""
    run_id_set = set(run_ids)
    round_rows = [row for row in report.rows if row.run_id in run_id_set]
    rows = [_row_to_dict(row) for row in round_rows]
    best_row = select_best_row(
        round_rows,
        primary_metric=primary_metric,
        tie_break_metric=tie_break_metric,
    )
    return {
        "round_label": round_state.round_label,
        "experiment_id": round_state.experiment_id,
        "path_id": round_state.path_id,
        "decision_action": round_state.decision_action,
        "experiment_question": round_state.experiment_question,
        "winner_criteria": round_state.winner_criteria,
        "decision_rationale": round_state.decision_rationale,
        "parent_run_id": round_state.parent_run_id,
        "parent_config_filename": round_state.parent_config_filename,
        "change_set": list(round_state.change_set),
        "llm_rationale": round_state.llm_rationale,
        "phase_id": round_state.phase_id,
        "phase_action": round_state.phase_action,
        "phase_transition_rationale": round_state.phase_transition_rationale,
        "run_ids": list(run_ids),
        "rows": rows,
        "best_row": _row_to_dict(best_row) if best_row is not None else None,
    }


def select_best_row(
    rows: list[ExperimentReportRow],
    *,
    primary_metric: str = "bmc_last_200_eras.mean",
    tie_break_metric: str = "bmc.mean",
) -> ExperimentReportRow | None:
    """Select the best row using the primary metric and tie-break metric."""
    best: ExperimentReportRow | None = None
    best_metrics: tuple[float, float] | None = None
    for row in rows:
        row_primary = _report_row_metric(row, primary_metric)
        if row_primary is None:
            continue
        row_metrics = (row_primary, _report_row_metric(row, tie_break_metric) or float("-inf"))
        if best_metrics is None or row_metrics > best_metrics:
            best = row
            best_metrics = row_metrics
    return best


def update_best_overall(
    *,
    current_best: ResearchBestRun,
    round_best: ExperimentReportRow | None,
    experiment_id: str,
    threshold: float,
    primary_metric: str = "bmc_last_200_eras.mean",
) -> tuple[ResearchBestRun, bool]:
    """Update the program-level best run tracker and return whether it improved."""
    round_value = _report_row_metric(round_best, primary_metric) if round_best is not None else None
    if round_best is None or round_value is None:
        return current_best, False
    prior_value = _best_metric_value(current_best, primary_metric)
    improved = prior_value is None or (round_value - prior_value) >= threshold
    if not improved:
        return current_best, False
    return _best_run_from_row(experiment_id=experiment_id, row=round_best), True


def update_path_after_round(
    *,
    path: ResearchPathState,
    round_best: ExperimentReportRow | None,
    improved: bool,
    round_policy: ResearchProgramRoundPolicy = _DEFAULT_ROUND_POLICY,
) -> ResearchPathState:
    """Return the updated path state after one completed round."""
    now = utc_now_iso()
    if improved:
        return replace(
            path,
            rounds_completed=path.rounds_completed + 1,
            plateau_streak=0,
            needs_scale_confirmation=False,
            scale_confirmation_rounds_completed=0,
            best_run_id=round_best.run_id if round_best is not None else path.best_run_id,
            updated_at=now,
        )
    new_streak = path.plateau_streak + 1
    needs_scale_confirmation = False
    scale_rounds_completed = path.scale_confirmation_rounds_completed
    if round_policy.require_scale_confirmation:
        if path.scale_confirmation_used:
            if scale_rounds_completed < round_policy.scale_confirmation_rounds:
                scale_rounds_completed += 1
            needs_scale_confirmation = scale_rounds_completed < round_policy.scale_confirmation_rounds
        elif new_streak >= round_policy.plateau_non_improving_rounds:
            needs_scale_confirmation = True
    return replace(
        path,
        rounds_completed=path.rounds_completed + 1,
        plateau_streak=new_streak,
        needs_scale_confirmation=needs_scale_confirmation,
        scale_confirmation_rounds_completed=scale_rounds_completed,
        updated_at=now,
    )


def force_action_for_path(
    path: ResearchPathState,
    *,
    round_policy: ResearchProgramRoundPolicy = _DEFAULT_ROUND_POLICY,
) -> str | None:
    """Return the forced next action implied by path plateau rules."""
    if round_policy.require_scale_confirmation and path.needs_scale_confirmation:
        return "scale"
    return None


def should_pivot_path(
    path: ResearchPathState,
    *,
    round_policy: ResearchProgramRoundPolicy = _DEFAULT_ROUND_POLICY,
) -> bool:
    """Return whether the next planning step should pivot into a new child experiment."""
    if path.plateau_streak < round_policy.plateau_non_improving_rounds:
        return False
    if not round_policy.require_scale_confirmation:
        return True
    completed_scale_rounds = path.scale_confirmation_rounds_completed
    if (
        path.scale_confirmation_used
        and not path.needs_scale_confirmation
        and completed_scale_rounds == 0
        and round_policy.scale_confirmation_rounds == 1
    ):
        completed_scale_rounds = 1
    return (
        path.scale_confirmation_used
        and not path.needs_scale_confirmation
        and completed_scale_rounds >= round_policy.scale_confirmation_rounds
    )


def child_experiment_id(*, root_experiment_id: str, generation: int, path_slug: str) -> str:
    """Build the canonical child experiment id for one new path."""
    slug = _FILENAME_SLUG_RE.sub("-", path_slug.strip().lower()).strip("-") or "new-path"
    return f"{root_experiment_id}--p{generation:02d}-{slug}"


def _apply_overrides(
    *,
    target: dict[str, object],
    overrides: dict[str, object],
    path: tuple[str, ...],
    allowed_paths: tuple[str, ...],
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
            _apply_overrides(
                target=target[key],
                overrides=value,
                path=candidate_path,
                allowed_paths=allowed_paths,
            )
            continue
        dotted = ".".join(candidate_path)
        if not _path_allowed(candidate_path, allowed_paths=allowed_paths):
            raise ValueError(f"agentic_research_override_path_not_allowed:{dotted}")
        target[key] = value


def _path_allowed(path: tuple[str, ...], *, allowed_paths: tuple[str, ...]) -> bool:
    for allowed in allowed_paths:
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


def _override_items(overrides: dict[str, object], *, path: tuple[str, ...] = ()) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for key, value in overrides.items():
        if isinstance(value, dict):
            items.extend(_override_items(value, path=(*path, key)))
            continue
        items.append(
            {
                "path": ".".join((*path, key)),
                "value_json": json.dumps(value, sort_keys=True),
            }
        )
    return items


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


def _report_row_metric(row: ExperimentReportRow, metric_key: str) -> float | None:
    field_name = _REPORT_METRIC_FIELDS.get(metric_key)
    if field_name is None:
        return row.metric_value
    value = getattr(row, field_name)
    return float(value) if isinstance(value, (int, float)) else None


def _best_metric_value(best: ResearchBestRun, metric_key: str) -> float | None:
    field_name = _REPORT_METRIC_FIELDS.get(metric_key, "bmc_last_200_eras_mean")
    value = getattr(best, field_name)
    return float(value) if isinstance(value, (int, float)) else None


def _best_run_from_row(*, experiment_id: str, row: ExperimentReportRow) -> ResearchBestRun:
    return ResearchBestRun(
        experiment_id=experiment_id,
        run_id=row.run_id,
        bmc_last_200_eras_mean=row.bmc_last_200_eras_mean,
        bmc_mean=row.bmc_mean,
        corr_mean=row.corr_mean,
        mmc_mean=row.mmc_mean,
        cwmm_mean=row.cwmm_mean,
        updated_at=utc_now_iso(),
    )
