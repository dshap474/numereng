"""Agentic research supervisor services."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from numereng.features.agentic_research.codex_exec import (
    AgenticResearchCodexError,
    default_codex_command,
    run_codex_planner,
    run_codex_raw_planner,
)
from numereng.features.agentic_research.config_evolution import (
    MaterializedMutationConfig,
    ParentConfigSelection,
    materialize_mutation_config,
    parse_mutation_response,
    render_mutation_prompt,
    select_parent_config,
    write_materialized_config,
)
from numereng.features.agentic_research.contracts import (
    CodexPlannerExecution,
    MutationPlannerExecution,
    RawPlannerExecution,
    ResearchBestRun,
    ResearchInitResult,
    ResearchLineageLink,
    ResearchLineageState,
    ResearchPathState,
    ResearchPhaseState,
    ResearchProgramState,
    ResearchRunResult,
    ResearchStatusResult,
    ResearchStrategyId,
)
from numereng.features.agentic_research.openrouter_exec import (
    AgenticResearchOpenRouterError,
    run_openrouter_planner,
    run_openrouter_raw_planner,
)
from numereng.features.agentic_research.planner import (
    build_round_state,
    child_experiment_id,
    force_action_for_path,
    materialize_configs,
    render_prompt,
    round_summary_from_report,
    select_best_row,
    select_reference_configs,
    should_pivot_path,
    update_best_overall,
    update_path_after_round,
    validate_decision,
)
from numereng.features.agentic_research.state import (
    append_jsonl_artifact,
    append_text_artifact,
    ensure_agentic_research_dirs,
    lineage_path,
    llm_trace_markdown_path,
    llm_trace_path,
    load_lineage_state,
    load_program_state,
    program_path,
    proposal_to_dict,
    round_dir,
    round_markdown_path,
    round_record_path,
    save_lineage_state,
    save_program_state,
    save_round_artifact,
    save_text_artifact,
    utc_now_iso,
)
from numereng.features.agentic_research.strategy import (
    ResearchStrategyDefinition,
    get_strategy_definition,
    initial_phase_state,
    next_phase_definition,
)
from numereng.features.experiments import (
    ExperimentError,
    ExperimentRecord,
    create_experiment,
    get_experiment,
    report_experiment,
    score_experiment_round,
    train_experiment,
)
from numereng.features.store import StoreError, resolve_store_root, upsert_experiment
from numereng.features.telemetry import bind_launch_metadata
from numereng.platform.clients.openrouter import active_model_source, load_openrouter_config

_AGENTIC_RESEARCH_DIRNAME = "agentic_research"
_DEFAULT_CODEX_COMMAND = default_codex_command()
_DEFAULT_PRIMARY_METRIC = "bmc_last_200_eras.mean"
_DEFAULT_SCORING_STAGE = "post_training_full"
_INITIAL_PATH_ID = "p00"


class AgenticResearchError(Exception):
    """Base error for agentic research workflows."""


class AgenticResearchNotInitializedError(AgenticResearchError):
    """Raised when the supervisor state has not been initialized."""


class AgenticResearchValidationError(AgenticResearchError):
    """Raised when agentic research inputs or state are invalid."""


def init_research(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str,
    strategy: ResearchStrategyId,
    improvement_threshold: float = 0.0002,
) -> ResearchInitResult:
    """Initialize one agentic research supervisor rooted at one experiment."""
    if improvement_threshold <= 0.0:
        raise AgenticResearchValidationError("agentic_research_improvement_threshold_invalid")

    root = resolve_store_root(store_root)
    experiment = get_experiment(store_root=root, experiment_id=experiment_id)
    strategy_definition = _resolve_strategy_definition(strategy)
    if experiment.status == "archived":
        raise AgenticResearchValidationError("agentic_research_root_experiment_archived")

    auto_dir = _agentic_research_dir(experiment)
    ensure_agentic_research_dirs(auto_dir)
    program_state_path = program_path(auto_dir)
    lineage_state_path = lineage_path(auto_dir)
    if program_state_path.is_file() and lineage_state_path.is_file():
        state = load_program_state(program_state_path)
        return ResearchInitResult(
            root_experiment_id=state.root_experiment_id,
            strategy=state.strategy,
            strategy_description=state.strategy_description,
            status=state.status,
            active_experiment_id=state.active_experiment_id,
            active_path_id=state.active_path_id,
            improvement_threshold=state.improvement_threshold,
            current_phase=state.current_phase,
            agentic_research_dir=auto_dir,
            program_path=program_state_path,
            lineage_path=lineage_state_path,
        )

    now = utc_now_iso()
    root_path = ResearchPathState(
        path_id=_INITIAL_PATH_ID,
        experiment_id=experiment.experiment_id,
        parent_experiment_id=None,
        generation=0,
        hypothesis=experiment.hypothesis or f"Agentic research root path for {experiment.experiment_id}",
        status="active",
        pivot_reason=None,
        source_round=None,
        rounds_completed=0,
        plateau_streak=0,
        scale_confirmation_used=False,
        needs_scale_confirmation=False,
        best_run_id=None,
        created_at=now,
        updated_at=now,
    )
    best_overall = _seed_best_run(root=root, experiment=experiment)
    program = ResearchProgramState(
        root_experiment_id=experiment.experiment_id,
        program_experiment_id=experiment.experiment_id,
        strategy=strategy_definition.strategy_id,
        strategy_description=strategy_definition.description,
        status="initialized",
        active_path_id=root_path.path_id,
        active_experiment_id=experiment.experiment_id,
        next_round_number=1,
        total_rounds_completed=0,
        total_paths_created=1,
        improvement_threshold=improvement_threshold,
        scoring_stage=_DEFAULT_SCORING_STAGE,
        codex_command=list(_DEFAULT_CODEX_COMMAND),
        last_checkpoint="initialized",
        stop_reason=None,
        current_round=None,
        current_phase=initial_phase_state(strategy_definition),
        best_overall=best_overall,
        paths=[root_path],
        created_at=now,
        updated_at=now,
    )
    lineage = ResearchLineageState(
        root_experiment_id=experiment.experiment_id,
        program_experiment_id=experiment.experiment_id,
        active_path_id=root_path.path_id,
        links=[
            ResearchLineageLink(
                path_id=root_path.path_id,
                experiment_id=experiment.experiment_id,
                parent_experiment_id=None,
                generation=0,
                source_round=None,
                pivot_reason=None,
                created_at=now,
            )
        ],
    )

    save_program_state(program_state_path, program)
    save_lineage_state(lineage_state_path, lineage)
    _upsert_agentic_research_metadata(
        root=root,
        experiment=experiment,
        metadata_update={
            "root_experiment_id": experiment.experiment_id,
            "program_experiment_id": experiment.experiment_id,
            "strategy": strategy_definition.strategy_id,
            "parent_experiment_id": None,
            "path_id": root_path.path_id,
            "pivot_reason": None,
            "source_round": None,
            "generation": 0,
        },
    )
    return ResearchInitResult(
        root_experiment_id=experiment.experiment_id,
        strategy=program.strategy,
        strategy_description=program.strategy_description,
        status=program.status,
        active_experiment_id=program.active_experiment_id,
        active_path_id=program.active_path_id,
        improvement_threshold=program.improvement_threshold,
        current_phase=program.current_phase,
        agentic_research_dir=auto_dir,
        program_path=program_state_path,
        lineage_path=lineage_state_path,
    )


def get_research_status(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str,
) -> ResearchStatusResult:
    """Load the current supervisor status payload."""
    auto_dir = _agentic_research_dir(get_experiment(store_root=store_root, experiment_id=experiment_id))
    state = load_program_state(program_path(auto_dir))
    return ResearchStatusResult(
        root_experiment_id=state.root_experiment_id,
        strategy=state.strategy,
        strategy_description=state.strategy_description,
        status=state.status,
        active_experiment_id=state.active_experiment_id,
        active_path_id=state.active_path_id,
        next_round_number=state.next_round_number,
        total_rounds_completed=state.total_rounds_completed,
        total_paths_created=state.total_paths_created,
        improvement_threshold=state.improvement_threshold,
        last_checkpoint=state.last_checkpoint,
        stop_reason=state.stop_reason,
        best_overall=state.best_overall,
        current_round=state.current_round,
        current_phase=state.current_phase,
        program_path=program_path(auto_dir),
        lineage_path=lineage_path(auto_dir),
    )


def run_research(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str,
    max_rounds: int | None = None,
    max_paths: int | None = None,
) -> ResearchRunResult:
    """Run the supervisor loop in the foreground until stopped or interrupted."""
    root = resolve_store_root(store_root)
    auto_dir = _agentic_research_dir(get_experiment(store_root=root, experiment_id=experiment_id))
    program_state_path = program_path(auto_dir)
    lineage_state_path = lineage_path(auto_dir)
    if not program_state_path.is_file():
        raise AgenticResearchNotInitializedError(f"agentic_research_not_initialized:{experiment_id}")
    state = load_program_state(program_state_path)
    lineage = load_lineage_state(lineage_state_path)
    rounds_started = 0
    resumed_round_active = state.current_round is not None

    try:
        while True:
            if max_rounds is not None and state.current_round is None and rounds_started >= max_rounds:
                state = _stop_program(state, reason="max_rounds_reached")
                save_program_state(program_state_path, state)
                break

            active_path = _require_active_path(state)
            if should_pivot_path(active_path):
                if max_paths is not None and state.total_paths_created >= max_paths:
                    state = _stop_program(state, reason="max_paths_reached")
                    save_program_state(program_state_path, state)
                    break
                state, lineage = _pivot_to_child_path(
                    root=root,
                    state=state,
                    lineage=lineage,
                    active_path=active_path,
                )
                save_program_state(program_state_path, state)
                save_lineage_state(lineage_state_path, lineage)
                active_path = _require_active_path(state)

            if state.current_round is None:
                state = replace(
                    state,
                    status="running",
                    current_round=build_round_state(
                        round_number=state.next_round_number,
                        experiment_id=state.active_experiment_id,
                        path_id=state.active_path_id,
                        phase_id=state.current_phase.phase_id if state.current_phase is not None else None,
                    ),
                    last_checkpoint="round_created",
                    stop_reason=None,
                    updated_at=utc_now_iso(),
                )
                save_program_state(program_state_path, state)
                rounds_started += 1

            state, lineage, line_action = _run_current_round(
                root=root,
                state=state,
                lineage=lineage,
                auto_dir=auto_dir,
            )
            save_program_state(program_state_path, state)
            save_lineage_state(lineage_state_path, lineage)
            if resumed_round_active and state.current_round is None:
                rounds_started = max(rounds_started, 1)
                resumed_round_active = False
            if line_action == "stop":
                break
    except KeyboardInterrupt:
        if program_state_path.is_file():
            try:
                state = load_program_state(program_state_path)
            except ValueError:
                pass
        state = replace(
            state,
            status="interrupted",
            stop_reason="keyboard_interrupt",
            last_checkpoint="interrupted",
            updated_at=utc_now_iso(),
        )
        save_program_state(program_state_path, state)
        return ResearchRunResult(
            root_experiment_id=state.root_experiment_id,
            strategy=state.strategy,
            strategy_description=state.strategy_description,
            status=state.status,
            active_experiment_id=state.active_experiment_id,
            active_path_id=state.active_path_id,
            next_round_number=state.next_round_number,
            total_rounds_completed=state.total_rounds_completed,
            total_paths_created=state.total_paths_created,
            last_checkpoint=state.last_checkpoint,
            stop_reason=state.stop_reason,
            current_phase=state.current_phase,
            interrupted=True,
        )

    return ResearchRunResult(
        root_experiment_id=state.root_experiment_id,
        strategy=state.strategy,
        strategy_description=state.strategy_description,
        status=state.status,
        active_experiment_id=state.active_experiment_id,
        active_path_id=state.active_path_id,
        next_round_number=state.next_round_number,
        total_rounds_completed=state.total_rounds_completed,
        total_paths_created=state.total_paths_created,
        last_checkpoint=state.last_checkpoint,
        stop_reason=state.stop_reason,
        current_phase=state.current_phase,
        interrupted=False,
    )


def _run_current_round(
    *,
    root: Path,
    state: ResearchProgramState,
    lineage: ResearchLineageState,
    auto_dir: Path,
) -> tuple[ResearchProgramState, ResearchLineageState, str | None]:
    round_state = state.current_round
    if round_state is None:
        return state, lineage, None
    active_path = _require_active_path(state)
    strategy_definition = _resolve_strategy_definition(state.strategy)
    experiment = get_experiment(store_root=root, experiment_id=state.active_experiment_id)
    round_artifact_dir = round_dir(auto_dir, round_state.round_label)
    round_artifact_dir.mkdir(parents=True, exist_ok=True)

    if round_state.status == "planning":
        if strategy_definition.planner_contract == "config_mutation":
            state, lineage, stop_signal = _plan_mutation_round(
                root=root,
                state=state,
                lineage=lineage,
                auto_dir=auto_dir,
                strategy_definition=strategy_definition,
                experiment=experiment,
                active_path=active_path,
                round_state=round_state,
                round_artifact_dir=round_artifact_dir,
            )
        else:
            state, lineage, stop_signal = _plan_structured_round(
                root=root,
                state=state,
                lineage=lineage,
                auto_dir=auto_dir,
                strategy_definition=strategy_definition,
                experiment=experiment,
                active_path=active_path,
                round_state=round_state,
                round_artifact_dir=round_artifact_dir,
            )
        if stop_signal is not None:
            return state, lineage, stop_signal

    round_state = state.current_round
    if round_state is None:
        return state, lineage, None
    if round_state.status in {"planned", "running"}:
        run_ids = list(round_state.run_ids)
        config_filenames = list(round_state.config_filenames)
        for idx in range(round_state.next_config_index, len(config_filenames)):
            config_dir = _experiment_config_dir(root=root, experiment_id=round_state.experiment_id)
            config_path = config_dir / config_filenames[idx]
            with bind_launch_metadata(source="feature.agentic_research.train", operation_type="run", job_type="run"):
                result = train_experiment(
                    store_root=root,
                    experiment_id=round_state.experiment_id,
                    config_path=config_path,
                )
            run_ids.append(result.run_id)
            round_state = replace(
                round_state,
                status="running",
                next_config_index=idx + 1,
                run_ids=run_ids,
                updated_at=utc_now_iso(),
            )
            state = replace(
                state,
                current_round=round_state,
                last_checkpoint="config_completed",
                updated_at=utc_now_iso(),
            )
            save_program_state(program_path(auto_dir), state)
            _save_round_bundle(round_artifact_dir=round_artifact_dir, round_state=round_state)
        with bind_launch_metadata(source="feature.agentic_research.score_round", operation_type="run", job_type="run"):
            score_experiment_round(
                store_root=root,
                experiment_id=round_state.experiment_id,
                round=round_state.round_label,
                stage=state.scoring_stage,
            )
        round_state = replace(round_state, status="scored", updated_at=utc_now_iso())
        state = replace(state, current_round=round_state, last_checkpoint="round_scored", updated_at=utc_now_iso())
        _save_round_bundle(round_artifact_dir=round_artifact_dir, round_state=round_state)

    round_state = state.current_round
    if round_state is None:
        return state, lineage, None
    if round_state.status == "scored":
        report = report_experiment(
            store_root=root,
            experiment_id=round_state.experiment_id,
            metric=_DEFAULT_PRIMARY_METRIC,
            limit=1000,
        )
        round_rows = [row for row in report.rows if row.run_id in set(round_state.run_ids)]
        best_row = select_best_row(round_rows)
        best_overall, improved = update_best_overall(
            current_best=state.best_overall,
            round_best=best_row,
            experiment_id=round_state.experiment_id,
            threshold=state.improvement_threshold,
        )
        path_state = _require_active_path(state)
        updated_path = update_path_after_round(
            path=path_state,
            round_best=best_row,
            improved=improved,
        )
        _save_round_bundle(
            round_artifact_dir=round_artifact_dir,
            round_state=round_state,
            results_payload=_round_results_payload(
                report=report,
                round_state=round_state,
                improved_best_overall=improved,
            ),
        )
        current_phase = _increment_phase_round_count(state.current_phase)
        state = _replace_path(state, updated_path)
        state = replace(
            state,
            best_overall=best_overall,
            total_rounds_completed=state.total_rounds_completed + 1,
            next_round_number=state.next_round_number + 1,
            current_round=None,
            current_phase=current_phase,
            last_checkpoint="round_completed",
            updated_at=utc_now_iso(),
        )
        return state, lineage, None

    return state, lineage, None


def _pivot_to_child_path(
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
    slug = child_slug or child_hypothesis or f"path-{generation}"
    child_id = child_experiment_id(
        root_experiment_id=state.root_experiment_id,
        generation=generation,
        path_slug=slug,
    )
    child = create_experiment(
        store_root=root,
        experiment_id=child_id,
        name=f"Agentic Research Path {generation}: {slug}",
        hypothesis=child_hypothesis or active_path.hypothesis,
        tags=["agentic-research", f"path:{generation:02d}"],
    )
    _upsert_agentic_research_metadata(
        root=root,
        experiment=child,
        metadata_update={
            "root_experiment_id": state.root_experiment_id,
            "program_experiment_id": state.program_experiment_id,
            "strategy": state.strategy,
            "parent_experiment_id": active_path.experiment_id,
            "path_id": f"p{generation:02d}",
            "pivot_reason": pivot_reason or "plateau",
            "source_round": source_round,
            "generation": generation,
        },
    )
    retired_path = replace(
        active_path,
        status="pivoted",
        pivot_reason=pivot_reason or "plateau",
        updated_at=utc_now_iso(),
    )
    child_path = ResearchPathState(
        path_id=f"p{generation:02d}",
        experiment_id=child.experiment_id,
        parent_experiment_id=active_path.experiment_id,
        generation=generation,
        hypothesis=child_hypothesis or active_path.hypothesis,
        status="active",
        pivot_reason=pivot_reason or "plateau",
        source_round=source_round,
        rounds_completed=0,
        plateau_streak=0,
        scale_confirmation_used=False,
        needs_scale_confirmation=False,
        best_run_id=None,
        created_at=utc_now_iso(),
        updated_at=utc_now_iso(),
    )
    new_paths = []
    for item in state.paths:
        if item.path_id == retired_path.path_id:
            new_paths.append(retired_path)
        else:
            new_paths.append(item)
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
        updated_at=utc_now_iso(),
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
                pivot_reason=pivot_reason or "plateau",
                created_at=utc_now_iso(),
            ),
        ],
    )
    return updated_state, updated_lineage


def _plan_structured_round(
    *,
    root: Path,
    state: ResearchProgramState,
    lineage: ResearchLineageState,
    auto_dir: Path,
    strategy_definition: ResearchStrategyDefinition,
    experiment: ExperimentRecord,
    active_path: ResearchPathState,
    round_state: Any,
    round_artifact_dir: Path,
) -> tuple[ResearchProgramState, ResearchLineageState, str | None]:
    report = _safe_report(root=root, experiment_id=state.active_experiment_id)
    recent_rounds = _recent_round_summaries(auto_dir=auto_dir)
    forced_action = force_action_for_path(active_path)
    base_config_snapshot, base_config_source, valid_config_examples = select_reference_configs(
        config_dirs=_planner_reference_config_dirs(root=root, state=state, active_path=active_path)
    )
    validation_feedback: str | None = None
    executions: list[CodexPlannerExecution] = []
    decision: Any | None = None
    for _attempt in range(2):
        prompt = render_prompt(
            program=state,
            active_path=active_path,
            experiment=experiment,
            report=report,
            recent_round_summaries=recent_rounds,
            forced_action=forced_action,
            strategy=strategy_definition,
            current_phase=state.current_phase,
            base_config_snapshot=base_config_snapshot,
            base_config_source=base_config_source,
            valid_config_examples=valid_config_examples,
            validation_feedback=validation_feedback,
        )
        try:
            execution = _run_planner(
                prompt=prompt,
                codex_command=state.codex_command,
                schema_path=strategy_definition.schema_path,
                artifact_dir=round_artifact_dir,
            )
            executions.append(execution)
            decision = execution.decision
            validate_decision(decision, strategy=strategy_definition, current_phase=state.current_phase)
            break
        except (AgenticResearchCodexError, AgenticResearchOpenRouterError, ValueError, json.JSONDecodeError) as exc:
            validation_feedback = str(exc)
    if decision is None:
        error = validation_feedback or "agentic_research_structured_planning_failed"
        planner_payload = _planner_trace_payload(
            round_state=round_state,
            round_artifact_dir=round_artifact_dir,
            prompt_text=prompt,
            executions=executions,
            status="failed",
            error=error,
        )
        _save_round_bundle(
            round_artifact_dir=round_artifact_dir,
            round_state=round_state,
            planner_payload=planner_payload,
        )
        _append_planner_trace(
            auto_dir=auto_dir,
            round_artifact_dir=round_artifact_dir,
            round_state=round_state,
            prompt_text=prompt,
            executions=executions,
            status="failed",
            error=error,
        )
        return _stop_program(state, reason="codex_planning_failed"), lineage, "stop"
    planner_payload = _planner_trace_payload(
        round_state=round_state,
        round_artifact_dir=round_artifact_dir,
        prompt_text=prompt,
        executions=executions,
        status="succeeded",
    )
    _append_planner_trace(
        auto_dir=auto_dir,
        round_artifact_dir=round_artifact_dir,
        round_state=round_state,
        prompt_text=prompt,
        executions=executions,
        status="succeeded",
    )
    state = _apply_strategy_phase_action(
        state=state,
        strategy=strategy_definition,
        decision=decision,
    )
    round_state = state.current_round or round_state
    active_path = _require_active_path(state)
    if forced_action == "scale" and decision.next_action != "scale":
        decision = replace(decision, next_action="scale")
    if decision.next_action == "stop":
        return _stop_program(state, reason="codex_requested_stop"), lineage, "stop"
    if decision.next_action == "pivot":
        state, lineage = _pivot_to_child_path(
            root=root,
            state=state,
            lineage=lineage,
            active_path=active_path,
            pivot_reason=decision.decision_rationale,
            child_hypothesis=decision.path_hypothesis,
            child_slug=decision.path_slug,
            source_round=round_state.round_label,
        )
        active_path = _require_active_path(state)
        round_state = replace(
            round_state,
            experiment_id=state.active_experiment_id,
            path_id=state.active_path_id,
            updated_at=utc_now_iso(),
        )
    config_dir = _experiment_config_dir(root=root, experiment_id=round_state.experiment_id)
    filenames = materialize_configs(
        round_label=round_state.round_label,
        config_dir=config_dir,
        base_config=base_config_snapshot,
        configs=decision.configs,
    )
    if force_action_for_path(active_path) == "scale":
        state = _replace_path(
            state,
            replace(
                active_path,
                scale_confirmation_used=True,
                needs_scale_confirmation=False,
                updated_at=utc_now_iso(),
            ),
        )
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
    _save_round_bundle(round_artifact_dir=round_artifact_dir, round_state=round_state, planner_payload=planner_payload)
    return (
        replace(
            state,
            current_round=round_state,
            last_checkpoint="round_planned",
            updated_at=utc_now_iso(),
        ),
        lineage,
        None,
    )


def _plan_mutation_round(
    *,
    root: Path,
    state: ResearchProgramState,
    lineage: ResearchLineageState,
    auto_dir: Path,
    strategy_definition: ResearchStrategyDefinition,
    experiment: ExperimentRecord,
    active_path: ResearchPathState,
    round_state: Any,
    round_artifact_dir: Path,
) -> tuple[ResearchProgramState, ResearchLineageState, str | None]:
    report = _safe_report(root=root, experiment_id=state.active_experiment_id)
    recent_rounds = _recent_round_summaries(auto_dir=auto_dir)
    comparison_dirs = _planner_reference_config_dirs(root=root, state=state, active_path=active_path)
    try:
        parent = select_parent_config(
            root=root,
            experiment=experiment,
            report=report,
            config_dirs=comparison_dirs,
        )
    except ValueError as exc:
        error = str(exc)
        planner_payload = _planner_trace_payload(
            round_state=round_state,
            round_artifact_dir=round_artifact_dir,
            prompt_text="",
            executions=[],
            status="failed",
            error=error,
        )
        _save_round_bundle(
            round_artifact_dir=round_artifact_dir,
            round_state=round_state,
            planner_payload=planner_payload,
        )
        _append_planner_trace(
            auto_dir=auto_dir,
            round_artifact_dir=round_artifact_dir,
            round_state=round_state,
            prompt_text="",
            executions=[],
            status="failed",
            error=error,
        )
        return _stop_program(state, reason="codex_planning_failed"), lineage, "stop"
    validation_feedback: str | None = None
    executions: list[Any] = []
    proposal: Any | None = None
    candidate: MaterializedMutationConfig | None = None
    for _attempt in range(2):
        raw_execution: RawPlannerExecution | None = None
        prompt = render_mutation_prompt(
            prompt_path=strategy_definition.prompt_path,
            parent=parent,
            recent_round_summaries=recent_rounds,
            validation_feedback=validation_feedback,
        )
        try:
            raw_execution = _run_mutation_planner(
                prompt=prompt,
                codex_command=state.codex_command,
                artifact_dir=round_artifact_dir,
            )
            proposal = parse_mutation_response(raw_execution.raw_response_text)
            candidate = materialize_mutation_config(
                round_label=round_state.round_label,
                config_dir=_experiment_config_dir(root=root, experiment_id=round_state.experiment_id),
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
                    last_message=proposal_to_dict(proposal),
                    raw_response_text=raw_execution.raw_response_text,
                )
            )
            break
        except (AgenticResearchCodexError, AgenticResearchOpenRouterError, ValueError, json.JSONDecodeError) as exc:
            if raw_execution is not None:
                executions.append(raw_execution)
            validation_feedback = str(exc)
    if proposal is None or candidate is None:
        error = validation_feedback or "agentic_research_mutation_planning_failed"
        planner_payload = _planner_trace_payload(
            round_state=round_state,
            round_artifact_dir=round_artifact_dir,
            prompt_text=prompt,
            executions=executions,
            status="failed",
            error=error,
        )
        _save_round_bundle(
            round_artifact_dir=round_artifact_dir,
            round_state=round_state,
            planner_payload=planner_payload,
        )
        _append_planner_trace(
            auto_dir=auto_dir,
            round_artifact_dir=round_artifact_dir,
            round_state=round_state,
            prompt_text=prompt,
            executions=executions,
            status="failed",
            error=error,
        )
        return _stop_program(state, reason="codex_planning_failed"), lineage, "stop"
    planner_payload = _planner_trace_payload(
        round_state=round_state,
        round_artifact_dir=round_artifact_dir,
        prompt_text=prompt,
        executions=executions,
        status="succeeded",
    )
    _append_planner_trace(
        auto_dir=auto_dir,
        round_artifact_dir=round_artifact_dir,
        round_state=round_state,
        prompt_text=prompt,
        executions=executions,
        status="succeeded",
    )
    action = _next_mutation_action(active_path)
    if action == "pivot":
        state, lineage = _pivot_to_child_path(
            root=root,
            state=state,
            lineage=lineage,
            active_path=active_path,
            pivot_reason=f"Deterministic pivot after plateau before {round_state.round_label}.",
            child_hypothesis=active_path.hypothesis,
            child_slug=f"{active_path.path_id}-{round_state.round_label}",
            source_round=round_state.round_label,
        )
        round_state = replace(
            round_state,
            experiment_id=state.active_experiment_id,
            path_id=state.active_path_id,
            updated_at=utc_now_iso(),
        )
    config_dir = _experiment_config_dir(root=root, experiment_id=round_state.experiment_id)
    write_materialized_config(config_dir=config_dir, candidate=candidate)
    if action == "scale":
        state = _replace_path(
            state,
            replace(
                active_path,
                scale_confirmation_used=True,
                needs_scale_confirmation=False,
                updated_at=utc_now_iso(),
            ),
        )
    round_state = replace(
        round_state,
        status="planned",
        config_filenames=[candidate.filename],
        decision_action=action,
        experiment_question=_mutation_experiment_question(parent),
        winner_criteria=_mutation_winner_criteria(),
        decision_rationale=proposal.rationale,
        parent_run_id=parent.run_id,
        parent_config_filename=parent.config_filename,
        change_set=candidate.change_set,
        llm_rationale=proposal.rationale,
        updated_at=utc_now_iso(),
    )
    _save_round_bundle(round_artifact_dir=round_artifact_dir, round_state=round_state, planner_payload=planner_payload)
    return (
        replace(
            state,
            current_round=round_state,
            last_checkpoint="round_planned",
            updated_at=utc_now_iso(),
        ),
        lineage,
        None,
    )


def _seed_best_run(*, root: Path, experiment: ExperimentRecord) -> ResearchBestRun:
    if not experiment.runs:
        return ResearchBestRun()
    try:
        report = report_experiment(
            store_root=root,
            experiment_id=experiment.experiment_id,
            metric=_DEFAULT_PRIMARY_METRIC,
            limit=1,
        )
    except ExperimentError:
        return ResearchBestRun()
    if not report.rows:
        return ResearchBestRun()
    top = report.rows[0]
    return ResearchBestRun(
        experiment_id=experiment.experiment_id,
        run_id=top.run_id,
        bmc_last_200_eras_mean=top.bmc_last_200_eras_mean,
        bmc_mean=top.bmc_mean,
        corr_mean=top.corr_mean,
        mmc_mean=top.mmc_mean,
        cwmm_mean=top.cwmm_mean,
        updated_at=utc_now_iso(),
    )


def _safe_report(*, root: Path, experiment_id: str) -> Any | None:
    try:
        return report_experiment(store_root=root, experiment_id=experiment_id, metric=_DEFAULT_PRIMARY_METRIC, limit=10)
    except ExperimentError:
        return None


def _planner_reference_config_dirs(
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
            config_dirs.append(_experiment_config_dir(root=root, experiment_id=experiment_id))
            seen_experiment_ids.add(experiment_id)
        parent_experiment_id = current_path.parent_experiment_id
        if parent_experiment_id is None:
            break
        if parent_experiment_id in seen_experiment_ids:
            break
        parent_path = paths_by_experiment_id.get(parent_experiment_id)
        if parent_path is None:
            config_dirs.append(_experiment_config_dir(root=root, experiment_id=parent_experiment_id))
            seen_experiment_ids.add(parent_experiment_id)
            break
        current_path = parent_path
    if state.program_experiment_id not in seen_experiment_ids:
        config_dirs.append(_experiment_config_dir(root=root, experiment_id=state.program_experiment_id))
    return config_dirs


def _resolve_strategy_definition(strategy_id: str) -> ResearchStrategyDefinition:
    try:
        return get_strategy_definition(strategy_id)
    except ValueError as exc:
        raise AgenticResearchValidationError(str(exc)) from exc


def _apply_strategy_phase_action(
    *,
    state: ResearchProgramState,
    strategy: ResearchStrategyDefinition,
    decision: Any,
) -> ResearchProgramState:
    if not strategy.phase_aware:
        return state
    current_phase = state.current_phase
    if current_phase is None:
        raise AgenticResearchValidationError("agentic_research_phase_state_missing")
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
    next_phase = next_phase_definition(strategy, current_phase.phase_id)
    if next_phase is None:
        raise AgenticResearchValidationError("agentic_research_phase_advance_invalid")
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


def _increment_phase_round_count(current_phase: ResearchPhaseState | None) -> ResearchPhaseState | None:
    if current_phase is None or current_phase.status != "active":
        return current_phase
    return replace(
        current_phase,
        round_count=current_phase.round_count + 1,
        updated_at=utc_now_iso(),
    )


_LEGACY_ROUND_ARTIFACT_FILENAMES = (
    "codex_prompt.txt",
    "codex_usage.json",
    "codex_stdout.jsonl",
    "codex_stderr.txt",
    "codex_last_message.json",
    "codex_last_message.txt",
    "codex_decision.json",
    "planned_configs.json",
    "report.json",
    "round_summary.json",
    "llm_trace.jsonl",
    "llm_trace.md",
    "codex_failure.txt",
    "codex_validation_error.txt",
)


def _planner_attempt_payloads(executions: list[Any]) -> list[dict[str, object]]:
    return [
        {
            "attempt_number": attempt.attempt_number,
            "elapsed_seconds": attempt.elapsed_seconds,
            "returncode": attempt.returncode,
            "thread_id": attempt.thread_id,
            "input_tokens": attempt.input_tokens,
            "cached_input_tokens": attempt.cached_input_tokens,
            "output_tokens": attempt.output_tokens,
            "stdout_line_count": attempt.stdout_line_count,
            "validation_feedback": attempt.validation_feedback,
        }
        for execution in executions
        for attempt in execution.attempts
    ]


def _planner_usage_payload(executions: list[Any]) -> dict[str, object]:
    attempts = _planner_attempt_payloads(executions)
    return {
        "attempts": attempts,
        "final_attempt": attempts[-1] if attempts else None,
        "total_input_tokens": sum(item["input_tokens"] or 0 for item in attempts),
        "total_cached_input_tokens": sum(item["cached_input_tokens"] or 0 for item in attempts),
        "total_output_tokens": sum(item["output_tokens"] or 0 for item in attempts),
        "total_elapsed_seconds": round(sum(float(item["elapsed_seconds"]) for item in attempts), 6),
    }


def _planner_trace_payload(
    *,
    round_state: Any,
    round_artifact_dir: Path,
    prompt_text: str,
    executions: list[Any],
    status: str,
    error: str | None = None,
) -> dict[str, object]:
    source = active_model_source()
    model_name: str | None = None
    if source == "openrouter":
        try:
            model_name = load_openrouter_config().active_model
        except Exception:  # noqa: BLE001
            model_name = None
    attempts = _planner_attempt_payloads(executions)
    parsed_response = None
    if executions:
        last_message = getattr(executions[-1], "last_message", None)
        if isinstance(last_message, dict):
            parsed_response = last_message
    return {
        "timestamp": utc_now_iso(),
        "event": "planner_trace",
        "status": status,
        "planner_source": source,
        "planner_model": model_name if model_name is not None else source,
        "experiment_id": round_state.experiment_id,
        "path_id": round_state.path_id,
        "round_label": round_state.round_label,
        "round_number": round_state.round_number,
        "round_path": str(round_record_path(round_artifact_dir)),
        "round_markdown_path": str(round_markdown_path(round_artifact_dir)),
        "attempts": attempts,
        "usage": _planner_usage_payload(executions),
        "prompt_text": prompt_text,
        "raw_response_text": _trace_response_text(executions[-1]) if executions else "",
        "parsed_response": parsed_response,
        "decision": parsed_response,
        "error": error,
    }


def _append_planner_trace(
    *,
    auto_dir: Path,
    round_artifact_dir: Path,
    round_state: Any,
    prompt_text: str,
    executions: list[Any],
    status: str,
    error: str | None = None,
) -> None:
    payload = _planner_trace_payload(
        round_state=round_state,
        round_artifact_dir=round_artifact_dir,
        prompt_text=prompt_text,
        executions=executions,
        status=status,
        error=error,
    )
    append_jsonl_artifact(llm_trace_path(auto_dir), payload)
    append_text_artifact(llm_trace_markdown_path(auto_dir), _render_planner_trace_markdown(payload))


def _render_planner_trace_markdown(payload: dict[str, object]) -> str:
    prompt_text = str(payload.get("prompt_text") or "")
    raw_response_text = str(payload.get("raw_response_text") or "")
    usage = payload.get("usage")
    stderr_text = ""
    parsed_response = payload.get("parsed_response")
    decision_text = json.dumps(parsed_response, indent=2, sort_keys=True) if isinstance(parsed_response, dict) else ""
    lines = [
        f"## {payload.get('timestamp', '')} {payload.get('round_label', '')}",
        "",
        f"- Status: `{payload.get('status', '')}`",
        f"- Planner source: `{payload.get('planner_source', '')}`",
        f"- Planner model: `{payload.get('planner_model', '')}`",
        (f"- Round record: `{payload.get('round_path', '')}`" if payload.get("round_path") else ""),
        "",
        "### Sent To LLM",
        "",
        "```text",
        prompt_text,
        "```",
        "",
        "### Raw LLM Response",
        "",
        "```text",
        raw_response_text,
        "```",
        "",
        "### Parsed Final Response",
        "",
        "```json",
        decision_text,
        "```",
    ]
    if isinstance(usage, dict):
        lines.extend(
            [
                "",
                "### Usage",
                "",
                "```json",
                json.dumps(usage, indent=2, sort_keys=True),
                "```",
            ]
        )
    if stderr_text:
        lines.extend(["", "### Raw Stderr", "", "```text", stderr_text, "```"])
    error = payload.get("error")
    if error:
        lines.extend(["", "### Error", "", str(error)])
    lines.extend(["", "---", ""])
    return "\n".join(line for line in lines if line != "")


def _run_planner(
    *,
    prompt: str,
    codex_command: list[str],
    schema_path: Path | None,
    artifact_dir: Path,
) -> CodexPlannerExecution:
    if schema_path is None:
        raise AgenticResearchValidationError("agentic_research_schema_missing")
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


def _run_mutation_planner(
    *,
    prompt: str,
    codex_command: list[str],
    artifact_dir: Path,
) -> RawPlannerExecution:
    if active_model_source() == "openrouter":
        return run_openrouter_raw_planner(prompt=prompt)
    return run_codex_raw_planner(prompt=prompt, command=codex_command, artifact_dir=artifact_dir)


def _next_mutation_action(active_path: ResearchPathState) -> str:
    if should_pivot_path(active_path):
        return "pivot"
    if force_action_for_path(active_path) == "scale":
        return "scale"
    return "continue"


def _mutation_experiment_question(parent: ParentConfigSelection) -> str:
    return f"What is the next targeted mutation after `{parent.config_filename}`?"


def _mutation_winner_criteria() -> str:
    return "Improve bmc_last_200_eras_mean, use bmc_mean as the tie-break, and sanity-check corr_mean."


def _trace_response_text(execution: Any) -> str:
    raw_response_text = getattr(execution, "raw_response_text", "")
    if isinstance(raw_response_text, str) and raw_response_text.strip():
        return raw_response_text
    stdout_jsonl = getattr(execution, "stdout_jsonl", "")
    if isinstance(stdout_jsonl, str) and stdout_jsonl.strip():
        return stdout_jsonl
    if isinstance(raw_response_text, str):
        return raw_response_text
    return ""


def _round_results_payload(
    *,
    report: Any,
    round_state: Any,
    improved_best_overall: bool,
) -> dict[str, object]:
    summary = round_summary_from_report(report=report, run_ids=round_state.run_ids, round_state=round_state)
    return {
        "metric": report.metric,
        "total_runs": report.total_runs,
        "champion_run_id": report.champion_run_id,
        "improved_best_overall": improved_best_overall,
        "best_row": summary.get("best_row"),
        "rows": summary.get("rows"),
    }


def _save_round_bundle(
    *,
    round_artifact_dir: Path,
    round_state: Any,
    planner_payload: dict[str, object] | None = None,
    results_payload: dict[str, object] | None = None,
) -> None:
    existing = _load_round_bundle(round_artifact_dir)
    if planner_payload is None and isinstance(existing.get("planner"), dict):
        planner_payload = existing.get("planner")
    if results_payload is None and isinstance(existing.get("results"), dict):
        results_payload = existing.get("results")
    payload = {
        "round_number": round_state.round_number,
        "round_label": round_state.round_label,
        "experiment_id": round_state.experiment_id,
        "path_id": round_state.path_id,
        "status": round_state.status,
        "started_at": round_state.started_at,
        "updated_at": round_state.updated_at,
        "decision": {
            "action": round_state.decision_action,
            "experiment_question": round_state.experiment_question,
            "winner_criteria": round_state.winner_criteria,
            "decision_rationale": round_state.decision_rationale,
            "path_hypothesis": round_state.decision_path_hypothesis,
            "path_slug": round_state.decision_path_slug,
            "phase_id": round_state.phase_id,
            "phase_action": round_state.phase_action,
            "phase_transition_rationale": round_state.phase_transition_rationale,
        },
        "lineage": {
            "parent_run_id": round_state.parent_run_id,
            "parent_config_filename": round_state.parent_config_filename,
            "child_config_filenames": list(round_state.config_filenames),
            "change_set": list(round_state.change_set),
            "llm_rationale": round_state.llm_rationale,
        },
        "execution": {
            "run_ids": list(round_state.run_ids),
            "next_config_index": round_state.next_config_index,
        },
        "planner": planner_payload,
        "results": results_payload,
    }
    save_round_artifact(round_record_path(round_artifact_dir), payload)
    save_text_artifact(round_markdown_path(round_artifact_dir), _render_round_markdown(payload))
    _cleanup_legacy_round_artifacts(round_artifact_dir)


def _load_round_bundle(round_artifact_dir: Path) -> dict[str, object]:
    path = round_record_path(round_artifact_dir)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _render_round_markdown(payload: dict[str, object]) -> str:
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    lineage = payload.get("lineage") if isinstance(payload.get("lineage"), dict) else {}
    execution = payload.get("execution") if isinstance(payload.get("execution"), dict) else {}
    planner = payload.get("planner") if isinstance(payload.get("planner"), dict) else {}
    results = payload.get("results") if isinstance(payload.get("results"), dict) else {}
    lines = [
        f"# {payload.get('round_label', '')}",
        "",
        f"- Status: `{payload.get('status', '')}`",
        f"- Experiment: `{payload.get('experiment_id', '')}`",
        f"- Path: `{payload.get('path_id', '')}`",
        f"- Decision action: `{decision.get('action') or 'n/a'}`",
    ]
    parent_config = lineage.get("parent_config_filename")
    if parent_config:
        lines.append(f"- Parent config: `{parent_config}`")
    parent_run_id = lineage.get("parent_run_id")
    if parent_run_id:
        lines.append(f"- Parent run: `{parent_run_id}`")
    child_configs = lineage.get("child_config_filenames")
    if isinstance(child_configs, list) and child_configs:
        lines.append(f"- Child config(s): `{', '.join(str(item) for item in child_configs)}`")
    run_ids = execution.get("run_ids")
    if isinstance(run_ids, list) and run_ids:
        lines.append(f"- Run id(s): `{', '.join(str(item) for item in run_ids)}`")
    results_best = results.get("best_row") if isinstance(results.get("best_row"), dict) else None
    if results_best is not None:
        lines.extend(
            [
                "",
                "## Outcome",
                "",
                f"- Improved best overall: `{results.get('improved_best_overall')}`",
                f"- Best run: `{results_best.get('run_id')}`",
                f"- `bmc_last_200_eras_mean`: `{results_best.get('bmc_last_200_eras_mean')}`",
                f"- `bmc_mean`: `{results_best.get('bmc_mean')}`",
                f"- `corr_mean`: `{results_best.get('corr_mean')}`",
            ]
        )
    change_set = lineage.get("change_set")
    if isinstance(change_set, list) and change_set:
        lines.extend(["", "## Change Set", ""])
        for item in change_set:
            if not isinstance(item, dict):
                continue
            lines.append(f"- `{item.get('path')}` = `{json.dumps(item.get('value'), sort_keys=True)}`")
    llm_rationale = lineage.get("llm_rationale")
    if isinstance(llm_rationale, str) and llm_rationale.strip():
        lines.extend(["", "## LLM Rationale", "", llm_rationale.strip()])
    prompt_text = planner.get("prompt_text")
    if isinstance(prompt_text, str) and prompt_text.strip():
        lines.extend(["", "## Sent To LLM", "", "```text", prompt_text.strip(), "```"])
    raw_response_text = planner.get("raw_response_text")
    if isinstance(raw_response_text, str) and raw_response_text.strip():
        lines.extend(["", "## Raw LLM Response", "", "```text", raw_response_text.strip(), "```"])
    parsed_response = planner.get("parsed_response")
    if isinstance(parsed_response, dict):
        lines.extend(
            [
                "",
                "## Parsed Final Response",
                "",
                "```json",
                json.dumps(parsed_response, indent=2, sort_keys=True),
                "```",
            ]
        )
    usage = planner.get("usage")
    if isinstance(usage, dict):
        lines.extend(["", "## Usage", "", "```json", json.dumps(usage, indent=2, sort_keys=True), "```"])
    planner_error = planner.get("error")
    if planner_error:
        lines.extend(["", "## Error", "", str(planner_error)])
    return "\n".join(lines) + "\n"


def _cleanup_legacy_round_artifacts(round_artifact_dir: Path) -> None:
    for filename in _LEGACY_ROUND_ARTIFACT_FILENAMES:
        path = round_artifact_dir / filename
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            continue


def _recent_round_summaries(*, auto_dir: Path) -> list[dict[str, object]]:
    rounds_root = auto_dir / "rounds"
    if not rounds_root.is_dir():
        return []
    items: list[dict[str, object]] = []
    for path in sorted(rounds_root.iterdir(), key=lambda item: item.name):
        record_path = round_record_path(path)
        if record_path.is_file():
            try:
                payload = json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict):
                items.append(_round_summary_view(payload))
                continue
        legacy_summary_path = path / "round_summary.json"
        if not legacy_summary_path.is_file():
            continue
        try:
            payload = json.loads(legacy_summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            items.append(payload)
    return items


def _round_summary_view(payload: dict[str, object]) -> dict[str, object]:
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    lineage = payload.get("lineage") if isinstance(payload.get("lineage"), dict) else {}
    results = payload.get("results") if isinstance(payload.get("results"), dict) else {}
    return {
        "round_label": payload.get("round_label"),
        "experiment_id": payload.get("experiment_id"),
        "path_id": payload.get("path_id"),
        "decision_action": decision.get("action"),
        "experiment_question": decision.get("experiment_question"),
        "winner_criteria": decision.get("winner_criteria"),
        "decision_rationale": decision.get("decision_rationale"),
        "parent_run_id": lineage.get("parent_run_id"),
        "parent_config_filename": lineage.get("parent_config_filename"),
        "change_set": lineage.get("change_set") if isinstance(lineage.get("change_set"), list) else [],
        "llm_rationale": lineage.get("llm_rationale"),
        "best_row": results.get("best_row") if isinstance(results.get("best_row"), dict) else None,
    }


def _agentic_research_dir(experiment: ExperimentRecord) -> Path:
    return experiment.manifest_path.parent / _AGENTIC_RESEARCH_DIRNAME


def _replace_path(state: ResearchProgramState, updated_path: ResearchPathState) -> ResearchProgramState:
    new_paths = [updated_path if item.path_id == updated_path.path_id else item for item in state.paths]
    return replace(state, paths=new_paths, updated_at=utc_now_iso())


def _require_active_path(state: ResearchProgramState) -> ResearchPathState:
    for item in state.paths:
        if item.path_id == state.active_path_id:
            return item
    raise AgenticResearchValidationError("agentic_research_active_path_missing")


def _stop_program(state: ResearchProgramState, *, reason: str) -> ResearchProgramState:
    return replace(
        state,
        status="stopped",
        stop_reason=reason,
        last_checkpoint="stopped",
        updated_at=utc_now_iso(),
    )


def _experiment_config_dir(*, root: Path, experiment_id: str) -> Path:
    record = get_experiment(store_root=root, experiment_id=experiment_id)
    return record.manifest_path.parent / "configs"


def _upsert_agentic_research_metadata(
    *,
    root: Path,
    experiment: ExperimentRecord,
    metadata_update: dict[str, object],
) -> None:
    manifest_path = experiment.manifest_path
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgenticResearchValidationError(f"agentic_research_manifest_invalid:{manifest_path}") from exc
    if not isinstance(payload, dict):
        raise AgenticResearchValidationError(f"agentic_research_manifest_invalid:{manifest_path}")
    metadata = payload.get("metadata")
    normalized: dict[str, object] = metadata if isinstance(metadata, dict) else {}
    normalized["agentic_research"] = metadata_update
    payload["metadata"] = normalized
    payload["updated_at"] = utc_now_iso()
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    try:
        upsert_experiment(
            store_root=root,
            experiment_id=experiment.experiment_id,
            name=experiment.name,
            status=str(payload.get("status", experiment.status)),
            created_at=str(payload.get("created_at", experiment.created_at)),
            updated_at=str(payload["updated_at"]),
            metadata={
                **normalized,
                "hypothesis": payload.get("hypothesis"),
                "tags": list(experiment.tags),
                "champion_run_id": payload.get("champion_run_id"),
                "runs": list(experiment.runs),
            },
        )
    except StoreError as exc:
        raise AgenticResearchValidationError("agentic_research_metadata_index_failed") from exc
