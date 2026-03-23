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
)
from numereng.features.agentic_research.contracts import (
    CodexPlannerExecution,
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
)
from numereng.features.agentic_research.planner import (
    build_round_state,
    child_experiment_id,
    force_action_for_path,
    materialize_config_payload,
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
    round_dir,
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
        report = _safe_report(root=root, experiment_id=state.active_experiment_id)
        recent_rounds = _recent_round_summaries(auto_dir=auto_dir)
        forced_action = force_action_for_path(active_path)
        base_config_snapshot, base_config_source, valid_config_examples = select_reference_configs(
            config_dirs=_planner_reference_config_dirs(root=root, state=state, active_path=active_path)
        )
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
        )
        save_text_artifact(round_artifact_dir / "codex_prompt.txt", prompt)
        validation_feedback: str | None = None
        executions: list[CodexPlannerExecution] = []
        materialized_decision_configs: list[dict[str, object]] = []
        try:
            execution = _run_planner(
                prompt=prompt,
                codex_command=state.codex_command,
                schema_path=strategy_definition.schema_path,
                artifact_dir=round_artifact_dir,
                validation_feedback=validation_feedback,
            )
            executions.append(execution)
            decision = execution.decision
            validate_decision(decision, strategy=strategy_definition, current_phase=state.current_phase)
            materialized_decision_configs = _materialized_decision_configs(
                base_config=base_config_snapshot,
                decision=decision,
            )
        except (AgenticResearchCodexError, AgenticResearchOpenRouterError, ValueError, json.JSONDecodeError) as exc:
            retry_feedback = str(exc)
            save_text_artifact(round_artifact_dir / "codex_validation_error.txt", retry_feedback)
            try:
                execution = _run_planner(
                    prompt=prompt,
                    codex_command=state.codex_command,
                    schema_path=strategy_definition.schema_path,
                    artifact_dir=round_artifact_dir,
                    validation_feedback=retry_feedback,
                )
                executions.append(execution)
                decision = execution.decision
                validate_decision(decision, strategy=strategy_definition, current_phase=state.current_phase)
                materialized_decision_configs = _materialized_decision_configs(
                    base_config=base_config_snapshot,
                    decision=decision,
                )
            except (
                AgenticResearchCodexError,
                AgenticResearchOpenRouterError,
                ValueError,
                json.JSONDecodeError,
            ) as retry_exc:
                if executions:
                    _save_codex_execution_artifacts(round_artifact_dir=round_artifact_dir, executions=executions)
                save_text_artifact(round_artifact_dir / "codex_failure.txt", str(retry_exc))
                _append_planner_trace(
                    auto_dir=auto_dir,
                    round_artifact_dir=round_artifact_dir,
                    round_state=round_state,
                    executions=executions,
                    status="failed",
                    error=str(retry_exc),
                )
                stopped_state = _stop_program(state, reason="codex_planning_failed")
                return stopped_state, lineage, "stop"
        _save_codex_execution_artifacts(round_artifact_dir=round_artifact_dir, executions=executions)
        save_round_artifact(
            round_artifact_dir / "codex_decision.json",
            {
                "experiment_question": decision.experiment_question,
                "winner_criteria": decision.winner_criteria,
                "decision_rationale": decision.decision_rationale,
                "next_action": decision.next_action,
                "path_hypothesis": decision.path_hypothesis,
                "path_slug": decision.path_slug,
                "phase_action": decision.phase_action,
                "phase_transition_rationale": decision.phase_transition_rationale,
                "configs": materialized_decision_configs,
            },
        )
        _append_planner_trace(
            auto_dir=auto_dir,
            round_artifact_dir=round_artifact_dir,
            round_state=round_state,
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
            state = _stop_program(state, reason="codex_requested_stop")
            return state, lineage, "stop"
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
        save_round_artifact(
            round_artifact_dir / "planned_configs.json",
            {
                "experiment_id": round_state.experiment_id,
                "filenames": filenames,
            },
        )
        path_after_plan = active_path
        if force_action_for_path(active_path) == "scale":
            path_after_plan = replace(
                active_path,
                scale_confirmation_used=True,
                needs_scale_confirmation=False,
                updated_at=utc_now_iso(),
            )
            state = _replace_path(state, path_after_plan)
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
        state = replace(
            state,
            current_round=round_state,
            last_checkpoint="round_planned",
            updated_at=utc_now_iso(),
        )

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
        with bind_launch_metadata(source="feature.agentic_research.score_round", operation_type="run", job_type="run"):
            score_experiment_round(
                store_root=root,
                experiment_id=round_state.experiment_id,
                round=round_state.round_label,
                stage=state.scoring_stage,
            )
        round_state = replace(round_state, status="scored", updated_at=utc_now_iso())
        state = replace(state, current_round=round_state, last_checkpoint="round_scored", updated_at=utc_now_iso())

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
        summary = round_summary_from_report(report=report, run_ids=round_state.run_ids, round_state=round_state)
        summary["improved_best_overall"] = improved
        save_round_artifact(round_artifact_dir / "round_summary.json", summary)
        save_round_artifact(
            round_artifact_dir / "report.json",
            {
                "experiment_id": report.experiment_id,
                "metric": report.metric,
                "total_runs": report.total_runs,
                "champion_run_id": report.champion_run_id,
                "rows": [
                    {
                        "run_id": item.run_id,
                        "metric_value": item.metric_value,
                        "corr_mean": item.corr_mean,
                        "mmc_mean": item.mmc_mean,
                        "cwmm_mean": item.cwmm_mean,
                        "bmc_mean": item.bmc_mean,
                        "bmc_last_200_eras_mean": item.bmc_last_200_eras_mean,
                        "is_champion": item.is_champion,
                    }
                    for item in report.rows
                ],
            },
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
    config_dirs = [_experiment_config_dir(root=root, experiment_id=state.active_experiment_id)]
    if active_path.parent_experiment_id is not None:
        config_dirs.append(_experiment_config_dir(root=root, experiment_id=active_path.parent_experiment_id))
    if state.program_experiment_id not in {state.active_experiment_id, active_path.parent_experiment_id}:
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


def _save_codex_execution_artifacts(
    *,
    round_artifact_dir: Path,
    executions: list[CodexPlannerExecution],
) -> None:
    if not executions:
        return
    attempts = [
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
    save_round_artifact(
        round_artifact_dir / "codex_usage.json",
        {
            "attempts": attempts,
            "final_attempt": attempts[-1] if attempts else None,
            "total_input_tokens": sum(item["input_tokens"] or 0 for item in attempts),
            "total_cached_input_tokens": sum(item["cached_input_tokens"] or 0 for item in attempts),
            "total_output_tokens": sum(item["output_tokens"] or 0 for item in attempts),
            "total_elapsed_seconds": round(sum(float(item["elapsed_seconds"]) for item in attempts), 6),
        },
    )
    save_text_artifact(
        round_artifact_dir / "codex_stdout.jsonl",
        "".join(execution.stdout_jsonl for execution in executions),
    )
    save_text_artifact(
        round_artifact_dir / "codex_stderr.txt",
        "\n".join(execution.stderr_text for execution in executions if execution.stderr_text).strip(),
    )
    save_round_artifact(round_artifact_dir / "codex_last_message.json", executions[-1].last_message)


def _planner_trace_payload(
    *,
    round_state: Any,
    round_artifact_dir: Path,
    executions: list[CodexPlannerExecution],
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
    attempts = [
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
        "prompt_path": str(round_artifact_dir / "codex_prompt.txt"),
        "usage_path": str(round_artifact_dir / "codex_usage.json"),
        "stdout_path": str(round_artifact_dir / "codex_stdout.jsonl"),
        "stderr_path": str(round_artifact_dir / "codex_stderr.txt"),
        "last_message_path": str(round_artifact_dir / "codex_last_message.json"),
        "decision_path": str(round_artifact_dir / "codex_decision.json"),
        "attempts": attempts,
        "decision": executions[-1].last_message if executions else None,
        "error": error,
    }


def _append_planner_trace(
    *,
    auto_dir: Path,
    round_artifact_dir: Path,
    round_state: Any,
    executions: list[CodexPlannerExecution],
    status: str,
    error: str | None = None,
) -> None:
    payload = _planner_trace_payload(
        round_state=round_state,
        round_artifact_dir=round_artifact_dir,
        executions=executions,
        status=status,
        error=error,
    )
    append_jsonl_artifact(llm_trace_path(auto_dir), payload)
    append_jsonl_artifact(round_artifact_dir / "llm_trace.jsonl", payload)
    append_text_artifact(llm_trace_markdown_path(auto_dir), _render_planner_trace_markdown(payload))
    append_text_artifact(round_artifact_dir / "llm_trace.md", _render_planner_trace_markdown(payload))


def _render_planner_trace_markdown(payload: dict[str, object]) -> str:
    prompt_text = _read_trace_text(payload.get("prompt_path"))
    raw_response_text = _read_trace_text(payload.get("stdout_path"))
    stderr_text = _read_trace_text(payload.get("stderr_path"))
    decision = payload.get("decision")
    decision_text = json.dumps(decision, indent=2, sort_keys=True) if isinstance(decision, dict) else ""
    lines = [
        f"## {payload.get('timestamp', '')} {payload.get('round_label', '')}",
        "",
        f"- Status: `{payload.get('status', '')}`",
        f"- Planner source: `{payload.get('planner_source', '')}`",
        f"- Planner model: `{payload.get('planner_model', '')}`",
        "",
        "### Sent To LLM",
        "",
        "```text",
        prompt_text,
        "```",
        "",
        "### Raw LLM Response",
        "",
        "```jsonl",
        raw_response_text,
        "```",
        "",
        "### Parsed Final Response",
        "",
        "```json",
        decision_text,
        "```",
    ]
    if stderr_text:
        lines.extend(["", "### Raw Stderr", "", "```text", stderr_text, "```"])
    error = payload.get("error")
    if error:
        lines.extend(["", "### Error", "", str(error)])
    lines.extend(["", "---", ""])
    return "\n".join(lines)


def _read_trace_text(path_value: object) -> str:
    if not isinstance(path_value, str) or not path_value.strip():
        return ""
    path = Path(path_value)
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _run_planner(
    *,
    prompt: str,
    codex_command: list[str],
    schema_path: Path,
    artifact_dir: Path,
    validation_feedback: str | None,
) -> CodexPlannerExecution:
    source = active_model_source()
    if source == "openrouter":
        return run_openrouter_planner(
            prompt=prompt,
            schema_path=schema_path,
            artifact_dir=artifact_dir,
            validation_feedback=validation_feedback,
        )
    return run_codex_planner(
        prompt=prompt,
        command=codex_command,
        schema_path=schema_path,
        artifact_dir=artifact_dir,
        validation_feedback=validation_feedback,
    )


def _materialized_decision_configs(
    *,
    base_config: dict[str, object],
    decision: Any,
) -> list[dict[str, object]]:
    return [
        {
            "filename": item.filename,
            "rationale": item.rationale,
            "overrides": item.overrides,
            "materialized_config": materialize_config_payload(
                base_config=base_config,
                overrides=item.overrides,
            ),
        }
        for item in decision.configs
    ]


def _recent_round_summaries(*, auto_dir: Path) -> list[dict[str, object]]:
    rounds_root = auto_dir / "rounds"
    if not rounds_root.is_dir():
        return []
    items: list[dict[str, object]] = []
    for path in sorted(rounds_root.iterdir(), key=lambda item: item.name):
        summary_path = path / "round_summary.json"
        if not summary_path.is_file():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            items.append(payload)
    return items


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
