"""Agentic research supervisor services."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from numereng.features.agentic_research.utils.llm import default_codex_command
from numereng.features.agentic_research.utils.planning import (
    build_round_state,
    experiment_config_dir,
    increment_phase_round_count,
    pivot_to_child_path,
    plan_mutation_round,
    plan_structured_round,
    replace_path_state,
    require_active_path,
    round_summary_from_report,
    seed_best_run,
    select_best_row,
    should_pivot_path,
    stop_program,
    update_best_overall,
    update_path_after_round,
)
from numereng.features.agentic_research.utils.programs import (
    initial_phase_state,
    list_program_catalog,
    load_program_details,
    program_markdown_sha256,
)
from numereng.features.agentic_research.utils.store import (
    agentic_research_dir,
    ensure_agentic_research_dirs,
    lineage_path,
    load_and_persist_program_state,
    load_lineage_state,
    program_path,
    round_dir,
    save_lineage_state,
    save_program_state,
    save_round_bundle,
    save_text_artifact,
    session_program_path,
    upsert_agentic_research_metadata,
    utc_now_iso,
)
from numereng.features.agentic_research.utils.types import (
    ResearchInitResult,
    ResearchLineageLink,
    ResearchLineageState,
    ResearchPathState,
    ResearchProgramCatalogEntry,
    ResearchProgramDetails,
    ResearchProgramState,
    ResearchRunResult,
    ResearchStatusResult,
)
from numereng.features.experiments import (
    get_experiment,
    report_experiment,
    score_experiment_round,
    train_experiment,
)
from numereng.features.store import resolve_store_root, resolve_workspace_layout_from_store_root
from numereng.features.telemetry import bind_launch_metadata

_DEFAULT_CODEX_COMMAND = default_codex_command()
_INITIAL_PATH_ID = "p00"


class AgenticResearchError(Exception):
    """Base error for agentic research workflows."""


class AgenticResearchNotInitializedError(AgenticResearchError):
    """Raised when the supervisor state has not been initialized."""


class AgenticResearchValidationError(AgenticResearchError):
    """Raised when agentic research inputs or state are invalid."""


def list_research_programs(*, user_programs_dir: str | Path | None = None) -> tuple[ResearchProgramCatalogEntry, ...]:
    """Return the merged research program catalog."""
    try:
        return list_program_catalog(user_dir=user_programs_dir)
    except ValueError as exc:
        raise AgenticResearchValidationError(str(exc)) from exc


def get_research_program(
    program_id: str,
    *,
    user_programs_dir: str | Path | None = None,
) -> ResearchProgramDetails:
    """Load one research program from the merged catalog."""
    try:
        return load_program_details(program_id, user_dir=user_programs_dir)
    except ValueError as exc:
        raise AgenticResearchValidationError(str(exc)) from exc


def init_research(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str,
    program_id: str,
    improvement_threshold: float | None = None,
) -> ResearchInitResult:
    """Initialize one agentic research supervisor rooted at one experiment."""
    root = resolve_store_root(store_root)
    workspace_layout = resolve_workspace_layout_from_store_root(root)
    experiment = get_experiment(store_root=root, experiment_id=experiment_id)
    if experiment.status == "archived":
        raise AgenticResearchValidationError("agentic_research_root_experiment_archived")
    user_programs_dir = os.getenv("NUMERENG_AGENTIC_RESEARCH_PROGRAMS_DIR") or workspace_layout.research_programs_root
    selected = get_research_program(program_id, user_programs_dir=user_programs_dir)
    threshold = improvement_threshold or selected.definition.improvement_threshold_default
    if threshold <= 0.0:
        raise AgenticResearchValidationError("agentic_research_improvement_threshold_invalid")

    auto_dir = agentic_research_dir(experiment)
    ensure_agentic_research_dirs(auto_dir)
    program_state_path = program_path(auto_dir)
    lineage_state_path = lineage_path(auto_dir)
    session_source_path = session_program_path(auto_dir)
    if program_state_path.is_file() and lineage_state_path.is_file():
        state = load_and_persist_program_state(auto_dir=auto_dir)
        return _init_result_from_state(state=state, auto_dir=auto_dir)

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
    program = ResearchProgramState(
        root_experiment_id=experiment.experiment_id,
        program_experiment_id=experiment.experiment_id,
        program_id=selected.definition.program_id,
        program_title=selected.definition.title,
        program_source=selected.definition.source,
        program_sha256=program_markdown_sha256(selected.raw_markdown),
        program_snapshot=selected.definition,
        status="initialized",
        active_path_id=root_path.path_id,
        active_experiment_id=experiment.experiment_id,
        next_round_number=1,
        total_rounds_completed=0,
        total_paths_created=1,
        improvement_threshold=threshold,
        scoring_stage=selected.definition.scoring_stage,
        codex_command=list(_DEFAULT_CODEX_COMMAND),
        last_checkpoint="initialized",
        stop_reason=None,
        current_round=None,
        current_phase=initial_phase_state(selected.definition, now_iso=now),
        best_overall=seed_best_run(
            root=root,
            experiment=experiment,
            primary_metric=selected.definition.metric_policy.primary,
        ),
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
    save_text_artifact(session_source_path, selected.raw_markdown)
    try:
        upsert_agentic_research_metadata(
            root=root,
            experiment=experiment,
            metadata_update={
                "root_experiment_id": experiment.experiment_id,
                "program_experiment_id": experiment.experiment_id,
                "program_id": program.program_id,
                "program_title": program.program_title,
                "program_source": program.program_source,
                "parent_experiment_id": None,
                "path_id": root_path.path_id,
                "pivot_reason": None,
                "source_round": None,
                "generation": 0,
            },
        )
    except ValueError as exc:
        raise AgenticResearchValidationError(str(exc)) from exc
    return _init_result_from_state(state=program, auto_dir=auto_dir)


def get_research_status(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str,
) -> ResearchStatusResult:
    """Load the current supervisor status payload."""
    auto_dir = agentic_research_dir(get_experiment(store_root=store_root, experiment_id=experiment_id))
    state = load_and_persist_program_state(auto_dir=auto_dir)
    return _status_result_from_state(state=state, auto_dir=auto_dir)


def run_research(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str,
    max_rounds: int | None = None,
    max_paths: int | None = None,
) -> ResearchRunResult:
    """Run the supervisor loop in the foreground until stopped or interrupted."""
    root = resolve_store_root(store_root)
    auto_dir = agentic_research_dir(get_experiment(store_root=root, experiment_id=experiment_id))
    program_state_path = program_path(auto_dir)
    lineage_state_path = lineage_path(auto_dir)
    if not program_state_path.is_file():
        raise AgenticResearchNotInitializedError(f"agentic_research_not_initialized:{experiment_id}")
    state = load_and_persist_program_state(auto_dir=auto_dir)
    lineage = load_lineage_state(lineage_state_path)
    rounds_started = 0
    resumed_round_active = state.current_round is not None

    try:
        while True:
            if max_rounds is not None and state.current_round is None and rounds_started >= max_rounds:
                state = stop_program(state, reason="max_rounds_reached")
                save_program_state(program_state_path, state)
                break

            active_path = require_active_path(state)
            if should_pivot_path(active_path, round_policy=state.program_snapshot.round_policy):
                if max_paths is not None and state.total_paths_created >= max_paths:
                    state = stop_program(state, reason="max_paths_reached")
                    save_program_state(program_state_path, state)
                    break
                state, lineage = pivot_to_child_path(
                    root=root,
                    state=state,
                    lineage=lineage,
                    active_path=active_path,
                )
                save_program_state(program_state_path, state)
                save_lineage_state(lineage_state_path, lineage)

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
                state = load_and_persist_program_state(auto_dir=auto_dir)
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
        return _run_result_from_state(state=state, interrupted=True)

    return _run_result_from_state(state=state, interrupted=False)


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
    active_path = require_active_path(state)
    definition = state.program_snapshot
    experiment = get_experiment(store_root=root, experiment_id=state.active_experiment_id)
    round_artifact_dir = round_dir(auto_dir, round_state.round_label)
    round_artifact_dir.mkdir(parents=True, exist_ok=True)

    if round_state.status == "planning":
        planner = plan_mutation_round if definition.planner_contract == "config_mutation" else plan_structured_round
        state, lineage, stop_signal = planner(
            root=root,
            state=state,
            lineage=lineage,
            auto_dir=auto_dir,
            definition=definition,
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
        config_dir = experiment_config_dir(root=root, experiment_id=round_state.experiment_id)
        for idx in range(round_state.next_config_index, len(round_state.config_filenames)):
            config_path = config_dir / round_state.config_filenames[idx]
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
            save_round_bundle(
                round_artifact_dir=round_artifact_dir,
                round_state=round_state,
                program_state=state,
                session_source_path=session_program_path(auto_dir),
            )
        with bind_launch_metadata(source="feature.agentic_research.score_round", operation_type="run", job_type="run"):
            score_experiment_round(
                store_root=root,
                experiment_id=round_state.experiment_id,
                round=round_state.round_label,
                stage=definition.scoring_stage,
            )
        round_state = replace(round_state, status="scored", updated_at=utc_now_iso())
        state = replace(state, current_round=round_state, last_checkpoint="round_scored", updated_at=utc_now_iso())
        save_round_bundle(
            round_artifact_dir=round_artifact_dir,
            round_state=round_state,
            program_state=state,
            session_source_path=session_program_path(auto_dir),
        )

    round_state = state.current_round
    if round_state is None or round_state.status != "scored":
        return state, lineage, None

    report = report_experiment(
        store_root=root,
        experiment_id=round_state.experiment_id,
        metric=definition.metric_policy.primary,
        limit=1000,
    )
    round_rows = [row for row in report.rows if row.run_id in set(round_state.run_ids)]
    best_row = select_best_row(
        round_rows,
        primary_metric=definition.metric_policy.primary,
        tie_break_metric=definition.metric_policy.tie_break,
    )
    best_overall, improved = update_best_overall(
        current_best=state.best_overall,
        round_best=best_row,
        experiment_id=round_state.experiment_id,
        threshold=state.improvement_threshold,
        primary_metric=definition.metric_policy.primary,
    )
    updated_path = update_path_after_round(
        path=require_active_path(state),
        round_best=best_row,
        improved=improved,
        round_policy=definition.round_policy,
    )
    summary = round_summary_from_report(
        report=report,
        run_ids=round_state.run_ids,
        round_state=round_state,
        primary_metric=definition.metric_policy.primary,
        tie_break_metric=definition.metric_policy.tie_break,
    )
    save_round_bundle(
        round_artifact_dir=round_artifact_dir,
        round_state=round_state,
        program_state=state,
        session_source_path=session_program_path(auto_dir),
        results_payload={
            "metric": report.metric,
            "total_runs": report.total_runs,
            "champion_run_id": report.champion_run_id,
            "improved_best_overall": improved,
            "best_row": summary.get("best_row"),
            "rows": summary.get("rows"),
        },
    )
    state = replace_path_state(state, updated_path)
    state = replace(
        state,
        best_overall=best_overall,
        total_rounds_completed=state.total_rounds_completed + 1,
        next_round_number=state.next_round_number + 1,
        current_round=None,
        current_phase=increment_phase_round_count(state.current_phase),
        last_checkpoint="round_completed",
        updated_at=utc_now_iso(),
    )
    return state, lineage, None


def _init_result_from_state(*, state: ResearchProgramState, auto_dir: Path) -> ResearchInitResult:
    return ResearchInitResult(
        root_experiment_id=state.root_experiment_id,
        program_id=state.program_id,
        program_title=state.program_title,
        status=state.status,
        active_experiment_id=state.active_experiment_id,
        active_path_id=state.active_path_id,
        improvement_threshold=state.improvement_threshold,
        current_phase=state.current_phase,
        agentic_research_dir=auto_dir,
        program_path=program_path(auto_dir),
        lineage_path=lineage_path(auto_dir),
        session_program_path=session_program_path(auto_dir),
    )


def _status_result_from_state(*, state: ResearchProgramState, auto_dir: Path) -> ResearchStatusResult:
    return ResearchStatusResult(
        root_experiment_id=state.root_experiment_id,
        program_id=state.program_id,
        program_title=state.program_title,
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
        session_program_path=session_program_path(auto_dir),
    )


def _run_result_from_state(*, state: ResearchProgramState, interrupted: bool) -> ResearchRunResult:
    return ResearchRunResult(
        root_experiment_id=state.root_experiment_id,
        program_id=state.program_id,
        program_title=state.program_title,
        status=state.status,
        active_experiment_id=state.active_experiment_id,
        active_path_id=state.active_path_id,
        next_round_number=state.next_round_number,
        total_rounds_completed=state.total_rounds_completed,
        total_paths_created=state.total_paths_created,
        last_checkpoint=state.last_checkpoint,
        stop_reason=state.stop_reason,
        current_phase=state.current_phase,
        interrupted=interrupted,
    )
