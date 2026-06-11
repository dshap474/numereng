"""Session lifecycle and round driver. Holds the five seams as module globals (monkeypatch anchor)."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

from numereng.features.agentic_research import boundary, context, llm, memory
from numereng.features.agentic_research.context import _safe_report
from numereng.features.agentic_research.llm import _call_research_llm
from numereng.features.agentic_research.types import (
    CONSECUTIVE_FAILURE_BAIL_THRESHOLD,
    PRIMARY_METRIC_FIELD,
    PROGRAM_PATH,
    SCORING_STAGE,
    STATE_FILENAME,
    AgenticResearchDuplicateCandidate,
    AgenticResearchValidationError,
    ResearchAction,
    ResearchBestRun,
    ResearchRoundResult,
    ResearchRunResult,
    ResearchStatusResult,
    as_int,
    optional_float,
    optional_str,
    status_value,
    utc_now_iso,
)
from numereng.features.experiments import (
    ExperimentRecord,
    get_experiment,
    score_experiment_round,
    train_experiment,
)
from numereng.features.store import index_run, resolve_store_root
from numereng.features.telemetry import bind_launch_metadata
from numereng.features.training.errors import TrainingError

# Seam globals. The contract suite monkeypatches these five names ON THIS MODULE, and
# every round calls them bare through loop's globals so the patches bite.
__all__ = ["get_research_status", "program_markdown", "run_research"]


def program_markdown() -> str:
    """Return the active default research prompt."""
    return PROGRAM_PATH.read_text(encoding="utf-8")


def get_research_status(*, store_root: str | Path = ".numereng", experiment_id: str) -> ResearchStatusResult:
    """Return current research-loop state, synthesizing a blank state if none exists."""
    root = resolve_store_root(store_root)
    experiment = get_experiment(store_root=root, experiment_id=experiment_id)
    state = memory.load_state(memory.state_path(experiment)) or memory.initial_state(experiment)
    best = context.best_run_from_report(_safe_report(root=root, experiment_id=experiment.experiment_id))
    return _status_result(experiment=experiment, state=state, best=best)


def run_research(*, store_root: str | Path = ".numereng", experiment_id: str, max_rounds: int = 1) -> ResearchRunResult:
    """Run one or more config-mutation research rounds in the foreground."""
    if max_rounds < 1:
        raise AgenticResearchValidationError("agentic_research_max_rounds_invalid")
    root = resolve_store_root(store_root)
    experiment = get_experiment(store_root=root, experiment_id=experiment_id)
    if experiment.status == "archived":
        raise AgenticResearchValidationError("agentic_research_experiment_archived")
    boundary.assert_scoring_paths_frozen(experiment)  # invariant 2: evaluator is read-only

    memory.agentic_dir(experiment).mkdir(parents=True, exist_ok=True)
    state = memory.load_state(memory.state_path(experiment)) or memory.initial_state(experiment)
    state["status"] = "running"
    state["stop_reason"] = None
    memory.heartbeat(state)
    memory.save_state(experiment, state)

    rounds: list[ResearchRoundResult] = []
    try:
        for _ in range(max_rounds):
            if _is_terminal_stop(state):
                break
            try:
                result = _run_one_round(root=root, experiment_id=experiment.experiment_id, state=state)
            except (KeyboardInterrupt, SystemExit):
                raise
            except AgenticResearchDuplicateCandidate as exc:
                result = _record_duplicate_skip_round(experiment=experiment, state=state, error=exc)
            except Exception as exc:
                result = _record_failed_round(experiment=experiment, state=state, error=exc)
            rounds.append(result)
            experiment = get_experiment(store_root=root, experiment_id=experiment_id)
            state = memory.load_state(memory.state_path(experiment)) or memory.initial_state(experiment)
    except KeyboardInterrupt:
        state.update({"status": "interrupted", "stop_reason": "keyboard_interrupt", "last_checkpoint": "interrupted"})
        memory.heartbeat(state)
        state["updated_at"] = utc_now_iso()
        memory.save_state(experiment, state)
        return _run_result(experiment=experiment, state=state, rounds=rounds, interrupted=True)
    except Exception:
        state.update({"status": "failed", "last_checkpoint": "failed"})
        memory.heartbeat(state)
        state["updated_at"] = utc_now_iso()
        memory.save_state(experiment, state)
        raise

    if state.get("status") == "running":
        state.update({"status": "stopped", "stop_reason": "max_rounds_reached", "last_checkpoint": "stopped"})
        memory.heartbeat(state)
        state["updated_at"] = utc_now_iso()
        memory.save_state(experiment, state)
    return _run_result(experiment=experiment, state=state, rounds=rounds, interrupted=False)


def _run_one_round(*, root: Path, experiment_id: str, state: dict[str, object]) -> ResearchRoundResult:
    experiment = get_experiment(store_root=root, experiment_id=experiment_id)
    round_number = as_int(state.get("next_round_number"), default=1)
    round_label = f"r{round_number:03d}"
    artifact_dir = memory.rounds_dir(experiment)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    report = _safe_report(root=root, experiment_id=experiment_id)
    if not context.has_scored_primary_row(report):
        return _run_baseline_round(
            root=root, experiment=experiment, state=state, round_number=round_number, round_label=round_label
        )

    prog_path = memory.program_path(experiment)
    ctx = context.build_context(root=root, experiment=experiment, report=report, state=state)
    prompt = llm.render_prompt(ctx, program_path=prog_path)
    try:
        raw_response, model_source = _call_research_llm(
            prompt=prompt, artifact_dir=artifact_dir, round_label=round_label
        )
    except Exception as exc:
        memory.write_failure_debug(artifact_dir=artifact_dir, round_label=round_label, prompt=prompt, error=str(exc))
        raise
    try:
        llm_response = llm.parse_llm_response(raw_response)
    except Exception as exc:
        memory.write_failure_debug(
            artifact_dir=artifact_dir, round_label=round_label, prompt=prompt, raw_response=raw_response, error=str(exc)
        )
        raise
    decision = llm_response.decision
    state["_pending_changes"] = [{"path": c.path, "value": c.value} for c in decision.changes]
    state["_pending_parent"] = decision.parent_config
    config_path = boundary.materialize_config(experiment=experiment, round_label=round_label, decision=decision)
    return _train_score_record_round(
        root=root,
        experiment=experiment,
        state=state,
        round_number=round_number,
        round_label=round_label,
        action="run",
        config_path=config_path,
        parent_config=decision.parent_config,
        learning=decision.learning,
        next_hypothesis=decision.next_hypothesis,
        memo=llm_response.round_markdown,
        experiment_markdown=llm_response.experiment_markdown,
    )


def _run_baseline_round(
    *, root: Path, experiment: ExperimentRecord, state: dict[str, object], round_number: int, round_label: str
) -> ResearchRoundResult:
    config_path = boundary.baseline_config(experiment, round_label)
    parent_name = config_path.name
    learning = f"Baseline round (copy of seed `{parent_name}`) before asking the LLM for mutations."
    return _train_score_record_round(
        root=root,
        experiment=experiment,
        state=state,
        round_number=round_number,
        round_label=round_label,
        action="baseline",
        config_path=config_path,
        parent_config=parent_name,
        learning=learning,
        next_hypothesis=None,
        memo=f"# {round_label} Research State\n\n{learning}\n",
        experiment_markdown=None,
    )


def _train_score_record_round(
    *,
    root: Path,
    experiment: ExperimentRecord,
    state: dict[str, object],
    round_number: int,
    round_label: str,
    action: ResearchAction,
    config_path: Path,
    parent_config: str | None,
    learning: str,
    next_hypothesis: str | None,
    memo: str,
    experiment_markdown: str | None,
) -> ResearchRoundResult:
    started_at = time.monotonic()
    reused = False
    with bind_launch_metadata(source="feature.agentic_research.train", operation_type="run", job_type="run"):
        try:
            trained = train_experiment(store_root=root, experiment_id=experiment.experiment_id, config_path=config_path)
        except TrainingError as exc:
            recovered = boundary.reuse_finished_run_on_hash_collision(
                root=root, experiment=experiment, exc=exc, index_run=index_run
            )
            if recovered is None:
                raise
            trained = recovered
            reused = True
    # run_plan row must exist before scoring: the scorer resolves the round's config by it.
    boundary.record_round_config_in_run_plan(experiment=experiment, round_label=round_label, config_path=config_path)
    # Scored-or-failed rule (copied from run.py:780): only skip scoring when reusing a run
    # that already has its primary metric on disk; otherwise score now or fail the round.
    if not reused or context.run_primary_metric_from_disk(root=root, run_id=trained.run_id) is None:
        with bind_launch_metadata(source="feature.agentic_research.score_round", operation_type="run", job_type="run"):
            score_experiment_round(
                store_root=root, experiment_id=experiment.experiment_id, round=round_label, stage=SCORING_STAGE
            )
    wall_seconds = max(0.0, time.monotonic() - started_at)

    report = _safe_report(root=root, experiment_id=experiment.experiment_id)
    row = context.row_for_run(report, trained.run_id)
    metric_value = getattr(row, PRIMARY_METRIC_FIELD) if row is not None else None
    if metric_value is None:
        metric_value = context.run_primary_metric_from_disk(root=root, run_id=trained.run_id)
    best = context.best_run_from_report(report)
    is_champion = _advance_champion(
        state=state,
        round_label=round_label,
        config_path=config_path,
        run_id=trained.run_id,
        metric=metric_value,
    )

    entry = _journal_entry(
        round_number=round_number,
        round_label=round_label,
        action=action,
        status="completed",
        config_path=config_path,
        parent_config=parent_config,
        run_id=trained.run_id,
        metric=optional_float(metric_value),
        is_champion=is_champion,
        learning=learning,
        next_hypothesis=next_hypothesis,
        changes=_take_pending_changes(state),
        wall_seconds=wall_seconds,
    )
    memory.append_journal(experiment, entry)
    memory.write_round_markdown(experiment, entry, memo=memo)
    memory.write_experiment_markdown(experiment, experiment_markdown)
    state.update(
        {
            "status": "running",
            "next_round_number": round_number + 1,
            "total_rounds_completed": as_int(state.get("total_rounds_completed"), default=0) + 1,
            "last_checkpoint": "round_completed",
            "last_round_label": round_label,
            "last_run_id": trained.run_id,
            "last_error": None,
            "best_overall": asdict(best),
            "failed_rounds_counter": 0,
            "updated_at": utc_now_iso(),
        }
    )
    state.pop("_pending_parent", None)
    memory.rotate_run_artifacts(root=root, experiment=experiment, state=state, last_round_number=round_number)
    memory.heartbeat(state)
    memory.save_state(experiment, state)
    return ResearchRoundResult(
        round_number=round_number,
        round_label=round_label,
        action=action,
        status="completed",
        config_path=config_path,
        run_id=trained.run_id,
        metric_value=metric_value,
        learning=learning,
        artifact_dir=memory.rounds_dir(experiment),
    )


def _advance_champion(
    *, state: dict[str, object], round_label: str, config_path: Path, run_id: str, metric: object
) -> bool:
    """Mechanical keep/discard: champion advances iff metric strictly exceeds the incumbent."""
    typed = optional_float(metric)
    if typed is None:
        return False
    champion = state.get("champion")
    current = optional_float(champion.get("metric")) if isinstance(champion, dict) else None
    if current is not None and typed <= current:
        return False
    state["champion"] = {
        "config": config_path.name,
        "run_id": run_id,
        "metric": typed,
        "round": int(round_label.removeprefix("r")) if round_label.removeprefix("r").isdigit() else None,
    }
    return True


def _record_failed_round(
    *, experiment: ExperimentRecord, state: dict[str, object], error: Exception
) -> ResearchRoundResult:
    return _record_terminal_round(experiment=experiment, state=state, error=error, status="failed")


def _record_duplicate_skip_round(
    *, experiment: ExperimentRecord, state: dict[str, object], error: AgenticResearchDuplicateCandidate
) -> ResearchRoundResult:
    return _record_terminal_round(experiment=experiment, state=state, error=error, status="skipped")


def _record_terminal_round(
    *, experiment: ExperimentRecord, state: dict[str, object], error: Exception, status: str
) -> ResearchRoundResult:
    """Record a failed (counts toward bail) or skipped (soft, resets counter) round."""
    round_number = as_int(state.get("next_round_number"), default=1)
    round_label = f"r{round_number:03d}"
    artifact_dir = memory.rounds_dir(experiment)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    message = str(error) or error.__class__.__name__
    learning = f"Round skipped: {message}"
    entry = _journal_entry(
        round_number=round_number,
        round_label=round_label,
        action="run",
        status=status,
        config_path=None,
        parent_config=optional_str(state.get("_pending_parent")),
        run_id=None,
        metric=None,
        is_champion=False,
        learning=learning,
        next_hypothesis=None,
        changes=_take_pending_changes(state),
        wall_seconds=None,
        error=message,
    )
    memory.append_journal(experiment, entry)
    memory.write_round_markdown(experiment, entry, memo=None)
    state.update({"status": "running", "next_round_number": round_number + 1, "updated_at": utc_now_iso()})
    state.pop("_pending_parent", None)
    if status == "skipped":
        # Soft skip: a no-progress round that resets the failure counter.
        state.update(
            {
                "total_rounds_completed": as_int(state.get("total_rounds_completed"), default=0) + 1,
                "last_checkpoint": "round_completed",
                "last_round_label": round_label,
                "failed_rounds_counter": 0,
                "last_error": None,
            }
        )
    else:
        failures = as_int(state.get("failed_rounds_counter"), default=0) + 1
        state.update({"last_checkpoint": "round_failed", "failed_rounds_counter": failures, "last_error": message})
        if failures >= CONSECUTIVE_FAILURE_BAIL_THRESHOLD:
            state.update(
                {
                    "status": "stopped",
                    "stop_reason": f"consecutive_failures:{failures}",
                    "last_checkpoint": "consecutive_failures_bail",
                }
            )
    memory.heartbeat(state)
    memory.save_state(experiment, state)
    return ResearchRoundResult(
        round_number=round_number,
        round_label=round_label,
        action="run",
        status=status,
        config_path=None,
        run_id=None,
        metric_value=None,
        learning=learning,
        artifact_dir=artifact_dir,
    )


def _take_pending_changes(state: dict[str, object]) -> list[dict[str, object]]:
    pending = state.pop("_pending_changes", None)
    return pending if isinstance(pending, list) else []


def _journal_entry(
    *,
    round_number: int,
    round_label: str,
    action: ResearchAction,
    status: str,
    config_path: Path | None,
    parent_config: str | None,
    run_id: str | None,
    metric: float | None,
    is_champion: bool,
    learning: str,
    next_hypothesis: str | None,
    changes: list[dict[str, object]],
    wall_seconds: float | None,
    error: str | None = None,
) -> dict[str, object]:
    now = utc_now_iso()
    return {
        "round": round_number,
        "round_label": round_label,
        "action": action,
        "status": status,
        "config": config_path.name if config_path is not None else None,
        "parent_config": parent_config,
        "run_id": run_id,
        "seed": None,
        "metric": metric,
        "is_champion": is_champion,
        "error": error,
        "learning": learning,
        "next_hypothesis": next_hypothesis,
        "changes": changes,
        "created_at": now,
        "completed_at": now,
        "wall_seconds": round(wall_seconds, 1) if wall_seconds is not None else None,
    }


def _status_result(
    *, experiment: ExperimentRecord, state: dict[str, object], best: ResearchBestRun
) -> ResearchStatusResult:
    auto_dir = memory.agentic_dir(experiment)
    journal = memory.journal_path(experiment)
    return ResearchStatusResult(
        experiment_id=experiment.experiment_id,
        status=status_value(state.get("status")),
        next_round_number=as_int(state.get("next_round_number"), default=1),
        total_rounds_completed=as_int(state.get("total_rounds_completed"), default=0),
        last_checkpoint=str(state.get("last_checkpoint") or "initialized"),
        last_round_label=optional_str(state.get("last_round_label")),
        last_run_id=optional_str(state.get("last_run_id")),
        stop_reason=optional_str(state.get("stop_reason")),
        best_overall=best,
        agentic_research_dir=auto_dir,
        state_path=auto_dir / STATE_FILENAME,
        trace_path=journal,
        decision_path=journal,
        program_path=memory.program_path(experiment),
    )


def _run_result(
    *, experiment: ExperimentRecord, state: dict[str, object], rounds: list[ResearchRoundResult], interrupted: bool
) -> ResearchRunResult:
    return ResearchRunResult(
        experiment_id=experiment.experiment_id,
        status=status_value(state.get("status")),
        next_round_number=as_int(state.get("next_round_number"), default=1),
        total_rounds_completed=as_int(state.get("total_rounds_completed"), default=0),
        last_checkpoint=str(state.get("last_checkpoint") or "initialized"),
        stop_reason=optional_str(state.get("stop_reason")),
        best_overall=_best_from_state(state),
        rounds=tuple(rounds),
        interrupted=interrupted,
    )


def _best_from_state(state: dict[str, object]) -> ResearchBestRun:
    payload = state.get("best_overall")
    if not isinstance(payload, dict):
        return ResearchBestRun()
    return ResearchBestRun(
        experiment_id=optional_str(payload.get("experiment_id")),
        run_id=optional_str(payload.get("run_id")),
        bmc_last_200_eras_mean=optional_float(payload.get("bmc_last_200_eras_mean")),
        bmc_mean=optional_float(payload.get("bmc_mean")),
        corr_mean=optional_float(payload.get("corr_mean")),
        mmc_mean=optional_float(payload.get("mmc_mean")),
        cwmm_mean=optional_float(payload.get("cwmm_mean")),
        updated_at=optional_str(payload.get("updated_at")),
    )


def _is_terminal_stop(state: dict[str, object]) -> bool:
    return status_value(state.get("status")) == "stopped"
