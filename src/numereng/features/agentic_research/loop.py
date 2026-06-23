"""Session lifecycle and round driver."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

from numereng.config.training import load_training_config_json
from numereng.features.agentic_research import aggregate, boundary, context, llm, memory
from numereng.features.agentic_research import types as ar_types
from numereng.features.agentic_research.context import _safe_report
from numereng.features.agentic_research.llm import _call_research_llm
from numereng.features.agentic_research.types import (
    AgenticResearchDuplicateCandidate,
    AgenticResearchValidationError,
    ResearchAction,
    ResearchBestRun,
    ResearchRoundResult,
    ResearchRunResult,
    ResearchStatusResult,
)
from numereng.features.experiments import ExperimentRecord, get_experiment, score_experiment_round, train_experiment
from numereng.features.store import index_run, resolve_store_root
from numereng.features.telemetry import bind_launch_metadata
from numereng.features.training.errors import TrainingError

__all__ = ["get_research_status", "program_markdown", "run_research"]


def program_markdown() -> str:
    return ar_types.PROGRAM_PATH.read_text(encoding="utf-8")


def get_research_status(*, store_root: str | Path = ".numereng", experiment_id: str) -> ResearchStatusResult:
    root = resolve_store_root(store_root)
    experiment = get_experiment(store_root=root, experiment_id=experiment_id)
    state = memory.load_state(memory.state_path(experiment)) or memory.initial_state(experiment)
    best = context.best_run_from_report(_safe_report(root=root, experiment_id=experiment.experiment_id))
    return _status_result(experiment=experiment, state=state, best=best)


def run_research(*, store_root: str | Path = ".numereng", experiment_id: str, max_rounds: int = 1) -> ResearchRunResult:
    if max_rounds < 1:
        raise AgenticResearchValidationError("agentic_research_max_rounds_invalid")
    root = resolve_store_root(store_root)
    experiment = get_experiment(store_root=root, experiment_id=experiment_id)
    if experiment.status == "archived":
        raise AgenticResearchValidationError("agentic_research_experiment_archived")
    boundary.assert_scoring_paths_frozen(experiment)
    memory.agentic_dir(experiment).mkdir(parents=True, exist_ok=True)
    state = memory.load_state(memory.state_path(experiment)) or memory.initial_state(experiment)
    state.update({"status": "running", "stop_reason": None})
    _save(experiment, state)

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
                result = _record_terminal_round(experiment=experiment, state=state, error=exc, status="skipped")
            except Exception as exc:
                result = _record_terminal_round(experiment=experiment, state=state, error=exc, status="failed")
            rounds.append(result)
            experiment = get_experiment(store_root=root, experiment_id=experiment_id)
            state = memory.load_state(memory.state_path(experiment)) or memory.initial_state(experiment)
    except KeyboardInterrupt:
        state.update({"status": "interrupted", "stop_reason": "keyboard_interrupt", "last_checkpoint": "interrupted"})
        state["updated_at"] = ar_types.utc_now_iso()
        _save(experiment, state)
        return _run_result(experiment=experiment, state=state, rounds=rounds, interrupted=True)
    except Exception:
        state.update({"status": "failed", "last_checkpoint": "failed", "updated_at": ar_types.utc_now_iso()})
        _save(experiment, state)
        raise

    if state.get("status") == "running":
        state.update({"status": "stopped", "stop_reason": "max_rounds_reached", "last_checkpoint": "stopped"})
        state["updated_at"] = ar_types.utc_now_iso()
        _save(experiment, state)
    return _run_result(experiment=experiment, state=state, rounds=rounds, interrupted=False)


def _run_one_round(*, root: Path, experiment_id: str, state: dict[str, object]) -> ResearchRoundResult:
    experiment = get_experiment(store_root=root, experiment_id=experiment_id)
    round_number = ar_types.as_int(state.get("next_round_number"), default=1)
    round_label = f"r{round_number:03d}"
    artifact_dir = memory.rounds_dir(experiment)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report = _safe_report(root=root, experiment_id=experiment_id)
    if not context.has_scored_primary_row(report):
        config_path = boundary.baseline_config(experiment, round_label)
        learning = f"Baseline round (copy of seed `{config_path.name}`) before asking the LLM for mutations."
        return _train_score_record_round(
            root=root,
            experiment=experiment,
            state=state,
            round_number=round_number,
            round_label=round_label,
            action="baseline",
            config_path=config_path,
            parent_config=config_path.name,
            learning=learning,
            next_hypothesis=None,
            believed_best=config_path.name,
            memo=f"# {round_label} Research State\n\n{learning}\n",
            experiment_markdown=None,
        )
    prompt = llm.render_prompt(
        context.build_context(root=root, experiment=experiment, report=report, state=state),
        program_path=memory.program_path(experiment),
    )
    try:
        raw_response, _model_source = _call_research_llm(
            prompt=prompt, artifact_dir=artifact_dir, round_label=round_label
        )
    except Exception as exc:
        memory.write_failure_debug(artifact_dir=artifact_dir, round_label=round_label, prompt=prompt, error=str(exc))
        raise
    try:
        llm_response = llm.parse_llm_response(raw_response)
    except Exception as exc:
        memory.write_failure_debug(
            artifact_dir=artifact_dir, round_label=round_label, prompt=prompt, error=str(exc), raw_response=raw_response
        )
        raise
    decision = llm_response.decision
    state["_pending_changes"] = [{"path": change.path, "value": change.value} for change in decision.changes]
    state["_pending_parent"] = decision.parent_config
    return _train_score_record_round(
        root=root,
        experiment=experiment,
        state=state,
        round_number=round_number,
        round_label=round_label,
        action="run",
        config_path=boundary.materialize_config(experiment=experiment, round_label=round_label, decision=decision),
        parent_config=decision.parent_config,
        learning=decision.learning,
        next_hypothesis=decision.next_hypothesis,
        believed_best=decision.believed_best,
        memo=llm_response.round_markdown,
        experiment_markdown=llm_response.experiment_markdown,
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
    believed_best: str | None,
    memo: str,
    experiment_markdown: str | None,
) -> ResearchRoundResult:
    started_at = time.monotonic()
    reused = False
    state["_pending_config"] = config_path.name
    state["_pending_config_path"] = str(config_path)
    with bind_launch_metadata(source="feature.agentic_research.train", operation_type="run", job_type="run"):
        try:
            trained = train_experiment(store_root=root, experiment_id=experiment.experiment_id, config_path=config_path)
        except TrainingError as exc:
            trained = boundary.reuse_finished_run_on_hash_collision(
                root=root, experiment=experiment, exc=exc, index_run=index_run
            )
            if trained is None:
                raise
            reused = True
    state["_pending_run_id"] = trained.run_id
    boundary.record_round_config_in_run_plan(experiment=experiment, round_label=round_label, config_path=config_path)
    if not reused or context.run_primary_metric_from_disk(root=root, run_id=trained.run_id) is None:
        with bind_launch_metadata(source="feature.agentic_research.score_round", operation_type="run", job_type="run"):
            score_experiment_round(
                store_root=root, experiment_id=experiment.experiment_id, round=round_label, stage=ar_types.SCORING_STAGE
            )
    report = _safe_report(root=root, experiment_id=experiment.experiment_id)
    row = context.row_for_run(report, trained.run_id)
    metric_value = getattr(row, ar_types.PRIMARY_METRIC_FIELD) if row is not None else None
    metric_value = (
        metric_value
        if metric_value is not None
        else context.run_primary_metric_from_disk(root=root, run_id=trained.run_id)
    )
    fnc_value = row.fnc_mean if row is not None else None
    fnc_value = fnc_value if fnc_value is not None else context.run_fnc_mean_from_disk(root=root, run_id=trained.run_id)
    is_champion = _advance_champion(
        state=state, round_label=round_label, config_path=config_path, run_id=trained.run_id, metric=metric_value
    )
    entry = _journal_entry(
        round=round_number,
        round_label=round_label,
        action=action,
        status="completed",
        config=config_path.name,
        parent_config=parent_config,
        run_id=trained.run_id,
        seed=_config_seed(config_path),
        metric=ar_types.optional_float(metric_value),
        fnc=ar_types.optional_float(fnc_value),
        is_champion=is_champion,
        learning=learning,
        next_hypothesis=next_hypothesis,
        changes=_take_pending_changes(state),
        wall_seconds=max(0.0, time.monotonic() - started_at),
    )
    memory.append_journal(experiment, entry)
    memory.write_round_markdown(experiment, entry, memo=memo)
    memory.write_experiment_markdown(experiment, experiment_markdown)
    believed, believed_changed_round = _resolve_believed_best(
        experiment=experiment,
        state=state,
        declared=believed_best,
        fallback_config=config_path.name,
        round_number=round_number,
    )
    state.update(
        {
            "status": "running",
            "next_round_number": round_number + 1,
            "total_rounds_completed": ar_types.as_int(state.get("total_rounds_completed"), default=0) + 1,
            "last_checkpoint": "round_completed",
            "last_round_label": round_label,
            "last_run_id": trained.run_id,
            "last_error": None,
            "best_overall": asdict(context.best_run_from_report(report)),
            "believed_best": believed,
            "believed_best_changed_round": believed_changed_round,
            "failed_rounds_counter": 0,
            "updated_at": ar_types.utc_now_iso(),
        }
    )
    state.pop("_pending_parent", None)
    state.pop("_pending_config", None)
    state.pop("_pending_config_path", None)
    state.pop("_pending_run_id", None)
    _save(experiment, state)
    return ResearchRoundResult(
        round_number,
        round_label,
        action,
        "completed",
        config_path,
        trained.run_id,
        metric_value,
        learning,
        memory.rounds_dir(experiment),
    )


def _advance_champion(
    *, state: dict[str, object], round_label: str, config_path: Path, run_id: str, metric: object
) -> bool:
    typed = ar_types.optional_float(metric)
    champion = state.get("champion")
    current = ar_types.optional_float(champion.get("metric")) if isinstance(champion, dict) else None
    if typed is None or current is not None and typed <= current:
        return False
    state["champion"] = {
        "config": config_path.name,
        "run_id": run_id,
        "metric": typed,
        "round": int(round_label.removeprefix("r")) if round_label.removeprefix("r").isdigit() else None,
    }
    return True


def _resolve_believed_best(
    *,
    experiment: ExperimentRecord,
    state: dict[str, object],
    declared: str | None,
    fallback_config: str,
    round_number: int,
) -> tuple[dict[str, object], int | None]:
    """Persist the model-declared trusted recipe, enriched with its seed-trio stats.

    Falls back to the current champion config, then this round's config, so the field is always
    populated (the model may omit it on the openrouter transport). The change-round counter shipped
    here is consumed by the plateau signal; updating it on declared-config change is the only state
    bookkeeping — belief itself remains the model's to declare.
    """
    champion = state.get("champion")
    champion_config = champion.get("config") if isinstance(champion, dict) else None
    config_name = declared or (champion_config if isinstance(champion_config, str) else None) or fallback_config
    configs = aggregate.load_config_cache(experiment.manifest_path.parent / "configs")
    groups = aggregate.aggregate_recipes(memory.journal_all(experiment), configs=configs)
    group = aggregate.group_for_config(groups, config_name, configs)
    record: dict[str, object] = {
        "config": config_name,
        "recipe_key": group.recipe_key if group else None,
        "trio_mean": group.trio_mean if group else None,
        "trio_fnc": group.trio_fnc_mean if group else None,
        "seed_count": group.count if group else None,
        "run_ids": list(group.run_ids) if group else [],
        "declared_round": round_number,
    }
    prior = state.get("believed_best")
    prior_config = prior.get("config") if isinstance(prior, dict) else None
    if config_name != prior_config:
        return record, round_number
    return record, ar_types.as_int(state.get("believed_best_changed_round"), default=round_number)


def _record_terminal_round(
    *, experiment: ExperimentRecord, state: dict[str, object], error: Exception, status: str
) -> ResearchRoundResult:
    round_number = ar_types.as_int(state.get("next_round_number"), default=1)
    round_label = f"r{round_number:03d}"
    artifact_dir = memory.rounds_dir(experiment)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    message = str(error) or error.__class__.__name__
    learning = f"Round skipped: {message}"
    config_name = ar_types.optional_str(state.get("_pending_config"))
    config_path_raw = ar_types.optional_str(state.get("_pending_config_path"))
    config_path = Path(config_path_raw) if config_path_raw is not None else None
    run_id = ar_types.optional_str(state.get("_pending_run_id"))
    typed_action: ResearchAction = "run"
    entry = _journal_entry(
        round=round_number,
        round_label=round_label,
        action=typed_action,
        status=status,
        config=config_name,
        parent_config=ar_types.optional_str(state.get("_pending_parent")),
        run_id=run_id,
        seed=None,
        metric=None,
        fnc=None,
        is_champion=False,
        learning=learning,
        next_hypothesis=None,
        changes=_take_pending_changes(state),
        wall_seconds=None,
        error=message,
    )
    memory.append_journal(experiment, entry)
    memory.write_round_markdown(experiment, entry, memo=None)
    state.update({"status": "running", "next_round_number": round_number + 1, "updated_at": ar_types.utc_now_iso()})
    state.pop("_pending_parent", None)
    state.pop("_pending_config", None)
    state.pop("_pending_config_path", None)
    state.pop("_pending_run_id", None)
    if status == "skipped":
        state.update(
            {
                "total_rounds_completed": ar_types.as_int(state.get("total_rounds_completed"), default=0) + 1,
                "last_checkpoint": "round_completed",
                "last_round_label": round_label,
                "failed_rounds_counter": 0,
                "last_error": None,
            }
        )
    else:
        failures = ar_types.as_int(state.get("failed_rounds_counter"), default=0) + 1
        state.update({"last_checkpoint": "round_failed", "failed_rounds_counter": failures, "last_error": message})
        if run_id is not None:
            state["last_run_id"] = run_id
        if failures >= ar_types.CONSECUTIVE_FAILURE_BAIL_THRESHOLD:
            state.update(
                {
                    "status": "stopped",
                    "stop_reason": f"consecutive_failures:{failures}",
                    "last_checkpoint": "consecutive_failures_bail",
                }
            )
    _save(experiment, state)
    return ResearchRoundResult(
        round_number, round_label, typed_action, status, config_path, run_id, None, learning, artifact_dir
    )


def _take_pending_changes(state: dict[str, object]) -> list[dict[str, object]]:
    pending = state.pop("_pending_changes", None)
    return pending if isinstance(pending, list) else []


def _journal_entry(**entry: object) -> dict[str, object]:
    now = ar_types.utc_now_iso()
    wall_seconds = entry.get("wall_seconds")
    entry["wall_seconds"] = round(wall_seconds, 1) if isinstance(wall_seconds, (int, float)) else None
    entry["created_at"] = now
    entry["completed_at"] = now
    return entry


def _status_result(
    *, experiment: ExperimentRecord, state: dict[str, object], best: ResearchBestRun
) -> ResearchStatusResult:
    auto_dir = memory.agentic_dir(experiment)
    journal = memory.journal_path(experiment)
    return ResearchStatusResult(
        experiment_id=experiment.experiment_id,
        status=ar_types.status_value(state.get("status")),
        next_round_number=ar_types.as_int(state.get("next_round_number"), default=1),
        total_rounds_completed=ar_types.as_int(state.get("total_rounds_completed"), default=0),
        last_checkpoint=str(state.get("last_checkpoint") or "initialized"),
        last_round_label=ar_types.optional_str(state.get("last_round_label")),
        last_run_id=ar_types.optional_str(state.get("last_run_id")),
        stop_reason=ar_types.optional_str(state.get("stop_reason")),
        best_overall=best,
        agentic_research_dir=auto_dir,
        state_path=auto_dir / ar_types.STATE_FILENAME,
        trace_path=journal,
        decision_path=journal,
        program_path=memory.program_path(experiment),
    )


def _run_result(
    *, experiment: ExperimentRecord, state: dict[str, object], rounds: list[ResearchRoundResult], interrupted: bool
) -> ResearchRunResult:
    return ResearchRunResult(
        experiment_id=experiment.experiment_id,
        status=ar_types.status_value(state.get("status")),
        next_round_number=ar_types.as_int(state.get("next_round_number"), default=1),
        total_rounds_completed=ar_types.as_int(state.get("total_rounds_completed"), default=0),
        last_checkpoint=str(state.get("last_checkpoint") or "initialized"),
        stop_reason=ar_types.optional_str(state.get("stop_reason")),
        best_overall=_best_from_state(state),
        rounds=tuple(rounds),
        interrupted=interrupted,
    )


def _best_from_state(state: dict[str, object]) -> ResearchBestRun:
    payload = state.get("best_overall")
    if not isinstance(payload, dict):
        return ResearchBestRun()
    return ResearchBestRun(**{key: payload.get(key) for key in ResearchBestRun.__dataclass_fields__})


def _config_seed(config_path: Path) -> int | None:
    model = load_training_config_json(config_path).get("model")
    params = model.get("params") if isinstance(model, dict) else None
    seed = params.get("random_state") if isinstance(params, dict) else None
    return seed if isinstance(seed, int) and not isinstance(seed, bool) else None


def _save(experiment: ExperimentRecord, state: dict[str, object]) -> None:
    memory.heartbeat(state)
    memory.save_state(experiment, state)


def _is_terminal_stop(state: dict[str, object]) -> bool:
    return ar_types.status_value(state.get("status")) == "stopped"
