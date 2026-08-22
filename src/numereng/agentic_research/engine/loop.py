"""Session lifecycle and round driver."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

from numereng.agentic_research.engine import aggregate, boundary, context, llm, memory
from numereng.agentic_research.engine import types as ar_types

# Bound under their underscore names on purpose: these two locals are the monkeypatch seam the
# loop tests use (they patch ``loop._safe_report`` / ``loop._call_research_llm``).
from numereng.agentic_research.engine.context import safe_report as _safe_report
from numereng.agentic_research.engine.llm import call_research_llm as _call_research_llm
from numereng.agentic_research.engine.types import (
    AgenticResearchDuplicateCandidate,
    AgenticResearchValidationError,
    ResearchAction,
    ResearchBestRun,
    ResearchRoundResult,
    ResearchRunResult,
    ResearchStatusResult,
)
from numereng.config.training import load_training_config_json
from numereng.features.experiments import (
    ExperimentRecord,
    ExperimentReport,
    freeze_experiment_holdout,
    get_experiment,
    score_experiment_round,
    train_experiment,
)
from numereng.features.store import index_run, resolve_store_root
from numereng.features.telemetry import bind_launch_metadata
from numereng.features.training.errors import TrainingError

__all__ = ["get_research_status", "program_markdown", "run_research"]

# Per-round scratch keys stripped from the session state once the round is recorded.
_PENDING_KEYS = ("_pending_parent", "_pending_config", "_pending_config_path", "_pending_run_id", "_pending_changes")


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
    boundary.program_allowed_paths(experiment)  # fail a misconfigured allowlist at round 0, not mid-round
    _prevalidate_seed_configs(experiment)
    _prevalidate_program_core(experiment)
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


def _prevalidate_seed_configs(experiment: ExperimentRecord) -> None:
    """Validate authored seed configs against the training contract before entering the loop.

    A config key that parses loosely but the training dispatcher rejects (e.g. the removed legacy
    `training.engine.embargo_eras`/`window_size_eras`) otherwise fails every round and bails the
    session after five failures. Loading through the contract here surfaces it up front, with the
    training-config error naming the offending key.
    """
    config_dir = memory.configs_dir(experiment)
    configs = sorted(config_dir.glob("*.json"))
    seeds = boundary.seed_config_paths(config_dir)
    for path in seeds or configs[:1]:
        try:
            load_training_config_json(path)
        except Exception as exc:
            raise AgenticResearchValidationError(f"agentic_research_seed_config_invalid:{path.name}:{exc}") from exc


def _prevalidate_program_core(experiment: ExperimentRecord) -> None:
    """Fail fast if a custom program's CORE sections have drifted from the base PROGRAM.md.

    The runner loads exactly one self-contained program file, so every custom program must copy the
    invariant CORE (frozen evaluator, evidence doctrine, output contract, ...) verbatim. A stale CORE
    line silently changes the contract the model runs under; catch it before round 1. The base
    PROGRAM.md is exempt (it is the canonical CORE).
    """
    program = memory.program_path(experiment)
    if program == ar_types.PROGRAM_PATH:
        return
    diverged = ar_types.first_diverging_core_section(
        program.read_text(encoding="utf-8"), ar_types.PROGRAM_PATH.read_text(encoding="utf-8")
    )
    if diverged is not None:
        raise AgenticResearchValidationError(f"agentic_research_program_core_drift:{program.name}:section:{diverged}")


def _run_one_round(*, root: Path, experiment_id: str, state: dict[str, object]) -> ResearchRoundResult:
    experiment = get_experiment(store_root=root, experiment_id=experiment_id)
    round_number = ar_types.as_int(state.get("next_round_number"), default=1)
    round_label = memory.round_label(round_number)
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
    if decision.seeds:
        return _run_seed_trio_round(
            root=root,
            experiment=experiment,
            state=state,
            round_number=round_number,
            round_label=round_label,
            decision=decision,
            memo=llm_response.round_markdown,
            experiment_markdown=llm_response.experiment_markdown,
        )
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
    """Single-run round (baseline + single-seed mutation): train/score one config, then finalize.

    Any failure raised here propagates to `run_research`, which records a terminal failed/skipped
    round — this preserves the pre-multi-seed error path exactly.
    """
    started_at = time.monotonic()
    report, run_id, metric_value, fnc_value, benchmark_corr_value = _train_and_score(
        root=root, experiment=experiment, state=state, round_label=round_label, config_path=config_path
    )
    is_champion = _advance_champion(
        state=state, round_number=round_number, config_path=config_path, run_id=run_id, metric=metric_value
    )
    entry = _journal_entry(
        round=round_number,
        round_label=round_label,
        action=action,
        status="completed",
        config=config_path.name,
        parent_config=parent_config,
        run_id=run_id,
        seed=_config_seed(config_path),
        metric=ar_types.optional_float(metric_value),
        fnc=ar_types.optional_float(fnc_value),
        benchmark_corr=ar_types.optional_float(benchmark_corr_value),
        is_champion=is_champion,
        learning=learning,
        next_hypothesis=next_hypothesis,
        changes=_take_pending_changes(state),
        wall_seconds=max(0.0, time.monotonic() - started_at),
    )
    memory.append_journal(experiment, entry)
    return _finalize_round(
        root=root,
        experiment=experiment,
        state=state,
        round_number=round_number,
        round_label=round_label,
        action=action,
        round_status="completed",
        primary_entry=entry,
        primary_config_path=config_path,
        primary_run_id=run_id,
        primary_metric=metric_value,
        learning=learning,
        memo=memo,
        experiment_markdown=experiment_markdown,
        declared_believed_best=believed_best,
        fallback_config=config_path.name,
        report=report,
        seed_lines=None,
    )


def _train_and_score(
    *, root: Path, experiment: ExperimentRecord, state: dict[str, object], round_label: str, config_path: Path
) -> tuple[ExperimentReport | None, str, float | None, float | None, float | None]:
    """Train one config, materialize deferred scoring, and read back its metrics from the report/disk."""
    reused = False
    state["_pending_config"] = config_path.name
    state["_pending_config_path"] = str(config_path)
    state["_pending_run_id"] = None
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
    # Pin the frozen holdout from the first trained run's eras BEFORE any scoring, so the
    # holdout block is excluded from every metric the LLM loop sees. No-op when no holdout
    # was requested and idempotent once frozen.
    freeze_experiment_holdout(store_root=root, experiment_id=experiment.experiment_id, run_id=trained.run_id)
    if not reused or context.run_primary_metric_from_disk(root=root, run_id=trained.run_id) is None:
        with bind_launch_metadata(source="feature.agentic_research.score_round", operation_type="run", job_type="run"):
            score_experiment_round(
                store_root=root, experiment_id=experiment.experiment_id, round=round_label, stage=ar_types.SCORING_STAGE
            )
    report = _safe_report(root=root, experiment_id=experiment.experiment_id)
    row = context.row_for_run(report, trained.run_id)
    metrics = context.load_run_metrics(root=root, run_id=trained.run_id)
    metric_value = getattr(row, ar_types.PRIMARY_METRIC_FIELD) if row is not None else None
    if metric_value is None:
        metric_value = context.metric_from_metrics(metrics, ar_types.PRIMARY_METRIC)
    fnc_value = row.fnc_mean if row is not None else None
    if fnc_value is None:
        fnc_value = context.metric_from_metrics(metrics, "fnc.mean")
    benchmark_corr_value = context.metric_from_metrics(metrics, context.BENCHMARK_CORR_METRIC)
    return report, trained.run_id, metric_value, fnc_value, benchmark_corr_value


def _run_seed_trio_round(
    *,
    root: Path,
    experiment: ExperimentRecord,
    state: dict[str, object],
    round_number: int,
    round_label: str,
    decision: ar_types.ResearchDecision,
    memo: str,
    experiment_markdown: str | None,
) -> ResearchRoundResult:
    """Multi-seed round: materialize/train/score one child config per requested seed, sequentially.

    One decision is still one round (one round-number/total advance). Each run appends its own journal
    line (same `round`, distinct `seed`/`config`/`run_id`); champion advances per completed run exactly
    as a single-seed round would. A per-seed failure is recorded and the remaining seeds continue; the
    round only counts as failed when every run failed (a duplicate seed soft-skips, like single-seed).
    """
    changes = state.get("_pending_changes")
    changes = changes if isinstance(changes, list) else []
    outcomes: list[dict[str, object]] = []
    report: ExperimentReport | None = None
    for seed in decision.seeds:
        outcome = _execute_seed(
            root=root,
            experiment=experiment,
            state=state,
            round_number=round_number,
            round_label=round_label,
            decision=decision,
            seed=seed,
            changes=changes,
        )
        outcomes.append(outcome)
        if outcome.get("report") is not None:
            report = outcome["report"]  # type: ignore[assignment]
    completed = [o for o in outcomes if o["status"] == "completed"]
    if completed:
        round_status = "completed"
    elif any(o["status"] == "skipped" for o in outcomes):
        round_status = "skipped"
    else:
        round_status = "failed"
    primary = max(completed, key=_metric_sort_key) if completed else outcomes[-1]
    seed_lines = ["- per-seed results:"] + [
        f"  - seed {o.get('seed')}: status={o['status']} run_id={o.get('run_id') or 'none'} "
        f"{ar_types.PRIMARY_METRIC_FIELD}={o['metric'] if o.get('metric') is not None else 'none'}"
        for o in outcomes
    ]
    return _finalize_round(
        root=root,
        experiment=experiment,
        state=state,
        round_number=round_number,
        round_label=round_label,
        action="run",
        round_status=round_status,
        primary_entry=primary["entry"],  # type: ignore[index]
        primary_config_path=primary.get("config_path"),  # type: ignore[arg-type]
        primary_run_id=ar_types.optional_str(primary.get("run_id")),
        primary_metric=ar_types.optional_float(primary.get("metric")),
        learning=decision.learning,
        memo=memo,
        experiment_markdown=experiment_markdown,
        declared_believed_best=decision.believed_best,
        fallback_config=str(primary.get("config") or ""),
        report=report,
        seed_lines=seed_lines,
    )


def _execute_seed(
    *,
    root: Path,
    experiment: ExperimentRecord,
    state: dict[str, object],
    round_number: int,
    round_label: str,
    decision: ar_types.ResearchDecision,
    seed: int,
    changes: list[dict[str, object]],
) -> dict[str, object]:
    started_at = time.monotonic()
    try:
        config_path = boundary.materialize_config(
            experiment=experiment, round_label=round_label, decision=decision, seed=seed
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        return _failed_seed_outcome(
            experiment=experiment,
            round_number=round_number,
            round_label=round_label,
            decision=decision,
            seed=seed,
            status="skipped" if isinstance(exc, AgenticResearchDuplicateCandidate) else "failed",
            error=str(exc),
            config=None,
            run_id=None,
            changes=changes,
            wall_seconds=time.monotonic() - started_at,
        )
    try:
        report, run_id, metric_value, fnc_value, benchmark_corr_value = _train_and_score(
            root=root, experiment=experiment, state=state, round_label=round_label, config_path=config_path
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        return _failed_seed_outcome(
            experiment=experiment,
            round_number=round_number,
            round_label=round_label,
            decision=decision,
            seed=seed,
            status="failed",
            error=str(exc),
            config=config_path.name,
            run_id=ar_types.optional_str(state.get("_pending_run_id")),
            changes=changes,
            wall_seconds=time.monotonic() - started_at,
        )
    is_champion = _advance_champion(
        state=state, round_number=round_number, config_path=config_path, run_id=run_id, metric=metric_value
    )
    entry = _journal_entry(
        round=round_number,
        round_label=round_label,
        action="run",
        status="completed",
        config=config_path.name,
        parent_config=decision.parent_config,
        run_id=run_id,
        seed=seed,
        metric=ar_types.optional_float(metric_value),
        fnc=ar_types.optional_float(fnc_value),
        benchmark_corr=ar_types.optional_float(benchmark_corr_value),
        is_champion=is_champion,
        learning=decision.learning,
        next_hypothesis=decision.next_hypothesis,
        changes=list(changes),
        wall_seconds=max(0.0, time.monotonic() - started_at),
    )
    memory.append_journal(experiment, entry)
    return {
        "status": "completed",
        "seed": entry["seed"],
        "run_id": run_id,
        "metric": metric_value,
        "config": config_path.name,
        "config_path": config_path,
        "report": report,
        "entry": entry,
    }


def _failed_seed_outcome(
    *,
    experiment: ExperimentRecord,
    round_number: int,
    round_label: str,
    decision: ar_types.ResearchDecision,
    seed: int,
    status: str,
    error: str,
    config: str | None,
    run_id: str | None,
    changes: list[dict[str, object]],
    wall_seconds: float,
) -> dict[str, object]:
    entry = _journal_entry(
        round=round_number,
        round_label=round_label,
        action="run",
        status=status,
        config=config,
        parent_config=decision.parent_config,
        run_id=run_id,
        seed=seed,
        metric=None,
        fnc=None,
        benchmark_corr=None,
        is_champion=False,
        learning=decision.learning,
        next_hypothesis=decision.next_hypothesis,
        changes=list(changes),
        wall_seconds=wall_seconds,
        error=error,
    )
    memory.append_journal(experiment, entry)
    return {
        "status": status,
        "seed": seed,
        "run_id": run_id,
        "metric": None,
        "config": config,
        "config_path": None,
        "report": None,
        "entry": entry,
    }


def _metric_sort_key(outcome: dict[str, object]) -> float:
    metric = ar_types.optional_float(outcome.get("metric"))
    return metric if metric is not None else float("-inf")


def _finalize_round(
    *,
    root: Path,
    experiment: ExperimentRecord,
    state: dict[str, object],
    round_number: int,
    round_label: str,
    action: ResearchAction,
    round_status: str,
    primary_entry: dict[str, object],
    primary_config_path: Path | None,
    primary_run_id: str | None,
    primary_metric: float | None,
    learning: str,
    memo: str | None,
    experiment_markdown: str | None,
    declared_believed_best: str | None,
    fallback_config: str,
    report: ExperimentReport | None,
    seed_lines: list[str] | None,
) -> ResearchRoundResult:
    """Round-level bookkeeping shared by single-run and multi-seed rounds (one decision = one round)."""
    memory.write_round_markdown(experiment, primary_entry, memo=memo, extra_lines=seed_lines)
    memory.write_experiment_markdown(experiment, experiment_markdown)
    _clear_pending(state)
    update: dict[str, object] = {
        "status": "running",
        "next_round_number": round_number + 1,
        "last_round_label": round_label,
        "updated_at": ar_types.utc_now_iso(),
    }
    if primary_run_id is not None:
        update["last_run_id"] = primary_run_id
    if round_status == "failed":
        state.update(update)
        _apply_failure(state, ar_types.optional_str(primary_entry.get("error")))
        _save(experiment, state)
        return ResearchRoundResult(
            round_number,
            round_label,
            action,
            round_status,
            primary_config_path,
            primary_run_id,
            None,
            learning,
            memory.rounds_dir(experiment),
        )
    believed, believed_changed_round = _resolve_believed_best(
        experiment=experiment,
        state=state,
        declared=declared_believed_best,
        fallback_config=fallback_config,
        round_number=round_number,
    )
    update.update(
        {
            "total_rounds_completed": ar_types.as_int(state.get("total_rounds_completed"), default=0) + 1,
            "last_checkpoint": "round_completed",
            "last_error": None,
            "believed_best": believed,
            "believed_best_changed_round": believed_changed_round,
            "failed_rounds_counter": 0,
        }
    )
    if report is not None:
        update["best_overall"] = asdict(context.best_run_from_report(report))
    state.update(update)
    _save(experiment, state)
    return ResearchRoundResult(
        round_number,
        round_label,
        action,
        round_status,
        primary_config_path,
        primary_run_id,
        primary_metric,
        learning,
        memory.rounds_dir(experiment),
    )


def _advance_champion(
    *, state: dict[str, object], round_number: int, config_path: Path, run_id: str, metric: object
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
        "round": round_number,
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
    configs = aggregate.load_config_cache(memory.configs_dir(experiment))
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
    round_label = memory.round_label(round_number)
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
        benchmark_corr=None,
        is_champion=False,
        learning=learning,
        next_hypothesis=None,
        changes=_take_pending_changes(state),
        wall_seconds=None,
        error=message,
    )
    memory.append_journal(experiment, entry)
    memory.write_round_markdown(experiment, entry, memo=None)
    state.update(
        {
            "status": "running",
            "next_round_number": round_number + 1,
            "last_round_label": round_label,
            "updated_at": ar_types.utc_now_iso(),
        }
    )
    _clear_pending(state)
    if status == "skipped":
        state.update(
            {
                "total_rounds_completed": ar_types.as_int(state.get("total_rounds_completed"), default=0) + 1,
                "last_checkpoint": "round_completed",
                "failed_rounds_counter": 0,
                "last_error": None,
            }
        )
    else:
        if run_id is not None:
            state["last_run_id"] = run_id
        _apply_failure(state, message)
    _save(experiment, state)
    return ResearchRoundResult(
        round_number, round_label, typed_action, status, config_path, run_id, None, learning, artifact_dir
    )


def _clear_pending(state: dict[str, object]) -> None:
    """Drop the per-round scratch keys once the round has been recorded."""
    for key in _PENDING_KEYS:
        state.pop(key, None)


def _apply_failure(state: dict[str, object], message: str | None) -> None:
    """Record a failed round: bump the consecutive-failure counter and bail at the threshold."""
    failures = ar_types.as_int(state.get("failed_rounds_counter"), default=0) + 1
    state.update({"last_checkpoint": "round_failed", "failed_rounds_counter": failures, "last_error": message})
    if failures >= ar_types.CONSECUTIVE_FAILURE_BAIL_THRESHOLD:
        state.update(
            {
                "status": "stopped",
                "stop_reason": f"consecutive_failures:{failures}",
                "last_checkpoint": "consecutive_failures_bail",
            }
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
