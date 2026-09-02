"""Session lifecycle and round driver: one execution path per round, one in-round retry.

Baseline, single-seed, seed trio, boundary rejection, duplicate, and terminal LLM failure all take
the same three steps - `_decide`, `_execute_round` (one outcome per seed), `_finalize_round`.

USAGE:
    from numereng.agentic_research.engine.loop import get_research_status, run_research

    run_research(store_root=".numereng", experiment_id="2026-09-01_my-exp", max_rounds=50)
    status = get_research_status(store_root=".numereng", experiment_id="2026-09-01_my-exp")
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

from numereng.agentic_research.engine import aggregate, boundary, context, llm, memory
from numereng.agentic_research.engine import types as ar_types

# Imported under their underscore aliases on purpose: these two are the monkeypatch seam the loop
# tests reach for (`loop._safe_report`, `loop._call_research_llm`).
from numereng.agentic_research.engine.context import safe_report as _safe_report
from numereng.agentic_research.engine.llm import call_research_llm as _call_research_llm
from numereng.agentic_research.engine.types import (
    AgenticResearchDuplicateCandidate,
    AgenticResearchValidationError,
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

__all__ = ["get_research_status", "run_research"]

# Per-round scratch keys, dropped from the session state once the round is recorded. The two config
# keys are only written by older versions of this module; they stay listed so a state file resumed
# from one is still cleaned.
_PENDING_KEYS = (
    "_pending_parent",
    "_pending_config",
    "_pending_config_path",
    "_pending_run_id",
    "_pending_changes",
    "_pending_llm",
)


# --------------------------------------------------------------------------- #
# Session state and preflight
# --------------------------------------------------------------------------- #


def _save(experiment: ExperimentRecord, state: dict[str, object]) -> None:
    memory.heartbeat(state)
    memory.save_state(experiment, state)


def _is_terminal_stop(state: dict[str, object]) -> bool:
    return ar_types.status_value(state.get("status")) == "stopped"


def _clear_pending(state: dict[str, object]) -> None:
    """Drop the per-round scratch keys once the round has been recorded."""
    for key in _PENDING_KEYS:
        state.pop(key, None)


def _take_pending_changes(state: dict[str, object]) -> list[dict[str, object]]:
    pending = state.pop("_pending_changes", None)
    return pending if isinstance(pending, list) else []


def _apply_failure(state: dict[str, object], message: str | None) -> None:
    """Record a failed round: bump the consecutive-failure counter and bail at the threshold."""
    failures = ar_types.as_int(state.get("failed_rounds_counter"), default=0) + 1
    state.update({"last_checkpoint": "round_failed", "failed_rounds_counter": failures, "last_error": message})
    if failures >= ar_types.CONSECUTIVE_FAILURE_BAIL_THRESHOLD:
        state["status"] = "stopped"
        state["stop_reason"] = f"consecutive_failures:{failures}"
        state["last_checkpoint"] = "consecutive_failures_bail"


def _best_from_state(state: dict[str, object]) -> ar_types.ResearchBestRun:
    stored = state.get("best_overall")
    payload = stored if isinstance(stored, dict) else {}
    return ar_types.ResearchBestRun(**{k: payload.get(k) for k in ar_types.ResearchBestRun.__dataclass_fields__})


def _run_result(
    *,
    experiment: ExperimentRecord,
    state: dict[str, object],
    rounds: list[ar_types.ResearchRoundResult],
    interrupted: bool,
) -> ar_types.ResearchRunResult:
    return ar_types.ResearchRunResult(
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


def _prevalidate_seed_configs(experiment: ExperimentRecord) -> None:
    """Validate authored seed configs against the training contract before entering the loop.

    A key that parses loosely but the dispatcher rejects otherwise fails every round and bails the
    session after five failures; loading through the contract here names it up front.
    """
    config_dir = memory.configs_dir(experiment)
    configs = sorted(config_dir.glob("*.json"))
    for path in boundary.seed_config_paths(config_dir) or configs[:1]:
        try:
            load_training_config_json(path)
        except Exception as exc:
            raise AgenticResearchValidationError(f"agentic_research_seed_config_invalid:{path.name}:{exc}") from exc


def _prevalidate_prompt_placeholders(experiment: ExperimentRecord) -> None:
    """Fail fast if the round prompt would not compose: PROGRAM.md must carry each placeholder
    exactly once and the experiment brief neither, so composing stays two string replacements.
    """
    program_text = ar_types.PROGRAM_PATH.read_text(encoding="utf-8")
    strategy = memory.strategy_path(experiment)
    strategy_text = strategy.read_text(encoding="utf-8")
    for placeholder in (ar_types.STRATEGY_PLACEHOLDER, ar_types.CONTEXT_PLACEHOLDER):
        if program_text.count(placeholder) != 1 or placeholder in strategy_text:
            raise AgenticResearchValidationError(
                f"agentic_research_program_placeholder_invalid:{placeholder}:{strategy.name}"
            )


# --------------------------------------------------------------------------- #
# The three round steps: decide, execute, finalize
# --------------------------------------------------------------------------- #


def _decide(
    root: Path,
    experiment: ExperimentRecord,
    state: dict[str, object],
    report: ExperimentReport | None,
    round_label: str,
    baseline: bool,
) -> tuple[ar_types.ResearchDecision, str, str | None]:
    """One proposal: the synthetic baseline decision, or one rendered, called, parsed LLM round.

    The baseline round asks nobody - it copies the authored seed and returns a decision whose parent
    is that copy - so it records like any LLM round, whose transport and parse failures dump the
    prompt (and any raw response) beside the round artifacts.
    """
    if baseline:
        name = boundary.baseline_config(experiment, round_label).name
        learning = f"Baseline round (copy of seed `{name}`) before asking the LLM for mutations."
        decision = ar_types.ResearchDecision("run", learning, "", None, name, (), None, name)
        return decision, f"# {round_label} Research State\n\n{learning}\n", None
    artifact_dir = memory.rounds_dir(experiment)
    prompt = llm.render_prompt(
        context.build_context(root=root, experiment=experiment, report=report, state=state),
        strategy_text=memory.strategy_path(experiment).read_text(encoding="utf-8"),
    )
    raw_response: str | None = None
    try:
        raw_response, model_source = _call_research_llm(
            prompt=prompt, artifact_dir=artifact_dir, round_label=round_label
        )
        state["_pending_llm"] = model_source
        response = llm.parse_llm_response(raw_response)
    except Exception as exc:
        memory.write_failure_debug(
            artifact_dir=artifact_dir, round_label=round_label, prompt=prompt, error=str(exc), raw_response=raw_response
        )
        raise
    state["_pending_changes"] = [{"path": change.path, "value": change.value} for change in response.decision.changes]
    state["_pending_parent"] = response.decision.parent_config
    return response.decision, response.round_markdown, response.experiment_markdown


def _execute_round(
    root: Path,
    experiment: ExperimentRecord,
    state: dict[str, object],
    round_number: int,
    round_label: str,
    action: ar_types.ResearchAction,
    decision: ar_types.ResearchDecision,
) -> list[dict[str, object]]:
    """Materialize, train, score, and champion-check one config per requested seed.

    One decision is still one round: `decision.seeds or (None,)` is a single unseeded run or a seed
    trio, each seed carries its own outcome (config, run, journal line), a per-seed failure never
    aborts the rest, and the champion advances per completed run. An outcome is flagged `rejected`
    when the boundary refused it outright - unanimous, that flag earns the round its one retry.
    """
    changes = [{"path": change.path, "value": change.value} for change in decision.changes]
    seed_path = boundary.seed_change_path(experiment)
    baseline_path = memory.configs_dir(experiment) / str(decision.parent_config) if action == "baseline" else None
    outcomes: list[dict[str, object]] = []
    for seed in decision.seeds or (None,):
        started_at = time.monotonic()
        state["_pending_run_id"] = None
        config_path: Path | None = None
        outcome: dict[str, object] = {"seed": seed, "changes": list(changes), "learning": decision.learning}
        try:
            config_path = baseline_path or boundary.materialize_config(
                experiment=experiment, round_label=round_label, decision=decision, seed=seed
            )
            outcome.update({"config": config_path.name, "config_path": config_path})
            if seed is None:
                outcome["seed"] = _config_seed(config_path, seed_path=seed_path)
            report, run_id, metric, fnc, corr = _train_and_score(root, experiment, state, round_label, config_path)
            outcome.update(
                {
                    "status": "completed",
                    "run_id": run_id,
                    "metric": ar_types.optional_float(metric),
                    "fnc": ar_types.optional_float(fnc),
                    "benchmark_corr": ar_types.optional_float(corr),
                    "is_champion": _advance_champion(state, round_number, config_path, run_id, metric),
                    "report": report,
                }
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            outcome.update(
                {
                    "status": "skipped" if isinstance(exc, AgenticResearchDuplicateCandidate) else "failed",
                    "rejected": config_path is None and isinstance(exc, AgenticResearchValidationError),
                    "run_id": ar_types.optional_str(state.get("_pending_run_id")),
                    "learning": f"Round skipped: {exc}",
                    "error": str(exc),
                }
            )
        outcome.update({"llm": ar_types.optional_str(state.get("_pending_llm")), "at": ar_types.utc_now_iso()})
        outcome["wall_seconds"] = max(0.0, time.monotonic() - started_at)
        outcomes.append(outcome)
    return outcomes


def _train_and_score(
    root: Path, experiment: ExperimentRecord, state: dict[str, object], round_label: str, config_path: Path
) -> tuple[ExperimentReport | None, str, float | None, float | None, float | None]:
    """Train one config, materialize deferred scoring, and read back its metrics from the report/disk."""
    reused = False
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
    return (
        report,
        trained.run_id,
        metric_value,
        fnc_value,
        context.metric_from_metrics(metrics, context.BENCHMARK_CORR_METRIC),
    )


def _advance_champion(
    state: dict[str, object], round_number: int, config_path: Path, run_id: str, metric: object
) -> bool:
    typed = ar_types.optional_float(metric)
    champion = state.get("champion")
    current = ar_types.optional_float(champion.get("metric")) if isinstance(champion, dict) else None
    if typed is None or current is not None and typed <= current:
        return False
    state["champion"] = {"config": config_path.name, "run_id": run_id, "metric": typed, "round": round_number}
    return True


def _config_seed(config_path: Path, *, seed_path: str | None = None) -> int | None:
    """Read the seed a config trains under: the experiment's seed path first, then the known names.

    Model families name it differently (LGBM `random_state`, NN models `seed`), so the manifest's
    `agentic_research_seed_path` wins.
    """
    payload = load_training_config_json(config_path)
    for dotted in (seed_path, ar_types.DEFAULT_SEED_PATH, "model.params.seed"):
        seed = ar_types.get_dotted(payload, dotted) if dotted else None
        if isinstance(seed, int) and not isinstance(seed, bool):
            return seed
    return None


def _finalize_round(
    experiment: ExperimentRecord,
    state: dict[str, object],
    round_number: int,
    round_label: str,
    action: ar_types.ResearchAction,
    decision: ar_types.ResearchDecision,
    outcomes: list[dict[str, object]],
    memo: str | None,
    experiment_markdown: str | None,
    retry_token: str | None,
) -> ar_types.ResearchRoundResult:
    """The one journal write site, memo write, and state update: one decision is one round.

    The round completes when any seed did, is skipped when a duplicate was the best that happened,
    and fails otherwise; the primary outcome (best completed seed, else the last) is what the memo,
    the state, and the result speak for.
    """
    entries = [_entry_from_outcome(o, round_number, round_label, action, decision) for o in outcomes]
    for entry in entries:
        memory.append_journal(experiment, entry)
    completed = [index for index, outcome in enumerate(outcomes) if outcome["status"] == "completed"]
    skipped = any(outcome["status"] == "skipped" for outcome in outcomes)
    round_status = "completed" if completed else "skipped" if skipped else "failed"
    index = max(completed, key=lambda i: _metric_sort_key(outcomes[i])) if completed else len(outcomes) - 1
    primary, primary_entry = outcomes[index], entries[index]

    extra_lines = [f"- retry: {retry_token}"] if retry_token else []
    if decision.seeds:
        extra_lines.append("- per-seed results:")
        extra_lines.extend(
            f"  - seed {o.get('seed')}: status={o['status']} run_id={o.get('run_id') or 'none'} "
            f"{ar_types.PRIMARY_METRIC_FIELD}={o['metric'] if o.get('metric') is not None else 'none'}"
            for o in outcomes
        )
    memory.write_round_markdown(experiment, primary_entry, memo=memo, extra_lines=extra_lines or None)
    memory.write_experiment_markdown(experiment, experiment_markdown)
    _clear_pending(state)

    run_id = ar_types.optional_str(primary.get("run_id"))
    state.update({"status": "running", "next_round_number": round_number + 1, "last_round_label": round_label})
    state["updated_at"] = ar_types.utc_now_iso()
    if run_id is not None:
        state["last_run_id"] = run_id
    if round_status == "failed":
        _apply_failure(state, ar_types.optional_str(primary_entry.get("error")))
    else:
        fallback = str(primary.get("config") or "")
        believed, changed_round = _resolve_believed_best(
            experiment, state, decision.believed_best, fallback, round_number
        )
        state["total_rounds_completed"] = ar_types.as_int(state.get("total_rounds_completed"), default=0) + 1
        state.update({"last_checkpoint": "round_completed", "last_error": None, "failed_rounds_counter": 0})
        state.update({"believed_best": believed, "believed_best_changed_round": changed_round})
        report = next((o["report"] for o in reversed(outcomes) if o.get("report") is not None), None)
        if report is not None:
            state["best_overall"] = asdict(context.best_run_from_report(report))
    _save(experiment, state)
    config_path = primary.get("config_path")
    return ar_types.ResearchRoundResult(
        round_number,
        round_label,
        action,
        round_status,
        config_path if isinstance(config_path, Path) else None,
        run_id,
        ar_types.optional_float(primary.get("metric")),
        str(primary_entry.get("learning") or ""),
        memory.rounds_dir(experiment),
    )


def _entry_from_outcome(
    outcome: dict[str, object],
    round_number: int,
    round_label: str,
    action: ar_types.ResearchAction,
    decision: ar_types.ResearchDecision,
) -> dict[str, object]:
    """The one journal-line builder: every seed, rejection, and terminal failure is shaped here."""
    stamp = ar_types.optional_str(outcome.get("at")) or ar_types.utc_now_iso()
    wall_seconds = ar_types.optional_float(outcome.get("wall_seconds"))
    entry: dict[str, object] = {
        "round": round_number,
        "round_label": round_label,
        "action": action,
        "status": outcome["status"],
        "config": ar_types.optional_str(outcome.get("config")),
        "parent_config": decision.parent_config,
        "run_id": ar_types.optional_str(outcome.get("run_id")),
        "seed": outcome.get("seed"),
        "metric": ar_types.optional_float(outcome.get("metric")),
        "fnc": ar_types.optional_float(outcome.get("fnc")),
        "benchmark_corr": ar_types.optional_float(outcome.get("benchmark_corr")),
        "is_champion": bool(outcome.get("is_champion")),
        "learning": str(outcome.get("learning") or ""),
        "next_hypothesis": decision.next_hypothesis,
        "changes": ar_types.as_list(outcome.get("changes")),
        "wall_seconds": round(wall_seconds, 1) if wall_seconds is not None else None,
        "llm": ar_types.optional_str(outcome.get("llm")),
        "created_at": stamp,
        "completed_at": stamp,
    }
    error = ar_types.optional_str(outcome.get("error"))
    if error is not None:
        entry["error"] = error
    return entry


def _metric_sort_key(outcome: dict[str, object]) -> float:
    metric = ar_types.optional_float(outcome.get("metric"))
    return metric if metric is not None else float("-inf")


def _resolve_believed_best(
    experiment: ExperimentRecord,
    state: dict[str, object],
    declared: str | None,
    fallback_config: str,
    round_number: int,
) -> tuple[dict[str, object], int | None]:
    """Persist the model-declared trusted recipe, enriched with its seed-trio stats.

    Falls back to the champion config, then this round's config, so the field is always populated.
    The change-round counter feeds the plateau signal; belief stays the model's to declare.
    """
    champion = state.get("champion")
    champion_config = champion.get("config") if isinstance(champion, dict) else None
    config_name = declared or (champion_config if isinstance(champion_config, str) else None) or fallback_config
    configs = aggregate.load_config_cache(memory.configs_dir(experiment))
    seed_path = boundary.seed_change_path(experiment)
    groups = aggregate.aggregate_recipes(memory.journal_all(experiment), configs=configs, seed_path=seed_path)
    group = aggregate.group_for_config(groups, config_name, configs, seed_path=seed_path)
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


# --------------------------------------------------------------------------- #
# Round driver
# --------------------------------------------------------------------------- #


def _run_one_round(*, root: Path, experiment_id: str, state: dict[str, object]) -> ar_types.ResearchRoundResult:
    """Decide, execute, finalize - with one retry when the boundary refused every seed.

    A rejection, a duplicate, or an unparseable response burns a round on an error the model can fix
    in seconds, so the token goes back as `last_error`, the context is rebuilt, and the model is
    asked once more. It is never repaired here, a second failure is recorded and counted exactly as a
    single failure was, and a partial seed failure is not a refused round and never retries. Anything
    escaping decide or execute becomes one failed outcome through the same finalize.
    """
    experiment = get_experiment(store_root=root, experiment_id=experiment_id)
    round_number = ar_types.as_int(state.get("next_round_number"), default=1)
    round_label = memory.round_label(round_number)
    memory.rounds_dir(experiment).mkdir(parents=True, exist_ok=True)
    report = _safe_report(root=root, experiment_id=experiment_id)
    baseline = not context.has_scored_primary_row(report)
    # Fixed before the try so a terminal failure is journaled under the round's real action.
    action: ar_types.ResearchAction = "baseline" if baseline else "run"
    memo: str | None = None
    experiment_markdown: str | None = None
    retry_token: str | None = None
    try:
        for attempt in (1, 2):
            try:
                decision, memo, experiment_markdown = _decide(root, experiment, state, report, round_label, baseline)
            except AgenticResearchValidationError as exc:
                # A response the parser refuses is the same kind of error as a refused proposal, so
                # it spends the round's one retry instead of the whole round.
                if baseline or attempt == 2:
                    raise
                retry_token = f"llm_response_invalid:{exc}"
                state["last_error"] = retry_token
                continue
            outcomes = _execute_round(root, experiment, state, round_number, round_label, action, decision)
            if baseline or attempt == 2 or not all(outcome.get("rejected") for outcome in outcomes):
                break
            retry_token = ar_types.optional_str(outcomes[0].get("error"))
            state["last_error"] = retry_token
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        message = str(exc) or exc.__class__.__name__
        memo, experiment_markdown = None, None
        parent = ar_types.optional_str(state.get("_pending_parent"))
        decision = ar_types.ResearchDecision("run", message, "", None, parent, (), None)
        outcomes = [
            {
                "status": "skipped" if isinstance(exc, AgenticResearchDuplicateCandidate) else "failed",
                "run_id": ar_types.optional_str(state.get("_pending_run_id")),
                "changes": _take_pending_changes(state),
                "learning": f"Round skipped: {message}",
                "error": message,
                "llm": ar_types.optional_str(state.get("_pending_llm")),
            }
        ]
    return _finalize_round(
        experiment, state, round_number, round_label, action, decision, outcomes, memo, experiment_markdown, retry_token
    )


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def get_research_status(*, store_root: str | Path = ".numereng", experiment_id: str) -> ar_types.ResearchStatusResult:
    root = resolve_store_root(store_root)
    experiment = get_experiment(store_root=root, experiment_id=experiment_id)
    state = memory.load_state(memory.state_path(experiment)) or memory.initial_state(experiment)
    auto_dir, journal = memory.agentic_dir(experiment), memory.journal_path(experiment)
    return ar_types.ResearchStatusResult(
        experiment_id=experiment.experiment_id,
        status=ar_types.status_value(state.get("status")),
        next_round_number=ar_types.as_int(state.get("next_round_number"), default=1),
        total_rounds_completed=ar_types.as_int(state.get("total_rounds_completed"), default=0),
        last_checkpoint=str(state.get("last_checkpoint") or "initialized"),
        last_round_label=ar_types.optional_str(state.get("last_round_label")),
        last_run_id=ar_types.optional_str(state.get("last_run_id")),
        stop_reason=ar_types.optional_str(state.get("stop_reason")),
        best_overall=context.best_run_from_report(_safe_report(root=root, experiment_id=experiment_id)),
        agentic_research_dir=auto_dir,
        state_path=auto_dir / ar_types.STATE_FILENAME,
        trace_path=journal,
        decision_path=journal,
        strategy_path=memory.strategy_path(experiment),
    )


def run_research(
    *, store_root: str | Path = ".numereng", experiment_id: str, max_rounds: int = 1
) -> ar_types.ResearchRunResult:
    if max_rounds < 1:
        raise AgenticResearchValidationError("agentic_research_max_rounds_invalid")
    root = resolve_store_root(store_root)
    experiment = get_experiment(store_root=root, experiment_id=experiment_id)
    if experiment.status == "archived":
        raise AgenticResearchValidationError("agentic_research_experiment_archived")
    boundary.assert_scoring_paths_frozen(experiment)
    boundary.program_allowed_paths(experiment)  # fail a misconfigured allowlist at round 0, not mid-round
    _prevalidate_seed_configs(experiment)
    _prevalidate_prompt_placeholders(experiment)
    memory.agentic_dir(experiment).mkdir(parents=True, exist_ok=True)
    state = memory.load_state(memory.state_path(experiment)) or memory.initial_state(experiment)
    state.update({"status": "running", "stop_reason": None})
    _save(experiment, state)

    rounds: list[ar_types.ResearchRoundResult] = []
    try:
        for _ in range(max_rounds):
            if _is_terminal_stop(state):
                break
            rounds.append(_run_one_round(root=root, experiment_id=experiment.experiment_id, state=state))
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
