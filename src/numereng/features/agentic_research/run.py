"""Minimal agentic research loop: prompt LLM for config mutations, then run numereng."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from numereng.config.training import TrainingConfig, load_training_config_json
from numereng.features.experiments import (
    ExperimentError,
    ExperimentRecord,
    ExperimentReport,
    ExperimentReportRow,
    ExperimentTrainResult,
    get_experiment,
    report_experiment,
    score_experiment_round,
    train_experiment,
)
from numereng.features.store import index_run, resolve_store_root
from numereng.features.telemetry import bind_launch_metadata
from numereng.features.training.errors import TrainingError
from numereng.features.training.run_store import compute_config_hash
from numereng.platform.clients.openrouter import OpenRouterClient, OpenRouterConfig, load_openrouter_config
from numereng.platform.errors import OpenRouterClientError

ResearchStatus = Literal["initialized", "running", "interrupted", "stopped", "failed"]
ResearchAction = Literal["baseline", "run", "stop"]

PROGRAM_PATH = Path(__file__).with_name("PROGRAM.md")
CUSTOM_PROGRAM_DIR = Path(__file__).with_name("custom_programs")
PROGRAM_METADATA_KEY = "agentic_research_program"
ALLOWED_PATHS_METADATA_KEY = "agentic_research_allowed_change_paths"
VALUE_CAPS_METADATA_KEY = "agentic_research_value_caps"
PHASES_METADATA_KEY = "agentic_research_phases"
_MISSING_SENTINEL: object = object()
PHASE_IMPROVEMENT_THRESHOLD = 3e-4
CANONICAL_SEED_TRIO: tuple[int, ...] = (42, 17, 99)
CONFIRMATION_PROMOTION_THRESHOLD = 3e-4
ARTIFACT_ROTATION_METADATA_KEY = "agentic_research_artifact_rotation"
ARTIFACT_ROTATION_HEAVY_NAMES: frozenset[str] = frozenset({"predictions.parquet", "meta.parquet"})
ARTIFACT_ROTATION_HEAVY_SIZE_THRESHOLD = 1_000_000
ARTIFACT_ROTATION_PRESERVE_NAMES: frozenset[str] = frozenset(
    {
        "metrics.json",
        "run.json",
        "score_provenance.json",
        "runtime.json",
        "manifest.json",
    }
)
ARTIFACT_ROTATION_RECENT_ROUND_GRACE = 10
CONSECUTIVE_FAILURE_BAIL_THRESHOLD = 5
TRIED_SIGNATURES_WINDOW = 100
AGENTIC_DIRNAME = "agentic_research"
STATE_FILENAME = "state.json"
TRACE_FILENAME = "trace.jsonl"
PRIMARY_METRIC = "bmc_last_200_eras.mean"
PRIMARY_METRIC_FIELD = "bmc_last_200_eras_mean"
SECONDARY_METRICS: tuple[tuple[str, str], ...] = (
    ("bmc.mean", "bmc_mean"),
    ("corr.mean", "corr_mean"),
    ("mmc.mean", "mmc_mean"),
    ("cwmm.mean", "cwmm_mean"),
)
SCORING_STAGE = "post_training_core"
RUN_PLAN_FIELDS = ("plan_index", "round", "seed", "target", "horizon", "config_path", "score_stage_default")
MAX_CONTEXT_CHARS = 12_000
CODEX_TIMEOUT_SECONDS = 600.0
ALLOWED_CHANGE_PATHS = (
    "data.feature_set",
    "data.target_col",
    "data.scoring_targets",
    "data.target_horizon",
    "preprocessing.nan_missing_all_twos",
    "preprocessing.missing_value",
    "model.type",
    "model.module_path",
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
)


class AgenticResearchError(Exception):
    """Base error for agentic research workflows."""


class AgenticResearchValidationError(AgenticResearchError):
    """Raised when an LLM decision or local research state is invalid."""


@dataclass(frozen=True)
class ResearchBestRun:
    """Best known run for the primary research metric."""

    experiment_id: str | None = None
    run_id: str | None = None
    bmc_last_200_eras_mean: float | None = None
    bmc_mean: float | None = None
    corr_mean: float | None = None
    mmc_mean: float | None = None
    cwmm_mean: float | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class ResearchRoundResult:
    """One completed or terminal research-loop round."""

    round_number: int
    round_label: str
    action: ResearchAction
    status: str
    config_path: Path | None
    run_id: str | None
    metric_value: float | None
    learning: str
    artifact_dir: Path


@dataclass(frozen=True)
class ResearchStatusResult:
    """Current lightweight state for one experiment's research loop."""

    experiment_id: str
    status: ResearchStatus
    next_round_number: int
    total_rounds_completed: int
    last_checkpoint: str
    last_round_label: str | None
    last_run_id: str | None
    stop_reason: str | None
    best_overall: ResearchBestRun
    agentic_research_dir: Path
    state_path: Path
    trace_path: Path
    decision_path: Path
    program_path: Path


@dataclass(frozen=True)
class ResearchRunResult:
    """Result for a foreground research-loop invocation."""

    experiment_id: str
    status: ResearchStatus
    next_round_number: int
    total_rounds_completed: int
    last_checkpoint: str
    stop_reason: str | None
    best_overall: ResearchBestRun
    rounds: tuple[ResearchRoundResult, ...]
    interrupted: bool = False


@dataclass(frozen=True)
class ResearchChange:
    """One validated config change requested by the LLM."""

    path: str
    value: object
    reason: str


@dataclass(frozen=True)
class ResearchDecision:
    """Parsed LLM decision for the next research step."""

    action: Literal["run", "stop"]
    learning: str
    belief_update: str
    next_hypothesis: str | None
    parent_config: str | None
    changes: tuple[ResearchChange, ...]
    stop_reason: str | None


@dataclass(frozen=True)
class ResearchLLMResponse:
    """Validated LLM response: research form plus cumulative round memo."""

    decision: ResearchDecision
    round_markdown: str
    experiment_markdown: str | None


def get_research_status(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str,
) -> ResearchStatusResult:
    """Return current research-loop state, synthesizing a blank state if needed."""
    root = resolve_store_root(store_root)
    experiment = get_experiment(store_root=root, experiment_id=experiment_id)
    state = _load_state(_state_path(experiment)) or _initial_state(experiment)
    best = _best_run_from_report(_safe_report(root=root, experiment_id=experiment.experiment_id))
    return _status_result(experiment=experiment, state=state, best=best)


def run_research(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str,
    max_rounds: int = 1,
) -> ResearchRunResult:
    """Run one or more config-mutation research rounds in the foreground."""
    if max_rounds < 1:
        raise AgenticResearchValidationError("agentic_research_max_rounds_invalid")

    root = resolve_store_root(store_root)
    experiment = get_experiment(store_root=root, experiment_id=experiment_id)
    if experiment.status == "archived":
        raise AgenticResearchValidationError("agentic_research_experiment_archived")

    auto_dir = _agentic_dir(experiment)
    auto_dir.mkdir(parents=True, exist_ok=True)
    state = _load_state(_state_path(experiment)) or _initial_state(experiment)
    state["status"] = "running"
    state["stop_reason"] = None
    _save_state(experiment, state)

    rounds: list[ResearchRoundResult] = []
    try:
        for _ in range(max_rounds):
            if _is_terminal_stop(state):
                break
            try:
                result = _run_one_round(root=root, experiment_id=experiment.experiment_id, state=state)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                result = _record_failed_round(experiment=experiment, state=state, error=exc)
            rounds.append(result)
            experiment = get_experiment(store_root=root, experiment_id=experiment_id)
            state = _load_state(_state_path(experiment)) or _initial_state(experiment)
            if result.action == "stop":
                break
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        state["stop_reason"] = "keyboard_interrupt"
        state["last_checkpoint"] = "interrupted"
        state["updated_at"] = _utc_now_iso()
        _save_state(experiment, state)
        return _run_result(experiment=experiment, state=state, rounds=rounds, interrupted=True)
    except Exception:
        state["status"] = "failed"
        state["last_checkpoint"] = "failed"
        state["updated_at"] = _utc_now_iso()
        _save_state(experiment, state)
        raise

    if state.get("status") == "running":
        state["status"] = "stopped"
        state["stop_reason"] = "max_rounds_reached"
        state["last_checkpoint"] = "stopped"
        state["updated_at"] = _utc_now_iso()
        _save_state(experiment, state)
    return _run_result(experiment=experiment, state=state, rounds=rounds, interrupted=False)


def program_markdown() -> str:
    """Return the active research prompt."""
    return PROGRAM_PATH.read_text(encoding="utf-8")


def _run_one_round(*, root: Path, experiment_id: str, state: dict[str, object]) -> ResearchRoundResult:
    experiment = get_experiment(store_root=root, experiment_id=experiment_id)
    round_number = _as_int(state.get("next_round_number"), default=1)
    round_label = f"r{round_number:03d}"
    artifact_dir = _rounds_dir(experiment)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    report = _safe_report(root=root, experiment_id=experiment_id)
    if not _has_scored_primary_row(report):
        return _run_baseline_round(
            root=root,
            experiment=experiment,
            state=state,
            round_number=round_number,
            round_label=round_label,
            artifact_dir=artifact_dir,
        )

    context = _build_context(root=root, experiment=experiment, report=report, state=state)
    program_path = _program_path(experiment)
    prompt = _render_prompt(context, program_path=program_path)
    _append_trace(
        experiment,
        round_number=round_number,
        round_label=round_label,
        event="prompt_rendered",
        payload={"program_path": str(program_path), "prompt": prompt},
    )

    try:
        raw_response, model_source = _call_research_llm(
            prompt=prompt,
            artifact_dir=artifact_dir,
            round_label=round_label,
        )
    except Exception as exc:
        _append_trace(
            experiment,
            round_number=round_number,
            round_label=round_label,
            event="llm_call_failed",
            payload={"error": str(exc)},
        )
        _write_failure_debug(artifact_dir=artifact_dir, round_label=round_label, prompt=prompt, error=str(exc))
        raise
    _append_trace(
        experiment,
        round_number=round_number,
        round_label=round_label,
        event="llm_response",
        payload={"model_source": model_source, "raw_response": raw_response},
    )
    try:
        llm_response = _parse_llm_response(raw_response)
    except Exception as exc:
        _append_trace(
            experiment,
            round_number=round_number,
            round_label=round_label,
            event="decision_parse_failed",
            payload={"raw_response": raw_response, "error": str(exc)},
        )
        _write_failure_debug(
            artifact_dir=artifact_dir,
            round_label=round_label,
            prompt=prompt,
            raw_response=raw_response,
            error=str(exc),
        )
        raise
    decision = llm_response.decision
    decision_payload = _decision_payload(decision, model_source=model_source)
    # Stash so _record_failed_round can surface what the LLM proposed when a
    # post-parse step (materialization, train, score) blows up. Cleared on
    # success in _train_score_record_round / _record_stop_round.
    state["pending_decision"] = decision_payload
    _append_trace(
        experiment,
        round_number=round_number,
        round_label=round_label,
        event="decision_parsed",
        payload={"decision": decision_payload},
    )

    if decision.action == "stop":
        result = _record_stop_round(
            experiment=experiment,
            state=state,
            decision=decision,
            decision_payload=decision_payload,
            round_number=round_number,
            round_label=round_label,
            artifact_dir=artifact_dir,
            round_markdown=llm_response.round_markdown,
        )
        bytes_written = _write_experiment_markdown(experiment, llm_response.experiment_markdown)
        if bytes_written > 0:
            _append_trace(
                experiment,
                round_number=round_number,
                round_label=round_label,
                event="experiment_markdown_updated",
                payload={"bytes_written": bytes_written},
            )
        _append_trace(
            experiment,
            round_number=round_number,
            round_label=round_label,
            event="round_stopped",
            payload={"stop_reason": decision.stop_reason},
        )
        return result

    try:
        config_path = _materialize_decision_config(
            experiment=experiment,
            round_label=round_label,
            decision=decision,
            state=state,
        )
    except Exception as exc:
        _append_trace(
            experiment,
            round_number=round_number,
            round_label=round_label,
            event="config_materialization_failed",
            payload={"decision": decision_payload, "error": str(exc)},
        )
        raise
    _append_trace(
        experiment,
        round_number=round_number,
        round_label=round_label,
        event="config_written",
        payload={"config_path": str(config_path)},
    )
    result = _train_score_record_round(
        root=root,
        experiment=experiment,
        state=state,
        round_number=round_number,
        round_label=round_label,
        action="run",
        config_path=config_path,
        artifact_dir=artifact_dir,
        learning=decision.learning,
        decision_payload=decision_payload,
        round_markdown=llm_response.round_markdown,
    )
    bytes_written = _write_experiment_markdown(experiment, llm_response.experiment_markdown)
    if bytes_written > 0:
        _append_trace(
            experiment,
            round_number=round_number,
            round_label=round_label,
            event="experiment_markdown_updated",
            payload={"bytes_written": bytes_written},
        )
    _append_trace(
        experiment,
        round_number=round_number,
        round_label=round_label,
        event="round_completed",
        payload={
            "config_path": str(result.config_path) if result.config_path is not None else None,
            "run_id": result.run_id,
            "metric_value": result.metric_value,
        },
    )
    return result


_TRAIN_RUN_DIR_NOT_FRESH_PREFIX = "training_run_dir_not_fresh:"


def _reuse_finished_run_on_hash_collision(
    *, root: Path, experiment: ExperimentRecord, exc: TrainingError
) -> ExperimentTrainResult | None:
    """Reuse an existing FINISHED run dir when a new round's config hashes to it.

    Runs are content-addressed: byte-identical training configs across experiments
    share `.numereng/runs/<run_id>/`. The training service's freshness check then
    rejects a second attempt. When the prior run is FINISHED, treat that run as
    the result of this round instead of failing the round. Also link the reused
    run into this experiment's manifest so downstream scoring/report flows find
    it via the normal resolver."""
    msg = str(exc)
    if not msg.startswith(_TRAIN_RUN_DIR_NOT_FRESH_PREFIX):
        return None
    parts = msg.split(":")
    if len(parts) < 3:
        return None
    run_id = parts[1]
    run_dir = root / "runs" / run_id
    run_json_path = run_dir / "run.json"
    if not run_json_path.is_file():
        return None
    try:
        run_manifest = json.loads(run_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if run_manifest.get("status") != "FINISHED":
        return None
    predictions_name = None
    output_block = run_manifest.get("output")
    if isinstance(output_block, dict):
        predictions_name = output_block.get("predictions_name")
    if not isinstance(predictions_name, str) or not predictions_name:
        predictions_name = "predictions"
    predictions_path = run_dir / "artifacts" / "predictions" / f"{predictions_name}.parquet"
    results_path = run_dir / "results.json"
    _link_reused_run_to_experiment(experiment=experiment, run_id=run_id)
    try:
        index_run(store_root=root, run_id=run_id)
    except Exception:
        pass
    return ExperimentTrainResult(
        experiment_id=experiment.experiment_id,
        run_id=run_id,
        predictions_path=predictions_path,
        results_path=results_path,
    )


def _link_reused_run_to_experiment(*, experiment: ExperimentRecord, run_id: str) -> None:
    """Append run_id to the experiment manifest's runs list if not already present."""
    manifest_path = experiment.manifest_path
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        runs = []
    if run_id in runs:
        return
    runs.append(run_id)
    manifest["runs"] = runs
    if manifest.get("status") == "draft":
        manifest["status"] = "active"
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    _write_json(manifest_path, manifest)


def _run_baseline_round(
    *,
    root: Path,
    experiment: ExperimentRecord,
    state: dict[str, object],
    round_number: int,
    round_label: str,
    artifact_dir: Path,
) -> ResearchRoundResult:
    parent_path = _first_config_path(experiment)
    config_dir = experiment.manifest_path.parent / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = _unique_config_path(config_dir, _round_config_filename(round_label))
    payload = load_training_config_json(parent_path)
    _write_json(config_path, payload)
    learning = f"Baseline round from `{parent_path.name}` before asking the LLM for mutations."
    decision_payload: dict[str, object] = {
        "action": "baseline",
        "parent_config": parent_path.name,
        "generated_config": config_path.name,
        "changes": [],
        "learning": learning,
    }
    return _train_score_record_round(
        root=root,
        experiment=experiment,
        state=state,
        round_number=round_number,
        round_label=round_label,
        action="baseline",
        config_path=config_path,
        artifact_dir=artifact_dir,
        learning=learning,
        decision_payload=decision_payload,
        round_markdown=_baseline_round_markdown(
            round_label=round_label,
            parent_name=parent_path.name,
            config_name=config_path.name,
        ),
    )


def _baseline_round_markdown(*, round_label: str, parent_name: str, config_name: str) -> str:
    """Render the baseline round using the same 5-section template as LLM rounds
    so the rounds dir has one consistent shape that scripts can pattern-match against."""
    return (
        f"# {round_label} Research State\n\n"
        f"## Phase\n\n"
        f"Initial phase - baseline establishment, no LLM mutation yet.\n\n"
        f"## What this decision tests\n\n"
        f"Whether `{parent_name}` produces a positive primary metric on the configured "
        f"target. Establishes the incumbent before any LLM-driven hypothesis is run.\n\n"
        f"## Evidence cited\n\n"
        f"No prior rounds. This is the experiment baseline.\n\n"
        f"## What changed and why\n\n"
        f"`{config_name}` is a full copy of `{parent_name}` with no mutation; future "
        f"rounds will mutate from this anchor.\n\n"
        f"## Open questions and caveats\n\n"
        f"All open frontiers are open. The first LLM decision should propose a single-axis "
        f"probe inside the configured cost envelope.\n"
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
    artifact_dir: Path,
    learning: str,
    decision_payload: dict[str, object],
    round_markdown: str,
) -> ResearchRoundResult:
    round_started_at = time.monotonic()
    reused_existing_run = False
    with bind_launch_metadata(source="feature.agentic_research.train", operation_type="run", job_type="run"):
        try:
            trained = train_experiment(store_root=root, experiment_id=experiment.experiment_id, config_path=config_path)
        except TrainingError as exc:
            reused = _reuse_finished_run_on_hash_collision(root=root, experiment=experiment, exc=exc)
            if reused is None:
                raise
            trained = reused
            reused_existing_run = True
    # Must run before score_experiment_round: score reads run_plan.csv to resolve
    # the round's configs (configs/config_NNN.json does not match the fallback glob).
    _record_round_config_in_run_plan(experiment=experiment, round_label=round_label, config_path=config_path)
    # Reused runs were already scored when first trained; the resolver matches runs to
    # rounds by config stem in run.json, and a content-shared run's stem points at the
    # original experiment's config name. Skip the score step to avoid a false miss.
    if not reused_existing_run:
        with bind_launch_metadata(source="feature.agentic_research.score_round", operation_type="run", job_type="run"):
            score_experiment_round(
                store_root=root,
                experiment_id=experiment.experiment_id,
                round=round_label,
                stage=SCORING_STAGE,
            )
    round_seconds = max(0.0, time.monotonic() - round_started_at)

    report = _safe_report(root=root, experiment_id=experiment.experiment_id)
    row = _row_for_run(report, trained.run_id)
    metric_value = getattr(row, PRIMARY_METRIC_FIELD) if row is not None else None
    if metric_value is None:
        metric_value = _run_primary_metric_from_disk(root=root, run_id=trained.run_id)
    best = _best_run_from_report(report)
    secondary_metrics = _load_secondary_metrics_from_disk(root=root, run_id=trained.run_id)
    diff_vs_parent_markdown = _render_diff_vs_parent(experiment=experiment, decision_payload=decision_payload)
    confirmation_round = _is_confirmation_round(decision_payload)
    champion_seed42_before = _champion_seed42_metric(state)
    round_payload: dict[str, object] = {
        "round_number": round_number,
        "round_label": round_label,
        "action": action,
        "status": "completed",
        "config_path": _relative_to_experiment(experiment, config_path),
        "run_id": trained.run_id,
        "metric_value": metric_value,
        "learning": learning,
        "decision": decision_payload,
        "completed_at": _utc_now_iso(),
        "wall_time_seconds": round_seconds,
        "secondary_metrics": secondary_metrics,
        "diff_vs_parent_markdown": diff_vs_parent_markdown,
        "confirmation_round": confirmation_round,
        "champion_seed42_before": champion_seed42_before,
    }
    state.pop("pending_decision", None)
    state.update(
        {
            "status": "running",
            "next_round_number": round_number + 1,
            "total_rounds_completed": _as_int(state.get("total_rounds_completed"), default=0) + 1,
            "last_checkpoint": "round_completed",
            "last_round_label": round_label,
            "last_run_id": trained.run_id,
            "best_overall": asdict(best),
            "failed_rounds_counter": 0,
            "updated_at": _utc_now_iso(),
        }
    )
    _accumulate_phase_cost(state, round_seconds)
    typed_metric = _coerce_metric(metric_value)
    _update_phase_progress(state=state, metric_value=typed_metric)
    _append_tried_signature(
        state,
        _extract_lgbm_signature(
            config_path=config_path,
            round_label=round_label,
            run_id=trained.run_id,
            primary_metric=typed_metric,
            action=action,
        ),
    )
    promotion_payload: dict[str, object] | None = None
    if _is_confirmation_round(decision_payload):
        parent_config = str(decision_payload.get("parent_config"))
        changes_list = decision_payload.get("changes")
        seed_value = (
            changes_list[0].get("value")
            if isinstance(changes_list, list) and changes_list and isinstance(changes_list[0], dict)
            else None
        )
        if isinstance(seed_value, int) and not isinstance(seed_value, bool):
            metric_for_record = float(metric_value) if isinstance(metric_value, (int, float)) else None
            _record_confirmation_attempt(
                state=state,
                parent_config=parent_config,
                seed=seed_value,
                run_id=trained.run_id,
                metric_value=metric_for_record,
                round_number=round_number,
            )
            _append_trace(
                experiment,
                round_number=round_number,
                round_label=round_label,
                event="confirmation_attempt",
                payload={
                    "parent_config": parent_config,
                    "seed": seed_value,
                    "run_id": trained.run_id,
                    "primary_metric": metric_for_record,
                },
            )
            promotion = _maybe_promote_confirmation(state=state, parent_config=parent_config, round_number=round_number)
            if promotion is not None:
                promotion_payload = promotion
                _append_trace(
                    experiment,
                    round_number=round_number,
                    round_label=round_label,
                    event="confirmation_promoted",
                    payload=promotion,
                )
    elif action == "run":
        # Auto-credit the discovery seed against the freshly-materialized config.
        # Without this, the seed trio can never complete: the LLM cannot re-run
        # the generated config with random_state=42 because the duplicate-by-hash
        # gate would reject it.
        discovery_seed = _config_random_state(config_path)
        if (
            isinstance(discovery_seed, int)
            and not isinstance(discovery_seed, bool)
            and discovery_seed in CANONICAL_SEED_TRIO
        ):
            _record_confirmation_attempt(
                state=state,
                parent_config=config_path.name,
                seed=discovery_seed,
                run_id=trained.run_id,
                metric_value=typed_metric,
                round_number=round_number,
            )
            _append_trace(
                experiment,
                round_number=round_number,
                round_label=round_label,
                event="discovery_seed_auto_credited",
                payload={
                    "generated_config": config_path.name,
                    "seed": discovery_seed,
                    "run_id": trained.run_id,
                    "primary_metric": typed_metric,
                },
            )
            promotion = _maybe_promote_confirmation(
                state=state, parent_config=config_path.name, round_number=round_number
            )
            if promotion is not None:
                promotion_payload = promotion
                _append_trace(
                    experiment,
                    round_number=round_number,
                    round_label=round_label,
                    event="confirmation_promoted",
                    payload=promotion,
                )
    phase_snapshot = _phase_snapshot_for_md(experiment=experiment, state=state)
    round_payload["promotion"] = promotion_payload
    round_payload["phase_snapshot"] = phase_snapshot
    _write_llm_round_markdown(
        artifact_dir=artifact_dir,
        round_label=round_label,
        round_markdown=round_markdown,
        round_payload=round_payload,
    )
    _append_decision_log(_decision_log_path(experiment), round_payload)
    transition_payload = _maybe_transition_phase(
        experiment=experiment, state=state, round_number=round_number, report=report
    )
    if transition_payload is not None:
        kind = transition_payload.get("transition")
        if kind == "misconfigured":
            event = "phase_misconfigured"
        elif kind == "blocked_no_champion":
            event = "phase_blocked_no_champion"
        else:
            event = "phase_transition"
        _append_trace(
            experiment,
            round_number=round_number,
            round_label=round_label,
            event=event,
            payload=transition_payload,
        )
    rotation_payload = _rotate_run_artifacts(
        root=root, experiment=experiment, state=state, last_round_number=round_number
    )
    if rotation_payload is not None:
        _append_trace(
            experiment,
            round_number=round_number,
            round_label=round_label,
            event="run_artifacts_rotated",
            payload=rotation_payload,
        )
    _save_state(experiment, state)
    return ResearchRoundResult(
        round_number=round_number,
        round_label=round_label,
        action=action,
        status="completed",
        config_path=config_path,
        run_id=trained.run_id,
        metric_value=metric_value,
        learning=learning,
        artifact_dir=artifact_dir,
    )


def _record_stop_round(
    *,
    experiment: ExperimentRecord,
    state: dict[str, object],
    decision: ResearchDecision,
    decision_payload: dict[str, object],
    round_number: int,
    round_label: str,
    artifact_dir: Path,
    round_markdown: str,
) -> ResearchRoundResult:
    learning = decision.learning or decision.stop_reason or "LLM stopped the research loop."
    diff_vs_parent_markdown = _render_diff_vs_parent(experiment=experiment, decision_payload=decision_payload)
    payload: dict[str, object] = {
        "round_number": round_number,
        "round_label": round_label,
        "action": "stop",
        "status": "stopped",
        "run_id": None,
        "metric_value": None,
        "learning": learning,
        "stop_reason": decision.stop_reason,
        "decision": decision_payload,
        "completed_at": _utc_now_iso(),
        "diff_vs_parent_markdown": diff_vs_parent_markdown,
    }
    _write_llm_round_markdown(
        artifact_dir=artifact_dir,
        round_label=round_label,
        round_markdown=round_markdown,
        round_payload=payload,
    )
    _append_decision_log(_decision_log_path(experiment), payload)
    state.pop("pending_decision", None)
    state.update(
        {
            "status": "stopped",
            "stop_reason": f"llm_stop:{decision.stop_reason or 'no_next_run'}",
            "last_checkpoint": "llm_stopped",
            "updated_at": _utc_now_iso(),
        }
    )
    _save_state(experiment, state)
    return ResearchRoundResult(
        round_number=round_number,
        round_label=round_label,
        action="stop",
        status="stopped",
        config_path=None,
        run_id=None,
        metric_value=None,
        learning=learning,
        artifact_dir=artifact_dir,
    )


def _record_failed_round(
    *,
    experiment: ExperimentRecord,
    state: dict[str, object],
    error: Exception,
) -> ResearchRoundResult:
    round_number = _as_int(state.get("next_round_number"), default=1)
    round_label = f"r{round_number:03d}"
    artifact_dir = _rounds_dir(experiment)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    message = str(error) or error.__class__.__name__
    error_class = error.__class__.__name__
    learning = f"Round skipped: {message}"
    pending_decision_raw = state.pop("pending_decision", None)
    pending_decision = pending_decision_raw if isinstance(pending_decision_raw, dict) else None
    payload: dict[str, object] = {
        "round_number": round_number,
        "round_label": round_label,
        "action": "run",
        "status": "failed",
        "run_id": None,
        "metric_value": None,
        "learning": learning,
        "error": message,
        "error_class": error_class,
        "completed_at": _utc_now_iso(),
    }
    if pending_decision is not None:
        payload["pending_decision"] = pending_decision
    _append_decision_log(_decision_log_path(experiment), payload)
    _append_trace(
        experiment,
        round_number=round_number,
        round_label=round_label,
        event="round_failed",
        payload={"error": message},
    )
    _write_failure_round_markdown(
        artifact_dir=artifact_dir,
        round_label=round_label,
        round_payload=payload,
    )
    failures = _as_int(state.get("failed_rounds_counter"), default=0) + 1
    state.update(
        {
            "status": "running",
            "next_round_number": round_number + 1,
            "last_checkpoint": "round_failed",
            "failed_rounds_counter": failures,
            "updated_at": _utc_now_iso(),
        }
    )
    if failures >= CONSECUTIVE_FAILURE_BAIL_THRESHOLD:
        state["status"] = "stopped"
        state["stop_reason"] = f"consecutive_failures:{failures}"
        state["last_checkpoint"] = "consecutive_failures_bail"
        _append_trace(
            experiment,
            round_number=round_number,
            round_label=round_label,
            event="round_failed_with_retry_exhausted",
            payload={"failed_rounds_counter": failures, "threshold": CONSECUTIVE_FAILURE_BAIL_THRESHOLD},
        )
    _save_state(experiment, state)
    return ResearchRoundResult(
        round_number=round_number,
        round_label=round_label,
        action="run",
        status="failed",
        config_path=None,
        run_id=None,
        metric_value=None,
        learning=learning,
        artifact_dir=artifact_dir,
    )


def _build_context(
    *,
    root: Path,
    experiment: ExperimentRecord,
    report: ExperimentReport | None,
    state: dict[str, object],
) -> dict[str, object]:
    phase_caps = _current_phase_caps(experiment=experiment, state=state)
    return {
        "objective": {
            "primary_metric": PRIMARY_METRIC_FIELD,
            "tie_break": "bmc_mean",
            "sanity_checks": ["corr_mean", "mmc_mean", "cwmm_mean"],
            "scoring_stage": SCORING_STAGE,
        },
        "allowed_change_paths": list(ALLOWED_CHANGE_PATHS),
        "experiment": {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "status": experiment.status,
            "hypothesis": experiment.hypothesis,
            "tags": list(experiment.tags),
            "champion_run_id": experiment.champion_run_id,
            "run_count": len(experiment.runs),
        },
        "phase": state.get("phase"),
        "phase_history": state.get("phase_history", []),
        "phase_value_caps": {path: list(bounds) for path, bounds in phase_caps.items()},
        "cost_summary": _cost_summary(state),
        "canonical_seed_trio": list(CANONICAL_SEED_TRIO),
        "confirmations": _recent_confirmations(state, limit=30),
        "confirmed_champion": state.get("confirmed_champion"),
        "tried_signatures": state.get("tried_signatures", []),
        "state": state,
        "configs": _config_context(experiment),
        "report": _report_context(report),
        "recent_rounds": _recent_decisions(_decision_log_path(experiment), limit=8),
        "latest_round_markdown": _latest_round_markdown(experiment),
        "experiment_notes": _read_text(_experiment_markdown_path(experiment), limit=MAX_CONTEXT_CHARS),
        "research_memory": _read_text(root / "notes" / "__RESEARCH_MEMORY__" / "CURRENT.md", limit=MAX_CONTEXT_CHARS),
    }


def _render_prompt(context: dict[str, object], *, program_path: Path = PROGRAM_PATH) -> str:
    context_json = json.dumps(context, indent=2, sort_keys=True, default=str)
    return program_path.read_text(encoding="utf-8").replace("{{CONTEXT_JSON}}", context_json)


def _call_research_llm(*, prompt: str, artifact_dir: Path, round_label: str) -> tuple[str, str]:
    config = load_openrouter_config()
    if config.active_model_source == "openrouter":
        return _call_openrouter(prompt, config=config), "openrouter"
    return (
        _call_codex_exec(prompt=prompt, artifact_dir=artifact_dir, round_label=round_label, config=config),
        "codex-exec",
    )


def _call_openrouter(prompt: str, *, config: OpenRouterConfig) -> str:
    payload: dict[str, object] = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    if config.active_model_reasoning_effort is not None:
        payload["reasoning"] = {"effort": config.active_model_reasoning_effort}
    try:
        response = OpenRouterClient(timeout_seconds=180.0).chat_completions(payload=payload)
    except OpenRouterClientError as exc:
        raise AgenticResearchError(str(exc)) from exc
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AgenticResearchError("agentic_research_openrouter_response_missing")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise AgenticResearchError("agentic_research_openrouter_content_missing")
    return content


def _call_codex_exec(*, prompt: str, artifact_dir: Path, round_label: str, config: OpenRouterConfig) -> str:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=artifact_dir, prefix=".codex_output_", suffix=".txt", delete=False) as handle:
        output_path = Path(handle.name)
    with tempfile.NamedTemporaryFile(dir=artifact_dir, prefix=".codex_schema_", suffix=".json", delete=False) as handle:
        schema_path = Path(handle.name)
    _write_json(schema_path, _llm_response_schema())
    cmd = [
        _resolve_codex_executable(),
        "exec",
    ]
    if config.active_model is not None:
        cmd.extend(["--model", config.active_model])
    if config.active_model_reasoning_effort is not None:
        cmd.extend(["-c", f'model_reasoning_effort="{config.active_model_reasoning_effort}"'])
    cmd.extend(
        [
            "--skip-git-repo-check",
            "--ephemeral",
            "--output-schema",
            str(schema_path),
            "--json",
            "--color",
            "never",
            "-",
            "-o",
            str(output_path),
        ]
    )
    try:
        try:
            completed = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
                timeout=CODEX_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            error = f"agentic_research_codex_timeout:{int(CODEX_TIMEOUT_SECONDS)}"
            _write_failure_debug(
                artifact_dir=artifact_dir,
                round_label=round_label,
                prompt=prompt,
                codex_stdout=_decode_subprocess_stream(exc.stdout),
                codex_stderr=_decode_subprocess_stream(exc.stderr),
                error=error,
            )
            raise AgenticResearchError(error) from exc
        except FileNotFoundError as exc:
            error = f"agentic_research_codex_executable_missing:{cmd[0]}"
            _write_failure_debug(
                artifact_dir=artifact_dir,
                round_label=round_label,
                prompt=prompt,
                error=error,
            )
            raise AgenticResearchError(error) from exc
        if completed.returncode != 0:
            error = f"agentic_research_codex_failed:{completed.returncode}:{completed.stderr.strip()}"
            _write_failure_debug(
                artifact_dir=artifact_dir,
                round_label=round_label,
                prompt=prompt,
                codex_stdout=completed.stdout,
                codex_stderr=completed.stderr,
                error=error,
            )
            raise AgenticResearchError(error)
        return output_path.read_text(encoding="utf-8")
    finally:
        try:
            output_path.unlink()
        except OSError:
            pass
        try:
            schema_path.unlink()
        except OSError:
            pass


def _decode_subprocess_stream(stream: object) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    if isinstance(stream, str):
        return stream
    return str(stream)


def _resolve_codex_executable() -> str:
    if os.name == "nt":
        return shutil.which("codex.cmd") or shutil.which("codex.exe") or shutil.which("codex") or "codex.cmd"
    return shutil.which("codex") or "codex"


def _parse_llm_response(raw_response: str) -> ResearchLLMResponse:
    payload = _extract_json_object(raw_response)
    decision_form = payload.get("decision_form")
    if not isinstance(decision_form, dict):
        raise AgenticResearchValidationError("agentic_research_decision_form_missing")
    experiment_markdown_raw = payload.get("experiment_markdown")
    if experiment_markdown_raw is not None and not isinstance(experiment_markdown_raw, str):
        raise AgenticResearchValidationError("agentic_research_experiment_markdown_invalid")
    return ResearchLLMResponse(
        decision=_parse_decision_object(cast(dict[str, object], decision_form)),
        round_markdown=_required_str(payload, "round_markdown"),
        experiment_markdown=experiment_markdown_raw,
    )


def _parse_decision_object(payload: dict[str, object]) -> ResearchDecision:
    action_raw = payload.get("action")
    if action_raw not in {"run", "stop"}:
        raise AgenticResearchValidationError("agentic_research_action_invalid")
    action = cast(Literal["run", "stop"], action_raw)
    changes = tuple(_parse_change(item) for item in _as_list(payload.get("changes")))
    decision = ResearchDecision(
        action=action,
        learning=_required_str(payload, "learning"),
        belief_update=_required_str(payload, "belief_update"),
        next_hypothesis=_optional_str(payload.get("next_hypothesis")),
        parent_config=_optional_str(payload.get("parent_config")),
        changes=changes,
        stop_reason=_optional_str(payload.get("stop_reason")),
    )
    if decision.action == "run":
        if decision.parent_config is None:
            raise AgenticResearchValidationError("agentic_research_parent_config_missing")
        if not 1 <= len(decision.changes) <= 5:
            raise AgenticResearchValidationError("agentic_research_change_count_invalid")
    if decision.action == "stop" and not decision.stop_reason:
        raise AgenticResearchValidationError("agentic_research_stop_reason_missing")
    return decision


def _llm_response_schema() -> dict[str, object]:
    value_schema: dict[str, object] = {
        "anyOf": [
            {"type": "string"},
            {"type": "number"},
            {"type": "integer"},
            {"type": "boolean"},
            {"type": "null"},
            {
                "type": "array",
                "items": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "number"},
                        {"type": "integer"},
                        {"type": "boolean"},
                        {"type": "null"},
                    ]
                },
            },
        ]
    }
    decision_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["run", "stop"]},
            "learning": {"type": "string"},
            "belief_update": {"type": "string"},
            "next_hypothesis": {"type": ["string", "null"]},
            "parent_config": {"type": ["string", "null"]},
            "changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "value": value_schema,
                        "reason": {"type": "string"},
                    },
                    "required": ["path", "value", "reason"],
                    "additionalProperties": False,
                },
            },
            "stop_reason": {"type": ["string", "null"]},
        },
        "required": [
            "action",
            "learning",
            "belief_update",
            "next_hypothesis",
            "parent_config",
            "changes",
            "stop_reason",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "decision_form": decision_schema,
            "round_markdown": {"type": "string"},
            "experiment_markdown": {"type": ["string", "null"]},
        },
        "required": ["decision_form", "round_markdown", "experiment_markdown"],
        "additionalProperties": False,
    }


def _parse_change(payload: object) -> ResearchChange:
    if not isinstance(payload, dict):
        raise AgenticResearchValidationError("agentic_research_change_invalid")
    parsed = cast(dict[str, object], payload)
    path = _required_str(parsed, "path")
    if not _change_path_allowed(path):
        raise AgenticResearchValidationError(f"agentic_research_change_path_not_allowed:{path}")
    return ResearchChange(
        path=path,
        value=deepcopy(parsed.get("value")),
        reason=_required_str(parsed, "reason"),
    )


def _materialize_decision_config(
    *,
    experiment: ExperimentRecord,
    round_label: str,
    decision: ResearchDecision,
    state: dict[str, object] | None = None,
) -> Path:
    config_dir = experiment.manifest_path.parent / "configs"
    parent_path = config_dir / str(decision.parent_config)
    if not parent_path.is_file():
        raise AgenticResearchValidationError(f"agentic_research_parent_config_not_found:{decision.parent_config}")
    allowed_paths = _program_allowed_paths(experiment)
    value_caps = (
        _current_phase_caps(experiment=experiment, state=state)
        if state is not None
        else _program_value_caps(experiment)
    )
    for change in decision.changes:
        if not _matches_any_path(change.path, allowed_paths):
            raise AgenticResearchValidationError(f"agentic_research_change_path_not_allowed:{change.path}")
        if change.path in value_caps:
            # None signals "drop this param" (e.g. during family switches). The
            # auto-cleanup below strips conflicting family params, so a None on a
            # capped path is harmless. Skip the range check; treat as a no-op.
            if change.value is None:
                continue
            bounds = value_caps[change.path]
            if isinstance(change.value, bool) or not isinstance(change.value, (int, float)):
                raise AgenticResearchValidationError(f"agentic_research_change_value_not_numeric:{change.path}")
            value = float(change.value)
            lo, hi = bounds
            if not (lo <= value <= hi):
                raise AgenticResearchValidationError(
                    f"agentic_research_change_value_out_of_range:{change.path}:{value}:[{lo},{hi}]"
                )
    payload = load_training_config_json(parent_path)
    for change in decision.changes:
        _assign_dotted(payload, change.path.split("."), deepcopy(change.value))
    _apply_family_switch_cleanup(payload)
    validated = TrainingConfig.model_validate(payload).model_dump(mode="python", exclude_none=True)
    _normalize_lgbm_effective_params(validated)
    candidate_hash = compute_config_hash(validated)
    existing_hashes = _existing_config_hashes(config_dir)
    if candidate_hash in existing_hashes:
        raise AgenticResearchValidationError(f"agentic_research_candidate_duplicate:{candidate_hash[:12]}")
    filename = _round_config_filename(round_label)
    path = _unique_config_path(config_dir, filename)
    _write_json(path, validated)
    return path


def _round_config_filename(round_label: str) -> str:
    suffix = round_label.removeprefix("r")
    return f"config_{suffix}.json" if suffix.isdigit() else f"{round_label}_config.json"


def _relative_to_experiment(experiment: ExperimentRecord, path: Path) -> str:
    """Render `path` relative to the experiment directory so artifacts stay portable.

    `experiments/run_plan.py:_resolve_config_path` already accepts relative paths,
    so writing them here doesn't break scoring or resume."""
    try:
        return str(path.relative_to(experiment.manifest_path.parent))
    except ValueError:
        return str(path)


def _record_round_config_in_run_plan(*, experiment: ExperimentRecord, round_label: str, config_path: Path) -> None:
    path = experiment.manifest_path.parent / "run_plan.csv"
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = [{key: value or "" for key, value in row.items()} for row in reader]

    for field in RUN_PLAN_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    config_stem = config_path.stem
    for row in rows:
        if row.get("round", "").strip() == round_label and Path(row.get("config_path", "")).stem == config_stem:
            return

    rows.append(
        {
            **{field: "" for field in fieldnames},
            "plan_index": str(_next_run_plan_index(rows)),
            "round": round_label,
            "config_path": _relative_to_experiment(experiment, config_path),
            "score_stage_default": SCORING_STAGE,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _next_run_plan_index(rows: list[dict[str, str]]) -> int:
    indexes: list[int] = []
    for row in rows:
        try:
            indexes.append(int(row.get("plan_index", "")))
        except ValueError:
            continue
    return max(indexes, default=len(rows)) + 1


def _first_config_path(experiment: ExperimentRecord) -> Path:
    config_dir = experiment.manifest_path.parent / "configs"
    configs = sorted(config_dir.glob("*.json"))
    if not configs:
        raise AgenticResearchValidationError(f"agentic_research_config_missing:{experiment.experiment_id}")
    return configs[0]


def _existing_config_hashes(config_dir: Path) -> set[str]:
    hashes: set[str] = set()
    for path in sorted(config_dir.glob("*.json")):
        try:
            payload = load_training_config_json(path)
        except Exception:
            continue
        _normalize_lgbm_effective_params(payload)
        hashes.add(compute_config_hash(payload))
    return hashes


_LGBM_ONLY_PARAM_KEYS = ("num_leaves", "min_child_samples", "bagging_freq", "reg_alpha", "device_type")
_XGBOOST_ONLY_PARAM_KEYS = ("max_leaves", "min_child_weight")


def _apply_family_switch_cleanup(payload: dict[str, object]) -> None:
    """Strip params that conflict with the active `model.type`.

    Family switches need to drop LGBM-only or XGBoost-only knobs from the parent
    config. The LLM cannot express "delete this key" via a value change, so the
    controller does it: when `model.type` is XGBoost, drop `model.device` and the
    LGBM-only params; when LGBM, drop the XGBoost-only params and any orphaned
    `model.module_path`. Also strips entries the LLM nulled out for the same
    reason. Operates in place.
    """
    model = payload.get("model")
    if not isinstance(model, dict):
        return
    model_type = model.get("type")
    params = model.get("params")
    if not isinstance(params, dict):
        params = None
    if model_type == "XGBoostRegressor":
        model.pop("device", None)
        if params is not None:
            for key in _LGBM_ONLY_PARAM_KEYS:
                params.pop(key, None)
    elif model_type == "LGBMRegressor":
        # LGBM uses the built-in module so any custom module_path the LLM kept
        # over from an XGBoost parent would point at the wrong file.
        if model.get("module_path") in (None, "xgboost_model.py"):
            model.pop("module_path", None)
        if params is not None:
            for key in _XGBOOST_ONLY_PARAM_KEYS:
                params.pop(key, None)
    if params is not None:
        for key in list(params.keys()):
            if params[key] is None:
                params.pop(key)


def _normalize_lgbm_effective_params(payload: dict[str, object]) -> dict[str, object]:
    """Clamp `num_leaves` to `2 ** max_depth` when both are set for LGBM models.

    LightGBM stops splits at `max_depth` regardless of `num_leaves`, so two configs
    that differ only by `num_leaves` above the depth-implied cap produce identical
    trees. Normalizing before hashing makes the duplicate-by-hash gate catch this.
    Skipped when `max_depth <= 0` (LightGBM's "unlimited depth" sentinel) because
    then `num_leaves` is the sole limiter.
    """
    model = payload.get("model")
    if not isinstance(model, dict) or model.get("type") != "LGBMRegressor":
        return payload
    params = model.get("params")
    if not isinstance(params, dict):
        return payload
    depth = params.get("max_depth")
    leaves = params.get("num_leaves")
    if not isinstance(depth, int) or isinstance(depth, bool) or depth <= 0:
        return payload
    if not isinstance(leaves, int) or isinstance(leaves, bool):
        return payload
    cap = 2**depth
    if leaves > cap:
        params["num_leaves"] = cap
    return payload


def _assign_dotted(payload: dict[str, object], parts: list[str], value: object) -> None:
    cursor: dict[str, object] = payload
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            raise AgenticResearchValidationError(f"agentic_research_change_target_not_mapping:{'.'.join(parts)}")
        cursor = cast(dict[str, object], child)
    cursor[parts[-1]] = value


def _change_path_allowed(path: str) -> bool:
    return _matches_any_path(path, ALLOWED_CHANGE_PATHS)


def _matches_any_path(path: str, allowed: tuple[str, ...]) -> bool:
    for entry in allowed:
        if entry.endswith(".*") and path.startswith(entry[:-1]):
            return True
        if path == entry:
            return True
    return False


def _program_allowed_paths(experiment: ExperimentRecord) -> tuple[str, ...]:
    raw = experiment.metadata.get(ALLOWED_PATHS_METADATA_KEY)
    if not isinstance(raw, list):
        return ALLOWED_CHANGE_PATHS
    narrowed: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        candidate = item.strip()
        if candidate and _matches_any_path(candidate, ALLOWED_CHANGE_PATHS):
            narrowed.append(candidate)
    return tuple(narrowed) if narrowed else ALLOWED_CHANGE_PATHS


def _program_value_caps(experiment: ExperimentRecord) -> dict[str, tuple[float, float]]:
    raw = experiment.metadata.get(VALUE_CAPS_METADATA_KEY)
    return _parse_value_caps(raw)


def _parse_value_caps(raw: object) -> dict[str, tuple[float, float]]:
    if not isinstance(raw, dict):
        return {}
    caps: dict[str, tuple[float, float]] = {}
    for key, bounds in raw.items():
        if not isinstance(key, str) or not key.strip():
            continue
        if not isinstance(bounds, list) or len(bounds) != 2:
            continue
        lo, hi = bounds
        if isinstance(lo, bool) or isinstance(hi, bool):
            continue
        if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
            continue
        caps[key.strip()] = (float(lo), float(hi))
    return caps


def _phases_config(experiment: ExperimentRecord) -> dict[str, object] | None:
    raw = experiment.metadata.get(PHASES_METADATA_KEY)
    if not isinstance(raw, dict) or not raw:
        return None
    return cast(dict[str, object], raw)


def _phase_spec(phases_config: dict[str, object], phase: str) -> dict[str, object] | None:
    spec = phases_config.get(phase)
    return cast(dict[str, object], spec) if isinstance(spec, dict) else None


def _initial_phase(phases_config: dict[str, object]) -> str:
    initial = phases_config.get("initial_phase")
    if not isinstance(initial, str) or not initial.strip():
        raise AgenticResearchValidationError("agentic_research_initial_phase_missing")
    if not isinstance(phases_config.get(initial), dict):
        raise AgenticResearchValidationError(f"agentic_research_initial_phase_invalid:{initial}")
    return initial


def _current_phase_caps(*, experiment: ExperimentRecord, state: dict[str, object]) -> dict[str, tuple[float, float]]:
    phases_config = _phases_config(experiment)
    if phases_config is None:
        return _program_value_caps(experiment)
    phase = _optional_str(state.get("phase"))
    if phase is None:
        return _program_value_caps(experiment)
    spec = _phase_spec(phases_config, phase)
    if spec is None:
        return _program_value_caps(experiment)
    return _parse_value_caps(spec.get("value_caps"))


def _phase_transition_spec(*, experiment: ExperimentRecord, state: dict[str, object]) -> dict[str, object] | None:
    phases_config = _phases_config(experiment)
    if phases_config is None:
        return None
    phase = _optional_str(state.get("phase"))
    if phase is None:
        return None
    spec = _phase_spec(phases_config, phase)
    if spec is None:
        return None
    transition = spec.get("transition")
    return cast(dict[str, object], transition) if isinstance(transition, dict) else None


def _update_phase_progress(*, state: dict[str, object], metric_value: float | None) -> None:
    if "phase" not in state:
        return
    if metric_value is None:
        return
    state["phase_successful_rounds"] = _as_int(state.get("phase_successful_rounds"), default=0) + 1
    current_best = state.get("phase_best_metric")
    improved = current_best is None or (
        isinstance(current_best, (int, float))
        and not isinstance(current_best, bool)
        and metric_value > float(current_best) + PHASE_IMPROVEMENT_THRESHOLD
    )
    if improved:
        state["phase_best_metric"] = float(metric_value)
        state["phase_plateau_counter"] = 0
    else:
        state["phase_plateau_counter"] = _as_int(state.get("phase_plateau_counter"), default=0) + 1


def _phase_has_confirmed_champion(state: dict[str, object], phase: str) -> bool:
    confirmations = state.get("confirmations")
    if not isinstance(confirmations, dict):
        return False
    for value in confirmations.values():
        if not isinstance(value, dict):
            continue
        if value.get("promoted_in_phase") == phase:
            return True
    return False


def _maybe_transition_phase(
    *,
    experiment: ExperimentRecord,
    state: dict[str, object],
    round_number: int,
    report: ExperimentReport | None = None,
) -> dict[str, object] | None:
    """Evaluate the phase-transition predicate; mutate state only if a real transition fires.

    Returns a trace payload when a transition (or terminal stop) fires, otherwise None.
    A non-terminal phase missing a valid `next_phase` emits a `phase_misconfigured` trace
    event and leaves state unchanged so the predicate doesn't re-fire forever.
    """
    transition = _phase_transition_spec(experiment=experiment, state=state)
    if transition is None:
        return None
    current_phase = _optional_str(state.get("phase"))
    if current_phase is None:
        return None

    min_rounds = _as_int(transition.get("min_rounds_in_phase"), default=0)
    plateau_threshold = _as_int(transition.get("plateau_threshold"), default=0)
    require_champion = bool(transition.get("require_confirmed_champion", False))
    is_terminal = bool(transition.get("is_terminal", False))

    successful_rounds = _as_int(state.get("phase_successful_rounds"), default=0)
    plateau = _as_int(state.get("phase_plateau_counter"), default=0)
    champion_ok = (not require_champion) or _phase_has_confirmed_champion(state, current_phase)
    if successful_rounds < min_rounds or plateau < plateau_threshold:
        return None
    if not champion_ok:
        # Transition gates are otherwise met but the phase exhausted without a
        # confirmed champion. Surface that distinctly so operators can grep
        # trace.jsonl for stuck phases instead of inferring it from silence.
        return {
            "transition": "blocked_no_champion",
            "from": current_phase,
            "successful_rounds": successful_rounds,
            "plateau": plateau,
        }

    next_phase: str | None = None
    if not is_terminal:
        phases_config = _phases_config(experiment)
        spec = _phase_spec(phases_config, current_phase) if phases_config else None
        raw_next = spec.get("next_phase") if spec else None
        if not isinstance(raw_next, str) or raw_next not in (phases_config or {}):
            return {
                "transition": "misconfigured",
                "from": current_phase,
                "reason": "next_phase_missing_or_invalid",
            }
        next_phase = raw_next

    history_record: dict[str, object] = {
        "phase": current_phase,
        "started_round": _as_int(state.get("phase_round_start"), default=1),
        "ended_round": round_number,
        "exit_reason": "all_phases_done" if is_terminal else "phase_transition",
        "best_metric": state.get("phase_best_metric"),
        "successful_rounds": successful_rounds,
        "best_run_id": _best_run_id_for_phase(state=state, report=report),
    }
    history = state.get("phase_history")
    if not isinstance(history, list):
        history = []
    history.append(history_record)
    state["phase_history"] = history

    if is_terminal:
        state["status"] = "stopped"
        state["stop_reason"] = f"all_phases_done:{current_phase}_plateau"
        state["last_checkpoint"] = "all_phases_done"
        return {"transition": "terminal", "from": current_phase, "to": None}

    state["phase"] = next_phase
    state["phase_round_start"] = round_number + 1
    state["phase_best_metric"] = None
    state["phase_plateau_counter"] = 0
    state["phase_successful_rounds"] = 0
    return {"transition": "phase_change", "from": current_phase, "to": next_phase}


def _best_run_id_for_phase(*, state: dict[str, object], report: ExperimentReport | None) -> str | None:
    """Return the run_id whose primary metric matches phase_best_metric, if any."""
    best_metric = state.get("phase_best_metric")
    if not isinstance(best_metric, (int, float)) or isinstance(best_metric, bool):
        return None
    if report is None:
        return None
    target = float(best_metric)
    for row in report.rows:
        value = getattr(row, PRIMARY_METRIC_FIELD, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) == target:
            return row.run_id
    return None


def _accumulate_phase_cost(state: dict[str, object], seconds: float) -> None:
    if "phase" not in state:
        return
    phase = _optional_str(state.get("phase")) or "unknown"
    totals = state.get("phase_cost_totals")
    if not isinstance(totals, dict):
        totals = {}
    bucket_raw = totals.get(phase)
    bucket: dict[str, float] = bucket_raw if isinstance(bucket_raw, dict) else {}
    bucket["rounds"] = float(bucket.get("rounds", 0)) + 1
    bucket["total_seconds"] = float(bucket.get("total_seconds", 0.0)) + max(0.0, seconds)
    totals[phase] = bucket
    state["phase_cost_totals"] = totals


def _extract_lgbm_signature(
    *,
    config_path: Path,
    round_label: str,
    run_id: str | None,
    primary_metric: float | None,
    action: str,
) -> dict[str, object]:
    """Compact per-round summary the LLM consults instead of carrying a ledger in the memo."""
    sig: dict[str, object] = {
        "r": round_label,
        "run_id": run_id,
        "action": action,
        "primary": primary_metric,
    }
    try:
        payload = load_training_config_json(config_path)
    except Exception:
        return sig
    data_section = payload.get("data") if isinstance(payload, dict) else None
    model_section = payload.get("model") if isinstance(payload, dict) else None
    params_section = model_section.get("params") if isinstance(model_section, dict) else None
    if isinstance(data_section, dict):
        sig["target"] = data_section.get("target_col")
    if isinstance(params_section, dict):
        sig["lr"] = params_section.get("learning_rate")
        sig["n"] = params_section.get("n_estimators")
        sig["depth"] = params_section.get("max_depth")
        sig["leaves"] = params_section.get("num_leaves")
        sig["cs"] = params_section.get("colsample_bytree")
        sig["mcs"] = params_section.get("min_child_samples")
        sig["seed"] = params_section.get("random_state")
    return sig


def _config_random_state(config_path: Path) -> object | None:
    """Return `model.params.random_state` for a materialized config, or None on any parse error."""
    try:
        payload = load_training_config_json(config_path)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    params = model.get("params") if isinstance(model, dict) else None
    return params.get("random_state") if isinstance(params, dict) else None


def _append_tried_signature(state: dict[str, object], signature: dict[str, object]) -> None:
    signatures = state.get("tried_signatures")
    if not isinstance(signatures, list):
        signatures = []
    signatures.append(signature)
    state["tried_signatures"] = signatures[-TRIED_SIGNATURES_WINDOW:]


def _is_confirmation_round(decision_payload: dict[str, object]) -> bool:
    """A confirmation round mutates only `model.params.random_state` and reuses a
    previously-LLM-generated parent config (`config_NNN.json` where NNN > 001).
    The baseline (`config_001.json`) is a copy of the seed, not an LLM hypothesis,
    so confirming it would satisfy the promotion gate without testing anything."""
    if decision_payload.get("action") != "run":
        return False
    changes = decision_payload.get("changes")
    if not isinstance(changes, list) or len(changes) != 1:
        return False
    change = changes[0]
    if not isinstance(change, dict):
        return False
    if change.get("path") != "model.params.random_state":
        return False
    parent = decision_payload.get("parent_config")
    if not isinstance(parent, str):
        return False
    if not (parent.startswith("config_") and parent.endswith(".json")):
        return False
    return parent != "config_001.json"


def _record_confirmation_attempt(
    *,
    state: dict[str, object],
    parent_config: str,
    seed: int,
    run_id: str,
    metric_value: float | None,
    round_number: int,
) -> dict[str, object]:
    """Append a single-seed result to state.confirmations[parent_config]."""
    confirmations_raw = state.get("confirmations")
    confirmations: dict[str, dict[str, object]] = confirmations_raw if isinstance(confirmations_raw, dict) else {}
    entry_raw = confirmations.get(parent_config)
    entry: dict[str, object] = entry_raw if isinstance(entry_raw, dict) else {}
    seeds_completed = entry.get("seeds_completed")
    if not isinstance(seeds_completed, list):
        seeds_completed = []
    if seed not in seeds_completed:
        seeds_completed.append(seed)
    runs_raw = entry.get("runs")
    runs: dict[str, str] = runs_raw if isinstance(runs_raw, dict) else {}
    runs[str(seed)] = run_id
    metrics_raw = entry.get("primary_metric_by_seed")
    metrics: dict[str, float] = metrics_raw if isinstance(metrics_raw, dict) else {}
    if metric_value is not None:
        metrics[str(seed)] = float(metric_value)
    entry["seeds_completed"] = seeds_completed
    entry["runs"] = runs
    entry["primary_metric_by_seed"] = metrics
    entry.setdefault("first_attempt_at_round", round_number)
    entry["last_attempt_at_round"] = round_number
    confirmations[parent_config] = entry
    state["confirmations"] = confirmations
    return entry


def _maybe_promote_confirmation(
    *, state: dict[str, object], parent_config: str, round_number: int
) -> dict[str, object] | None:
    """If the canonical seed trio has completed and beats the current champion, promote.

    Returns a trace payload describing the promotion, or None if no promotion happened.
    """
    confirmations_raw = state.get("confirmations")
    if not isinstance(confirmations_raw, dict):
        return None
    entry_raw = confirmations_raw.get(parent_config)
    if not isinstance(entry_raw, dict):
        return None
    seeds = entry_raw.get("seeds_completed")
    metrics_raw = entry_raw.get("primary_metric_by_seed")
    if not isinstance(seeds, list) or not isinstance(metrics_raw, dict):
        return None
    canonical = set(CANONICAL_SEED_TRIO)
    if not canonical.issubset(set(seeds)):
        return None
    metric_values: list[float] = []
    for seed in CANONICAL_SEED_TRIO:
        raw = metrics_raw.get(str(seed))
        if not isinstance(raw, (int, float)):
            return None
        metric_values.append(float(raw))
    mean = sum(metric_values) / len(metric_values)
    entry_raw["seed_trio_primary_mean"] = mean
    prior = state.get("confirmed_champion")
    prior_mean: float | None = None
    if isinstance(prior, dict):
        raw = prior.get("seed_trio_primary_mean")
        if isinstance(raw, (int, float)):
            prior_mean = float(raw)
    if prior_mean is not None and mean <= prior_mean + CONFIRMATION_PROMOTION_THRESHOLD:
        return None
    entry_raw["promoted_at_round"] = round_number
    entry_raw["promoted_in_phase"] = state.get("phase")
    state["confirmed_champion"] = {
        "parent_config": parent_config,
        "seed_trio_primary_mean": mean,
        "promoted_at_round": round_number,
        "promoted_in_phase": state.get("phase"),
        "runs": dict(entry_raw.get("runs") or {}),
    }
    return {
        "parent_config": parent_config,
        "seed_trio_primary_mean": mean,
        "prior_seed_trio_primary_mean": prior_mean,
        "phase": state.get("phase"),
    }


def _recent_confirmations(state: dict[str, object], *, limit: int = 30) -> dict[str, object]:
    """Return the most-recent N confirmation entries by last_attempt_at_round."""
    confirmations_raw = state.get("confirmations")
    if not isinstance(confirmations_raw, dict):
        return {}
    items: list[tuple[int, str, dict[str, object]]] = []
    for parent, entry in confirmations_raw.items():
        if not isinstance(entry, dict):
            continue
        last = entry.get("last_attempt_at_round")
        last_int = int(last) if isinstance(last, (int, float)) else 0
        items.append((last_int, str(parent), entry))
    items.sort(key=lambda triple: triple[0], reverse=True)
    return {parent: entry for _, parent, entry in items[:limit]}


def _cost_summary(state: dict[str, object]) -> dict[str, object]:
    totals_raw = state.get("phase_cost_totals")
    totals: dict[str, dict[str, float]] = {}
    grand_rounds = 0.0
    grand_seconds = 0.0
    if isinstance(totals_raw, dict):
        for phase, bucket in totals_raw.items():
            if not isinstance(bucket, dict):
                continue
            rounds = float(bucket.get("rounds", 0))
            seconds = float(bucket.get("total_seconds", 0.0))
            totals[str(phase)] = {
                "rounds": rounds,
                "total_seconds": seconds,
                "avg_seconds": (seconds / rounds) if rounds else 0.0,
            }
            grand_rounds += rounds
            grand_seconds += seconds
    return {
        "total_rounds": grand_rounds,
        "total_seconds": grand_seconds,
        "by_phase": totals,
    }


def _config_context(experiment: ExperimentRecord) -> list[dict[str, object]]:
    config_dir = experiment.manifest_path.parent / "configs"
    paths = sorted(config_dir.glob("*.json"))
    parsed: list[tuple[Path, dict[str, object] | None, str | None]] = []
    for path in paths:
        try:
            payload = load_training_config_json(path)
            parsed.append((path, payload, compute_config_hash(payload)))
        except Exception as exc:
            parsed.append((path, None, str(exc)))

    by_hash: dict[str, list[Path]] = {}
    for path, payload, hash_or_error in parsed:
        if payload is not None and hash_or_error is not None:
            by_hash.setdefault(hash_or_error, []).append(path)

    skip: set[Path] = set()
    for group in by_hash.values():
        if len(group) < 2:
            continue
        # If a generated `config_*.json` shares a hash with the seed, hide the seed.
        generated = [p for p in group if p.name.startswith("config_")]
        seeds = [p for p in group if not p.name.startswith("config_")]
        if generated and seeds:
            skip.update(seeds)

    items: list[dict[str, object]] = []
    for path, payload, hash_or_error in parsed:
        if path in skip:
            continue
        if payload is None:
            items.append({"filename": path.name, "error": hash_or_error})
            continue
        items.append({"filename": path.name, "config": _mutable_config_view(payload)})
    return items


def _mutable_config_view(payload: dict[str, object]) -> dict[str, object]:
    view: dict[str, object] = {}
    for path in ALLOWED_CHANGE_PATHS:
        if path.endswith(".*"):
            prefix = path[:-2]
            parts = _split_path(prefix)
            value = _get_dotted(payload, parts)
            if value is not None:
                _assign_view(view, parts, value)
            continue
        parts = _split_path(path)
        value = _get_dotted(payload, parts)
        if value is not None:
            _assign_view(view, parts, value)
    return view


def _split_path(path: str) -> list[str]:
    return path.split(".")


def _get_dotted(payload: dict[str, object], parts: list[str]) -> object | None:
    cursor: object = payload
    for part in parts:
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        parsed = cast(dict[str, object], cursor)
        cursor = parsed[part]
    return cursor


def _assign_view(payload: dict[str, object], parts: list[str], value: object) -> None:
    cursor = payload
    for part in parts[:-1]:
        child = cursor.get(part)
        if child is None:
            child = {}
            cursor[part] = child
        if not isinstance(child, dict):
            return
        cursor = cast(dict[str, object], child)
    cursor[parts[-1]] = value


def _report_context(report: ExperimentReport | None) -> dict[str, object]:
    if report is None:
        return {"rows": []}
    return {
        "metric": report.metric,
        "total_runs": report.total_runs,
        "champion_run_id": report.champion_run_id,
        "rows": [_row_payload(row) for row in report.rows],
    }


def _safe_report(*, root: Path, experiment_id: str) -> ExperimentReport | None:
    try:
        return report_experiment(store_root=root, experiment_id=experiment_id, metric=PRIMARY_METRIC, limit=25)
    except ExperimentError:
        return None


def _champion_seed42_metric(state: dict[str, object]) -> float | None:
    """Return the seed-42 primary metric of the current champion, or None.

    Used by the Outcome footer to decide whether a discovery round cleared the
    "above champion seed-42" threshold without trying to read the LLM's own
    EXPERIMENT.md trigger value.
    """
    champion = state.get("confirmed_champion")
    if not isinstance(champion, dict):
        return None
    parent = champion.get("parent_config")
    if not isinstance(parent, str):
        return None
    confirmations = state.get("confirmations")
    if not isinstance(confirmations, dict):
        return None
    entry = confirmations.get(parent)
    if not isinstance(entry, dict):
        return None
    metrics_by_seed = entry.get("primary_metric_by_seed")
    if not isinstance(metrics_by_seed, dict):
        return None
    raw = metrics_by_seed.get("42")
    return float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None


def _phase_snapshot_for_md(*, experiment: ExperimentRecord, state: dict[str, object]) -> dict[str, object]:
    """Capture the post-round, pre-transition phase counters for the Outcome footer."""
    transition = _phase_transition_spec(experiment=experiment, state=state)
    plateau_threshold = _as_int(transition.get("plateau_threshold"), default=0) if isinstance(transition, dict) else 0
    min_rounds = _as_int(transition.get("min_rounds_in_phase"), default=0) if isinstance(transition, dict) else 0
    return {
        "phase": state.get("phase"),
        "plateau_counter": _as_int(state.get("phase_plateau_counter"), default=0),
        "plateau_threshold": plateau_threshold,
        "successful_rounds": _as_int(state.get("phase_successful_rounds"), default=0),
        "min_rounds_in_phase": min_rounds,
    }


def _lookup_dotted(payload: object, dotted: str) -> object:
    """Walk `dotted` (e.g. `model.params.learning_rate`) through `payload`. Returns
    `_MISSING_SENTINEL` if any segment is missing or non-mapping. Caller renders this
    as `?` so a missing-old-value doesn't masquerade as None."""
    current: object = payload
    for token in dotted.split("."):
        if not isinstance(current, dict):
            return _MISSING_SENTINEL
        if token not in current:
            return _MISSING_SENTINEL
        current = current.get(token)
    return current


def _render_diff_vs_parent(*, experiment: ExperimentRecord, decision_payload: dict[str, object]) -> str:
    """Render a `## Diff vs parent` block from the structured decision changes.

    Returns the empty string if there is no parent or no diffable content. Loads
    the parent config from disk to surface the pre-change values; falls back to
    `?` for any segment that doesn't resolve."""
    parent_name = decision_payload.get("parent_config")
    if not isinstance(parent_name, str):
        return ""
    action = decision_payload.get("action")
    changes = decision_payload.get("changes")
    if action == "baseline":
        return f"## Diff vs parent\n\n- parent: {parent_name} (full copy; no mutation)\n"
    if not isinstance(changes, list) or not changes:
        return ""
    parent_path = experiment.manifest_path.parent / "configs" / parent_name
    parent_payload: object = None
    try:
        parent_payload = load_training_config_json(parent_path)
    except Exception:
        parent_payload = None
    lines = ["## Diff vs parent", "", f"- parent: {parent_name}"]
    for change in changes:
        if not isinstance(change, dict):
            continue
        path = change.get("path")
        new_value = change.get("value")
        if not isinstance(path, str):
            continue
        old_value = _lookup_dotted(parent_payload, path) if parent_payload is not None else _MISSING_SENTINEL
        old_str = "?" if old_value is _MISSING_SENTINEL else repr(old_value)
        lines.append(f"- {path}: {old_str} → {new_value!r}")
    return "\n".join(lines) + "\n"


def _load_secondary_metrics_from_disk(*, root: Path, run_id: str | None) -> dict[str, float]:
    """Read the secondary metric block from runs/<run_id>/metrics.json.

    Returns a flat dict keyed by canonical viz names (`bmc_mean`, `corr_mean`, ...).
    Silently returns {} if the file is missing or malformed; secondary metrics are
    cosmetic for the round md, not load-bearing for any decision."""
    if run_id is None:
        return {}
    metrics_path = root / "runs" / run_id / "metrics.json"
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    result: dict[str, float] = {}
    for dotted, label in SECONDARY_METRICS:
        current: object = payload
        for token in dotted.split("."):
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(token)
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            result[label] = float(current)
    return result


def _run_primary_metric_from_disk(*, root: Path, run_id: str) -> float | None:
    """Read the primary metric directly from runs/<run_id>/metrics.json.

    Fallback for rounds whose run scored below the truncated leaderboard
    returned by `_safe_report`. The on-disk metric is authoritative; the
    leaderboard is just a paginated view.
    """
    metrics_path = root / "runs" / run_id / "metrics.json"
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    current: object = payload
    for token in PRIMARY_METRIC.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(token)
        if current is None:
            return None
    return float(current) if isinstance(current, (int, float)) else None


def _has_scored_primary_row(report: ExperimentReport | None) -> bool:
    return any(getattr(row, PRIMARY_METRIC_FIELD) is not None for row in (report.rows if report else ()))


def _row_for_run(report: ExperimentReport | None, run_id: str) -> ExperimentReportRow | None:
    for row in report.rows if report else ():
        if row.run_id == run_id:
            return row
    return None


def _best_run_from_report(report: ExperimentReport | None) -> ResearchBestRun:
    if report is None or not report.rows:
        return ResearchBestRun()
    for row in report.rows:
        if getattr(row, PRIMARY_METRIC_FIELD) is not None:
            return ResearchBestRun(
                experiment_id=report.experiment_id,
                run_id=row.run_id,
                bmc_last_200_eras_mean=row.bmc_last_200_eras_mean,
                bmc_mean=row.bmc_mean,
                corr_mean=row.corr_mean,
                mmc_mean=row.mmc_mean,
                cwmm_mean=row.cwmm_mean,
                updated_at=row.created_at,
            )
    return ResearchBestRun()


def _row_payload(row: ExperimentReportRow) -> dict[str, object]:
    return {
        "run_id": row.run_id,
        "status": row.status,
        "created_at": row.created_at,
        "metric_value": row.metric_value,
        "corr_mean": row.corr_mean,
        "mmc_mean": row.mmc_mean,
        "cwmm_mean": row.cwmm_mean,
        "bmc_mean": row.bmc_mean,
        "bmc_last_200_eras_mean": row.bmc_last_200_eras_mean,
        "is_champion": row.is_champion,
    }


def _decision_payload(decision: ResearchDecision, *, model_source: str) -> dict[str, object]:
    return {
        "action": decision.action,
        "learning": decision.learning,
        "belief_update": decision.belief_update,
        "next_hypothesis": decision.next_hypothesis,
        "parent_config": decision.parent_config,
        "changes": [asdict(change) for change in decision.changes],
        "stop_reason": decision.stop_reason,
        "model_source": model_source,
    }


def _extract_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise AgenticResearchValidationError("agentic_research_json_missing")
    try:
        payload = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise AgenticResearchValidationError("agentic_research_json_invalid") from exc
    if not isinstance(payload, dict):
        raise AgenticResearchValidationError("agentic_research_json_object_required")
    return payload


def _initial_state(experiment: ExperimentRecord) -> dict[str, object]:
    now = _utc_now_iso()
    state: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": experiment.experiment_id,
        "status": "initialized",
        "next_round_number": 1,
        "total_rounds_completed": 0,
        "last_checkpoint": "initialized",
        "stop_reason": None,
        "best_overall": asdict(ResearchBestRun()),
        "created_at": now,
        "updated_at": now,
    }
    phases_config = _phases_config(experiment)
    if phases_config is not None:
        state["phase"] = _initial_phase(phases_config)
        state["phase_round_start"] = 1
        state["phase_best_metric"] = None
        state["phase_plateau_counter"] = 0
        state["phase_successful_rounds"] = 0
        state["phase_history"] = []
        state["phase_cost_totals"] = {}
    return state


def _status_result(
    *,
    experiment: ExperimentRecord,
    state: dict[str, object],
    best: ResearchBestRun,
) -> ResearchStatusResult:
    auto_dir = _agentic_dir(experiment)
    return ResearchStatusResult(
        experiment_id=experiment.experiment_id,
        status=_status_value(state.get("status")),
        next_round_number=_as_int(state.get("next_round_number"), default=1),
        total_rounds_completed=_as_int(state.get("total_rounds_completed"), default=0),
        last_checkpoint=str(state.get("last_checkpoint") or "initialized"),
        last_round_label=_optional_str(state.get("last_round_label")),
        last_run_id=_optional_str(state.get("last_run_id")),
        stop_reason=_optional_str(state.get("stop_reason")),
        best_overall=best,
        agentic_research_dir=auto_dir,
        state_path=auto_dir / STATE_FILENAME,
        trace_path=auto_dir / TRACE_FILENAME,
        decision_path=_decision_log_path(experiment),
        program_path=_program_path(experiment),
    )


def _run_result(
    *,
    experiment: ExperimentRecord,
    state: dict[str, object],
    rounds: list[ResearchRoundResult],
    interrupted: bool,
) -> ResearchRunResult:
    best = _best_from_state(state)
    return ResearchRunResult(
        experiment_id=experiment.experiment_id,
        status=_status_value(state.get("status")),
        next_round_number=_as_int(state.get("next_round_number"), default=1),
        total_rounds_completed=_as_int(state.get("total_rounds_completed"), default=0),
        last_checkpoint=str(state.get("last_checkpoint") or "initialized"),
        stop_reason=_optional_str(state.get("stop_reason")),
        best_overall=best,
        rounds=tuple(rounds),
        interrupted=interrupted,
    )


def _best_from_state(state: dict[str, object]) -> ResearchBestRun:
    payload = state.get("best_overall")
    if not isinstance(payload, dict):
        return ResearchBestRun()
    best = cast(dict[str, object], payload)
    return ResearchBestRun(
        experiment_id=_optional_str(best.get("experiment_id")),
        run_id=_optional_str(best.get("run_id")),
        bmc_last_200_eras_mean=_optional_float(best.get("bmc_last_200_eras_mean")),
        bmc_mean=_optional_float(best.get("bmc_mean")),
        corr_mean=_optional_float(best.get("corr_mean")),
        mmc_mean=_optional_float(best.get("mmc_mean")),
        cwmm_mean=_optional_float(best.get("cwmm_mean")),
        updated_at=_optional_str(best.get("updated_at")),
    )


def _status_value(value: object) -> ResearchStatus:
    if value in {"initialized", "running", "interrupted", "stopped", "failed"}:
        return cast(ResearchStatus, value)
    return "initialized"


def _is_terminal_stop(state: dict[str, object]) -> bool:
    """Has the loop reached any terminal stop (LLM stop, all phases done, failure bail)?"""
    return _status_value(state.get("status")) == "stopped"


def _agentic_dir(experiment: ExperimentRecord) -> Path:
    return experiment.manifest_path.parent / AGENTIC_DIRNAME


def _rounds_dir(experiment: ExperimentRecord) -> Path:
    return _agentic_dir(experiment) / "rounds"


def _program_path(experiment: ExperimentRecord) -> Path:
    raw = experiment.metadata.get(PROGRAM_METADATA_KEY)
    if raw is None:
        return PROGRAM_PATH
    if not isinstance(raw, str) or not raw.strip():
        raise AgenticResearchValidationError("agentic_research_program_invalid")
    name = raw.strip()
    if Path(name).name != name or not name.endswith(".md"):
        raise AgenticResearchValidationError(f"agentic_research_program_invalid:{name}")
    if name == PROGRAM_PATH.name:
        return PROGRAM_PATH
    path = CUSTOM_PROGRAM_DIR / name
    if not path.is_file():
        raise AgenticResearchValidationError(f"agentic_research_program_missing:{name}")
    return path


def _state_path(experiment: ExperimentRecord) -> Path:
    return _agentic_dir(experiment) / STATE_FILENAME


def _trace_path(experiment: ExperimentRecord) -> Path:
    return _agentic_dir(experiment) / TRACE_FILENAME


def _decision_log_path(experiment: ExperimentRecord) -> Path:
    return _rounds_dir(experiment) / "decision.json"


def _latest_round_markdown(experiment: ExperimentRecord) -> str | None:
    rounds_dir = _rounds_dir(experiment)
    if not rounds_dir.is_dir():
        return None
    candidates = [path for path in rounds_dir.glob("r*.md") if re.fullmatch(r"r\d{3,}\.md", path.name)]
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: int(p.stem[1:]))
    return _read_text(latest, limit=MAX_CONTEXT_CHARS)


def _load_state(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgenticResearchValidationError(f"agentic_research_state_invalid:{path}") from exc
    if not isinstance(payload, dict):
        raise AgenticResearchValidationError(f"agentic_research_state_invalid:{path}")
    return payload


def _save_state(experiment: ExperimentRecord, state: dict[str, object]) -> None:
    _write_json(_state_path(experiment), state)


def _recent_decisions(path: Path, *, limit: int) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows[-limit:]


def _append_decision_log(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _append_trace(
    experiment: ExperimentRecord,
    *,
    round_number: int,
    round_label: str,
    event: str,
    payload: dict[str, object],
) -> None:
    row: dict[str, object] = {
        "created_at": _utc_now_iso(),
        "round_number": round_number,
        "round_label": round_label,
        "event": event,
        "payload": payload,
    }
    path = _trace_path(experiment)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _write_failure_round_markdown(
    *,
    artifact_dir: Path,
    round_label: str,
    round_payload: dict[str, object],
) -> None:
    error = _optional_str(round_payload.get("error")) or "unknown error"
    error_class = _optional_str(round_payload.get("error_class")) or "Exception"
    pending = round_payload.get("pending_decision")
    pending_decision = pending if isinstance(pending, dict) else None
    reached_llm = pending_decision is not None

    lines = [
        f"# {round_label} Research State",
        "",
        "## What was attempted",
    ]
    if reached_llm:
        lines.append(
            f"The LLM produced a decision but the round failed during materialization, "
            f"train, or score. Cause: `{error_class}`."
        )
    else:
        lines.append(f"The round failed before the LLM produced a usable decision. Cause: `{error_class}`.")

    if pending_decision is not None:
        lines.extend(["", "## LLM proposal"])
        parent = _optional_str(pending_decision.get("parent_config"))
        action = _optional_str(pending_decision.get("action"))
        if parent is not None:
            lines.append(f"- parent: {parent}")
        if action is not None:
            lines.append(f"- action: {action}")
        changes = pending_decision.get("changes")
        if isinstance(changes, list) and changes:
            lines.append("- changes:")
            for change in changes:
                if not isinstance(change, dict):
                    continue
                path = change.get("path")
                value = change.get("value")
                lines.append(f"  - {path}: → {value}")

    debug_files = [
        artifact_dir / f"{round_label}.debug.error.txt",
        artifact_dir / f"{round_label}.debug.llm_response.txt",
        artifact_dir / f"{round_label}.debug.prompt.md",
        artifact_dir / f"{round_label}.debug.codex_stdout.jsonl",
        artifact_dir / f"{round_label}.debug.codex_stderr.txt",
    ]
    present = [p.name for p in debug_files if p.is_file()]
    if present:
        lines.extend(["", "## Debug artifacts"])
        for name in present:
            lines.append(f"- {name}")

    lines.extend(
        [
            "",
            "## Execution Result",
            f"- Action: {round_payload.get('action')}",
            f"- Status: {round_payload.get('status')}",
            f"- Run ID: {_notes_value(round_payload.get('run_id'))}",
            f"- Config: {_notes_value(round_payload.get('config_path'))}",
            f"- {PRIMARY_METRIC_FIELD}: {_notes_value(round_payload.get('metric_value'))}",
            f"- Completed at: {_notes_value(round_payload.get('completed_at'))}",
            f"- Error: {error}",
        ]
    )
    _write_text(artifact_dir / f"{round_label}.md", "\n".join(lines).rstrip() + "\n")


def _write_llm_round_markdown(
    *,
    artifact_dir: Path,
    round_label: str,
    round_markdown: str,
    round_payload: dict[str, object],
) -> None:
    wall = round_payload.get("wall_time_seconds")
    wall_str = f"{wall:.1f}" if isinstance(wall, (int, float)) and not isinstance(wall, bool) else "none"
    diff_md = round_payload.get("diff_vs_parent_markdown")
    diff_block = diff_md.rstrip() if isinstance(diff_md, str) and diff_md.strip() else None
    lines = [
        round_markdown.rstrip(),
        "",
    ]
    if diff_block is not None:
        lines.extend([diff_block, ""])
    lines.extend(
        [
            "## Execution Result",
            f"- Action: {round_payload.get('action')}",
            f"- Status: {round_payload.get('status')}",
            f"- Run ID: {_notes_value(round_payload.get('run_id'))}",
            f"- Config: {_notes_value(round_payload.get('config_path'))}",
            f"- {PRIMARY_METRIC_FIELD}: {_notes_value(round_payload.get('metric_value'))}",
            f"- Completed at: {_notes_value(round_payload.get('completed_at'))}",
            f"- Wall time: {wall_str}s",
        ]
    )
    stop_reason = _optional_str(round_payload.get("stop_reason"))
    if stop_reason is not None:
        lines.append(f"- Stop reason: {stop_reason}")
    secondary = round_payload.get("secondary_metrics")
    if isinstance(secondary, dict) and secondary:
        lines.extend(["", "## Secondary Metrics"])
        for _, label in SECONDARY_METRICS:
            if label in secondary:
                lines.append(f"- {label}: {secondary[label]}")
    outcome_block = _render_outcome_footer(round_payload)
    if outcome_block:
        lines.extend(["", outcome_block.rstrip()])
    _write_text(artifact_dir / f"{round_label}.md", "\n".join(lines).rstrip() + "\n")


def _render_outcome_footer(round_payload: dict[str, object]) -> str:
    """Render Python-authoritative round classification: trigger / confirmation / promoted / phase."""
    lines = ["## Outcome"]

    metric = round_payload.get("metric_value")
    champion_seed42 = round_payload.get("champion_seed42_before")
    if (
        isinstance(metric, (int, float))
        and not isinstance(metric, bool)
        and isinstance(champion_seed42, (int, float))
        and not isinstance(champion_seed42, bool)
    ):
        cleared = float(metric) > float(champion_seed42)
        verb = "above" if cleared else "below"
        lines.append(
            f"- Trigger cleared: {'yes' if cleared else 'no'} "
            f"(metric {metric} {verb} champion seed-42 {champion_seed42})"
        )
    else:
        lines.append("- Trigger cleared: n/a (no champion yet)")

    confirmation = bool(round_payload.get("confirmation_round"))
    lines.append(f"- Confirmation round: {'yes' if confirmation else 'no'}")

    promotion = round_payload.get("promotion")
    if isinstance(promotion, dict) and promotion:
        mean = promotion.get("seed_trio_primary_mean")
        if isinstance(mean, (int, float)) and not isinstance(mean, bool):
            lines.append(f"- Promoted: yes (trio mean {mean})")
        else:
            lines.append("- Promoted: yes")
    else:
        lines.append("- Promoted: no")

    snap = round_payload.get("phase_snapshot")
    if isinstance(snap, dict):
        phase = snap.get("phase")
        plateau = snap.get("plateau_counter")
        plateau_threshold = snap.get("plateau_threshold")
        successful = snap.get("successful_rounds")
        min_rounds = snap.get("min_rounds_in_phase")
        if phase is not None:
            lines.append(
                f"- Phase: {phase} plateau {plateau}/{plateau_threshold}, successful {successful}/{min_rounds}"
            )
    return "\n".join(lines) + "\n"


def _notes_value(value: object) -> str:
    if value is None:
        return "none"
    return str(value)


def _write_failure_debug(
    *,
    artifact_dir: Path,
    round_label: str,
    prompt: str,
    error: str,
    raw_response: str | None = None,
    codex_stdout: str | None = None,
    codex_stderr: str | None = None,
) -> None:
    prefix = artifact_dir / f"{round_label}.debug"
    _write_text(Path(f"{prefix}.prompt.md"), prompt)
    _write_text(Path(f"{prefix}.error.txt"), error.strip() + "\n")
    if raw_response is not None:
        _write_text(Path(f"{prefix}.llm_response.txt"), raw_response)
    if codex_stdout is not None:
        _write_text(Path(f"{prefix}.codex_stdout.jsonl"), codex_stdout)
    if codex_stderr is not None:
        _write_text(Path(f"{prefix}.codex_stderr.txt"), codex_stderr)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _experiment_markdown_path(experiment: ExperimentRecord) -> Path:
    return experiment.manifest_path.parent / "EXPERIMENT.md"


def _write_experiment_markdown(experiment: ExperimentRecord, content: str | None) -> int:
    """Atomically overwrite the experiment-level EXPERIMENT.md curated by the agent.

    Returns the number of bytes written. If `content` is missing or empty, the
    prior file is left untouched and 0 is returned. Uses temp+os.replace so a
    crash mid-write cannot leave a partial file the next round would read.
    """
    if not content:
        return 0
    path = _experiment_markdown_path(experiment)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)
    return len(content)


def _artifact_rotation_mode(experiment: ExperimentRecord) -> str:
    """Return one of `disabled`, `dry_run`, `enabled`. Default is `disabled`."""
    raw = experiment.metadata.get(ARTIFACT_ROTATION_METADATA_KEY)
    if isinstance(raw, str) and raw in ("disabled", "dry_run", "enabled"):
        return raw
    return "disabled"


def _essential_run_ids(
    *,
    report: ExperimentReport | None,
    state: dict[str, object],
    last_round_number: int,
) -> set[str]:
    """Compute the set of run_ids whose heavy artifacts must be preserved."""
    essential: set[str] = set()
    if report is not None:
        for row in report.rows:
            run_id = getattr(row, "run_id", None)
            if isinstance(run_id, str):
                essential.add(run_id)
    confirmations = state.get("confirmations")
    if isinstance(confirmations, dict):
        for entry in confirmations.values():
            if not isinstance(entry, dict):
                continue
            runs = entry.get("runs")
            if isinstance(runs, dict):
                for value in runs.values():
                    if isinstance(value, str):
                        essential.add(value)
    history = state.get("phase_history")
    if isinstance(history, list):
        for record in history:
            if not isinstance(record, dict):
                continue
            best_run = record.get("best_run_id")
            if isinstance(best_run, str):
                essential.add(best_run)
    champion = state.get("confirmed_champion")
    if isinstance(champion, dict):
        runs = champion.get("runs")
        if isinstance(runs, dict):
            for value in runs.values():
                if isinstance(value, str):
                    essential.add(value)
    grace_start = max(1, last_round_number - ARTIFACT_ROTATION_RECENT_ROUND_GRACE + 1)
    signatures = state.get("tried_signatures")
    if isinstance(signatures, list):
        for entry in signatures:
            if not isinstance(entry, dict):
                continue
            round_label = entry.get("r")
            if not isinstance(round_label, str) or not round_label.startswith("r"):
                continue
            try:
                round_num = int(round_label[1:])
            except ValueError:
                continue
            if round_num < grace_start:
                continue
            run_id = entry.get("run_id")
            if isinstance(run_id, str):
                essential.add(run_id)
    return essential


def _heavy_artifact_targets(run_dir: Path) -> list[Path]:
    """Return concrete files in a run dir that are eligible for rotation."""
    if not run_dir.is_dir():
        return []
    targets: list[Path] = []
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name in ARTIFACT_ROTATION_PRESERVE_NAMES:
            continue
        if path.name in ARTIFACT_ROTATION_HEAVY_NAMES:
            targets.append(path)
            continue
        if "predictions" in path.parts and path.suffix == ".parquet":
            targets.append(path)
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size >= ARTIFACT_ROTATION_HEAVY_SIZE_THRESHOLD:
            targets.append(path)
    return targets


def _rotate_run_artifacts(
    *,
    root: Path,
    experiment: ExperimentRecord,
    state: dict[str, object],
    last_round_number: int,
) -> dict[str, object] | None:
    """Prune heavy artifacts from non-essential runs OWNED BY THIS EXPERIMENT.

    Runs are owned if listed in `experiment.runs` or referenced anywhere in state
    (state references cannot escape the current experiment by construction).
    Sibling experiments' run dirs in the shared `runs/` root are never touched.

    Returns a trace payload when rotation produced action (or would have, in dry run),
    or None when the rotation mode is disabled.
    """
    mode = _artifact_rotation_mode(experiment)
    if mode == "disabled":
        return None
    report = _safe_report(root=root, experiment_id=experiment.experiment_id)
    essential = _essential_run_ids(report=report, state=state, last_round_number=last_round_number)
    owned = set(experiment.runs) | essential | _state_run_ids(state)
    runs_root = root / "runs"
    if not runs_root.is_dir():
        return None
    rotated: list[str] = []
    bytes_freed = 0
    dry_run_targets: list[str] = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name not in owned or run_dir.name in essential:
            continue
        targets = _heavy_artifact_targets(run_dir)
        if not targets:
            continue
        for target in targets:
            try:
                bytes_freed += target.stat().st_size
            except OSError:
                pass
            if mode == "enabled":
                try:
                    target.unlink()
                except OSError:
                    continue
            elif mode == "dry_run" and len(dry_run_targets) < 100:
                dry_run_targets.append(str(target))
        rotated.append(run_dir.name)
    if not rotated:
        return None
    payload: dict[str, object] = {
        "mode": mode,
        "rotated_run_ids": rotated,
        "bytes_freed": bytes_freed,
        "kept_count": len(essential),
    }
    if mode == "dry_run":
        payload["targets"] = dry_run_targets
    return payload


def _state_run_ids(state: dict[str, object]) -> set[str]:
    """Collect every run_id referenced anywhere in state (confirmations, history,
    tried_signatures, confirmed_champion, best_overall)."""
    run_ids: set[str] = set()
    confirmations = state.get("confirmations")
    if isinstance(confirmations, dict):
        for entry in confirmations.values():
            if isinstance(entry, dict):
                runs = entry.get("runs")
                if isinstance(runs, dict):
                    run_ids.update(v for v in runs.values() if isinstance(v, str))
    history = state.get("phase_history")
    if isinstance(history, list):
        for record in history:
            if isinstance(record, dict):
                best = record.get("best_run_id")
                if isinstance(best, str):
                    run_ids.add(best)
    champion = state.get("confirmed_champion")
    if isinstance(champion, dict):
        runs = champion.get("runs")
        if isinstance(runs, dict):
            run_ids.update(v for v in runs.values() if isinstance(v, str))
    best_overall = state.get("best_overall")
    if isinstance(best_overall, dict):
        run_id = best_overall.get("run_id")
        if isinstance(run_id, str):
            run_ids.add(run_id)
    signatures = state.get("tried_signatures")
    if isinstance(signatures, list):
        for entry in signatures:
            if isinstance(entry, dict):
                run_id = entry.get("run_id")
                if isinstance(run_id, str):
                    run_ids.add(run_id)
    return run_ids


def _read_text(path: Path, *, limit: int) -> str | None:
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _unique_config_path(config_dir: Path, filename: str) -> Path:
    path = config_dir / filename
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    index = 2
    while True:
        candidate = config_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _required_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AgenticResearchValidationError(f"agentic_research_field_missing:{key}")
    return value.strip()


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _coerce_metric(value: object) -> float | None:
    """Coerce a primary-metric candidate to float, rejecting booleans (subclass of int)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_int(value: object, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    return default


def _as_list(value: object) -> list[object]:
    return cast(list[object], value) if isinstance(value, list) else []


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "AgenticResearchError",
    "AgenticResearchValidationError",
    "ResearchBestRun",
    "ResearchRoundResult",
    "ResearchRunResult",
    "ResearchStatusResult",
    "get_research_status",
    "program_markdown",
    "run_research",
]
