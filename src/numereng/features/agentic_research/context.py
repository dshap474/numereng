"""Bounded context assembly: no term grows with round count (the 860 KB-prompt lesson)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast

from numereng.config.training import load_training_config_json
from numereng.features.agentic_research import memory
from numereng.features.agentic_research.boundary import ALLOWED_CHANGE_PATHS
from numereng.features.agentic_research.types import (
    CONFIG_CONTEXT_RECENT,
    MAX_CONTEXT_CHARS,
    PAYOUT_TARGET_COL,
    PRIMARY_METRIC,
    PRIMARY_METRIC_FIELD,
    RECENT_JOURNAL_LIMIT,
    REPORT_LIMIT,
    SCORING_STAGE,
    ResearchBestRun,
    as_int,
    optional_str,
)
from numereng.features.experiments import (
    ExperimentError,
    ExperimentRecord,
    ExperimentReport,
    ExperimentReportRow,
    report_experiment,
)


def _safe_report(*, root: Path, experiment_id: str) -> ExperimentReport | None:
    """Return a ranked report (capped at REPORT_LIMIT), or None if it cannot be built."""
    try:
        return report_experiment(
            store_root=root, experiment_id=experiment_id, metric=PRIMARY_METRIC, limit=REPORT_LIMIT
        )
    except ExperimentError:
        return None


def build_context(
    *,
    root: Path,
    experiment: ExperimentRecord,
    report: ExperimentReport | None,
    state: dict[str, object],
) -> dict[str, object]:
    """Assemble the bounded context dict shown to the model each round."""
    return {
        "objective": {
            "primary_metric": PRIMARY_METRIC_FIELD,
            "tie_break": "bmc_mean",
            "sanity_checks": ["corr_mean", "mmc_mean", "cwmm_mean"],
            "scoring_stage": SCORING_STAGE,
            "payout_target": PAYOUT_TARGET_COL,
        },
        "experiment": {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "status": experiment.status,
            "hypothesis": experiment.hypothesis,
            "tags": list(experiment.tags),
            "champion_run_id": experiment.champion_run_id,
            "run_count": len(experiment.runs),
        },
        "budget": {
            "next_round_number": as_int(state.get("next_round_number"), default=1),
            "total_rounds_completed": as_int(state.get("total_rounds_completed"), default=0),
            "failed_rounds_counter": as_int(state.get("failed_rounds_counter"), default=0),
        },
        "allowed_change_paths": list(ALLOWED_CHANGE_PATHS),
        "value_caps": {path: list(bounds) for path, bounds in _value_caps(experiment).items()},
        "champion": state.get("champion"),
        "report": _report_context(report),
        "recent_journal": memory.journal_tail(experiment, limit=RECENT_JOURNAL_LIMIT),
        "configs": _config_context(experiment, state=state),
        "last_round_memo": memory.latest_round_markdown(experiment),
        "experiment_notes": memory.read_text(memory.experiment_markdown_path(experiment), limit=MAX_CONTEXT_CHARS),
        "research_memory": memory.read_text(
            root / "notes" / "__RESEARCH_MEMORY__" / "CURRENT.md", limit=MAX_CONTEXT_CHARS
        ),
        "last_error": optional_str(state.get("last_error")),
    }


def _value_caps(experiment: ExperimentRecord) -> dict[str, tuple[float, float]]:
    from numereng.features.agentic_research.boundary import program_value_caps

    return program_value_caps(experiment)


def _report_context(report: ExperimentReport | None) -> dict[str, object]:
    if report is None:
        return {"rows": []}
    return {
        "metric": report.metric,
        "total_runs": report.total_runs,
        "champion_run_id": report.champion_run_id,
        "rows": [_row_payload(row) for row in report.rows],
    }


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


def has_scored_primary_row(report: ExperimentReport | None) -> bool:
    """True if any report row carries a primary metric (used to decide baseline vs LLM)."""
    return any(getattr(row, PRIMARY_METRIC_FIELD) is not None for row in (report.rows if report else ()))


def row_for_run(report: ExperimentReport | None, run_id: str) -> ExperimentReportRow | None:
    """Return the report row for one run, or None."""
    for row in report.rows if report else ():
        if row.run_id == run_id:
            return row
    return None


def best_run_from_report(report: ExperimentReport | None) -> ResearchBestRun:
    """Return the best scored run from a ranked (best-first) report."""
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


def run_primary_metric_from_disk(*, root: Path, run_id: str) -> float | None:
    """Read the primary metric directly from runs/<run_id>/metrics.json (authoritative)."""
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


def _config_context(experiment: ExperimentRecord, *, state: dict[str, object]) -> list[dict[str, object]]:
    config_dir = experiment.manifest_path.parent / "configs"
    all_paths = sorted(config_dir.glob("*.json"))
    keep = _relevant_config_paths(all_paths, state=state)
    items: list[dict[str, object]] = []
    for path in all_paths:
        if path not in keep:
            continue
        try:
            payload = load_training_config_json(path)
        except Exception as exc:
            items.append({"filename": path.name, "error": str(exc)})
            continue
        items.append({"filename": path.name, "config": _mutable_config_view(payload)})
    return items


def _relevant_config_paths(paths: list[Path], *, state: dict[str, object]) -> set[Path]:
    """Bound the config menu to all seeds + the champion config + the most recent N generated."""
    keep: set[Path] = set()
    generated: list[Path] = []
    for path in paths:
        (generated.append(path) if path.name.startswith("config_") else keep.add(path))
    champion = state.get("champion")
    if isinstance(champion, dict):
        champion_config = cast(dict[str, object], champion).get("config")
        if isinstance(champion_config, str):
            keep.update(p for p in generated if p.name == champion_config)
    generated.sort(key=_config_suffix_num)
    keep.update(generated[-CONFIG_CONTEXT_RECENT:])
    return keep


def _config_suffix_num(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def _mutable_config_view(payload: dict[str, object]) -> dict[str, object]:
    view: dict[str, object] = {}
    for path in ALLOWED_CHANGE_PATHS:
        parts = (path[:-2] if path.endswith(".*") else path).split(".")
        value = _get_dotted(payload, parts)
        if value is not None:
            _assign_view(view, parts, value)
    return view


def _get_dotted(payload: dict[str, object], parts: list[str]) -> object | None:
    cursor: object = payload
    for part in parts:
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cast(dict[str, object], cursor)[part]
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
