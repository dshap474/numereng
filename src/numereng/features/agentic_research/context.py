"""Bounded context assembly."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from numereng.config.training import load_training_config_json
from numereng.features.agentic_research import boundary, memory
from numereng.features.agentic_research import types as ar_types
from numereng.features.agentic_research.boundary import ALLOWED_CHANGE_PATHS
from numereng.features.experiments import (
    ExperimentError,
    ExperimentRecord,
    ExperimentReport,
    ExperimentReportRow,
    report_experiment,
)


def _safe_report(*, root: Path, experiment_id: str) -> ExperimentReport | None:
    try:
        return report_experiment(
            store_root=root, experiment_id=experiment_id, metric=ar_types.PRIMARY_METRIC, limit=ar_types.REPORT_LIMIT
        )
    except ExperimentError:
        return None


def build_context(
    *, root: Path, experiment: ExperimentRecord, report: ExperimentReport | None, state: dict[str, object]
) -> dict[str, object]:
    raw_budget = experiment.metadata.get(ar_types.BUDGET_ROUNDS_METADATA_KEY)
    budget_rounds = ar_types.as_int(raw_budget, default=0) if raw_budget is not None else None
    return {
        "objective": {
            "primary_metric": ar_types.PRIMARY_METRIC_FIELD,
            "tie_break": "bmc_mean",
            "sanity_checks": ["corr_mean", "mmc_mean", "cwmm_mean", "fnc_mean"],
            "scoring_stage": ar_types.SCORING_STAGE,
            "payout_target": ar_types.PAYOUT_TARGET_COL,
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
            "next_round_number": ar_types.as_int(state.get("next_round_number"), default=1),
            "total_rounds_completed": ar_types.as_int(state.get("total_rounds_completed"), default=0),
            "failed_rounds_counter": ar_types.as_int(state.get("failed_rounds_counter"), default=0),
            "budget_rounds": budget_rounds if budget_rounds and budget_rounds > 0 else None,
        },
        "allowed_change_paths": list(ALLOWED_CHANGE_PATHS),
        "value_caps": {path: list(bounds) for path, bounds in boundary.program_value_caps(experiment).items()},
        "champion": state.get("champion"),
        "report": _report_context(report),
        "recent_journal": memory.journal_tail(experiment, limit=ar_types.RECENT_JOURNAL_LIMIT),
        "configs": _config_context(experiment, state=state),
        "last_round_memo": memory.latest_round_markdown(experiment),
        "experiment_notes": memory.read_text(
            memory.experiment_markdown_path(experiment), limit=ar_types.MAX_CONTEXT_CHARS
        ),
        "last_error": ar_types.optional_str(state.get("last_error")),
    }


def _report_context(report: ExperimentReport | None) -> dict[str, object]:
    if report is None:
        return {"rows": []}
    return {
        "metric": report.metric,
        "total_runs": report.total_runs,
        "champion_run_id": report.champion_run_id,
        "rows": [asdict(row) for row in report.rows],
    }


def has_scored_primary_row(report: ExperimentReport | None) -> bool:
    return any(getattr(row, ar_types.PRIMARY_METRIC_FIELD) is not None for row in (report.rows if report else ()))


def row_for_run(report: ExperimentReport | None, run_id: str) -> ExperimentReportRow | None:
    for row in report.rows if report else ():
        if row.run_id == run_id:
            return row
    return None


def best_run_from_report(report: ExperimentReport | None) -> ar_types.ResearchBestRun:
    if report is not None:
        for row in report.rows:
            if getattr(row, ar_types.PRIMARY_METRIC_FIELD) is not None:
                return ar_types.ResearchBestRun(
                    experiment_id=report.experiment_id,
                    run_id=row.run_id,
                    bmc_last_200_eras_mean=row.bmc_last_200_eras_mean,
                    bmc_mean=row.bmc_mean,
                    corr_mean=row.corr_mean,
                    mmc_mean=row.mmc_mean,
                    cwmm_mean=row.cwmm_mean,
                    updated_at=row.created_at,
                )
    return ar_types.ResearchBestRun()


def run_primary_metric_from_disk(*, root: Path, run_id: str) -> float | None:
    try:
        current: object = json.loads((root / "runs" / run_id / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for token in ar_types.PRIMARY_METRIC.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(token)
        if current is None:
            return None
    return float(current) if isinstance(current, (int, float)) else None


def _config_context(experiment: ExperimentRecord, *, state: dict[str, object]) -> list[dict[str, object]]:
    config_dir = experiment.manifest_path.parent / "configs"
    keep = _relevant_config_paths(sorted(config_dir.glob("*.json")), state=state)
    items: list[dict[str, object]] = []
    for path in sorted(keep, key=lambda path: path.name):
        try:
            items.append({"filename": path.name, "config": _mutable_config_view(load_training_config_json(path))})
        except Exception as exc:
            items.append({"filename": path.name, "error": str(exc)})
    return items


def _relevant_config_paths(paths: list[Path], *, state: dict[str, object]) -> set[Path]:
    keep: set[Path] = set()
    generated: list[Path] = []
    for path in paths:
        if path.name.startswith("config_"):
            generated.append(path)
        else:
            keep.add(path)
    champion = state.get("champion")
    champion_config = champion.get("config") if isinstance(champion, dict) else None
    if isinstance(champion_config, str):
        keep.update(path for path in generated if path.name == champion_config)
    keep.update(sorted(generated, key=_config_suffix_num)[-ar_types.CONFIG_CONTEXT_RECENT :])
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
        cursor = cursor[part]
    return cursor


def _assign_view(payload: dict[str, object], parts: list[str], value: object) -> None:
    cursor = payload
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            return
        cursor = child
    cursor[parts[-1]] = value
