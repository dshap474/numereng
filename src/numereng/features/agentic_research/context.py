"""Bounded context assembly."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from numereng.config.training import load_training_config_json
from numereng.features.agentic_research import aggregate, boundary, memory
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
    configs = aggregate.load_config_cache(experiment.manifest_path.parent / "configs")
    journal_entries = memory.journal_all(experiment)
    recipe_groups = aggregate.aggregate_recipes(journal_entries, configs=configs)
    assembled = {
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
        "believed_best": state.get("believed_best"),
        "recipe_leaderboard": _recipe_leaderboard(recipe_groups, configs),
        "rounds_since_new_believed_best": _rounds_since_new_believed_best(journal_entries, state),
        "coverage": _coverage(configs),
        "caps_binding": _caps_binding(experiment, configs, state),
        "observed_seed_noise": aggregate.observed_seed_noise(recipe_groups),
        "report": _report_context(report),
        "recent_journal": memory.journal_tail(experiment, limit=ar_types.RECENT_JOURNAL_LIMIT),
        "configs": _config_context(experiment, state=state),
        "last_round_memo": memory.latest_round_markdown(experiment),
        "experiment_notes": memory.read_text(
            memory.experiment_markdown_path(experiment), limit=ar_types.MAX_CONTEXT_CHARS
        ),
        "last_error": ar_types.optional_str(state.get("last_error")),
    }
    return _apply_size_guard(assembled)


def _recipe_leaderboard(
    groups: list[aggregate.RecipeGroup], configs: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    """Top recipes by seed-trio mean — the harness-owned replacement for the model's seed ledger."""
    leaderboard: list[dict[str, object]] = []
    for group in groups[: ar_types.RECIPE_LEADERBOARD_LIMIT]:
        config = configs.get(group.representative_config)
        leaderboard.append(
            {
                "representative_config": group.representative_config,
                "params": _mutable_config_view(config) if config is not None else {},
                "seeds": list(group.seeds),
                "seed_count": group.count,
                "trio_mean": group.trio_mean,
                "trio_fnc_mean": group.trio_fnc_mean,
                "bmc_std": group.bmc_std,
                "per_seed": [
                    {"seed": row.get("seed"), "bmc": row.get("bmc"), "fnc": row.get("fnc")} for row in group.per_seed
                ],
                "run_ids": list(group.run_ids),
            }
        )
    return leaderboard


def _rounds_since_new_believed_best(entries: list[dict[str, object]], state: dict[str, object]) -> int:
    """Completed rounds since the believed-best last changed (resume-safe; derived, not stored)."""
    changed = state.get("believed_best_changed_round")
    if not isinstance(changed, int) or isinstance(changed, bool):
        return 0
    return sum(
        1
        for entry in entries
        if entry.get("status") == "completed" and ar_types.as_int(entry.get("round"), default=0) > changed
    )


def _coverage(configs: dict[str, dict[str, object]]) -> dict[str, object]:
    """Per allowed-path map of distinct values tried, cardinality-capped to stay bounded."""
    buckets: dict[str, list[object]] = {}
    for config in configs.values():
        for path, value in _flatten_dotted(_mutable_config_view(config)):
            buckets.setdefault(path, []).append(value)
    coverage: dict[str, object] = {}
    for path, values in sorted(buckets.items()):
        distinct = _distinct(values)
        if len(distinct) <= ar_types.COVERAGE_VALUE_LIMIT:
            coverage[path] = _sorted_values(distinct)
        elif all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in distinct):
            coverage[path] = {
                "min": min(distinct),
                "max": max(distinct),
                "count": len(distinct),
                "recent_samples": distinct[-ar_types.COVERAGE_VALUE_LIMIT :],
            }
        else:
            coverage[path] = {"count": len(distinct), "recent_samples": distinct[-ar_types.COVERAGE_VALUE_LIMIT :]}
    return coverage


def _caps_binding(
    experiment: ExperimentRecord, configs: dict[str, dict[str, object]], state: dict[str, object]
) -> list[dict[str, object]]:
    """Allowed paths where the believed-best config sits at a value_caps edge (optimum may lie beyond)."""
    believed = state.get("believed_best")
    config_name = believed.get("config") if isinstance(believed, dict) else None
    if not isinstance(config_name, str) or config_name not in configs:
        return []
    flat = dict(_flatten_dotted(_mutable_config_view(configs[config_name])))
    binding: list[dict[str, object]] = []
    for path, (lo, hi) in boundary.program_value_caps(experiment).items():
        value = flat.get(path)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if value <= lo:
            binding.append({"path": path, "value": value, "edge": "min", "cap": [lo, hi]})
        elif value >= hi:
            binding.append({"path": path, "value": value, "edge": "max", "cap": [lo, hi]})
    return binding


def _apply_size_guard(context: dict[str, object]) -> dict[str, object]:
    """Defensive backstop: trim the bulkiest discretionary term if the whole context blows past budget.

    Every term is bounded by construction; this only fires on a pathological config blowup, and records
    what it dropped so the trim is never silent (the documented run-killer was an 860 KB prompt).
    """
    if _json_len(context) <= ar_types.MAX_TOTAL_CONTEXT_CHARS:
        return context
    configs = context.get("configs")
    if isinstance(configs, list) and configs:
        kept = list(configs)
        while kept and _json_len(context) > ar_types.MAX_TOTAL_CONTEXT_CHARS:
            kept = kept[1:]
            context["configs"] = kept
        context["_trimmed"] = {"configs_kept": len(kept), "reason": "context_exceeded_max_total_chars"}
    return context


def _json_len(payload: object) -> int:
    return len(json.dumps(payload, default=str))


def _flatten_dotted(payload: dict[str, object], prefix: str = "") -> list[tuple[str, object]]:
    items: list[tuple[str, object]] = []
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.extend(_flatten_dotted(value, path))
        else:
            items.append((path, value))
    return items


def _distinct(values: list[object]) -> list[object]:
    seen: set[str] = set()
    distinct: list[object] = []
    for value in values:
        marker = json.dumps(value, sort_keys=True, default=str)
        if marker not in seen:
            seen.add(marker)
            distinct.append(value)
    return distinct


def _sorted_values(values: list[object]) -> list[object]:
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        return sorted(values)
    return sorted(values, key=lambda value: json.dumps(value, sort_keys=True, default=str))


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
    return _run_metric_from_disk(root=root, run_id=run_id, dotted=ar_types.PRIMARY_METRIC)


def run_fnc_mean_from_disk(*, root: Path, run_id: str) -> float | None:
    """Read fnc.mean from a run's metrics.json (the report row caps below top-N drop fnc)."""
    return _run_metric_from_disk(root=root, run_id=run_id, dotted="fnc.mean")


def _run_metric_from_disk(*, root: Path, run_id: str, dotted: str) -> float | None:
    try:
        current: object = json.loads((root / "runs" / run_id / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for token in dotted.split("."):
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
