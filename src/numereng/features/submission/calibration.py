"""Materialize local-vs-live calibration artifacts for submitted models."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from numereng.features.store.layout import WorkspaceLayout, resolve_workspace_layout

CALIBRATION_RELATIVE_DIR = Path("analysis") / "live_calibration"
CALIBRATION_ROWS_FILENAME = "calibration_rows.parquet"
CALIBRATION_REPORT_FILENAME = "report.json"
CALIBRATION_MANIFEST_FILENAME = "manifest.json"

LOCAL_METRICS: tuple[str, ...] = (
    "local_bmc200_mean",
    "local_bmc_mean",
    "local_corr_mean",
    "local_mmc_mean",
    "local_fnc_mean",
)
LIVE_METRICS: tuple[str, ...] = (
    "live_mmc20",
    "live_corr20",
    "live_mmc60",
    "live_corr60",
    "live_bmc",
    "live_season_score",
)
CALIBRATION_COLUMNS: tuple[str, ...] = (
    "model_name",
    "model_id",
    "experiment_id",
    "package_id",
    "recipe",
    "feature_scope",
    "target",
    "target_horizon",
    "family",
    "round_number",
    "state",
    "is_estimate",
    "open_date",
    "close_date",
    "resolve_date",
    "has_live_score",
    "local_bmc200_mean",
    "local_bmc200_sharpe",
    "local_bmc200_max_drawdown",
    "local_bmc_mean",
    "local_corr_mean",
    "local_mmc_mean",
    "local_fnc_mean",
    "local_metric_source",
    "live_bmc",
    "live_bmc_percentile",
    "live_mmc20",
    "live_mmc20_percentile",
    "live_corr20",
    "live_corr20_percentile",
    "live_mmc60",
    "live_mmc60_percentile",
    "live_corr60",
    "live_corr60_percentile",
    "live_season_score",
    "live_season_score_percentile",
    "live_started_at",
    "pulled_at",
)


@dataclass(frozen=True)
class CalibrationMaterializeResult:
    """Result from materializing calibration artifacts."""

    workspace_root: Path
    artifact_root: Path
    rows_path: Path
    report_path: Path
    manifest_path: Path
    row_count: int
    model_count: int
    scored_row_count: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CalibrationReportResult:
    """Loaded calibration report plus row artifact location."""

    artifact_root: Path
    rows_path: Path
    report_path: Path
    manifest_path: Path
    row_count: int
    report: dict[str, Any]
    manifest: dict[str, Any]


def calibration_artifact_root(*, workspace_root: str | Path = ".") -> Path:
    """Return the canonical local calibration artifact directory."""

    layout = resolve_workspace_layout(workspace_root)
    return layout.store_root / CALIBRATION_RELATIVE_DIR


def materialize_live_calibration(*, workspace_root: str | Path = ".") -> CalibrationMaterializeResult:
    """Build local-vs-live model-round rows and report artifacts."""

    layout = resolve_workspace_layout(workspace_root)
    artifact_root = layout.store_root / CALIBRATION_RELATIVE_DIR
    rows_path = artifact_root / CALIBRATION_ROWS_FILENAME
    report_path = artifact_root / CALIBRATION_REPORT_FILENAME
    manifest_path = artifact_root / CALIBRATION_MANIFEST_FILENAME
    generated_at = datetime.now(UTC).isoformat()

    rows, source_files, warnings = _build_calibration_rows(layout)
    frame = pd.DataFrame.from_records(rows, columns=list(CALIBRATION_COLUMNS))
    report = build_live_calibration_report(rows, generated_at=generated_at)
    manifest = {
        "generated_at": generated_at,
        "workspace_root": str(layout.workspace_root),
        "artifact_root": str(artifact_root),
        "rows_path": str(rows_path),
        "report_path": str(report_path),
        "source_files": sorted(source_files),
        "row_count": len(rows),
        "model_count": len({row["model_name"] for row in rows}),
        "scored_row_count": sum(1 for row in rows if row.get("has_live_score")),
        "warnings": warnings,
    }

    artifact_root.mkdir(parents=True, exist_ok=True)
    _write_parquet_atomic(frame, rows_path)
    _write_json_atomic(report, report_path)
    _write_json_atomic(manifest, manifest_path)

    return CalibrationMaterializeResult(
        workspace_root=layout.workspace_root,
        artifact_root=artifact_root,
        rows_path=rows_path,
        report_path=report_path,
        manifest_path=manifest_path,
        row_count=len(rows),
        model_count=len({row["model_name"] for row in rows}),
        scored_row_count=sum(1 for row in rows if row.get("has_live_score")),
        warnings=tuple(warnings),
    )


def load_live_calibration_report(
    *,
    workspace_root: str | Path = ".",
    resolved_only: bool = False,
) -> CalibrationReportResult:
    """Load the latest materialized calibration report."""

    artifact_root = calibration_artifact_root(workspace_root=workspace_root)
    rows_path = artifact_root / CALIBRATION_ROWS_FILENAME
    report_path = artifact_root / CALIBRATION_REPORT_FILENAME
    manifest_path = artifact_root / CALIBRATION_MANIFEST_FILENAME
    rows = load_live_calibration_rows(workspace_root=workspace_root)
    manifest = _read_json_dict(manifest_path)
    report = _read_json_dict(report_path)
    if not report:
        report = build_live_calibration_report(rows, generated_at=datetime.now(UTC).isoformat())
    elif resolved_only:
        resolved = report.get("scopes", {}).get("resolved_only") if isinstance(report.get("scopes"), dict) else None
        report = {
            "generated_at": report.get("generated_at"),
            "scope": "resolved_only",
            **(resolved if isinstance(resolved, dict) else {}),
        }
    return CalibrationReportResult(
        artifact_root=artifact_root,
        rows_path=rows_path,
        report_path=report_path,
        manifest_path=manifest_path,
        row_count=len(rows),
        report=report,
        manifest=manifest,
    )


def load_live_calibration_rows(*, workspace_root: str | Path = ".") -> list[dict[str, Any]]:
    """Load the latest materialized model-round calibration rows."""

    rows_path = calibration_artifact_root(workspace_root=workspace_root) / CALIBRATION_ROWS_FILENAME
    if not rows_path.is_file():
        return []
    try:
        frame = pd.read_parquet(rows_path)
    except Exception:
        return []
    return [_sanitize_row(row) for row in frame.to_dict(orient="records")]


def build_live_calibration_report(
    rows: list[dict[str, Any]],
    *,
    generated_at: str,
) -> dict[str, Any]:
    """Build calibration correlations, ranks, and group summaries from rows."""

    return {
        "generated_at": generated_at,
        "row_count": len(rows),
        "model_count": len({row.get("model_name") for row in rows if row.get("model_name")}),
        "scored_row_count": sum(1 for row in rows if row.get("has_live_score")),
        "scopes": {
            "all_scored": _scope_report(rows, resolved_only=False),
            "resolved_only": _scope_report(rows, resolved_only=True),
        },
    }


def _build_calibration_rows(layout: WorkspaceLayout) -> tuple[list[dict[str, Any]], set[str], list[str]]:
    rows: list[dict[str, Any]] = []
    source_files: set[str] = set()
    warnings: list[str] = []
    submissions_root = layout.submissions_root
    if not submissions_root.is_dir():
        warnings.append(f"submissions_root_missing:{submissions_root}")
        return rows, source_files, warnings

    for submission_dir in sorted(item for item in submissions_root.iterdir() if item.is_dir()):
        model_name = submission_dir.name
        metadata_path = submission_dir / "submission.json"
        rounds_path = submission_dir / "live_rounds.parquet"
        if not metadata_path.is_file() and not rounds_path.is_file():
            continue
        metadata = _read_json_dict(metadata_path)
        if metadata_path.is_file():
            source_files.add(str(metadata_path))
        if not rounds_path.is_file():
            warnings.append(f"live_rounds_missing:{model_name}")
            continue
        source_files.add(str(rounds_path))

        round_rows = _read_round_rows(rounds_path)
        if not round_rows:
            warnings.append(f"live_rounds_empty:{model_name}")
            continue
        local_metrics, local_sources = _submission_offline_metrics(layout, metadata)
        source_files.update(local_sources)
        tags = _submission_model_tags(model_name=model_name, metadata=metadata)
        source = metadata.get("source") if isinstance(metadata.get("source"), dict) else {}
        live_started_at = _submission_live_started_at(metadata)

        for round_row in round_rows:
            rows.append(
                _calibration_row(
                    model_name=model_name,
                    metadata=metadata,
                    source=source,
                    tags=tags,
                    local_metrics=local_metrics,
                    live_started_at=live_started_at,
                    round_row=round_row,
                )
            )
    return rows, source_files, warnings


def _calibration_row(
    *,
    model_name: str,
    metadata: dict[str, Any],
    source: dict[str, Any],
    tags: dict[str, Any],
    local_metrics: dict[str, Any],
    live_started_at: str | None,
    round_row: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "model_name": model_name,
        "model_id": _to_non_empty_str(metadata.get("model_id")) or _to_non_empty_str(metadata.get("numerai_model_id")),
        "experiment_id": _to_non_empty_str(source.get("experiment_id")),
        "package_id": _to_non_empty_str(source.get("package_id")),
        "recipe": _to_non_empty_str(source.get("recipe")) or tags.get("recipe"),
        "feature_scope": tags.get("feature_scope"),
        "target": tags.get("target"),
        "target_horizon": tags.get("target_horizon"),
        "family": tags.get("family"),
        "round_number": _to_int(round_row.get("round_number") or round_row.get("round")),
        "state": _to_non_empty_str(round_row.get("state")) or _to_non_empty_str(round_row.get("status")),
        "is_estimate": bool(round_row.get("is_estimate")),
        "open_date": _to_non_empty_str(round_row.get("open_date")),
        "close_date": _to_non_empty_str(round_row.get("close_date")),
        "resolve_date": _to_non_empty_str(round_row.get("resolve_date")),
        "local_bmc200_mean": local_metrics.get("local_bmc200_mean"),
        "local_bmc200_sharpe": local_metrics.get("local_bmc200_sharpe"),
        "local_bmc200_max_drawdown": local_metrics.get("local_bmc200_max_drawdown"),
        "local_bmc_mean": local_metrics.get("local_bmc_mean"),
        "local_corr_mean": local_metrics.get("local_corr_mean"),
        "local_mmc_mean": local_metrics.get("local_mmc_mean"),
        "local_fnc_mean": local_metrics.get("local_fnc_mean"),
        "local_metric_source": local_metrics.get("local_metric_source"),
        "live_bmc": _to_float(round_row.get("bmc")),
        "live_bmc_percentile": _to_float(round_row.get("bmc_percentile")),
        "live_mmc20": _to_float(round_row.get("mmc20")),
        "live_mmc20_percentile": _to_float(round_row.get("mmc20_percentile") or round_row.get("mmc_percentile")),
        "live_corr20": _to_float(round_row.get("corr20")),
        "live_corr20_percentile": _to_float(round_row.get("corr20_percentile") or round_row.get("corr_percentile")),
        "live_mmc60": _to_float(round_row.get("mmc60")),
        "live_mmc60_percentile": _to_float(round_row.get("mmc60_percentile")),
        "live_corr60": _to_float(round_row.get("corr60")),
        "live_corr60_percentile": _to_float(round_row.get("corr60_percentile")),
        "live_season_score": _to_float(round_row.get("season_score")),
        "live_season_score_percentile": _to_float(round_row.get("season_score_percentile")),
        "live_started_at": live_started_at,
        "pulled_at": _to_non_empty_str(round_row.get("pulled_at")),
    }
    row["has_live_score"] = any(row.get(metric) is not None for metric in LIVE_METRICS)
    return row


def _scope_report(rows: list[dict[str, Any]], *, resolved_only: bool) -> dict[str, Any]:
    scoped = [row for row in rows if row.get("has_live_score")]
    if resolved_only:
        scoped = [row for row in scoped if str(row.get("state") or "").lower() == "resolved"]

    return {
        "row_count": len(scoped),
        "model_count": len({row.get("model_name") for row in scoped if row.get("model_name")}),
        "correlations": _correlation_grid(scoped),
        "rank_deltas": _rank_deltas(scoped),
        "groups": {
            "target": _group_summaries(scoped, "target"),
            "feature_scope": _group_summaries(scoped, "feature_scope"),
            "recipe": _group_summaries(scoped, "recipe"),
        },
    }


def _correlation_grid(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grid: dict[str, dict[str, dict[str, Any]]] = {}
    for local_metric in LOCAL_METRICS:
        live_grid: dict[str, dict[str, Any]] = {}
        for live_metric in LIVE_METRICS:
            pairs = _metric_pairs(rows, local_metric, live_metric)
            live_grid[live_metric] = _linear_stats(pairs)
        grid[local_metric] = live_grid
    return grid


def _rank_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        model_name = _to_non_empty_str(row.get("model_name"))
        if model_name:
            by_model[model_name].append(row)

    items: list[dict[str, Any]] = []
    for model_name, model_rows in sorted(by_model.items()):
        first = model_rows[0]
        local_bmc200 = _first_numeric(model_rows, "local_bmc200_mean")
        live_mmc20 = _mean_numeric(model_rows, "live_mmc20")
        live_corr20 = _mean_numeric(model_rows, "live_corr20")
        items.append(
            {
                "model_name": model_name,
                "target": first.get("target"),
                "feature_scope": first.get("feature_scope"),
                "recipe": first.get("recipe"),
                "row_count": len(model_rows),
                "local_bmc200_mean": local_bmc200,
                "live_mmc20_mean": live_mmc20,
                "live_corr20_mean": live_corr20,
                "local_rank": None,
                "live_rank": None,
                "rank_delta": None,
            }
        )

    _assign_rank(items, value_key="local_bmc200_mean", rank_key="local_rank")
    _assign_rank(items, value_key="live_mmc20_mean", rank_key="live_rank")
    for item in items:
        local_rank = item.get("local_rank")
        live_rank = item.get("live_rank")
        if isinstance(local_rank, int) and isinstance(live_rank, int):
            item["rank_delta"] = live_rank - local_rank
    return items


def _group_summaries(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = _to_non_empty_str(row.get(group_key))
        if value:
            grouped[value].append(row)

    summaries: list[dict[str, Any]] = []
    for value, group_rows in sorted(grouped.items()):
        pairs = _metric_pairs(group_rows, "local_bmc200_mean", "live_mmc20")
        if len(pairs) < 3:
            continue
        summaries.append(
            {
                group_key: value,
                "row_count": len(group_rows),
                "model_count": len({row.get("model_name") for row in group_rows if row.get("model_name")}),
                "local_bmc200_mean": _mean_numeric(group_rows, "local_bmc200_mean"),
                "live_mmc20_mean": _mean_numeric(group_rows, "live_mmc20"),
                "stats": _linear_stats(pairs),
            }
        )
    return summaries


def _metric_pairs(rows: Iterable[dict[str, Any]], x_key: str, y_key: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        x_value = _to_float(row.get(x_key))
        y_value = _to_float(row.get(y_key))
        if x_value is not None and y_value is not None:
            pairs.append((x_value, y_value))
    return pairs


def _linear_stats(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    n = len(pairs)
    if n < 3:
        return {"n": n, "pearson_r": None, "spearman_rho": None, "r2": None, "slope": None, "intercept": None}
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((value - mean_x) ** 2 for value in xs)
    syy = sum((value - mean_y) ** 2 for value in ys)
    sxy = sum((x_value - mean_x) * (y_value - mean_y) for x_value, y_value in pairs)
    if sxx == 0 or syy == 0:
        return {"n": n, "pearson_r": None, "spearman_rho": None, "r2": None, "slope": None, "intercept": None}
    pearson = sxy / math.sqrt(sxx * syy)
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    spearman = _pearson_on_ranked(xs, ys)
    return {
        "n": n,
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "r2": pearson * pearson,
        "slope": slope,
        "intercept": intercept,
    }


def _pearson_on_ranked(xs: list[float], ys: list[float]) -> float | None:
    ranks_x = _average_ranks(xs)
    ranks_y = _average_ranks(ys)
    stats = _linear_stats_without_spearman(list(zip(ranks_x, ranks_y, strict=True)))
    return stats.get("pearson_r")


def _linear_stats_without_spearman(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    n = len(pairs)
    if n < 3:
        return {"pearson_r": None}
    mean_x = sum(pair[0] for pair in pairs) / n
    mean_y = sum(pair[1] for pair in pairs) / n
    sxx = sum((pair[0] - mean_x) ** 2 for pair in pairs)
    syy = sum((pair[1] - mean_y) ** 2 for pair in pairs)
    sxy = sum((x_value - mean_x) * (y_value - mean_y) for x_value, y_value in pairs)
    if sxx == 0 or syy == 0:
        return {"pearson_r": None}
    return {"pearson_r": sxy / math.sqrt(sxx * syy)}


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + 1 + end) / 2
        for original_index, _ in indexed[index:end]:
            ranks[original_index] = rank
        index = end
    return ranks


def _assign_rank(items: list[dict[str, Any]], *, value_key: str, rank_key: str) -> None:
    ranked = [(index, _to_float(item.get(value_key))) for index, item in enumerate(items)]
    ranked = [(index, value) for index, value in ranked if value is not None]
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    for rank, (index, _) in enumerate(ranked, start=1):
        items[index][rank_key] = rank


def _submission_offline_metrics(layout: WorkspaceLayout, metadata: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    offline = metadata.get("offline_metrics")
    if not isinstance(offline, dict):
        offline = metadata.get("offline_snapshot") if isinstance(metadata.get("offline_snapshot"), dict) else {}

    metrics = {
        "local_bmc200_mean": _first_float(
            offline,
            "bmc_last_200_eras_mean",
            "local_bmc_last_200_mean",
            "bmc_last_200_eras.mean",
        ),
        "local_bmc200_sharpe": _first_float(
            offline,
            "bmc_last_200_eras_sharpe",
            "bmc_last_200_eras.sharpe",
        ),
        "local_bmc200_max_drawdown": _first_float(
            offline,
            "bmc_last_200_eras_max_drawdown",
            "bmc_last_200_eras.max_drawdown",
        ),
        "local_bmc_mean": _first_float(offline, "bmc_mean", "local_bmc_mean", "bmc.mean"),
        "local_corr_mean": _first_float(offline, "corr_mean", "local_corr_mean", "corr.mean"),
        "local_mmc_mean": _first_float(offline, "mmc_mean", "local_mmc_mean", "mmc.mean"),
        "local_fnc_mean": _first_float(offline, "fnc_mean", "local_fnc_mean", "fnc.mean"),
        "local_metric_source": "submission.json",
    }
    if any(value is not None for key, value in metrics.items() if key != "local_metric_source"):
        return metrics, set()

    summaries, source_files = _submission_package_summaries(layout, metadata)
    if summaries:
        tags = _submission_model_tags(
            model_name=_to_non_empty_str(metadata.get("model_name")) or "",
            metadata=metadata,
        )
        target_suffix = _target_group_suffix(_to_non_empty_str(tags.get("target")))
        metrics = {
            "local_bmc200_mean": _summary_stat(
                summaries,
                "mean",
                *_metric_groups("bmc_last_200_eras", target_suffix),
            ),
            "local_bmc200_sharpe": _summary_stat(
                summaries,
                "sharpe",
                *_metric_groups("bmc_last_200_eras", target_suffix),
            ),
            "local_bmc200_max_drawdown": _summary_stat(
                summaries,
                "max_drawdown",
                *_metric_groups("bmc_last_200_eras", target_suffix),
            ),
            "local_bmc_mean": _summary_stat(summaries, "mean", *_metric_groups("bmc", target_suffix)),
            "local_corr_mean": _summary_stat(summaries, "mean", *_metric_groups("corr", target_suffix)),
            "local_mmc_mean": _summary_stat(summaries, "mean", *_metric_groups("mmc", target_suffix)),
            "local_fnc_mean": _summary_stat(summaries, "mean", *_metric_groups("fnc", target_suffix)),
            "local_metric_source": "package_validation_summary",
        }
        return metrics, source_files

    return {
        "local_bmc200_mean": None,
        "local_bmc200_sharpe": None,
        "local_bmc200_max_drawdown": None,
        "local_bmc_mean": None,
        "local_corr_mean": None,
        "local_mmc_mean": None,
        "local_fnc_mean": None,
        "local_metric_source": None,
    }, source_files


def _submission_package_summaries(
    layout: WorkspaceLayout,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], set[str]]:
    source_files: set[str] = set()
    source = metadata.get("source") if isinstance(metadata.get("source"), dict) else {}
    package_path_raw = _to_non_empty_str(source.get("package_path"))
    package_path = Path(package_path_raw) if package_path_raw else None
    if package_path is not None and not package_path.is_absolute():
        package_path = layout.workspace_root / package_path
    if package_path is None:
        experiment_id = _to_non_empty_str(source.get("experiment_id"))
        package_id = _to_non_empty_str(source.get("package_id"))
        if experiment_id and package_id:
            package_path = layout.experiments_root / experiment_id / "submission_packages" / package_id
    if package_path is None:
        return {}, source_files

    package_path = package_path.expanduser()
    package_json_path = package_path / "package.json"
    package = _read_json_dict(package_json_path)
    if package_json_path.is_file():
        source_files.add(str(package_json_path))
    artifacts = package.get("artifacts") if isinstance(package.get("artifacts"), dict) else {}
    summaries_path_raw = _to_non_empty_str(artifacts.get("last_validation_eval_summaries_path"))
    summaries_path = Path(summaries_path_raw) if summaries_path_raw else None
    if summaries_path is not None and not summaries_path.is_absolute():
        summaries_path = package_path / summaries_path
    if summaries_path is None:
        candidate = package_path / "artifacts" / "eval" / "validation" / "pickle" / "summaries.json"
        summaries_path = candidate if candidate.is_file() else None
    summaries = _read_json_dict(summaries_path) if summaries_path is not None else {}
    if summaries_path is not None and summaries_path.is_file():
        source_files.add(str(summaries_path))
    return summaries, source_files


def _metric_groups(metric: str, target_suffix: str | None) -> tuple[str, ...]:
    groups = []
    if target_suffix:
        groups.append(f"{metric}_{target_suffix}")
    groups.extend((f"{metric}_target", metric))
    return tuple(groups)


def _target_group_suffix(target: str | None) -> str | None:
    if target == "ender20":
        return "target_ender_20"
    if target == "ender60":
        return "target_ender_60"
    if target == "cyrusd20":
        return "target_cyrusd_20"
    if target == "cyrusd60":
        return "target_cyrusd_60"
    return None


def _summary_stat(summaries: dict[str, Any], stat: str, *groups: str) -> float | None:
    for group in groups:
        value = summaries.get(group)
        if isinstance(value, dict):
            number = _to_float(value.get(stat))
            if number is not None:
                return number
        dotted = _to_float(summaries.get(f"{group}.{stat}"))
        if dotted is not None:
            return dotted
    return None


def _submission_live_started_at(metadata: dict[str, Any]) -> str | None:
    hosted = metadata.get("hosted_pickle") if isinstance(metadata.get("hosted_pickle"), dict) else {}
    numerai = metadata.get("numerai") if isinstance(metadata.get("numerai"), dict) else {}
    return (
        _to_non_empty_str(hosted.get("uploaded_at"))
        or _to_non_empty_str(numerai.get("hosted_pickle_inserted_at"))
        or _to_non_empty_str(metadata.get("woke_at"))
        or _to_non_empty_str(metadata.get("created_at"))
    )


def _submission_model_tags(*, model_name: str, metadata: dict[str, Any]) -> dict[str, Any]:
    source = metadata.get("source") if isinstance(metadata.get("source"), dict) else {}
    recipe = _to_non_empty_str(source.get("recipe"))
    package_id = _to_non_empty_str(source.get("package_id"))
    name = model_name.lower()

    if "cross_scope" in name:
        feature_scope = "blend"
        target = "cross_scope"
        target_horizon = None
        recipe = recipe or "cross_scope"
    else:
        feature_scope = None
        if name.startswith("lgbm_s_"):
            feature_scope = "small"
        elif name.startswith("lgbm_m_"):
            feature_scope = "medium"
        elif name.startswith("lgbm_deep_"):
            feature_scope = "deep"

        target = None
        target_horizon = None
        for candidate in ("ender20", "ender60", "cyrusd20", "cyrusd60"):
            if candidate in name:
                target = candidate
                target_horizon = "60" if candidate.endswith("60") else "20"
                break

    family = "lgbm" if "lgbm" in name else name.split("_", 1)[0]
    return {
        "family": family,
        "feature_scope": feature_scope,
        "target": target,
        "target_horizon": target_horizon,
        "recipe": recipe,
        "package_id": package_id,
    }


def _read_round_rows(path: Path) -> list[dict[str, Any]]:
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return []
    return [_sanitize_row(row) for row in frame.to_dict(orient="records")]


def _read_json_dict(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(
            json.dumps(_sanitize_json(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temp_path, index=False)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _sanitize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _sanitize_scalar(value) for key, value in row.items()}


def _sanitize_json(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(key): _sanitize_json(value) for key, value in payload.items()}
    if isinstance(payload, list | tuple):
        return [_sanitize_json(value) for value in payload]
    return _sanitize_scalar(payload)


def _sanitize_scalar(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return value


def _to_non_empty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_float(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _to_float(payload.get(key))
        if value is not None:
            return value
    return None


def _first_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    for row in rows:
        value = _to_float(row.get(key))
        if value is not None:
            return value
    return None


def _mean_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_to_float(row.get(key)) for row in rows]
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


__all__ = [
    "CALIBRATION_MANIFEST_FILENAME",
    "CALIBRATION_RELATIVE_DIR",
    "CALIBRATION_REPORT_FILENAME",
    "CALIBRATION_ROWS_FILENAME",
    "CalibrationMaterializeResult",
    "CalibrationReportResult",
    "build_live_calibration_report",
    "calibration_artifact_root",
    "load_live_calibration_report",
    "load_live_calibration_rows",
    "materialize_live_calibration",
]
