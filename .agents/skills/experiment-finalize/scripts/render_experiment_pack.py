#!/usr/bin/env python3
"""Render a finalized numereng experiment pack from run or ensemble artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PACK_FILENAME = "EXPERIMENT.pack.md"
BASE_PROMPT_PATH = Path(__file__).resolve().parents[1] / "references" / "base-prompt.md"
FALLBACK_METRIC_COLUMNS: tuple[str, ...] = (
    "bmc_last_200_eras_mean",
    "bmc_mean",
    "bmc_std",
    "bmc_sharpe",
    "corr_mean",
    "corr_std",
    "corr_sharpe",
    "fnc_mean",
    "fnc_std",
    "fnc_sharpe",
    "mmc_mean",
    "mmc_std",
    "mmc_sharpe",
    "mmc_coverage_ratio_rows",
    "cwmm_mean",
    "cwmm_std",
    "cwmm_sharpe",
    "max_drawdown",
)

METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "bmc_last_200_eras_mean": (
        "bmc_last_200_eras_mean",
        "bmc_ender20_last_200_eras.mean",
        "bmc_last_200_eras.mean",
    ),
    "bmc_mean": ("bmc_mean", "bmc_ender20.mean", "bmc.mean"),
    "bmc_std": ("bmc_std", "bmc_ender20.std", "bmc.std"),
    "bmc_sharpe": ("bmc_sharpe", "bmc_ender20.sharpe", "bmc.sharpe"),
    "corr_mean": ("corr_native_mean", "corr_mean", "corr.mean", "corr20v2_mean"),
    "corr_std": ("corr_native_std", "corr_std", "corr.std", "corr20v2_std"),
    "corr_sharpe": (
        "corr_native_sharpe",
        "corr_sharpe",
        "corr.sharpe",
        "corr20v2_sharpe",
        "sharpe",
    ),
    "fnc_mean": ("fnc_native_mean", "fnc_mean", "fnc.mean"),
    "fnc_std": ("fnc_native_std", "fnc_std", "fnc.std"),
    "fnc_sharpe": ("fnc_native_sharpe", "fnc_sharpe", "fnc.sharpe"),
    "mmc_mean": ("mmc_mean", "mmc_ender20.mean", "mmc.mean"),
    "mmc_std": ("mmc_std", "mmc_ender20.std", "mmc.std"),
    "mmc_sharpe": ("mmc_sharpe", "mmc_ender20.sharpe", "mmc.sharpe"),
    "cwmm_mean": ("cwmm_mean", "cwmm.mean"),
    "cwmm_std": ("cwmm_std", "cwmm.std"),
    "cwmm_sharpe": ("cwmm_sharpe", "cwmm.sharpe"),
    "max_drawdown": ("corr_native_max_drawdown", "max_drawdown", "corr.max_drawdown"),
}

REQUIRED_RUN_FILES: tuple[str, ...] = (
    "run.json",
    "metrics.json",
    "results.json",
    "resolved.json",
    "score_provenance.json",
)
REQUIRED_SCORING_FILES: tuple[str, ...] = (
    "manifest.json",
    "run_metric_series.parquet",
    "post_training_core_summary.parquet",
)


@dataclass(frozen=True)
class RunRow:
    run_id: str
    config: str
    model: str
    target: str
    feature_set: str
    status: str
    metrics: dict[str, float | None]


class PackError(Exception):
    """Raised when experiment pack rendering cannot proceed safely."""


def main() -> int:
    args = parse_args()
    try:
        workspace = args.workspace.resolve()
        store_root = (workspace / ".numereng").resolve()
        experiment_id = resolve_experiment_id(args.experiment_id, args.experiment_path)
        experiment_dir = resolve_experiment_dir(store_root, experiment_id, args.experiment_path)
        rendered, rows, columns = render_pack(
            workspace=workspace,
            store_root=store_root,
            experiment_dir=experiment_dir,
            experiment_context_path=args.experiment_context_path,
        )
    except PackError as exc:
        print(f"experiment_pack_failed:{exc}", file=sys.stderr)
        return 2

    output_path = experiment_dir / PACK_FILENAME
    if args.dry_run:
        print(rendered)
        print(
            f"\n<!-- dry_run: output_path={output_path} rows={len(rows)} "
            f"metric_columns={len(columns)} -->"
        )
        return 0

    output_path.write_text(rendered, encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "rows": len(rows), "metric_columns": list(columns)}))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render EXPERIMENT.pack.md for one completed numereng experiment.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--experiment-id", help="Experiment id under .numereng/experiments.")
    target.add_argument("--experiment-path", type=Path, help="Path to an experiment directory.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Repo workspace root. Defaults to cwd.")
    parser.add_argument("--dry-run", action="store_true", help="Render to stdout without writing EXPERIMENT.pack.md.")
    parser.add_argument(
        "--experiment-context-path",
        type=Path,
        help="Generated experiment-specific prompt context to prepend after the base prompt.",
    )
    return parser.parse_args()


def resolve_experiment_id(experiment_id: str | None, experiment_path: Path | None) -> str:
    if experiment_id:
        return experiment_id
    if experiment_path is None:
        raise PackError("experiment_target_missing")
    return experiment_path.name


def resolve_experiment_dir(store_root: Path, experiment_id: str, experiment_path: Path | None) -> Path:
    if experiment_path is not None:
        path = experiment_path.resolve()
    else:
        path = store_root / "experiments" / experiment_id
    if not path.is_dir():
        raise PackError(f"experiment_dir_missing:{path}")
    return path


def render_pack(
    *,
    workspace: Path,
    store_root: Path,
    experiment_dir: Path,
    experiment_context_path: Path | None,
) -> tuple[str, list[RunRow], tuple[str, ...]]:
    manifest = load_mapping(experiment_dir / "experiment.json", "experiment_manifest_invalid")
    experiment_id = str(manifest.get("experiment_id") or experiment_dir.name)
    run_ids = normalize_run_ids(manifest.get("runs"))

    doc_path = experiment_dir / "EXPERIMENT.md"
    if not doc_path.is_file():
        raise PackError(f"experiment_doc_missing:{doc_path}")
    notes_body = doc_path.read_text(encoding="utf-8").strip()

    metric_columns = load_runops_metric_columns(workspace)
    if run_ids:
        rows = [load_run_row(store_root=store_root, run_id=run_id, metric_columns=metric_columns) for run_id in run_ids]
        row_label = "run"
    else:
        rows = load_ensemble_rows(store_root=store_root, experiment_dir=experiment_dir, metric_columns=metric_columns)
        row_label = "ensemble"
    rows.sort(key=run_sort_key)

    pack_body = "\n".join(
        [
            "# Experiment Pack",
            "",
            "## Pack Metadata",
            f"- experiment_id: `{markdown_inline(experiment_id)}`",
            f"- name: {markdown_code(str(manifest.get('name') or experiment_dir.name))}",
            f"- status: `{markdown_inline(str(manifest.get('status') or 'draft'))}`",
            f"- champion_run_id: `{markdown_inline(str(manifest.get('champion_run_id') or 'none'))}`",
            f"- packed_at: `{datetime.now(UTC).isoformat()}`",
            f"- source_markdown_path: `{markdown_inline('EXPERIMENT.md')}`",
            f"- output_path: `{markdown_inline(PACK_FILENAME)}`",
            f"- table_row_type: `{row_label}`",
            f"- run_table_rows: `{len(rows)}`",
            "",
            "## Experiment Notes",
            "",
            notes_body or "_Empty_",
            "",
            "## Run Ops Metrics Table",
            "",
            *render_run_table(rows, metric_columns),
            "",
        ]
    )
    rendered = prepend_prompt_context(pack_body, experiment_context_path)
    return rendered, rows, metric_columns


def prepend_prompt_context(pack_body: str, experiment_context_path: Path | None) -> str:
    if experiment_context_path is None:
        return pack_body
    base_prompt = read_required_text(BASE_PROMPT_PATH, "base_prompt_missing")
    experiment_context = read_required_text(experiment_context_path, "experiment_context_missing")
    return "\n\n".join((base_prompt, experiment_context, pack_body))


def read_required_text(path: Path, error_code: str) -> str:
    resolved = path.resolve()
    if not resolved.is_file():
        raise PackError(f"{error_code}:{resolved}")
    try:
        text = resolved.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise PackError(f"{error_code}:{resolved}") from exc
    if not text:
        raise PackError(f"{error_code}:{resolved}:empty")
    return text


def load_run_row(*, store_root: Path, run_id: str, metric_columns: tuple[str, ...]) -> RunRow:
    run_dir = store_root / "runs" / run_id
    if not run_dir.is_dir():
        raise PackError(f"run_dir_missing:{run_id}:{run_dir}")

    for file_name in REQUIRED_RUN_FILES:
        if not (run_dir / file_name).is_file():
            raise PackError(f"run_required_file_missing:{run_id}:{file_name}")
    scoring_dir = run_dir / "artifacts" / "scoring"
    for file_name in REQUIRED_SCORING_FILES:
        if not (scoring_dir / file_name).is_file():
            raise PackError(f"run_required_scoring_file_missing:{run_id}:{file_name}")

    run_manifest = load_mapping(run_dir / "run.json", "run_manifest_invalid")
    status = str(run_manifest.get("status") or "")
    if status != "FINISHED":
        raise PackError(f"run_not_finished:{run_id}:{status or 'unknown'}")

    metrics_payload = load_mapping(run_dir / "metrics.json", "run_metrics_invalid")
    score_provenance = load_mapping(run_dir / "score_provenance.json", "score_provenance_invalid")
    scoring_metrics = load_scoring_summary_metrics(scoring_dir, run_id=run_id)
    normalized = normalize_metrics(
        metrics_payload,
        score_provenance=score_provenance,
        scoring_metrics=scoring_metrics,
        metric_columns=metric_columns,
    )
    return RunRow(
        run_id=run_id,
        config=resolve_config_name(run_manifest),
        model=resolve_model_name(run_manifest),
        target=resolve_target(run_manifest),
        feature_set=resolve_feature_set(run_manifest),
        status=status,
        metrics=normalized,
    )


def load_ensemble_rows(*, store_root: Path, experiment_dir: Path, metric_columns: tuple[str, ...]) -> list[RunRow]:
    ensemble_dirs = sorted(path for path in (experiment_dir / "ensembles").glob("*") if path.is_dir())
    if not ensemble_dirs:
        raise PackError(f"experiment_manifest_has_no_runs:{experiment_dir.name}")
    score_map = compute_ensemble_score_map(store_root=store_root, ensemble_dirs=ensemble_dirs)
    return [
        load_ensemble_row(
            store_root=store_root,
            ensemble_dir=ensemble_dir,
            metric_columns=metric_columns,
            score_map=score_map,
        )
        for ensemble_dir in ensemble_dirs
    ]


def load_ensemble_row(
    *,
    store_root: Path,
    ensemble_dir: Path,
    metric_columns: tuple[str, ...],
    score_map: dict[str, dict[str, float]],
) -> RunRow:
    ensemble_id = ensemble_dir.name
    for file_name in ("lineage.json", "metrics.json", "weights.parquet", "era_metrics.parquet", "predictions.parquet"):
        if not (ensemble_dir / file_name).is_file():
            raise PackError(f"ensemble_required_file_missing:{ensemble_id}:{file_name}")
    lineage = load_mapping(ensemble_dir / "lineage.json", "ensemble_lineage_invalid")
    metrics_payload = load_mapping(ensemble_dir / "metrics.json", "ensemble_metrics_invalid")
    target = str(lineage.get("target") or "-")
    run_ids = normalize_run_ids(lineage.get("run_ids"))
    computed = score_map.get(ensemble_id)
    if computed is None:
        raise PackError(f"ensemble_score_missing:{ensemble_id}")
    flattened = flatten_metrics(metrics_payload)
    metrics = metrics_payload.get("metrics")
    if isinstance(metrics, dict):
        flattened.update(metrics)
    elif isinstance(metrics, list):
        for metric in metrics:
            if isinstance(metric, dict) and isinstance(metric.get("name"), str):
                flattened[metric["name"]] = metric.get("value")
    flattened.update(computed)
    normalized = {
        metric_name: extract_metric(flattened, METRIC_ALIASES.get(metric_name, (metric_name,)))
        for metric_name in metric_columns
    }
    return RunRow(
        run_id=ensemble_id,
        config=ensemble_id,
        model="rank_avg_ensemble",
        target=target,
        feature_set=resolve_ensemble_feature_set(store_root=store_root, run_ids=run_ids),
        status=str(metrics_payload.get("status") or "completed"),
        metrics=normalized,
    )


def compute_ensemble_score_map(*, store_root: Path, ensemble_dirs: list[Path]) -> dict[str, dict[str, float]]:
    try:
        import numpy as np  # type: ignore[import-untyped]
        import pandas as pd  # type: ignore[import-untyped]

        from numereng.features.ensemble.selection import (
            _attach_active_benchmark,
            _era_ranges,
            _score_weight_matrix,
        )
    except Exception as exc:  # pragma: no cover - dependency/runtime guard.
        raise PackError("ensemble_scoring_import_failed") from exc

    by_target: dict[str, list[dict[str, Any]]] = {}
    for ensemble_dir in ensemble_dirs:
        ensemble_id = ensemble_dir.name
        lineage = load_mapping(ensemble_dir / "lineage.json", "ensemble_lineage_invalid")
        target = str(lineage.get("target") or "-")
        run_ids = normalize_run_ids(lineage.get("run_ids"))
        if not run_ids:
            raise PackError(f"ensemble_lineage_has_no_runs:{ensemble_id}")
        predictions = pd.read_parquet(
            ensemble_dir / "predictions.parquet",
            columns=["era", "id", "prediction"],
        ).rename(columns={"prediction": ensemble_id})
        by_target.setdefault(target, []).append(
            {"ensemble_id": ensemble_id, "predictions": predictions, "run_ids": run_ids}
        )

    score_map: dict[str, dict[str, float]] = {}
    for target, items in by_target.items():
        target_frame = load_target_frame(
            store_root=store_root,
            target=target,
            run_ids=items[0]["run_ids"],
            pandas_module=pd,
        )
        merged = _attach_active_benchmark(store_root=store_root, prediction_frame=target_frame)
        ensemble_ids: list[str] = []
        for item in items:
            ensemble_id = str(item["ensemble_id"])
            merged = merged.merge(item["predictions"], on=["era", "id"], how="inner")
            ensemble_ids.append(ensemble_id)
        if merged.empty:
            raise PackError(f"ensemble_target_join_empty:{target}")
        merged = merged.sort_values(["era", "id"]).reset_index(drop=True)
        summary = _score_weight_matrix(
            prediction_matrix=merged[ensemble_ids].to_numpy(dtype=np.float64, copy=False),
            target_vector=merged[target].to_numpy(dtype=np.float64, copy=False),
            benchmark_vector=merged["active_benchmark"].to_numpy(dtype=np.float64, copy=False),
            era_ranges=_era_ranges(merged["era"].astype(str).tolist()),
            weight_matrix=np.eye(len(ensemble_ids), dtype=np.float64),
        )
        for index, ensemble_id in enumerate(ensemble_ids):
            if coerce_float(summary["bmc_mean"][index]) is None:
                raise PackError(f"ensemble_scoring_empty:{ensemble_id}")
            score_map[ensemble_id] = {
                "bmc_last_200_eras_mean": float(summary["bmc_last_200_eras_mean"][index]),
                "bmc_mean": float(summary["bmc_mean"][index]),
                "bmc_std": float(summary["bmc_std"][index]),
                "bmc_sharpe": float(summary["bmc_sharpe"][index]),
                "corr_mean": float(summary["corr_mean"][index]),
                "corr_std": float(summary["corr_std"][index]),
                "corr_sharpe": float(summary["corr_sharpe"][index]),
            }
    return score_map


def load_target_frame(*, store_root: Path, target: str, run_ids: list[str], pandas_module: Any) -> Any:
    for run_id in run_ids:
        run_dir = store_root / "runs" / run_id
        manifest = load_mapping(run_dir / "run.json", "run_manifest_invalid")
        artifact_path = manifest.get("artifacts", {}).get("predictions")
        if not isinstance(artifact_path, str) or not artifact_path:
            continue
        predictions_path = run_dir / artifact_path.replace("\\", "/")
        if not predictions_path.is_file():
            continue
        columns = pandas_module.read_parquet(predictions_path).columns
        if target in columns:
            return pandas_module.read_parquet(predictions_path, columns=["era", "id", target])
    raise PackError(f"ensemble_target_column_missing:{target}")


def resolve_ensemble_feature_set(*, store_root: Path, run_ids: list[str]) -> str:
    values: list[str] = []
    for run_id in run_ids:
        run_dir = store_root / "runs" / run_id
        if not (run_dir / "run.json").is_file():
            continue
        value = resolve_feature_set(load_mapping(run_dir / "run.json", "run_manifest_invalid"))
        if value != "-" and value not in values:
            values.append(value)
    return ",".join(values) if values else "-"


def load_runops_metric_columns(workspace: Path) -> tuple[str, ...]:
    canonical_path = workspace / "viz" / "web" / "src" / "lib" / "metrics" / "canonical.ts"
    if not canonical_path.is_file():
        return FALLBACK_METRIC_COLUMNS
    text = canonical_path.read_text(encoding="utf-8")
    match = re.search(r"RUNOPS_ALL_SCORING_METRICS\s*=\s*\[(.*?)\]\s+as const", text, flags=re.S)
    if match is None:
        return FALLBACK_METRIC_COLUMNS
    columns = tuple(re.findall(r"'([^']+)'", match.group(1)))
    return columns or FALLBACK_METRIC_COLUMNS


def normalize_metrics(
    metrics_payload: dict[str, Any],
    *,
    score_provenance: dict[str, Any],
    scoring_metrics: dict[str, float],
    metric_columns: tuple[str, ...],
) -> dict[str, float | None]:
    flattened = flatten_metrics(metrics_payload)
    flattened.update(scoring_metrics)
    normalized: dict[str, float | None] = {}
    for metric_name in metric_columns:
        if metric_name == "mmc_coverage_ratio_rows":
            normalized[metric_name] = coverage_ratio(score_provenance)
            continue
        normalized[metric_name] = extract_metric(flattened, METRIC_ALIASES.get(metric_name, (metric_name,)))
    return normalized


def load_scoring_summary_metrics(scoring_dir: Path, *, run_id: str) -> dict[str, float]:
    merged: dict[str, float] = {}
    for file_name in ("post_training_core_summary.parquet", "post_training_full_summary.parquet"):
        path = scoring_dir / file_name
        if not path.is_file():
            continue
        try:
            import pandas as pd  # type: ignore[import-untyped]

            frame = pd.read_parquet(path)
        except Exception as exc:
            raise PackError(f"run_scoring_summary_invalid:{run_id}:{file_name}") from exc
        if frame.empty:
            raise PackError(f"run_scoring_summary_empty:{run_id}:{file_name}")
        row = frame.iloc[0].to_dict()
        for key, value in row.items():
            number = coerce_float(value)
            if number is not None:
                merged[str(key)] = number
    return merged


def flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key_raw, value in metrics.items():
        key = str(key_raw)
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_metrics(value, full_key))
        else:
            flattened[full_key] = value
    return flattened


def extract_metric(metrics: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = coerce_float(metrics.get(key))
        if value is not None:
            return value
    return None


def coverage_ratio(score_provenance: dict[str, Any]) -> float | None:
    joins = score_provenance.get("joins")
    if not isinstance(joins, dict):
        return None
    overlap = coerce_float(joins.get("meta_overlap_rows"))
    predictions = coerce_float(joins.get("predictions_rows"))
    if overlap is None or predictions is None or predictions == 0:
        return None
    return overlap / predictions


def run_sort_key(row: RunRow) -> tuple[bool, float, str]:
    primary = row.metrics.get("bmc_last_200_eras_mean")
    if primary is None:
        return (True, 0.0, row.run_id)
    return (False, -primary, row.run_id)


def render_run_table(rows: list[RunRow], metric_columns: tuple[str, ...]) -> list[str]:
    columns = ("run_id", "config", "model", "target", "feature_set", "status", *metric_columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [
            row.run_id,
            row.config,
            row.model,
            row.target,
            row.feature_set,
            row.status,
            *(format_metric(row.metrics.get(metric_name)) for metric_name in metric_columns),
        ]
        lines.append("| " + " | ".join(markdown_cell(value) for value in values) + " |")
    return lines


def resolve_config_name(run_manifest: dict[str, Any]) -> str:
    config = run_manifest.get("config")
    if isinstance(config, dict):
        path = config.get("path")
        if isinstance(path, str) and path:
            return Path(path).name
    name = run_manifest.get("run_name")
    return str(name) if name else "-"


def resolve_model_name(run_manifest: dict[str, Any]) -> str:
    model = run_manifest.get("model")
    if isinstance(model, dict):
        for key in ("type", "model_type", "name"):
            value = model.get(key)
            if isinstance(value, str) and value:
                return value
    value = run_manifest.get("model_type")
    return str(value) if value else "-"


def resolve_target(run_manifest: dict[str, Any]) -> str:
    for section_name in ("data", "training", "config"):
        section = run_manifest.get(section_name)
        if isinstance(section, dict):
            for key in ("target_payout", "target_train", "target_col", "target"):
                value = section.get(key)
                if isinstance(value, str) and value:
                    return value
    return "-"


def resolve_feature_set(run_manifest: dict[str, Any]) -> str:
    for section_name in ("features", "data", "training"):
        section = run_manifest.get(section_name)
        if isinstance(section, dict):
            for key in ("feature_set", "feature_scope", "x_groups"):
                value = section.get(key)
                if isinstance(value, str) and value:
                    return value
                if isinstance(value, list) and value:
                    return ",".join(str(item) for item in value)
    return "-"


def normalize_run_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def load_mapping(path: Path, error_code: str) -> dict[str, Any]:
    if not path.is_file():
        raise PackError(f"{error_code}:{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackError(f"{error_code}:{path}") from exc
    if not isinstance(payload, dict):
        raise PackError(f"{error_code}:{path}")
    return payload


def coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def markdown_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_inline(value: str) -> str:
    return value.replace("`", "\\`")


def markdown_code(value: str) -> str:
    return f"`{markdown_inline(value)}`"


if __name__ == "__main__":
    raise SystemExit(main())
