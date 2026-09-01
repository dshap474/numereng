#!/usr/bin/env python3
"""Summarize run or ensemble experiment evidence for final report writing."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

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
class EvidenceResult:
    experiment_id: str
    output_path: Path
    rows: int
    targets: int
    finished: int


class EvidenceError(Exception):
    """Raised when experiment evidence cannot be summarized safely."""


def main() -> int:
    args = parse_args()
    try:
        workspace = args.workspace.resolve()
        store_root = (workspace / ".numereng").resolve()
        experiment_id = resolve_experiment_id(args.experiment_id, args.experiment_path)
        experiment_dir = resolve_experiment_dir(store_root, experiment_id, args.experiment_path)
        rendered, result = render_evidence(
            store_root=store_root,
            experiment_dir=experiment_dir,
            output_path=args.output_path,
        )
    except EvidenceError as exc:
        print(f"experiment_evidence_failed:{exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(rendered)
        print(
            f"\n<!-- dry_run: output_path={result.output_path} rows={result.rows} "
            f"targets={result.targets} finished={result.finished} -->"
        )
        return 0

    result.output_path.parent.mkdir(parents=True, exist_ok=True)
    result.output_path.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment_id": result.experiment_id,
                "output_path": str(result.output_path),
                "rows": result.rows,
                "targets": result.targets,
                "finished": result.finished,
            }
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize evidence for a completed numereng experiment.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--experiment-id", help="Experiment id under .numereng/experiments.")
    target.add_argument("--experiment-path", type=Path, help="Path to an experiment directory.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Repo workspace root. Defaults to cwd.")
    parser.add_argument("--output-path", type=Path, help="Output markdown path. Defaults to .numereng/tmp.")
    parser.add_argument("--dry-run", action="store_true", help="Print evidence markdown without writing.")
    return parser.parse_args()


def resolve_experiment_id(experiment_id: str | None, experiment_path: Path | None) -> str:
    if experiment_id:
        return experiment_id
    if experiment_path is None:
        raise EvidenceError("experiment_target_missing")
    return experiment_path.name


def resolve_experiment_dir(store_root: Path, experiment_id: str, experiment_path: Path | None) -> Path:
    if experiment_path is not None:
        path = experiment_path.resolve()
    else:
        path = store_root / "experiments" / experiment_id
    if not path.is_dir():
        raise EvidenceError(f"experiment_dir_missing:{path}")
    return path


def render_evidence(
    *,
    store_root: Path,
    experiment_dir: Path,
    output_path: Path | None,
) -> tuple[str, EvidenceResult]:
    manifest = load_mapping(experiment_dir / "experiment.json", "experiment_manifest_invalid")
    experiment_id = str(manifest.get("experiment_id") or experiment_dir.name)
    run_ids = normalize_run_ids(manifest.get("runs"))
    if run_ids:
        rows = [load_run_evidence(store_root=store_root, run_id=run_id) for run_id in run_ids]
        rendered = build_markdown(experiment_id=experiment_id, manifest=manifest, frame=pd.DataFrame(rows))
    else:
        rows = load_ensemble_evidence_rows(store_root=store_root, experiment_dir=experiment_dir)
        rendered = build_ensemble_markdown(experiment_id=experiment_id, manifest=manifest, frame=pd.DataFrame(rows))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise EvidenceError(f"experiment_has_no_rows:{experiment_id}")

    target_path = output_path or store_root / "tmp" / "experiment-finalize" / f"{experiment_id}.evidence.md"
    result = EvidenceResult(
        experiment_id=experiment_id,
        output_path=target_path,
        rows=len(frame),
        targets=int(frame["target"].nunique()),
        finished=int(frame["status"].isin(["FINISHED", "completed"]).sum()),
    )
    return rendered, result


def load_run_evidence(*, store_root: Path, run_id: str) -> dict[str, Any]:
    run_dir = store_root / "runs" / run_id
    if not run_dir.is_dir():
        raise EvidenceError(f"run_dir_missing:{run_id}:{run_dir}")
    for file_name in REQUIRED_RUN_FILES:
        if not (run_dir / file_name).is_file():
            raise EvidenceError(f"run_required_file_missing:{run_id}:{file_name}")

    scoring_dir = run_dir / "artifacts" / "scoring"
    for file_name in REQUIRED_SCORING_FILES:
        if not (scoring_dir / file_name).is_file():
            raise EvidenceError(f"run_required_scoring_file_missing:{run_id}:{file_name}")

    run_manifest = load_mapping(run_dir / "run.json", "run_manifest_invalid")
    metrics = load_mapping(run_dir / "metrics.json", "run_metrics_invalid")
    status = str(run_manifest.get("status") or "")
    if status != "FINISHED":
        raise EvidenceError(f"run_not_finished:{run_id}:{status or 'unknown'}")

    core = read_summary_row(scoring_dir / "post_training_core_summary.parquet", run_id=run_id)
    full_path = scoring_dir / "post_training_full_summary.parquet"
    full = read_summary_row(full_path, run_id=run_id) if full_path.is_file() else {}
    config = resolve_config_name(run_manifest)
    target = resolve_target(config, run_manifest)
    return {
        "run_id": run_id,
        "config": config,
        "model": resolve_model_name(run_manifest),
        "target": target,
        "horizon": resolve_horizon(target),
        "seed": resolve_seed(config),
        "feature_set": resolve_feature_set(run_manifest),
        "status": status,
        "bmc_last_200_eras_mean": number(core.get("bmc_last_200_eras_mean")),
        "bmc_mean": number(core.get("bmc_mean")),
        "bmc_std": number(core.get("bmc_std")),
        "bmc_sharpe": number(core.get("bmc_sharpe")),
        "corr_mean": number(core.get("corr_native_mean")),
        "corr_sharpe": number(core.get("corr_native_sharpe")),
        "fnc_mean": number(full.get("fnc_native_mean")),
        "fnc_sharpe": number(full.get("fnc_native_sharpe")),
        "mmc_mean": number(core.get("mmc_mean")),
        "cwmm_mean": nested_number(metrics, "cwmm", "mean"),
        "max_drawdown": number(core.get("corr_native_max_drawdown")),
    }


def load_ensemble_evidence_rows(*, store_root: Path, experiment_dir: Path) -> list[dict[str, Any]]:
    ensemble_dirs = sorted(path for path in (experiment_dir / "ensembles").glob("*") if path.is_dir())
    if not ensemble_dirs:
        raise EvidenceError(f"experiment_manifest_has_no_runs:{experiment_dir.name}")
    score_map = compute_ensemble_score_map(store_root=store_root, ensemble_dirs=ensemble_dirs)
    return [
        load_ensemble_evidence(
            store_root=store_root,
            ensemble_dir=path,
            score_map=score_map,
        )
        for path in ensemble_dirs
    ]


def load_ensemble_evidence(
    *,
    store_root: Path,
    ensemble_dir: Path,
    score_map: dict[str, dict[str, float]],
) -> dict[str, Any]:
    ensemble_id = ensemble_dir.name
    for file_name in ("lineage.json", "metrics.json", "weights.parquet", "era_metrics.parquet", "predictions.parquet"):
        if not (ensemble_dir / file_name).is_file():
            raise EvidenceError(f"ensemble_required_file_missing:{ensemble_id}:{file_name}")
    lineage = load_mapping(ensemble_dir / "lineage.json", "ensemble_lineage_invalid")
    metrics_payload = load_mapping(ensemble_dir / "metrics.json", "ensemble_metrics_invalid")
    target = str(lineage.get("target") or "-")
    run_ids = normalize_run_ids(lineage.get("run_ids"))
    computed = score_map.get(ensemble_id)
    if computed is None:
        raise EvidenceError(f"ensemble_score_missing:{ensemble_id}")
    persisted = flatten_metric_list(metrics_payload)
    family_weights = compute_family_weights(
        store_root=store_root,
        run_ids=run_ids,
        weights_path=ensemble_dir / "weights.parquet",
    )
    return {
        "run_id": ensemble_id,
        "ensemble_id": ensemble_id,
        "config": ensemble_id,
        "model": "rank_avg_ensemble",
        "target": target,
        "horizon": resolve_horizon(target),
        "seed": None,
        "feature_set": resolve_ensemble_feature_set(store_root=store_root, run_ids=run_ids),
        "status": str(metrics_payload.get("status") or "completed"),
        "source_runs": len(run_ids),
        "bmc_last_200_eras_mean": computed["bmc_last_200_eras_mean"],
        "bmc_mean": computed["bmc_mean"],
        "bmc_std": computed["bmc_std"],
        "bmc_sharpe": computed["bmc_sharpe"],
        "corr_mean": computed["corr_mean"],
        "corr_sharpe": computed["corr_sharpe"],
        "fnc_mean": None,
        "fnc_sharpe": None,
        "mmc_mean": None,
        "cwmm_mean": None,
        "max_drawdown": number(persisted.get("max_drawdown")),
        "jasper60_weight": family_weights.get("target_jasper_60", 0.0),
        "ender60_weight": family_weights.get("target_ender_60", 0.0),
        "ender20_weight": family_weights.get("target_ender_20", 0.0),
        "alpha20_weight": family_weights.get("target_alpha_20", 0.0),
    }


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
        raise EvidenceError("ensemble_scoring_import_failed") from exc

    by_target: dict[str, list[dict[str, Any]]] = {}
    for ensemble_dir in ensemble_dirs:
        ensemble_id = ensemble_dir.name
        lineage = load_mapping(ensemble_dir / "lineage.json", "ensemble_lineage_invalid")
        target = str(lineage.get("target") or "-")
        run_ids = normalize_run_ids(lineage.get("run_ids"))
        if not run_ids:
            raise EvidenceError(f"ensemble_lineage_has_no_runs:{ensemble_id}")
        predictions = pd.read_parquet(
            ensemble_dir / "predictions.parquet",
            columns=["era", "id", "prediction"],
        ).rename(columns={"prediction": ensemble_id})
        by_target.setdefault(target, []).append(
            {"ensemble_id": ensemble_id, "predictions": predictions, "run_ids": run_ids}
        )

    score_map: dict[str, dict[str, float]] = {}
    for target, items in by_target.items():
        target_frame = load_target_frame(store_root=store_root, target=target, run_ids=items[0]["run_ids"])
        merged = _attach_active_benchmark(store_root=store_root, prediction_frame=target_frame)
        ensemble_ids: list[str] = []
        for item in items:
            ensemble_id = str(item["ensemble_id"])
            merged = merged.merge(item["predictions"], on=["era", "id"], how="inner")
            ensemble_ids.append(ensemble_id)
        if merged.empty:
            raise EvidenceError(f"ensemble_target_join_empty:{target}")
        merged = merged.sort_values(["era", "id"]).reset_index(drop=True)
        summary = _score_weight_matrix(
            prediction_matrix=merged[ensemble_ids].to_numpy(dtype=np.float64, copy=False),
            target_vector=merged[target].to_numpy(dtype=np.float64, copy=False),
            benchmark_vector=merged["active_benchmark"].to_numpy(dtype=np.float64, copy=False),
            era_ranges=_era_ranges(merged["era"].astype(str).tolist()),
            weight_matrix=np.eye(len(ensemble_ids), dtype=np.float64),
        )
        for index, ensemble_id in enumerate(ensemble_ids):
            if number(summary["bmc_mean"][index]) is None:
                raise EvidenceError(f"ensemble_scoring_empty:{ensemble_id}")
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


def load_target_frame(*, store_root: Path, target: str, run_ids: list[str]) -> pd.DataFrame:
    for run_id in run_ids:
        run_dir = store_root / "runs" / run_id
        manifest = load_mapping(run_dir / "run.json", "run_manifest_invalid")
        artifact_path = manifest.get("artifacts", {}).get("predictions")
        if not isinstance(artifact_path, str) or not artifact_path:
            continue
        predictions_path = run_dir / artifact_path.replace("\\", "/")
        if not predictions_path.is_file():
            continue
        columns = pd.read_parquet(predictions_path).columns
        if target in columns:
            return pd.read_parquet(predictions_path, columns=["era", "id", target])
    raise EvidenceError(f"ensemble_target_column_missing:{target}")


def flatten_metric_list(metrics_payload: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    metrics = metrics_payload.get("metrics")
    if isinstance(metrics, dict):
        flattened.update(metrics)
    elif isinstance(metrics, list):
        for metric in metrics:
            if isinstance(metric, dict) and isinstance(metric.get("name"), str):
                flattened[metric["name"]] = metric.get("value")
    return flattened


def compute_family_weights(*, store_root: Path, run_ids: list[str], weights_path: Path) -> dict[str, float]:
    weights_frame = pd.read_parquet(weights_path)
    weight_by_run = dict(zip(weights_frame["run_id"], weights_frame["weight"], strict=False))
    family_weights: dict[str, float] = {}
    for run_id in run_ids:
        run_dir = store_root / "runs" / run_id
        if not (run_dir / "run.json").is_file():
            continue
        config = resolve_config_name(load_mapping(run_dir / "run.json", "run_manifest_invalid"))
        target = resolve_target(config, load_mapping(run_dir / "run.json", "run_manifest_invalid"))
        family_weights[target] = family_weights.get(target, 0.0) + float(weight_by_run.get(run_id, 0.0))
    return family_weights


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


def build_markdown(*, experiment_id: str, manifest: dict[str, Any], frame: pd.DataFrame) -> str:
    target_agg = aggregate_targets(frame)
    seed_agg = aggregate_seeds(frame)
    horizon_agg = aggregate_horizons(frame)
    top_runs = frame.sort_values(["bmc_last_200_eras_mean", "run_id"], ascending=[False, True]).head(12)
    bottom_targets = target_agg.sort_values(["bmc_l200_mean", "target"], ascending=[True, True]).head(8)
    top_targets = target_agg.sort_values(["bmc_l200_mean", "target"], ascending=[False, True]).head(8)
    high_corr_traps = target_agg[
        (target_agg["corr_mean"] >= target_agg["corr_mean"].quantile(0.75))
        & ((target_agg["bmc_l200_mean"] <= 0) | (target_agg["bmc_mean"] <= 0))
    ].sort_values(["corr_mean", "target"], ascending=[False, True])
    bmc_mmc_conflicts = target_agg[
        ((target_agg["bmc_l200_mean"] > 0) & (target_agg["mmc_mean"] < 0))
        | ((target_agg["bmc_mean"] < 0) & (target_agg["mmc_mean"] > 0))
    ].sort_values(["bmc_l200_mean", "target"], ascending=[False, True])
    drawdown_warnings = frame[frame["max_drawdown"] >= 0.25].sort_values(
        ["max_drawdown", "run_id"], ascending=[False, True]
    )

    best_target = top_targets.iloc[0]
    best_run = top_runs.iloc[0]
    target_l200 = frame.groupby("target")["bmc_last_200_eras_mean"]
    target_table_columns = [
        "target",
        "runs",
        "bmc_l200_mean",
        "bmc_l200_max",
        "bmc_mean",
        "corr_mean",
        "mmc_mean",
        "cwmm_mean",
        "max_drawdown_mean",
    ]
    lines = [
        "# Experiment Evidence Brief",
        "",
        "## Metadata",
        f"- experiment_id: `{experiment_id}`",
        f"- name: `{manifest.get('name') or experiment_id}`",
        f"- manifest_status: `{manifest.get('status') or 'unknown'}`",
        f"- champion_run_id: `{manifest.get('champion_run_id') or 'none'}`",
        "",
        "## Matrix Totals",
        "",
        markdown_table(
            [
                ("manifest_runs", len(frame)),
                ("finished_runs", int((frame["status"] == "FINISHED").sum())),
                ("targets", int(frame["target"].nunique())),
                ("seeds", int(frame["seed"].nunique())),
                ("positive_bmc_last_200_runs", int((frame["bmc_last_200_eras_mean"] > 0).sum())),
                ("positive_bmc_mean_runs", int((frame["bmc_mean"] > 0).sum())),
                ("positive_mmc_mean_runs", int((frame["mmc_mean"] > 0).sum())),
                ("targets_positive_any_seed_l200", int(target_l200.max().gt(0).sum())),
                ("targets_positive_mean_l200", int(target_l200.mean().gt(0).sum())),
                ("mean_bmc_last_200_eras_mean", frame["bmc_last_200_eras_mean"].mean()),
                ("median_bmc_last_200_eras_mean", frame["bmc_last_200_eras_mean"].median()),
                ("mean_corr_mean", frame["corr_mean"].mean()),
                ("mean_mmc_mean", frame["mmc_mean"].mean()),
                ("mean_cwmm_mean", frame["cwmm_mean"].mean()),
            ],
            ("metric", "value"),
        ),
        "",
        "## Primary Candidate Signals",
        "",
        f"- best_single_run: `{best_run['run_id']}` / `{best_run['config']}`",
        f"- best_single_run_bmc_last_200_eras_mean: `{fmt(best_run['bmc_last_200_eras_mean'])}`",
        f"- best_target_family: `{best_target['target']}`",
        f"- best_target_family_mean_bmc_last_200_eras_mean: `{fmt(best_target['bmc_l200_mean'])}`",
        "- interpret best row separately from best candidate family and from champion status.",
        "",
        "## Top Individual Runs",
        "",
        frame_table(
            top_runs,
            [
                "run_id",
                "config",
                "target",
                "seed",
                "bmc_last_200_eras_mean",
                "bmc_mean",
                "bmc_sharpe",
                "corr_mean",
                "corr_sharpe",
                "fnc_mean",
                "mmc_mean",
                "cwmm_mean",
                "max_drawdown",
            ],
        ),
        "",
        "## Top Target Families",
        "",
        frame_table(top_targets, target_table_columns),
        "",
        "## Weakest Target Families",
        "",
        frame_table(bottom_targets, target_table_columns),
        "",
        "## Horizon Split",
        "",
        frame_table(horizon_agg, ["horizon", "runs", "bmc_l200_mean", "bmc_mean", "corr_mean", "mmc_mean"]),
        "",
        "## Seed Split",
        "",
        frame_table(seed_agg, ["seed", "runs", "bmc_l200_mean", "bmc_l200_std", "corr_mean", "mmc_mean"]),
        "",
        "## High-CORR / Weak-BMC Traps",
        "",
        frame_table(
            high_corr_traps.head(10),
            ["target", "runs", "bmc_l200_mean", "bmc_mean", "corr_mean", "mmc_mean", "cwmm_mean"],
        ),
        "",
        "## BMC / MMC Conflicts",
        "",
        frame_table(
            bmc_mmc_conflicts.head(12),
            ["target", "runs", "bmc_l200_mean", "bmc_mean", "mmc_mean", "corr_mean", "cwmm_mean"],
        ),
        "",
        "## Drawdown Warnings",
        "",
        frame_table(
            drawdown_warnings.head(12),
            ["run_id", "config", "target", "seed", "bmc_last_200_eras_mean", "bmc_mean", "corr_mean", "max_drawdown"],
        ),
        "",
        "## Report Writing Reminders",
        "",
        "- State hypothesis verdict explicitly.",
        "- Do not promote a single-run champion unless candidate-family evidence and risk checks support it.",
        "- Explain metric conflicts before recommending follow-up work.",
        "- Give pass criteria for the next experiment.",
    ]
    return "\n".join(lines) + "\n"


def build_ensemble_markdown(*, experiment_id: str, manifest: dict[str, Any], frame: pd.DataFrame) -> str:
    ranked = frame.sort_values(["bmc_last_200_eras_mean", "ensemble_id"], ascending=[False, True])
    top = ranked.head(12)
    best = ranked.iloc[0]
    practical = ranked[ranked["ensemble_id"] == "B31_jasper55_alpha05"]
    practical_candidate = practical.iloc[0] if not practical.empty else ranked.iloc[0]
    lines = [
        "# Experiment Evidence Brief",
        "",
        "## Metadata",
        f"- experiment_id: `{experiment_id}`",
        f"- name: `{manifest.get('name') or experiment_id}`",
        f"- manifest_status: `{manifest.get('status') or 'unknown'}`",
        "- evidence_mode: `ensemble_artifacts`",
        "",
        "## Matrix Totals",
        "",
        markdown_table(
            [
                ("ensemble_rows", len(frame)),
                ("completed_ensembles", int((frame["status"] == "completed").sum())),
                ("targets", int(frame["target"].nunique())),
                ("positive_bmc_last_200_ensembles", int((frame["bmc_last_200_eras_mean"] > 0).sum())),
                ("positive_bmc_mean_ensembles", int((frame["bmc_mean"] > 0).sum())),
                ("mean_bmc_last_200_eras_mean", frame["bmc_last_200_eras_mean"].mean()),
                ("median_bmc_last_200_eras_mean", frame["bmc_last_200_eras_mean"].median()),
                ("mean_corr_mean", frame["corr_mean"].mean()),
                ("mean_max_drawdown", frame["max_drawdown"].mean()),
            ],
            ("metric", "value"),
        ),
        "",
        "## Primary Candidate Signals",
        "",
        f"- max_bmc_ensemble: `{best['ensemble_id']}`",
        f"- max_bmc_ensemble_bmc_last_200_eras_mean: `{fmt(best['bmc_last_200_eras_mean'])}`",
        f"- practical_candidate: `{practical_candidate['ensemble_id']}`",
        f"- practical_candidate_bmc_last_200_eras_mean: `{fmt(practical_candidate['bmc_last_200_eras_mean'])}`",
        "- interpret ensemble candidates separately from champion status; "
        "no live/production evidence is included here.",
        "",
        "## Top Ensembles",
        "",
        frame_table(
            top,
            [
                "ensemble_id",
                "target",
                "source_runs",
                "bmc_last_200_eras_mean",
                "bmc_mean",
                "bmc_sharpe",
                "corr_mean",
                "corr_sharpe",
                "max_drawdown",
                "jasper60_weight",
                "ender60_weight",
                "ender20_weight",
                "alpha20_weight",
            ],
        ),
        "",
        "## Stability Warnings",
        "",
        frame_table(
            ranked.sort_values(["max_drawdown", "ensemble_id"], ascending=[False, True]).head(10),
            [
                "ensemble_id",
                "bmc_last_200_eras_mean",
                "bmc_mean",
                "corr_mean",
                "max_drawdown",
                "jasper60_weight",
                "alpha20_weight",
            ],
        ),
        "",
        "## Jasper Weight Readout",
        "",
        frame_table(
            ranked.sort_values(["jasper60_weight", "bmc_last_200_eras_mean"], ascending=[False, False]).head(15),
            [
                "ensemble_id",
                "jasper60_weight",
                "alpha20_weight",
                "bmc_last_200_eras_mean",
                "bmc_mean",
                "corr_mean",
                "max_drawdown",
            ],
        ),
        "",
        "## Report Writing Reminders",
        "",
        "- This is an ensemble-artifact closeout, not a manifest-run closeout.",
        "- State that high Jasper60 weight improved BMC but traded away CORR/drawdown.",
        "- Separate raw max-BMC winner from practical candidate.",
        "- Do not promote a champion without production/live validation.",
    ]
    return "\n".join(lines) + "\n"


def aggregate_targets(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("target", dropna=False)
        .agg(
            runs=("run_id", "count"),
            bmc_l200_mean=("bmc_last_200_eras_mean", "mean"),
            bmc_l200_std=("bmc_last_200_eras_mean", "std"),
            bmc_l200_max=("bmc_last_200_eras_mean", "max"),
            bmc_mean=("bmc_mean", "mean"),
            corr_mean=("corr_mean", "mean"),
            fnc_mean=("fnc_mean", "mean"),
            mmc_mean=("mmc_mean", "mean"),
            cwmm_mean=("cwmm_mean", "mean"),
            max_drawdown_mean=("max_drawdown", "mean"),
        )
        .reset_index()
    )


def aggregate_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("seed", dropna=False)
        .agg(
            runs=("run_id", "count"),
            bmc_l200_mean=("bmc_last_200_eras_mean", "mean"),
            bmc_l200_std=("bmc_last_200_eras_mean", "std"),
            corr_mean=("corr_mean", "mean"),
            mmc_mean=("mmc_mean", "mean"),
        )
        .reset_index()
        .sort_values("seed")
    )


def aggregate_horizons(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("horizon", dropna=False)
        .agg(
            runs=("run_id", "count"),
            bmc_l200_mean=("bmc_last_200_eras_mean", "mean"),
            bmc_mean=("bmc_mean", "mean"),
            corr_mean=("corr_mean", "mean"),
            mmc_mean=("mmc_mean", "mean"),
        )
        .reset_index()
        .sort_values("horizon")
    )


def read_summary_row(path: Path, *, run_id: str) -> dict[str, Any]:
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise EvidenceError(f"run_scoring_summary_invalid:{run_id}:{path.name}") from exc
    if frame.empty:
        raise EvidenceError(f"run_scoring_summary_empty:{run_id}:{path.name}")
    return frame.iloc[0].to_dict()


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


def resolve_target(config_name: str, run_manifest: dict[str, Any]) -> str:
    if config_name.startswith("r1_target_") and "_seed" in config_name:
        return config_name.removesuffix(".json").replace("r1_", "").rsplit("_seed", 1)[0]
    for section_name in ("data", "training", "config"):
        section = run_manifest.get(section_name)
        if isinstance(section, dict):
            for key in ("target_payout", "target_train", "target_col", "target"):
                value = section.get(key)
                if isinstance(value, str) and value:
                    return value
    return "-"


def resolve_horizon(target: str) -> str:
    match = re.search(r"_(20|60)$", target)
    return f"{match.group(1)}d" if match else "-"


def resolve_seed(config_name: str) -> int | None:
    match = re.search(r"seed(\d+)", config_name)
    return int(match.group(1)) if match else None


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
        raise EvidenceError(f"{error_code}:{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"{error_code}:{path}") from exc
    if not isinstance(payload, dict):
        raise EvidenceError(f"{error_code}:{path}")
    return payload


def nested_number(mapping: dict[str, Any], section: str, key: str) -> float | None:
    value = mapping.get(section)
    if not isinstance(value, dict):
        return None
    return number(value.get(key))


def number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        candidate = float(value)
        return candidate if math.isfinite(candidate) else None
    return None


def markdown_table(rows: list[tuple[str, Any]], headers: tuple[str, str]) -> str:
    lines = [
        f"| {headers[0]} | {headers[1]} |",
        "|---|---:|",
    ]
    for key, value in rows:
        lines.append(f"| {markdown_cell(key)} | {markdown_cell(fmt(value))} |")
    return "\n".join(lines)


def frame_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_None_"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame[columns].iterrows():
        values = [fmt_table_value(row[column], column) for column in columns]
        lines.append("| " + " | ".join(markdown_cell(value) for value in values) + " |")
    return "\n".join(lines)


def fmt_table_value(value: Any, column: str) -> str:
    if column in {"runs", "seed"} and value is not None and not pd.isna(value):
        number_value = float(value)
        if math.isfinite(number_value) and number_value.is_integer():
            return str(int(number_value))
    return fmt(value)


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.6f}"
    if pd.isna(value):
        return "n/a"
    return str(value)


def markdown_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
