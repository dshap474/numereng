"""Experiment-aware ensemble selection workflow."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from numereng.features.ensemble.contracts import (
    EnsembleSelectionRequest,
    EnsembleSelectionResult,
    EnsembleSelectionSelectionMode,
    EnsembleSelectionSourceRule,
    EnsembleSelectionVariantName,
)
from numereng.features.ensemble.metrics import correlation_matrix
from numereng.features.scoring._fastops import correlation_contribution_matrix, numerai_corr_matrix_vs_target
from numereng.features.scoring.metrics import (
    attach_benchmark_predictions,
    attach_scoring_targets,
    load_benchmark_predictions_from_path,
    load_fold_data_lazy,
    resolve_fnc_source_paths,
)
from numereng.features.store import resolve_store_root, resolve_workspace_layout_from_store_root
from numereng.features.training.client import TrainingDataClient, create_training_data_client
from numereng.features.training.repo import resolve_active_benchmark_predictions_path
from numereng.platform.parquet import write_parquet

_SAFE_ID = re.compile(r"^[\w\-.]+$")
_ACTIVE_BENCHMARK_ALIAS = "active_benchmark"
_THIRD_METRIC = "corr.mean"
_DEFAULT_SELECTION_ID = "default"
_WEIGHT_CHUNK_SIZE = 256


class EnsembleSelectionError(Exception):
    """Base error for ensemble-selection workflows."""


class EnsembleSelectionValidationError(EnsembleSelectionError):
    """Raised when ensemble-selection inputs are invalid."""


class EnsembleSelectionExecutionError(EnsembleSelectionError):
    """Raised when ensemble-selection execution fails."""


@dataclass(frozen=True)
class _RunRecord:
    run_id: str
    experiment_id: str
    feature_set: str
    target_col: str
    seed: int | None
    status: str
    data_version: str
    dataset_variant: str
    dataset_scope: str
    predictions_path: Path
    metrics_summary: dict[str, Any]


@dataclass(frozen=True)
class _FrozenCandidate:
    candidate_id: str
    feature_set: str
    target_col: str
    source_experiment_id: str
    run_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    freeze_rank: int
    freeze_metrics: dict[str, float | None]
    bundle_path: Path
    bundle_summary_path: Path
    bundle_row_count: int
    bundle_metrics: dict[str, Any]


@dataclass(frozen=True)
class _BlendMatrix:
    frame: pd.DataFrame
    candidate_ids: tuple[str, ...]
    target_col: str
    prediction_matrix: np.ndarray
    target_vector: np.ndarray
    benchmark_vector: np.ndarray
    era_ranges: tuple[tuple[str, int, int], ...]


def select_ensemble(
    *,
    store_root: str | Path = ".numereng",
    request: EnsembleSelectionRequest,
) -> EnsembleSelectionResult:
    """Run the ensemble-selection workflow and persist experiment-local artifacts."""

    _validate_request(request)

    resolved_store = resolve_store_root(store_root)
    layout = resolve_workspace_layout_from_store_root(resolved_store)
    experiment_root = (layout.experiments_root / request.experiment_id).resolve()
    if not (experiment_root / "experiment.json").is_file():
        raise EnsembleSelectionValidationError(f"ensemble_select_experiment_not_found:{request.experiment_id}")

    selection_id = request.selection_id or _DEFAULT_SELECTION_ID
    artifacts_root = (experiment_root / "ensemble_selection" / selection_id).resolve()
    created_at = _utc_now_iso()

    _ensure_dirs(
        artifacts_root,
        (
            "candidates",
            "candidates/predictions",
            "candidates/summaries",
            "correlation",
            "blends",
            "blends/predictions",
        ),
    )
    _write_status(
        artifacts_root,
        {
            "status": "running",
            "stage": "freeze_candidates",
            "selection_id": selection_id,
            "experiment_id": request.experiment_id,
            "created_at": created_at,
            "updated_at": created_at,
        },
    )

    client_holder: dict[str, TrainingDataClient | None] = {"client": None}

    def _client_factory() -> TrainingDataClient:
        if client_holder["client"] is None:
            client_holder["client"] = create_training_data_client()
        return client_holder["client"]

    frozen_candidates = _freeze_candidates(
        store_root=resolved_store,
        request=request,
        artifacts_root=artifacts_root,
        client_factory=_client_factory,
    )
    frozen_candidate_count = len(frozen_candidates)
    _write_status(
        artifacts_root,
        {
            "status": "running",
            "stage": "correlation_prune",
            "selection_id": selection_id,
            "experiment_id": request.experiment_id,
            "created_at": created_at,
            "updated_at": _utc_now_iso(),
            "frozen_candidate_count": frozen_candidate_count,
        },
    )

    blend_matrix = _build_ranked_candidate_matrix(frozen_candidates, target=request.target)
    correlation_frame, pruning_frame = _write_correlation_stage(
        request=request,
        frozen_candidates=frozen_candidates,
        blend_matrix=blend_matrix,
        artifacts_root=artifacts_root,
    )

    surviving_ids = pruning_frame[pruning_frame["status"] == "kept"]["candidate_id"].astype(str).tolist()
    surviving_candidate_count = len(surviving_ids)
    if surviving_candidate_count < 1:
        raise EnsembleSelectionExecutionError("ensemble_select_no_surviving_candidates")

    _write_status(
        artifacts_root,
        {
            "status": "running",
            "stage": "equal_weight",
            "selection_id": selection_id,
            "experiment_id": request.experiment_id,
            "created_at": created_at,
            "updated_at": _utc_now_iso(),
            "frozen_candidate_count": frozen_candidate_count,
            "surviving_candidate_count": surviving_candidate_count,
        },
    )

    equal_rows = _evaluate_equal_weight_variants(
        request=request,
        frozen_candidates=frozen_candidates,
        surviving_ids=surviving_ids,
        blend_matrix=blend_matrix,
        artifacts_root=artifacts_root,
    )
    scored_equal_rows = [row for row in equal_rows if row["status"] == "scored"]
    ranked_equal_rows = _sort_result_rows(scored_equal_rows, request=request)
    top_weighted_equal_rows = ranked_equal_rows[: request.top_weighted_variants]

    _write_status(
        artifacts_root,
        {
            "status": "running",
            "stage": "weighted_search",
            "selection_id": selection_id,
            "experiment_id": request.experiment_id,
            "created_at": created_at,
            "updated_at": _utc_now_iso(),
            "frozen_candidate_count": frozen_candidate_count,
            "surviving_candidate_count": surviving_candidate_count,
            "equal_weight_variant_count": len(scored_equal_rows),
            "top_weighted_variants": len(top_weighted_equal_rows),
        },
    )

    weighted_rows, promoted_rows = _evaluate_weighted_variants(
        request=request,
        top_equal_rows=top_weighted_equal_rows,
        blend_matrix=blend_matrix,
        artifacts_root=artifacts_root,
    )

    final_candidates = [*scored_equal_rows, *promoted_rows]
    final_winner = _sort_result_rows(final_candidates, request=request)[0]
    final_selection = {
        "winner": final_winner,
        "selection_id": selection_id,
        "experiment_id": request.experiment_id,
        "target": request.target,
        "primary_metric": request.primary_metric,
        "tie_break_metric": request.tie_break_metric,
        "artifacts_path": str(artifacts_root),
        "frozen_candidate_count": frozen_candidate_count,
        "surviving_candidate_count": surviving_candidate_count,
        "equal_weight_variant_count": len(scored_equal_rows),
        "weighted_candidate_count": len(weighted_rows),
        "created_at": created_at,
        "updated_at": _utc_now_iso(),
    }
    _write_json(artifacts_root / "blends" / "final_selection.json", final_selection)
    _write_status(
        artifacts_root,
        {
            "status": "completed",
            "stage": "completed",
            "selection_id": selection_id,
            "experiment_id": request.experiment_id,
            "created_at": created_at,
            "updated_at": final_selection["updated_at"],
            "winner_blend_id": final_winner["blend_id"],
            "winner_selection_mode": final_winner["selection_mode"],
        },
    )

    return EnsembleSelectionResult(
        selection_id=selection_id,
        experiment_id=request.experiment_id,
        target=request.target,
        primary_metric=request.primary_metric,
        tie_break_metric=request.tie_break_metric,
        status="completed",
        artifacts_path=artifacts_root,
        frozen_candidate_count=frozen_candidate_count,
        surviving_candidate_count=surviving_candidate_count,
        equal_weight_variant_count=len(scored_equal_rows),
        weighted_candidate_count=len(weighted_rows),
        winner_blend_id=str(final_winner["blend_id"]),
        winner_selection_mode=_coerce_selection_mode(final_winner["selection_mode"]),
        winner_component_ids=tuple(str(item) for item in final_winner["component_ids"]),
        winner_weights=tuple(float(item) for item in final_winner["weights"]),
        winner_metrics={
            request.primary_metric: final_winner["primary_metric_value"],
            request.tie_break_metric: final_winner["tie_break_metric_value"],
            _THIRD_METRIC: final_winner["third_metric_value"],
            "corr_mean": final_winner["corr_mean"],
            "bmc_mean": final_winner["bmc_mean"],
            "bmc_last_200_eras_mean": final_winner["bmc_last_200_eras_mean"],
        },
        created_at=created_at,
        updated_at=str(final_selection["updated_at"]),
    )


def _validate_request(request: EnsembleSelectionRequest) -> None:
    if not _SAFE_ID.match(request.experiment_id):
        raise EnsembleSelectionValidationError(f"ensemble_select_experiment_id_invalid:{request.experiment_id}")
    if request.selection_id is not None and not _SAFE_ID.match(request.selection_id):
        raise EnsembleSelectionValidationError(f"ensemble_select_selection_id_invalid:{request.selection_id}")
    if tuple(rule.experiment_id for rule in request.source_rules) != request.source_experiment_ids:
        raise EnsembleSelectionValidationError("ensemble_select_source_rule_order_mismatch")
    if len(request.source_experiment_ids) != len(set(request.source_experiment_ids)):
        raise EnsembleSelectionValidationError("ensemble_select_source_experiment_ids_duplicate")
    if request.primary_metric.count(".") != 1 or request.tie_break_metric.count(".") != 1:
        raise EnsembleSelectionValidationError("ensemble_select_metric_path_invalid")
    if request.correlation_threshold <= 0.0 or request.correlation_threshold > 1.0:
        raise EnsembleSelectionValidationError("ensemble_select_correlation_threshold_invalid")
    if request.top_weighted_variants < 1:
        raise EnsembleSelectionValidationError("ensemble_select_top_weighted_variants_invalid")
    if request.bundle_policy != "seed_avg":
        raise EnsembleSelectionValidationError("ensemble_select_bundle_policy_invalid")
    if request.required_seed_count < 1:
        raise EnsembleSelectionValidationError("ensemble_select_required_seed_count_invalid")
    if request.weighted_promotion_min_gain < 0.0:
        raise EnsembleSelectionValidationError("ensemble_select_weighted_promotion_min_gain_invalid")
    units = round(1.0 / request.weight_step)
    if request.weight_step <= 0.0 or not math.isclose(request.weight_step * units, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise EnsembleSelectionValidationError("ensemble_select_weight_step_invalid")
    for rule in request.source_rules:
        _validate_source_rule(rule)


def _validate_source_rule(rule: EnsembleSelectionSourceRule) -> None:
    if not _SAFE_ID.match(rule.experiment_id):
        raise EnsembleSelectionValidationError(f"ensemble_select_source_experiment_id_invalid:{rule.experiment_id}")
    if rule.selection_mode == "explicit_targets":
        if not rule.explicit_targets:
            raise EnsembleSelectionValidationError(f"ensemble_select_source_rule_targets_required:{rule.experiment_id}")
        if rule.top_n is not None:
            raise EnsembleSelectionValidationError(f"ensemble_select_source_rule_top_n_conflict:{rule.experiment_id}")
        return
    if rule.selection_mode == "top_n":
        if rule.top_n is None or rule.top_n < 1:
            raise EnsembleSelectionValidationError(f"ensemble_select_source_rule_top_n_invalid:{rule.experiment_id}")
        if rule.explicit_targets:
            raise EnsembleSelectionValidationError(
                f"ensemble_select_source_rule_explicit_targets_conflict:{rule.experiment_id}"
            )
        return
    raise EnsembleSelectionValidationError(f"ensemble_select_source_rule_mode_invalid:{rule.experiment_id}")


def _freeze_candidates(
    *,
    store_root: Path,
    request: EnsembleSelectionRequest,
    artifacts_root: Path,
    client_factory: Any,
) -> list[_FrozenCandidate]:
    candidates_dir = artifacts_root / "candidates"
    frozen_rows: list[dict[str, Any]] = []
    frozen_candidates: list[_FrozenCandidate] = []
    freeze_rank = 1

    for rule in request.source_rules:
        run_records = _load_source_run_records(store_root=store_root, experiment_id=rule.experiment_id)
        ranked_rows = _rank_source_targets(request=request, run_records=run_records)
        selected_rows = _select_source_rows(request=request, rule=rule, ranked_rows=ranked_rows)
        for selected in selected_rows:
            candidate = _materialize_candidate_bundle(
                store_root=store_root,
                request=request,
                selected_row=selected,
                freeze_rank=freeze_rank,
                candidates_dir=candidates_dir,
                client_factory=client_factory,
            )
            frozen_candidates.append(candidate)
            frozen_rows.append(_candidate_row(candidate))
            freeze_rank += 1

    if not frozen_candidates:
        raise EnsembleSelectionExecutionError("ensemble_select_no_candidates_frozen")

    frozen_frame = pd.DataFrame(frozen_rows)
    _write_csv(candidates_dir / "frozen_candidates.csv", frozen_frame)
    _write_json(candidates_dir / "frozen_candidates.json", frozen_rows)
    return frozen_candidates


def _load_source_run_records(*, store_root: Path, experiment_id: str) -> list[_RunRecord]:
    manifest_path = (
        resolve_workspace_layout_from_store_root(store_root).experiments_root / experiment_id / "experiment.json"
    )
    if not manifest_path.is_file():
        raise EnsembleSelectionValidationError(f"ensemble_select_source_experiment_not_found:{experiment_id}")
    payload = _read_json(manifest_path)
    run_ids = payload.get("runs")
    if not isinstance(run_ids, list):
        raise EnsembleSelectionExecutionError(f"ensemble_select_source_runs_missing:{experiment_id}")
    return [
        _load_run_record(
            store_root=store_root,
            run_id=str(run_id),
            experiment_id=experiment_id,
        )
        for run_id in run_ids
    ]


def _load_run_record(*, store_root: Path, run_id: str, experiment_id: str) -> _RunRecord:
    run_dir = store_root / "runs" / run_id
    manifest_path = run_dir / "run.json"
    if not manifest_path.is_file():
        raise EnsembleSelectionExecutionError(f"ensemble_select_run_manifest_missing:{run_id}")
    manifest = _read_json(manifest_path)
    predictions_rel = manifest.get("artifacts", {}).get("predictions")
    if not isinstance(predictions_rel, str):
        raise EnsembleSelectionExecutionError(f"ensemble_select_run_predictions_missing:{run_id}")
    predictions_path = (run_dir / predictions_rel).resolve()
    if not predictions_path.is_file():
        raise EnsembleSelectionExecutionError(f"ensemble_select_run_predictions_not_found:{run_id}")
    data = manifest.get("data", {})
    training = manifest.get("training", {})
    training_data = training.get("data", {}) if isinstance(training, dict) else {}
    metrics_summary = manifest.get("metrics_summary")
    resolved = _read_json(run_dir / "resolved.json") if (run_dir / "resolved.json").is_file() else {}
    model = resolved.get("model", {}) if isinstance(resolved, dict) else {}
    params = model.get("params", {}) if isinstance(model, dict) else {}
    seed = params.get("random_state")
    if not isinstance(seed, int):
        seed = _seed_from_config_path(manifest.get("config", {}).get("path"))
    feature_set = data.get("feature_set")
    target_col = data.get("target_col")
    data_version = data.get("version")
    if not isinstance(feature_set, str) or not isinstance(target_col, str) or not isinstance(data_version, str):
        raise EnsembleSelectionExecutionError(f"ensemble_select_run_data_missing:{run_id}")
    if not isinstance(metrics_summary, dict):
        metrics_summary = {}
    _require_active_benchmark_results(store_root=store_root, run_id=run_id)
    return _RunRecord(
        run_id=run_id,
        experiment_id=experiment_id,
        feature_set=feature_set,
        target_col=target_col,
        seed=seed,
        status=str(manifest.get("status") or ""),
        data_version=data_version,
        dataset_variant=str(training_data.get("dataset_variant") or "non_downsampled"),
        dataset_scope=str(training_data.get("dataset_scope") or "train_plus_validation"),
        predictions_path=predictions_path,
        metrics_summary=metrics_summary,
    )


def _require_active_benchmark_results(*, store_root: Path, run_id: str) -> None:
    results_path = store_root / "runs" / run_id / "results.json"
    if not results_path.is_file():
        raise EnsembleSelectionExecutionError(f"ensemble_select_results_missing:{run_id}")
    payload = _read_json(results_path)
    benchmark = payload.get("benchmark")
    if not isinstance(benchmark, dict):
        raise EnsembleSelectionExecutionError(f"ensemble_select_results_benchmark_missing:{run_id}")
    file_value = benchmark.get("file")
    mode_value = benchmark.get("mode")
    if mode_value == "active":
        return
    if isinstance(file_value, str) and "active_benchmark" in file_value:
        return
    raise EnsembleSelectionValidationError(f"ensemble_select_run_not_active_scored:{run_id}")


def _rank_source_targets(*, request: EnsembleSelectionRequest, run_records: list[_RunRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[_RunRecord]] = {}
    for record in run_records:
        if record.status != "FINISHED":
            continue
        grouped.setdefault(record.target_col, []).append(record)

    ranked_rows: list[dict[str, Any]] = []
    for target_col, records in grouped.items():
        ordered = sorted(records, key=lambda item: (item.seed is None, item.seed if item.seed is not None else -1))
        seeds = tuple(seed for seed in (item.seed for item in ordered) if seed is not None)
        distinct_seed_count = len(set(seeds))
        if request.require_full_seed_bundle:
            eligible_bundle = (
                len(ordered) == request.required_seed_count and distinct_seed_count == request.required_seed_count
            )
        else:
            if seeds:
                eligible_bundle = distinct_seed_count >= request.required_seed_count
            else:
                eligible_bundle = len(ordered) >= request.required_seed_count
        feature_set = ordered[0].feature_set
        ranked_rows.append(
            {
                "feature_set": feature_set,
                "target_col": target_col,
                "experiment_id": ordered[0].experiment_id,
                "run_ids": tuple(item.run_id for item in ordered),
                "seeds": seeds,
                "eligible_bundle": eligible_bundle,
                "bundle_size": len(ordered),
                "distinct_seed_count": distinct_seed_count,
                "primary_metric_value": _mean_metric_path(ordered, request.primary_metric),
                "tie_break_metric_value": _mean_metric_path(ordered, request.tie_break_metric),
                "third_metric_value": _mean_metric_path(ordered, _THIRD_METRIC),
            }
        )

    ranked_rows.sort(
        key=lambda row: (
            float(row["primary_metric_value"] if row["primary_metric_value"] is not None else float("-inf")),
            float(row["tie_break_metric_value"] if row["tie_break_metric_value"] is not None else float("-inf")),
            float(row["third_metric_value"] if row["third_metric_value"] is not None else float("-inf")),
            row["target_col"],
        ),
        reverse=True,
    )
    return ranked_rows


def _select_source_rows(
    *,
    request: EnsembleSelectionRequest,
    rule: EnsembleSelectionSourceRule,
    ranked_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    eligible_rows = [row for row in ranked_rows if (row["eligible_bundle"] or not request.require_full_seed_bundle)]
    if rule.selection_mode == "top_n":
        assert rule.top_n is not None  # validated
        selected = eligible_rows[: rule.top_n]
        if len(selected) < rule.top_n:
            raise EnsembleSelectionExecutionError(
                f"ensemble_select_source_top_n_insufficient:{rule.experiment_id}:{len(selected)}"
            )
        return selected

    wanted = list(rule.explicit_targets)
    selected: list[dict[str, Any]] = []
    by_target = {str(row["target_col"]): row for row in eligible_rows}
    for target_col in wanted:
        row = by_target.get(target_col)
        if row is None:
            raise EnsembleSelectionExecutionError(
                f"ensemble_select_source_target_missing_or_ineligible:{rule.experiment_id}:{target_col}"
            )
        selected.append(row)
    return selected


def _materialize_candidate_bundle(
    *,
    store_root: Path,
    request: EnsembleSelectionRequest,
    selected_row: dict[str, Any],
    freeze_rank: int,
    candidates_dir: Path,
    client_factory: Any,
) -> _FrozenCandidate:
    feature_set = str(selected_row["feature_set"])
    target_col = str(selected_row["target_col"])
    run_ids = tuple(str(item) for item in selected_row["run_ids"])
    records = [
        _load_run_record(
            store_root=store_root,
            run_id=run_id,
            experiment_id=str(selected_row["experiment_id"]),
        )
        for run_id in run_ids
    ]
    candidate_id = _candidate_id(feature_set=feature_set, target_col=target_col)
    bundle_path = candidates_dir / "predictions" / f"{candidate_id}.parquet"
    bundle_summary_path = candidates_dir / "summaries" / f"{candidate_id}.json"
    bundle_frame = _build_bundle_frame(
        store_root=store_root,
        records=records,
        target=request.target,
        client_factory=client_factory,
    )
    write_parquet(bundle_frame, bundle_path, index=False)
    bundle_metrics = _score_prediction_frame(bundle_frame, target=request.target)
    _write_json(bundle_summary_path, bundle_metrics)
    return _FrozenCandidate(
        candidate_id=_candidate_id(feature_set=feature_set, target_col=target_col),
        feature_set=feature_set,
        target_col=target_col,
        source_experiment_id=str(selected_row["experiment_id"]),
        run_ids=run_ids,
        seeds=tuple(int(seed) for seed in selected_row["seeds"]),
        freeze_rank=freeze_rank,
        freeze_metrics={
            "primary_metric_value": selected_row["primary_metric_value"],
            "tie_break_metric_value": selected_row["tie_break_metric_value"],
            "third_metric_value": selected_row["third_metric_value"],
        },
        bundle_path=bundle_path,
        bundle_summary_path=bundle_summary_path,
        bundle_row_count=int(len(bundle_frame)),
        bundle_metrics=bundle_metrics,
    )


def _build_bundle_frame(
    *,
    store_root: Path,
    records: list[_RunRecord],
    target: str,
    client_factory: Any,
) -> pd.DataFrame:
    base_record = records[0]
    frames: list[pd.DataFrame] = []
    for index, record in enumerate(records, start=1):
        frame = pd.read_parquet(record.predictions_path)
        required_cols = {"id", "era", "prediction", record.target_col}
        missing = required_cols.difference(frame.columns)
        if missing:
            raise EnsembleSelectionExecutionError(
                f"ensemble_select_bundle_columns_missing:{record.run_id}:{sorted(missing)}"
            )
        keep_cols = ["id", "era", record.target_col]
        if target != record.target_col and target in frame.columns:
            keep_cols.append(target)
        subset = frame[[*keep_cols, "prediction"]].copy()
        subset["id"] = subset["id"].astype(str)
        subset["era"] = subset["era"].astype(str)
        subset = subset.rename(columns={"prediction": f"prediction_seed_{index}"})
        frames.append(subset)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame[["id", "era", frame.columns[-1]]], on=["id", "era"], how="inner")
    if merged.empty:
        raise EnsembleSelectionExecutionError(f"ensemble_select_bundle_alignment_empty:{base_record.target_col}")

    prediction_cols = [col for col in merged.columns if col.startswith("prediction_seed_")]
    merged["prediction"] = merged[prediction_cols].mean(axis=1)
    selected_cols = ["id", "era", base_record.target_col]
    if target != base_record.target_col and target in merged.columns:
        selected_cols.append(target)
    merged = merged[[*selected_cols, "prediction"]].copy()
    merged = _attach_missing_targets(
        store_root=store_root,
        prediction_frame=merged,
        record=base_record,
        target_cols=tuple(col for col in (target,) if col != base_record.target_col),
        client_factory=client_factory,
    )
    merged = _attach_active_benchmark(store_root=store_root, prediction_frame=merged)
    return merged.sort_values(["era", "id"]).reset_index(drop=True)


def _attach_missing_targets(
    *,
    store_root: Path,
    prediction_frame: pd.DataFrame,
    record: _RunRecord,
    target_cols: tuple[str, ...],
    client_factory: Any,
) -> pd.DataFrame:
    missing = [target_col for target_col in target_cols if target_col not in prediction_frame.columns]
    if not missing:
        return prediction_frame
    client = client_factory()
    source_paths, include_validation_only = resolve_fnc_source_paths(
        client=client,
        data_version=record.data_version,
        dataset_variant=record.dataset_variant,
        feature_source_paths=None,
        dataset_scope=record.dataset_scope,
        data_root=(store_root / "datasets").resolve(),
    )
    target_frame = load_fold_data_lazy(
        source_paths,
        eras=sorted({str(value) for value in prediction_frame["era"].tolist()}),
        columns=["era", "id", *missing],
        era_col="era",
        id_col="id",
        include_validation_only=include_validation_only,
    )
    return attach_scoring_targets(
        prediction_frame,
        target_frame,
        target_cols=missing,
        era_col="era",
        id_col="id",
    )


def _attach_active_benchmark(*, store_root: Path, prediction_frame: pd.DataFrame) -> pd.DataFrame:
    benchmark_frame, benchmark_col = load_benchmark_predictions_from_path(
        resolve_active_benchmark_predictions_path(data_root=(store_root / "datasets").resolve()),
        benchmark_model="prediction",
        benchmark_name=_ACTIVE_BENCHMARK_ALIAS,
        prediction_cols=["prediction"],
        era_col="era",
        id_col="id",
        data_root=(store_root / "datasets").resolve(),
    )
    attached = attach_benchmark_predictions(
        prediction_frame,
        benchmark_frame,
        benchmark_col,
        era_col="era",
        id_col="id",
        min_overlap_ratio=0.0,
    )
    return attached.rename(columns={benchmark_col: _ACTIVE_BENCHMARK_ALIAS})


def _score_prediction_frame(frame: pd.DataFrame, *, target: str) -> dict[str, Any]:
    working = frame[["era", "id", "prediction", target, _ACTIVE_BENCHMARK_ALIAS]].copy()
    corr_scores = _score_weight_matrix(
        prediction_matrix=working[["prediction"]].to_numpy(dtype=np.float64, copy=False),
        target_vector=working[target].to_numpy(dtype=np.float64, copy=False),
        benchmark_vector=working[_ACTIVE_BENCHMARK_ALIAS].to_numpy(dtype=np.float64, copy=False),
        era_ranges=_era_ranges(working["era"].astype(str).tolist()),
        weight_matrix=np.asarray([[1.0]], dtype=np.float64),
    )
    return _metrics_payload_from_summary(corr_scores, index=0)


def _build_ranked_candidate_matrix(
    candidates: list[_FrozenCandidate],
    *,
    target: str = "target_ender_20",
) -> _BlendMatrix:
    merged: pd.DataFrame | None = None
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    for candidate in candidates:
        frame = pd.read_parquet(candidate.bundle_path)
        required = {"era", "id", "prediction", _ACTIVE_BENCHMARK_ALIAS}
        missing = required.difference(frame.columns)
        if missing:
            raise EnsembleSelectionExecutionError(
                f"ensemble_select_candidate_bundle_missing_columns:{candidate.candidate_id}:{sorted(missing)}"
            )
        ranked = frame.copy()
        ranked["prediction"] = ranked.groupby("era")["prediction"].rank(method="average", pct=True)
        subset = ranked[["era", "id", _ACTIVE_BENCHMARK_ALIAS, "prediction"]].copy()
        if target not in ranked.columns:
            raise EnsembleSelectionExecutionError(
                f"ensemble_select_candidate_bundle_missing_target:{candidate.candidate_id}:{target}"
            )
        subset[target] = ranked[target].to_numpy()
        subset = subset.rename(columns={"prediction": candidate.candidate_id})
        if merged is None:
            merged = subset
            continue
        merged = merged.merge(subset[["era", "id", candidate.candidate_id]], on=["era", "id"], how="inner")
    if merged is None or merged.empty:
        raise EnsembleSelectionExecutionError("ensemble_select_ranked_component_matrix_empty")
    merged = merged.sort_values(["era", "id"]).reset_index(drop=True)
    return _BlendMatrix(
        frame=merged,
        candidate_ids=tuple(candidate_ids),
        target_col=target,
        prediction_matrix=merged[candidate_ids].to_numpy(dtype=np.float64, copy=False),
        target_vector=merged[target].to_numpy(dtype=np.float64, copy=False),
        benchmark_vector=merged[_ACTIVE_BENCHMARK_ALIAS].to_numpy(dtype=np.float64, copy=False),
        era_ranges=_era_ranges(merged["era"].astype(str).tolist()),
    )


def _write_correlation_stage(
    *,
    request: EnsembleSelectionRequest,
    frozen_candidates: list[_FrozenCandidate],
    blend_matrix: _BlendMatrix,
    artifacts_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr = correlation_matrix(
        ranked_predictions=pd.DataFrame(blend_matrix.prediction_matrix, columns=blend_matrix.candidate_ids),
        run_ids=blend_matrix.candidate_ids,
    )
    corr_reset = corr.reset_index(names=["candidate_id"])
    candidates_frame = pd.DataFrame([_candidate_row(candidate) for candidate in frozen_candidates])
    candidate_metric_order = (
        candidates_frame.sort_values(
            by=["primary_metric_value", "tie_break_metric_value", "third_metric_value", "freeze_rank"],
            ascending=[False, False, False, True],
        )["candidate_id"]
        .astype(str)
        .tolist()
    )
    pruning_rows: list[dict[str, Any]] = []
    kept_ids: list[str] = []
    for candidate_id in candidate_metric_order:
        max_corr = None
        pruned_by = None
        for keeper in kept_ids:
            value = float(corr.loc[candidate_id, keeper])
            if max_corr is None or value > max_corr:
                max_corr = value
                pruned_by = keeper
        keep = max_corr is None or max_corr <= request.correlation_threshold
        if keep:
            kept_ids.append(candidate_id)
        pruning_rows.append(
            {
                "candidate_id": candidate_id,
                "status": "kept" if keep else "pruned",
                "pruned_by": pruned_by,
                "max_corr_with_kept": max_corr,
                "threshold": request.correlation_threshold,
            }
        )
    pruning_frame = pd.DataFrame(pruning_rows).merge(
        candidates_frame[
            [
                "candidate_id",
                "feature_set",
                "target_col",
                "freeze_rank",
                "primary_metric_value",
                "tie_break_metric_value",
                "third_metric_value",
            ]
        ],
        on="candidate_id",
        how="left",
    )
    correlation_dir = artifacts_root / "correlation"
    write_parquet(corr_reset, correlation_dir / "correlation_matrix.parquet", index=False)
    _write_csv(correlation_dir / "correlation_matrix.csv", corr_reset)
    _write_json(correlation_dir / "correlation_matrix.json", corr_reset.to_dict(orient="records"))
    _write_csv(correlation_dir / "pruning_recommendation.csv", pruning_frame)
    _write_json(correlation_dir / "pruning_recommendation.json", pruning_frame.to_dict(orient="records"))
    return corr_reset, pruning_frame


def _evaluate_equal_weight_variants(
    *,
    request: EnsembleSelectionRequest,
    frozen_candidates: list[_FrozenCandidate],
    surviving_ids: list[str],
    blend_matrix: _BlendMatrix,
    artifacts_root: Path,
) -> list[dict[str, Any]]:
    candidates_frame = pd.DataFrame([_candidate_row(candidate) for candidate in frozen_candidates])
    surviving_frame = candidates_frame[candidates_frame["candidate_id"].isin(surviving_ids)].copy()
    top_overall = (
        surviving_frame.sort_values(
            by=["primary_metric_value", "tie_break_metric_value", "third_metric_value", "freeze_rank"],
            ascending=[False, False, False, True],
        )["candidate_id"]
        .astype(str)
        .tolist()
    )
    medium_ids = surviving_frame[surviving_frame["feature_set"] == "medium"]["candidate_id"].astype(str).tolist()
    small_ids = surviving_frame[surviving_frame["feature_set"] == "small"]["candidate_id"].astype(str).tolist()

    rows: list[dict[str, Any]] = []
    for variant_name in request.blend_variants:
        component_ids = _resolve_variant_components(
            variant_name=variant_name,
            top_overall=top_overall,
            medium_ids=medium_ids,
            small_ids=small_ids,
        )
        if not component_ids:
            rows.append(
                {
                    "blend_id": str(variant_name),
                    "source_variant_id": str(variant_name),
                    "selection_mode": "equal_weight",
                    "status": "skipped",
                    "component_ids": [],
                    "weights": [],
                    "predictions_path": None,
                    "primary_metric_value": None,
                    "tie_break_metric_value": None,
                    "third_metric_value": None,
                }
            )
            continue
        weights = tuple([1.0 / len(component_ids)] * len(component_ids))
        metrics = _score_variant(
            blend_matrix=blend_matrix,
            component_ids=component_ids,
            weight_vectors=np.asarray([weights], dtype=np.float64),
        )
        predictions_path = artifacts_root / "blends" / "predictions" / f"{variant_name}__equal_weight.parquet"
        _write_variant_prediction(
            path=predictions_path,
            blend_matrix=blend_matrix,
            component_ids=component_ids,
            weights=weights,
        )
        rows.append(
            _result_row(
                blend_id=str(variant_name),
                source_variant_id=str(variant_name),
                selection_mode="equal_weight",
                component_ids=component_ids,
                weights=weights,
                predictions_path=predictions_path,
                metrics=metrics,
            )
        )

    equal_frame = pd.DataFrame(rows)
    blends_dir = artifacts_root / "blends"
    _write_csv(blends_dir / "equal_weight_results.csv", equal_frame)
    _write_json(blends_dir / "equal_weight_results.json", equal_frame.to_dict(orient="records"))
    return rows


def _evaluate_weighted_variants(
    *,
    request: EnsembleSelectionRequest,
    top_equal_rows: list[dict[str, Any]],
    blend_matrix: _BlendMatrix,
    artifacts_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    weighted_rows: list[dict[str, Any]] = []
    promoted_rows: list[dict[str, Any]] = []
    blends_dir = artifacts_root / "blends"

    for equal_row in top_equal_rows:
        component_ids = tuple(str(item) for item in equal_row["component_ids"])
        if not component_ids:
            continue
        weight_matrix = np.asarray(_weight_simplex(len(component_ids), request.weight_step), dtype=np.float64)
        metrics_list = _score_variant(
            blend_matrix=blend_matrix,
            component_ids=component_ids,
            weight_vectors=weight_matrix,
            return_all=True,
        )
        source_variant_id = str(equal_row["source_variant_id"])
        branch_rows = [
            _result_row(
                blend_id=f"{source_variant_id}__weighted",
                source_variant_id=source_variant_id,
                selection_mode="weighted",
                component_ids=component_ids,
                weights=tuple(float(value) for value in weight_matrix[index].tolist()),
                predictions_path=None,
                metrics=metrics,
            )
            for index, metrics in enumerate(metrics_list)
        ]
        weighted_rows.extend(branch_rows)
        weighted_frame = pd.DataFrame(weighted_rows)
        _write_csv(blends_dir / "weighted_candidates.csv", weighted_frame)
        _write_json(blends_dir / "weighted_candidates.json", weighted_frame.to_dict(orient="records"))

        best_weighted = _sort_result_rows(branch_rows, request=request)[0]
        best_weighted_path = blends_dir / "predictions" / f"{source_variant_id}__weighted_best.parquet"
        _write_variant_prediction(
            path=best_weighted_path,
            blend_matrix=blend_matrix,
            component_ids=component_ids,
            weights=tuple(float(item) for item in best_weighted["weights"]),
        )
        best_weighted["predictions_path"] = str(best_weighted_path)

        improvement = float(best_weighted["primary_metric_value"]) - float(equal_row["primary_metric_value"])
        if improvement >= request.weighted_promotion_min_gain:
            promoted_rows.append(
                {
                    **best_weighted,
                    "promoted_over_blend_id": str(equal_row["blend_id"]),
                    "primary_improvement_vs_equal": improvement,
                }
            )
        else:
            promoted_rows.append(
                {
                    **equal_row,
                    "selection_mode": "equal_weight_retained",
                    "promoted_over_blend_id": str(equal_row["blend_id"]),
                    "primary_improvement_vs_equal": improvement,
                }
            )

        promoted_frame = pd.DataFrame(promoted_rows)
        _write_csv(blends_dir / "weighted_promotions.csv", promoted_frame)
        _write_json(blends_dir / "weighted_promotions.json", promoted_frame.to_dict(orient="records"))

    return weighted_rows, promoted_rows


def _resolve_variant_components(
    *,
    variant_name: EnsembleSelectionVariantName,
    top_overall: list[str],
    medium_ids: list[str],
    small_ids: list[str],
) -> tuple[str, ...]:
    if variant_name == "all_surviving":
        return tuple(top_overall)
    if variant_name == "medium_only":
        return tuple(medium_ids)
    if variant_name == "small_only":
        return tuple(small_ids)
    if variant_name == "top2_medium_top2_small":
        return tuple([*medium_ids[:2], *small_ids[:2]])
    if variant_name == "top3_overall":
        return tuple(top_overall[:3])
    raise EnsembleSelectionValidationError(f"ensemble_select_variant_invalid:{variant_name}")


def _score_variant(
    *,
    blend_matrix: _BlendMatrix,
    component_ids: tuple[str, ...],
    weight_vectors: np.ndarray,
    return_all: bool = False,
) -> dict[str, Any] | list[dict[str, Any]]:
    component_indices = [blend_matrix.candidate_ids.index(component_id) for component_id in component_ids]
    predictions = blend_matrix.prediction_matrix[:, component_indices]
    summaries = _score_weight_matrix(
        prediction_matrix=predictions,
        target_vector=blend_matrix.target_vector,
        benchmark_vector=blend_matrix.benchmark_vector,
        era_ranges=blend_matrix.era_ranges,
        weight_matrix=weight_vectors,
    )
    if return_all:
        return [_metrics_payload_from_summary(summaries, index=index) for index in range(weight_vectors.shape[0])]
    return _metrics_payload_from_summary(summaries, index=0)


def _score_weight_matrix(
    *,
    prediction_matrix: np.ndarray,
    target_vector: np.ndarray,
    benchmark_vector: np.ndarray,
    era_ranges: tuple[tuple[str, int, int], ...],
    weight_matrix: np.ndarray,
) -> dict[str, np.ndarray]:
    n_eras = len(era_ranges)
    n_variants = weight_matrix.shape[0]
    corr_scores = np.full((n_eras, n_variants), np.nan, dtype=np.float64)
    bmc_scores = np.full((n_eras, n_variants), np.nan, dtype=np.float64)

    for era_index, (_era, start, end) in enumerate(era_ranges):
        pred_slice = prediction_matrix[start:end]
        target_slice = target_vector[start:end]
        benchmark_slice = benchmark_vector[start:end]
        for offset in range(0, n_variants, _WEIGHT_CHUNK_SIZE):
            chunk = weight_matrix[offset : offset + _WEIGHT_CHUNK_SIZE]
            blended = pred_slice @ chunk.T
            corr_scores[era_index, offset : offset + len(chunk)] = numerai_corr_matrix_vs_target(blended, target_slice)
            bmc_scores[era_index, offset : offset + len(chunk)] = correlation_contribution_matrix(
                blended,
                benchmark_slice,
                target_slice,
            )

    recent_window = bmc_scores[-min(200, n_eras) :, :]
    corr_summary = _summary_columns(corr_scores)
    bmc_summary = _summary_columns(bmc_scores)
    recent_summary = _summary_columns(recent_window)
    return {
        "corr_mean": corr_summary["mean"],
        "corr_std": corr_summary["std"],
        "corr_sharpe": corr_summary["sharpe"],
        "corr_max_drawdown": corr_summary["max_drawdown"],
        "bmc_mean": bmc_summary["mean"],
        "bmc_std": bmc_summary["std"],
        "bmc_sharpe": bmc_summary["sharpe"],
        "bmc_max_drawdown": bmc_summary["max_drawdown"],
        "bmc_last_200_eras_mean": recent_summary["mean"],
        "bmc_last_200_eras_std": recent_summary["std"],
        "bmc_last_200_eras_sharpe": recent_summary["sharpe"],
        "bmc_last_200_eras_max_drawdown": recent_summary["max_drawdown"],
    }


def _metrics_payload_from_summary(summary: dict[str, np.ndarray], *, index: int) -> dict[str, Any]:
    return {key: float(value[index]) if np.isfinite(value[index]) else None for key, value in summary.items()}


def _summary_columns(values: np.ndarray) -> dict[str, np.ndarray]:
    mean = np.full(values.shape[1], np.nan, dtype=np.float64)
    std = np.full(values.shape[1], np.nan, dtype=np.float64)
    for column_index in range(values.shape[1]):
        finite = values[np.isfinite(values[:, column_index]), column_index]
        if finite.size == 0:
            continue
        mean[column_index] = float(np.mean(finite))
        std[column_index] = float(np.std(finite, ddof=0))
    sharpe = np.divide(mean, std, out=np.full_like(mean, np.nan), where=std != 0.0)
    return {
        "mean": mean,
        "std": std,
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown_per_column(values),
    }


def _max_drawdown_per_column(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[1], np.nan, dtype=np.float64)
    for column_index in range(values.shape[1]):
        series = values[:, column_index]
        running = 0.0
        peak = 0.0
        worst = 0.0
        for value in series:
            if not np.isfinite(value):
                continue
            running += float(value)
            if running > peak:
                peak = running
            drawdown = peak - running
            if drawdown > worst:
                worst = drawdown
        out[column_index] = worst
    return out


def _weight_simplex(n_components: int, step: float) -> list[list[float]]:
    units = round(1.0 / step)
    if n_components == 1:
        return [[1.0]]
    rows: list[list[float]] = []

    def _walk(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            rows.append([(value / units) for value in (*prefix, remaining)])
            return
        for value in range(remaining + 1):
            _walk([*prefix, value], remaining - value, slots - 1)

    _walk([], units, n_components)
    return rows


def _write_variant_prediction(
    *,
    path: Path,
    blend_matrix: _BlendMatrix,
    component_ids: tuple[str, ...],
    weights: tuple[float, ...],
) -> None:
    component_indices = [blend_matrix.candidate_ids.index(component_id) for component_id in component_ids]
    predictions = blend_matrix.prediction_matrix[:, component_indices] @ np.asarray(weights, dtype=np.float64)
    output = blend_matrix.frame[["era", "id", "target_ender_20", _ACTIVE_BENCHMARK_ALIAS]].copy()
    if blend_matrix.target_col != "target_ender_20":
        output = blend_matrix.frame[["era", "id", blend_matrix.target_col, _ACTIVE_BENCHMARK_ALIAS]].copy()
    output["prediction"] = predictions
    write_parquet(output, path, index=False)


def _result_row(
    *,
    blend_id: str,
    source_variant_id: str,
    selection_mode: EnsembleSelectionSelectionMode,
    component_ids: tuple[str, ...],
    weights: tuple[float, ...],
    predictions_path: Path | None,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "blend_id": blend_id,
        "source_variant_id": source_variant_id,
        "selection_mode": selection_mode,
        "status": "scored",
        "component_ids": list(component_ids),
        "weights": [float(value) for value in weights],
        "component_count": len(component_ids),
        "predictions_path": None if predictions_path is None else str(predictions_path),
        "primary_metric_value": metrics.get("bmc_last_200_eras_mean"),
        "tie_break_metric_value": metrics.get("bmc_mean"),
        "third_metric_value": metrics.get("corr_mean"),
        **metrics,
    }


def _sort_result_rows(rows: list[dict[str, Any]], *, request: EnsembleSelectionRequest) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row.get("primary_metric_value") if row.get("primary_metric_value") is not None else float("-inf")),
            float(
                row.get("tie_break_metric_value") if row.get("tie_break_metric_value") is not None else float("-inf")
            ),
            float(row.get("third_metric_value") if row.get("third_metric_value") is not None else float("-inf")),
            -int(row.get("component_count") or 0),
            str(row.get("blend_id") or ""),
        ),
        reverse=True,
    )


def _candidate_row(candidate: _FrozenCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "feature_set": candidate.feature_set,
        "target_col": candidate.target_col,
        "source_experiment_id": candidate.source_experiment_id,
        "run_ids": list(candidate.run_ids),
        "seeds": list(candidate.seeds),
        "freeze_rank": candidate.freeze_rank,
        "bundle_path": str(candidate.bundle_path),
        "bundle_summary_path": str(candidate.bundle_summary_path),
        "bundle_row_count": candidate.bundle_row_count,
        "primary_metric_value": candidate.freeze_metrics["primary_metric_value"],
        "tie_break_metric_value": candidate.freeze_metrics["tie_break_metric_value"],
        "third_metric_value": candidate.freeze_metrics["third_metric_value"],
        "bundle_metrics": candidate.bundle_metrics,
    }


def _candidate_id(*, feature_set: str, target_col: str) -> str:
    return f"{feature_set}_{target_col}"


def _mean_metric_path(records: list[_RunRecord], metric_path: str) -> float | None:
    values = [_metric_path_value(record.metrics_summary, metric_path) for record in records]
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def _metric_path_value(payload: dict[str, Any], metric_path: str) -> float | None:
    current: Any = payload
    for part in metric_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if current is None:
        return None
    try:
        value = float(current)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _seed_from_config_path(raw_value: Any) -> int | None:
    if not isinstance(raw_value, str):
        return None
    match = re.search(r"_seed(\d+)\.json$", raw_value)
    if match is None:
        return None
    return int(match.group(1))


def _era_ranges(eras: list[str]) -> tuple[tuple[str, int, int], ...]:
    if not eras:
        return ()
    rows: list[tuple[str, int, int]] = []
    start = 0
    current = eras[0]
    for index, era in enumerate(eras[1:], start=1):
        if era == current:
            continue
        rows.append((current, start, index))
        current = era
        start = index
    rows.append((current, start, len(eras)))
    return tuple(rows)


def _coerce_selection_mode(value: object) -> EnsembleSelectionSelectionMode:
    if value in {"equal_weight", "weighted", "equal_weight_retained"}:
        return value
    raise EnsembleSelectionExecutionError(f"ensemble_select_selection_mode_invalid:{value}")


def _ensure_dirs(root: Path, dirs: tuple[str, ...]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for rel in dirs:
        (root / rel).mkdir(parents=True, exist_ok=True)


def _write_status(root: Path, payload: dict[str, Any]) -> None:
    _write_json(root / "status.json", payload)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EnsembleSelectionExecutionError(f"ensemble_select_json_object_expected:{path}")
    return payload


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "EnsembleSelectionError",
    "EnsembleSelectionExecutionError",
    "EnsembleSelectionValidationError",
    "select_ensemble",
]
