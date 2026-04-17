"""Package-native scoring and diagnostics sync for submission packages."""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

from numereng.features.scoring.metrics import (
    DEFAULT_PAYOUT_TARGET_COL,
    build_cumulative_scores,
    per_era_reference_corr,
    score_prediction_file_with_details,
    summarize_scores,
)
from numereng.features.serving.contracts import (
    PackageDiagnosticsSyncResult,
    PackageEvaluationDataset,
    PackageScoreResult,
    PackageScoreRuntime,
    PackageScoreStage,
    SubmissionPackageRecord,
)
from numereng.features.serving.evaluation_io import (
    ensure_validation_dataset_path,
    materialize_pickle_validation_predictions,
)
from numereng.features.serving.repo import (
    ServingValidationError,
    load_package,
    save_package,
    utc_now_iso,
)
from numereng.features.serving.runtime import (
    ServingRuntimeError,
    ServingUnsupportedConfigError,
    blend_component_predictions,
)
from numereng.features.serving.service import (
    _fit_and_predict_package,
    build_submission_pickle,
    create_serving_client,
    inspect_package,
)
from numereng.features.store import resolve_workspace_layout
from numereng.features.training.client import create_training_data_client
from numereng.features.training.errors import TrainingDataError
from numereng.features.training.repo import (
    resolve_data_version_root,
    resolve_variant_dataset_filename,
)
from numereng.platform.parquet import write_parquet

_PACKAGE_SCORE_TARGETS = ("target", DEFAULT_PAYOUT_TARGET_COL, "target_cyrusd_20")
_PACKAGE_SCORE_PRED_COL = "prediction"
_DIAGNOSTICS_POLL_INTERVAL_SECONDS = 30
_DIAGNOSTICS_POLL_TIMEOUT_SECONDS = 30 * 60
_SUCCESS_STATUSES = {"success", "succeeded", "complete", "completed", "done", "ok"}
_FAILURE_STATUSES = {"failed", "failure", "error", "errored", "rejected", "cancelled", "canceled"}
_PENDING_STATUSES = {"pending", "queued", "running", "triggered", "validating", "diagnosing", "processing"}


def score_submission_package(
    *,
    workspace_root: str | Path,
    experiment_id: str,
    package_id: str,
    dataset: PackageEvaluationDataset = "validation",
    runtime: PackageScoreRuntime = "auto",
    stage: PackageScoreStage = "post_training_full",
    client: Any | None = None,
) -> PackageScoreResult:
    """Score one final submission package artifact on local validation data."""

    if dataset != "validation":
        raise ServingValidationError("serving_package_score_dataset_not_supported")

    package = load_package(workspace_root=workspace_root, experiment_id=experiment_id, package_id=package_id)
    original_status = package.status
    runtime_client = create_serving_client() if client is None else client
    layout = resolve_workspace_layout(workspace_root)
    data_root = (layout.store_root / "datasets").resolve()
    inspection = inspect_package(
        workspace_root=workspace_root,
        experiment_id=experiment_id,
        package_id=package_id,
    )
    validation_frame: pd.DataFrame | None = None
    runtime_used: Literal["pickle", "local"]
    package_after_runtime: SubmissionPackageRecord
    predictions: pd.DataFrame

    if runtime == "pickle" or (runtime == "auto" and inspection.model_upload_compatible):
        try:
            validation_path = ensure_validation_dataset_path(
                client=runtime_client,
                data_root=data_root,
                data_version=package.data_version,
            )
            package_after_runtime, predictions = materialize_pickle_validation_predictions(
                workspace_root=workspace_root,
                experiment_id=experiment_id,
                package_id=package_id,
                package=inspection.package,
                validation_path=validation_path,
                client=runtime_client,
                scoring_target_cols=_PACKAGE_SCORE_TARGETS,
            )
            runtime_used = "pickle"
        except (ServingUnsupportedConfigError, ServingValidationError, ServingRuntimeError):
            if runtime == "pickle":
                raise
            validation_frame = _load_validation_frame(
                client=runtime_client,
                data_root=data_root,
                data_version=package.data_version,
            )
            runtime_used, package_after_runtime, predictions = _materialize_package_predictions(
                workspace_root=workspace_root,
                inspection_package=inspection.package,
                inspection=inspection,
                validation_frame=validation_frame,
                runtime="local",
                client=runtime_client,
            )
    else:
        validation_frame = _load_validation_frame(
            client=runtime_client,
            data_root=data_root,
            data_version=package.data_version,
        )
        runtime_used, package_after_runtime, predictions = _materialize_package_predictions(
            workspace_root=workspace_root,
            inspection_package=inspection.package,
            inspection=inspection,
            validation_frame=validation_frame,
            runtime=runtime,
            client=runtime_client,
        )

    if "target" not in predictions.columns:
        if validation_frame is None:
            validation_frame = _load_validation_frame(
                client=runtime_client,
                data_root=data_root,
                data_version=package.data_version,
                columns=["era", "id", "target"],
            )
        predictions = _attach_native_target(predictions=predictions, validation_frame=validation_frame)

    eval_dir = inspection.package.package_path / "artifacts" / "eval" / dataset / runtime_used
    predictions_path = write_parquet(predictions, eval_dir / "predictions.parquet", index=False)
    scoring_client = create_training_data_client() if client is None else runtime_client
    include_feature_neutral_metrics = stage == "post_training_full"
    summaries, score_provenance, metric_frames = score_prediction_file_with_details(
        predictions_path=predictions_path,
        pred_cols=[_PACKAGE_SCORE_PRED_COL],
        target_col="target",
        scoring_target_cols=_PACKAGE_SCORE_TARGETS,
        scoring_targets_explicit=True,
        data_version=inspection.package.data_version,
        client=scoring_client,
        data_root=data_root,
        benchmark_data_path=_resolve_validation_benchmark_path(
            client=scoring_client,
            data_root=data_root,
            data_version=inspection.package.data_version,
        ),
        include_feature_neutral_metrics=include_feature_neutral_metrics,
    )

    example_metric = _score_example_predictions(
        client=scoring_client,
        data_root=data_root,
        data_version=inspection.package.data_version,
        predictions=predictions,
    )
    if example_metric is not None:
        example_key, example_summary, example_frame, example_meta = example_metric
        summaries[example_key] = example_summary
        metric_frames[example_key] = example_frame
        joins_payload = score_provenance.setdefault("joins", {})
        if isinstance(joins_payload, dict):
            joins_payload["example_predictions"] = example_meta

    explicit_summaries = _summaries_payload(
        summaries=summaries,
        scoring_targets=_PACKAGE_SCORE_TARGETS,
        native_target_col="target",
    )
    explicit_metric_frames = _rewrite_metric_frames(
        metric_frames=metric_frames,
        scoring_targets=_PACKAGE_SCORE_TARGETS,
        native_target_col="target",
    )
    metric_series = _build_metric_series(
        metric_frames=explicit_metric_frames,
        package=inspection.package,
        dataset=dataset,
        runtime_used=runtime_used,
        prediction_col=_PACKAGE_SCORE_PRED_COL,
    )

    score_provenance_payload = {
        "package": {
            "package_id": inspection.package.package_id,
            "experiment_id": inspection.package.experiment_id,
            "status_preserved_from": original_status,
        },
        "evaluation": {
            "dataset": dataset,
            "runtime_requested": runtime,
            "runtime_used": runtime_used,
            "stage": stage,
            "predictions_path": str(predictions_path),
            "row_count": int(len(predictions)),
            "era_count": int(predictions["era"].nunique(dropna=True)),
        },
        "scoring": _json_ready(score_provenance),
        "explicit_metric_keys": sorted(explicit_summaries.keys()),
    }
    score_provenance_path = eval_dir / "score_provenance.json"
    score_provenance_path.write_text(json.dumps(score_provenance_payload, indent=2, sort_keys=True), encoding="utf-8")

    summaries_path = eval_dir / "summaries.json"
    summaries_path.write_text(json.dumps(explicit_summaries, indent=2, sort_keys=True), encoding="utf-8")

    metric_series_path = write_parquet(metric_series, eval_dir / "metric_series.parquet", index=False)
    manifest_payload = {
        "package_id": inspection.package.package_id,
        "experiment_id": inspection.package.experiment_id,
        "dataset": dataset,
        "data_version": inspection.package.data_version,
        "runtime_requested": runtime,
        "runtime_used": runtime_used,
        "stage": stage,
        "predictions_path": str(predictions_path),
        "score_provenance_path": str(score_provenance_path),
        "summaries_path": str(summaries_path),
        "metric_series_path": str(metric_series_path),
        "completed_at": utc_now_iso(),
        "row_count": int(len(predictions)),
        "era_count": int(predictions["era"].nunique(dropna=True)),
        "metric_keys": sorted(explicit_summaries.keys()),
    }
    manifest_path = eval_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")

    updated_package = _persist_package_artifacts(
        package=package_after_runtime,
        preserved_status=original_status,
        artifacts={
            "last_validation_eval_dataset": dataset,
            "last_validation_eval_runtime_requested": runtime,
            "last_validation_eval_runtime_used": runtime_used,
            "last_validation_eval_stage": stage,
            "last_validation_eval_predictions_path": str(predictions_path),
            "last_validation_eval_score_provenance_path": str(score_provenance_path),
            "last_validation_eval_summaries_path": str(summaries_path),
            "last_validation_eval_metric_series_path": str(metric_series_path),
            "last_validation_eval_manifest_path": str(manifest_path),
            "last_validation_eval_completed_at": manifest_payload["completed_at"],
            "last_validation_eval_row_count": str(manifest_payload["row_count"]),
            "last_validation_eval_era_count": str(manifest_payload["era_count"]),
        },
    )
    return PackageScoreResult(
        package=updated_package,
        dataset=dataset,
        data_version=inspection.package.data_version,
        stage=stage,
        runtime_requested=runtime,
        runtime_used=runtime_used,
        predictions_path=predictions_path,
        score_provenance_path=score_provenance_path,
        summaries_path=summaries_path,
        metric_series_path=metric_series_path,
        manifest_path=manifest_path,
        row_count=int(manifest_payload["row_count"]),
        era_count=int(manifest_payload["era_count"]),
    )


def sync_submission_package_diagnostics(
    *,
    workspace_root: str | Path,
    experiment_id: str,
    package_id: str,
    wait: bool = True,
    client: Any | None = None,
) -> PackageDiagnosticsSyncResult:
    """Persist the latest Numerai diagnostics snapshot after one uploaded package pickle."""

    package = load_package(workspace_root=workspace_root, experiment_id=experiment_id, package_id=package_id)
    upload_id = package.artifacts.get("last_pickle_upload_id")
    model_id = package.artifacts.get("last_pickle_model_id")
    if not upload_id or not model_id:
        raise ServingValidationError("serving_package_diagnostics_upload_missing")

    diagnostics_client = create_serving_client() if client is None else client
    diagnostics_dir = package.package_path / "artifacts" / "diagnostics" / upload_id
    status_deadline = time.monotonic() + _DIAGNOSTICS_POLL_TIMEOUT_SECONDS
    latest_status: dict[str, Any] | None = None
    latest_logs: list[dict[str, Any]] = []

    while True:
        latest_status = diagnostics_client.compute_pickle_status(pickle_id=upload_id, model_id=model_id)
        if latest_status is None:
            latest_status = _pending_compute_status_payload(upload_id=upload_id, model_id=model_id)
        latest_logs = diagnostics_client.compute_pickle_diagnostics_logs(pickle_id=upload_id)
        diagnostics_state = _diagnostics_state(latest_status)
        if diagnostics_state["terminal"] or not wait or time.monotonic() >= status_deadline:
            break
        time.sleep(_DIAGNOSTICS_POLL_INTERVAL_SECONDS)

    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    compute_status_path = diagnostics_dir / "compute_status.json"
    compute_status_path.write_text(
        json.dumps(_json_ready(latest_status), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logs_path = diagnostics_dir / "diagnostics_logs.json"
    logs_path.write_text(json.dumps(_json_ready(latest_logs), indent=2, sort_keys=True), encoding="utf-8")

    raw_path: Path | None = None
    summary_path: Path | None = None
    per_era_path: Path | None = None
    payload_provenance: dict[str, str | None] = {}
    if diagnostics_state["success"]:
        diagnostics_payload = diagnostics_client.diagnostics(model_id=model_id)
        payload_provenance = _diagnostics_payload_provenance(
            diagnostics_payload=diagnostics_payload,
            upload_id=upload_id,
            model_id=model_id,
        )
        raw_path = diagnostics_dir / "raw.json"
        raw_path.write_text(json.dumps(_json_ready(diagnostics_payload), indent=2, sort_keys=True), encoding="utf-8")

        summary_payload = _diagnostics_summary_payload(
            diagnostics_payload=diagnostics_payload,
            upload_id=upload_id,
            model_id=model_id,
            payload_provenance=payload_provenance,
        )
        summary_path = diagnostics_dir / "summary.json"
        summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

        per_era = pd.DataFrame(list(diagnostics_payload.get("perEraDiagnostics", []) or []))
        per_era_path = write_parquet(per_era, diagnostics_dir / "per_era.parquet", index=False)

    synced_at = utc_now_iso()
    updated_package = _persist_package_artifacts(
        package=package,
        preserved_status=package.status,
        artifacts={
            "last_diagnostics_upload_id": upload_id,
            "last_diagnostics_status": str(diagnostics_state["status"]),
            "last_diagnostics_synced_at": synced_at,
            "last_diagnostics_compute_status_path": str(compute_status_path),
            "last_diagnostics_logs_path": str(logs_path),
            "last_diagnostics_raw_path": None if raw_path is None else str(raw_path),
            "last_diagnostics_summary_path": None if summary_path is None else str(summary_path),
            "last_diagnostics_per_era_path": None if per_era_path is None else str(per_era_path),
            "last_diagnostics_payload_scope": payload_provenance.get("payload_scope"),
            "last_diagnostics_payload_model_id": payload_provenance.get("payload_model_id"),
            "last_diagnostics_payload_synced_upload_id": payload_provenance.get("synced_upload_id"),
            "last_diagnostics_payload_selected_updated_at": payload_provenance.get("selected_updated_at"),
            "last_diagnostics_payload_source_diagnostics_id": payload_provenance.get("source_diagnostics_id"),
        },
        clear_keys={
            "last_diagnostics_raw_path",
            "last_diagnostics_summary_path",
            "last_diagnostics_per_era_path",
            "last_diagnostics_payload_scope",
            "last_diagnostics_payload_model_id",
            "last_diagnostics_payload_synced_upload_id",
            "last_diagnostics_payload_selected_updated_at",
            "last_diagnostics_payload_source_diagnostics_id",
        },
    )
    return PackageDiagnosticsSyncResult(
        package=updated_package,
        model_id=model_id,
        upload_id=upload_id,
        wait_requested=wait,
        diagnostics_status=str(diagnostics_state["status"]),
        terminal=bool(diagnostics_state["terminal"]),
        timed_out=bool(wait and not diagnostics_state["terminal"]),
        synced_at=synced_at,
        compute_status_path=compute_status_path,
        logs_path=logs_path,
        raw_path=raw_path,
        summary_path=summary_path,
        per_era_path=per_era_path,
    )


def _materialize_package_predictions(
    *,
    workspace_root: str | Path,
    inspection_package: SubmissionPackageRecord,
    inspection: Any,
    validation_frame: pd.DataFrame,
    runtime: PackageScoreRuntime,
    client: Any,
) -> tuple[Literal["pickle", "local"], SubmissionPackageRecord, pd.DataFrame]:
    if runtime == "local":
        return _predict_with_local_runtime(
            workspace_root=workspace_root,
            package=inspection_package,
            inspection=inspection,
            validation_frame=validation_frame,
            client=client,
        )
    if runtime == "pickle":
        return _predict_with_pickle_runtime(
            workspace_root=workspace_root,
            experiment_id=inspection_package.experiment_id,
            package_id=inspection_package.package_id,
            package=inspection_package,
            validation_frame=validation_frame,
            client=client,
        )
    if inspection.model_upload_compatible:
        try:
            return _predict_with_pickle_runtime(
                workspace_root=workspace_root,
                experiment_id=inspection_package.experiment_id,
                package_id=inspection_package.package_id,
                package=inspection_package,
                validation_frame=validation_frame,
                client=client,
            )
        except (ServingUnsupportedConfigError, ServingValidationError):
            pass
        except ServingRuntimeError:
            pass
    return _predict_with_local_runtime(
        workspace_root=workspace_root,
        package=inspection_package,
        inspection=inspection,
        validation_frame=validation_frame,
        client=client,
    )


def _predict_with_local_runtime(
    *,
    workspace_root: str | Path,
    package: SubmissionPackageRecord,
    inspection: Any,
    validation_frame: pd.DataFrame,
    client: Any,
) -> tuple[Literal["local"], SubmissionPackageRecord, pd.DataFrame]:
    if not inspection.local_live_compatible:
        raise ServingUnsupportedConfigError("serving_live_preflight_failed")
    component_predictions = _fit_and_predict_package(
        workspace_root=workspace_root,
        client=client,
        package=package,
        live_features=validation_frame,
    )
    internal, _ = blend_component_predictions(
        component_predictions=component_predictions,
        live_features=validation_frame,
        blend_rule=package.blend_rule,
        neutralization=package.neutralization,
    )
    return "local", package, internal[["era", "id", "prediction"]].copy()


def _predict_with_pickle_runtime(
    *,
    workspace_root: str | Path,
    experiment_id: str,
    package_id: str,
    package: SubmissionPackageRecord,
    validation_frame: pd.DataFrame,
    client: Any,
) -> tuple[Literal["pickle"], SubmissionPackageRecord, pd.DataFrame]:
    built = build_submission_pickle(
        workspace_root=workspace_root,
        experiment_id=experiment_id,
        package_id=package_id,
        client=client,
    )
    predictor = pd.read_pickle(built.pickle_path)
    submission = predictor(validation_frame.copy(), None)
    if "prediction" not in submission.columns:
        raise ServingValidationError("serving_package_pickle_predictions_invalid")
    predictions = validation_frame[["era", "id"]].copy()
    predictions["prediction"] = submission["prediction"].to_numpy(dtype=float)
    return "pickle", built.package, predictions


def _load_validation_frame(
    *,
    client: Any,
    data_root: Path,
    data_version: str,
    dataset: str = "validation",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    version_root = resolve_data_version_root(data_root=data_root, data_version=data_version)
    filename = resolve_variant_dataset_filename(dataset_variant="non_downsampled", filename=f"{dataset}.parquet")
    dataset_path = (version_root / filename).resolve()
    if not dataset_path.exists():
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        client.download_dataset(f"{data_version}/{dataset}.parquet", dest_path=str(dataset_path))
    frame = pd.read_parquet(dataset_path, columns=columns)
    if "data_type" in frame.columns:
        filtered = frame[frame["data_type"].astype(str) == dataset].copy()
        if not filtered.empty:
            frame = filtered
    if "id" not in frame.columns:
        if frame.index.name == "id":
            frame = frame.reset_index()
        else:
            raise TrainingDataError("training_data_id_col_missing")
    if "era" not in frame.columns:
        frame["era"] = dataset
    return frame.reset_index(drop=True)


def _attach_native_target(*, predictions: pd.DataFrame, validation_frame: pd.DataFrame) -> pd.DataFrame:
    if "target" in predictions.columns:
        return predictions
    if "target" not in validation_frame.columns:
        raise TrainingDataError("training_predictions_missing_columns:target")
    native = validation_frame[["era", "id", "target"]].copy()
    joined = predictions.merge(native, on=["era", "id"], how="left", validate="one_to_one")
    if joined["target"].isna().any():
        raise TrainingDataError("training_predictions_missing_columns:target")
    return joined


def _score_example_predictions(
    *,
    client: Any,
    data_root: Path,
    data_version: str,
    predictions: pd.DataFrame,
) -> tuple[str, pd.DataFrame, pd.DataFrame, dict[str, Any]] | None:
    version_root = resolve_data_version_root(data_root=data_root, data_version=data_version)
    filename = resolve_variant_dataset_filename(
        dataset_variant="non_downsampled",
        filename="validation_example_preds.parquet",
    )
    example_path = (version_root / filename).resolve()
    if not example_path.exists():
        try:
            client.download_dataset(f"{data_version}/validation_example_preds.parquet", dest_path=str(example_path))
        except Exception:
            return None
    frame = pd.read_parquet(example_path)
    if "id" not in frame.columns:
        if frame.index.name == "id":
            frame = frame.reset_index()
        else:
            return None
    if "era" not in frame.columns:
        return None
    example_cols = [col for col in frame.columns if col not in {"id", "era", "data_type"}]
    if not example_cols:
        return None
    example_col = "prediction" if "prediction" in example_cols else str(example_cols[0])
    reference_col = "example_prediction"
    joined = predictions.merge(
        frame[["era", "id", example_col]].rename(columns={example_col: reference_col}),
        on=["era", "id"],
        how="inner",
    )
    if joined.empty:
        return None
    per_era = per_era_reference_corr(joined, [_PACKAGE_SCORE_PRED_COL], reference_col, era_col="era")
    return (
        "corr_with_example_preds",
        summarize_scores(per_era),
        per_era,
        {
            "path": str(example_path),
            "prediction_col": example_col,
            "overlap_rows": int(len(joined)),
            "overlap_eras": int(joined["era"].nunique(dropna=True)),
        },
    )


def _resolve_validation_benchmark_path(*, client: Any, data_root: Path, data_version: str) -> Path:
    version_root = resolve_data_version_root(data_root=data_root, data_version=data_version)
    filename = resolve_variant_dataset_filename(
        dataset_variant="non_downsampled",
        filename="validation_benchmark_models.parquet",
    )
    benchmark_path = (version_root / filename).resolve()
    if not benchmark_path.exists():
        benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        client.download_dataset(f"{data_version}/validation_benchmark_models.parquet", dest_path=str(benchmark_path))
    return benchmark_path


def _summaries_payload(
    *,
    summaries: dict[str, pd.DataFrame],
    scoring_targets: tuple[str, ...],
    native_target_col: str,
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for key, frame in summaries.items():
        explicit_key = _rewrite_summary_metric_key(
            metric_key=key,
            scoring_targets=scoring_targets,
            native_target_col=native_target_col,
        )
        if frame.empty:
            payload[explicit_key] = {}
            continue
        row = frame.iloc[0].to_dict()
        payload[explicit_key] = {str(col): _json_ready(value) for col, value in row.items()}
    return payload


def _rewrite_metric_frames(
    *,
    metric_frames: dict[str, pd.DataFrame],
    scoring_targets: tuple[str, ...],
    native_target_col: str,
) -> dict[str, pd.DataFrame]:
    explicit: dict[str, pd.DataFrame] = {}
    for key, frame in metric_frames.items():
        explicit_key = _rewrite_series_metric_key(
            metric_key=key,
            scoring_targets=scoring_targets,
            native_target_col=native_target_col,
        )
        explicit[explicit_key] = frame
    return explicit


def _build_metric_series(
    *,
    metric_frames: dict[str, pd.DataFrame],
    package: SubmissionPackageRecord,
    dataset: str,
    runtime_used: str,
    prediction_col: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for metric_key, frame in metric_frames.items():
        if frame.empty:
            continue
        per_era = _normalize_metric_frame(frame)
        cumulative = build_cumulative_scores(per_era)
        for series_type, series_frame in (("per_era", per_era), ("cumulative", cumulative)):
            reset = series_frame.reset_index()
            era_column = reset.columns[0]
            value_col = reset.columns[1]
            for row in reset.itertuples(index=False):
                era_value = getattr(row, era_column)
                value = getattr(row, value_col)
                records.append(
                    {
                        "package_id": package.package_id,
                        "experiment_id": package.experiment_id,
                        "dataset": dataset,
                        "runtime": runtime_used,
                        "prediction_col": prediction_col,
                        "metric_key": metric_key,
                        "series_type": series_type,
                        "era": str(era_value),
                        "value": float(value) if pd.notna(value) else float("nan"),
                    }
                )
    return pd.DataFrame.from_records(records)


def _normalize_metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "era" in frame.columns:
        value_cols = [col for col in frame.columns if col != "era"]
        if len(value_cols) != 1:
            raise ServingValidationError("serving_package_score_single_prediction_required")
        return frame[["era", value_cols[0]]].rename(columns={value_cols[0]: "value"}).set_index("era")
    if len(frame.columns) != 1:
        raise ServingValidationError("serving_package_score_single_prediction_required")
    normalized = frame.copy()
    normalized.columns = ["value"]
    return normalized


def _rewrite_summary_metric_key(
    *,
    metric_key: str,
    scoring_targets: tuple[str, ...],
    native_target_col: str,
) -> str:
    alias_map = {_target_alias(target_col): target_col for target_col in scoring_targets}
    alias_map["native"] = native_target_col
    return _rewrite_metric_key(
        metric_key=metric_key,
        alias_map=alias_map,
        native_target_col=native_target_col,
        native_summary_keys=("corr", "fnc"),
        contribution_keys=("bmc_last_200_eras", "corr_delta_vs_baseline", "bmc", "mmc"),
    )


def _rewrite_series_metric_key(
    *,
    metric_key: str,
    scoring_targets: tuple[str, ...],
    native_target_col: str,
) -> str:
    alias_map = {_target_alias(target_col): target_col for target_col in scoring_targets}
    alias_map["native"] = native_target_col
    return _rewrite_metric_key(
        metric_key=metric_key,
        alias_map=alias_map,
        native_target_col=native_target_col,
        native_summary_keys=("corr", "fnc"),
        contribution_keys=("corr_delta_vs_baseline", "bmc", "mmc"),
    )


def _rewrite_metric_key(
    *,
    metric_key: str,
    alias_map: dict[str, str],
    native_target_col: str,
    native_summary_keys: tuple[str, ...],
    contribution_keys: tuple[str, ...],
) -> str:
    passthrough = {
        "corr_with_benchmark",
        "corr_with_example_preds",
        "cwmm",
    }
    if metric_key in passthrough:
        return metric_key

    for prefix in native_summary_keys:
        if metric_key == prefix:
            return f"{prefix}_{native_target_col}"
        if metric_key.startswith(f"{prefix}_"):
            alias = metric_key[len(prefix) + 1 :]
            target_col = alias_map.get(alias)
            if target_col is not None:
                return f"{prefix}_{target_col}"
    for prefix in contribution_keys:
        if metric_key == prefix:
            return f"{prefix}_{DEFAULT_PAYOUT_TARGET_COL}"
        if metric_key.startswith(f"{prefix}_"):
            alias = metric_key[len(prefix) + 1 :]
            target_col = alias_map.get(alias)
            if target_col is not None:
                return f"{prefix}_{target_col}"
    return metric_key


def _target_alias(target_col: str) -> str:
    alias = target_col
    if alias.startswith("target_"):
        alias = alias[len("target_") :]
    return "".join(part for part in alias.split("_") if part)


def _persist_package_artifacts(
    *,
    package: SubmissionPackageRecord,
    preserved_status: str,
    artifacts: dict[str, str | None],
    clear_keys: set[str] | None = None,
) -> SubmissionPackageRecord:
    merged_artifacts = dict(package.artifacts)
    for key in clear_keys or set():
        merged_artifacts.pop(str(key), None)
    for key, value in artifacts.items():
        if value is not None:
            merged_artifacts[str(key)] = str(value)
    updated = replace(
        package,
        status=preserved_status,
        artifacts=merged_artifacts,
        updated_at=utc_now_iso(),
    )
    return save_package(updated)


def _diagnostics_summary_payload(
    *,
    diagnostics_payload: dict[str, Any],
    upload_id: str,
    model_id: str,
    payload_provenance: dict[str, str | None],
) -> dict[str, Any]:
    summary = {
        key: value
        for key, value in diagnostics_payload.items()
        if key != "perEraDiagnostics" and not isinstance(value, (dict, list))
    }
    summary["upload_id"] = upload_id
    summary["model_id"] = model_id
    for key, value in payload_provenance.items():
        if value is not None:
            summary[key] = value
    return cast(dict[str, Any], _json_ready(summary))


def _diagnostics_state(compute_status: dict[str, Any]) -> dict[str, Any]:
    diagnostics_status = _normalized_status(compute_status.get("diagnosticsStatus"))
    trigger_status = _normalized_status(compute_status.get("triggerStatus"))
    validation_status = _normalized_status(compute_status.get("validationStatus"))
    status = diagnostics_status or trigger_status or validation_status or "unknown"
    if diagnostics_status in _SUCCESS_STATUSES:
        return {"status": diagnostics_status, "terminal": True, "success": True}
    if diagnostics_status in _FAILURE_STATUSES or trigger_status in _FAILURE_STATUSES:
        return {"status": diagnostics_status or trigger_status or "failed", "terminal": True, "success": False}
    if status in _PENDING_STATUSES or status == "unknown":
        return {"status": status, "terminal": False, "success": False}
    if status in _SUCCESS_STATUSES:
        return {"status": status, "terminal": True, "success": True}
    if status in _FAILURE_STATUSES:
        return {"status": status, "terminal": True, "success": False}
    return {"status": status, "terminal": False, "success": False}


def _pending_compute_status_payload(*, upload_id: str, model_id: str) -> dict[str, Any]:
    return {
        "id": upload_id,
        "modelId": model_id,
        "diagnosticsStatus": "pending",
        "status": "pending",
        "reason": "compute_pickle_status_missing",
    }


def _diagnostics_payload_provenance(
    *,
    diagnostics_payload: dict[str, Any],
    upload_id: str,
    model_id: str,
) -> dict[str, str | None]:
    selected_updated_at = _json_ready(diagnostics_payload.get("updatedAt"))
    source_diagnostics_id = _json_ready(diagnostics_payload.get("id"))
    return {
        "payload_scope": "latest_model",
        "payload_model_id": str(model_id),
        "synced_upload_id": str(upload_id),
        "selected_updated_at": None if selected_updated_at is None else str(selected_updated_at),
        "source_diagnostics_id": None if source_diagnostics_id is None else str(source_diagnostics_id),
    }


def _normalized_status(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if hasattr(value, "item"):
        try:
            return _json_ready(value.item())
        except Exception:
            return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


__all__ = [
    "score_submission_package",
    "sync_submission_package_diagnostics",
]
