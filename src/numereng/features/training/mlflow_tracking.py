"""Optional MLflow tracking hooks for training runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def maybe_log_training_run(
    *,
    run_id: str,
    config: dict[str, object],
    metrics_payload: dict[str, object],
    artifacts: dict[str, str],
    output_root: Path,
) -> str | None:
    """Log training metadata to MLflow when explicitly enabled.

    Returns `None` on success or if disabled. Returns an error marker string when
    logging is enabled but could not complete.
    """

    if not _is_mlflow_enabled():
        return None

    try:
        import mlflow  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return "mlflow_not_installed"

    tracking_uri = os.getenv("NUMERENG_MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.getenv("NUMERENG_MLFLOW_EXPERIMENT")
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    try:
        with mlflow.start_run(run_name=run_id):
            for key, param_value in _flatten_scalar_params(config).items():
                mlflow.log_param(key, param_value)

            for key, metric_value in _flatten_numeric_metrics(metrics_payload).items():
                mlflow.log_metric(key, metric_value)

            for relative_path in sorted(artifacts.values()):
                artifact_path = (output_root / relative_path).resolve()
                if artifact_path.is_file():
                    mlflow.log_artifact(str(artifact_path))
        return None
    except Exception as exc:  # pragma: no cover - depends on external backend state
        return f"mlflow_log_failed:{type(exc).__name__}"


def _is_mlflow_enabled() -> bool:
    raw = os.getenv("NUMERENG_MLFLOW_ENABLED", "")
    return raw.lower() in {"1", "true", "yes", "on"}


def _flatten_scalar_params(config: dict[str, object]) -> dict[str, str]:
    flat: dict[str, str] = {}

    def _walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                _walk(child_prefix, nested)
            return

        if isinstance(value, (list, tuple)):
            return

        if value is None:
            flat[prefix] = "null"
            return

        flat[prefix] = str(value)

    _walk("", config)
    return flat


def _flatten_numeric_metrics(metrics_payload: dict[str, object]) -> dict[str, float]:
    flat: dict[str, float] = {}

    def _walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                _walk(child_prefix, nested)
            return
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            flat[prefix] = float(value)

    _walk("", metrics_payload)
    return flat
