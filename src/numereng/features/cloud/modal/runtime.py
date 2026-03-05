"""Remote runtime helpers for Modal-executed training runs."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from numereng.features.cloud.modal.contracts import (
    ModalRuntimePayload,
    ModalRuntimeResult,
    ModalTrainingProfile,
)
from numereng.features.telemetry import bind_launch_metadata, get_launch_metadata


def _sanitize_payload_config_filename(raw_filename: str) -> str:
    """Return a safe basename for one payload config filename."""
    if not raw_filename or raw_filename in {".", ".."}:
        raise ValueError("runtime_config_filename_invalid")
    if "/" in raw_filename or "\\" in raw_filename:
        raise ValueError("runtime_config_filename_invalid")
    candidate = Path(raw_filename)
    if candidate.is_absolute() or candidate.name != raw_filename:
        raise ValueError("runtime_config_filename_invalid")
    return raw_filename


def _load_run_training() -> Callable[..., Any]:
    module = importlib.import_module("numereng.features.training")
    run_training_impl = getattr(module, "run_training", None)
    if run_training_impl is None or not callable(run_training_impl):
        raise RuntimeError("training_run_function_missing")
    return cast(Callable[..., Any], run_training_impl)


def build_runtime_payload_from_config(
    *,
    config_path: str | Path,
    output_dir: str | None = None,
    profile: ModalTrainingProfile | None = None,
    engine_mode: str | None = None,
    window_size_eras: int | None = None,
    embargo_eras: int | None = None,
) -> ModalRuntimePayload:
    """Build runtime payload from one local config path."""
    resolved_config_path = Path(config_path).expanduser().resolve()
    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"config_path_not_found:{resolved_config_path}")
    return ModalRuntimePayload(
        config_text=resolved_config_path.read_text(encoding="utf-8"),
        config_filename=resolved_config_path.name,
        output_dir=output_dir,
        profile=profile,
        engine_mode=engine_mode,
        window_size_eras=window_size_eras,
        embargo_eras=embargo_eras,
    )


def run_training_payload(payload: ModalRuntimePayload | dict[str, Any]) -> dict[str, str]:
    """Execute one serialized training payload and return serialized result."""
    runtime_payload = ModalRuntimePayload.model_validate(payload)
    config_filename = _sanitize_payload_config_filename(runtime_payload.config_filename)
    with TemporaryDirectory(prefix="numereng-modal-") as tmp_dir:
        config_path = Path(tmp_dir) / config_filename
        config_path.write_text(runtime_payload.config_text, encoding="utf-8")
        launch_scope = (
            nullcontext()
            if get_launch_metadata() is not None
            else bind_launch_metadata(source="cloud.modal.runtime", operation_type="run", job_type="run")
        )
        with launch_scope:
            run_kwargs: dict[str, Any] = {
                "config_path": config_path,
                "output_dir": runtime_payload.output_dir,
                "engine_mode": runtime_payload.engine_mode,
                "window_size_eras": runtime_payload.window_size_eras,
                "embargo_eras": runtime_payload.embargo_eras,
            }
            if runtime_payload.profile is not None:
                run_kwargs["profile"] = runtime_payload.profile
            run_result = _load_run_training()(**run_kwargs)

    output_dir = str(Path(run_result.results_path).resolve().parent)
    result = ModalRuntimeResult(
        run_id=run_result.run_id,
        predictions_path=str(run_result.predictions_path),
        results_path=str(run_result.results_path),
        output_dir=output_dir,
    )
    dumped = result.model_dump(mode="json")
    return {key: str(value) for key, value in dumped.items()}
