"""Feature-neutralization business logic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from numereng.features.feature_neutralization.contracts import (
    NeutralizationMode,
    NeutralizationResult,
    NeutralizePredictionsRequest,
)
from numereng.features.feature_neutralization.io import (
    ensure_unique_join_keys,
    normalize_join_keys,
    read_table,
    resolve_neutralizer_path,
    resolve_output_path,
    resolve_predictions_path,
    resolve_run_predictions_path,
    write_table,
)

_KEY_COLS = ("era", "id")
_EXCLUDED_AUTO_NEUTRALIZER_COLS = {
    "era",
    "id",
    "prediction",
    "target",
    "target_ender_20",
    "target_cyrus_20",
}


class NeutralizationError(Exception):
    """Base error for feature-neutralization workflows."""


class NeutralizationValidationError(NeutralizationError):
    """Raised when neutralization inputs are invalid."""


class NeutralizationDataError(NeutralizationError):
    """Raised when neutralization data cannot be aligned/processed."""


class NeutralizationExecutionError(NeutralizationError):
    """Raised when neutralization math fails."""


def load_neutralizer_table(
    *,
    neutralizer_path: str | Path,
    neutralizer_cols: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Load one neutralizer table and resolve selected neutralizer columns."""

    try:
        resolved_path = resolve_neutralizer_path(neutralizer_path)
        frame = read_table(resolved_path)
    except ValueError as exc:
        raise NeutralizationValidationError(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive IO guard
        raise NeutralizationDataError(f"neutralization_neutralizer_read_failed:{neutralizer_path}") from exc

    required_cols = ["era", "id"]
    missing = [col for col in required_cols if col not in frame.columns]
    if missing:
        raise NeutralizationValidationError(f"neutralization_neutralizer_missing_columns:{','.join(missing)}")

    normalized = normalize_join_keys(frame)
    try:
        ensure_unique_join_keys(normalized, keys=_KEY_COLS)
    except ValueError as exc:
        raise NeutralizationValidationError(str(exc)) from exc

    selected_cols = _resolve_neutralizer_columns(frame=normalized, neutralizer_cols=neutralizer_cols)

    payload = normalized[["era", "id", *selected_cols]].copy()
    for col in selected_cols:
        payload[col] = pd.to_numeric(payload[col], errors="coerce").fillna(0.0)
    return payload, selected_cols


def neutralize_prediction_frame(
    *,
    predictions: pd.DataFrame,
    neutralizers: pd.DataFrame,
    neutralizer_cols: tuple[str, ...],
    proportion: float = 0.5,
    mode: NeutralizationMode = "era",
    rank_output: bool = True,
) -> pd.DataFrame:
    """Neutralize one prediction dataframe using provided neutralizer data."""

    _validate_proportion(proportion)
    _validate_mode(mode)

    if predictions.empty:
        raise NeutralizationDataError("neutralization_predictions_empty")

    required_pred_cols = ["era", "id", "prediction"]
    missing_pred_cols = [col for col in required_pred_cols if col not in predictions.columns]
    if missing_pred_cols:
        raise NeutralizationValidationError(f"neutralization_predictions_missing_columns:{','.join(missing_pred_cols)}")

    if not neutralizer_cols:
        raise NeutralizationValidationError("neutralization_neutralizer_columns_empty")

    required_neutralizer_cols = ["era", "id", *neutralizer_cols]
    missing_neutralizer_cols = [col for col in required_neutralizer_cols if col not in neutralizers.columns]
    if missing_neutralizer_cols:
        raise NeutralizationValidationError(
            f"neutralization_neutralizer_missing_columns:{','.join(missing_neutralizer_cols)}"
        )

    source = predictions.copy()
    source["prediction"] = pd.to_numeric(source["prediction"], errors="coerce")
    if bool(source["prediction"].isna().any()):
        raise NeutralizationDataError("neutralization_predictions_contains_nan")
    source_normalized = normalize_join_keys(source)

    neutralizers_normalized = normalize_join_keys(neutralizers)
    try:
        ensure_unique_join_keys(neutralizers_normalized, keys=_KEY_COLS)
    except ValueError as exc:
        raise NeutralizationValidationError(str(exc)) from exc

    merged = source_normalized.merge(
        neutralizers_normalized[["era", "id", *neutralizer_cols]],
        on=["era", "id"],
        how="left",
        sort=False,
        validate="many_to_one",
    )

    missing_rows = int(merged[list(neutralizer_cols)].isna().any(axis=1).sum())
    if missing_rows > 0:
        raise NeutralizationDataError(f"neutralization_missing_neutralizer_rows:{missing_rows}")

    x_matrix = np.asarray(merged[list(neutralizer_cols)].to_numpy(dtype=float), dtype=float)
    y_values = np.asarray(merged["prediction"].to_numpy(dtype=float), dtype=float)

    try:
        neutralized = _neutralize_values(
            values=y_values,
            neutralizers=x_matrix,
            eras=merged["era"],
            proportion=proportion,
            mode=mode,
            rank_output=rank_output,
        )
    except NeutralizationExecutionError:
        raise
    except Exception as exc:  # pragma: no cover - defensive math guard
        raise NeutralizationExecutionError("neutralization_math_failed") from exc

    output = source.copy()
    output["prediction"] = neutralized
    return output


def neutralize_predictions_file(
    *,
    request: NeutralizePredictionsRequest,
    run_id: str | None = None,
) -> NeutralizationResult:
    """Neutralize one predictions file and persist sidecar output."""

    _validate_proportion(request.proportion)
    _validate_mode(request.mode)

    try:
        source_path = resolve_predictions_path(request.predictions_path)
        output_path = resolve_output_path(source_path=source_path, output_path=request.output_path)
    except ValueError as exc:
        raise NeutralizationValidationError(str(exc)) from exc

    neutralizers, selected_cols = load_neutralizer_table(
        neutralizer_path=request.neutralizer_path,
        neutralizer_cols=request.neutralizer_cols,
    )

    try:
        source_frame = read_table(source_path)
    except Exception as exc:  # pragma: no cover - defensive IO guard
        raise NeutralizationDataError(f"neutralization_predictions_read_failed:{source_path}") from exc

    neutralized = neutralize_prediction_frame(
        predictions=source_frame,
        neutralizers=neutralizers,
        neutralizer_cols=selected_cols,
        proportion=request.proportion,
        mode=request.mode,
        rank_output=request.rank_output,
    )

    try:
        write_table(frame=neutralized, path=output_path)
    except ValueError as exc:
        raise NeutralizationValidationError(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive IO guard
        raise NeutralizationDataError(f"neutralization_output_write_failed:{output_path}") from exc

    return NeutralizationResult(
        source_path=source_path,
        output_path=output_path,
        run_id=run_id,
        neutralizer_path=Path(request.neutralizer_path).expanduser().resolve(),
        neutralizer_cols=selected_cols,
        proportion=request.proportion,
        mode=request.mode,
        rank_output=request.rank_output,
        source_rows=int(len(source_frame)),
        neutralizer_rows=int(len(neutralizers)),
        matched_rows=int(len(neutralized)),
    )


def neutralize_run_predictions(
    *,
    run_id: str,
    neutralizer_path: str | Path,
    store_root: str | Path = ".numereng",
    output_path: str | Path | None = None,
    proportion: float = 0.5,
    mode: NeutralizationMode = "era",
    neutralizer_cols: tuple[str, ...] | None = None,
    rank_output: bool = True,
) -> NeutralizationResult:
    """Resolve run predictions and persist a neutralized sidecar file."""

    try:
        source_path = resolve_run_predictions_path(store_root=store_root, run_id=run_id)
    except ValueError as exc:
        raise NeutralizationValidationError(str(exc)) from exc

    return neutralize_predictions_file(
        request=NeutralizePredictionsRequest(
            predictions_path=source_path,
            neutralizer_path=Path(neutralizer_path),
            output_path=Path(output_path).expanduser().resolve() if output_path is not None else None,
            proportion=proportion,
            mode=mode,
            neutralizer_cols=neutralizer_cols,
            rank_output=rank_output,
        ),
        run_id=run_id,
    )


def _resolve_neutralizer_columns(
    *,
    frame: pd.DataFrame,
    neutralizer_cols: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if neutralizer_cols is not None:
        cleaned = tuple(col for col in (item.strip() for item in neutralizer_cols) if col)
        if not cleaned:
            raise NeutralizationValidationError("neutralization_neutralizer_columns_empty")

        missing = [col for col in cleaned if col not in frame.columns]
        if missing:
            raise NeutralizationValidationError(f"neutralization_neutralizer_columns_missing:{','.join(missing)}")

        non_numeric = [col for col in cleaned if not pd.api.types.is_numeric_dtype(frame[col])]
        if non_numeric:
            raise NeutralizationValidationError(
                f"neutralization_neutralizer_columns_non_numeric:{','.join(non_numeric)}"
            )
        return cleaned

    selected = tuple(
        str(col)
        for col in frame.columns
        if str(col) not in _EXCLUDED_AUTO_NEUTRALIZER_COLS and pd.api.types.is_numeric_dtype(frame[col])
    )
    if not selected:
        raise NeutralizationValidationError("neutralization_neutralizer_columns_missing_numeric")
    return selected


def _neutralize_values(
    *,
    values: np.ndarray,
    neutralizers: np.ndarray,
    eras: pd.Series,
    proportion: float,
    mode: NeutralizationMode,
    rank_output: bool,
) -> np.ndarray:
    if values.ndim != 1:
        raise NeutralizationExecutionError("neutralization_prediction_vector_invalid")
    if neutralizers.ndim != 2:
        raise NeutralizationExecutionError("neutralization_matrix_invalid")
    if neutralizers.shape[0] != values.shape[0]:
        raise NeutralizationExecutionError("neutralization_row_count_mismatch")

    resolved = np.asarray(values, dtype=float).copy()

    if mode == "global":
        resolved = _neutralize_vector(values=resolved, neutralizers=neutralizers, proportion=proportion)
    else:
        era_values = eras.astype(str).to_numpy()
        for era in pd.unique(era_values):
            idx = np.where(era_values == era)[0]
            if idx.size == 0:
                continue
            resolved[idx] = _neutralize_vector(
                values=resolved[idx],
                neutralizers=neutralizers[idx, :],
                proportion=proportion,
            )

    if rank_output:
        ranked = pd.Series(resolved, dtype=float)
        if mode == "global":
            resolved = np.asarray(ranked.rank(pct=True, method="average").to_numpy(dtype=float), dtype=float)
        else:
            ranked_by_era = ranked.groupby(eras.astype(str), sort=False).rank(pct=True, method="average")
            resolved = np.asarray(ranked_by_era.to_numpy(dtype=float), dtype=float)

    if not np.all(np.isfinite(resolved)):
        raise NeutralizationExecutionError("neutralization_output_non_finite")
    return resolved


def _neutralize_vector(
    *,
    values: np.ndarray,
    neutralizers: np.ndarray,
    proportion: float,
) -> np.ndarray:
    if values.size == 0:
        return values

    if neutralizers.shape[1] == 0 or values.size < 2:
        return values

    matrix = np.nan_to_num(neutralizers.astype(float), copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    vector = np.nan_to_num(values.astype(float), copy=True, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        beta = np.linalg.pinv(matrix) @ vector
        component = matrix @ beta
    except np.linalg.LinAlgError as exc:
        raise NeutralizationExecutionError("neutralization_linalg_failed") from exc

    resolved = vector - (proportion * component)
    return np.asarray(resolved, dtype=float)


def _validate_proportion(proportion: float) -> None:
    if not np.isfinite(proportion):
        raise NeutralizationValidationError("neutralization_proportion_invalid")
    if proportion < 0.0 or proportion > 1.0:
        raise NeutralizationValidationError("neutralization_proportion_out_of_bounds")


def _validate_mode(mode: NeutralizationMode) -> None:
    if mode not in {"era", "global"}:
        raise NeutralizationValidationError(f"neutralization_mode_invalid:{mode}")
