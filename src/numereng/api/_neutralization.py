"""Feature-neutralization API handlers."""

from __future__ import annotations

from pathlib import Path

from numereng.api.contracts import NeutralizeRequest, NeutralizeResponse
from numereng.features.feature_neutralization import (
    NeutralizationDataError,
    NeutralizationError,
    NeutralizationExecutionError,
    NeutralizationValidationError,
)
from numereng.features.feature_neutralization import (
    NeutralizePredictionsRequest as FeatureNeutralizePredictionsRequest,
)
from numereng.platform.errors import PackageError


def neutralize_apply(request: NeutralizeRequest) -> NeutralizeResponse:
    """Apply feature neutralization to one run/file prediction source."""

    from numereng import api as api_module

    try:
        if request.run_id is not None:
            result = api_module.neutralize_run_prediction_artifact(
                run_id=request.run_id,
                neutralizer_path=request.neutralizer_path,
                store_root=request.store_root,
                output_path=request.output_path,
                proportion=request.neutralization_proportion,
                mode=request.neutralization_mode,
                neutralizer_cols=None if request.neutralizer_cols is None else tuple(request.neutralizer_cols),
                rank_output=request.neutralization_rank_output,
            )
        else:
            if request.predictions_path is None:  # pragma: no cover - guarded by model validation
                raise PackageError("neutralization_request_invalid")
            result = api_module.neutralize_prediction_artifact(
                request=FeatureNeutralizePredictionsRequest(
                    predictions_path=Path(request.predictions_path),
                    neutralizer_path=Path(request.neutralizer_path),
                    output_path=Path(request.output_path) if request.output_path else None,
                    proportion=request.neutralization_proportion,
                    mode=request.neutralization_mode,
                    neutralizer_cols=None if request.neutralizer_cols is None else tuple(request.neutralizer_cols),
                    rank_output=request.neutralization_rank_output,
                )
            )
    except (
        NeutralizationValidationError,
        NeutralizationDataError,
        NeutralizationExecutionError,
        NeutralizationError,
    ) as exc:
        raise PackageError(str(exc)) from exc

    return NeutralizeResponse(
        source_path=str(result.source_path),
        output_path=str(result.output_path),
        run_id=result.run_id,
        neutralizer_path=str(result.neutralizer_path),
        neutralizer_cols=list(result.neutralizer_cols),
        neutralization_proportion=result.proportion,
        neutralization_mode=result.mode,
        neutralization_rank_output=result.rank_output,
        source_rows=result.source_rows,
        neutralizer_rows=result.neutralizer_rows,
        matched_rows=result.matched_rows,
    )


__all__ = ["neutralize_apply"]
