"""Dataset-tools downsampling API handlers."""

from __future__ import annotations

from pathlib import Path

from numereng.api.contracts import (
    DatasetToolsBuildDownsampleRequest,
    DatasetToolsBuildDownsampleResponse,
)
from numereng.features.dataset_tools import (
    BuildDownsampledFullRequest,
    DatasetToolsError,
    DatasetToolsExecutionError,
    DatasetToolsValidationError,
    build_downsampled_full,
    build_downsampled_full_response_payload,
)
from numereng.platform.errors import PackageError


def dataset_tools_build_downsampled_full(
    request: DatasetToolsBuildDownsampleRequest,
) -> DatasetToolsBuildDownsampleResponse:
    """Build full + downsampled full datasets using official era downsampling."""

    try:
        from numereng import api as api_module

        result = build_downsampled_full(
            BuildDownsampledFullRequest(
                data_dir=Path(request.data_dir),
                data_version=request.data_version,
                rebuild=request.rebuild,
                downsample_eras_step=request.downsample_eras_step,
                downsample_eras_offset=request.downsample_eras_offset,
                skip_downsample=request.skip_downsample,
            ),
            client=api_module._create_numerai_client(tournament="classic"),
        )
    except (
        DatasetToolsValidationError,
        DatasetToolsExecutionError,
        DatasetToolsError,
    ) as exc:
        raise PackageError(str(exc)) from exc

    payload = build_downsampled_full_response_payload(result)
    return DatasetToolsBuildDownsampleResponse(**payload)


__all__ = ["dataset_tools_build_downsampled_full"]
