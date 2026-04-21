"""Docs sync and dataset utility contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from numereng.api._contracts.shared import WorkspaceBoundRequest


class DatasetToolsBuildDownsampleRequest(BaseModel):
    data_version: str = "v5.2"
    data_dir: str = ".numereng/datasets"
    rebuild: bool = False
    downsample_eras_step: int = 4
    downsample_eras_offset: int = 0


class DatasetToolsBuildDownsampleResponse(BaseModel):
    data_version: str
    data_dir: str
    downsampled_full_path: str
    downsampled_full_benchmark_path: str
    downsampled_rows: int
    downsampled_full_benchmark_rows: int
    total_eras: int
    kept_eras: int
    downsample_step: int
    downsample_offset: int


class DocsSyncRequest(WorkspaceBoundRequest):
    domain: Literal["numerai"] = "numerai"


class DocsSyncResponse(BaseModel):
    workspace_root: str
    destination_root: str
    sync_meta_path: str
    upstream_commit: str
    synced_at: str
    synced_files: int = Field(default=0, ge=0)


__all__ = [
    "DatasetToolsBuildDownsampleRequest",
    "DatasetToolsBuildDownsampleResponse",
    "DocsSyncRequest",
    "DocsSyncResponse",
]
