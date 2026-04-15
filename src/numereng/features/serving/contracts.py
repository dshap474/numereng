"""Typed contracts for serving and production submission packaging."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

NeutralizationMode = Literal["era", "global"]
RankMethod = Literal["average", "min", "max", "first", "dense"]


@dataclass(frozen=True)
class ServingComponentSpec:
    """One explicit component frozen into a submission package."""

    component_id: str
    weight: float
    config_path: Path | None = None
    run_id: str | None = None
    source_label: str | None = None


@dataclass(frozen=True)
class ServingBlendRule:
    """Blend rule applied to component predictions."""

    per_era_rank: bool = True
    rank_method: RankMethod = "average"
    rank_pct: bool = True
    final_rerank: bool = False


@dataclass(frozen=True)
class ServingNeutralizationSpec:
    """Optional final-step neutralization against live features."""

    enabled: bool = False
    proportion: float = 0.5
    mode: NeutralizationMode = "era"
    neutralizer_cols: tuple[str, ...] | None = None
    rank_output: bool = True


@dataclass(frozen=True)
class SubmissionPackageRecord:
    """Persisted submission package metadata."""

    package_id: str
    experiment_id: str
    tournament: str
    data_version: str
    package_path: Path
    status: str
    components: tuple[ServingComponentSpec, ...]
    blend_rule: ServingBlendRule
    neutralization: ServingNeutralizationSpec | None
    artifacts: dict[str, str] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    provenance: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ServingComponentInspection:
    """Compatibility findings for one frozen package component."""

    component_id: str
    local_live_compatible: bool
    model_upload_compatible: bool
    artifact_backed: bool = False
    artifact_ready: bool = False
    local_live_blockers: tuple[str, ...] = ()
    model_upload_blockers: tuple[str, ...] = ()
    artifact_blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ServingInspectionResult:
    """Preflight inspection result for one submission package."""

    package: SubmissionPackageRecord
    checked_at: str
    local_live_compatible: bool
    model_upload_compatible: bool
    artifact_backed: bool = False
    artifact_ready: bool = False
    artifact_live_ready: bool = False
    pickle_upload_ready: bool = False
    deployment_classification: str = "not_live_ready"
    local_live_blockers: tuple[str, ...] = ()
    model_upload_blockers: tuple[str, ...] = ()
    artifact_blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    components: tuple[ServingComponentInspection, ...] = ()
    report_path: Path | None = None


@dataclass(frozen=True)
class LiveBuildResult:
    """Artifacts produced by one local live build."""

    package: SubmissionPackageRecord
    current_round: int | None
    live_dataset_name: str
    live_benchmark_dataset_name: str | None
    live_dataset_path: Path
    live_benchmark_dataset_path: Path | None
    component_prediction_paths: tuple[Path, ...]
    blended_predictions_path: Path
    submission_predictions_path: Path


@dataclass(frozen=True)
class PickleBuildResult:
    """Artifacts produced by one pickle build."""

    package: SubmissionPackageRecord
    pickle_path: Path
    docker_image: str
    smoke_verified: bool = False


@dataclass(frozen=True)
class ModelUploadResult:
    """Result returned after uploading one model pickle."""

    package: SubmissionPackageRecord
    model_name: str
    model_id: str
    pickle_path: Path
    upload_id: str
    data_version: str | None = None
    docker_image: str | None = None


@dataclass(frozen=True)
class LiveSubmitResult:
    """Result returned after building and submitting a live parquet."""

    live_build: LiveBuildResult
    submission_id: str
    model_name: str
    model_id: str


__all__ = [
    "LiveBuildResult",
    "LiveSubmitResult",
    "ModelUploadResult",
    "ServingComponentInspection",
    "ServingBlendRule",
    "ServingComponentSpec",
    "ServingInspectionResult",
    "ServingNeutralizationSpec",
    "PickleBuildResult",
    "RankMethod",
    "SubmissionPackageRecord",
]
