"""Experiment/HPO/ensemble/store/dataset API contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from numereng.api._contracts_base import NeutralizationMode


class HpoStudyCreateRequest(BaseModel):
    study_name: str
    config_path: str
    experiment_id: str | None = None
    metric: str = "post_fold_champion_objective"
    direction: Literal["maximize", "minimize"] = "maximize"
    n_trials: int = Field(default=100, ge=1)
    sampler: Literal["tpe", "random"] = "tpe"
    seed: int | None = 1337
    search_space: dict[str, dict[str, object]] | None = None
    neutralize: bool = False
    neutralizer_path: str | None = None
    neutralization_proportion: float = Field(default=0.5, ge=0.0, le=1.0)
    neutralization_mode: NeutralizationMode = "era"
    neutralizer_cols: list[str] | None = None
    neutralization_rank_output: bool = True
    store_root: str = ".numereng"

    @model_validator(mode="after")
    def _validate_neutralization(self) -> HpoStudyCreateRequest:
        if self.neutralize and self.neutralizer_path is None:
            raise ValueError("neutralizer_path is required when neutralize is true")
        return self


class HpoStudyListRequest(BaseModel):
    experiment_id: str | None = None
    status: str | None = None
    limit: int = Field(default=50, ge=1)
    offset: int = Field(default=0, ge=0)
    store_root: str = ".numereng"


class HpoStudyGetRequest(BaseModel):
    study_id: str
    store_root: str = ".numereng"


class HpoStudyTrialsRequest(BaseModel):
    study_id: str
    store_root: str = ".numereng"


class HpoTrialResponse(BaseModel):
    study_id: str
    trial_number: int
    status: str
    value: float | None = None
    run_id: str | None = None
    config_path: str | None = None
    params: dict[str, object] = Field(default_factory=dict)
    error_message: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    updated_at: str


class HpoStudyResponse(BaseModel):
    study_id: str
    experiment_id: str | None = None
    study_name: str
    status: str
    metric: str
    direction: Literal["maximize", "minimize"]
    n_trials: int
    sampler: Literal["tpe", "random"]
    seed: int | None = None
    best_trial_number: int | None = None
    best_value: float | None = None
    best_run_id: str | None = None
    config: dict[str, object] = Field(default_factory=dict)
    storage_path: str | None = None
    error_message: str | None = None
    created_at: str
    updated_at: str


class HpoStudyListResponse(BaseModel):
    studies: list[HpoStudyResponse]


class HpoStudyTrialsResponse(BaseModel):
    study_id: str
    trials: list[HpoTrialResponse]


class BaselineBuildRequest(BaseModel):
    run_ids: list[str]
    name: str = Field(min_length=1)
    default_target: str = "target_ender_20"
    description: str | None = None
    promote_active: bool = False
    store_root: str = ".numereng"


class BaselineBuildResponse(BaseModel):
    name: str
    baseline_dir: str
    predictions_path: str
    metadata_path: str
    available_targets: list[str]
    default_target: str
    source_run_ids: list[str]
    source_experiment_id: str | None = None
    active_predictions_path: str | None = None
    active_metadata_path: str | None = None
    created_at: str


class EnsembleBuildRequest(BaseModel):
    run_ids: list[str]
    experiment_id: str | None = None
    method: Literal["rank_avg"] = "rank_avg"
    metric: str = "corr_sharpe"
    target: str = "target_ender_20"
    name: str | None = None
    ensemble_id: str | None = None
    weights: list[float] | None = None
    optimize_weights: bool = False
    include_heavy_artifacts: bool = False
    selection_note: str | None = None
    regime_buckets: int = Field(default=4, ge=2, le=50)
    neutralize_members: bool = False
    neutralize_final: bool = False
    neutralizer_path: str | None = None
    neutralization_proportion: float = Field(default=0.5, ge=0.0, le=1.0)
    neutralization_mode: NeutralizationMode = "era"
    neutralizer_cols: list[str] | None = None
    neutralization_rank_output: bool = True
    store_root: str = ".numereng"

    @model_validator(mode="after")
    def _validate_neutralization(self) -> EnsembleBuildRequest:
        if (self.neutralize_members or self.neutralize_final) and self.neutralizer_path is None:
            raise ValueError("neutralizer_path is required when neutralize_members or neutralize_final is true")
        return self


class EnsembleListRequest(BaseModel):
    experiment_id: str | None = None
    limit: int = Field(default=50, ge=1)
    offset: int = Field(default=0, ge=0)
    store_root: str = ".numereng"


class EnsembleGetRequest(BaseModel):
    ensemble_id: str
    store_root: str = ".numereng"


class EnsembleComponentResponse(BaseModel):
    run_id: str
    weight: float
    rank: int


class EnsembleMetricResponse(BaseModel):
    name: str
    value: float | None = None


class EnsembleResponse(BaseModel):
    ensemble_id: str
    experiment_id: str | None = None
    name: str
    method: Literal["rank_avg"]
    target: str
    metric: str
    status: str
    components: list[EnsembleComponentResponse] = Field(default_factory=list)
    metrics: list[EnsembleMetricResponse] = Field(default_factory=list)
    artifacts_path: str | None = None
    config: dict[str, object] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class EnsembleListResponse(BaseModel):
    ensembles: list[EnsembleResponse]


class StoreInitRequest(BaseModel):
    store_root: str = ".numereng"


class StoreInitResponse(BaseModel):
    store_root: str
    db_path: str
    created: bool
    schema_migration: str


class StoreIndexRequest(BaseModel):
    run_id: str
    store_root: str = ".numereng"


class StoreIndexResponse(BaseModel):
    run_id: str
    status: str
    metrics_indexed: int
    artifacts_indexed: int
    run_path: str
    warnings: list[str]


class StoreRebuildRequest(BaseModel):
    store_root: str = ".numereng"


class StoreRebuildFailureResponse(BaseModel):
    run_id: str
    error: str


class StoreRebuildResponse(BaseModel):
    store_root: str
    db_path: str
    scanned_runs: int
    indexed_runs: int
    failed_runs: int
    failures: list[StoreRebuildFailureResponse]


class StoreDoctorRequest(BaseModel):
    store_root: str = ".numereng"
    fix_strays: bool = False


class StoreDoctorResponse(BaseModel):
    store_root: str
    db_path: str
    ok: bool
    issues: list[str]
    stats: dict[str, int]
    stray_cleanup_applied: bool = False
    deleted_paths: list[str] = Field(default_factory=list)
    missing_paths: list[str] = Field(default_factory=list)


class StoreRunLifecycleRepairRequest(BaseModel):
    store_root: str = ".numereng"
    run_id: str | None = None
    active_only: bool = True


class StoreRunLifecycleRepairResponse(BaseModel):
    store_root: str
    scanned_count: int
    unchanged_count: int
    reconciled_count: int
    reconciled_stale_count: int
    reconciled_canceled_count: int
    run_ids: list[str] = Field(default_factory=list)


class StoreMaterializeVizArtifactsRequest(BaseModel):
    store_root: str = ".numereng"
    kind: str
    run_id: str | None = None
    experiment_id: str | None = None
    all: bool = False

    @model_validator(mode="after")
    def _validate_scope(self) -> StoreMaterializeVizArtifactsRequest:
        scope_flags = int(self.run_id is not None) + int(self.experiment_id is not None) + int(self.all)
        if scope_flags != 1:
            raise ValueError("exactly one of run_id, experiment_id, or all=true is required")
        return self


class StoreMaterializeVizArtifactsFailureResponse(BaseModel):
    run_id: str
    error: str


class StoreMaterializeVizArtifactsResponse(BaseModel):
    store_root: str
    kind: str
    scoped_run_count: int
    created_count: int
    skipped_count: int
    failed_count: int
    failures: list[StoreMaterializeVizArtifactsFailureResponse]


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


__all__ = [
    "BaselineBuildRequest",
    "BaselineBuildResponse",
    "DatasetToolsBuildDownsampleRequest",
    "DatasetToolsBuildDownsampleResponse",
    "EnsembleBuildRequest",
    "EnsembleComponentResponse",
    "EnsembleGetRequest",
    "EnsembleListRequest",
    "EnsembleListResponse",
    "EnsembleMetricResponse",
    "EnsembleResponse",
    "HpoStudyCreateRequest",
    "HpoStudyGetRequest",
    "HpoStudyListRequest",
    "HpoStudyListResponse",
    "HpoStudyResponse",
    "HpoStudyTrialsRequest",
    "HpoStudyTrialsResponse",
    "HpoTrialResponse",
    "StoreDoctorRequest",
    "StoreDoctorResponse",
    "StoreIndexRequest",
    "StoreIndexResponse",
    "StoreInitRequest",
    "StoreInitResponse",
    "StoreMaterializeVizArtifactsFailureResponse",
    "StoreMaterializeVizArtifactsRequest",
    "StoreMaterializeVizArtifactsResponse",
    "StoreRunLifecycleRepairRequest",
    "StoreRunLifecycleRepairResponse",
    "StoreRebuildFailureResponse",
    "StoreRebuildRequest",
    "StoreRebuildResponse",
]
