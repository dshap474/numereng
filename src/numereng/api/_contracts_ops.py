"""Experiment/HPO/ensemble/store/dataset API contracts."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_serializer, model_validator

from numereng.api._contracts_base import NeutralizationMode, WorkspaceBoundRequest
from numereng.config.hpo.contracts import canonicalize_hpo_sampler_payload

_SAFE_ID = re.compile(r"^[\w\-.]+$")


class HpoSearchSpaceSpecRequest(BaseModel):
    type: Literal["float", "int", "categorical"]
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    log: bool = False
    choices: list[str | int | float] | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> HpoSearchSpaceSpecRequest:
        if self.type == "categorical":
            if self.choices is None or not self.choices:
                raise ValueError("choices is required when type is categorical")
            if self.low is not None or self.high is not None or self.step is not None:
                raise ValueError("low/high/step are not allowed when type is categorical")
            return self
        if self.low is None or self.high is None:
            raise ValueError("low and high are required when type is float or int")
        if self.choices is not None:
            raise ValueError("choices is not allowed when type is float or int")
        return self


class HpoNeutralizationRequest(BaseModel):
    enabled: bool = False
    neutralizer_path: str | None = None
    proportion: float = Field(default=0.5, ge=0.0, le=1.0)
    mode: NeutralizationMode = "era"
    neutralizer_cols: list[str] | None = None
    rank_output: bool = True

    @model_validator(mode="after")
    def _validate_shape(self) -> HpoNeutralizationRequest:
        if self.enabled and self.neutralizer_path is None:
            raise ValueError("neutralizer_path is required when neutralization.enabled is true")
        return self


class HpoObjectiveRequest(BaseModel):
    metric: str = "post_fold_champion_objective"
    direction: Literal["maximize", "minimize"] = "maximize"
    neutralization: HpoNeutralizationRequest = Field(default_factory=HpoNeutralizationRequest)

    @field_validator("metric")
    @classmethod
    def _validate_metric(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("metric must not be empty")
        return stripped


class HpoSamplerRequest(BaseModel):
    kind: Literal["tpe", "random"] = "tpe"
    seed: int | None = 1337
    n_startup_trials: int = Field(default=10, ge=1)
    multivariate: bool = True
    group: bool = False

    @model_validator(mode="after")
    def _validate_shape(self) -> HpoSamplerRequest:
        if self.kind == "random":
            unsupported = {"n_startup_trials", "multivariate", "group"} & self.model_fields_set
            if unsupported:
                fields = ",".join(sorted(unsupported))
                raise ValueError(f"{fields} are not allowed when sampler.kind is random")
            return self
        if self.group and not self.multivariate:
            raise ValueError("group requires multivariate=true when sampler.kind is tpe")
        return self

    @model_serializer(mode="plain")
    def _serialize(self) -> dict[str, object]:
        return canonicalize_hpo_sampler_payload(
            {
                "kind": self.kind,
                "seed": self.seed,
                "n_startup_trials": self.n_startup_trials,
                "multivariate": self.multivariate,
                "group": self.group,
            }
        )


class HpoPlateauRequest(BaseModel):
    enabled: bool = False
    min_completed_trials: int = Field(default=15, ge=1)
    patience_completed_trials: int = Field(default=10, ge=1)
    min_improvement_abs: float = Field(default=0.00025, ge=0.0)


class HpoStoppingRequest(BaseModel):
    max_trials: int = Field(default=100, ge=1)
    max_completed_trials: int | None = Field(default=None, ge=1)
    timeout_seconds: int | None = Field(default=None, ge=1)
    plateau: HpoPlateauRequest = Field(default_factory=HpoPlateauRequest)


class HpoStudySpecResponse(BaseModel):
    study_id: str
    study_name: str
    config_path: str
    experiment_id: str | None = None
    objective: HpoObjectiveRequest
    search_space: dict[str, HpoSearchSpaceSpecRequest]
    sampler: HpoSamplerRequest
    stopping: HpoStoppingRequest


class HpoStudyCreateRequest(WorkspaceBoundRequest):
    study_id: str
    study_name: str
    config_path: str
    experiment_id: str | None = None
    objective: HpoObjectiveRequest
    search_space: dict[str, HpoSearchSpaceSpecRequest]
    sampler: HpoSamplerRequest
    stopping: HpoStoppingRequest

    @field_validator("study_id")
    @classmethod
    def _validate_study_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("study_id must not be empty")
        if not _SAFE_ID.match(stripped):
            raise ValueError("study_id must contain only letters, numbers, underscore, dash, or dot")
        return stripped

    @field_validator("study_name")
    @classmethod
    def _validate_study_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("study_name must not be empty")
        return stripped

    @field_validator("experiment_id")
    @classmethod
    def _validate_experiment_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("experiment_id must not be empty")
        if not _SAFE_ID.match(stripped):
            raise ValueError("experiment_id must contain only letters, numbers, underscore, dash, or dot")
        return stripped

    @field_validator("search_space")
    @classmethod
    def _validate_search_space(
        cls,
        value: dict[str, HpoSearchSpaceSpecRequest],
    ) -> dict[str, HpoSearchSpaceSpecRequest]:
        if not value:
            raise ValueError("search_space must not be empty")
        return value


class HpoStudyListRequest(WorkspaceBoundRequest):
    experiment_id: str | None = None
    status: str | None = None
    limit: int = Field(default=50, ge=1)
    offset: int = Field(default=0, ge=0)


class HpoStudyGetRequest(WorkspaceBoundRequest):
    study_id: str


class HpoStudyTrialsRequest(WorkspaceBoundRequest):
    study_id: str


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
    best_trial_number: int | None = None
    best_value: float | None = None
    best_run_id: str | None = None
    spec: HpoStudySpecResponse
    attempted_trials: int
    completed_trials: int
    failed_trials: int
    stop_reason: str | None = None
    storage_path: str | None = None
    error_message: str | None = None
    created_at: str
    updated_at: str


class HpoStudyListResponse(BaseModel):
    studies: list[HpoStudyResponse]


class HpoStudyTrialsResponse(BaseModel):
    study_id: str
    trials: list[HpoTrialResponse]


class BaselineBuildRequest(WorkspaceBoundRequest):
    run_ids: list[str]
    name: str = Field(min_length=1)
    default_target: str = "target_ender_20"
    description: str | None = None
    promote_active: bool = False


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


class EnsembleBuildRequest(WorkspaceBoundRequest):
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

    @model_validator(mode="after")
    def _validate_neutralization(self) -> EnsembleBuildRequest:
        if (self.neutralize_members or self.neutralize_final) and self.neutralizer_path is None:
            raise ValueError("neutralizer_path is required when neutralize_members or neutralize_final is true")
        return self


class EnsembleSelectionSourceRuleRequest(BaseModel):
    experiment_id: str
    selection_mode: Literal["explicit_targets", "top_n"]
    explicit_targets: list[str] = Field(default_factory=list)
    top_n: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_mode_fields(self) -> EnsembleSelectionSourceRuleRequest:
        if self.selection_mode == "explicit_targets":
            if not self.explicit_targets:
                raise ValueError("explicit_targets is required when selection_mode is explicit_targets")
            if self.top_n is not None:
                raise ValueError("top_n must be omitted when selection_mode is explicit_targets")
            return self
        if self.top_n is None:
            raise ValueError("top_n is required when selection_mode is top_n")
        if self.explicit_targets:
            raise ValueError("explicit_targets must be empty when selection_mode is top_n")
        return self


class EnsembleSelectRequest(WorkspaceBoundRequest):
    experiment_id: str
    source_experiment_ids: list[str]
    source_rules: list[EnsembleSelectionSourceRuleRequest]
    selection_id: str | None = None
    target: str = "target_ender_20"
    primary_metric: str = "bmc_last_200_eras.mean"
    tie_break_metric: str = "bmc.mean"
    correlation_threshold: float = Field(default=0.85, gt=0.0, le=1.0)
    top_weighted_variants: int = Field(default=2, ge=1)
    weight_step: float = Field(default=0.05, gt=0.0)
    bundle_policy: Literal["seed_avg"] = "seed_avg"
    required_seed_count: int = Field(default=1, ge=1)
    require_full_seed_bundle: bool = False
    blend_variants: list[
        Literal["all_surviving", "medium_only", "small_only", "top2_medium_top2_small", "top3_overall"]
    ] = Field(
        default_factory=lambda: [
            "all_surviving",
            "medium_only",
            "small_only",
            "top2_medium_top2_small",
            "top3_overall",
        ]
    )
    weighted_promotion_min_gain: float = Field(default=0.0005, ge=0.0)

    @model_validator(mode="after")
    def _validate_sources(self) -> EnsembleSelectRequest:
        if [rule.experiment_id for rule in self.source_rules] != self.source_experiment_ids:
            raise ValueError("source_rules experiment_id order must match source_experiment_ids exactly")
        return self


class EnsembleListRequest(WorkspaceBoundRequest):
    experiment_id: str | None = None
    limit: int = Field(default=50, ge=1)
    offset: int = Field(default=0, ge=0)


class EnsembleGetRequest(WorkspaceBoundRequest):
    ensemble_id: str


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


class EnsembleSelectionWinnerResponse(BaseModel):
    blend_id: str
    selection_mode: Literal["equal_weight", "weighted", "equal_weight_retained"]
    component_ids: list[str] = Field(default_factory=list)
    weights: list[float] = Field(default_factory=list)
    metrics: dict[str, float | None] = Field(default_factory=dict)


class EnsembleSelectResponse(BaseModel):
    selection_id: str
    experiment_id: str
    target: str
    primary_metric: str
    tie_break_metric: str
    status: str
    artifacts_path: str
    frozen_candidate_count: int
    surviving_candidate_count: int
    equal_weight_variant_count: int
    weighted_candidate_count: int
    winner: EnsembleSelectionWinnerResponse
    created_at: str
    updated_at: str


class EnsembleListResponse(BaseModel):
    ensembles: list[EnsembleResponse]


class StoreInitRequest(WorkspaceBoundRequest):
    pass


class StoreInitResponse(BaseModel):
    store_root: str
    db_path: str
    created: bool
    schema_migration: str


class StoreIndexRequest(WorkspaceBoundRequest):
    run_id: str


class StoreIndexResponse(BaseModel):
    run_id: str
    status: str
    metrics_indexed: int
    artifacts_indexed: int
    run_path: str
    warnings: list[str]


class StoreRebuildRequest(WorkspaceBoundRequest):
    pass


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


class StoreDoctorRequest(WorkspaceBoundRequest):
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


class StoreRunLifecycleRepairRequest(WorkspaceBoundRequest):
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


class StoreRunExecutionBackfillRequest(WorkspaceBoundRequest):
    run_id: str | None = None
    all: bool = False

    @model_validator(mode="after")
    def _validate_scope(self) -> StoreRunExecutionBackfillRequest:
        scope_flags = int(self.run_id is not None) + int(self.all)
        if scope_flags != 1:
            raise ValueError("exactly one of run_id or all=true is required")
        return self


class StoreRunExecutionBackfillResponse(BaseModel):
    store_root: str
    scanned_count: int
    updated_count: int
    skipped_count: int
    ambiguous_runs: list[str] = Field(default_factory=list)
    updated_run_ids: list[str] = Field(default_factory=list)
    skipped_run_ids: list[str] = Field(default_factory=list)


class StoreMaterializeVizArtifactsRequest(WorkspaceBoundRequest):
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


class ServeComponentRequest(BaseModel):
    component_id: str | None = None
    weight: float = Field(ge=0.0)
    config_path: str | None = None
    run_id: str | None = None
    source_label: str | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> ServeComponentRequest:
        scope_flags = int(self.config_path is not None) + int(self.run_id is not None)
        if scope_flags != 1:
            raise ValueError("exactly one of config_path or run_id is required")
        return self


class ServeBlendRuleRequest(BaseModel):
    per_era_rank: bool = True
    rank_method: Literal["average", "min", "max", "first", "dense"] = "average"
    rank_pct: bool = True
    final_rerank: bool = False


class ServeNeutralizationRequest(BaseModel):
    enabled: bool = False
    proportion: float = Field(default=0.5, ge=0.0, le=1.0)
    mode: NeutralizationMode = "era"
    neutralizer_cols: list[str] | None = None
    rank_output: bool = True


class ServePackageCreateRequest(WorkspaceBoundRequest):
    experiment_id: str
    package_id: str
    tournament: Literal["classic"] = "classic"
    data_version: str = "v5.2"
    components: list[ServeComponentRequest]
    blend_rule: ServeBlendRuleRequest = Field(default_factory=ServeBlendRuleRequest)
    neutralization: ServeNeutralizationRequest | None = None
    provenance: dict[str, object] = Field(default_factory=dict)

    @field_validator("experiment_id", "package_id", "data_version")
    @classmethod
    def _validate_non_empty_string(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty")
        return stripped

    @field_validator("components")
    @classmethod
    def _validate_components_nonempty(cls, value: list[ServeComponentRequest]) -> list[ServeComponentRequest]:
        if not value:
            raise ValueError("components must not be empty")
        return value


class ServePackageListRequest(WorkspaceBoundRequest):
    experiment_id: str | None = None


class ServePackageResponse(BaseModel):
    package_id: str
    experiment_id: str
    tournament: str
    data_version: str
    package_path: str
    status: str
    components: list[ServeComponentRequest]
    blend_rule: ServeBlendRuleRequest
    neutralization: ServeNeutralizationRequest | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    provenance: dict[str, object] = Field(default_factory=dict)


class ServePackageListResponse(BaseModel):
    packages: list[ServePackageResponse]


class ServeComponentInspectionResponse(BaseModel):
    component_id: str
    local_live_compatible: bool
    model_upload_compatible: bool
    artifact_backed: bool = False
    artifact_ready: bool = False
    local_live_blockers: list[str] = Field(default_factory=list)
    model_upload_blockers: list[str] = Field(default_factory=list)
    artifact_blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ServePackageInspectRequest(WorkspaceBoundRequest):
    experiment_id: str
    package_id: str


class ServePackageInspectResponse(BaseModel):
    package: ServePackageResponse
    checked_at: str
    local_live_compatible: bool
    model_upload_compatible: bool
    artifact_backed: bool = False
    artifact_ready: bool = False
    artifact_live_ready: bool = False
    pickle_upload_ready: bool = False
    deployment_classification: str = "not_live_ready"
    local_live_blockers: list[str] = Field(default_factory=list)
    model_upload_blockers: list[str] = Field(default_factory=list)
    artifact_blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    components: list[ServeComponentInspectionResponse] = Field(default_factory=list)
    report_path: str | None = None


class ServeLiveBuildRequest(WorkspaceBoundRequest):
    experiment_id: str
    package_id: str


class ServeLiveBuildResponse(BaseModel):
    package: ServePackageResponse
    current_round: int | None = None
    live_dataset_name: str
    live_benchmark_dataset_name: str | None = None
    live_dataset_path: str
    live_benchmark_dataset_path: str | None = None
    component_prediction_paths: list[str]
    blended_predictions_path: str
    submission_predictions_path: str


class ServeLiveSubmitRequest(WorkspaceBoundRequest):
    experiment_id: str
    package_id: str
    model_name: str


class ServeLiveSubmitResponse(BaseModel):
    package: ServePackageResponse
    current_round: int | None = None
    submission_id: str
    model_name: str
    model_id: str
    submission_predictions_path: str


class ServePickleBuildRequest(WorkspaceBoundRequest):
    experiment_id: str
    package_id: str


class ServePickleBuildResponse(BaseModel):
    package: ServePackageResponse
    pickle_path: str


class ServePickleUploadRequest(WorkspaceBoundRequest):
    experiment_id: str
    package_id: str
    model_name: str
    data_version: str | None = None
    docker_image: str | None = None


class ServePickleUploadResponse(BaseModel):
    package: ServePackageResponse
    pickle_path: str
    model_name: str
    model_id: str
    upload_id: str
    data_version: str | None = None
    docker_image: str | None = None


class WorkspaceInitRequest(WorkspaceBoundRequest):
    pass


class WorkspaceInitResponse(BaseModel):
    workspace_root: str
    store_root: str
    created_paths: list[str] = Field(default_factory=list)
    skipped_existing_paths: list[str] = Field(default_factory=list)
    installed_skill_ids: list[str] = Field(default_factory=list)


__all__ = [
    "BaselineBuildRequest",
    "BaselineBuildResponse",
    "DatasetToolsBuildDownsampleRequest",
    "DatasetToolsBuildDownsampleResponse",
    "EnsembleBuildRequest",
    "EnsembleComponentResponse",
    "EnsembleSelectRequest",
    "EnsembleSelectResponse",
    "EnsembleSelectionSourceRuleRequest",
    "EnsembleSelectionWinnerResponse",
    "EnsembleGetRequest",
    "EnsembleListRequest",
    "EnsembleListResponse",
    "EnsembleMetricResponse",
    "EnsembleResponse",
    "HpoNeutralizationRequest",
    "HpoObjectiveRequest",
    "HpoPlateauRequest",
    "HpoSamplerRequest",
    "HpoSearchSpaceSpecRequest",
    "HpoStoppingRequest",
    "HpoStudyCreateRequest",
    "HpoStudyGetRequest",
    "HpoStudyListRequest",
    "HpoStudyListResponse",
    "HpoStudyResponse",
    "HpoStudySpecResponse",
    "HpoStudyTrialsRequest",
    "HpoStudyTrialsResponse",
    "HpoTrialResponse",
    "ServeBlendRuleRequest",
    "ServeComponentInspectionResponse",
    "ServeComponentRequest",
    "ServeLiveBuildRequest",
    "ServeLiveBuildResponse",
    "ServeLiveSubmitRequest",
    "ServeLiveSubmitResponse",
    "ServeNeutralizationRequest",
    "ServePackageCreateRequest",
    "ServePackageInspectRequest",
    "ServePackageInspectResponse",
    "ServePackageListRequest",
    "ServePackageListResponse",
    "ServePackageResponse",
    "ServePickleBuildRequest",
    "ServePickleBuildResponse",
    "ServePickleUploadRequest",
    "ServePickleUploadResponse",
    "StoreDoctorRequest",
    "StoreDoctorResponse",
    "StoreIndexRequest",
    "StoreIndexResponse",
    "StoreInitRequest",
    "StoreInitResponse",
    "StoreMaterializeVizArtifactsFailureResponse",
    "StoreMaterializeVizArtifactsRequest",
    "StoreMaterializeVizArtifactsResponse",
    "StoreRunExecutionBackfillRequest",
    "StoreRunExecutionBackfillResponse",
    "StoreRunLifecycleRepairRequest",
    "StoreRunLifecycleRepairResponse",
    "StoreRebuildFailureResponse",
    "StoreRebuildRequest",
    "StoreRebuildResponse",
    "WorkspaceInitRequest",
    "WorkspaceInitResponse",
]
