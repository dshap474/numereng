"""Canonical training config contracts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

TrainingProfile = Literal["simple", "purged_walk_forward", "full_history_refit"]
TrainingEngineMode = Literal["official", "custom", "full_history"]
DataLoadingMode = Literal["materialized", "fold_lazy"]
ScoringMode = Literal["materialized", "era_stream"]
DatasetScope = Literal["train_only", "train_plus_validation"]
DatasetVariant = Literal["non_downsampled", "downsampled"]
ParallelBackend = Literal["joblib"]
CacheMode = Literal["deterministic"]
ModelDevice = Literal["cpu", "cuda"]


class _StrictConfigModel(BaseModel):
    """Base model that rejects unknown config keys."""

    model_config = ConfigDict(extra="forbid")


class DataLoadingConfig(_StrictConfigModel):
    """Runtime loading/scoring controls for training and metrics."""

    mode: DataLoadingMode = "materialized"
    scoring_mode: ScoringMode = "materialized"
    era_chunk_size: int = Field(default=64, ge=1)
    include_feature_neutral_metrics: bool = True


class DataConfig(_StrictConfigModel):
    """Data loading and target-selection settings."""

    data_version: str = "v5.2"
    dataset_variant: DatasetVariant
    feature_set: str = "small"
    target_col: str = "target"
    scoring_targets: list[str] | None = None
    target_horizon: Literal["20d", "60d"] | None = None
    era_col: str = "era"
    id_col: str = "id"
    full_data_path: str | None = None
    benchmark_data_path: str | None = None
    meta_model_data_path: str | None = None
    meta_model_col: str = "numerai_meta_model"
    embargo_eras: int | None = Field(default=None, ge=0)
    benchmark_model: str = "v52_lgbm_ender20"
    baselines_dir: str | None = None
    dataset_scope: DatasetScope = "train_only"
    loading: DataLoadingConfig = Field(default_factory=DataLoadingConfig)


class PreprocessingConfig(_StrictConfigModel):
    """Preprocessing settings applied before model fitting."""

    nan_missing_all_twos: bool = False
    missing_value: float = 2.0


class ModelBaselineConfig(_StrictConfigModel):
    """Optional baseline prediction source used as an input feature."""

    name: str
    predictions_path: str
    pred_col: str = "prediction"


class ModelConfig(_StrictConfigModel):
    """Model identity, hyperparameters, and optional feature options."""

    type: str
    params: dict[str, object]
    device: ModelDevice | None = None
    x_groups: list[str] | None = None
    data_needed: list[str] | None = None
    module_path: str | None = None
    target_transform: dict[str, object] | None = None
    benchmark: dict[str, object] | None = None
    baseline: ModelBaselineConfig | None = None

    @field_validator("x_groups", "data_needed")
    @classmethod
    def _validate_input_groups(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        normalized = [str(item) for item in value]
        for item in normalized:
            if item in {"era", "id"}:
                raise ValueError(f"training_model_x_groups_not_supported:{item}")
            if item in {"benchmark", "benchmarks", "benchmark_models"}:
                raise ValueError("training_model_x_groups_benchmark_not_supported")
        return normalized

    @field_validator("device", mode="before")
    @classmethod
    def _normalize_device(cls, value: object) -> object:
        if value is None:
            return value
        normalized = str(value).strip().lower()
        if normalized not in {"cpu", "cuda"}:
            raise ValueError("training_model_device_invalid")
        return normalized

    @field_validator("params", mode="after")
    @classmethod
    def _validate_params_mapping(cls, value: dict[str, object]) -> dict[str, object]:
        raw_device = value.get("device_type")
        if raw_device is None:
            return value
        normalized = str(raw_device).strip().lower()
        if normalized not in {"cpu", "gpu", "cuda"}:
            raise ValueError("training_model_params_device_type_invalid")
        resolved = dict(value)
        resolved["device_type"] = normalized
        return resolved


class TrainingEngineConfig(_StrictConfigModel):
    """Training profile settings."""

    profile: TrainingProfile | None = None
    # Legacy compatibility keys retained for migration.
    mode: TrainingEngineMode | None = None
    window_size_eras: int | None = Field(default=None, ge=1)
    embargo_eras: int | None = Field(default=None, ge=1)

    @field_validator("profile", mode="before")
    @classmethod
    def _reject_submission_profile(cls, value: object) -> object:
        if value is not None and str(value) == "submission":
            raise ValueError("training profile 'submission' was renamed to 'full_history_refit'")
        return value


class TrainingResourcesConfig(_StrictConfigModel):
    """Parallelism and memory controls for fold execution."""

    parallel_folds: int = Field(default=1, ge=1)
    parallel_backend: ParallelBackend = "joblib"
    memmap_enabled: bool = True
    max_threads_per_worker: Annotated[int, Field(ge=1)] | Literal["default"] | None = "default"
    sklearn_working_memory_mib: int | None = Field(default=None, ge=1)


class TrainingCacheConfig(_StrictConfigModel):
    """Deterministic cache controls for fold/runtime artifacts."""

    mode: CacheMode = "deterministic"
    cache_fold_specs: bool = True
    cache_features: bool = True
    cache_labels: bool = True
    cache_fold_matrices: bool = False


class TrainingSectionConfig(_StrictConfigModel):
    """Training controls for engine behavior and execution policy."""

    engine: TrainingEngineConfig = Field(default_factory=TrainingEngineConfig)
    resources: TrainingResourcesConfig = Field(default_factory=TrainingResourcesConfig)
    cache: TrainingCacheConfig = Field(default_factory=TrainingCacheConfig)


class OutputConfig(_StrictConfigModel):
    """Optional output path and artifact naming overrides."""

    output_dir: str | None = None
    baselines_dir: str | None = None
    predictions_name: str | None = None
    results_name: str | None = None


class TrainingConfig(_StrictConfigModel):
    """Canonical top-level training config contract."""

    data: DataConfig
    model: ModelConfig
    training: TrainingSectionConfig
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


__all__ = [
    "CacheMode",
    "DataConfig",
    "DataLoadingConfig",
    "DataLoadingMode",
    "DatasetScope",
    "DatasetVariant",
    "ModelDevice",
    "ModelBaselineConfig",
    "ModelConfig",
    "OutputConfig",
    "ParallelBackend",
    "PreprocessingConfig",
    "ScoringMode",
    "TrainingCacheConfig",
    "TrainingConfig",
    "TrainingEngineConfig",
    "TrainingEngineMode",
    "TrainingProfile",
    "TrainingResourcesConfig",
    "TrainingSectionConfig",
]
