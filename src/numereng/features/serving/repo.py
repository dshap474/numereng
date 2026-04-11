"""Persistence helpers for submission packages."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from numereng.features.serving.contracts import (
    NeutralizationMode,
    ServingBlendRule,
    ServingComponentSpec,
    ServingNeutralizationSpec,
    SubmissionPackageRecord,
)
from numereng.features.store import resolve_workspace_layout
from numereng.features.training.repo import (
    resolve_model_artifact_path,
    resolve_model_manifest_path,
)

_SAFE_ID = re.compile(r"^[\w\-.]+$")
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PACKAGE_FILENAME = "package.json"


class ServingValidationError(Exception):
    """Raised when serving inputs are invalid."""


class ServingPackageNotFoundError(Exception):
    """Raised when a requested submission package cannot be found."""


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def validate_safe_id(value: str, *, field: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ServingValidationError(f"serving_{field}_empty")
    if not _SAFE_ID.fullmatch(stripped):
        raise ServingValidationError(f"serving_{field}_invalid")
    return stripped


def validate_run_id(value: str) -> str:
    stripped = value.strip()
    if not _RUN_ID_PATTERN.fullmatch(stripped):
        raise ServingValidationError("serving_run_id_invalid")
    return stripped


def package_dir(*, workspace_root: str | Path, experiment_id: str, package_id: str) -> Path:
    layout = resolve_workspace_layout(workspace_root)
    return (layout.experiments_root / experiment_id / "submission_packages" / package_id).resolve()


def package_manifest_path(*, workspace_root: str | Path, experiment_id: str, package_id: str) -> Path:
    return (
        package_dir(
            workspace_root=workspace_root,
            experiment_id=experiment_id,
            package_id=package_id,
        )
        / _PACKAGE_FILENAME
    )


def run_resolved_config_path(*, workspace_root: str | Path, run_id: str) -> Path:
    layout = resolve_workspace_layout(workspace_root)
    safe_run_id = validate_run_id(run_id)
    return (layout.store_root / "runs" / safe_run_id / "resolved.json").resolve()


def run_dir(*, workspace_root: str | Path, run_id: str) -> Path:
    layout = resolve_workspace_layout(workspace_root)
    safe_run_id = validate_run_id(run_id)
    return (layout.store_root / "runs" / safe_run_id).resolve()


def run_model_artifact_path(*, workspace_root: str | Path, run_id: str) -> Path:
    return resolve_model_artifact_path(run_dir(workspace_root=workspace_root, run_id=run_id)).resolve()


def run_model_manifest_path(*, workspace_root: str | Path, run_id: str) -> Path:
    return resolve_model_manifest_path(run_dir(workspace_root=workspace_root, run_id=run_id)).resolve()


def source_config_path(
    *,
    workspace_root: str | Path,
    config_path: str | Path | None,
    run_id: str | None,
) -> Path:
    if (config_path is None) == (run_id is None):
        raise ServingValidationError("serving_component_requires_exactly_one_source")
    if config_path is not None:
        candidate = Path(config_path).expanduser().resolve()
    else:
        candidate = run_resolved_config_path(workspace_root=workspace_root, run_id=str(run_id))
    if not candidate.is_file():
        raise ServingPackageNotFoundError("serving_component_config_not_found")
    return candidate


def build_component_id(
    *,
    workspace_root: str | Path,
    source_label: str | None,
    config_path: str | Path | None,
    run_id: str | None,
) -> str:
    if source_label is not None and source_label.strip():
        return validate_safe_id(source_label, field="component_id")
    if run_id is not None:
        return validate_run_id(run_id)
    candidate = source_config_path(workspace_root=workspace_root, config_path=config_path, run_id=run_id)
    return validate_safe_id(candidate.stem, field="component_id")


def save_package(record: SubmissionPackageRecord) -> SubmissionPackageRecord:
    payload = {
        "package_id": record.package_id,
        "experiment_id": record.experiment_id,
        "tournament": record.tournament,
        "data_version": record.data_version,
        "status": record.status,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "artifacts": dict(record.artifacts),
        "blend_rule": {
            "per_era_rank": record.blend_rule.per_era_rank,
            "rank_method": record.blend_rule.rank_method,
            "rank_pct": record.blend_rule.rank_pct,
            "final_rerank": record.blend_rule.final_rerank,
        },
        "neutralization": None
        if record.neutralization is None
        else {
            "enabled": record.neutralization.enabled,
            "proportion": record.neutralization.proportion,
            "mode": record.neutralization.mode,
            "neutralizer_cols": list(record.neutralization.neutralizer_cols or ()),
            "rank_output": record.neutralization.rank_output,
        },
        "components": [
            {
                "component_id": item.component_id,
                "weight": item.weight,
                "config_path": str(item.config_path) if item.config_path is not None else None,
                "run_id": item.run_id,
                "source_label": item.source_label,
            }
            for item in record.components
        ],
        "provenance": dict(record.provenance),
    }
    record.package_path.mkdir(parents=True, exist_ok=True)
    manifest_path = record.package_path / _PACKAGE_FILENAME
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return record


def load_package(*, workspace_root: str | Path, experiment_id: str, package_id: str) -> SubmissionPackageRecord:
    manifest_path = package_manifest_path(
        workspace_root=workspace_root,
        experiment_id=experiment_id,
        package_id=package_id,
    )
    if not manifest_path.is_file():
        raise ServingPackageNotFoundError("serving_package_not_found")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    components = tuple(
        ServingComponentSpec(
            component_id=str(item["component_id"]),
            weight=float(item["weight"]),
            config_path=None if item.get("config_path") is None else Path(str(item["config_path"])).resolve(),
            run_id=None if item.get("run_id") is None else str(item["run_id"]),
            source_label=None if item.get("source_label") is None else str(item["source_label"]),
        )
        for item in payload.get("components", [])
    )
    neutralization_payload = payload.get("neutralization")
    neutralization = None
    if isinstance(neutralization_payload, dict):
        mode_value = neutralization_payload.get("mode", "era")
        mode: NeutralizationMode = "global" if str(mode_value) == "global" else "era"
        neutralization = ServingNeutralizationSpec(
            enabled=bool(neutralization_payload.get("enabled", False)),
            proportion=float(neutralization_payload.get("proportion", 0.5)),
            mode=mode,
            neutralizer_cols=tuple(str(item) for item in neutralization_payload.get("neutralizer_cols", [])) or None,
            rank_output=bool(neutralization_payload.get("rank_output", True)),
        )
    return SubmissionPackageRecord(
        package_id=str(payload["package_id"]),
        experiment_id=str(payload["experiment_id"]),
        tournament=str(payload.get("tournament", "classic")),
        data_version=str(payload.get("data_version", "v5.2")),
        package_path=manifest_path.parent.resolve(),
        status=str(payload.get("status", "created")),
        components=components,
        blend_rule=ServingBlendRule(**payload.get("blend_rule", {})),
        neutralization=neutralization,
        artifacts={str(key): str(value) for key, value in dict(payload.get("artifacts", {})).items()},
        created_at=str(payload.get("created_at", "")),
        updated_at=str(payload.get("updated_at", "")),
        provenance=dict(payload.get("provenance", {})),
    )


def list_packages(*, workspace_root: str | Path, experiment_id: str | None = None) -> list[SubmissionPackageRecord]:
    layout = resolve_workspace_layout(workspace_root)
    experiments_root = layout.experiments_root
    experiment_dirs = (
        [experiments_root / experiment_id] if experiment_id is not None else sorted(experiments_root.glob("*"))
    )
    records: list[SubmissionPackageRecord] = []
    for experiment_dir in experiment_dirs:
        packages_root = experiment_dir / "submission_packages"
        if not packages_root.is_dir():
            continue
        for package_dir_path in sorted(path for path in packages_root.iterdir() if path.is_dir()):
            try:
                records.append(
                    load_package(
                        workspace_root=workspace_root,
                        experiment_id=experiment_dir.name,
                        package_id=package_dir_path.name,
                    )
                )
            except ServingPackageNotFoundError:
                continue
    return records


__all__ = [
    "ServingPackageNotFoundError",
    "ServingValidationError",
    "build_component_id",
    "load_package",
    "list_packages",
    "package_dir",
    "package_manifest_path",
    "run_dir",
    "run_model_artifact_path",
    "run_model_manifest_path",
    "run_resolved_config_path",
    "save_package",
    "source_config_path",
    "utc_now_iso",
    "validate_run_id",
    "validate_safe_id",
]
