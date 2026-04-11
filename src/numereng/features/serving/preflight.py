"""Read-only compatibility inspection for submission packages."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

from numereng.features.serving.contracts import (
    ServingComponentInspection,
    ServingInspectionResult,
    SubmissionPackageRecord,
)
from numereng.features.serving.repo import source_config_path, utc_now_iso
from numereng.features.serving.runtime import (
    ServingUnsupportedConfigError,
    load_run_backed_component,
    prepare_component_plan,
)


def inspect_submission_package(
    *,
    workspace_root: str | Path,
    package: SubmissionPackageRecord,
) -> ServingInspectionResult:
    """Classify one package for local live builds and Numerai model uploads."""

    component_reports: list[ServingComponentInspection] = []
    package_local_blockers: list[str] = []
    package_model_blockers: list[str] = []
    package_artifact_blockers: list[str] = []
    warnings: list[str] = []

    if package.tournament != "classic":
        package_local_blockers.append("serving_tournament_not_supported")
        package_model_blockers.append("serving_tournament_not_supported")
    if not package.data_version.startswith("v"):
        warnings.append("serving_data_version_unrecognized")

    for component in package.components:
        local_blockers: list[str] = []
        model_blockers: list[str] = []
        artifact_blockers: list[str] = []
        component_warnings: list[str] = []
        artifact_backed = component.run_id is not None
        artifact_ready = False
        try:
            config_path = source_config_path(
                workspace_root=workspace_root,
                config_path=component.config_path,
                run_id=component.run_id,
            )
            plan = prepare_component_plan(
                workspace_root=workspace_root,
                component=component,
                config_path=config_path,
            )
        except ServingUnsupportedConfigError as exc:
            local_blockers.append(str(exc))
            model_blockers.append(str(exc))
            component_reports.append(
                ServingComponentInspection(
                    component_id=component.component_id,
                    local_live_compatible=False,
                    model_upload_compatible=False,
                    artifact_backed=artifact_backed,
                    artifact_ready=False,
                    local_live_blockers=tuple(local_blockers),
                    model_upload_blockers=tuple(model_blockers),
                    artifact_blockers=tuple(artifact_blockers),
                )
            )
            package_local_blockers.extend(local_blockers)
            package_model_blockers.extend(model_blockers)
            package_artifact_blockers.extend(artifact_blockers)
            continue

        if plan.model_type == "LGBMRegressor" and find_spec("lightgbm") is None:
            local_blockers.append("serving_component_dependency_missing:lightgbm")
            model_blockers.append("serving_component_dependency_missing:lightgbm")
        if find_spec("cloudpickle") is None:
            model_blockers.append("serving_model_upload_dependency_missing:cloudpickle")
        if plan.baseline_predictions_path is not None:
            model_blockers.append("serving_model_upload_baseline_inputs_not_supported")
        if plan.uses_custom_module:
            model_blockers.append("serving_model_upload_custom_modules_not_supported")
        if plan.data_key.data_version != package.data_version:
            component_warnings.append("serving_package_component_data_version_mismatch")
        if artifact_backed:
            try:
                loaded = load_run_backed_component(workspace_root=workspace_root, component=component)
            except Exception as exc:
                artifact_blockers.append(str(exc))
                model_blockers.append("serving_model_upload_requires_persisted_model_artifact")
            else:
                artifact_ready = True
                if not loaded.model_upload_compatible:
                    model_blockers.append("serving_model_upload_artifact_declared_local_only")
                if loaded.uses_custom_module:
                    model_blockers.append("serving_model_upload_custom_modules_not_supported")
                if loaded.component.baseline_predictions_path is not None:
                    model_blockers.append("serving_model_upload_baseline_inputs_not_supported")
                if package.data_version != plan.data_key.data_version:
                    component_warnings.append("serving_package_component_data_version_mismatch")
        else:
            artifact_blockers.append("serving_component_config_backed_only")
            model_blockers.append("serving_model_upload_requires_persisted_model_artifact")

        component_reports.append(
            ServingComponentInspection(
                component_id=component.component_id,
                local_live_compatible=not local_blockers,
                model_upload_compatible=not model_blockers and artifact_ready,
                artifact_backed=artifact_backed,
                artifact_ready=artifact_ready,
                local_live_blockers=tuple(local_blockers),
                model_upload_blockers=tuple(model_blockers),
                artifact_blockers=tuple(artifact_blockers),
                warnings=tuple(component_warnings),
            )
        )
        package_local_blockers.extend(local_blockers)
        package_model_blockers.extend(model_blockers)
        package_artifact_blockers.extend(artifact_blockers)
        warnings.extend(component_warnings)

    local_live_compatible = not package_local_blockers and all(item.local_live_compatible for item in component_reports)
    artifact_backed = bool(component_reports) and all(item.artifact_backed for item in component_reports)
    artifact_ready = bool(component_reports) and all(item.artifact_ready for item in component_reports)
    artifact_live_ready = local_live_compatible and artifact_backed and artifact_ready
    model_upload_compatible = (
        artifact_live_ready
        and not package_model_blockers
        and all(item.model_upload_compatible for item in component_reports)
    )
    deployment_classification = _deployment_classification(
        local_live_compatible=local_live_compatible,
        artifact_live_ready=artifact_live_ready,
        pickle_upload_ready=model_upload_compatible,
    )
    return ServingInspectionResult(
        package=package,
        checked_at=utc_now_iso(),
        local_live_compatible=local_live_compatible,
        model_upload_compatible=model_upload_compatible,
        artifact_backed=artifact_backed,
        artifact_ready=artifact_ready,
        artifact_live_ready=artifact_live_ready,
        pickle_upload_ready=model_upload_compatible,
        deployment_classification=deployment_classification,
        local_live_blockers=tuple(_dedupe(package_local_blockers)),
        model_upload_blockers=tuple(_dedupe(package_model_blockers)),
        artifact_blockers=tuple(_dedupe(package_artifact_blockers)),
        warnings=tuple(_dedupe(warnings)),
        components=tuple(component_reports),
        report_path=None,
    )


def inspection_payload(result: ServingInspectionResult) -> dict[str, object]:
    """Convert one inspection result into a stable JSON payload."""

    return {
        "checked_at": result.checked_at,
        "local_live_compatible": result.local_live_compatible,
        "model_upload_compatible": result.model_upload_compatible,
        "artifact_backed": result.artifact_backed,
        "artifact_ready": result.artifact_ready,
        "artifact_live_ready": result.artifact_live_ready,
        "pickle_upload_ready": result.pickle_upload_ready,
        "deployment_classification": result.deployment_classification,
        "local_live_blockers": list(result.local_live_blockers),
        "model_upload_blockers": list(result.model_upload_blockers),
        "artifact_blockers": list(result.artifact_blockers),
        "warnings": list(result.warnings),
        "components": [
            {
                "component_id": item.component_id,
                "local_live_compatible": item.local_live_compatible,
                "model_upload_compatible": item.model_upload_compatible,
                "artifact_backed": item.artifact_backed,
                "artifact_ready": item.artifact_ready,
                "local_live_blockers": list(item.local_live_blockers),
                "model_upload_blockers": list(item.model_upload_blockers),
                "artifact_blockers": list(item.artifact_blockers),
                "warnings": list(item.warnings),
            }
            for item in result.components
        ],
    }


def _deployment_classification(
    *,
    local_live_compatible: bool,
    artifact_live_ready: bool,
    pickle_upload_ready: bool,
) -> str:
    if not local_live_compatible:
        return "not_live_ready"
    if pickle_upload_ready:
        return "pickle_upload_ready"
    if artifact_live_ready:
        return "artifact_backed_live_ready"
    return "local_live_only"


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        ordered.append(item)
        seen.add(item)
    return ordered


__all__ = ["inspect_submission_package", "inspection_payload"]
