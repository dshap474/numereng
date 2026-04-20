"""Serving orchestration for submission packages, live builds, and model uploads."""

from __future__ import annotations

import gc
import json
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Literal, Protocol

import pandas as pd

from numereng.features.serving.contracts import (
    LiveBuildResult,
    LiveSubmitResult,
    ModelUploadResult,
    PickleBuildResult,
    ServingBlendRule,
    ServingComponentSpec,
    ServingInspectionResult,
    ServingNeutralizationSpec,
    SubmissionPackageRecord,
)
from numereng.features.serving.preflight import inspect_submission_package, inspection_payload
from numereng.features.serving.repo import (
    ServingValidationError,
    build_component_id,
    list_packages,
    load_package,
    package_dir,
    save_package,
    source_config_path,
    utc_now_iso,
    validate_safe_id,
)
from numereng.features.serving.runtime import (
    FittedComponent,
    PreparedServingData,
    ServingDataClient,
    ServingDataContextKey,
    ServingPredictionMember,
    ServingRuntimeError,
    ServingUnsupportedConfigError,
    blend_component_predictions,
    build_pickled_predictor,
    fit_component,
    load_run_backed_component,
    predict_component_live,
    prediction_member_from_fitted,
    prepare_component_plan,
    prepare_training_context,
    write_blended_predictions,
    write_component_predictions,
)
from numereng.features.submission import submit_predictions_file, upload_model_pickle_file
from numereng.features.submission.client import SubmissionClient
from numereng.platform.numerai_client import NumeraiClient

NumeraiTournament = Literal["classic"]
_DEFAULT_MODEL_UPLOAD_DOCKER_IMAGE = "Python 3.12"

_LIVE_DATASET_PATTERN = re.compile(r"^v(?P<major>\d+)(?:\.(?P<minor>\d+))?(?:\.(?P<patch>\d+))?/live\.parquet$")
_LIVE_BENCHMARK_PATTERN = re.compile(
    r"^v(?P<major>\d+)(?:\.(?P<minor>\d+))?(?:\.(?P<patch>\d+))?/live_benchmark_models\.parquet$"
)


class ServingClient(ServingDataClient, SubmissionClient, Protocol):
    """Client surface required by serving workflows."""

    def list_datasets(self, round_num: int | None = None) -> list[str]:
        """Return dataset names."""

    def get_current_round(self) -> int | None:
        """Return current round when available."""

    def get_models(self) -> dict[str, str]:
        """Return model id mapping."""

    def model_upload(
        self,
        *,
        file_path: str,
        model_id: str,
        data_version: str | None = None,
        docker_image: str | None = None,
    ) -> str:
        """Upload one Numerai model pickle."""

    def model_upload_data_versions(self) -> list[str]:
        """Return supported model-upload data versions."""

    def model_upload_docker_images(self) -> list[str]:
        """Return supported model-upload docker images."""

    def diagnostics(self, *, model_id: str, diagnostics_id: str | None = None) -> dict[str, object]:
        """Return one normalized diagnostics payload."""

    def compute_pickle_status(
        self,
        *,
        pickle_id: str,
        model_id: str | None = None,
        unassigned: bool = False,
    ) -> dict[str, object] | None:
        """Return one upload-specific compute status payload."""

    def compute_pickle_diagnostics_logs(self, *, pickle_id: str) -> list[dict[str, object]]:
        """Return diagnostics-trigger logs for one upload."""


def create_submission_package(
    *,
    workspace_root: str | Path,
    experiment_id: str,
    package_id: str,
    components: tuple[ServingComponentSpec, ...],
    data_version: str = "v5.2",
    tournament: NumeraiTournament = "classic",
    blend_rule: ServingBlendRule | None = None,
    neutralization: ServingNeutralizationSpec | None = None,
    provenance: dict[str, object] | None = None,
) -> SubmissionPackageRecord:
    """Persist one explicit serving package under an experiment root."""

    safe_experiment_id = validate_safe_id(experiment_id, field="experiment_id")
    safe_package_id = validate_safe_id(package_id, field="package_id")
    if tournament != "classic":
        raise ServingValidationError("serving_tournament_not_supported")
    resolved_components = _resolve_components(workspace_root=workspace_root, components=components)
    created_at = utc_now_iso()
    record = SubmissionPackageRecord(
        package_id=safe_package_id,
        experiment_id=safe_experiment_id,
        tournament=tournament,
        data_version=data_version.strip() or "v5.2",
        package_path=package_dir(
            workspace_root=workspace_root, experiment_id=safe_experiment_id, package_id=safe_package_id
        ),
        status="created",
        components=resolved_components,
        blend_rule=blend_rule or ServingBlendRule(),
        neutralization=neutralization,
        artifacts={},
        created_at=created_at,
        updated_at=created_at,
        provenance=dict(provenance or {}),
    )
    return save_package(record)


def list_submission_packages(
    *,
    workspace_root: str | Path,
    experiment_id: str | None = None,
) -> list[SubmissionPackageRecord]:
    """List persisted submission packages."""

    return list_packages(workspace_root=workspace_root, experiment_id=experiment_id)


def inspect_package(
    *,
    workspace_root: str | Path,
    experiment_id: str,
    package_id: str,
) -> ServingInspectionResult:
    """Inspect one package for local-live and model-upload compatibility."""

    package = load_package(workspace_root=workspace_root, experiment_id=experiment_id, package_id=package_id)
    package = _normalize_existing_package_components(package=package, workspace_root=workspace_root)
    return _inspect_and_persist_package(package=package, workspace_root=workspace_root)


def build_live_submission_package(
    *,
    workspace_root: str | Path,
    experiment_id: str,
    package_id: str,
    client: ServingClient | None = None,
) -> LiveBuildResult:
    """Fit package components locally and emit a submit-ready live parquet."""

    package = load_package(workspace_root=workspace_root, experiment_id=experiment_id, package_id=package_id)
    package = _normalize_existing_package_components(package=package, workspace_root=workspace_root)
    inspection = _inspect_and_persist_package(package=package, workspace_root=workspace_root)
    if not inspection.local_live_compatible:
        raise ServingUnsupportedConfigError("serving_live_preflight_failed")
    package = inspection.package
    numerai_client = create_serving_client() if client is None else client
    try:
        current_round = numerai_client.get_current_round()
        live_name, benchmark_name = _resolve_live_dataset_names(client=numerai_client)
        datasets_dir = package.package_path / "artifacts" / "datasets" / _round_token(current_round)
        live_path = _download_dataset(
            client=numerai_client, dataset_name=live_name, destination=datasets_dir / "live.parquet"
        )
        benchmark_path = None
        if benchmark_name is not None:
            benchmark_path = _download_dataset(
                client=numerai_client,
                dataset_name=benchmark_name,
                destination=datasets_dir / "live_benchmark_models.parquet",
            )
        live_features = pd.read_parquet(live_path)
        component_predictions = _fit_and_predict_package(
            workspace_root=workspace_root,
            client=numerai_client,
            package=package,
            live_features=live_features,
        )
        live_dir = package.package_path / "artifacts" / "live" / _round_token(current_round)
        component_paths = write_component_predictions(
            component_predictions=component_predictions, component_dir=live_dir / "components"
        )
        internal, submission = blend_component_predictions(
            component_predictions=component_predictions,
            live_features=live_features,
            blend_rule=package.blend_rule,
            neutralization=package.neutralization,
        )
        blended_path, submission_path = write_blended_predictions(
            internal=internal, submission=submission, live_dir=live_dir
        )
        updated = _save_package_update(
            package,
            status="live_built",
            artifacts={
                "live_dataset_name": live_name,
                "live_dataset_path": str(live_path),
                "live_benchmark_dataset_name": benchmark_name,
                "live_benchmark_dataset_path": None if benchmark_path is None else str(benchmark_path),
                "live_blended_predictions_path": str(blended_path),
                "live_submission_predictions_path": str(submission_path),
                "current_round": None if current_round is None else str(current_round),
                "component_count": str(len(component_predictions)),
            },
        )
        return LiveBuildResult(
            package=updated,
            current_round=current_round,
            live_dataset_name=live_name,
            live_benchmark_dataset_name=benchmark_name,
            live_dataset_path=live_path,
            live_benchmark_dataset_path=benchmark_path,
            component_prediction_paths=component_paths,
            blended_predictions_path=blended_path,
            submission_predictions_path=submission_path,
        )
    except Exception as exc:
        _save_failure(package=package, stage="live_build", exc=exc)
        raise


def submit_live_package(
    *,
    workspace_root: str | Path,
    experiment_id: str,
    package_id: str,
    model_name: str,
    client: ServingClient | None = None,
) -> LiveSubmitResult:
    """Build a live parquet from one package and submit it."""

    numerai_client = create_serving_client() if client is None else client
    live_build = build_live_submission_package(
        workspace_root=workspace_root,
        experiment_id=experiment_id,
        package_id=package_id,
        client=numerai_client,
    )
    try:
        submission = submit_predictions_file(
            predictions_path=live_build.submission_predictions_path,
            model_name=model_name,
            tournament="classic",
            client=numerai_client,
        )
    except Exception as exc:
        _save_failure(package=live_build.package, stage="live_submit", exc=exc)
        raise
    updated = _save_package_update(
        live_build.package,
        status="live_submitted",
        artifacts={
            "last_submission_id": submission.submission_id,
            "last_submission_model_name": submission.model_name,
            "last_submission_model_id": submission.model_id,
        },
    )
    return LiveSubmitResult(
        live_build=replace(live_build, package=updated),
        submission_id=submission.submission_id,
        model_name=submission.model_name,
        model_id=submission.model_id,
    )


def build_submission_pickle(
    *,
    workspace_root: str | Path,
    experiment_id: str,
    package_id: str,
    docker_image: str | None = None,
    client: ServingClient | None = None,
) -> PickleBuildResult:
    """Fit package components and serialize one Numerai-compatible predict callable."""

    package = load_package(workspace_root=workspace_root, experiment_id=experiment_id, package_id=package_id)
    package = _normalize_existing_package_components(package=package, workspace_root=workspace_root)
    inspection = _inspect_and_persist_package(package=package, workspace_root=workspace_root)
    package = inspection.package
    resolved_docker_image = (
        docker_image or package.artifacts.get("pickle_runtime_docker_image") or _DEFAULT_MODEL_UPLOAD_DOCKER_IMAGE
    )
    _ = client
    try:
        fitted_components = _load_pickle_compatible_components(workspace_root=workspace_root, package=package)
        pickle_path = build_pickled_predictor(
            fitted_components=fitted_components,
            blend_rule=package.blend_rule,
            neutralization=package.neutralization,
            pickle_path=package.package_path / "artifacts" / "pickle" / "model.pkl",
        )
        smoke_details = _verify_isolated_pickle_runtime(
            pickle_path=pickle_path,
            fitted_components=fitted_components,
            docker_image=resolved_docker_image,
            working_dir=package.package_path / "artifacts" / "pickle",
        )
        updated = _save_package_update(
            package,
            status="pickle_built",
            artifacts={
                "pickle_path": str(pickle_path),
                "pickle_size_bytes": str(pickle_path.stat().st_size),
                "pickle_component_count": str(len(fitted_components)),
                "pickle_uses_baseline_inputs": str(
                    any(item.baseline_predictions_path is not None for item in fitted_components)
                ).lower(),
                "pickle_runtime_docker_image": resolved_docker_image,
                "pickle_smoke_verified": "true",
                "pickle_smoke_checked_at": smoke_details["checked_at"],
                "pickle_smoke_command": smoke_details["command"],
                "pickle_smoke_runtime": smoke_details["runtime"],
            },
        )
        refreshed = _inspect_and_persist_package(package=updated, workspace_root=workspace_root)
        return PickleBuildResult(
            package=refreshed.package,
            pickle_path=pickle_path,
            docker_image=resolved_docker_image,
            smoke_verified=True,
        )
    except Exception as exc:
        _save_package_update(
            package,
            status=package.status,
            artifacts={
                "pickle_runtime_docker_image": resolved_docker_image,
                "pickle_smoke_verified": "false",
                "pickle_smoke_checked_at": utc_now_iso(),
            },
        )
        _save_failure(package=package, stage="pickle_build", exc=exc)
        raise


def upload_submission_pickle(
    *,
    workspace_root: str | Path,
    experiment_id: str,
    package_id: str,
    model_name: str,
    data_version: str | None = None,
    docker_image: str | None = None,
    client: ServingClient | None = None,
) -> ModelUploadResult:
    """Build and upload one model pickle for Numerai-hosted inference."""

    numerai_client = create_serving_client() if client is None else client
    built = build_submission_pickle(
        workspace_root=workspace_root,
        experiment_id=experiment_id,
        package_id=package_id,
        docker_image=docker_image,
        client=numerai_client,
    )
    resolved_data_version = data_version or built.package.data_version
    resolved_docker_image = docker_image or built.docker_image
    _validate_model_upload_options(
        client=numerai_client,
        data_version=resolved_data_version,
        docker_image=resolved_docker_image,
    )
    _ensure_pickle_upload_ready(package=built.package, docker_image=resolved_docker_image)
    try:
        upload = upload_model_pickle_file(
            pickle_path=built.pickle_path,
            model_name=model_name,
            tournament="classic",
            data_version=resolved_data_version,
            docker_image=resolved_docker_image,
            client=numerai_client,
        )
    except Exception as exc:
        _save_failure(package=built.package, stage="pickle_upload", exc=exc)
        raise
    updated = _save_package_update(
        built.package,
        status="pickle_uploaded",
        artifacts={
            "last_pickle_upload_id": upload.upload_id,
            "last_pickle_model_name": upload.model_name,
            "last_pickle_model_id": upload.model_id,
        },
    )
    return ModelUploadResult(
        package=updated,
        model_name=upload.model_name,
        model_id=upload.model_id,
        pickle_path=upload.pickle_path,
        upload_id=upload.upload_id,
        data_version=resolved_data_version,
        docker_image=upload.docker_image,
    )


def create_serving_client() -> ServingClient:
    """Create the default Numerai Classic serving client."""

    return NumeraiClient(tournament="classic")


def _resolve_components(
    *,
    workspace_root: str | Path,
    components: tuple[ServingComponentSpec, ...],
) -> tuple[ServingComponentSpec, ...]:
    if not components:
        raise ServingValidationError("serving_components_empty")
    normalized: list[ServingComponentSpec] = []
    seen_ids: set[str] = set()
    weight_total = 0.0
    for component in components:
        if float(component.weight) <= 0.0:
            continue
        component_id = build_component_id(
            workspace_root=workspace_root,
            source_label=component.source_label or component.component_id,
            config_path=component.config_path,
            run_id=component.run_id,
        )
        if component_id in seen_ids:
            raise ServingValidationError("serving_component_ids_duplicate")
        seen_ids.add(component_id)
        if component.weight < 0.0:
            raise ServingValidationError("serving_component_weight_negative")
        weight_total += component.weight
        resolved_source_path = source_config_path(
            workspace_root=workspace_root,
            config_path=component.config_path,
            run_id=component.run_id,
        )
        normalized.append(
            ServingComponentSpec(
                component_id=component_id,
                weight=float(component.weight),
                config_path=resolved_source_path if component.config_path is not None else None,
                run_id=component.run_id,
                source_label=component.source_label,
            )
        )
    if not normalized:
        raise ServingValidationError("serving_components_empty")
    if abs(weight_total - 1.0) > 1e-6:
        raise ServingValidationError("serving_component_weights_must_sum_to_one")
    return tuple(normalized)


def _fit_and_predict_package(
    *,
    workspace_root: str | Path,
    client: ServingClient,
    package: SubmissionPackageRecord,
    live_features: pd.DataFrame,
) -> list[tuple[ServingPredictionMember, pd.DataFrame]]:
    component_predictions: list[tuple[ServingPredictionMember, pd.DataFrame]] = []
    pending_retrain: list[ServingComponentSpec] = []
    for component in package.components:
        loaded = _maybe_load_artifact_backed_component(
            workspace_root=workspace_root,
            component=component,
        )
        if loaded is None:
            pending_retrain.append(component)
            continue
        component_predictions.append(
            (
                prediction_member_from_fitted(loaded),
                predict_component_live(component=loaded, live_features=live_features),
            )
        )

    if not pending_retrain:
        return component_predictions
    if isinstance(client, NumeraiClient):
        component_predictions.extend(
            _fit_and_predict_package_subprocess(
                workspace_root=workspace_root,
                package=package,
                components=tuple(pending_retrain),
                live_features=live_features,
            )
        )
        return component_predictions

    prepared_context: PreparedServingData | None = None
    current_key: ServingDataContextKey | None = None
    for component in pending_retrain:
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
        if current_key != plan.data_key:
            prepared_context = None
            gc.collect()
            prepared_context = prepare_training_context(
                workspace_root=workspace_root,
                client=client,
                plan=plan,
            )
            current_key = plan.data_key
        prepared = prepared_context
        if prepared is None:  # pragma: no cover - defensive
            raise ServingRuntimeError("serving_training_context_missing")
        fitted = fit_component(
            workspace_root=workspace_root,
            client=client,
            component=component,
            config_path=config_path,
            plan=plan,
            prepared_data=prepared,
        )
        component_predictions.append(
            (
                prediction_member_from_fitted(fitted),
                predict_component_live(component=fitted, live_features=live_features),
            )
        )
        del fitted
        gc.collect()
    return component_predictions


def _fit_and_predict_package_subprocess(
    *,
    workspace_root: str | Path,
    package: SubmissionPackageRecord,
    components: tuple[ServingComponentSpec, ...],
    live_features: pd.DataFrame,
) -> list[tuple[ServingPredictionMember, pd.DataFrame]]:
    live_dir = package.package_path / "artifacts" / "live" / "tmp"
    live_dir.mkdir(parents=True, exist_ok=True)
    live_path = live_dir / "live_features.parquet"
    live_features.to_parquet(live_path, index=False)
    component_predictions: list[tuple[ServingPredictionMember, pd.DataFrame]] = []
    for component in components:
        config_path = source_config_path(
            workspace_root=workspace_root,
            config_path=component.config_path,
            run_id=component.run_id,
        )
        payload_path = live_dir / f"{component.component_id}.payload.json"
        output_path = live_dir / f"{component.component_id}.predictions.parquet"
        log_path = live_dir / f"{component.component_id}.worker.log"
        payload_path.write_text(
            json.dumps(
                {
                    "workspace_root": str(Path(workspace_root).resolve()),
                    "config_path": str(config_path),
                    "live_path": str(live_path),
                    "output_path": str(output_path),
                    "component": {
                        "component_id": component.component_id,
                        "weight": component.weight,
                        "config_path": None if component.config_path is None else str(component.config_path),
                        "run_id": component.run_id,
                        "source_label": component.source_label,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        with log_path.open("w", encoding="utf-8") as log_handle:
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "numereng.features.serving.worker",
                        "--payload",
                        str(payload_path),
                    ],
                    check=True,
                    cwd=str(Path(workspace_root).resolve()),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
            except subprocess.CalledProcessError as exc:
                raise ServingRuntimeError(
                    f"serving_component_worker_failed:{component.component_id}:{log_path}"
                ) from exc
        component_predictions.append(
            (
                ServingPredictionMember(component_id=component.component_id, weight=component.weight),
                pd.read_parquet(output_path),
            )
        )
    return component_predictions


def _maybe_load_artifact_backed_component(
    *,
    workspace_root: str | Path,
    component: ServingComponentSpec,
):
    if component.run_id is None:
        return None
    try:
        loaded = load_run_backed_component(workspace_root=workspace_root, component=component)
    except ServingRuntimeError:
        return None
    return loaded.component


def _load_component_for_pickle(
    *,
    workspace_root: str | Path,
    component: ServingComponentSpec,
):
    loaded = load_run_backed_component(workspace_root=workspace_root, component=component)
    return loaded.component


def _load_pickle_compatible_components(
    *,
    workspace_root: str | Path,
    package: SubmissionPackageRecord,
) -> tuple[FittedComponent, ...]:
    if package.neutralization is not None and package.neutralization.enabled:
        raise ServingUnsupportedConfigError("serving_model_upload_neutralization_not_supported")
    loaded_components = []
    for item in package.components:
        try:
            loaded = load_run_backed_component(workspace_root=workspace_root, component=item)
        except Exception as exc:
            raise ServingUnsupportedConfigError("serving_model_upload_preflight_failed") from exc
        if (
            not loaded.model_upload_compatible
            or loaded.uses_custom_module
            or loaded.model_type != "LGBMRegressor"
            or loaded.component.baseline_predictions_path is not None
        ):
            raise ServingUnsupportedConfigError("serving_model_upload_preflight_failed")
        loaded_components.append(loaded.component)
    return tuple(loaded_components)


def _verify_isolated_pickle_runtime(
    *,
    pickle_path: Path,
    fitted_components: tuple[FittedComponent, ...],
    docker_image: str,
    working_dir: Path,
) -> dict[str, str]:
    uvx = shutil.which("uvx")
    if uvx is None:
        raise ServingRuntimeError("serving_model_upload_smoke_runtime_unavailable")
    python_version = _docker_python_version(docker_image)
    feature_cols = sorted({col for item in fitted_components for col in item.feature_cols})
    id_cols = sorted({item.id_col for item in fitted_components})
    era_cols = sorted({item.era_col for item in fitted_components})
    python_args = [] if python_version is None else ["--python", python_version]
    script = _pickle_smoke_script(
        pickle_path=pickle_path,
        feature_cols=feature_cols,
        id_cols=id_cols,
        era_cols=era_cols,
    )
    command = [
        uvx,
        *python_args,
        "--with",
        "cloudpickle",
        "--with",
        "pandas",
        "--with",
        "numpy",
        "--with",
        "lightgbm",
        "python",
        "-c",
        script,
    ]
    try:
        subprocess.run(
            command,
            check=True,
            cwd=str(working_dir.resolve()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        message = (exc.stdout or "").strip()
        if message:
            raise ServingRuntimeError(f"serving_model_upload_smoke_failed:{message}") from exc
        raise ServingRuntimeError("serving_model_upload_smoke_failed") from exc
    return {
        "checked_at": utc_now_iso(),
        "command": " ".join(command[:-1] + ["<python-smoke-script>"]),
        "runtime": docker_image,
    }


def _pickle_smoke_script(
    *,
    pickle_path: Path,
    feature_cols: list[str],
    id_cols: list[str],
    era_cols: list[str],
) -> str:
    payload = {
        "pickle_path": str(pickle_path.resolve()),
        "feature_cols": feature_cols,
        "id_cols": id_cols,
        "era_cols": era_cols,
    }
    return (
        "import json\n"
        "import inspect\n"
        "import pandas as pd\n"
        f"payload = json.loads({json.dumps(json.dumps(payload))})\n"
        "frame = pd.DataFrame({col: [0.0, 1.0] for col in payload['feature_cols']})\n"
        "for col in payload['id_cols']:\n"
        "    frame[col] = ['row_1', 'row_2']\n"
        "for col in payload['era_cols']:\n"
        "    frame[col] = ['live', 'live']\n"
        "benchmark = pd.DataFrame(index=frame.index)\n"
        "predictor = pd.read_pickle(payload['pickle_path'])\n"
        "assert len(inspect.signature(predictor).parameters) in (1, 2)\n"
        "submission = predictor(frame, benchmark)\n"
        "assert isinstance(submission, pd.DataFrame)\n"
        "assert list(submission.columns) == ['prediction']\n"
        "assert len(submission) == len(frame)\n"
        "assert submission['prediction'].notna().all()\n"
        "assert submission['prediction'].between(0, 1).all()\n"
    )


def _docker_python_version(docker_image: str) -> str | None:
    match = re.search(r"Python\\s+(\\d+\\.\\d+)", docker_image)
    if match is not None:
        return match.group(1)
    return None


def _ensure_pickle_upload_ready(*, package: SubmissionPackageRecord, docker_image: str) -> None:
    smoke_verified = package.artifacts.get("pickle_smoke_verified") == "true"
    smoke_runtime = package.artifacts.get("pickle_runtime_docker_image")
    if not smoke_verified:
        raise ServingValidationError("serving_model_upload_smoke_not_verified")
    if smoke_runtime != docker_image:
        raise ServingValidationError("serving_model_upload_runtime_mismatch")


def _resolve_live_dataset_names(*, client: ServingClient) -> tuple[str, str | None]:
    dataset_names = client.list_datasets()
    live_candidates = [
        item
        for item in dataset_names
        if item.lower().endswith("/live.parquet") and not item.lower().startswith(("signals/", "crypto/"))
    ]
    if not live_candidates:
        raise ServingRuntimeError("serving_live_dataset_unavailable")
    benchmark_candidates = [
        item
        for item in dataset_names
        if item.lower().endswith("/live_benchmark_models.parquet")
        and not item.lower().startswith(("signals/", "crypto/"))
    ]
    live_name = max(live_candidates, key=_dataset_sort_key)
    benchmark_name = max(benchmark_candidates, key=_dataset_sort_key) if benchmark_candidates else None
    return live_name, benchmark_name


def _dataset_sort_key(dataset_name: str) -> tuple[int, int, int, str]:
    lowered = dataset_name.lower()
    match = _LIVE_DATASET_PATTERN.fullmatch(lowered) or _LIVE_BENCHMARK_PATTERN.fullmatch(lowered)
    if match is None:
        return (-1, -1, -1, dataset_name)
    return (
        int(match.group("major")),
        int(match.group("minor") or 0),
        int(match.group("patch") or 0),
        dataset_name,
    )


def _download_dataset(*, client: ServingClient, dataset_name: str, destination: Path) -> Path:
    local_path = Path(client.download_dataset(dataset_name, dest_path=str(destination)))
    return local_path.expanduser().resolve()


_STATUS_ORDER = {
    "created": 0,
    "inspected": 1,
    "live_built": 2,
    "live_submitted": 3,
    "pickle_built": 4,
    "pickle_uploaded": 5,
}


def _save_package_update(
    package: SubmissionPackageRecord,
    *,
    status: str,
    artifacts: dict[str, str | None],
) -> SubmissionPackageRecord:
    merged_artifacts = dict(package.artifacts)
    for key, value in artifacts.items():
        if value is None:
            continue
        merged_artifacts[str(key)] = str(value)
    next_status = _monotonic_status(current=package.status, new=status)
    updated = replace(
        package,
        status=next_status,
        artifacts=merged_artifacts,
        updated_at=utc_now_iso(),
    )
    return save_package(updated)


def _normalize_existing_package_components(
    *,
    package: SubmissionPackageRecord,
    workspace_root: str | Path,
) -> SubmissionPackageRecord:
    normalized_components = _resolve_components(workspace_root=workspace_root, components=package.components)
    if normalized_components == package.components:
        return package
    normalized = replace(
        package,
        components=normalized_components,
        updated_at=utc_now_iso(),
    )
    return save_package(normalized)


def _inspect_and_persist_package(
    *,
    package: SubmissionPackageRecord,
    workspace_root: str | Path,
) -> ServingInspectionResult:
    inspection = inspect_submission_package(workspace_root=workspace_root, package=package)
    report_path = package.package_path / "artifacts" / "preflight" / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(inspection_payload(inspection), indent=2, sort_keys=True), encoding="utf-8")
    updated = _save_package_update(
        package,
        status="inspected",
        artifacts={
            "preflight_report_path": str(report_path),
            "preflight_checked_at": inspection.checked_at,
            "preflight_local_live_compatible": str(inspection.local_live_compatible).lower(),
            "preflight_model_upload_compatible": str(inspection.model_upload_compatible).lower(),
            "preflight_artifact_backed": str(inspection.artifact_backed).lower(),
            "preflight_artifact_ready": str(inspection.artifact_ready).lower(),
            "preflight_artifact_live_ready": str(inspection.artifact_live_ready).lower(),
            "preflight_pickle_upload_ready": str(inspection.pickle_upload_ready).lower(),
            "preflight_deployment_classification": inspection.deployment_classification,
        },
    )
    return replace(inspection, package=updated, report_path=report_path)


def _save_failure(*, package: SubmissionPackageRecord, stage: str, exc: Exception) -> None:
    try:
        _save_package_update(
            package,
            status=package.status,
            artifacts={
                "last_failure_stage": stage,
                "last_failure_code": str(exc),
                "last_failure_at": utc_now_iso(),
            },
        )
    except Exception:
        return


def _validate_model_upload_options(
    *,
    client: ServingClient,
    data_version: str,
    docker_image: str | None,
) -> None:
    available_data_versions = client.model_upload_data_versions()
    if data_version not in available_data_versions:
        raise ServingValidationError("serving_model_upload_data_version_unsupported")
    if docker_image is None:
        return
    available_images = client.model_upload_docker_images()
    if docker_image not in available_images:
        raise ServingValidationError("serving_model_upload_docker_image_unsupported")


def _monotonic_status(*, current: str, new: str) -> str:
    current_rank = _STATUS_ORDER.get(current, -1)
    new_rank = _STATUS_ORDER.get(new, -1)
    if new_rank >= current_rank:
        return new
    return current


def _round_token(round_num: int | None) -> str:
    return "current" if round_num is None else f"round_{round_num}"


__all__ = [
    "ServingClient",
    "create_serving_client",
    "create_submission_package",
    "build_live_submission_package",
    "build_submission_pickle",
    "inspect_package",
    "list_submission_packages",
    "submit_live_package",
    "upload_submission_pickle",
]
