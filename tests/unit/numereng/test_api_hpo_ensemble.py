from __future__ import annotations

from pathlib import Path

import pytest

import numereng.api as api_module
from numereng.features.ensemble import (
    EnsembleComponent,
    EnsembleError,
    EnsembleMetric,
    EnsembleNotFoundError,
    EnsembleRecord,
    EnsembleResult,
    EnsembleSelectionResult,
    EnsembleSelectionSourceRule,
    EnsembleValidationError,
)
from numereng.features.hpo import (
    HpoDependencyError,
    HpoError,
    HpoNotFoundError,
    HpoStudyRecord,
    HpoStudyResult,
    HpoTrialRecord,
    HpoValidationError,
)
from numereng.platform.errors import PackageError


def _study_result() -> HpoStudyResult:
    return HpoStudyResult(
        study_id="study-1",
        study_name="study-a",
        experiment_id="exp-1",
        status="completed",
        best_trial_number=1,
        best_value=0.2,
        best_run_id="run-1",
        storage_path=Path("/tmp/study-1"),
        spec={
            "study_id": "study-1",
            "study_name": "study-a",
            "config_path": "configs/base.json",
            "experiment_id": "exp-1",
            "objective": {
                "metric": "bmc_last_200_eras.mean",
                "direction": "maximize",
                "neutralization": {
                    "enabled": False,
                    "neutralizer_path": None,
                    "proportion": 0.5,
                    "mode": "era",
                    "neutralizer_cols": None,
                    "rank_output": True,
                },
            },
            "search_space": {
                "model.params.learning_rate": {
                    "type": "float",
                    "low": 0.001,
                    "high": 0.1,
                    "step": None,
                    "log": True,
                    "choices": None,
                }
            },
            "sampler": {
                "kind": "tpe",
                "seed": 1337,
                "n_startup_trials": 10,
                "multivariate": True,
                "group": False,
            },
            "stopping": {
                "max_trials": 2,
                "max_completed_trials": None,
                "timeout_seconds": None,
                "plateau": {
                    "enabled": False,
                    "min_completed_trials": 15,
                    "patience_completed_trials": 10,
                    "min_improvement_abs": 0.00025,
                },
            },
        },
        attempted_trials=2,
        completed_trials=2,
        failed_trials=0,
        stop_reason="max_trials_reached",
        trials=(),
        created_at="2026-02-22T00:00:00+00:00",
        updated_at="2026-02-22T00:00:00+00:00",
    )


def _study_record() -> HpoStudyRecord:
    return HpoStudyRecord(
        study_id="study-1",
        experiment_id="exp-1",
        study_name="study-a",
        status="completed",
        best_trial_number=1,
        best_value=0.2,
        best_run_id="run-1",
        spec=_study_result().spec,
        attempted_trials=2,
        completed_trials=2,
        failed_trials=0,
        stop_reason="max_trials_reached",
        storage_path=Path("/tmp/study-1"),
        error_message=None,
        created_at="2026-02-22T00:00:00+00:00",
        updated_at="2026-02-22T00:00:00+00:00",
    )


def _trial_record() -> HpoTrialRecord:
    return HpoTrialRecord(
        study_id="study-1",
        trial_number=0,
        status="completed",
        value=0.1,
        run_id="run-0",
        config_path=Path("/tmp/study-1/configs/trial_0000.json"),
        params={"model.params.learning_rate": 0.01},
        error_message=None,
        started_at="2026-02-22T00:00:00+00:00",
        finished_at="2026-02-22T00:01:00+00:00",
        updated_at="2026-02-22T00:01:00+00:00",
    )


def _study_create_request() -> api_module.HpoStudyCreateRequest:
    return api_module.HpoStudyCreateRequest(
        study_id="study-a",
        study_name="study-a",
        config_path="configs/base.json",
        experiment_id="exp-1",
        objective=api_module.HpoObjectiveRequest(
            metric="bmc_last_200_eras.mean",
            direction="maximize",
            neutralization=api_module.HpoNeutralizationRequest(),
        ),
        search_space={
            "model.params.learning_rate": api_module.HpoSearchSpaceSpecRequest(
                type="float",
                low=0.001,
                high=0.1,
                log=True,
            )
        },
        sampler=api_module.HpoSamplerRequest(kind="tpe", seed=1337),
        stopping=api_module.HpoStoppingRequest(max_trials=2),
    )


def _ensemble_result() -> EnsembleResult:
    return EnsembleResult(
        ensemble_id="ens-1",
        experiment_id="exp-1",
        name="ens-1",
        method="rank_avg",
        target="target_ender_20",
        metric="corr_sharpe",
        status="completed",
        components=(
            EnsembleComponent(run_id="run-a", weight=0.6, rank=0),
            EnsembleComponent(run_id="run-b", weight=0.4, rank=1),
        ),
        metrics=(EnsembleMetric(name="corr_mean", value=0.1),),
        artifacts_path=Path("/tmp/ens-1"),
        config={"weights": [0.6, 0.4]},
        created_at="2026-02-22T00:00:00+00:00",
        updated_at="2026-02-22T00:00:00+00:00",
    )


def _ensemble_record() -> EnsembleRecord:
    result = _ensemble_result()
    return EnsembleRecord(
        ensemble_id=result.ensemble_id,
        experiment_id=result.experiment_id,
        name=result.name,
        method=result.method,
        target=result.target,
        metric=result.metric,
        status=result.status,
        components=result.components,
        metrics=result.metrics,
        artifacts_path=result.artifacts_path,
        config=result.config,
        created_at=result.created_at,
        updated_at=result.updated_at,
    )


def _ensemble_selection_result() -> EnsembleSelectionResult:
    return EnsembleSelectionResult(
        selection_id="selection-1",
        experiment_id="exp-1",
        target="target_ender_20",
        primary_metric="bmc_last_200_eras.mean",
        tie_break_metric="bmc.mean",
        status="completed",
        artifacts_path=Path("/tmp/selection-1"),
        frozen_candidate_count=8,
        surviving_candidate_count=6,
        equal_weight_variant_count=5,
        weighted_candidate_count=1771,
        winner_blend_id="small_only",
        winner_selection_mode="equal_weight",
        winner_component_ids=("small_target_a", "small_target_b"),
        winner_weights=(0.5, 0.5),
        winner_metrics={"bmc_last_200_eras.mean": 0.01, "bmc.mean": 0.008, "corr.mean": 0.02},
        created_at="2026-04-09T00:00:00+00:00",
        updated_at="2026-04-09T00:00:00+00:00",
    )


def test_hpo_create_delegates_to_feature_and_returns_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_create_study(
        *,
        store_root: str,
        request: object,
    ) -> HpoStudyResult:
        assert store_root == ".numereng"
        assert isinstance(request, object)
        return _study_result()

    monkeypatch.setattr(api_module, "hpo_create_study", fake_create_study)
    response = api_module.hpo_create(_study_create_request())
    assert response.study_id == "study-1"
    assert response.best_run_id == "run-1"
    assert response.storage_path == "/tmp/study-1"


def test_hpo_create_preserves_explicit_empty_neutralizer_cols(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_create_study(
        *,
        store_root: str,
        request: object,
    ) -> HpoStudyResult:
        assert store_root == ".numereng"
        neutralization = getattr(request, "objective").neutralization
        assert neutralization.neutralizer_cols == ()
        return _study_result()

    monkeypatch.setattr(api_module, "hpo_create_study", fake_create_study)

    request = _study_create_request()
    request.objective.neutralization.neutralizer_cols = []
    response = api_module.hpo_create(request)
    assert response.study_id == "study-1"


def test_hpo_get_normalizes_random_sampler_spec_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    record = _study_record()
    record.spec["sampler"] = {
        "kind": "random",
        "seed": 21,
        "n_startup_trials": 10,
        "multivariate": True,
        "group": False,
    }

    monkeypatch.setattr(api_module, "hpo_get_study", lambda **kwargs: record)

    response = api_module.hpo_get(api_module.HpoStudyGetRequest(study_id="study-1"))

    assert response.spec.sampler.kind == "random"
    assert response.spec.sampler.seed == 21
    assert response.model_dump()["spec"]["sampler"] == {
        "kind": "random",
        "seed": 21,
    }


@pytest.mark.parametrize(
    "error",
    [
        HpoValidationError("hpo_validation_error"),
        HpoDependencyError("hpo_dependency_error"),
        HpoError("hpo_generic_error"),
    ],
)
def test_hpo_create_translates_feature_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    def fake_create_study(
        *,
        store_root: str,
        request: object,
    ) -> HpoStudyResult:
        _ = (store_root, request)
        raise error

    monkeypatch.setattr(api_module, "hpo_create_study", fake_create_study)

    with pytest.raises(PackageError, match=str(error)):
        api_module.hpo_create(_study_create_request())


def test_hpo_list_get_and_trials_delegate_and_translate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_module, "hpo_list_studies", lambda **_: (_study_record(),))
    monkeypatch.setattr(api_module, "hpo_get_study", lambda **_: _study_record())
    monkeypatch.setattr(api_module, "hpo_get_study_trials", lambda **_: (_trial_record(),))

    listed = api_module.hpo_list()
    assert len(listed.studies) == 1
    assert listed.studies[0].study_id == "study-1"

    loaded = api_module.hpo_get(api_module.HpoStudyGetRequest(study_id="study-1"))
    assert loaded.study_id == "study-1"

    trials = api_module.hpo_trials(api_module.HpoStudyTrialsRequest(study_id="study-1"))
    assert len(trials.trials) == 1
    assert trials.trials[0].trial_number == 0

    monkeypatch.setattr(
        api_module,
        "hpo_get_study",
        lambda **_: (_ for _ in ()).throw(HpoNotFoundError("hpo_study_not_found:study-1")),
    )
    with pytest.raises(PackageError, match="hpo_study_not_found:study-1"):
        api_module.hpo_get(api_module.HpoStudyGetRequest(study_id="study-1"))


def test_ensemble_build_delegates_to_feature_and_returns_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_build_ensemble(
        *,
        store_root: str,
        request: object,
    ) -> EnsembleResult:
        assert store_root == ".numereng"
        assert getattr(request, "include_heavy_artifacts") is True
        assert getattr(request, "selection_note") == "top corr + low cluster"
        assert getattr(request, "regime_buckets") == 6
        return _ensemble_result()

    monkeypatch.setattr(api_module, "build_ensemble_record", fake_build_ensemble)
    response = api_module.ensemble_build(
        api_module.EnsembleBuildRequest(
            run_ids=["run-a", "run-b"],
            experiment_id="exp-1",
            include_heavy_artifacts=True,
            selection_note="top corr + low cluster",
            regime_buckets=6,
        )
    )
    assert response.ensemble_id == "ens-1"
    assert response.status == "completed"
    assert len(response.components) == 2


def test_ensemble_build_preserves_explicit_empty_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_build_ensemble(
        *,
        store_root: str,
        request: object,
    ) -> EnsembleResult:
        _ = store_root
        assert getattr(request, "weights") == ()
        return _ensemble_result()

    monkeypatch.setattr(api_module, "build_ensemble_record", fake_build_ensemble)

    response = api_module.ensemble_build(
        api_module.EnsembleBuildRequest(
            run_ids=["run-a", "run-b"],
            weights=[],
        )
    )
    assert response.ensemble_id == "ens-1"


def test_ensemble_select_delegates_to_feature_and_returns_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_select_ensemble(
        *,
        store_root: str,
        request: object,
    ) -> EnsembleSelectionResult:
        assert store_root == ".numereng"
        assert getattr(request, "selection_id") == "selection-1"
        assert getattr(request, "top_weighted_variants") == 3
        source_rules = getattr(request, "source_rules")
        assert len(source_rules) == 2
        assert source_rules[0] == EnsembleSelectionSourceRule(
            experiment_id="exp-medium",
            selection_mode="explicit_targets",
            explicit_targets=("target_alpha_20", "target_charlie_20"),
            top_n=None,
        )
        return _ensemble_selection_result()

    monkeypatch.setattr(api_module, "select_ensemble_record", fake_select_ensemble)
    response = api_module.ensemble_select(
        api_module.EnsembleSelectRequest(
            experiment_id="exp-1",
            source_experiment_ids=["exp-medium", "exp-small"],
            source_rules=[
                api_module.EnsembleSelectionSourceRuleRequest(
                    experiment_id="exp-medium",
                    selection_mode="explicit_targets",
                    explicit_targets=["target_alpha_20", "target_charlie_20"],
                ),
                api_module.EnsembleSelectionSourceRuleRequest(
                    experiment_id="exp-small",
                    selection_mode="top_n",
                    top_n=2,
                ),
            ],
            selection_id="selection-1",
            top_weighted_variants=3,
        )
    )
    assert response.selection_id == "selection-1"
    assert response.winner.blend_id == "small_only"
    assert response.artifacts_path == "/tmp/selection-1"


def test_ensemble_build_preserves_explicit_empty_neutralizer_cols(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_build_ensemble(
        *,
        store_root: str,
        request: object,
    ) -> EnsembleResult:
        _ = store_root
        assert getattr(request, "neutralizer_cols") == ()
        return _ensemble_result()

    monkeypatch.setattr(api_module, "build_ensemble_record", fake_build_ensemble)

    response = api_module.ensemble_build(
        api_module.EnsembleBuildRequest(
            run_ids=["run-a", "run-b"],
            neutralizer_cols=[],
        )
    )
    assert response.ensemble_id == "ens-1"


@pytest.mark.parametrize(
    "error",
    [
        EnsembleValidationError("ensemble_validation_error"),
        EnsembleError("ensemble_generic_error"),
    ],
)
def test_ensemble_build_translates_feature_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    def fake_build_ensemble(
        *,
        store_root: str,
        request: object,
    ) -> EnsembleResult:
        _ = (store_root, request)
        raise error

    monkeypatch.setattr(api_module, "build_ensemble_record", fake_build_ensemble)

    with pytest.raises(PackageError, match=str(error)):
        api_module.ensemble_build(api_module.EnsembleBuildRequest(run_ids=["run-a", "run-b"]))


def test_ensemble_list_and_get_delegate_and_translate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_module, "list_ensemble_records_api", lambda **_: (_ensemble_record(),))
    monkeypatch.setattr(api_module, "get_ensemble_record_api", lambda **_: _ensemble_record())

    listed = api_module.ensemble_list()
    assert len(listed.ensembles) == 1
    assert listed.ensembles[0].ensemble_id == "ens-1"

    loaded = api_module.ensemble_get(api_module.EnsembleGetRequest(ensemble_id="ens-1"))
    assert loaded.ensemble_id == "ens-1"

    monkeypatch.setattr(
        api_module,
        "get_ensemble_record_api",
        lambda **_: (_ for _ in ()).throw(EnsembleNotFoundError("ensemble_not_found:ens-1")),
    )
    with pytest.raises(PackageError, match="ensemble_not_found:ens-1"):
        api_module.ensemble_get(api_module.EnsembleGetRequest(ensemble_id="ens-1"))


def test_ensemble_list_and_get_translate_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api_module,
        "list_ensemble_records_api",
        lambda **_: (_ for _ in ()).throw(ValueError("ensemble_method_invalid:forward")),
    )
    with pytest.raises(PackageError, match="ensemble_method_invalid:forward"):
        api_module.ensemble_list()

    monkeypatch.setattr(
        api_module,
        "get_ensemble_record_api",
        lambda **_: (_ for _ in ()).throw(ValueError("ensemble_status_invalid:queued")),
    )
    with pytest.raises(PackageError, match="ensemble_status_invalid:queued"):
        api_module.ensemble_get(api_module.EnsembleGetRequest(ensemble_id="ens-1"))
