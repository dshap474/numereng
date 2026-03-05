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
        metric="bmc_last_200_eras.mean",
        direction="maximize",
        n_trials=2,
        sampler="tpe",
        seed=1337,
        best_trial_number=1,
        best_value=0.2,
        best_run_id="run-1",
        storage_path=Path("/tmp/study-1"),
        config={"search_space": []},
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
        metric="bmc_last_200_eras.mean",
        direction="maximize",
        n_trials=2,
        sampler="tpe",
        seed=1337,
        best_trial_number=1,
        best_value=0.2,
        best_run_id="run-1",
        config={"search_space": []},
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


def _ensemble_result() -> EnsembleResult:
    return EnsembleResult(
        ensemble_id="ens-1",
        experiment_id="exp-1",
        name="ens-1",
        method="rank_avg",
        target="target_ender_20",
        metric="corr20v2_sharpe",
        status="completed",
        components=(
            EnsembleComponent(run_id="run-a", weight=0.6, rank=0),
            EnsembleComponent(run_id="run-b", weight=0.4, rank=1),
        ),
        metrics=(EnsembleMetric(name="corr20v2_mean", value=0.1),),
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
    response = api_module.hpo_create(
        api_module.HpoStudyCreateRequest(
            study_name="study-a",
            config_path="configs/base.json",
            experiment_id="exp-1",
            n_trials=2,
        )
    )
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
        assert getattr(request, "neutralizer_cols") == ()
        return _study_result()

    monkeypatch.setattr(api_module, "hpo_create_study", fake_create_study)

    response = api_module.hpo_create(
        api_module.HpoStudyCreateRequest(
            study_name="study-a",
            config_path="configs/base.json",
            neutralizer_cols=[],
        )
    )
    assert response.study_id == "study-1"


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
        api_module.hpo_create(
            api_module.HpoStudyCreateRequest(
                study_name="study-a",
                config_path="configs/base.json",
            )
        )


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
