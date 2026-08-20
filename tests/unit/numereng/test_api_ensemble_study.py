"""API-layer tests for the combination-study handlers (JSON response shape + errors)."""

from __future__ import annotations

from pathlib import Path

import pytest

from numereng import api
from numereng.platform.errors import PackageError
from tests.unit.numereng.features.research_portfolio import _portfolio_fixtures as fx


def _freeze(store: fx.Store, tmp_path: Path) -> api.StudyFreezeResponse:
    config = fx.write_json_file(tmp_path, fx.freeze_payload(store), name="freeze.json")
    return api.study_freeze(api.StudyFreezeRequest(workspace_root=str(tmp_path), config_path=str(config)))


def test_study_freeze_response_shape(tmp_path: Path) -> None:
    store = fx.build_study_store(tmp_path)
    response = _freeze(store, tmp_path)
    assert isinstance(response, api.StudyFreezeResponse)
    assert response.frozen is True
    assert response.n_members == 2
    assert '"frozen":true' in response.model_dump_json()


def test_study_run_and_finalize_response_shape(tmp_path: Path) -> None:
    store = fx.build_study_store(tmp_path)
    _freeze(store, tmp_path)
    trials = fx.write_json_file(tmp_path, fx.trials_payload(), name="trials.json")

    run = api.study_run(api.StudyRunRequest(workspace_root=str(tmp_path), trials_path=str(trials)))
    assert isinstance(run, api.StudyRunResponse)
    assert run.executed == 1
    assert run.trials[0].trial_id == "trial_a"
    assert run.trials[0].pooled_search_bmc is not None

    final = api.study_finalize(api.StudyFinalizeRequest(workspace_root=str(tmp_path), study_id="S1", select="trial_a"))
    assert isinstance(final, api.StudyFinalizeResponse)
    assert final.sealed is True
    assert final.is_baseline is False


def test_study_status_response_shape(tmp_path: Path) -> None:
    store = fx.build_study_store(tmp_path)
    _freeze(store, tmp_path)
    status = api.study_status(api.StudyStatusRequest(workspace_root=str(tmp_path), study_id="S1"))
    assert isinstance(status, api.StudyStatusResponse)
    assert status.frozen is True
    assert status.sealed is False
    assert status.trials_executed == 0


def test_blocked_freeze_raises_package_error(tmp_path: Path) -> None:
    store = fx.build_study_store(tmp_path, policy_filled=False)
    config = fx.write_json_file(tmp_path, fx.freeze_payload(store), name="freeze.json")
    with pytest.raises(PackageError, match="policy_unset"):
        api.study_freeze(api.StudyFreezeRequest(workspace_root=str(tmp_path), config_path=str(config)))


def test_unknown_study_raises_package_error(tmp_path: Path) -> None:
    fx.build_study_store(tmp_path)
    with pytest.raises(PackageError, match="study_not_found"):
        api.study_status(api.StudyStatusRequest(workspace_root=str(tmp_path), study_id="missing"))
