"""API-layer tests for the research-portfolio handler (JSON response shape + errors)."""

from __future__ import annotations

from pathlib import Path

import pytest

from numereng import api
from numereng.platform.errors import PackageError
from tests.unit.numereng.features.research_portfolio import _portfolio_fixtures as fx


def _trio_store(tmp_path: Path) -> fx.Store:
    store = fx.build_store(tmp_path)
    for seed, name, run_id, bmc in (
        (42, "config_010_s42.json", "r42", 0.0050),
        (17, "config_010_s17.json", "r17", 0.0040),
        (99, "config_010_s99.json", "r99", 0.0045),
    ):
        config = fx.valid_config(random_state=seed, predictions_name=f"pred_s{seed}")
        fx.write_config(store, name, config)
        fx.build_run(store, run_id=run_id, config=config, bmc=bmc)
    fx.write_journal(
        store,
        [
            fx.journal_row("config_010_s42.json", seed=42, metric=0.0050, run_id="r42"),
            fx.journal_row("config_010_s17.json", seed=17, metric=0.0040, run_id="r17"),
            fx.journal_row("config_010_s99.json", seed=99, metric=0.0045, run_id="r99"),
        ],
    )
    fx.write_state(store, {"total_rounds_completed": 3, "believed_best": {"config": "config_010_s42.json"}})
    return store


def _candidate() -> dict:
    return {"candidate_id": "c1", "role": "believed_best", "anchor_config": "config_010_s42.json"}


def test_portfolio_status_response_shape(tmp_path: Path) -> None:
    store = _trio_store(tmp_path)
    fx.write_registry(store, fx.registry_payload(store=store, candidates=[_candidate()]))
    response = api.portfolio_status(api.PortfolioStatusRequest(workspace_root=str(tmp_path)))
    assert isinstance(response, api.PortfolioStatusResponse)
    assert response.portfolio_present is True
    assert response.schema_version == 1
    lane = response.lanes[0]
    candidate = lane.candidates[0]
    assert candidate.trio_complete is True
    assert candidate.evidence_tier == "scale-confirmed"
    assert len(candidate.per_seed) == 3
    # Round-trips as JSON.
    dumped = response.model_dump_json()
    assert '"portfolio_present":true' in dumped


def test_absent_portfolio_response(tmp_path: Path) -> None:
    fx.build_store(tmp_path)
    response = api.portfolio_status(api.PortfolioStatusRequest(workspace_root=str(tmp_path)))
    assert response.portfolio_present is False
    assert response.lanes == []


def test_write_persists_report_path(tmp_path: Path) -> None:
    store = _trio_store(tmp_path)
    fx.write_registry(store, fx.registry_payload(store=store, candidates=[_candidate()]))
    response = api.portfolio_status(api.PortfolioStatusRequest(workspace_root=str(tmp_path), write=True))
    assert response.report_path is not None
    assert Path(response.report_path).is_file()


def test_malformed_registry_raises_package_error(tmp_path: Path) -> None:
    store = fx.build_store(tmp_path)
    path = store.root / "portfolio" / "registry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(PackageError):
        api.portfolio_status(api.PortfolioStatusRequest(workspace_root=str(tmp_path)))
