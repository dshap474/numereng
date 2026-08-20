"""portfolio_status / portfolio_report orchestration tests (spec §2.4).

Covers the missing-registry empty portfolio, blank-policy blockers, the policy
hash, lane blocker aggregation, and report persistence.
"""

from __future__ import annotations

import json
from pathlib import Path

from numereng.features.research_portfolio.status import portfolio_report, portfolio_status
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


def test_missing_registry_is_absent_portfolio(tmp_path: Path) -> None:
    store = fx.build_store(tmp_path)
    report = portfolio_status(store_root=store.root)
    assert report.portfolio_present is False
    assert report.lanes == ()
    assert report.policy_hash is None


def test_full_portfolio_present_with_policy_hash(tmp_path: Path) -> None:
    store = _trio_store(tmp_path)
    fx.write_registry(store, fx.registry_payload(store=store, candidates=[_candidate()]))
    report = portfolio_status(store_root=store.root)
    assert report.portfolio_present is True
    assert report.policy_hash is not None
    assert report.policy_gaps == ()
    assert len(report.lanes) == 1
    assert report.lanes[0].candidates[0].trio_complete is True


def test_blank_policy_produces_gap_blockers(tmp_path: Path) -> None:
    store = _trio_store(tmp_path)
    fx.write_registry(store, fx.registry_payload(store=store, candidates=[_candidate()], policy_filled=False))
    report = portfolio_status(store_root=store.root)
    assert "scout_tranche_cap" in report.policy_gaps
    assert "cross_lane_weight_cap" in report.policy_gaps
    assert "policy_unset:scout_tranche_cap" in report.blockers


def test_lane_blockers_bubble_into_global(tmp_path: Path) -> None:
    store = _trio_store(tmp_path)
    fx.write_registry(
        store,
        fx.registry_payload(store=store, candidates=[_candidate()], scale="does-not-exist"),
    )
    report = portfolio_status(store_root=store.root)
    assert any("scale_experiment_not_found" in item for item in report.blockers)


def test_report_persists_status_file(tmp_path: Path) -> None:
    store = _trio_store(tmp_path)
    fx.write_registry(store, fx.registry_payload(store=store, candidates=[_candidate()]))
    report = portfolio_report(store_root=store.root)
    assert report.report_path is not None
    written = Path(report.report_path)
    assert written.is_file()
    assert written.parent == store.root / "portfolio" / "reports"
    payload = json.loads(written.read_text())
    assert payload["portfolio_present"] is True
    assert payload["report_path"] == report.report_path


def test_status_without_write_does_not_persist(tmp_path: Path) -> None:
    store = _trio_store(tmp_path)
    fx.write_registry(store, fx.registry_payload(store=store, candidates=[_candidate()]))
    report = portfolio_status(store_root=store.root, write=False)
    assert report.report_path is None
    assert not (store.root / "portfolio" / "reports").exists()
