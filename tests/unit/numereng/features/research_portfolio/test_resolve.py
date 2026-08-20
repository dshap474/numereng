"""Live lane/candidate resolution tests (spec §2.2, verification §6 P1).

Every case is driven off synthetic on-disk state built by ``_portfolio_fixtures``;
nothing here mocks the resolvers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from numereng.config.research_portfolio import load_registry_config
from numereng.features.research_portfolio.resolve import resolve_lane
from numereng.features.research_portfolio.types import PortfolioValidationError
from tests.unit.numereng.features.research_portfolio import _portfolio_fixtures as fx

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _lane(payload: dict):
    return load_registry_config(payload).lanes[0]


def _trio_store(tmp_path: Path) -> fx.Store:
    """A full, clean trio: seeds 42/17/99 all completed and materialized in scale."""

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


def _candidate(anchor: str = "config_010_s42.json") -> dict:
    return {"candidate_id": "c_primary", "role": "believed_best", "anchor_config": anchor}


# --------------------------------------------------------------------------- #
# Trio resolution
# --------------------------------------------------------------------------- #


def test_trio_resolves_scale_confirmed_with_mean_and_surface_match(tmp_path: Path) -> None:
    store = _trio_store(tmp_path)
    payload = fx.registry_payload(store=store, candidates=[_candidate()])
    lane = resolve_lane(store_root=store.root, lane=_lane(payload))

    candidate = lane.candidates[0]
    assert candidate.trio_complete is True
    assert candidate.seeds_present == (17, 42, 99)
    assert candidate.evidence_tier == "scale-confirmed"
    assert candidate.trio_bmc_mean == pytest.approx((0.0050 + 0.0040 + 0.0045) / 3)
    assert candidate.bmc_sd is not None and candidate.bmc_sd > 0
    # All three seeds share one comparison surface (same scope/target/panel/benchmark).
    assert candidate.surface_match is True
    assert len(set(candidate.surface_ids)) == 1
    assert lane.surface_match is True
    assert candidate.blockers == ()


def test_bmc_read_from_disk_not_journal(tmp_path: Path) -> None:
    store = _trio_store(tmp_path)
    # Rewrite one journal metric to disagree with metrics.json; disk must win.
    fx.write_journal(
        store,
        [
            fx.journal_row("config_010_s42.json", seed=42, metric=0.9999, run_id="r42"),
            fx.journal_row("config_010_s17.json", seed=17, metric=0.0040, run_id="r17"),
            fx.journal_row("config_010_s99.json", seed=99, metric=0.0045, run_id="r99"),
        ],
    )
    payload = fx.registry_payload(store=store, candidates=[_candidate()])
    lane = resolve_lane(store_root=store.root, lane=_lane(payload))
    seed42 = next(s for s in lane.candidates[0].per_seed if s.seed == 42)
    assert seed42.bmc == pytest.approx(0.0050)
    assert seed42.journal_vs_disk_bmc_delta == pytest.approx(0.9999 - 0.0050)


# --------------------------------------------------------------------------- #
# Duplicate (recipe, seed) -> diagnostic, never silent-latest
# --------------------------------------------------------------------------- #


def test_duplicate_seed_runs_surface_as_diagnostic(tmp_path: Path) -> None:
    store = _trio_store(tmp_path)
    # A second seed-42 completed row for the same recipe (changes=[random_state] shape).
    dup_config = fx.valid_config(random_state=42, predictions_name="pred_s42_again")
    fx.write_config(store, "config_099_s42.json", dup_config)
    fx.build_run(store, run_id="r42b", config=dup_config, bmc=0.0060)
    fx.write_journal(
        store,
        [
            fx.journal_row("config_010_s42.json", seed=42, metric=0.0050, run_id="r42"),
            fx.journal_row("config_099_s42.json", seed=42, metric=0.0060, run_id="r42b"),
            fx.journal_row("config_010_s17.json", seed=17, metric=0.0040, run_id="r17"),
            fx.journal_row("config_010_s99.json", seed=99, metric=0.0045, run_id="r99"),
        ],
    )
    payload = fx.registry_payload(store=store, candidates=[_candidate()])
    lane = resolve_lane(store_root=store.root, lane=_lane(payload))
    candidate = lane.candidates[0]
    assert "duplicate_seed_runs:42" in candidate.blockers
    seed42 = next(s for s in candidate.per_seed if s.seed == 42)
    assert set(seed42.duplicate_run_ids) == {"r42", "r42b"}


# --------------------------------------------------------------------------- #
# Malformed journal -> hard fail
# --------------------------------------------------------------------------- #


def test_malformed_journal_line_hard_fails(tmp_path: Path) -> None:
    store = _trio_store(tmp_path)
    fx.write_journal(
        store,
        [
            fx.journal_row("config_010_s42.json", seed=42, metric=0.0050, run_id="r42"),
            "{not valid json",
        ],
    )
    payload = fx.registry_payload(store=store, candidates=[_candidate()])
    with pytest.raises(PortfolioValidationError, match="malformed_journal_line"):
        resolve_lane(store_root=store.root, lane=_lane(payload))


# --------------------------------------------------------------------------- #
# Skipped-seed-then-completed trio
# --------------------------------------------------------------------------- #


def test_skipped_seed_then_completed_is_not_trio(tmp_path: Path) -> None:
    store = fx.build_store(tmp_path)
    for seed, name, run_id, bmc in (
        (42, "config_010_s42.json", "r42", 0.0050),
        (17, "config_010_s17.json", "r17", 0.0040),
    ):
        config = fx.valid_config(random_state=seed, predictions_name=f"pred_s{seed}")
        fx.write_config(store, name, config)
        fx.build_run(store, run_id=run_id, config=config, bmc=bmc)
    # seed 99 attempted but failed (skipped), then no completed row exists for it.
    fx.write_journal(
        store,
        [
            fx.journal_row("config_010_s42.json", seed=42, metric=0.0050, run_id="r42"),
            fx.journal_row("config_010_s17.json", seed=17, metric=0.0040, run_id="r17"),
            {"status": "failed", "config": "config_010_s99.json", "seed": 99, "run_id": None},
        ],
    )
    fx.write_config(store, "config_010_s99.json", fx.valid_config(random_state=99, predictions_name="pred_s99"))
    fx.write_state(store, {"total_rounds_completed": 3})
    payload = fx.registry_payload(store=store, candidates=[_candidate()])
    lane = resolve_lane(store_root=store.root, lane=_lane(payload))
    candidate = lane.candidates[0]
    assert candidate.trio_complete is False
    assert candidate.seeds_present == (17, 42)
    assert candidate.evidence_tier == "discovery"


# --------------------------------------------------------------------------- #
# Config-hash mismatch -> blocker
# --------------------------------------------------------------------------- #


def test_config_hash_mismatch_blocks(tmp_path: Path) -> None:
    store = _trio_store(tmp_path)
    # Corrupt r17's recorded hash so the config file no longer matches the run.
    manifest_path = store.root / "runs" / "r17" / "run.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["hash"] = "deadbeef"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    payload = fx.registry_payload(store=store, candidates=[_candidate()])
    lane = resolve_lane(store_root=store.root, lane=_lane(payload))
    candidate = lane.candidates[0]
    assert "config_hash_mismatch:r17" in candidate.blockers
    seed17 = next(s for s in candidate.per_seed if s.seed == 17)
    assert seed17.config_hash_ok is False


# --------------------------------------------------------------------------- #
# Superseded experiment exclusion
# --------------------------------------------------------------------------- #


def test_superseded_scale_experiment_is_excluded(tmp_path: Path) -> None:
    store = _trio_store(tmp_path)
    superseded = [{"experiment_id": store.experiment_id, "superseded_by": "new-exp", "decision_record_id": "DR-9"}]
    payload = fx.registry_payload(store=store, candidates=[_candidate()], superseded=superseded)
    lane = resolve_lane(store_root=store.root, lane=_lane(payload))
    assert f"scale_experiment_superseded:{store.experiment_id}" in lane.blockers
    # No candidate facts resolved because the experiment evidence is excluded.
    assert lane.candidates[0].recipe_key is None


def test_scale_experiment_unset_blocks(tmp_path: Path) -> None:
    store = fx.build_store(tmp_path)
    payload = fx.registry_payload(store=store, candidates=[_candidate()], scale="")
    # scale="" is falsy but a valid str; force None via direct edit.
    payload["lanes"][0]["experiments"]["scale"] = None
    lane = resolve_lane(store_root=store.root, lane=_lane(payload))
    assert "scale_experiment_unset" in lane.blockers


def test_scale_experiment_not_found_blocks(tmp_path: Path) -> None:
    store = fx.build_store(tmp_path)
    payload = fx.registry_payload(store=store, candidates=[_candidate()], scale="does-not-exist")
    lane = resolve_lane(store_root=store.root, lane=_lane(payload))
    assert "scale_experiment_not_found:does-not-exist" in lane.blockers


# --------------------------------------------------------------------------- #
# Drift vs expected_believed_best
# --------------------------------------------------------------------------- #


def test_drift_detected_when_state_believed_best_differs(tmp_path: Path) -> None:
    store = _trio_store(tmp_path)  # state believed_best == config_010_s42.json
    payload = fx.registry_payload(store=store, candidates=[_candidate()], expected_believed_best="config_777_s42.json")
    lane = resolve_lane(store_root=store.root, lane=_lane(payload))
    assert lane.drift is not None
    assert "drift:expected=config_777_s42.json" in lane.drift
    assert any(item.startswith("drift:") for item in lane.blockers)


def test_no_drift_when_expected_matches_state(tmp_path: Path) -> None:
    store = _trio_store(tmp_path)
    payload = fx.registry_payload(store=store, candidates=[_candidate()], expected_believed_best="config_010_s42.json")
    lane = resolve_lane(store_root=store.root, lane=_lane(payload))
    assert lane.drift is None


# --------------------------------------------------------------------------- #
# Artifact + anchor edge cases
# --------------------------------------------------------------------------- #


def test_incomplete_artifact_blocks(tmp_path: Path) -> None:
    store = _trio_store(tmp_path)
    # Wipe r99's required root artifacts so classify_run_mode -> incomplete.
    for name in ("resolved.json", "results.json", "metrics.json"):
        (store.root / "runs" / "r99" / name).unlink()
    payload = fx.registry_payload(store=store, candidates=[_candidate()])
    lane = resolve_lane(store_root=store.root, lane=_lane(payload))
    assert any(item.startswith("artifact_incomplete:r99") for item in lane.candidates[0].blockers)


def test_anchor_config_not_found(tmp_path: Path) -> None:
    store = _trio_store(tmp_path)
    payload = fx.registry_payload(store=store, candidates=[_candidate(anchor="config_missing.json")])
    lane = resolve_lane(store_root=store.root, lane=_lane(payload))
    candidate = lane.candidates[0]
    assert candidate.recipe_key is None
    assert "anchor_config_not_found" in candidate.blockers
