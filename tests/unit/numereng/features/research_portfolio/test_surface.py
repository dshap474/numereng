"""comparison_surface_id tests (spec §2.3, verification §6 P1).

Covers surface-ID equality and mismatch, the training-target vs contribution-target
distinction (the surface must key off the *contribution* target), and the
unavailable-reason paths when provenance/predictions are absent.
"""

from __future__ import annotations

from pathlib import Path

from numereng.features.research_portfolio.surface import compute_surface_id
from tests.unit.numereng.features.research_portfolio import _portfolio_fixtures as fx


def _run(store: fx.Store, run_id: str, **kwargs) -> Path:
    config = kwargs.pop("config", None) or fx.valid_config(random_state=42, predictions_name=f"p_{run_id}")
    return fx.build_run(store, run_id=run_id, config=config, bmc=0.005, **kwargs)


def test_identical_surface_is_equal(tmp_path: Path) -> None:
    store = fx.build_store(tmp_path)
    _run(store, "a")
    _run(store, "b")
    first = compute_surface_id(run_dir=store.root / "runs" / "a")
    second = compute_surface_id(run_dir=store.root / "runs" / "b")
    assert first.surface_id is not None
    assert first.surface_id == second.surface_id


def test_surface_keys_off_contribution_target_not_training_target(tmp_path: Path) -> None:
    store = fx.build_store(tmp_path)
    # Same contribution target (ender_20) and identical data scope, but a different
    # TRAINING target_col (not part of the data scope) -> same surface. This proves
    # the surface keys off the contribution target, not run.json's training target.
    trained_ender = fx.valid_config(random_state=42, predictions_name="p_e")
    trained_alpha = fx.valid_config(random_state=42, predictions_name="p_a")
    trained_alpha["data"]["target_col"] = "target_alpha_20"
    _run(store, "ender", config=trained_ender, contribution_target="target_ender_20")
    _run(store, "alpha", config=trained_alpha, contribution_target="target_ender_20")
    ender = compute_surface_id(run_dir=store.root / "runs" / "ender")
    alpha = compute_surface_id(run_dir=store.root / "runs" / "alpha")
    assert ender.surface_id == alpha.surface_id

    # Different CONTRIBUTION target -> different surface even with identical everything else.
    _run(store, "other", config=trained_ender, contribution_target="target_cyrus_20")
    other = compute_surface_id(run_dir=store.root / "runs" / "other")
    assert other.surface_id is not None
    assert other.surface_id != ender.surface_id


def test_different_benchmark_hash_changes_surface(tmp_path: Path) -> None:
    store = fx.build_store(tmp_path)
    _run(store, "bm1", benchmark_sha="sha-aaa")
    _run(store, "bm2", benchmark_sha="sha-bbb")
    assert (
        compute_surface_id(run_dir=store.root / "runs" / "bm1").surface_id
        != compute_surface_id(run_dir=store.root / "runs" / "bm2").surface_id
    )


def test_different_panel_changes_surface(tmp_path: Path) -> None:
    store = fx.build_store(tmp_path)
    _run(store, "p1", era_ids=(("e1", "id1"), ("e1", "id2")))
    _run(store, "p2", era_ids=(("e1", "id1"), ("e2", "id9")))
    assert (
        compute_surface_id(run_dir=store.root / "runs" / "p1").surface_id
        != compute_surface_id(run_dir=store.root / "runs" / "p2").surface_id
    )


def test_missing_predictions_is_unavailable(tmp_path: Path) -> None:
    store = fx.build_store(tmp_path)
    _run(store, "nopred", with_predictions=False)
    result = compute_surface_id(run_dir=store.root / "runs" / "nopred")
    assert result.surface_id is None
    assert result.unavailable_reason == "missing_predictions"


def test_missing_resolved_config_is_unavailable(tmp_path: Path) -> None:
    store = fx.build_store(tmp_path)
    result = compute_surface_id(run_dir=store.root / "runs" / "absent")
    assert result.surface_id is None
    assert result.unavailable_reason == "missing_resolved_config"


def test_panel_hash_is_cached_in_sidecar(tmp_path: Path) -> None:
    store = fx.build_store(tmp_path)
    _run(store, "cache")
    run_dir = store.root / "runs" / "cache"
    compute_surface_id(run_dir=run_dir)
    assert (run_dir / "surface.json").is_file()
