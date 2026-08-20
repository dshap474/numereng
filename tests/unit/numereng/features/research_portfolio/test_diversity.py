"""Cross-lane diversity orchestration tests (spec §3, verification §6 P2).

Everything is driven off synthetic on-disk state from ``_portfolio_fixtures``: two
lanes over one shared comparison surface plus an active-benchmark parquet. Covers the
happy path, the hard-fail gates (policy unset / one lane / surface mismatch / registry
absent / lane not found), tolerance exclusion vs the standalone exemption, equal
per-lane weighting under unequal candidate counts, LOO shape, artifact schemas, and
the status wiring of the latest report id.

USAGE:
    uv run pytest tests/unit/numereng/features/research_portfolio/test_diversity.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from numereng.features.ensemble.builder import EnsembleBuildError
from numereng.features.research_portfolio import diversity as div
from numereng.features.research_portfolio import portfolio_status
from numereng.features.research_portfolio.diversity import latest_diversity_report_id, portfolio_diversity
from numereng.features.research_portfolio.types import PortfolioError
from tests.unit.numereng.features.research_portfolio import _portfolio_fixtures as fx

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _run(store: fx.Store, **kwargs):
    kwargs.setdefault("block_length_eras", 2)
    kwargs.setdefault("n_resamples", 100)
    return portfolio_diversity(store_root=store.root, **kwargs)


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_happy_path_builds_two_lane_report(tmp_path: Path) -> None:
    store = fx.build_diversity_store(tmp_path)
    report = _run(store)

    assert report.surface_id is not None
    assert report.included_lanes == ("lane_alpha", "lane_beta")
    assert report.excluded_candidates == ()
    assert report.n_eras == 6
    assert report.diversity_bmc_tolerance == pytest.approx(0.0003)
    assert {member.candidate_id for member in report.members} == {"cand_alpha", "cand_beta"}
    for member in report.members:
        assert len(member.run_ids) == 3
        assert len(member.prediction_sha256) == 3
    # One unordered pair, both lanes leave-one-out'd.
    assert len(report.pairwise) == 1
    assert {loo.lane_id for loo in report.leave_one_out} == {"lane_alpha", "lane_beta"}


def test_bootstrap_populates_loo_ci_when_enough_blocks(tmp_path: Path) -> None:
    store = fx.build_diversity_store(tmp_path)
    report = _run(store, block_length_eras=2)  # 6 eras >= 2 full blocks of 2
    for loo in report.leave_one_out:
        assert loo.mean_diff is not None
        assert loo.ci90_low is not None
        assert loo.ci90_high is not None
        assert loo.prob_positive is not None


def test_loo_degrades_gracefully_without_enough_blocks(tmp_path: Path) -> None:
    store = fx.build_diversity_store(tmp_path)
    report = _run(store, block_length_eras=10)  # 6 eras < 2 full blocks of 10
    for loo in report.leave_one_out:
        assert loo.mean_diff is not None  # mean difference still computed
        assert loo.ci90_low is None  # but the bootstrap CI is skipped
        assert loo.ci90_high is None
        assert loo.prob_positive is None


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


def test_registry_absent_raises(tmp_path: Path) -> None:
    store = fx.build_store(tmp_path)
    with pytest.raises(PortfolioError, match="diversity_registry_absent"):
        portfolio_diversity(store_root=store.root)


def test_policy_tolerance_unset_raises(tmp_path: Path) -> None:
    store = fx.build_diversity_store(tmp_path, policy_filled=False)
    with pytest.raises(PortfolioError, match="policy_unset:diversity_bmc_tolerance"):
        _run(store)


def test_lane_not_found_raises(tmp_path: Path) -> None:
    store = fx.build_diversity_store(tmp_path)
    with pytest.raises(PortfolioError, match="diversity_lane_not_found:ghost"):
        _run(store, lanes=("lane_alpha", "ghost"))


def test_single_lane_selection_needs_two_lanes(tmp_path: Path) -> None:
    store = fx.build_diversity_store(tmp_path)
    with pytest.raises(PortfolioError, match="need_two_lanes"):
        _run(store, lanes=("lane_alpha",))


def test_surface_mismatch_raises(tmp_path: Path) -> None:
    store = fx.build_diversity_store(tmp_path)
    # Repoint lane_beta's whole trio at a different benchmark sha -> a second surface.
    for seed in (42, 17, 99):
        provenance = store.root / "runs" / f"r_lane_beta_s{seed}" / "score_provenance.json"
        payload = json.loads(provenance.read_text(encoding="utf-8"))
        payload["sources"]["benchmark"]["sha256"] = "other-benchmark-sha"
        provenance.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PortfolioError, match="surface_mismatch"):
        _run(store)


def test_panel_target_unavailable_raises(tmp_path: Path) -> None:
    # Rebuild lane runs without a target column in their prediction parquets.
    store = fx.build_store(tmp_path)
    era_ids = fx.diversity_era_ids()
    fx.write_active_benchmark(store, era_ids=era_ids, predictions=[0.5] * len(era_ids))
    lanes = []
    journal_rows: list[dict | str] = []
    for depth, cand_id, lane_id in ((9, "cand_alpha", "lane_alpha"), (6, "cand_beta", "lane_beta")):
        for seed in (42, 17, 99):
            name = f"config_{lane_id}_s{seed}.json"
            config = fx.valid_config(random_state=seed, predictions_name=f"p_{lane_id}_{seed}", max_depth=depth)
            fx.write_config(store, name, config)
            fx.build_run(store, run_id=f"r_{lane_id}_s{seed}", config=config, bmc=0.005, era_ids=era_ids)
            journal_rows.append(fx.journal_row(name, seed=seed, metric=0.005, run_id=f"r_{lane_id}_s{seed}"))
        lanes.append(
            fx.lane_block(
                lane_id=lane_id,
                store=store,
                candidates=[
                    {"candidate_id": cand_id, "role": "believed_best", "anchor_config": f"config_{lane_id}_s42.json"}
                ],
            )
        )
    fx.write_journal(store, journal_rows)
    fx.write_state(store, {"total_rounds_completed": 6})
    fx.write_registry(store, fx.registry_with_lanes(lanes=lanes))
    with pytest.raises(PortfolioError, match="diversity_panel_target_unavailable"):
        _run(store)


def test_row_key_mismatch_translated_to_portfolio_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Surface equality normally precludes a join mismatch; this pins the guard/translation.
    store = fx.build_diversity_store(tmp_path)

    def _boom(**_kwargs):
        raise EnsembleBuildError("ensemble_predictions_no_overlap")

    monkeypatch.setattr(div, "load_ranked_components", _boom)
    with pytest.raises(PortfolioError, match="diversity_panel_row_key_mismatch"):
        _run(store)


# --------------------------------------------------------------------------- #
# Tolerance vs standalone exemption
# --------------------------------------------------------------------------- #


def test_candidate_outside_tolerance_is_excluded(tmp_path: Path) -> None:
    store = fx.build_diversity_store(tmp_path, lane_a_bmc=0.0050, lane_b_bmc=0.0010)
    # lane_beta falls far below best - tolerance, so it is excluded -> only one lane left.
    with pytest.raises(PortfolioError, match="need_two_lanes") as excinfo:
        _run(store)
    assert "outside_tolerance" in str(excinfo.value)
    assert "cand_beta" in str(excinfo.value)


def test_standalone_role_exempt_from_tolerance(tmp_path: Path) -> None:
    store = fx.build_diversity_store(tmp_path, lane_a_bmc=0.0050, lane_b_bmc=0.0010, lane_b_role="standalone")
    report = _run(store)
    assert report.included_lanes == ("lane_alpha", "lane_beta")
    assert report.excluded_candidates == ()


# --------------------------------------------------------------------------- #
# Equal per-lane weighting under unequal candidate counts
# --------------------------------------------------------------------------- #


def _fixed_two_lane_store(tmp_path: Path, *, second_alpha: bool) -> fx.Store:
    """Two lanes with fixed predictions; lane_alpha optionally holds a duplicate candidate."""

    store = fx.build_store(tmp_path)
    era_ids = fx.diversity_era_ids(n_eras=6, ids_per_era=3)
    n_rows = len(era_ids)
    pattern = [0.1, 0.5, 0.9]
    pattern_b = [0.8, 0.2, 0.6]
    target = [[0.2, 0.7, 0.4][row % 3] for row in range(n_rows)]
    pred_a = [pattern[row % 3] for row in range(n_rows)]
    pred_b = [pattern_b[row % 3] for row in range(n_rows)]
    fx.write_active_benchmark(store, era_ids=era_ids, predictions=[[0.3, 0.55, 0.15][row % 3] for row in range(n_rows)])

    journal_rows: list[dict | str] = []
    alpha_candidates = [("cand_a1", 9, pred_a)]
    if second_alpha:
        # Second alpha candidate with identical predictions: lane column is unchanged.
        alpha_candidates.append(("cand_a2", 8, pred_a))

    def _emit(lane_id: str, entries: list[tuple[str, int, list[float]]]) -> dict:
        candidates = []
        for cand_id, depth, preds in entries:
            for seed in (42, 17, 99):
                name = f"config_{cand_id}_s{seed}.json"
                config = fx.valid_config(random_state=seed, predictions_name=f"p_{cand_id}_{seed}", max_depth=depth)
                fx.write_config(store, name, config)
                run_id = f"r_{cand_id}_s{seed}"
                fx.build_run(
                    store,
                    run_id=run_id,
                    config=config,
                    bmc=0.005,
                    era_ids=era_ids,
                    predictions=preds,
                    targets=target,
                )
                journal_rows.append(fx.journal_row(name, seed=seed, metric=0.005, run_id=run_id))
            candidates.append(
                {"candidate_id": cand_id, "role": "believed_best", "anchor_config": f"config_{cand_id}_s42.json"}
            )
        return fx.lane_block(lane_id=lane_id, store=store, candidates=candidates)

    lanes = [
        _emit("lane_alpha", alpha_candidates),
        _emit("lane_beta", [("cand_b", 6, pred_b)]),
    ]
    fx.write_journal(store, journal_rows)
    fx.write_state(store, {"total_rounds_completed": len(journal_rows)})
    fx.write_registry(store, fx.registry_with_lanes(lanes=lanes))
    return store


def test_member_count_does_not_buy_lane_weight(tmp_path: Path) -> None:
    one = _run(_fixed_two_lane_store(tmp_path / "one", second_alpha=False))
    two = _run(_fixed_two_lane_store(tmp_path / "two", second_alpha=True))
    # lane_alpha gains a second (identical-prediction) candidate; equal per-lane weighting
    # means the blend is unchanged, even though the member count rose.
    assert len(two.members) == len(one.members) + 1
    assert two.blend_bmc_mean == pytest.approx(one.blend_bmc_mean)


# --------------------------------------------------------------------------- #
# Artifact schemas
# --------------------------------------------------------------------------- #


def test_artifacts_written_with_expected_schema(tmp_path: Path) -> None:
    store = fx.build_diversity_store(tmp_path)
    report = _run(store)
    report_dir = Path(report.report_dir)
    assert report_dir.is_dir()

    payload = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["surface_id"] == report.surface_id
    assert payload["schema_version"] == report.schema_version
    assert payload["inference"]["block_length_eras"] == 2

    era_bmc = pd.read_parquet(report_dir / "era_bmc.parquet")
    assert list(era_bmc.columns) == ["era", "cand_alpha", "cand_beta"]
    assert len(era_bmc) == 6

    pairwise = pd.read_parquet(report_dir / "pairwise_correlation.parquet")
    assert set(pairwise.columns) == {
        "left",
        "right",
        "spearman_mean",
        "spearman_p10",
        "spearman_p90",
        "spearman_min",
        "n_eras",
    }

    corr = pd.read_parquet(report_dir / "correlation_matrix.parquet")
    assert list(corr.columns) == ["cand_alpha", "cand_beta"]
    assert corr.shape == (2, 2)


# --------------------------------------------------------------------------- #
# Status wiring: latest diversity report id
# --------------------------------------------------------------------------- #


def test_latest_report_id_and_status_stamp(tmp_path: Path) -> None:
    store = fx.build_diversity_store(tmp_path)
    assert latest_diversity_report_id(store_root=store.root) is None
    report = _run(store)
    assert latest_diversity_report_id(store_root=store.root) == report.report_id

    status = portfolio_status(store_root=store.root)
    for lane in status.lanes:
        assert lane.latest_diversity_report_id == report.report_id
