"""Feature tests for the bounded combination study (freeze/run/finalize/status, P3).

Covers spec §6 P3: freeze preflight rejections (FHR run, seed set != {42,17,99},
surface mismatch, blank policy params, missing DR, duplicate holdout fingerprint),
tamper-between-freeze-and-run abort, atomic over-cap trials reject, run-after-seal
reject, crash-resume idempotency, and the finalize --select baseline path.

USAGE:
    uv run pytest tests/unit/numereng/features/research_portfolio/test_combination.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from numereng.features.research_portfolio import study_finalize, study_freeze, study_run, study_status
from numereng.features.research_portfolio.types import PortfolioError
from tests.unit.numereng.features.research_portfolio import _portfolio_fixtures as fx

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _freeze(store: fx.Store, tmp_path: Path, **kwargs: object):
    payload = fx.freeze_payload(store, **kwargs)
    path = fx.write_json_file(tmp_path, payload, name="freeze.json")
    return study_freeze(store_root=store.root, config_path=path)


# --------------------------------------------------------------------------- #
# Freeze happy path
# --------------------------------------------------------------------------- #


def test_freeze_materializes_snapshot_without_scoring(tmp_path: Path) -> None:
    store = fx.build_study_store(tmp_path)
    result = _freeze(store, tmp_path)

    assert result.frozen is True
    assert result.n_members == 2
    assert result.n_lanes == 2
    assert result.n_search_folds >= 1
    assert result.holdout_n_eras == 6
    assert result.surface_id is not None

    study_dir = Path(result.study_dir)
    assert (study_dir / "frozen_manifest.json").is_file()
    assert (study_dir / "holdout_fingerprint.json").is_file()
    # Freeze scores nothing: no ledger, no seal, no artifacts.
    assert not (study_dir / "ledger.jsonl").exists()
    assert not (study_dir / "sealed.json").exists()
    assert not (study_dir / "artifacts").exists()


def test_freeze_refuses_double_freeze(tmp_path: Path) -> None:
    store = fx.build_study_store(tmp_path)
    _freeze(store, tmp_path)
    with pytest.raises(PortfolioError, match="study_already_frozen"):
        _freeze(store, tmp_path)


# --------------------------------------------------------------------------- #
# Freeze preflight rejections (spec §6 P3)
# --------------------------------------------------------------------------- #


def test_freeze_rejects_fhr_member(tmp_path: Path) -> None:
    store = fx.build_study_store(tmp_path)
    for seed in (42, 17, 99):
        fx.set_run_profile(store, f"r_lane_alpha_s{seed}", "full_history_refit")
    with pytest.raises(PortfolioError, match="not_oof"):
        _freeze(store, tmp_path)


def test_freeze_rejects_non_trio_seed_set(tmp_path: Path) -> None:
    store = fx.build_study_store(tmp_path)
    fx.add_extra_seed_run(store, lane_id="lane_alpha", seed=7, max_depth=fx.LANE_ALPHA_DEPTH)
    with pytest.raises(PortfolioError, match="not_trio_complete"):
        _freeze(store, tmp_path)


def test_freeze_rejects_surface_mismatch(tmp_path: Path) -> None:
    store = fx.build_study_store(tmp_path)
    for seed in (42, 17, 99):
        fx.set_run_benchmark_sha(store, f"r_lane_beta_s{seed}", "benchmark-sha-DIVERGENT")
    with pytest.raises(PortfolioError, match="surface_mismatch"):
        _freeze(store, tmp_path)


def test_freeze_rejects_blank_policy_params(tmp_path: Path) -> None:
    store = fx.build_study_store(tmp_path, policy_filled=False)
    with pytest.raises(PortfolioError, match="policy_unset"):
        _freeze(store, tmp_path)


def test_freeze_rejects_missing_decision_record(tmp_path: Path) -> None:
    store = fx.build_study_store(tmp_path)
    with pytest.raises(PortfolioError, match="decision_record_id_missing"):
        _freeze(store, tmp_path, decision_record_id="")


def test_freeze_rejects_single_lane(tmp_path: Path) -> None:
    store = fx.build_study_store(tmp_path)
    members = [{"candidate_id": "cand_alpha", "lane_id": "lane_alpha", "anchor_config": "config_lane_alpha_s42.json"}]
    with pytest.raises(PortfolioError, match="need_two_lanes"):
        _freeze(store, tmp_path, members=members)


def test_freeze_rejects_duplicate_holdout_fingerprint(tmp_path: Path) -> None:
    store = fx.build_study_store(tmp_path)
    _freeze(store, tmp_path, study_id="S1")
    # Same holdout + panel + decision record, new study id -> accidental-reuse guard fires.
    payload = fx.freeze_payload(store, study_id="S2")
    path = fx.write_json_file(tmp_path, payload, name="freeze_s2.json")
    with pytest.raises(PortfolioError, match="holdout_reuse"):
        study_freeze(store_root=store.root, config_path=path)


def test_freeze_exploratory_bypasses_reuse_guard(tmp_path: Path) -> None:
    store = fx.build_study_store(tmp_path)
    _freeze(store, tmp_path, study_id="S1")
    payload = fx.freeze_payload(store, study_id="S2", exploratory=True)
    path = fx.write_json_file(tmp_path, payload, name="freeze_s2.json")
    result = study_freeze(store_root=store.root, config_path=path)
    assert result.exploratory is True


# --------------------------------------------------------------------------- #
# Run: scoring, tamper guard, cap, resume
# --------------------------------------------------------------------------- #


def _freeze_and_write_trials(store: fx.Store, tmp_path: Path, *, trials: list[dict] | None = None, **freeze_kwargs):
    _freeze(store, tmp_path, **freeze_kwargs)
    payload = fx.trials_payload(trials=trials)
    return fx.write_json_file(tmp_path, payload, name="trials.json")


def test_run_scores_trials_against_search_folds(tmp_path: Path) -> None:
    store = fx.build_study_store(tmp_path)
    trials = [fx.study_trial(trial_id="trial_a"), fx.study_trial(trial_id="trial_b", alpha_weight=0.5, beta_weight=0.4)]
    trials_path = _freeze_and_write_trials(store, tmp_path, trials=trials)
    result = study_run(store_root=store.root, trials_path=trials_path)

    assert result.executed == 2
    assert result.skipped == 0
    assert {trial.trial_id for trial in result.trials} == {"trial_a", "trial_b"}
    for trial in result.trials:
        assert trial.pooled_search_bmc is not None
        assert trial.baseline_pooled_search_bmc is not None
        assert trial.n_folds >= 1
        assert trial.status == "complete"
    assert Path(result.ledger_path).is_file()


def test_run_aborts_on_tampered_member_parquet(tmp_path: Path) -> None:
    store = fx.build_study_store(tmp_path)
    trials_path = _freeze_and_write_trials(store, tmp_path)
    fx.tamper_run_predictions(store, "r_lane_alpha_s42")
    with pytest.raises(PortfolioError, match="frozen_input_tampered"):
        study_run(store_root=store.root, trials_path=trials_path)


def test_run_rejects_over_cap_trials_whole_file(tmp_path: Path) -> None:
    store = fx.build_study_store(tmp_path, n_eras=24)
    trials = [
        fx.study_trial(trial_id="t1", beta_weight=0.5),
        fx.study_trial(trial_id="t2", beta_weight=0.4),
        fx.study_trial(trial_id="t3", beta_weight=0.3),
    ]
    trials_path = _freeze_and_write_trials(store, tmp_path, trials=trials, study_trial_cap=2)
    with pytest.raises(PortfolioError, match="trials_over_cap"):
        study_run(store_root=store.root, trials_path=trials_path)
    # Whole-file reject: nothing was scored.
    ledger = store.root / "experiments" / store.experiment_id / "combination_study" / "S1" / "ledger.jsonl"
    assert not ledger.exists()


def test_run_is_idempotent_on_resume(tmp_path: Path) -> None:
    store = fx.build_study_store(tmp_path)
    trials = [fx.study_trial(trial_id="trial_a"), fx.study_trial(trial_id="trial_b", beta_weight=0.4)]
    trials_path = _freeze_and_write_trials(store, tmp_path, trials=trials)

    first = study_run(store_root=store.root, trials_path=trials_path)
    assert first.executed == 2

    ledger = store.root / "experiments" / store.experiment_id / "combination_study" / "S1" / "ledger.jsonl"
    lines_after_first = ledger.read_text(encoding="utf-8").strip().splitlines()

    second = study_run(store_root=store.root, trials_path=trials_path)
    assert second.executed == 0
    assert second.skipped == 2
    assert second.superseded == 0
    # No duplicate ledger lines appended on the idempotent re-run.
    lines_after_second = ledger.read_text(encoding="utf-8").strip().splitlines()
    assert lines_after_second == lines_after_first


def test_run_rejects_after_seal(tmp_path: Path) -> None:
    store = fx.build_study_store(tmp_path)
    trials_path = _freeze_and_write_trials(store, tmp_path)
    study_run(store_root=store.root, trials_path=trials_path)
    study_finalize(store_root=store.root, study_id="S1", select="baseline")
    with pytest.raises(PortfolioError, match="study_sealed"):
        study_run(store_root=store.root, trials_path=trials_path)


# --------------------------------------------------------------------------- #
# Finalize + status
# --------------------------------------------------------------------------- #


def test_finalize_select_baseline_seals_and_writes_artifacts(tmp_path: Path) -> None:
    store = fx.build_study_store(tmp_path)
    _freeze(store, tmp_path)
    result = study_finalize(store_root=store.root, study_id="S1", select="baseline")

    assert result.is_baseline is True
    assert result.selected_trial == "baseline"
    assert result.sealed is True
    assert result.holdout_bmc is not None

    study_dir = Path(result.study_dir)
    assert (study_dir / "sealed.json").is_file()
    assert (study_dir / "holdout_result.json").is_file()
    artifacts = Path(result.artifacts_dir)
    for name in (
        "predictions.parquet",
        "weights.parquet",
        "correlation_matrix.parquet",
        "era_metrics.parquet",
        "lineage.json",
    ):
        assert (artifacts / name).is_file(), name


def test_finalize_select_trial_scores_on_holdout(tmp_path: Path) -> None:
    store = fx.build_study_store(tmp_path)
    trials_path = _freeze_and_write_trials(store, tmp_path)
    study_run(store_root=store.root, trials_path=trials_path)
    result = study_finalize(store_root=store.root, study_id="S1", select="trial_a")

    assert result.is_baseline is False
    assert result.selected_trial == "trial_a"
    assert result.sealed is True
    assert result.holdout_bmc is not None
    assert result.baseline_holdout_bmc is not None


def test_finalize_unknown_trial_rejected(tmp_path: Path) -> None:
    store = fx.build_study_store(tmp_path)
    _freeze(store, tmp_path)
    with pytest.raises(PortfolioError, match="study_trial_not_executed"):
        study_finalize(store_root=store.root, study_id="S1", select="never_ran")


def test_status_reports_lifecycle(tmp_path: Path) -> None:
    store = fx.build_study_store(tmp_path)
    trials_path = _freeze_and_write_trials(store, tmp_path)

    frozen_status = study_status(store_root=store.root, study_id="S1")
    assert frozen_status.frozen is True
    assert frozen_status.sealed is False
    assert frozen_status.trials_executed == 0

    study_run(store_root=store.root, trials_path=trials_path)
    ran_status = study_status(store_root=store.root, study_id="S1")
    assert ran_status.trials_executed == 1

    study_finalize(store_root=store.root, study_id="S1", select="baseline")
    sealed_status = study_status(store_root=store.root, study_id="S1")
    assert sealed_status.sealed is True
    assert sealed_status.selected_trial == "baseline"


def test_status_unknown_study_rejected(tmp_path: Path) -> None:
    fx.build_study_store(tmp_path)
    root = tmp_path / ".numereng"
    with pytest.raises(PortfolioError, match="study_not_found"):
        study_status(store_root=root, study_id="missing")
