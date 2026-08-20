# --------------------------------------------------------------------------- #
# Module docstring
# --------------------------------------------------------------------------- #
"""Tests that loop-visible scoring resolves the holdout exclusion (and is default-off).

USAGE:
    uv run pytest tests/unit/numereng/features/test_holdout_scoring_plumbing.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

from numereng.features import holdout
from numereng.features.scoring.models import PostTrainingScoringRequest
from numereng.features.scoring.run_service import _resolve_loop_visible_era_filter

ERAS = tuple(f"{index:04d}" for index in range(1, 41))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _seed_frozen_experiment(store_root: Path, experiment_id: str) -> holdout.HoldoutSpec:
    spec = holdout.build_spec(era_order=ERAS, holdout_n_eras=8, era_gap=2)
    exp_dir = store_root / "experiments" / experiment_id
    exp_dir.mkdir(parents=True)
    (exp_dir / "experiment.json").write_text(
        json.dumps({"metadata": {holdout.METADATA_KEY: spec.to_metadata()}}), encoding="utf-8"
    )
    return spec


# --------------------------------------------------------------------------- #
# Default-off plumbing
# --------------------------------------------------------------------------- #


def test_request_defaults_to_no_era_filter() -> None:
    request = PostTrainingScoringRequest.__dataclass_fields__["era_filter"]
    assert request.default is None


def test_resolver_returns_none_without_experiment(tmp_path: Path) -> None:
    result = _resolve_loop_visible_era_filter(store_root=tmp_path / ".numereng", run_manifest={"run_id": "r1"})
    assert result is None


def test_resolver_returns_none_when_spec_unfrozen(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    exp_dir = store_root / "experiments" / "2026-07-16_x"
    exp_dir.mkdir(parents=True)
    requested = holdout.HoldoutSpec(holdout_n_eras=8, era_gap=2)
    (exp_dir / "experiment.json").write_text(
        json.dumps({"metadata": {holdout.METADATA_KEY: requested.to_metadata()}}), encoding="utf-8"
    )
    result = _resolve_loop_visible_era_filter(store_root=store_root, run_manifest={"experiment_id": "2026-07-16_x"})
    assert result is None


# --------------------------------------------------------------------------- #
# Frozen-on plumbing
# --------------------------------------------------------------------------- #


def test_resolver_excludes_holdout_and_gap_when_frozen(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    spec = _seed_frozen_experiment(store_root, "2026-07-16_x")
    result = _resolve_loop_visible_era_filter(store_root=store_root, run_manifest={"experiment_id": "2026-07-16_x"})
    assert result is not None
    assert result.mode == "exclude"
    expected = set(spec.holdout_eras or ()) | set(spec.gap_eras or ())
    assert result.eras == frozenset(expected)


# --------------------------------------------------------------------------- #
# Hash-collision reuse invariant (loop.py:302)
# --------------------------------------------------------------------------- #


def _loop_rescore_predicate(*, reused: bool, metric_on_disk: float | None) -> bool:
    """Transcription of the score-skip guard in loop.py `_run_config_round`.

    Mirror of `if not reused or context.run_primary_metric_from_disk(...) is None`.
    Kept in lockstep with that line; changing the production guard must update this.
    """
    return (not reused) or (metric_on_disk is None)


def test_reused_run_without_disk_metric_is_always_rescored() -> None:
    # A reused run whose metrics.json is absent is re-scored, so it can never be read
    # with unfiltered provenance. run_service writes metrics.json and the era_filter
    # provenance in one pass, so a present metric implies the filter was recorded.
    assert _loop_rescore_predicate(reused=False, metric_on_disk=None) is True
    assert _loop_rescore_predicate(reused=False, metric_on_disk=0.01) is True
    assert _loop_rescore_predicate(reused=True, metric_on_disk=None) is True
    # Only skip when the run was reused AND already carries an on-disk metric.
    assert _loop_rescore_predicate(reused=True, metric_on_disk=0.01) is False


def test_frozen_experiment_forces_era_filter_on_every_loop_score(tmp_path: Path) -> None:
    # The only skip path (reused + metric on disk) is safe because every loop-visible
    # score in a frozen-holdout experiment resolves the exclusion filter from the
    # manifest (not the caller), and metrics.json + era_filter provenance are co-written.
    store_root = tmp_path / ".numereng"
    _seed_frozen_experiment(store_root, "2026-07-16_x")
    manifest = {"experiment_id": "2026-07-16_x", "run_id": "reused_run"}
    assert _resolve_loop_visible_era_filter(store_root=store_root, run_manifest=manifest) is not None
