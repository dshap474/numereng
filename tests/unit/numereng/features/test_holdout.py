# --------------------------------------------------------------------------- #
# Module docstring
# --------------------------------------------------------------------------- #
"""Unit tests for the frozen chronological-suffix holdout helpers.

USAGE:
    uv run pytest tests/unit/numereng/features/test_holdout.py -q
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from numereng.features import holdout

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

ERAS = tuple(f"{index:04d}" for index in range(1, 101))


# --------------------------------------------------------------------------- #
# Partition math
# --------------------------------------------------------------------------- #


def test_partition_carves_chronological_suffix() -> None:
    search, gap, hold = holdout.partition_eras(ERAS, holdout_n_eras=10, era_gap=3)
    assert hold == ERAS[-10:]
    assert gap == ERAS[-13:-10]
    assert search == ERAS[:-13]
    assert search + gap + hold == ERAS


def test_partition_zero_gap_is_valid() -> None:
    search, gap, hold = holdout.partition_eras(ERAS, holdout_n_eras=5, era_gap=0)
    assert gap == ()
    assert hold == ERAS[-5:]
    assert search == ERAS[:-5]


@pytest.mark.parametrize("holdout_n_eras,era_gap", [(0, 1), (-1, 0), (90, 10), (95, 6)])
def test_partition_rejects_degenerate_splits(holdout_n_eras: int, era_gap: int) -> None:
    with pytest.raises(holdout.HoldoutError, match="holdout_split_invalid"):
        holdout.partition_eras(ERAS, holdout_n_eras=holdout_n_eras, era_gap=era_gap)


# --------------------------------------------------------------------------- #
# Fingerprint stability + tamper
# --------------------------------------------------------------------------- #


def test_fingerprint_is_stable_and_order_sensitive() -> None:
    spec = holdout.build_spec(era_order=ERAS, holdout_n_eras=10, era_gap=3)
    holdout.verify_fingerprint(spec, era_order=ERAS)  # no raise


def test_fingerprint_detects_changed_era_universe() -> None:
    spec = holdout.build_spec(era_order=ERAS, holdout_n_eras=10, era_gap=3)
    tampered = ERAS + ("0101",)
    with pytest.raises(holdout.HoldoutError, match="tampered"):
        holdout.verify_fingerprint(spec, era_order=tampered)


# --------------------------------------------------------------------------- #
# Era filter
# --------------------------------------------------------------------------- #


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"era": list(ERAS), "prediction": range(len(ERAS))})


def test_exclude_filter_drops_holdout_and_gap() -> None:
    spec = holdout.build_spec(era_order=ERAS, holdout_n_eras=10, era_gap=3)
    filtered = holdout.exclusion_filter(spec).apply(_frame(), era_col="era")
    assert len(filtered) == len(ERAS) - 13
    remaining = set(filtered["era"])
    assert remaining.isdisjoint(set(spec.holdout_eras or ()))
    assert remaining.isdisjoint(set(spec.gap_eras or ()))


def test_restrict_filter_keeps_only_holdout() -> None:
    spec = holdout.build_spec(era_order=ERAS, holdout_n_eras=10, era_gap=3)
    filtered = holdout.restriction_filter(spec).apply(_frame(), era_col="era")
    assert set(filtered["era"]) == set(spec.holdout_eras or ())


def test_restrict_filter_rejects_empty_selection() -> None:
    empty = holdout.EraFilter(mode="restrict", eras=frozenset())
    with pytest.raises(holdout.HoldoutError, match="holdout_restrict_empty"):
        empty.apply(_frame(), era_col="era")


def test_filter_requires_era_column() -> None:
    spec = holdout.build_spec(era_order=ERAS, holdout_n_eras=10, era_gap=3)
    with pytest.raises(holdout.HoldoutError, match="holdout_era_col_missing"):
        holdout.exclusion_filter(spec).apply(_frame(), era_col="nope")


# --------------------------------------------------------------------------- #
# Spec (de)serialization
# --------------------------------------------------------------------------- #


def test_metadata_roundtrip_frozen_spec() -> None:
    spec = holdout.build_spec(era_order=ERAS, holdout_n_eras=10, era_gap=3)
    restored = holdout.spec_from_metadata(spec.to_metadata())
    assert restored == spec


def test_metadata_roundtrip_requested_spec() -> None:
    requested = holdout.HoldoutSpec(holdout_n_eras=7, era_gap=2)
    restored = holdout.spec_from_metadata(requested.to_metadata())
    assert restored == requested
    assert restored is not None and not restored.is_frozen


def test_spec_from_metadata_none_when_absent() -> None:
    assert holdout.spec_from_metadata(None) is None
    assert holdout.spec_from_metadata("not-a-dict") is None


def test_spec_from_metadata_raises_on_corrupt_payload() -> None:
    with pytest.raises(holdout.HoldoutError, match="holdout_spec_invalid"):
        holdout.spec_from_metadata({"other": 1})


def test_spec_from_metadata_rejects_wrong_mode() -> None:
    with pytest.raises(holdout.HoldoutError, match="holdout_mode_invalid"):
        holdout.spec_from_metadata({"mode": "random", "holdout_n_eras": 5, "era_gap": 1})


# --------------------------------------------------------------------------- #
# Seal + double-open guard
# --------------------------------------------------------------------------- #


def test_seal_flips_once_then_refuses() -> None:
    spec = holdout.build_spec(era_order=ERAS, holdout_n_eras=10, era_gap=3)
    sealed = holdout.seal(spec)
    assert sealed.sealed is True
    with pytest.raises(holdout.HoldoutError, match="holdout_reuse_blocked"):
        holdout.seal(sealed)


def test_seal_refuses_unfrozen_spec() -> None:
    with pytest.raises(holdout.HoldoutError, match="holdout_not_frozen"):
        holdout.seal(holdout.HoldoutSpec(holdout_n_eras=5, era_gap=1))


# --------------------------------------------------------------------------- #
# Manifest reader
# --------------------------------------------------------------------------- #


def test_resolve_active_spec_reads_manifest(tmp_path) -> None:
    store = tmp_path / ".numereng"
    exp_dir = store / "experiments" / "2026-07-16_x"
    exp_dir.mkdir(parents=True)
    spec = holdout.build_spec(era_order=ERAS, holdout_n_eras=10, era_gap=3)
    (exp_dir / "experiment.json").write_text(
        json.dumps({"metadata": {holdout.METADATA_KEY: spec.to_metadata()}}), encoding="utf-8"
    )
    resolved = holdout.resolve_active_spec(store_root=store, experiment_id="2026-07-16_x")
    assert resolved == spec


def test_resolve_active_spec_none_when_missing(tmp_path) -> None:
    assert holdout.resolve_active_spec(store_root=tmp_path / ".numereng", experiment_id="absent") is None


# --------------------------------------------------------------------------- #
# Numeric era ordering (width-crossing tail)
# --------------------------------------------------------------------------- #


def test_read_prediction_era_order_is_numeric_not_lexicographic(tmp_path) -> None:
    # Bare, unpadded eras that cross a digit-width boundary; lexicographic order
    # would place "1000" before "999", corrupting the chronological suffix.
    eras = [str(n) for n in range(8, 1001)]
    frame = pd.DataFrame({"era": list(reversed(eras)), "prediction": range(len(eras))})
    path = tmp_path / "pred.parquet"
    frame.to_parquet(path)

    order = holdout.read_prediction_era_order(path)
    assert list(order) == eras
    assert order[-1] == "1000"  # numeric max, not lexicographic "999"

    spec = holdout.build_spec(era_order=order, holdout_n_eras=5, era_gap=2)
    assert spec.holdout_eras == ("996", "997", "998", "999", "1000")
    assert spec.gap_eras == ("994", "995")
