# --------------------------------------------------------------------------- #
# Module docstring
# --------------------------------------------------------------------------- #
"""Experiment-service tests for the frozen holdout lifecycle.

USAGE:
    uv run pytest tests/unit/numereng/features/experiments/test_holdout_ops.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import numereng.features.experiments.service as service_module
from numereng.features import holdout
from numereng.features.experiments import (
    ExperimentValidationError,
    freeze_experiment_holdout,
    get_experiment_holdout,
    seal_experiment_holdout,
)

EXP_ID = "2026-07-16_holdout"
ERAS = [f"{index:04d}" for index in range(1, 41)]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _write_run_with_predictions(store_root: Path, run_id: str) -> None:
    run_dir = store_root / "runs" / run_id
    predictions_rel = "artifacts/predictions/pred.parquet"
    predictions_path = run_dir / predictions_rel
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"era": ERAS, "prediction": range(len(ERAS))})
    frame.to_parquet(predictions_path)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "FINISHED",
                "experiment_id": EXP_ID,
                "artifacts": {"predictions": predictions_rel},
            }
        ),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Default-off
# --------------------------------------------------------------------------- #


def test_create_without_holdout_leaves_metadata_clean(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    record = service_module.create_experiment(store_root=store_root, experiment_id=EXP_ID)
    assert holdout.METADATA_KEY not in record.metadata
    assert get_experiment_holdout(store_root=store_root, experiment_id=EXP_ID) is None


def test_create_rejects_invalid_holdout_request(tmp_path: Path) -> None:
    with pytest.raises(ExperimentValidationError, match="experiment_holdout_request_invalid"):
        service_module.create_experiment(
            store_root=tmp_path / ".numereng",
            experiment_id=EXP_ID,
            holdout_n_eras=0,
        )


# --------------------------------------------------------------------------- #
# Freeze + seal lifecycle
# --------------------------------------------------------------------------- #


def test_create_seeds_requested_but_unfrozen_spec(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    service_module.create_experiment(
        store_root=store_root,
        experiment_id=EXP_ID,
        holdout_n_eras=8,
        holdout_era_gap=2,
    )
    spec = get_experiment_holdout(store_root=store_root, experiment_id=EXP_ID)
    assert spec is not None
    assert spec.holdout_n_eras == 8 and spec.era_gap == 2
    assert not spec.is_frozen


def test_lazy_freeze_is_idempotent(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    service_module.create_experiment(
        store_root=store_root,
        experiment_id=EXP_ID,
        holdout_n_eras=8,
        holdout_era_gap=2,
    )
    _write_run_with_predictions(store_root, "run_a")

    first = freeze_experiment_holdout(store_root=store_root, experiment_id=EXP_ID, run_id="run_a")
    assert first is not None and first.is_frozen
    assert first.holdout_eras == tuple(ERAS[-8:])
    assert first.gap_eras == tuple(ERAS[-10:-8])

    _write_run_with_predictions(store_root, "run_b")
    second = freeze_experiment_holdout(store_root=store_root, experiment_id=EXP_ID, run_id="run_b")
    assert second is not None
    assert second.fingerprint == first.fingerprint
    assert second.holdout_eras == first.holdout_eras


def test_freeze_noop_without_request(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    service_module.create_experiment(store_root=store_root, experiment_id=EXP_ID)
    _write_run_with_predictions(store_root, "run_a")
    assert freeze_experiment_holdout(store_root=store_root, experiment_id=EXP_ID, run_id="run_a") is None


def test_seal_flips_once_then_refuses(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    service_module.create_experiment(
        store_root=store_root,
        experiment_id=EXP_ID,
        holdout_n_eras=8,
        holdout_era_gap=2,
    )
    _write_run_with_predictions(store_root, "run_a")
    freeze_experiment_holdout(store_root=store_root, experiment_id=EXP_ID, run_id="run_a")

    sealed = seal_experiment_holdout(store_root=store_root, experiment_id=EXP_ID)
    assert sealed.sealed is True

    with pytest.raises(holdout.HoldoutError, match="holdout_reuse_blocked"):
        seal_experiment_holdout(store_root=store_root, experiment_id=EXP_ID)
