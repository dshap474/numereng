# --------------------------------------------------------------------------- #
# Module docstring
# --------------------------------------------------------------------------- #
"""Closeout tests for one-time frozen-holdout opening, sealing, and tamper refusal.

The scoring call is stubbed because real scoring needs Numerai datasets; these tests
exercise the open/seal/idempotency/tamper control flow, not metric computation.

USAGE:
    uv run pytest tests/unit/numereng/agentic_research/closeout/test_holdout_open.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from numereng.agentic_research.engine import aggregate
from numereng.agentic_research.engine.closeout import evidence
from numereng.agentic_research.engine.closeout import types as ct
from numereng.features import holdout
from numereng.features.experiments import (
    create_experiment,
    freeze_experiment_holdout,
    get_experiment,
)

EXP_ID = "2026-07-16_closeout-holdout"
ERAS = [f"{index:04d}" for index in range(1, 41)]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _write_run(store_root: Path, run_id: str, *, eras: list[str] = ERAS) -> None:
    run_dir = store_root / "runs" / run_id
    predictions_rel = "artifacts/predictions/pred.parquet"
    predictions_path = run_dir / predictions_rel
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"era": eras, "prediction": range(len(eras))}).to_parquet(predictions_path)
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


def _recipe_group(run_ids: tuple[str, ...]) -> aggregate.RecipeGroup:
    return aggregate.RecipeGroup(
        recipe_key="rk-1",
        representative_config="config_001.json",
        seeds=(42, 17),
        per_seed=(),
        trio_mean=0.0041,
        trio_fnc_mean=0.02,
        count=len(run_ids),
        bmc_std=None,
        run_ids=run_ids,
    )


def _frozen_experiment(store_root: Path, run_ids: tuple[str, ...]):
    create_experiment(store_root=store_root, experiment_id=EXP_ID, holdout_n_eras=8, holdout_era_gap=2)
    for run_id in run_ids:
        _write_run(store_root, run_id)
    freeze_experiment_holdout(store_root=store_root, experiment_id=EXP_ID, run_id=run_ids[0])
    return get_experiment(store_root=store_root, experiment_id=EXP_ID)


def _stub_scoring(monkeypatch: pytest.MonkeyPatch, value: float) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def _fake(*, run_id, era_filter, store_root, stage):
        calls.append({"run_id": run_id, "mode": era_filter.mode, "eras": set(era_filter.eras)})
        return {"bmc_last_200_eras": {"mean": value}}

    monkeypatch.setattr(evidence, "score_run_eras", _fake)
    return calls


# --------------------------------------------------------------------------- #
# No-op when unset
# --------------------------------------------------------------------------- #


def test_open_holdout_none_when_not_requested(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    create_experiment(store_root=store_root, experiment_id=EXP_ID)
    _write_run(store_root, "run_a")
    experiment = get_experiment(store_root=store_root, experiment_id=EXP_ID)
    result = evidence._open_holdout(
        experiment=experiment,
        believed_best=_recipe_group(("run_a",)),
        runs_dir=store_root / "runs",
    )
    assert result is None


# --------------------------------------------------------------------------- #
# One-time open + seal + idempotency
# --------------------------------------------------------------------------- #


def test_open_holdout_scores_restricted_then_seals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_root = tmp_path / ".numereng"
    experiment = _frozen_experiment(store_root, ("run_a", "run_b"))
    calls = _stub_scoring(monkeypatch, 0.006)

    record = evidence._open_holdout(
        experiment=experiment,
        believed_best=_recipe_group(("run_a", "run_b")),
        runs_dir=store_root / "runs",
    )

    assert record is not None
    assert record["sealed"] is True
    assert record["holdout_primary_mean"] == 0.006
    # Closeout scores ONLY the holdout eras (restriction), never the loop-visible search eras.
    assert len(calls) == 2
    assert all(call["mode"] == "restrict" for call in calls)
    assert all(call["eras"] == set(ERAS[-8:]) for call in calls)
    # Spec is now sealed in the manifest.
    reloaded = get_experiment(store_root=store_root, experiment_id=EXP_ID)
    spec = holdout.spec_from_metadata(reloaded.metadata.get(holdout.METADATA_KEY))
    assert spec is not None and spec.sealed


def test_open_holdout_second_pass_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_root = tmp_path / ".numereng"
    experiment = _frozen_experiment(store_root, ("run_a",))
    calls = _stub_scoring(monkeypatch, 0.006)

    first = evidence._open_holdout(
        experiment=experiment,
        believed_best=_recipe_group(("run_a",)),
        runs_dir=store_root / "runs",
    )
    assert len(calls) == 1

    reloaded = get_experiment(store_root=store_root, experiment_id=EXP_ID)
    second = evidence._open_holdout(
        experiment=reloaded,
        believed_best=_recipe_group(("run_a",)),
        runs_dir=store_root / "runs",
    )
    # Sealed already: returns the persisted record, no re-scoring.
    assert second == first
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
# Tamper refusal
# --------------------------------------------------------------------------- #


def test_open_holdout_refuses_tampered_era_universe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_root = tmp_path / ".numereng"
    experiment = _frozen_experiment(store_root, ("run_a",))
    calls = _stub_scoring(monkeypatch, 0.006)
    # Rewrite the run's predictions over a different era universe after freeze.
    _write_run(store_root, "run_a", eras=[f"{index:04d}" for index in range(1, 51)])

    with pytest.raises(ct.CloseoutError, match="holdout_frozen_input_tampered"):
        evidence._open_holdout(
            experiment=experiment,
            believed_best=_recipe_group(("run_a",)),
            runs_dir=store_root / "runs",
        )
    assert calls == []  # never reached scoring
