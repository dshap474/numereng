"""Batch scoring helpers for deferred experiment-round scoring."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from numereng.features.scoring.models import CanonicalScoringStage
from numereng.features.scoring.run_service import score_run
from numereng.features.training.client import create_training_data_client
from numereng.features.training.models import ScoreRunResult


def score_run_batch(
    *,
    run_ids: Sequence[str],
    store_root: str | Path = ".numereng",
    stage: CanonicalScoringStage,
) -> tuple[ScoreRunResult, ...]:
    """Score multiple persisted runs while sharing one data client."""

    resolved_run_ids = tuple(str(run_id).strip() for run_id in run_ids if str(run_id).strip())
    if not resolved_run_ids:
        return ()

    client = create_training_data_client()
    results: list[ScoreRunResult] = []
    for run_id in resolved_run_ids:
        results.append(score_run(run_id=run_id, store_root=store_root, stage=stage, client=client))
    return tuple(results)


__all__ = ["score_run_batch"]
