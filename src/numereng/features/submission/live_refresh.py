"""Refresh local submitted-model live score snapshots from Numerai."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from numereng.features.store.layout import resolve_workspace_layout
from numereng.features.submission.client import SubmissionClient, create_submission_client


@dataclass(frozen=True)
class SubmissionRefreshResult:
    """Refresh result for one submitted Numerai model."""

    model_name: str
    model_id: str | None
    live_rounds_path: Path
    submission_path: Path
    round_count: int
    scored_round_count: int
    resolved_round_count: int
    resolved_scored_round_count: int
    latest_scored_round: int | None = None
    latest_resolved_round: int | None = None
    skipped: bool = False
    warning: str | None = None


def refresh_submission_snapshots(
    *,
    workspace_root: str | Path = ".",
    model_names: Iterable[str] | None = None,
    client: SubmissionClient | None = None,
    dry_run: bool = False,
) -> list[SubmissionRefreshResult]:
    """Refresh local `.numereng/submissions/*` score snapshots."""

    layout = resolve_workspace_layout(workspace_root)
    submissions_root = layout.submissions_root
    resolved_client = client or create_submission_client(tournament="classic")
    model_map = resolved_client.get_models()
    names = list(model_names) if model_names is not None else _existing_submission_names(submissions_root)

    results: list[SubmissionRefreshResult] = []
    for model_name in names:
        submission_dir = submissions_root / model_name
        submission_path = submission_dir / "submission.json"
        live_rounds_path = submission_dir / "live_rounds.parquet"
        model_id = model_map.get(model_name)
        if model_id is None:
            results.append(
                SubmissionRefreshResult(
                    model_name=model_name,
                    model_id=None,
                    live_rounds_path=live_rounds_path,
                    submission_path=submission_path,
                    round_count=0,
                    scored_round_count=0,
                    resolved_round_count=0,
                    resolved_scored_round_count=0,
                    skipped=True,
                    warning="model_not_found_in_numerai_account",
                )
            )
            continue

        try:
            rows = resolved_client.round_model_performances_v2(model_id=model_id)
        except Exception as exc:
            results.append(
                SubmissionRefreshResult(
                    model_name=model_name,
                    model_id=model_id,
                    live_rounds_path=live_rounds_path,
                    submission_path=submission_path,
                    round_count=0,
                    scored_round_count=0,
                    resolved_round_count=0,
                    resolved_scored_round_count=0,
                    skipped=True,
                    warning=f"refresh_failed:{type(exc).__name__}:{exc}",
                )
            )
            continue

        pulled_at = datetime.now(UTC).isoformat()
        records = [_normalize_round(row, pulled_at=pulled_at) for row in rows]
        scored_count = sum(_has_score(row) for row in records)
        resolved_count = sum(row.get("state") == "resolved" for row in records)
        resolved_scored_count = sum(row.get("state") == "resolved" and _has_score(row) for row in records)
        latest_scored_round = _latest_round_number(row for row in records if _has_score(row))
        latest_resolved_round = _latest_round_number(row for row in records if row.get("state") == "resolved")

        if not dry_run:
            submission_dir.mkdir(parents=True, exist_ok=True)
            _write_parquet_atomic(pd.DataFrame.from_records(records), live_rounds_path)
            metadata = _read_metadata(submission_path)
            metadata["model_name"] = model_name
            metadata["model_id"] = model_id
            metadata["status"] = "live_scores_available" if scored_count else "awaiting_live_scores"
            metadata["refresh"] = {
                "pulled_at": pulled_at,
                "source": "numerai.round_model_performances_v2",
                "round_count": len(records),
                "scored_round_count": scored_count,
                "resolved_round_count": resolved_count,
                "resolved_scored_round_count": resolved_scored_count,
                "latest_scored_round": latest_scored_round,
                "latest_resolved_round": latest_resolved_round,
                "status": metadata["status"],
            }
            _write_json_atomic(metadata, submission_path)

        results.append(
            SubmissionRefreshResult(
                model_name=model_name,
                model_id=model_id,
                live_rounds_path=live_rounds_path,
                submission_path=submission_path,
                round_count=len(records),
                scored_round_count=scored_count,
                resolved_round_count=resolved_count,
                resolved_scored_round_count=resolved_scored_count,
                latest_scored_round=latest_scored_round,
                latest_resolved_round=latest_resolved_round,
            )
        )
    return results


def _existing_submission_names(submissions_root: Path) -> list[str]:
    if not submissions_root.is_dir():
        return []
    return sorted(
        item.name
        for item in submissions_root.iterdir()
        if item.is_dir() and ((item / "submission.json").is_file() or (item / "live_rounds.parquet").is_file())
    )


def _read_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temp_path, index=False)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _date_prefix(value: object) -> str | None:
    if hasattr(value, "isoformat"):
        try:
            return str(value.isoformat())[:10]
        except Exception:
            return None
    if not isinstance(value, str) or not value:
        return None
    return value[:10]


def _score_map(row: dict[str, Any]) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    raw_scores = row.get("submissionScores")
    if not isinstance(raw_scores, list):
        return scores
    for item in raw_scores:
        if not isinstance(item, dict):
            continue
        name = item.get("displayName")
        if isinstance(name, str) and name:
            scores[name] = item.get("value")
            scores[f"{name}_percentile"] = item.get("percentile")
    return scores


def _first_present(scores: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = scores.get(name)
        if value is not None:
            return value
    return None


def _normalize_round(row: dict[str, Any], *, pulled_at: str) -> dict[str, Any]:
    scores = _score_map(row)
    resolved = bool(row.get("roundResolved"))
    open_date = _date_prefix(row.get("roundOpenTime"))
    close_date = _date_prefix(row.get("roundCloseTime")) or open_date
    return {
        "round": row.get("roundNumber"),
        "round_number": row.get("roundNumber"),
        "state": "resolved" if resolved else "resolving",
        "open_date": open_date,
        "close_date": close_date,
        "resolve_date": _date_prefix(row.get("roundResolveTime")),
        "payout_factor": _to_float_or_none(row.get("roundPayoutFactor")),
        "at_risk_nmr": _to_float_or_none(row.get("atRisk")),
        "corr_multiplier": _to_float_or_none(row.get("corrMultiplier")),
        "mmc_multiplier": _to_float_or_none(row.get("mmcMultiplier")),
        "bmc": _first_present(scores, "bmc", "canon_bmc"),
        "bmc_percentile": _first_present(scores, "bmc_percentile", "canon_bmc_percentile"),
        "mmc": _first_present(scores, "mmc", "canon_mmc"),
        "mmc_percentile": _first_present(scores, "mmc_percentile", "canon_mmc_percentile"),
        "corr": _first_present(scores, "v2_corr20", "canon_corr", "corr60", "canon_corr60"),
        "corr_percentile": _first_present(
            scores,
            "v2_corr20_percentile",
            "canon_corr_percentile",
            "corr60_percentile",
            "canon_corr60_percentile",
        ),
        "fnc": _first_present(scores, "fnc_v3", "canon_fnc_v3"),
        "fnc_percentile": _first_present(scores, "fnc_v3_percentile", "canon_fnc_v3_percentile"),
        "mmc20": _first_present(scores, "mmc", "canon_mmc"),
        "mmc20_percentile": _first_present(scores, "mmc_percentile", "canon_mmc_percentile"),
        "corr20": _first_present(scores, "v2_corr20", "canon_corr"),
        "corr20_percentile": _first_present(scores, "v2_corr20_percentile", "canon_corr_percentile"),
        "mmc60": _first_present(scores, "mmc60", "canon_mmc60"),
        "mmc60_percentile": _first_present(scores, "mmc60_percentile", "canon_mmc60_percentile"),
        "corr60": _first_present(scores, "corr60", "canon_corr60"),
        "corr60_percentile": _first_present(scores, "corr60_percentile", "canon_corr60_percentile"),
        "season_score": scores.get("season_score"),
        "season_score_percentile": scores.get("season_score_percentile"),
        "source": "numerai.round_model_performances_v2",
        "pulled_at": pulled_at,
        "is_estimate": not resolved,
    }


def _to_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_score(row: dict[str, Any]) -> bool:
    return any(row.get(metric) is not None for metric in ("bmc", "mmc", "corr", "fnc", "mmc20", "corr20"))


def _latest_round_number(rows: Iterable[dict[str, Any]]) -> int | None:
    values: list[int] = []
    for row in rows:
        value = row.get("round_number") or row.get("round")
        try:
            values.append(int(value))
        except (TypeError, ValueError):
            continue
    return max(values) if values else None


__all__ = ["SubmissionRefreshResult", "refresh_submission_snapshots"]
