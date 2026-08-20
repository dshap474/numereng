"""Portfolio status/report orchestration (P1).

`portfolio_status` resolves every lane live and returns a `PortfolioReport`;
`portfolio_report` is the same computation but always persisted to
`reports/status-<ts>.json` (disk-first, no SQLite), stamped with the SHA-256 of
the policy block it ran under.

USAGE:
    from numereng.features.research_portfolio.status import portfolio_status, portfolio_report
    report = portfolio_status(store_root=".numereng")
    persisted = portfolio_report(store_root=".numereng")   # writes the report file
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from numereng.config.research_portfolio import REGISTRY_SCHEMA_VERSION, RegistryConfig
from numereng.features.research_portfolio.registry import load_registry, registry_path
from numereng.features.research_portfolio.resolve import resolve_lane
from numereng.features.research_portfolio.types import LaneStatus, PortfolioReport
from numereng.features.store import resolve_portfolio_reports_root

# Nullable policy params whose blank value blocks the transitions they gate (§2.4).
_NULLABLE_POLICY_FIELDS: tuple[str, ...] = (
    "scout_tranche_cap",
    "scout_quality_floor",
    "coverage_reserve",
    "diversity_bmc_tolerance",
    "capacity_class_rule",
    "live_review_min_resolved_rounds",
    "combination_trial_cap",
    "cross_lane_weight_cap",
)


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #


def portfolio_status(*, store_root: str | Path = ".numereng", write: bool = False) -> PortfolioReport:
    """Resolve every lane live; optionally persist the report to reports/."""

    registry = load_registry(store_root=store_root)
    generated_at = _utc_now_iso()
    path = str(registry_path(store_root=store_root))

    if registry is None:
        report = PortfolioReport(
            schema_version=REGISTRY_SCHEMA_VERSION,
            portfolio_present=False,
            generated_at=generated_at,
            policy_hash=None,
            policy_gaps=(),
            lanes=(),
            blockers=(),
            registry_path=path,
        )
        return _persist(report, store_root=store_root) if write else report

    lanes = tuple(resolve_lane(store_root=store_root, lane=lane) for lane in registry.lanes)
    lanes = _stamp_latest_diversity(lanes, store_root=store_root)
    policy_gaps = _policy_gaps(registry)
    report = PortfolioReport(
        schema_version=registry.schema_version,
        portfolio_present=True,
        generated_at=generated_at,
        policy_hash=_policy_hash(registry),
        policy_gaps=policy_gaps,
        lanes=lanes,
        blockers=_global_blockers(policy_gaps=policy_gaps, lanes=lanes),
        registry_path=path,
    )
    return _persist(report, store_root=store_root) if write else report


def portfolio_report(*, store_root: str | Path = ".numereng") -> PortfolioReport:
    """Resolve and persist a portfolio report to reports/status-<ts>.json."""

    return portfolio_status(store_root=store_root, write=True)


# --------------------------------------------------------------------------- #
# Policy + blockers
# --------------------------------------------------------------------------- #


def _policy_hash(registry: RegistryConfig) -> str:
    payload = registry.policy.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _policy_gaps(registry: RegistryConfig) -> tuple[str, ...]:
    return tuple(field_name for field_name in _NULLABLE_POLICY_FIELDS if getattr(registry.policy, field_name) is None)


def _global_blockers(*, policy_gaps: tuple[str, ...], lanes: tuple[LaneStatus, ...]) -> tuple[str, ...]:
    blockers = [f"policy_unset:{field_name}" for field_name in policy_gaps]
    for lane in lanes:
        blockers.extend(f"{lane.lane_id}:{item}" for item in lane.blockers)
    return tuple(blockers)


# --------------------------------------------------------------------------- #
# Latest diversity report pointer
# --------------------------------------------------------------------------- #


def _stamp_latest_diversity(
    lanes: tuple[LaneStatus, ...],
    *,
    store_root: str | Path,
) -> tuple[LaneStatus, ...]:
    """Point each lane at the newest diversity report that included it (§3)."""

    latest_by_lane = _latest_diversity_by_lane(store_root=store_root)
    if not latest_by_lane:
        return lanes
    return tuple(replace(lane, latest_diversity_report_id=latest_by_lane.get(lane.lane_id)) for lane in lanes)


def _latest_diversity_by_lane(*, store_root: str | Path) -> dict[str, str]:
    reports_root = resolve_portfolio_reports_root(store_root=store_root)
    if not reports_root.is_dir():
        return {}
    latest: dict[str, str] = {}
    for report_dir in sorted(reports_root.glob("diversity-*")):
        if not report_dir.is_dir():
            continue
        payload = _read_json(report_dir / "report.json")
        if payload is None:
            continue
        report_id = payload.get("report_id")
        included = payload.get("included_lanes")
        if not isinstance(report_id, str) or not isinstance(included, list):
            continue
        for lane_id in included:
            if isinstance(lane_id, str) and (lane_id not in latest or report_id > latest[lane_id]):
                latest[lane_id] = report_id
    return latest


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def _persist(report: PortfolioReport, *, store_root: str | Path) -> PortfolioReport:
    reports_root = resolve_portfolio_reports_root(store_root=store_root)
    reports_root.mkdir(parents=True, exist_ok=True)
    stamp = report.generated_at.replace(":", "").replace("-", "").replace(".", "")
    report_path = reports_root / f"status-{stamp}.json"
    written = PortfolioReport(
        schema_version=report.schema_version,
        portfolio_present=report.portfolio_present,
        generated_at=report.generated_at,
        policy_hash=report.policy_hash,
        policy_gaps=report.policy_gaps,
        lanes=report.lanes,
        blockers=report.blockers,
        registry_path=report.registry_path,
        report_path=str(report_path),
    )
    report_path.write_text(json.dumps(asdict(written), sort_keys=True, indent=2), encoding="utf-8")
    return written


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["portfolio_report", "portfolio_status"]
