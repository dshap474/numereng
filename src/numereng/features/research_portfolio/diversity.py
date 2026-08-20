"""Cross-lane diversity report orchestration (P2, spec §3).

Read-only: reads the registry policy + resolved lane evidence and writes only its
own artifacts under `.numereng/portfolio/reports/diversity-<id>/`. It never
allocates, stops, or deploys anything.

Hard-fail gates (no silent downsizing): `diversity_bmc_tolerance` must be set,
all included candidates must share one `comparison_surface_id`, and at least two
lanes must contribute an artifact-ready candidate. One global panel is loaded in a
single `load_ranked_components` call over every member seed run (ranking happens
after that join, so it must never be called once-per-recipe). Blending is
hierarchical (seed-avg within recipe -> equal within lane -> equal across lanes)
and leave-one-out removes a whole lane.

USAGE:
    from numereng.features.research_portfolio.diversity import portfolio_diversity
    report = portfolio_diversity(store_root=".numereng", lanes=("cyrus-lgbm", "ender-lgbm"))
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from numereng.config.research_portfolio import RegistryLane
from numereng.features.ensemble import diversity_metrics as dm
from numereng.features.ensemble import panel
from numereng.features.ensemble.builder import EnsembleBuildError, load_ranked_components
from numereng.features.research_portfolio.registry import load_registry
from numereng.features.research_portfolio.resolve import resolve_lane
from numereng.features.research_portfolio.types import (
    DIVERSITY_SCHEMA_VERSION,
    CandidateFact,
    DiversityInference,
    DiversityMember,
    DiversityReport,
    LaneLeaveOneOut,
    LaneStatus,
    PairwiseDiagnostic,
    PortfolioError,
)
from numereng.features.scoring.metrics import (
    attach_benchmark_predictions,
    load_benchmark_predictions_from_path,
)
from numereng.features.store import resolve_portfolio_reports_root, resolve_store_root
from numereng.features.training.repo import resolve_active_benchmark_predictions_path

# --------------------------------------------------------------------------- #
# Constants (recorded into every report; never hidden)
# --------------------------------------------------------------------------- #

DEFAULT_BLOCK_LENGTH_ERAS = 10
DEFAULT_N_RESAMPLES = 2000
DEFAULT_RNG_SEED = 7
_BENCHMARK_ALIAS = "active_benchmark"
_BOTTOM_DECILE = 0.10


@dataclass(frozen=True)
class _Panel:
    """Aligned global panel: era ranges, target/benchmark vectors, candidate matrix."""

    ranges: tuple[tuple[str, int, int], ...]
    target: np.ndarray
    benchmark: np.ndarray
    candidate_matrix: np.ndarray
    candidate_ids: tuple[str, ...]


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def portfolio_diversity(
    *,
    store_root: str | Path = ".numereng",
    lanes: tuple[str, ...] | None = None,
    block_length_eras: int = DEFAULT_BLOCK_LENGTH_ERAS,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    rng_seed: int = DEFAULT_RNG_SEED,
) -> DiversityReport:
    """Build a cross-lane diversity report over artifact-ready candidates."""

    root = resolve_store_root(store_root)
    registry = load_registry(store_root=root)
    if registry is None:
        raise PortfolioError("diversity_registry_absent")

    tolerance = registry.policy.diversity_bmc_tolerance
    if tolerance is None:
        raise PortfolioError("diversity_blocked:policy_unset:diversity_bmc_tolerance")

    selected = _select_lanes(registry.lanes, lanes)
    resolved = [resolve_lane(store_root=root, lane=lane) for lane in selected]

    included, excluded = _partition_candidates(resolved, tolerance=tolerance)
    surface_id = _require_single_surface(included, excluded)
    _require_two_lanes(included, excluded)

    inference = DiversityInference(
        block_length_eras=block_length_eras,
        n_resamples=n_resamples,
        rng_seed=rng_seed,
    )
    return _build_report(
        root=root,
        policy_hash=_policy_hash(registry.policy.model_dump(mode="json")),
        surface_id=surface_id,
        tolerance=tolerance,
        included=included,
        excluded=excluded,
        inference=inference,
    )


def latest_diversity_report_id(*, store_root: str | Path = ".numereng") -> str | None:
    """Return the newest persisted diversity report id, or None when none exist."""

    reports_root = resolve_portfolio_reports_root(store_root=store_root)
    if not reports_root.is_dir():
        return None
    ids = sorted(item.name.removeprefix("diversity-") for item in reports_root.glob("diversity-*") if item.is_dir())
    return ids[-1] if ids else None


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


def _select_lanes(lanes: list[RegistryLane], requested: tuple[str, ...] | None) -> list[RegistryLane]:
    if requested is None:
        return list(lanes)
    by_id = {lane.lane_id: lane for lane in lanes}
    missing = [lane_id for lane_id in requested if lane_id not in by_id]
    if missing:
        raise PortfolioError(f"diversity_lane_not_found:{','.join(missing)}")
    return [by_id[lane_id] for lane_id in requested]


def _partition_candidates(
    resolved: list[LaneStatus],
    *,
    tolerance: float,
) -> tuple[list[tuple[LaneStatus, CandidateFact]], list[tuple[str, str]]]:
    """Split candidates into gate-passing (lane, candidate) pairs vs excluded reasons."""

    artifact_ready: list[tuple[LaneStatus, CandidateFact]] = []
    excluded: list[tuple[str, str]] = []
    for lane in resolved:
        for candidate in lane.candidates:
            reason = _artifact_ready_reason(candidate)
            if reason is not None:
                excluded.append((candidate.candidate_id, reason))
            else:
                artifact_ready.append((lane, candidate))

    if not artifact_ready:
        return [], excluded

    best_bmc = max(
        (candidate.trio_bmc_mean for _lane, candidate in artifact_ready if candidate.trio_bmc_mean is not None),
        default=None,
    )
    included: list[tuple[LaneStatus, CandidateFact]] = []
    for lane, candidate in artifact_ready:
        if candidate.role == "standalone":
            included.append((lane, candidate))
            continue
        if best_bmc is None or candidate.trio_bmc_mean is None:
            excluded.append((candidate.candidate_id, "tolerance_unresolvable:trio_bmc_missing"))
            continue
        if candidate.trio_bmc_mean < best_bmc - tolerance:
            excluded.append(
                (candidate.candidate_id, f"outside_tolerance:{candidate.trio_bmc_mean:.6f}<{best_bmc - tolerance:.6f}")
            )
            continue
        included.append((lane, candidate))
    return included, excluded


def _artifact_ready_reason(candidate: CandidateFact) -> str | None:
    if not candidate.trio_complete:
        return "not_trio_complete"
    if not candidate.surface_match:
        return "surface_unresolved"
    modes = {fact.artifact_mode for fact in candidate.per_seed}
    if modes - {"full"}:
        return f"artifact_not_full:{sorted(modes - {'full'})}"
    return None


def _require_single_surface(
    included: list[tuple[LaneStatus, CandidateFact]],
    excluded: list[tuple[str, str]],
) -> str:
    surfaces = {surface for _lane, candidate in included for surface in candidate.surface_ids}
    if len(surfaces) > 1:
        raise PortfolioError(
            f"diversity_blocked:surface_mismatch:{sorted(surfaces)}:excluded={_excluded_summary(excluded)}"
        )
    if not surfaces:
        raise PortfolioError(f"diversity_blocked:no_resolvable_surface:excluded={_excluded_summary(excluded)}")
    return next(iter(surfaces))


def _require_two_lanes(
    included: list[tuple[LaneStatus, CandidateFact]],
    excluded: list[tuple[str, str]],
) -> None:
    lane_ids = {lane.lane_id for lane, _candidate in included}
    if len(lane_ids) < 2:
        raise PortfolioError(
            f"diversity_blocked:need_two_lanes:got={sorted(lane_ids)}:excluded={_excluded_summary(excluded)}"
        )


def _excluded_summary(excluded: list[tuple[str, str]]) -> str:
    return ";".join(f"{candidate_id}={reason}" for candidate_id, reason in excluded) or "none"


# --------------------------------------------------------------------------- #
# Report construction
# --------------------------------------------------------------------------- #


def _build_report(
    *,
    root: Path,
    policy_hash: str,
    surface_id: str,
    tolerance: float,
    included: list[tuple[LaneStatus, CandidateFact]],
    excluded: list[tuple[str, str]],
    inference: DiversityInference,
) -> DiversityReport:
    panel_data = _assemble_panel(root=root, included=included)
    candidate_ids = list(panel_data.candidate_ids)
    matrix = panel_data.candidate_matrix

    era_bmc = panel.score_on_panel(matrix, panel_data.target, panel_data.benchmark, panel_data.ranges).bmc
    spearman = dm.per_era_pairwise_spearman(matrix, tuple(candidate_ids), panel_data.ranges)
    joint = dm.joint_drawdown(era_bmc, tuple(candidate_ids), decile=_BOTTOM_DECILE)
    pairwise = _pairwise_diagnostics(candidate_ids, era_bmc, spearman, joint)
    pooled_corr = _pooled_correlation(matrix, candidate_ids)

    lane_columns, lane_order = _lane_columns(matrix, candidate_ids, included)
    blend_era_bmc, loo = _blend_and_loo(
        lane_columns=lane_columns,
        lane_order=lane_order,
        target=panel_data.target,
        benchmark=panel_data.benchmark,
        ranges=panel_data.ranges,
        inference=inference,
    )

    report_id = _report_id()
    report_dir = resolve_portfolio_reports_root(store_root=root) / f"diversity-{report_id}"
    report = DiversityReport(
        schema_version=DIVERSITY_SCHEMA_VERSION,
        report_id=report_id,
        generated_at=_utc_now_iso(),
        report_dir=str(report_dir),
        surface_id=surface_id,
        policy_hash=policy_hash,
        diversity_bmc_tolerance=tolerance,
        inference=inference,
        n_eras=len(panel_data.ranges),
        members=_members(root=root, included=included),
        included_lanes=lane_order,
        excluded_candidates=tuple(excluded),
        blend_bmc_mean=_finite_mean(blend_era_bmc),
        pairwise=pairwise,
        leave_one_out=loo,
    )
    _write_artifacts(
        report_dir=report_dir,
        report=report,
        era_labels=[era for era, _s, _e in panel_data.ranges],
        candidate_ids=candidate_ids,
        era_bmc=era_bmc,
        spearman=spearman,
        pooled_corr=pooled_corr,
    )
    return report


def _assemble_panel(*, root: Path, included: list[tuple[LaneStatus, CandidateFact]]) -> _Panel:
    """One global load + benchmark attach; all vectors stay row-aligned by construction."""

    run_ids = _member_run_ids(included)
    if not run_ids:
        raise PortfolioError("diversity_panel_no_member_runs")
    target_col = _contribution_target(root=root, run_id=run_ids[0])
    try:
        ranked, era_series, id_series, target_series = load_ranked_components(
            store_root=root,
            run_ids=run_ids,
            target_col=target_col,
        )
    except EnsembleBuildError as exc:
        raise PortfolioError(f"diversity_panel_row_key_mismatch:{exc}") from exc
    if target_series is None:
        raise PortfolioError(f"diversity_panel_target_unavailable:{target_col}")

    working = pd.DataFrame(
        {
            "era": era_series.astype(str).to_numpy(),
            "id": id_series.astype(str).to_numpy(),
            target_col: target_series.to_numpy(dtype=np.float64),
        }
    )
    candidate_ids: list[str] = []
    for _lane, candidate in included:
        cols = [f"pred_{fact.run_id}" for fact in candidate.per_seed if fact.run_id is not None]
        missing = [col for col in cols if col not in ranked.columns]
        if missing:
            raise PortfolioError(f"diversity_panel_missing_columns:{candidate.candidate_id}:{missing}")
        working[candidate.candidate_id] = ranked[cols].to_numpy(dtype=np.float64).mean(axis=1)
        candidate_ids.append(candidate.candidate_id)

    attached = _attach_benchmark_frame(root=root, frame=working)
    attached = attached.sort_values(["era", "id"]).reset_index(drop=True)
    ranges = panel.era_ranges(attached["era"].astype(str).tolist())
    return _Panel(
        ranges=ranges,
        target=attached[target_col].to_numpy(dtype=np.float64),
        benchmark=attached[_BENCHMARK_ALIAS].to_numpy(dtype=np.float64),
        candidate_matrix=attached[candidate_ids].to_numpy(dtype=np.float64),
        candidate_ids=tuple(candidate_ids),
    )


def _attach_benchmark_frame(*, root: Path, frame: pd.DataFrame) -> pd.DataFrame:
    data_root = (root / "datasets").resolve()
    benchmark_frame, benchmark_col = load_benchmark_predictions_from_path(
        resolve_active_benchmark_predictions_path(data_root=data_root),
        benchmark_model="prediction",
        benchmark_name=_BENCHMARK_ALIAS,
        prediction_cols=["prediction"],
        era_col="era",
        id_col="id",
        data_root=data_root,
    )
    attached = attach_benchmark_predictions(
        frame,
        benchmark_frame,
        benchmark_col,
        era_col="era",
        id_col="id",
        min_overlap_ratio=0.0,
    )
    return attached.rename(columns={benchmark_col: _BENCHMARK_ALIAS})


def _lane_columns(
    matrix: np.ndarray,
    candidate_ids: list[str],
    included: list[tuple[LaneStatus, CandidateFact]],
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    """Equal-within-lane column per lane; member count never buys weight."""

    index_by_id = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    grouped: dict[str, list[int]] = {}
    order: list[str] = []
    for lane, candidate in included:
        if lane.lane_id not in grouped:
            grouped[lane.lane_id] = []
            order.append(lane.lane_id)
        grouped[lane.lane_id].append(index_by_id[candidate.candidate_id])
    lane_columns = {lane_id: matrix[:, indices].mean(axis=1) for lane_id, indices in grouped.items()}
    return lane_columns, tuple(order)


def _blend_and_loo(
    *,
    lane_columns: dict[str, np.ndarray],
    lane_order: tuple[str, ...],
    target: np.ndarray,
    benchmark: np.ndarray,
    ranges: tuple[tuple[str, int, int], ...],
    inference: DiversityInference,
) -> tuple[np.ndarray, tuple[LaneLeaveOneOut, ...]]:
    """Equal-across-lanes blend, then leave-one-lane-out with paired block bootstrap."""

    lane_stack = np.column_stack([lane_columns[lane_id] for lane_id in lane_order])
    blend_era_bmc = _blend_era_bmc(lane_stack.mean(axis=1), target, benchmark, ranges)

    loo_results: list[LaneLeaveOneOut] = []
    for dropped in lane_order:
        remaining = [lane_id for lane_id in lane_order if lane_id != dropped]
        loo_column = np.column_stack([lane_columns[lane_id] for lane_id in remaining]).mean(axis=1)
        loo_era_bmc = _blend_era_bmc(loo_column, target, benchmark, ranges)
        loo_results.append(
            _leave_one_out(
                lane_id=dropped,
                blend_era_bmc=blend_era_bmc,
                loo_era_bmc=loo_era_bmc,
                inference=inference,
            )
        )
    return blend_era_bmc, tuple(loo_results)


def _blend_era_bmc(
    column: np.ndarray,
    target: np.ndarray,
    benchmark: np.ndarray,
    ranges: tuple[tuple[str, int, int], ...],
) -> np.ndarray:
    return panel.score_on_panel(column.reshape(-1, 1), target, benchmark, ranges).bmc[:, 0]


def _leave_one_out(
    *,
    lane_id: str,
    blend_era_bmc: np.ndarray,
    loo_era_bmc: np.ndarray,
    inference: DiversityInference,
) -> LaneLeaveOneOut:
    both = np.isfinite(blend_era_bmc) & np.isfinite(loo_era_bmc)
    diffs = (blend_era_bmc - loo_era_bmc)[both]
    boot = None
    if diffs.size >= 2 * inference.block_length_eras:
        boot = dm.paired_block_bootstrap(
            diffs,
            block_length_eras=inference.block_length_eras,
            n_resamples=inference.n_resamples,
            rng_seed=inference.rng_seed,
        )
    return LaneLeaveOneOut(
        lane_id=lane_id,
        blend_bmc_mean=_finite_mean(blend_era_bmc),
        loo_bmc_mean=_finite_mean(loo_era_bmc),
        mean_diff=float(np.mean(diffs)) if diffs.size else None,
        ci90_low=boot.ci90_low if boot else None,
        ci90_high=boot.ci90_high if boot else None,
        prob_positive=boot.prob_positive if boot else None,
    )


def _pairwise_diagnostics(
    candidate_ids: list[str],
    era_bmc: np.ndarray,
    spearman: tuple[dm.PairwiseSpearman, ...],
    joint: tuple[dm.JointDrawdown, ...],
) -> tuple[PairwiseDiagnostic, ...]:
    index_by_id = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    joint_by_pair = {(item.left, item.right): item.fraction for item in joint}
    results: list[PairwiseDiagnostic] = []
    for item in spearman:
        left_series = era_bmc[:, index_by_id[item.left]]
        right_series = era_bmc[:, index_by_id[item.right]]
        results.append(
            PairwiseDiagnostic(
                left=item.left,
                right=item.right,
                spearman_mean=item.mean,
                spearman_p10=item.p10,
                spearman_p90=item.p90,
                spearman_min=item.minimum,
                bmc_series_corr=_series_correlation(left_series, right_series),
                joint_drawdown_fraction=joint_by_pair.get((item.left, item.right)),
            )
        )
    return tuple(results)


def _members(
    *,
    root: Path,
    included: list[tuple[LaneStatus, CandidateFact]],
) -> tuple[DiversityMember, ...]:
    members: list[DiversityMember] = []
    for lane, candidate in included:
        run_ids = tuple(fact.run_id for fact in candidate.per_seed if fact.run_id is not None)
        shas = tuple(_prediction_sha256(root=root, run_id=run_id) for run_id in run_ids)
        members.append(
            DiversityMember(
                candidate_id=candidate.candidate_id,
                lane_id=lane.lane_id,
                recipe_key=candidate.recipe_key,
                run_ids=run_ids,
                prediction_sha256=shas,
                trio_bmc200=candidate.trio_bmc_mean,
            )
        )
    return tuple(members)


# --------------------------------------------------------------------------- #
# Member run / target / benchmark resolution
# --------------------------------------------------------------------------- #


def _member_run_ids(included: list[tuple[LaneStatus, CandidateFact]]) -> tuple[str, ...]:
    ordered: list[str] = []
    for _lane, candidate in included:
        for fact in candidate.per_seed:
            if fact.run_id is not None and fact.run_id not in ordered:
                ordered.append(fact.run_id)
    return tuple(ordered)


def _contribution_target(*, root: Path, run_id: str) -> str:
    provenance = _load_json(root / "runs" / run_id / "score_provenance.json")
    columns = _nested_dict(provenance, "columns")
    targets = columns.get("contribution_target_cols")
    if not isinstance(targets, list) or not targets:
        raise PortfolioError(f"diversity_contribution_target_unresolved:{run_id}")
    return str(sorted(str(item) for item in targets)[0])


def _prediction_sha256(*, root: Path, run_id: str) -> str:
    manifest = _load_json(root / "runs" / run_id / "run.json")
    rel = _nested_dict(manifest, "artifacts").get("predictions")
    path = (root / "runs" / run_id / rel) if isinstance(rel, str) and rel else None
    if path is None or not path.is_file():
        matches = sorted((root / "runs" / run_id / "artifacts" / "predictions").glob("*.parquet"))
        path = matches[0] if matches else None
    if path is None:
        raise PortfolioError(f"diversity_prediction_missing:{run_id}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #


def _write_artifacts(
    *,
    report_dir: Path,
    report: DiversityReport,
    era_labels: list[str],
    candidate_ids: list[str],
    era_bmc: np.ndarray,
    spearman: tuple[dm.PairwiseSpearman, ...],
    pooled_corr: pd.DataFrame,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
    era_bmc_frame = pd.DataFrame(era_bmc, columns=candidate_ids)
    era_bmc_frame.insert(0, "era", era_labels)
    era_bmc_frame.to_parquet(report_dir / "era_bmc.parquet")
    _spearman_frame(spearman).to_parquet(report_dir / "pairwise_correlation.parquet")
    pooled_corr.to_parquet(report_dir / "correlation_matrix.parquet")


def _spearman_frame(spearman: tuple[dm.PairwiseSpearman, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "left": item.left,
                "right": item.right,
                "spearman_mean": item.mean,
                "spearman_p10": item.p10,
                "spearman_p90": item.p90,
                "spearman_min": item.minimum,
                "n_eras": item.n_eras,
            }
            for item in spearman
        ]
    )


def _pooled_correlation(matrix: np.ndarray, candidate_ids: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(matrix, columns=candidate_ids)
    return frame.corr().replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _series_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    both = np.isfinite(left) & np.isfinite(right)
    if int(np.count_nonzero(both)) < 2:
        return None
    left_finite = left[both]
    right_finite = right[both]
    if float(np.std(left_finite)) == 0.0 or float(np.std(right_finite)) == 0.0:
        return None
    return float(np.corrcoef(left_finite, right_finite)[0, 1])


def _finite_mean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else None


def _policy_hash(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _report_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _load_json(path: Path) -> dict[str, object] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _nested_dict(payload: dict[str, object] | None, key: str) -> dict[str, object]:
    if payload is None:
        return {}
    value = payload.get(key)
    return cast("dict[str, object]", value) if isinstance(value, dict) else {}


__all__ = ["latest_diversity_report_id", "portfolio_diversity"]
