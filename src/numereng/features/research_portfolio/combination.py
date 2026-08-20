"""Bounded combination study orchestration (P3, spec §4).

Deterministic, ledgered, two-phase. `study_freeze` runs the readiness preflight
and materializes an immutable snapshot (frozen manifest + holdout fingerprint)
WITHOUT scoring anything. `study_run` scores an immutable trials file against the
frozen search folds, appending one ledger line per trial. `study_finalize`
scores exactly one selected trial (or the baseline) on the held-out eras, writes
ensemble-format artifacts, and seals the study. `study_status` is read-only.

The holdout fingerprint is an accidental-reuse guard for one cooperative human
(refuses re-freezing the same holdout under the same decision record). It is
explicitly NOT a security boundary — a determined caller can bypass it, and that
is fine; its job is to catch an honest mistake.

USAGE:
    from numereng.features.research_portfolio.combination import study_freeze, study_run
    frozen = study_freeze(store_root=".numereng", config_path="freeze.json")
    result = study_run(store_root=".numereng", trials_path="trials.json")
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from numereng.config.research_portfolio import (
    FreezeConfig,
    RegistryConfig,
    RegistryLane,
    Trial,
    TrialsConfig,
    load_freeze_config,
    load_trials_config,
)
from numereng.features.ensemble import panel as panel_ops
from numereng.features.ensemble.builder import EnsembleBuildError, load_ranked_components
from numereng.features.ensemble.diversity_metrics import DiversityMetricError, paired_block_bootstrap
from numereng.features.feature_neutralization import (
    NeutralizationError,
    load_neutralizer_table,
    neutralize_prediction_frame,
)
from numereng.features.research_portfolio.registry import load_registry
from numereng.features.research_portfolio.resolve import resolve_lane
from numereng.features.research_portfolio.types import (
    CandidateFact,
    LaneStatus,
    PortfolioError,
    StudyFinalizeResult,
    StudyFreezeResult,
    StudyRunResult,
    StudyStatusResult,
    StudyTrialResult,
)
from numereng.features.scoring import SCORING_CONTRACT_VERSION
from numereng.features.scoring.metrics import (
    attach_benchmark_predictions,
    load_benchmark_predictions_from_path,
)
from numereng.features.store import resolve_store_root
from numereng.features.training.repo import resolve_active_benchmark_predictions_path

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

_STUDY_DIRNAME = "combination_study"
_FROZEN_MANIFEST = "frozen_manifest.json"
_HOLDOUT_FINGERPRINT = "holdout_fingerprint.json"
_SEALED = "sealed.json"
_LEDGER = "ledger.jsonl"
_HOLDOUT_RESULT = "holdout_result.json"
_ARTIFACTS_DIR = "artifacts"
_BENCHMARK_ALIAS = "active_benchmark"
_OOF_PROFILE = "purged_walk_forward"
_MAX_DISTINCT_NEUTRALIZATION_P = 3
_LEDGER_COMPLETE = "complete"
_LEDGER_SUPERSEDED = "superseded"


@dataclass(frozen=True)
class _StudyPanel:
    """One aligned scoring panel shared by freeze/run/finalize."""

    era_order: tuple[str, ...]
    ranges: tuple[tuple[str, int, int], ...]
    target: np.ndarray
    benchmark: np.ndarray
    eras: np.ndarray
    ids: np.ndarray
    candidate_columns: dict[str, np.ndarray]
    ordered_index: list[list[str]]
    panel_hash: str


@dataclass(frozen=True)
class _ResolvedMember:
    """A study member's resolved trio run ids and surface ids."""

    candidate_id: str
    lane_id: str
    run_ids: tuple[str, ...]
    surface_ids: tuple[str, ...]


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #


def study_freeze(*, store_root: str | Path = ".numereng", config_path: str | Path) -> StudyFreezeResult:
    """Run the readiness preflight and materialize an unsealed study snapshot."""

    root = resolve_store_root(store_root)
    freeze = _load_freeze(config_path)
    study_dir = _study_dir(root=root, experiment_id=freeze.experiment_id, study_id=freeze.study_id)
    if (study_dir / _FROZEN_MANIFEST).is_file():
        raise PortfolioError(f"study_already_frozen:{freeze.study_id}")

    registry = load_registry(store_root=root)
    if registry is None:
        raise PortfolioError("study_registry_absent")
    _preflight_policy(registry=registry, freeze=freeze)
    _preflight_decision_record(freeze)

    members = _preflight_members(root=root, registry=registry, freeze=freeze)
    surface_id = _preflight_single_surface(members)

    panel = _assemble_panel(root=root, members=members)
    split = _partition_eras(panel.era_order, freeze=freeze)
    contribution_target = _contribution_target(root=root, run_id=members[0].run_ids[0])
    neutralizer_sha = _preflight_neutralization(freeze=freeze, panel=panel)

    _guard_holdout_reuse(root=root, freeze=freeze, holdout_fingerprint=split.holdout_fingerprint(panel.panel_hash))
    _write_frozen(
        study_dir=study_dir,
        root=root,
        freeze=freeze,
        members=members,
        panel=panel,
        split=split,
        surface_id=surface_id,
        contribution_target=contribution_target,
        registry=registry,
        neutralizer_sha=neutralizer_sha,
    )
    return StudyFreezeResult(
        study_id=freeze.study_id,
        study_dir=str(study_dir),
        experiment_id=freeze.experiment_id,
        frozen=True,
        n_members=len(members),
        n_lanes=len({member.lane_id for member in members}),
        n_search_folds=len(split.search_folds),
        holdout_n_eras=len(split.holdout),
        surface_id=surface_id,
        holdout_fingerprint=split.holdout_fingerprint(panel.panel_hash),
        exploratory=freeze.exploratory,
    )


def study_run(
    *,
    store_root: str | Path = ".numereng",
    trials_path: str | Path,
    experiment_id: str | None = None,
) -> StudyRunResult:
    """Score an immutable trials file against a frozen, unsealed study."""

    root = resolve_store_root(store_root)
    trials = _load_trials(trials_path)
    study_dir = _locate_study(root=root, study_id=trials.study_id, experiment_id=experiment_id)
    manifest = _load_frozen(study_dir)
    _reject_if_sealed(study_dir)

    freeze = load_freeze_config(manifest["freeze_config"])  # snapshot round-trips through the same contract
    members = _members_from_manifest(manifest)
    panel = _assemble_panel(root=root, members=members)
    _verify_frozen_inputs(root=root, manifest=manifest, freeze=freeze, members=members, panel=panel)

    registry = load_registry(store_root=root)
    if registry is None:
        raise PortfolioError("study_registry_absent")
    cross_lane_cap = _require_policy_value(registry, "cross_lane_weight_cap")

    specs = _canonicalize_trials(trials=trials, freeze=freeze, members=members, cross_lane_cap=cross_lane_cap)
    split = _split_from_manifest(manifest)
    baseline = _baseline_bmc(panel=panel, freeze=freeze, split=split)

    return _execute_trials(
        study_dir=study_dir,
        panel=panel,
        freeze=freeze,
        split=split,
        specs=specs,
        baseline=baseline,
    )


def study_finalize(
    *,
    store_root: str | Path = ".numereng",
    study_id: str,
    select: str,
    experiment_id: str | None = None,
) -> StudyFinalizeResult:
    """Score the selected trial (or baseline) on the holdout eras and seal the study."""

    root = resolve_store_root(store_root)
    study_dir = _locate_study(root=root, study_id=study_id, experiment_id=experiment_id)
    manifest = _load_frozen(study_dir)
    _reject_if_sealed(study_dir)

    freeze = load_freeze_config(manifest["freeze_config"])
    members = _members_from_manifest(manifest)
    panel = _assemble_panel(root=root, members=members)
    _verify_frozen_inputs(root=root, manifest=manifest, freeze=freeze, members=members, panel=panel)

    split = _split_from_manifest(manifest)
    return _finalize_selection(
        study_dir=study_dir,
        manifest=manifest,
        freeze=freeze,
        members=members,
        panel=panel,
        split=split,
        select=select,
    )


def study_status(
    *,
    store_root: str | Path = ".numereng",
    study_id: str,
    experiment_id: str | None = None,
) -> StudyStatusResult:
    """Return a read-only lifecycle snapshot for one study."""

    root = resolve_store_root(store_root)
    study_dir = _locate_study(root=root, study_id=study_id, experiment_id=experiment_id)
    manifest = _load_frozen(study_dir)
    freeze = load_freeze_config(manifest["freeze_config"])
    sealed_payload = _load_json(study_dir / _SEALED)
    executed = len(_complete_ledger_specs(study_dir))
    return StudyStatusResult(
        study_id=study_id,
        study_dir=str(study_dir),
        frozen=True,
        sealed=sealed_payload is not None,
        trials_executed=executed,
        trial_cap=freeze.study_trial_cap,
        selected_trial=(str(sealed_payload.get("selected_trial")) if isinstance(sealed_payload, dict) else None),
    )


# --------------------------------------------------------------------------- #
# Freeze preflight
# --------------------------------------------------------------------------- #


def _preflight_policy(*, registry: RegistryConfig, freeze: FreezeConfig) -> None:
    for field_name in ("combination_trial_cap", "cross_lane_weight_cap", "diversity_bmc_tolerance"):
        if getattr(registry.policy, field_name) is None:
            raise PortfolioError(f"study_blocked:policy_unset:{field_name}")
    cap = registry.policy.combination_trial_cap
    if cap is not None and freeze.study_trial_cap > cap:
        raise PortfolioError(f"study_blocked:study_trial_cap_over_policy:{freeze.study_trial_cap}>{cap}")


def _preflight_decision_record(freeze: FreezeConfig) -> None:
    if not freeze.decision_record_id.strip():
        raise PortfolioError("study_blocked:decision_record_id_missing")


def _preflight_members(*, root: Path, registry: RegistryConfig, freeze: FreezeConfig) -> tuple[_ResolvedMember, ...]:
    if len({member.lane_id for member in freeze.members}) < 2:
        raise PortfolioError("study_blocked:need_two_lanes")

    lanes = {lane.lane_id: lane for lane in registry.lanes}
    resolved_cache: dict[str, LaneStatus] = {}
    members: list[_ResolvedMember] = []
    for member in freeze.members:
        lane = lanes.get(member.lane_id)
        if lane is None:
            raise PortfolioError(f"study_blocked:lane_not_in_registry:{member.lane_id}")
        status = resolved_cache.get(member.lane_id) or _resolve_cached(root=root, lane=lane, cache=resolved_cache)
        candidate = _find_candidate(status, member.candidate_id)
        if candidate is None:
            raise PortfolioError(f"study_blocked:candidate_not_found:{member.candidate_id}")
        members.append(_check_member(member_run_ids=tuple(member.run_ids), candidate=candidate, lane_id=member.lane_id))
    return tuple(members)


def _resolve_cached(*, root: Path, lane: RegistryLane, cache: dict[str, LaneStatus]) -> LaneStatus:
    status = resolve_lane(store_root=root, lane=lane)
    cache[lane.lane_id] = status
    return status


def _find_candidate(status: LaneStatus, candidate_id: str) -> CandidateFact | None:
    for candidate in status.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def _check_member(*, member_run_ids: tuple[str, ...], candidate: CandidateFact, lane_id: str) -> _ResolvedMember:
    if not candidate.trio_complete:
        raise PortfolioError(
            f"study_blocked:not_trio_complete:{candidate.candidate_id}:{list(candidate.seeds_present)}"
        )
    if candidate.evidence_tier != "scale-confirmed":
        raise PortfolioError(f"study_blocked:not_scale_confirmed:{candidate.candidate_id}:{candidate.evidence_tier}")
    for fact in candidate.per_seed:
        if fact.training_profile != _OOF_PROFILE:
            raise PortfolioError(
                f"study_blocked:not_oof:{candidate.candidate_id}:{fact.run_id}:{fact.training_profile}"
            )
        if fact.artifact_mode != "full":
            raise PortfolioError(
                f"study_blocked:artifact_not_full:{candidate.candidate_id}:{fact.run_id}:{fact.artifact_mode}"
            )
    resolved_run_ids = tuple(fact.run_id for fact in candidate.per_seed if fact.run_id is not None)
    if member_run_ids and set(member_run_ids) != set(resolved_run_ids):
        raise PortfolioError(f"study_blocked:member_run_ids_mismatch:{candidate.candidate_id}")
    return _ResolvedMember(
        candidate_id=candidate.candidate_id,
        lane_id=lane_id,
        run_ids=resolved_run_ids,
        surface_ids=candidate.surface_ids,
    )


def _preflight_single_surface(members: tuple[_ResolvedMember, ...]) -> str:
    surfaces = {surface for member in members for surface in member.surface_ids}
    if len(surfaces) != 1:
        raise PortfolioError(f"study_blocked:surface_mismatch:{sorted(surfaces)}")
    return next(iter(surfaces))


def _preflight_neutralization(*, freeze: FreezeConfig, panel: _StudyPanel) -> str | None:
    block = freeze.neutralization
    if block is None:
        return None
    source = Path(block.source_path).expanduser()
    if not source.is_file():
        raise PortfolioError(f"study_blocked:neutralizer_missing:{block.source_path}")
    content_sha = _file_sha256(source)
    if block.content_sha256 and block.content_sha256 != content_sha:
        raise PortfolioError("study_blocked:neutralizer_sha_mismatch")
    if not block.columns:
        raise PortfolioError("study_blocked:neutralizer_columns_missing")
    if not block.justification.strip():
        raise PortfolioError("study_blocked:neutralizer_justification_missing")
    _assert_neutralizer_coverage(block_columns=tuple(block.columns), source=source, panel=panel)
    return content_sha


def _assert_neutralizer_coverage(*, block_columns: tuple[str, ...], source: Path, panel: _StudyPanel) -> None:
    try:
        neutralizers, cols = load_neutralizer_table(neutralizer_path=source, neutralizer_cols=block_columns)
        frame = pd.DataFrame({"era": panel.eras.astype(str), "id": panel.ids.astype(str), "prediction": panel.target})
        neutralize_prediction_frame(predictions=frame, neutralizers=neutralizers, neutralizer_cols=cols, proportion=0.0)
    except NeutralizationError as exc:
        raise PortfolioError(f"study_blocked:neutralizer_coverage:{exc}") from exc


# --------------------------------------------------------------------------- #
# Panel assembly + era partition
# --------------------------------------------------------------------------- #


def _assemble_panel(*, root: Path, members: tuple[_ResolvedMember, ...]) -> _StudyPanel:
    run_ids = _ordered_run_ids(members)
    if not run_ids:
        raise PortfolioError("study_panel_no_member_runs")
    target_col = _contribution_target(root=root, run_id=run_ids[0])
    try:
        ranked, era_series, id_series, target_series = load_ranked_components(
            store_root=root, run_ids=run_ids, target_col=target_col
        )
    except EnsembleBuildError as exc:
        raise PortfolioError(f"study_panel_row_key_mismatch:{exc}") from exc
    if target_series is None:
        raise PortfolioError(f"study_panel_target_unavailable:{target_col}")

    working = pd.DataFrame(
        {
            "era": era_series.astype(str).to_numpy(),
            "id": id_series.astype(str).to_numpy(),
            target_col: target_series.to_numpy(dtype=np.float64),
        }
    )
    for member in members:
        cols = [f"pred_{run_id}" for run_id in member.run_ids]
        missing = [col for col in cols if col not in ranked.columns]
        if missing:
            raise PortfolioError(f"study_panel_missing_columns:{member.candidate_id}:{missing}")
        working[member.candidate_id] = ranked[cols].to_numpy(dtype=np.float64).mean(axis=1)

    attached = _attach_benchmark(root=root, frame=working).sort_values(["era", "id"]).reset_index(drop=True)
    eras = attached["era"].astype(str).to_numpy()
    ids = attached["id"].astype(str).to_numpy()
    ranges = panel_ops.era_ranges(eras.tolist())
    ordered_index = [[str(era), str(row_id)] for era, row_id in zip(eras, ids, strict=True)]
    panel_hash = hashlib.sha256(json.dumps(ordered_index, separators=(",", ":")).encode("utf-8")).hexdigest()
    return _StudyPanel(
        era_order=tuple(era for era, _s, _e in ranges),
        ranges=ranges,
        target=attached[target_col].to_numpy(dtype=np.float64),
        benchmark=attached[_BENCHMARK_ALIAS].to_numpy(dtype=np.float64),
        eras=eras,
        ids=ids,
        candidate_columns={
            member.candidate_id: attached[member.candidate_id].to_numpy(dtype=np.float64) for member in members
        },
        ordered_index=ordered_index,
        panel_hash=panel_hash,
    )


def _attach_benchmark(*, root: Path, frame: pd.DataFrame) -> pd.DataFrame:
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
        frame, benchmark_frame, benchmark_col, era_col="era", id_col="id", min_overlap_ratio=0.0
    )
    return attached.rename(columns={benchmark_col: _BENCHMARK_ALIAS})


@dataclass(frozen=True)
class _Split:
    """Chronological era partition + evaluation folds over the search region."""

    search_region: tuple[str, ...]
    gap: tuple[str, ...]
    holdout: tuple[str, ...]
    search_folds: tuple[tuple[str, ...], ...]

    def holdout_fingerprint(self, panel_hash: str) -> str:
        payload = {"holdout": list(self.holdout), "panel_hash": panel_hash}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _partition_eras(era_order: tuple[str, ...], *, freeze: FreezeConfig) -> _Split:
    eras = list(era_order)
    holdout_n = freeze.split.holdout_n_eras
    gap_n = freeze.split.era_gap
    if holdout_n <= 0 or holdout_n + gap_n >= len(eras):
        raise PortfolioError(f"study_split_invalid:eras={len(eras)}:holdout={holdout_n}:gap={gap_n}")
    holdout = tuple(eras[len(eras) - holdout_n :])
    gap = tuple(eras[len(eras) - holdout_n - gap_n : len(eras) - holdout_n])
    search = tuple(eras[: len(eras) - holdout_n - gap_n])
    folds = _search_folds(search, freeze=freeze)
    if not folds:
        raise PortfolioError(f"study_split_no_folds:search={len(search)}")
    return _Split(search_region=search, gap=gap, holdout=holdout, search_folds=folds)


def _search_folds(search: tuple[str, ...], *, freeze: FreezeConfig) -> tuple[tuple[str, ...], ...]:
    meta = freeze.meta_validation
    width = meta.validation_width_eras
    folds: list[tuple[str, ...]] = []
    val_start = meta.min_history_eras + meta.gap_eras
    while val_start + width <= len(search):
        folds.append(tuple(search[val_start : val_start + width]))
        val_start += meta.step_eras
    return tuple(folds)


# --------------------------------------------------------------------------- #
# Frozen manifest write + verify
# --------------------------------------------------------------------------- #


def _write_frozen(
    *,
    study_dir: Path,
    root: Path,
    freeze: FreezeConfig,
    members: tuple[_ResolvedMember, ...],
    panel: _StudyPanel,
    split: _Split,
    surface_id: str,
    contribution_target: str,
    registry: RegistryConfig,
    neutralizer_sha: str | None,
) -> None:
    study_dir.mkdir(parents=True, exist_ok=True)
    hashes = _input_hashes(root=root, members=members, panel=panel, registry=registry, neutralizer_sha=neutralizer_sha)
    manifest = {
        "schema_version": freeze.schema_version,
        "study_id": freeze.study_id,
        "experiment_id": freeze.experiment_id,
        "decision_record_id": freeze.decision_record_id,
        "exploratory": freeze.exploratory,
        "surface_id": surface_id,
        "contribution_target": contribution_target,
        "baseline_candidate_id": freeze.baseline_candidate_id,
        "members": [
            {"candidate_id": m.candidate_id, "lane_id": m.lane_id, "run_ids": list(m.run_ids)} for m in members
        ],
        "split": {
            "search_region": list(split.search_region),
            "gap": list(split.gap),
            "holdout": list(split.holdout),
            "search_folds": [list(fold) for fold in split.search_folds],
        },
        "meta_validation": freeze.meta_validation.model_dump(),
        "inference": freeze.inference.model_dump(),
        "panel_hash": panel.panel_hash,
        "hashes": hashes,
        "freeze_config": freeze.model_dump(),
    }
    (study_dir / _FROZEN_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    fingerprint = {
        "study_id": freeze.study_id,
        "decision_record_id": freeze.decision_record_id,
        "exploratory": freeze.exploratory,
        "holdout_fingerprint": split.holdout_fingerprint(panel.panel_hash),
    }
    (study_dir / _HOLDOUT_FINGERPRINT).write_text(json.dumps(fingerprint, indent=2, sort_keys=True), encoding="utf-8")


def _input_hashes(
    *,
    root: Path,
    members: tuple[_ResolvedMember, ...],
    panel: _StudyPanel,
    registry: RegistryConfig,
    neutralizer_sha: str | None,
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for run_id in _ordered_run_ids(members):
        hashes[f"member_parquet:{run_id}"] = _prediction_sha256(root=root, run_id=run_id)
    hashes["benchmark"] = _file_sha256(
        resolve_active_benchmark_predictions_path(data_root=(root / "datasets").resolve())
    )
    hashes["target"] = hashlib.sha256(np.ascontiguousarray(panel.target, dtype=np.float64).tobytes()).hexdigest()
    hashes["policy_block"] = _sha256_canonical(registry.policy.model_dump(mode="json"))
    hashes["panel"] = panel.panel_hash
    hashes["scoring_contract_version"] = hashlib.sha256(str(SCORING_CONTRACT_VERSION).encode("utf-8")).hexdigest()
    if neutralizer_sha is not None:
        hashes["neutralizer"] = neutralizer_sha
    return hashes


def _verify_frozen_inputs(
    *,
    root: Path,
    manifest: dict[str, object],
    freeze: FreezeConfig,
    members: tuple[_ResolvedMember, ...],
    panel: _StudyPanel,
) -> None:
    registry = load_registry(store_root=root)
    if registry is None:
        raise PortfolioError("study_registry_absent")
    neutralizer_sha = None
    if freeze.neutralization is not None:
        source = Path(freeze.neutralization.source_path).expanduser()
        neutralizer_sha = _file_sha256(source) if source.is_file() else None
    current = _input_hashes(root=root, members=members, panel=panel, registry=registry, neutralizer_sha=neutralizer_sha)
    recorded = manifest.get("hashes")
    recorded = recorded if isinstance(recorded, dict) else {}
    for key, value in recorded.items():
        if current.get(key) != value:
            raise PortfolioError(f"frozen_input_tampered:{key}")
    if manifest.get("panel_hash") != panel.panel_hash:
        raise PortfolioError("frozen_input_tampered:panel_hash")


# --------------------------------------------------------------------------- #
# Trials canonicalization + cap preflight
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _TrialSpec:
    """A canonicalized, validated trial ready to score."""

    trial_id: str
    selection: dict[str, tuple[str, ...]]
    lane_weights_norm: dict[str, float]
    neutralization_p: float
    spec_hash: str
    spec: dict[str, object]


def _canonicalize_trials(
    *,
    trials: TrialsConfig,
    freeze: FreezeConfig,
    members: tuple[_ResolvedMember, ...],
    cross_lane_cap: float,
) -> tuple[_TrialSpec, ...]:
    if trials.study_id != freeze.study_id:
        raise PortfolioError(f"trials_study_id_mismatch:{trials.study_id}!={freeze.study_id}")

    member_lanes = {member.candidate_id: member.lane_id for member in members}
    specs: list[_TrialSpec] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    positive_ps: set[float] = set()
    for trial in trials.trials:
        if trial.trial_id in seen_ids:
            raise PortfolioError(f"trials_duplicate_id:{trial.trial_id}")
        seen_ids.add(trial.trial_id)
        spec = _validate_trial(trial=trial, freeze=freeze, member_lanes=member_lanes, cross_lane_cap=cross_lane_cap)
        if spec.spec_hash in seen_hashes:
            raise PortfolioError(f"trials_duplicate_spec:{trial.trial_id}")
        seen_hashes.add(spec.spec_hash)
        if spec.neutralization_p > 0.0:
            positive_ps.add(spec.neutralization_p)
        specs.append(spec)

    if len(positive_ps) > _MAX_DISTINCT_NEUTRALIZATION_P:
        raise PortfolioError(f"trials_too_many_neutralization_p:{sorted(positive_ps)}")
    if len(specs) > freeze.study_trial_cap:
        raise PortfolioError(f"trials_over_cap:{len(specs)}>{freeze.study_trial_cap}")
    return tuple(specs)


def _validate_trial(
    *,
    trial: Trial,
    freeze: FreezeConfig,
    member_lanes: dict[str, str],
    cross_lane_cap: float,
) -> _TrialSpec:
    selection = trial.selection
    lane_weights = trial.lane_weights
    neutralization_p = float(trial.neutralization_p)
    trial_id = str(trial.trial_id)

    if not selection:
        raise PortfolioError(f"trial_selection_empty:{trial_id}")
    if set(lane_weights) != set(selection):
        raise PortfolioError(f"trial_weight_lane_mismatch:{trial_id}")
    for lane_id, candidate_ids in selection.items():
        if not candidate_ids:
            raise PortfolioError(f"trial_lane_selection_empty:{trial_id}:{lane_id}")
        for candidate_id in candidate_ids:
            if member_lanes.get(candidate_id) != lane_id:
                raise PortfolioError(f"trial_candidate_not_member:{trial_id}:{lane_id}:{candidate_id}")
    for lane_id, weight in lane_weights.items():
        if weight < 0.0:
            raise PortfolioError(f"trial_negative_weight:{trial_id}:{lane_id}")
        if weight > cross_lane_cap:
            raise PortfolioError(f"trial_weight_over_cap:{trial_id}:{lane_id}:{weight}>{cross_lane_cap}")
    total = float(sum(lane_weights.values()))
    if total <= 0.0:
        raise PortfolioError(f"trial_weight_total_zero:{trial_id}")
    if neutralization_p > 0.0 and freeze.neutralization is None:
        raise PortfolioError(f"trial_neutralization_without_block:{trial_id}")

    normalized = {lane_id: round(weight / total, 12) for lane_id, weight in lane_weights.items()}
    canonical_selection = {lane_id: sorted(set(candidate_ids)) for lane_id, candidate_ids in selection.items()}
    spec: dict[str, object] = {
        "selection": {lane_id: canonical_selection[lane_id] for lane_id in sorted(canonical_selection)},
        "lane_weights": {lane_id: normalized[lane_id] for lane_id in sorted(normalized)},
        "neutralization_p": neutralization_p,
    }
    spec_hash = _sha256_canonical(spec)
    return _TrialSpec(
        trial_id=trial_id,
        selection={lane_id: tuple(canonical_selection[lane_id]) for lane_id in canonical_selection},
        lane_weights_norm=normalized,
        neutralization_p=neutralization_p,
        spec_hash=spec_hash,
        spec=spec,
    )


# --------------------------------------------------------------------------- #
# Blending + scoring
# --------------------------------------------------------------------------- #


def _blend_column(
    *, panel: _StudyPanel, selection: dict[str, tuple[str, ...]], weights_norm: dict[str, float]
) -> np.ndarray:
    lane_columns: list[np.ndarray] = []
    lane_weight: list[float] = []
    for lane_id in sorted(selection):
        candidate_ids = selection[lane_id]
        stack = np.column_stack([panel.candidate_columns[cid] for cid in candidate_ids])
        lane_columns.append(stack.mean(axis=1))
        lane_weight.append(weights_norm[lane_id])
    matrix = np.column_stack(lane_columns)
    return np.asarray(matrix @ np.asarray(lane_weight, dtype=np.float64), dtype=np.float64)


def _neutralize_column(
    *, column: np.ndarray, panel: _StudyPanel, freeze: FreezeConfig, proportion: float
) -> np.ndarray:
    block = freeze.neutralization
    if block is None or proportion <= 0.0:
        return column
    neutralizers, cols = load_neutralizer_table(
        neutralizer_path=Path(block.source_path).expanduser(), neutralizer_cols=tuple(block.columns)
    )
    frame = pd.DataFrame({"era": panel.eras.astype(str), "id": panel.ids.astype(str), "prediction": column})
    out = neutralize_prediction_frame(
        predictions=frame,
        neutralizers=neutralizers,
        neutralizer_cols=cols,
        proportion=proportion,
        mode=block.mode,  # type: ignore[arg-type]
        rank_output=block.rank_output,
    )
    return out["prediction"].to_numpy(dtype=np.float64)


def _era_bmc_map(*, panel: _StudyPanel, column: np.ndarray) -> dict[str, float]:
    era_bmc = panel_ops.score_on_panel(column.reshape(-1, 1), panel.target, panel.benchmark, panel.ranges).bmc[:, 0]
    return {era: float(era_bmc[index]) for index, era in enumerate(panel.era_order)}


def _pooled(era_bmc: dict[str, float], eras: tuple[str, ...]) -> float | None:
    values = [era_bmc[era] for era in eras if era in era_bmc and np.isfinite(era_bmc[era])]
    return float(np.mean(values)) if values else None


@dataclass(frozen=True)
class _BaselineScores:
    era_bmc: dict[str, float]
    column: np.ndarray


def _baseline_bmc(*, panel: _StudyPanel, freeze: FreezeConfig, split: _Split) -> _BaselineScores:
    if freeze.baseline_candidate_id not in panel.candidate_columns:
        raise PortfolioError(f"study_baseline_not_member:{freeze.baseline_candidate_id}")
    column = panel.candidate_columns[freeze.baseline_candidate_id]
    return _BaselineScores(era_bmc=_era_bmc_map(panel=panel, column=column), column=column)


def _paired_diff(
    *,
    trial_bmc: dict[str, float],
    baseline_bmc: dict[str, float],
    eras: tuple[str, ...],
    freeze: FreezeConfig,
) -> tuple[float | None, float | None, float | None, float | None]:
    diffs = np.asarray(
        [trial_bmc[era] - baseline_bmc[era] for era in eras if _both_finite(trial_bmc, baseline_bmc, era)],
        dtype=np.float64,
    )
    if diffs.size == 0:
        return None, None, None, None
    block = freeze.inference.block_length_eras
    if diffs.size >= 2 * block:
        try:
            boot = paired_block_bootstrap(
                diffs,
                block_length_eras=block,
                n_resamples=freeze.inference.n_resamples,
                rng_seed=freeze.inference.rng_seed,
            )
        except DiversityMetricError:
            return float(np.mean(diffs)), None, None, None
        return boot.mean, boot.ci90_low, boot.ci90_high, boot.prob_positive
    return float(np.mean(diffs)), None, None, None


# --------------------------------------------------------------------------- #
# Run execution + ledger
# --------------------------------------------------------------------------- #


def _execute_trials(
    *,
    study_dir: Path,
    panel: _StudyPanel,
    freeze: FreezeConfig,
    split: _Split,
    specs: tuple[_TrialSpec, ...],
    baseline: _BaselineScores,
) -> StudyRunResult:
    ledger_path = study_dir / _LEDGER
    existing = _read_ledger(ledger_path)
    complete_by_id = {line["trial_id"]: line for line in existing if line.get("status") == _LEDGER_COMPLETE}
    search_eras = split.search_region

    executed = 0
    skipped = 0
    superseded = 0
    results: list[StudyTrialResult] = []
    for spec in specs:
        prior = complete_by_id.get(spec.trial_id)
        if prior is not None and prior.get("spec_hash") == spec.spec_hash:
            skipped += 1
            results.append(_trial_result_from_line(prior))
            continue
        if prior is not None:
            _append_ledger(
                ledger_path,
                {
                    "trial_id": spec.trial_id,
                    "status": _LEDGER_SUPERSEDED,
                    "supersedes_spec_hash": prior.get("spec_hash"),
                    "scored_at": _utc_now_iso(),
                },
            )
            superseded += 1
        line = _score_trial(panel=panel, freeze=freeze, spec=spec, baseline=baseline, search_eras=search_eras)
        _append_ledger(ledger_path, line)
        executed += 1
        results.append(_trial_result_from_line(line))

    return StudyRunResult(
        study_id=freeze.study_id,
        study_dir=str(study_dir),
        executed=executed,
        skipped=skipped,
        superseded=superseded,
        trial_cap=freeze.study_trial_cap,
        ledger_path=str(ledger_path),
        trials=tuple(results),
    )


def _score_trial(
    *,
    panel: _StudyPanel,
    freeze: FreezeConfig,
    spec: _TrialSpec,
    baseline: _BaselineScores,
    search_eras: tuple[str, ...],
) -> dict[str, object]:
    started = time.perf_counter()
    blended = _blend_column(panel=panel, selection=spec.selection, weights_norm=spec.lane_weights_norm)
    blended = _neutralize_column(column=blended, panel=panel, freeze=freeze, proportion=spec.neutralization_p)
    trial_bmc = _era_bmc_map(panel=panel, column=blended)
    fold_bmc = [_pooled(trial_bmc, fold) for fold in _folds_for(freeze=freeze, search_eras=search_eras)]
    pooled = _pooled(trial_bmc, search_eras)
    baseline_pooled = _pooled(baseline.era_bmc, search_eras)
    mean, ci_low, ci_high, prob = _paired_diff(
        trial_bmc=trial_bmc, baseline_bmc=baseline.era_bmc, eras=search_eras, freeze=freeze
    )
    return {
        "trial_id": spec.trial_id,
        "status": _LEDGER_COMPLETE,
        "spec_hash": spec.spec_hash,
        "spec": spec.spec,
        "fold_bmc": fold_bmc,
        "n_folds": len(fold_bmc),
        "pooled_search_bmc": pooled,
        "baseline_pooled_search_bmc": baseline_pooled,
        "diff": {"mean": mean, "ci90_low": ci_low, "ci90_high": ci_high, "prob_positive": prob},
        "wall_seconds": round(time.perf_counter() - started, 6),
        "scored_at": _utc_now_iso(),
    }


def _folds_for(*, freeze: FreezeConfig, search_eras: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    return _search_folds(search_eras, freeze=freeze)


def _trial_result_from_line(line: dict[str, object]) -> StudyTrialResult:
    diff = line.get("diff")
    diff = diff if isinstance(diff, dict) else {}
    return StudyTrialResult(
        trial_id=str(line.get("trial_id")),
        pooled_search_bmc=_opt_float(line.get("pooled_search_bmc")),
        baseline_pooled_search_bmc=_opt_float(line.get("baseline_pooled_search_bmc")),
        diff_mean=_opt_float(diff.get("mean")),
        diff_ci90_low=_opt_float(diff.get("ci90_low")),
        diff_ci90_high=_opt_float(diff.get("ci90_high")),
        diff_prob_positive=_opt_float(diff.get("prob_positive")),
        n_folds=int(line.get("n_folds") or 0),
        status=str(line.get("status")),
    )


# --------------------------------------------------------------------------- #
# Finalize
# --------------------------------------------------------------------------- #


def _finalize_selection(
    *,
    study_dir: Path,
    manifest: dict[str, object],
    freeze: FreezeConfig,
    members: tuple[_ResolvedMember, ...],
    panel: _StudyPanel,
    split: _Split,
    select: str,
) -> StudyFinalizeResult:
    baseline = _baseline_bmc(panel=panel, freeze=freeze, split=split)
    is_baseline = select == "baseline"
    if is_baseline:
        column = baseline.column
        selection: dict[str, tuple[str, ...]] = {members[0].lane_id: (freeze.baseline_candidate_id,)}
        weights_norm = {members[0].lane_id: 1.0}
        pooled_search = _pooled(baseline.era_bmc, split.search_region)
        neutralization_p = 0.0
    else:
        ledger_line = _complete_ledger_specs(study_dir).get(select)
        if ledger_line is None:
            raise PortfolioError(f"study_trial_not_executed:{select}")
        spec = _spec_from_line(ledger_line)
        column = _blend_column(panel=panel, selection=spec.selection, weights_norm=spec.lane_weights_norm)
        column = _neutralize_column(column=column, panel=panel, freeze=freeze, proportion=spec.neutralization_p)
        selection = spec.selection
        weights_norm = spec.lane_weights_norm
        pooled_search = _opt_float(ledger_line.get("pooled_search_bmc"))
        neutralization_p = spec.neutralization_p

    trial_bmc = _era_bmc_map(panel=panel, column=column)
    holdout_bmc = _pooled(trial_bmc, split.holdout)
    baseline_holdout = _pooled(baseline.era_bmc, split.holdout)
    holdout_diff = (
        (holdout_bmc - baseline_holdout) if (holdout_bmc is not None and baseline_holdout is not None) else None
    )
    degradation = (pooled_search - holdout_bmc) if (pooled_search is not None and holdout_bmc is not None) else None
    _, ci_low, ci_high, prob = _paired_diff(
        trial_bmc=trial_bmc, baseline_bmc=baseline.era_bmc, eras=split.holdout, freeze=freeze
    )

    artifacts_dir = study_dir / _ARTIFACTS_DIR
    _write_holdout_result(
        study_dir=study_dir,
        select=select,
        is_baseline=is_baseline,
        holdout_bmc=holdout_bmc,
        baseline_holdout=baseline_holdout,
        holdout_diff=holdout_diff,
        degradation=degradation,
        ci_low=ci_low,
        ci_high=ci_high,
        prob=prob,
    )
    _write_study_artifacts(
        artifacts_dir=artifacts_dir,
        manifest=manifest,
        freeze=freeze,
        members=members,
        panel=panel,
        column=column,
        selection=selection,
        weights_norm=weights_norm,
        select=select,
        is_baseline=is_baseline,
        neutralization_p=neutralization_p,
    )
    (study_dir / _SEALED).write_text(
        json.dumps(
            {
                "study_id": freeze.study_id,
                "selected_trial": select,
                "is_baseline": is_baseline,
                "sealed_at": _utc_now_iso(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return StudyFinalizeResult(
        study_id=freeze.study_id,
        study_dir=str(study_dir),
        selected_trial=select,
        is_baseline=is_baseline,
        holdout_bmc=holdout_bmc,
        baseline_holdout_bmc=baseline_holdout,
        holdout_diff=holdout_diff,
        degradation=degradation,
        holdout_ci90_low=ci_low,
        holdout_ci90_high=ci_high,
        holdout_prob_positive=prob,
        sealed=True,
        artifacts_dir=str(artifacts_dir),
    )


def _write_holdout_result(
    *,
    study_dir: Path,
    select: str,
    is_baseline: bool,
    holdout_bmc: float | None,
    baseline_holdout: float | None,
    holdout_diff: float | None,
    degradation: float | None,
    ci_low: float | None,
    ci_high: float | None,
    prob: float | None,
) -> None:
    payload = {
        "selected_trial": select,
        "is_baseline": is_baseline,
        "holdout_bmc": holdout_bmc,
        "baseline_holdout_bmc": baseline_holdout,
        "holdout_diff": holdout_diff,
        "degradation_vs_search": degradation,
        "holdout_ci90_low": ci_low,
        "holdout_ci90_high": ci_high,
        "holdout_prob_positive": prob,
        "finalized_at": _utc_now_iso(),
    }
    (study_dir / _HOLDOUT_RESULT).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Study artifacts (ensemble-format reuse)
# --------------------------------------------------------------------------- #


def _write_study_artifacts(
    *,
    artifacts_dir: Path,
    manifest: dict[str, object],
    freeze: FreezeConfig,
    members: tuple[_ResolvedMember, ...],
    panel: _StudyPanel,
    column: np.ndarray,
    selection: dict[str, tuple[str, ...]],
    weights_norm: dict[str, float],
    select: str,
    is_baseline: bool,
    neutralization_p: float,
) -> None:
    from numereng.platform.parquet import write_parquet

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame({"era": panel.eras, "id": panel.ids, "prediction": column})
    write_parquet(predictions, artifacts_dir / "predictions.parquet", index=False)

    run_weights = _effective_run_weights(
        members=members,
        selection=selection,
        weights_norm=weights_norm,
        is_baseline=is_baseline,
        baseline_candidate_id=freeze.baseline_candidate_id,
    )
    weights_frame = pd.DataFrame(
        {
            "run_id": list(run_weights),
            "weight": [run_weights[r] for r in run_weights],
            "rank": list(range(len(run_weights))),
        }
    )
    write_parquet(weights_frame, artifacts_dir / "weights.parquet", index=False)

    selected_candidates = sorted({cid for cids in selection.values() for cid in cids})
    corr = pd.DataFrame({cid: panel.candidate_columns[cid] for cid in selected_candidates}).corr()
    write_parquet(corr.replace([np.inf, -np.inf], np.nan), artifacts_dir / "correlation_matrix.parquet", index=True)

    era_bmc = panel_ops.score_on_panel(column.reshape(-1, 1), panel.target, panel.benchmark, panel.ranges).bmc[:, 0]
    era_metrics = pd.DataFrame({"era": list(panel.era_order), "bmc": era_bmc})
    write_parquet(era_metrics, artifacts_dir / "era_metrics.parquet", index=False)

    lineage = {
        "study_id": freeze.study_id,
        "experiment_id": freeze.experiment_id,
        "method": "combination_study",
        "selected_trial": select,
        "is_baseline": is_baseline,
        "target": manifest.get("contribution_target"),
        "neutralization_p": neutralization_p,
        "run_ids": list(run_weights),
        "weights": run_weights,
        "rows": int(len(predictions)),
        "artifacts": {
            "predictions": "predictions.parquet",
            "weights": "weights.parquet",
            "correlation_matrix": "correlation_matrix.parquet",
            "era_metrics": "era_metrics.parquet",
        },
    }
    (artifacts_dir / "lineage.json").write_text(json.dumps(lineage, indent=2, sort_keys=True), encoding="utf-8")


def _effective_run_weights(
    *,
    members: tuple[_ResolvedMember, ...],
    selection: dict[str, tuple[str, ...]],
    weights_norm: dict[str, float],
    is_baseline: bool,
    baseline_candidate_id: str,
) -> dict[str, float]:
    runs_by_candidate = {member.candidate_id: member.run_ids for member in members}
    weights: dict[str, float] = {}
    if is_baseline:
        run_ids = runs_by_candidate.get(baseline_candidate_id, ())
        for run_id in run_ids:
            weights[run_id] = round(1.0 / len(run_ids), 12) if run_ids else 0.0
        return weights
    for lane_id, candidate_ids in selection.items():
        lane_weight = weights_norm.get(lane_id, 0.0)
        per_candidate = lane_weight / len(candidate_ids) if candidate_ids else 0.0
        for candidate_id in candidate_ids:
            run_ids = runs_by_candidate.get(candidate_id, ())
            per_run = per_candidate / len(run_ids) if run_ids else 0.0
            for run_id in run_ids:
                weights[run_id] = round(weights.get(run_id, 0.0) + per_run, 12)
    return weights


# --------------------------------------------------------------------------- #
# Manifest / ledger / study location helpers
# --------------------------------------------------------------------------- #


def _study_dir(*, root: Path, experiment_id: str, study_id: str) -> Path:
    experiment_dir = root / "experiments" / experiment_id
    return experiment_dir / _STUDY_DIRNAME / study_id


def _locate_study(*, root: Path, study_id: str, experiment_id: str | None) -> Path:
    if experiment_id is not None:
        study_dir = _study_dir(root=root, experiment_id=experiment_id, study_id=study_id)
        if (study_dir / _FROZEN_MANIFEST).is_file():
            return study_dir
        raise PortfolioError(f"study_not_found:{study_id}")
    matches = sorted((root / "experiments").glob(f"*/{_STUDY_DIRNAME}/{study_id}"))
    for match in matches:
        if (match / _FROZEN_MANIFEST).is_file():
            return match
    raise PortfolioError(f"study_not_found:{study_id}")


def _guard_holdout_reuse(*, root: Path, freeze: FreezeConfig, holdout_fingerprint: str) -> None:
    if freeze.exploratory:
        return
    for path in sorted((root / "experiments").glob(f"*/{_STUDY_DIRNAME}/*/{_HOLDOUT_FINGERPRINT}")):
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        if payload.get("holdout_fingerprint") != holdout_fingerprint:
            continue
        if bool(payload.get("exploratory")):
            continue
        if payload.get("decision_record_id") == freeze.decision_record_id:
            raise PortfolioError(f"study_blocked:holdout_reuse:{path.parent.name}")


def _load_frozen(study_dir: Path) -> dict[str, object]:
    payload = _load_json(study_dir / _FROZEN_MANIFEST)
    if not isinstance(payload, dict):
        raise PortfolioError(f"study_frozen_manifest_unreadable:{study_dir}")
    return payload


def _reject_if_sealed(study_dir: Path) -> None:
    if (study_dir / _SEALED).is_file():
        raise PortfolioError(f"study_sealed:{study_dir.name}")


def _members_from_manifest(manifest: dict[str, object]) -> tuple[_ResolvedMember, ...]:
    raw = manifest.get("members")
    raw = raw if isinstance(raw, list) else []
    members: list[_ResolvedMember] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        members.append(
            _ResolvedMember(
                candidate_id=str(item.get("candidate_id")),
                lane_id=str(item.get("lane_id")),
                run_ids=tuple(str(run_id) for run_id in (item.get("run_ids") or [])),
                surface_ids=(),
            )
        )
    if not members:
        raise PortfolioError("study_manifest_no_members")
    return tuple(members)


def _split_from_manifest(manifest: dict[str, object]) -> _Split:
    raw = manifest.get("split")
    raw = raw if isinstance(raw, dict) else {}
    return _Split(
        search_region=tuple(str(era) for era in (raw.get("search_region") or [])),
        gap=tuple(str(era) for era in (raw.get("gap") or [])),
        holdout=tuple(str(era) for era in (raw.get("holdout") or [])),
        search_folds=tuple(tuple(str(era) for era in fold) for fold in (raw.get("search_folds") or [])),
    )


def _read_ledger(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    lines: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            lines.append(payload)
    return lines


def _append_ledger(path: Path, line: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, sort_keys=True) + "\n")


def _complete_ledger_specs(study_dir: Path) -> dict[str, dict[str, object]]:
    complete: dict[str, dict[str, object]] = {}
    for line in _read_ledger(study_dir / _LEDGER):
        if line.get("status") == _LEDGER_COMPLETE:
            complete[str(line.get("trial_id"))] = line
    return complete


def _spec_from_line(line: dict[str, object]) -> _TrialSpec:
    spec = line.get("spec")
    spec = spec if isinstance(spec, dict) else {}
    selection_raw = spec.get("selection")
    selection_raw = selection_raw if isinstance(selection_raw, dict) else {}
    weights_raw = spec.get("lane_weights")
    weights_raw = weights_raw if isinstance(weights_raw, dict) else {}
    return _TrialSpec(
        trial_id=str(line.get("trial_id")),
        selection={lane_id: tuple(str(cid) for cid in cids) for lane_id, cids in selection_raw.items()},
        lane_weights_norm={lane_id: float(weight) for lane_id, weight in weights_raw.items()},
        neutralization_p=float(spec.get("neutralization_p") or 0.0),
        spec_hash=str(line.get("spec_hash")),
        spec=spec,
    )


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _load_freeze(config_path: str | Path) -> FreezeConfig:
    payload = _load_json(Path(config_path))
    if payload is None:
        raise PortfolioError(f"study_freeze_config_unreadable:{config_path}")
    return load_freeze_config(payload)


def _load_trials(trials_path: str | Path) -> TrialsConfig:
    payload = _load_json(Path(trials_path))
    if payload is None:
        raise PortfolioError(f"study_trials_config_unreadable:{trials_path}")
    return load_trials_config(payload)


def _require_policy_value(registry: RegistryConfig, field_name: str) -> float:
    value = getattr(registry.policy, field_name)
    if value is None:
        raise PortfolioError(f"study_blocked:policy_unset:{field_name}")
    return float(value)


def _ordered_run_ids(members: tuple[_ResolvedMember, ...]) -> tuple[str, ...]:
    ordered: list[str] = []
    for member in members:
        for run_id in member.run_ids:
            if run_id not in ordered:
                ordered.append(run_id)
    return tuple(ordered)


def _contribution_target(*, root: Path, run_id: str) -> str:
    provenance = _load_json(root / "runs" / run_id / "score_provenance.json")
    columns = provenance.get("columns") if isinstance(provenance, dict) else None
    targets = columns.get("contribution_target_cols") if isinstance(columns, dict) else None
    if not isinstance(targets, list) or not targets:
        raise PortfolioError(f"study_contribution_target_unresolved:{run_id}")
    return str(sorted(str(item) for item in targets)[0])


def _prediction_sha256(*, root: Path, run_id: str) -> str:
    manifest = _load_json(root / "runs" / run_id / "run.json")
    artifacts = manifest.get("artifacts") if isinstance(manifest, dict) else None
    rel = artifacts.get("predictions") if isinstance(artifacts, dict) else None
    path = (root / "runs" / run_id / rel) if isinstance(rel, str) and rel else None
    if path is None or not path.is_file():
        matches = sorted((root / "runs" / run_id / "artifacts" / "predictions").glob("*.parquet"))
        path = matches[0] if matches else None
    if path is None:
        raise PortfolioError(f"study_prediction_missing:{run_id}")
    return _file_sha256(path)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha256_canonical(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _both_finite(left: dict[str, float], right: dict[str, float], era: str) -> bool:
    return era in left and era in right and np.isfinite(left[era]) and np.isfinite(right[era])


def _opt_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _load_json(path: Path) -> dict[str, object] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["study_finalize", "study_freeze", "study_run", "study_status"]
