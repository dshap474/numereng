"""Live resolution of lane/candidate facts from state + journal + metrics + artifacts.

Nothing here is stored in the registry: metrics come from each run's metrics.json
(disk is canonical), recipe grouping comes from the agentic journal, artifact
modes come from the public store helper, and surface ids come from surface.py.

Hard-fail rules (§2.2): malformed journal lines abort the lane's resolution;
duplicate (recipe, seed) rows are surfaced as diagnostics, never silently
collapsed to the latest.

USAGE:
    from numereng.features.research_portfolio.resolve import resolve_lane
    lane_status = resolve_lane(store_root=".numereng", lane=lane, believed_best="config_044_s42.json")
"""

from __future__ import annotations

import json
from pathlib import Path

from numereng.agentic_research.engine import context as ar_context
from numereng.agentic_research.engine.aggregate import (
    aggregate_recipes,
    load_config_cache,
    observed_seed_noise,
    recipe_key,
)
from numereng.config.research_portfolio import RegistryLane
from numereng.features.experiments import ExperimentError, get_experiment
from numereng.features.research_portfolio.surface import compute_surface_id
from numereng.features.research_portfolio.types import (
    EVIDENCE_TIERS,
    REQUIRED_TRIO_SEEDS,
    CandidateFact,
    LaneStatus,
    PortfolioValidationError,
    SeedFact,
)
from numereng.features.store import classify_run_mode, resolve_store_root
from numereng.features.training.run_store import compute_config_hash

# --------------------------------------------------------------------------- #
# Lane resolution
# --------------------------------------------------------------------------- #


def resolve_lane(
    *,
    store_root: str | Path = ".numereng",
    lane: RegistryLane,
) -> LaneStatus:
    """Resolve one lane's candidate facts, evidence, drift, and blockers."""

    root = resolve_store_root(store_root)
    blockers: list[str] = []

    scale_id = lane.experiments.scale
    superseded_ids = {item.experiment_id for item in lane.experiments.superseded}

    experiment = None
    configs: dict[str, dict[str, object]] = {}
    journal_entries: list[dict[str, object]] = []
    state: dict[str, object] = {}

    if scale_id is None:
        blockers.append("scale_experiment_unset")
    elif scale_id in superseded_ids:
        blockers.append(f"scale_experiment_superseded:{scale_id}")
    else:
        try:
            experiment = get_experiment(store_root=root, experiment_id=scale_id)
        except ExperimentError:
            blockers.append(f"scale_experiment_not_found:{scale_id}")
        if experiment is not None:
            experiment_dir = experiment.manifest_path.parent
            configs = load_config_cache(experiment_dir / "configs")
            journal_entries = _read_journal_strict(
                experiment_dir / "agentic_research" / "journal.jsonl",
                lane_id=lane.lane_id,
            )
            state = _load_state(experiment_dir / "agentic_research" / "state.json")

    candidates = tuple(
        _resolve_candidate(
            root=root,
            candidate=candidate,
            scale_id=scale_id,
            configs=configs,
            journal_entries=journal_entries,
        )
        for candidate in lane.candidates
    )
    for candidate in candidates:
        blockers.extend(f"{candidate.candidate_id}:{item}" for item in candidate.blockers)

    lane_noise = _lane_seed_noise(journal_entries, configs)
    lane_surface_match = _lane_surface_match(candidates)
    drift = _drift(lane=lane, state=state)
    if drift is not None:
        blockers.append(drift)

    last_error = state.get("last_error")
    if isinstance(last_error, str) and last_error:
        blockers.append(f"state_last_error:{last_error}")

    return LaneStatus(
        lane_id=lane.lane_id,
        axis=lane.axis,
        structural=lane.structural,
        research_stage_asserted=lane.research_stage,
        research_stage_evidenced=_lane_evidence_tier(candidates),
        deployment_stage_asserted=lane.deployment_stage,
        deployment_stage_evidenced=("live-bound" if lane.live is not None else "unbound"),
        combination_stage_asserted=lane.combination_stage,
        combination_stage_evidenced=("artifact-ready" if lane_surface_match else "not-ready"),
        tranche_rounds_completed=_int_or_none(state.get("total_rounds_completed")),
        tranche_approved_rounds=lane.envelope.approved_tranche_rounds,
        tranche_max_rounds=lane.envelope.max_rounds,
        observed_seed_noise=lane_noise,
        candidates=candidates,
        surface_match=lane_surface_match,
        drift=drift,
        blockers=tuple(blockers),
    )


# --------------------------------------------------------------------------- #
# Candidate resolution
# --------------------------------------------------------------------------- #


def _resolve_candidate(
    *,
    root: Path,
    candidate: object,
    scale_id: str | None,
    configs: dict[str, dict[str, object]],
    journal_entries: list[dict[str, object]],
) -> CandidateFact:
    anchor_config = candidate.anchor_config  # type: ignore[attr-defined]
    candidate_id = candidate.candidate_id  # type: ignore[attr-defined]
    role = candidate.role  # type: ignore[attr-defined]
    blockers: list[str] = []

    anchor = configs.get(anchor_config)
    if anchor is None:
        return CandidateFact(
            candidate_id=candidate_id,
            role=role,
            anchor_config=anchor_config,
            recipe_key=None,
            evidence_tier="discovery",
            seeds_present=(),
            trio_complete=False,
            trio_bmc_mean=None,
            bmc_sd=None,
            per_seed=(),
            surface_ids=(),
            surface_match=False,
            blockers=("anchor_config_not_found",),
        )

    key = recipe_key(anchor)
    rows_by_seed = _match_rows(key, configs=configs, journal_entries=journal_entries)

    per_seed: list[SeedFact] = []
    for seed in sorted(rows_by_seed, key=lambda value: (value is None, value)):
        rows = rows_by_seed[seed]
        chosen = rows[-1]
        duplicate_run_ids = tuple(str(row["run_id"]) for row in rows if row.get("run_id"))
        if len(rows) > 1:
            blockers.append(f"duplicate_seed_runs:{seed}")
        per_seed.append(
            _seed_fact(
                root=root,
                seed=seed,
                row=chosen,
                duplicate_run_ids=duplicate_run_ids if len(rows) > 1 else (),
            )
        )

    for fact in per_seed:
        if fact.config_hash_ok is False:
            blockers.append(f"config_hash_mismatch:{fact.run_id}")
        if fact.artifact_mode in {"missing", "incomplete"}:
            blockers.append(f"artifact_{fact.artifact_mode}:{fact.run_id}")

    seeds_present = tuple(sorted(seed for seed in rows_by_seed if seed is not None))
    trio_complete = set(seeds_present) == set(REQUIRED_TRIO_SEEDS)
    trio_bmcs = [fact.bmc for fact in per_seed if fact.seed in REQUIRED_TRIO_SEEDS and fact.bmc is not None]
    trio_bmc_mean = (sum(trio_bmcs) / len(trio_bmcs)) if trio_bmcs else None
    bmc_sd = _sample_std(trio_bmcs)

    surface_ids = tuple(fact.surface_id for fact in per_seed if fact.surface_id is not None)
    surface_match = len(surface_ids) == len(per_seed) and len(set(surface_ids)) == 1 and bool(surface_ids)

    return CandidateFact(
        candidate_id=candidate_id,
        role=role,
        anchor_config=anchor_config,
        recipe_key=key,
        evidence_tier=_candidate_evidence_tier(per_seed, trio_complete=trio_complete, scale_id=scale_id),
        seeds_present=seeds_present,
        trio_complete=trio_complete,
        trio_bmc_mean=trio_bmc_mean,
        bmc_sd=bmc_sd,
        per_seed=tuple(per_seed),
        surface_ids=surface_ids,
        surface_match=surface_match,
        blockers=tuple(blockers),
    )


def _seed_fact(
    *,
    root: Path,
    seed: int | None,
    row: dict[str, object],
    duplicate_run_ids: tuple[str, ...],
) -> SeedFact:
    run_id = row.get("run_id")
    run_id = str(run_id) if run_id else None
    config = row.get("config")
    config = str(config) if isinstance(config, str) else None

    artifact_mode = "missing"
    training_profile: str | None = None
    experiment_id: str | None = None
    config_hash_ok: bool | None = None
    surface_id: str | None = None
    surface_reason: str | None = None
    disk_bmc: float | None = None

    if run_id is not None:
        run_dir = root / "runs" / run_id
        artifact_mode = classify_run_mode(run_dir=run_dir)
        manifest = _load_json(run_dir / "run.json")
        training_profile = _training_profile(manifest)
        experiment_id = _manifest_experiment_id(manifest)
        config_hash_ok = _config_hash_ok(root=root, run_id=run_id, config=config, manifest=manifest)
        disk_bmc = ar_context.run_primary_metric_from_disk(root=root, run_id=run_id)
        surface = compute_surface_id(run_dir=run_dir)
        surface_id = surface.surface_id
        surface_reason = surface.unavailable_reason

    journal_bmc = _float_or_none(row.get("metric"))
    delta = (journal_bmc - disk_bmc) if (journal_bmc is not None and disk_bmc is not None) else None

    return SeedFact(
        seed=seed,
        run_id=run_id,
        bmc=disk_bmc if disk_bmc is not None else journal_bmc,
        fnc=(ar_context.run_fnc_mean_from_disk(root=root, run_id=run_id) if run_id else None),
        artifact_mode=artifact_mode,
        training_profile=training_profile,
        experiment_id=experiment_id,
        config=config,
        config_hash_ok=config_hash_ok,
        surface_id=surface_id,
        surface_unavailable_reason=surface_reason,
        journal_vs_disk_bmc_delta=delta,
        duplicate_run_ids=duplicate_run_ids,
    )


# --------------------------------------------------------------------------- #
# Journal + state readers
# --------------------------------------------------------------------------- #


def _read_journal_strict(path: Path, *, lane_id: str) -> list[dict[str, object]]:
    """Read journal.jsonl, hard-failing on any malformed line (§2.2.2)."""

    if not path.is_file():
        return []
    entries: list[dict[str, object]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PortfolioValidationError(f"malformed_journal_line:{lane_id}:{path}:{number}") from exc
        if not isinstance(payload, dict):
            raise PortfolioValidationError(f"malformed_journal_line:{lane_id}:{path}:{number}")
        entries.append(payload)
    return entries


def _load_state(path: Path) -> dict[str, object]:
    payload = _load_json(path)
    return payload if isinstance(payload, dict) else {}


def _match_rows(
    key: str,
    *,
    configs: dict[str, dict[str, object]],
    journal_entries: list[dict[str, object]],
) -> dict[int | None, list[dict[str, object]]]:
    """Group completed journal rows sharing the recipe key by seed (chronological)."""

    grouped: dict[int | None, list[dict[str, object]]] = {}
    for entry in journal_entries:
        if entry.get("status") != "completed":
            continue
        config_name = entry.get("config")
        config = configs.get(config_name) if isinstance(config_name, str) else None
        if config is None or recipe_key(config) != key:
            continue
        seed = entry.get("seed")
        seed = seed if isinstance(seed, int) and not isinstance(seed, bool) else None
        grouped.setdefault(seed, []).append(entry)
    return grouped


# --------------------------------------------------------------------------- #
# Evidence + drift + noise
# --------------------------------------------------------------------------- #


def _candidate_evidence_tier(per_seed: list[SeedFact], *, trio_complete: bool, scale_id: str | None) -> str:
    if not trio_complete:
        return "discovery"
    trio = [fact for fact in per_seed if fact.seed in REQUIRED_TRIO_SEEDS]
    if scale_id is not None and trio and all(fact.experiment_id == scale_id for fact in trio):
        return "scale-confirmed"
    return "seed-confirmed"


def _lane_evidence_tier(candidates: tuple[CandidateFact, ...]) -> str:
    if not candidates:
        return "discovery"
    return min(candidates, key=lambda item: EVIDENCE_TIERS.index(item.evidence_tier)).evidence_tier


def _lane_surface_match(candidates: tuple[CandidateFact, ...]) -> bool:
    resolvable = [item for item in candidates if item.surface_ids]
    if len(resolvable) < 1 or len(resolvable) != len(candidates):
        return False
    ids = {surface for item in resolvable for surface in item.surface_ids}
    return len(ids) == 1 and all(item.surface_match for item in resolvable)


def _drift(*, lane: RegistryLane, state: dict[str, object]) -> str | None:
    expected = lane.expected_believed_best
    if expected is None:
        return None
    believed = state.get("believed_best")
    actual = believed.get("config") if isinstance(believed, dict) else None
    if actual is None:
        return f"drift:expected={expected},actual=none"
    if actual != expected:
        return f"drift:expected={expected},actual={actual}"
    return None


def _lane_seed_noise(
    journal_entries: list[dict[str, object]],
    configs: dict[str, dict[str, object]],
) -> float | None:
    if not journal_entries or not configs:
        return None
    return observed_seed_noise(aggregate_recipes(journal_entries, configs=configs))


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _config_hash_ok(
    *,
    root: Path,
    run_id: str,
    config: str | None,
    manifest: dict[str, object] | None,
) -> bool | None:
    if config is None or manifest is None:
        return None
    recorded = manifest.get("config")
    recorded_hash = recorded.get("hash") if isinstance(recorded, dict) else None
    if not isinstance(recorded_hash, str) or not recorded_hash:
        return None
    config_path = _config_source_path(root=root, run_id=run_id, config=config, manifest=manifest)
    loaded = _load_json(config_path) if config_path is not None else None
    if loaded is None:
        return None
    return compute_config_hash(loaded) == recorded_hash


def _config_source_path(
    *,
    root: Path,
    run_id: str,
    config: str,
    manifest: dict[str, object],
) -> Path | None:
    experiment_id = _manifest_experiment_id(manifest)
    if experiment_id is not None:
        candidate = root / "experiments" / experiment_id / "configs" / config
        if candidate.is_file():
            return candidate
    fallback = root / "runs" / run_id / "resolved.json"
    return fallback if fallback.is_file() else None


def _training_profile(manifest: dict[str, object] | None) -> str | None:
    if not isinstance(manifest, dict):
        return None
    training = manifest.get("training")
    engine = training.get("engine") if isinstance(training, dict) else None
    if isinstance(engine, str) and engine:
        return engine
    if isinstance(engine, dict):
        profile = engine.get("profile")
        return str(profile) if isinstance(profile, str) and profile else None
    return None


def _manifest_experiment_id(manifest: dict[str, object] | None) -> str | None:
    if not isinstance(manifest, dict):
        return None
    value = manifest.get("experiment_id")
    return value if isinstance(value, str) and value else None


def _sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _float_or_none(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _int_or_none(value: object) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _load_json(path: Path) -> dict[str, object] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


__all__ = ["resolve_lane"]
