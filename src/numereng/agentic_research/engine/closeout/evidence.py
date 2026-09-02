"""Phase 0: strict, deterministic evidence build (no LLM).

The in-run readers (``memory.journal_all``, ``aggregate.load_config_cache``,
``aggregate.aggregate_recipes``) are lenient: they silently drop malformed journal lines, skip
invalid configs, and ignore completed entries without a config/metric. Closeout cannot distill
corrupted evidence, so this module re-parses the journal strictly and computes the deterministic
run-record summaries the memo depends on. The output survives context truncation, so the real
distillation signal (sweep ranges, failure archaeology) is never lost to a size guard.

USAGE:
    from numereng.agentic_research.engine.closeout import evidence
    summary = evidence.build_evidence(experiment=rec, state=state_dict, store_root=Path(".numereng"))
"""

from __future__ import annotations

import json
from pathlib import Path

from numereng.agentic_research.engine import aggregate, boundary, memory
from numereng.agentic_research.engine import types as ar_types
from numereng.agentic_research.engine.closeout import types as ct
from numereng.config.training import load_training_config_json
from numereng.features import holdout
from numereng.features.experiments import (
    ExperimentRecord,
    get_experiment_holdout,
    seal_experiment_holdout,
)
from numereng.features.scoring.run_service import score_run_eras
from numereng.features.training.repo import resolve_run_manifest_path


# --------------------------------------------------------------------------- #
# Strict journal + config parsing
# --------------------------------------------------------------------------- #
def _parse_journal_strict(journal_path: Path) -> list[tuple[int, dict[str, object]]]:
    """Return (lineno, entry) for each non-blank line; a malformed line fails closeout hard."""
    try:
        return list(memory.iter_journal_lines(journal_path, strict=True))
    except ar_types.JournalLineError as exc:
        raise ct.CloseoutError(f"{ct.ERROR_PREFIX}journal_malformed:{exc.lineno}") from exc


def _validate_completed_configs(
    parsed: list[tuple[int, dict[str, object]]], *, config_dir: Path
) -> dict[str, dict[str, object]]:
    """Every completed entry must name an existing, contract-valid config and a numeric metric."""
    cache: dict[str, dict[str, object]] = {}
    for lineno, entry in parsed:
        if entry.get("status") != "completed":
            continue
        name = entry.get("config")
        if not isinstance(name, str) or not name:
            raise ct.CloseoutError(f"{ct.ERROR_PREFIX}journal_entry_invalid:{lineno}:config")
        if name not in cache:
            path = config_dir / name
            if not path.is_file():
                raise ct.CloseoutError(f"{ct.ERROR_PREFIX}journal_entry_invalid:{lineno}:config")
            try:
                cache[name] = load_training_config_json(path)
            except Exception as exc:  # noqa: BLE001 - any loader failure is an invalid config
                raise ct.CloseoutError(f"{ct.ERROR_PREFIX}journal_entry_invalid:{lineno}:config") from exc
        metric = entry.get("metric")
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise ct.CloseoutError(f"{ct.ERROR_PREFIX}journal_entry_invalid:{lineno}:metric")
    return cache


# --------------------------------------------------------------------------- #
# Deterministic summaries
# --------------------------------------------------------------------------- #
def _failure_taxonomy(parsed: list[tuple[int, dict[str, object]]]) -> dict[str, int]:
    taxonomy: dict[str, int] = {}
    for _, entry in parsed:
        if entry.get("status") != "failed":
            continue
        error = entry.get("error")
        token = str(error).split(":", 1)[0] if isinstance(error, str) and error else "unknown"
        taxonomy[token] = taxonomy.get(token, 0) + 1
    return taxonomy


def _coverage(parsed: list[tuple[int, dict[str, object]]]) -> dict[str, object]:
    values_by_path: dict[str, list[object]] = {}
    for _, entry in parsed:
        for change in ar_types.as_list(entry.get("changes")):
            if not isinstance(change, dict):
                continue
            path = change.get("path")
            if not isinstance(path, str):
                continue
            values_by_path.setdefault(path, []).append(change.get("value"))
    coverage: dict[str, object] = {}
    for path, values in sorted(values_by_path.items()):
        numeric = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        distinct: list[object] = []
        for value in values:
            if value not in distinct:
                distinct.append(value)
        entry: dict[str, object] = {"count": len(values), "distinct": distinct[: ar_types.COVERAGE_VALUE_LIMIT]}
        if numeric:
            entry["numeric_range"] = [min(numeric), max(numeric)]
        coverage[path] = entry
    return coverage


def _parentage(parsed: list[tuple[int, dict[str, object]]]) -> dict[str, object]:
    children: dict[str, int] = {}
    for _, entry in parsed:
        parent = entry.get("parent_config")
        if isinstance(parent, str) and parent:
            children[parent] = children.get(parent, 0) + 1
    branch_points = sum(1 for count in children.values() if count > 1)
    return {
        "distinct_parents": len(children),
        "max_children": max(children.values()) if children else 0,
        "branch_points": branch_points,
    }


def _sweep_abandoned(experiment: ExperimentRecord) -> dict[str, object]:
    directory = memory.rounds_dir(experiment)
    rounds: list[int] = []
    if directory.is_dir():
        for path in sorted(directory.glob("r*.md")):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if "SWEEP ABANDONED" in text:
                number = memory.parse_round_label(path.stem)
                if number is not None:
                    rounds.append(number)
    return {"count": len(rounds), "rounds": sorted(rounds)}


def _rounds_table(parsed: list[tuple[int, dict[str, object]]]) -> list[dict[str, object]]:
    table: list[dict[str, object]] = []
    for _, entry in parsed:
        changes_digest = "; ".join(
            f"{c.get('path')}={c.get('value')}" for c in ar_types.as_list(entry.get("changes")) if isinstance(c, dict)
        )
        table.append(
            {
                "round": entry.get("round"),
                "config": entry.get("config"),
                "parent": entry.get("parent_config"),
                "seed": entry.get("seed"),
                "changes": changes_digest,
                "metric": ar_types.optional_float(entry.get("metric")),
                "fnc": ar_types.optional_float(entry.get("fnc")),
                "benchmark_corr": ar_types.optional_float(entry.get("benchmark_corr")),
                "status": entry.get("status"),
                "wall_seconds": ar_types.optional_float(entry.get("wall_seconds")),
            }
        )
    return table


# --------------------------------------------------------------------------- #
# Believed-best resolution + leaderboard projection
# --------------------------------------------------------------------------- #
def _believed_best_config(state: dict[str, object]) -> str | None:
    raw = state.get("believed_best")
    if isinstance(raw, dict):
        config = raw.get("config")
        return config if isinstance(config, str) and config else None
    if isinstance(raw, str) and raw:
        return raw
    return None


def _leaderboard_row(group: aggregate.RecipeGroup) -> dict[str, object]:
    return {
        "recipe_key": group.recipe_key,
        "representative_config": group.representative_config,
        "trio_mean": group.trio_mean,
        "trio_fnc_mean": group.trio_fnc_mean,
        "count": group.count,
        "bmc_std": group.bmc_std,
        "seeds": list(group.seeds),
        "run_ids": list(group.run_ids),
    }


# --------------------------------------------------------------------------- #
# Frozen holdout: one-time closeout opening
# --------------------------------------------------------------------------- #
def _open_holdout(
    *, experiment: ExperimentRecord, believed_best: aggregate.RecipeGroup, store_root: Path
) -> dict[str, object] | None:
    """Score the believed-best candidate on the frozen holdout exactly once, then seal it.

    No-op (None) when no holdout was frozen. On the first closeout pass it verifies the
    tamper fingerprint, scores the believed-best seed runs restricted to the holdout eras,
    writes ``holdout_result.json``, and seals. On any later pass (sealed) it returns the
    persisted record without re-scoring, so closeout restarts stay idempotent.
    """
    runs_dir = store_root / "runs"
    spec = get_experiment_holdout(store_root=store_root, experiment_id=experiment.experiment_id)
    if spec is None or not spec.is_frozen:
        return None

    result_path = memory.agentic_dir(experiment) / ct.CLOSEOUT_DIRNAME / ct.CLOSEOUT_HOLDOUT_FILENAME
    if spec.sealed:
        if result_path.is_file():
            record = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(record, dict):
                return record
        return {"enabled": True, "sealed": True, "note": "sealed_without_record"}

    run_ids = [run_id for run_id in believed_best.run_ids if run_id]
    if not run_ids:
        raise ct.CloseoutError(ct.ERR_BELIEVED_BEST_UNRESOLVED)

    era_order = _run_era_order(runs_dir, run_ids[0])
    try:
        holdout.verify_fingerprint(spec, era_order=era_order)
    except holdout.HoldoutError as exc:
        raise ct.CloseoutError(ct.ERR_HOLDOUT_TAMPERED) from exc

    era_filter = holdout.restriction_filter(spec)
    per_run: dict[str, object] = {}
    for run_id in run_ids:
        payload = score_run_eras(
            run_id=run_id, era_filter=era_filter, store_root=store_root, stage=ar_types.SCORING_STAGE
        )
        per_run[run_id] = ar_types.get_dotted(payload, ar_types.PRIMARY_METRIC)
    values = [value for value in per_run.values() if isinstance(value, (int, float)) and not isinstance(value, bool)]
    record: dict[str, object] = {
        "enabled": True,
        "mode": spec.mode,
        "holdout_n_eras": spec.holdout_n_eras,
        "era_gap": spec.era_gap,
        "holdout_eras": list(spec.holdout_eras or ()),
        "fingerprint": spec.fingerprint,
        "recipe_key": believed_best.recipe_key,
        "run_ids": run_ids,
        "primary_metric": ar_types.PRIMARY_METRIC,
        "per_run_primary": per_run,
        "holdout_primary_mean": (sum(values) / len(values)) if values else None,
        "search_trio_mean": believed_best.trio_mean,
        "sealed": True,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    ar_types.write_json(result_path, record)
    try:
        seal_experiment_holdout(store_root=store_root, experiment_id=experiment.experiment_id)
    except holdout.HoldoutError as exc:
        raise ct.CloseoutError(ct.ERR_HOLDOUT_REUSE) from exc
    return record


def _run_era_order(runs_dir: Path, run_id: str) -> tuple[str, ...]:
    run_dir = runs_dir / run_id
    run_manifest = json.loads(resolve_run_manifest_path(run_dir).read_text(encoding="utf-8"))
    artifacts = run_manifest.get("artifacts") if isinstance(run_manifest, dict) else None
    predictions_rel = artifacts.get("predictions") if isinstance(artifacts, dict) else None
    if isinstance(predictions_rel, str) and predictions_rel.strip():
        predictions_path = (run_dir / predictions_rel).resolve()
    else:
        parquet_files = sorted((run_dir / "artifacts" / "predictions").glob("*.parquet"))
        if len(parquet_files) != 1:
            raise ct.CloseoutError(ct.ERR_HOLDOUT_TAMPERED)
        predictions_path = parquet_files[0].resolve()
    return holdout.read_prediction_era_order(predictions_path)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def build_evidence(*, experiment: ExperimentRecord, state: dict[str, object], store_root: Path) -> dict[str, object]:
    """Strict, deterministic evidence build. Raises ``CloseoutError`` on any evidence corruption."""
    journal_path = memory.journal_path(experiment)
    config_dir = memory.configs_dir(experiment)
    parsed = _parse_journal_strict(journal_path)
    configs = _validate_completed_configs(parsed, config_dir=config_dir)

    entries = [entry for _, entry in parsed]
    seed_path = boundary.seed_change_path(experiment)
    leaderboard = aggregate.aggregate_recipes(entries, configs=configs, seed_path=seed_path)
    if not leaderboard:
        raise ct.CloseoutError(ct.ERR_LEADERBOARD_EMPTY)

    believed_best_config = _believed_best_config(state)
    believed_best_group = (
        aggregate.group_for_config(leaderboard, believed_best_config, configs, seed_path=seed_path)
        if believed_best_config is not None
        else None
    )
    if believed_best_config is None or believed_best_group is None:
        raise ct.CloseoutError(ct.ERR_BELIEVED_BEST_UNRESOLVED)

    champion = state.get("champion") if isinstance(state.get("champion"), dict) else None
    completed = sum(1 for e in entries if e.get("status") == "completed")
    failed = sum(1 for e in entries if e.get("status") == "failed")
    skipped = sum(1 for e in entries if e.get("status") == "skipped")

    return {
        "experiment_id": experiment.experiment_id,
        "believed_best": {
            "config": believed_best_config,
            "recipe_key": believed_best_group.recipe_key,
            "trio_mean": believed_best_group.trio_mean,
            "trio_fnc_mean": believed_best_group.trio_fnc_mean,
            "count": believed_best_group.count,
            "run_ids": list(believed_best_group.run_ids),
        },
        "champion": champion,
        "leaderboard": [_leaderboard_row(group) for group in leaderboard],
        "observed_seed_noise": aggregate.observed_seed_noise(leaderboard),
        "failure_taxonomy": _failure_taxonomy(parsed),
        "coverage": _coverage(parsed),
        "parentage": _parentage(parsed),
        "sweep_abandoned": _sweep_abandoned(experiment),
        "duplicate_skips": skipped,
        "rounds_table": _rounds_table(parsed),
        "holdout": _open_holdout(experiment=experiment, believed_best=believed_best_group, store_root=store_root),
        "totals": {
            "journal_entries": len(entries),
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
        },
    }
