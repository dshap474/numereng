"""Unit tests for the seed-normalized aggregation keystone and the believed-best / plateau /
caps / empirical-noise context fields layered on top of it.

These cover the harness-owned seed bookkeeping that replaced the model's hand-tracked ledger:
recipe grouping (seed + naming/execution knobs stripped), trio stats, empirical noise, the
machine-readable believed_best, and the plateau/coverage/caps signals — plus the size guard.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from numereng.features.agentic_research import aggregate, context, memory
from numereng.features.agentic_research import types as ar_types
from numereng.features.experiments import create_experiment, get_experiment

EXPERIMENT_ID = "2026-06-22_aggregate-exp"


def _config(*, random_state: int, predictions_name: str, max_depth: int = 6, parallel_folds: int = 1) -> dict:
    return {
        "data": {"data_version": "v5.2", "dataset_variant": "non_downsampled", "target_col": "target_ender_20"},
        "model": {"type": "LGBMRegressor", "params": {"max_depth": max_depth, "random_state": random_state}},
        "training": {"resources": {"parallel_folds": parallel_folds, "max_threads_per_worker": 10}},
        "output": {"predictions_name": predictions_name},
    }


# ---------------------------------------------------------------------------
# recipe_key
# ---------------------------------------------------------------------------


def test_recipe_key_collapses_seed_trio_and_ignores_naming_and_execution_knobs() -> None:
    # Same recipe under three seeds with seed-encoded predictions_name + different resources.
    s42 = _config(random_state=42, predictions_name="pred_s42", parallel_folds=1)
    s17 = _config(random_state=17, predictions_name="pred_s17", parallel_folds=4)
    s99 = _config(random_state=99, predictions_name="pred_s99", parallel_folds=2)
    assert aggregate.recipe_key(s42) == aggregate.recipe_key(s17) == aggregate.recipe_key(s99)


def test_recipe_key_splits_on_a_real_param_change() -> None:
    base = _config(random_state=42, predictions_name="p", max_depth=6)
    deeper = _config(random_state=42, predictions_name="p", max_depth=8)
    assert aggregate.recipe_key(base) != aggregate.recipe_key(deeper)


def test_recipe_key_does_not_mutate_input() -> None:
    cfg = _config(random_state=42, predictions_name="p")
    snapshot = copy.deepcopy(cfg)
    aggregate.recipe_key(cfg)
    assert cfg == snapshot


# ---------------------------------------------------------------------------
# aggregate_recipes / observed_seed_noise
# ---------------------------------------------------------------------------


def _entry(config: str, *, seed, metric, fnc, run_id, status="completed", round_n=None) -> dict:
    entry = {"status": status, "config": config, "seed": seed, "metric": metric, "fnc": fnc, "run_id": run_id}
    if round_n is not None:
        entry["round"] = round_n
    return entry


def _trio_setup() -> tuple[list[dict], dict[str, dict]]:
    configs = {
        "config_015.json": _config(random_state=42, predictions_name="pred_s42"),
        "config_016.json": _config(random_state=17, predictions_name="pred_s17"),
        "config_017.json": _config(random_state=99, predictions_name="pred_s99"),
        "config_009.json": _config(random_state=42, predictions_name="other", max_depth=8),
    }
    entries = [
        _entry("config_015.json", seed=42, metric=0.001483, fnc=0.020, run_id="a"),
        _entry("config_016.json", seed=17, metric=0.001372, fnc=0.021, run_id="b"),
        _entry("config_017.json", seed=99, metric=0.001340, fnc=0.019, run_id="c"),
        _entry("config_009.json", seed=42, metric=0.000900, fnc=0.015, run_id="d"),
        _entry("config_009.json", seed=None, metric=None, fnc=None, run_id=None, status="failed"),
    ]
    return entries, configs


def test_aggregate_recipes_computes_trio_mean_and_fnc() -> None:
    entries, configs = _trio_setup()
    groups = aggregate.aggregate_recipes(entries, configs=configs)
    assert len(groups) == 2  # trio recipe + the depth-8 recipe
    top = groups[0]
    assert top.count == 3
    assert top.trio_mean == (0.001483 + 0.001372 + 0.001340) / 3
    assert top.trio_fnc_mean == (0.020 + 0.021 + 0.019) / 3
    assert set(top.run_ids) == {"a", "b", "c"}


def test_aggregate_recipes_ignores_failed_and_unscored_entries() -> None:
    entries, configs = _trio_setup()
    groups = aggregate.aggregate_recipes(entries, configs=configs)
    depth8 = next(g for g in groups if g.count == 1)
    assert depth8.representative_config == "config_009.json"  # the failed line did not add a seed


def test_observed_seed_noise_null_until_multi_seed_then_positive() -> None:
    entries, configs = _trio_setup()
    groups = aggregate.aggregate_recipes(entries, configs=configs)
    noise = aggregate.observed_seed_noise(groups)
    assert noise is not None and noise > 0
    # A single-seed-only world has no within-recipe spread to measure.
    single = aggregate.aggregate_recipes(entries[3:4], configs=configs)
    assert aggregate.observed_seed_noise(single) is None


# ---------------------------------------------------------------------------
# context fields (believed_best, recipe_leaderboard, plateau, coverage, caps, size guard)
# ---------------------------------------------------------------------------


def _experiment_with_history(tmp_path: Path, *, configs: dict[str, dict], journal: list[dict], state_extra: dict):
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Agg")
    experiment_dir = experiment.manifest_path.parent
    config_dir = experiment_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in configs.items():
        (config_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    journal_path = experiment_dir / "agentic_research" / "journal.jsonl"
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("w", encoding="utf-8") as handle:
        for entry in journal:
            handle.write(json.dumps(entry) + "\n")
    experiment = get_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID)
    state = memory.initial_state(experiment)
    state.update(state_extra)
    return store_root, experiment, state


def test_build_context_surfaces_recipe_leaderboard_and_believed_best(tmp_path: Path) -> None:
    entries, configs = _trio_setup()
    state_extra = {"believed_best": {"config": "config_015.json", "trio_mean": 0.0014}}
    store_root, experiment, state = _experiment_with_history(
        tmp_path, configs=configs, journal=entries, state_extra=state_extra
    )
    ctx = context.build_context(root=store_root, experiment=experiment, report=None, state=state)
    assert ctx["believed_best"] == {"config": "config_015.json", "trio_mean": 0.0014}
    leaderboard = ctx["recipe_leaderboard"]
    assert leaderboard and leaderboard[0]["seed_count"] == 3
    assert leaderboard[0]["trio_mean"] == (0.001483 + 0.001372 + 0.001340) / 3
    assert ctx["observed_seed_noise"] is not None


def test_rounds_since_new_believed_best_counts_completed_rounds_after_change(tmp_path: Path) -> None:
    configs = {
        f"config_{n:03d}.json": _config(random_state=42, predictions_name=f"p{n}", max_depth=n) for n in (1, 2, 3, 4)
    }
    journal = [
        _entry("config_001.json", seed=42, metric=0.001, fnc=0.01, run_id="r1", round_n=1),
        _entry("config_002.json", seed=42, metric=0.002, fnc=0.01, run_id="r2", round_n=2),
        _entry("config_003.json", seed=None, metric=None, fnc=None, run_id=None, status="failed", round_n=3),
        _entry("config_004.json", seed=42, metric=0.0015, fnc=0.01, run_id="r4", round_n=4),
    ]
    # believed_best last changed at round 2 -> completed rounds after it = round 4 only (round 3 failed).
    store_root, experiment, state = _experiment_with_history(
        tmp_path, configs=configs, journal=journal, state_extra={"believed_best_changed_round": 2}
    )
    ctx = context.build_context(root=store_root, experiment=experiment, report=None, state=state)
    assert ctx["rounds_since_new_believed_best"] == 1


def test_coverage_is_cardinality_capped(tmp_path: Path) -> None:
    # 30 distinct max_depth values across configs -> coverage must summarize, not list all 30.
    configs = {
        f"config_{n:03d}.json": _config(random_state=42, predictions_name=f"p{n}", max_depth=n) for n in range(1, 31)
    }
    journal = [
        _entry(name, seed=42, metric=0.001, fnc=0.01, run_id=f"r{i}", round_n=i)
        for i, name in enumerate(configs, start=1)
    ]
    store_root, experiment, state = _experiment_with_history(tmp_path, configs=configs, journal=journal, state_extra={})
    ctx = context.build_context(root=store_root, experiment=experiment, report=None, state=state)
    depth_cov = ctx["coverage"]["model.params.max_depth"]
    assert isinstance(depth_cov, dict)  # summarized, not a 30-element list
    assert depth_cov["count"] == 30
    assert len(depth_cov["recent_samples"]) <= ar_types.COVERAGE_VALUE_LIMIT
    # A low-cardinality path stays an explicit list.
    assert ctx["coverage"]["data.feature_set"] == [] or isinstance(ctx["coverage"].get("data.target_col"), list)


def test_caps_binding_flags_believed_best_at_cap_edge(tmp_path: Path) -> None:
    configs = {"config_001.json": _config(random_state=42, predictions_name="p", max_depth=9)}
    journal = [_entry("config_001.json", seed=42, metric=0.001, fnc=0.01, run_id="r1", round_n=1)]
    state_extra = {"believed_best": {"config": "config_001.json"}}
    store_root, experiment, state = _experiment_with_history(
        tmp_path, configs=configs, journal=journal, state_extra=state_extra
    )
    experiment.metadata[ar_types.VALUE_CAPS_METADATA_KEY] = {"model.params.max_depth": [3, 9]}
    ctx = context.build_context(root=store_root, experiment=experiment, report=None, state=state)
    binding = ctx["caps_binding"]
    assert binding == [{"path": "model.params.max_depth", "value": 9, "edge": "max", "cap": [3.0, 9.0]}]
