"""Seed-normalized recipe aggregation.

The harness keeps the per-seed clerical bookkeeping the model used to hand-track: it groups
runs that share one training recipe (ignoring seed and pure execution/naming knobs) and computes
each recipe's seed-trio statistics. This is deterministic bookkeeping, not strategy — the model
still owns belief and judgment; it just consults these aggregates instead of maintaining them.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

from numereng.config.training import load_training_config_json
from numereng.features.agentic_research import types as ar_types
from numereng.features.training.run_store import compute_config_hash

# Paths normalized out of a config before hashing so seed-trio runs of one recipe collapse to one
# key. Mirrors what compute_run_hash strips for training identity (the whole `output` block and the
# legacy `data.loading`) plus the seed and the two pure-execution resource knobs.
_RECIPE_DROP_KEYS = ("output",)
_RECIPE_DROP_DOTTED = (
    ("data", "loading"),
    ("model", "params", "random_state"),
    ("training", "resources", "parallel_folds"),
    ("training", "resources", "max_threads_per_worker"),
)


@dataclass(frozen=True)
class RecipeGroup:
    recipe_key: str
    representative_config: str
    seeds: tuple[int | None, ...]
    per_seed: tuple[dict[str, object], ...]
    trio_mean: float
    trio_fnc_mean: float | None
    count: int
    bmc_std: float | None
    run_ids: tuple[str, ...]


def recipe_key(config: dict[str, object]) -> str:
    """Hash a config with seed + execution/naming knobs stripped (one key per training recipe)."""
    stripped = deepcopy(config)
    for key in _RECIPE_DROP_KEYS:
        stripped.pop(key, None)
    for parts in _RECIPE_DROP_DOTTED:
        cursor: object = stripped
        for part in parts[:-1]:
            cursor = cursor.get(part) if isinstance(cursor, dict) else None
        if isinstance(cursor, dict):
            cursor.pop(parts[-1], None)
    return compute_config_hash(stripped)


def load_config_cache(config_dir: Path) -> dict[str, dict[str, object]]:
    """Load every config file once into a {filename: config} cache (loaders are shared callers)."""
    cache: dict[str, dict[str, object]] = {}
    if not config_dir.is_dir():
        return cache
    for path in sorted(config_dir.glob("*.json")):
        try:
            cache[path.name] = load_training_config_json(path)
        except Exception:
            continue
    return cache


def aggregate_recipes(entries: list[dict[str, object]], *, configs: dict[str, dict[str, object]]) -> list[RecipeGroup]:
    """Group completed journal entries by recipe_key; one per-seed row per distinct seed."""
    grouped: dict[str, dict[object, dict[str, object]]] = {}
    for entry in entries:
        if entry.get("status") != "completed":
            continue
        name = entry.get("config")
        config = configs.get(name) if isinstance(name, str) else None
        bmc = ar_types.optional_float(entry.get("metric"))
        if config is None or bmc is None:
            continue
        seed = entry.get("seed")
        seed = seed if isinstance(seed, int) and not isinstance(seed, bool) else None
        key = recipe_key(config)
        # Latest completed entry for a (recipe, seed) wins; entries arrive in chronological order.
        grouped.setdefault(key, {})[seed if seed is not None else "_none"] = {
            "seed": seed,
            "bmc": bmc,
            "fnc": ar_types.optional_float(entry.get("fnc")),
            "config": name,
            "run_id": entry.get("run_id"),
        }
    groups = [_build_group(key, list(by_seed.values())) for key, by_seed in grouped.items()]
    groups.sort(key=lambda group: group.trio_mean, reverse=True)
    return groups


def observed_seed_noise(groups: list[RecipeGroup]) -> float | None:
    """Pooled within-recipe SD of per-seed BMC across recipes with >=2 seeds (None until available)."""
    deviations: list[float] = []
    contributing = 0
    for group in groups:
        if group.count < 2:
            continue
        contributing += 1
        for row in group.per_seed:
            bmc = ar_types.optional_float(row.get("bmc"))
            if bmc is not None:
                deviations.append(bmc - group.trio_mean)
    dof = len(deviations) - contributing
    if dof <= 0:
        return None
    return sqrt(sum(value * value for value in deviations) / dof)


def group_for_config(
    groups: list[RecipeGroup], config_name: str, configs: dict[str, dict[str, object]]
) -> RecipeGroup | None:
    """Return the recipe group a given config belongs to, if it has been scored."""
    config = configs.get(config_name)
    if config is None:
        return None
    key = recipe_key(config)
    return next((group for group in groups if group.recipe_key == key), None)


def _build_group(key: str, rows: list[dict[str, object]]) -> RecipeGroup:
    bmcs = [float(row["bmc"]) for row in rows]
    fncs = [float(row["fnc"]) for row in rows if isinstance(row.get("fnc"), (int, float))]
    best = max(rows, key=lambda row: float(row["bmc"]))
    trio_mean = sum(bmcs) / len(bmcs)
    return RecipeGroup(
        recipe_key=key,
        representative_config=str(best["config"]),
        seeds=tuple(row["seed"] for row in rows),  # type: ignore[misc]
        per_seed=tuple(rows),
        trio_mean=trio_mean,
        trio_fnc_mean=(sum(fncs) / len(fncs)) if fncs else None,
        count=len(rows),
        bmc_std=_sample_std(bmcs),
        run_ids=tuple(str(row["run_id"]) for row in rows if row.get("run_id")),
    )


def _sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
