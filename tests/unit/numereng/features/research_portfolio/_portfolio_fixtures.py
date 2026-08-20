"""Synthetic on-disk fixtures for research-portfolio P1 tests.

Builds a minimal-but-real store: a scale experiment with valid trio training
configs (differing only in ``model.params.random_state`` so they share one
recipe key), an agentic journal + state, and fully materialized run directories
(run.json / resolved.json / results.json / metrics.json + a prediction parquet
and score provenance) so ``resolve_lane`` and ``portfolio_status`` resolve
end-to-end. Nothing here mocks the resolvers — the disk is canonical.

USAGE:
    from tests.unit.numereng.features.research_portfolio import _portfolio_fixtures as fx
    store = fx.build_store(tmp_path)
    fx.build_run(store, run_id="r42", config=cfg, ...)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from numereng.features.training.run_store import compute_config_hash

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SCALE_EXPERIMENT_ID = "2026-07-12_portfolio-scale"
DEFAULT_CONTRIBUTION_TARGET = "target_ender_20"
DEFAULT_BENCHMARK_SHA = "benchmark-sha-000"


# --------------------------------------------------------------------------- #
# Config template (a real, strictly-valid TrainingConfig)
# --------------------------------------------------------------------------- #


def valid_config(*, random_state: int, predictions_name: str, max_depth: int = 9) -> dict:
    """Return one fully-valid training config; only random_state/name vary in a trio."""

    return {
        "data": {
            "benchmark_source": {"pred_col": "prediction", "source": "active"},
            "data_version": "v5.2",
            "dataset_scope": "train_plus_validation",
            "dataset_variant": "non_downsampled",
            "era_col": "era",
            "feature_set": "medium",
            "id_col": "id",
            "meta_model_col": "numerai_meta_model",
            "target_col": "target_ender_20",
            "target_horizon": "20d",
        },
        "model": {
            "params": {
                "colsample_bytree": 0.6,
                "device_type": "gpu",
                "learning_rate": 0.006,
                "max_depth": max_depth,
                "min_child_samples": 1000,
                "n_estimators": 6000,
                "num_leaves": 256,
                "random_state": random_state,
                "reg_alpha": 25,
                "reg_lambda": 0.0,
            },
            "type": "LGBMRegressor",
            "x_groups": ["features"],
        },
        "output": {"predictions_name": predictions_name},
        "preprocessing": {"missing_value": 2.0, "nan_missing_all_twos": False},
        "training": {
            "cache": {
                "cache_features": True,
                "cache_fold_matrices": False,
                "cache_fold_specs": True,
                "cache_labels": True,
                "mode": "deterministic",
            },
            "engine": {"profile": "purged_walk_forward"},
            "post_training_scoring": "none",
            "resources": {
                "max_threads_per_worker": 10,
                "memmap_enabled": True,
                "parallel_backend": "joblib",
                "parallel_folds": 1,
            },
        },
    }


# --------------------------------------------------------------------------- #
# Store + experiment scaffolding
# --------------------------------------------------------------------------- #


@dataclass
class Store:
    """Handles to the synthetic store roots used across a single test."""

    root: Path
    experiment_id: str

    @property
    def configs_dir(self) -> Path:
        return self.root / "experiments" / self.experiment_id / "configs"

    @property
    def agentic_dir(self) -> Path:
        return self.root / "experiments" / self.experiment_id / "agentic_research"


def build_store(tmp_path: Path, *, experiment_id: str = SCALE_EXPERIMENT_ID) -> Store:
    """Create ``.numereng`` with one scale experiment manifest + configs dir."""

    root = tmp_path / ".numereng"
    exp_dir = root / "experiments" / experiment_id
    (exp_dir / "configs").mkdir(parents=True)
    (exp_dir / "agentic_research").mkdir(parents=True)
    (exp_dir / "experiment.json").write_text(
        json.dumps({"schema_version": "1", "experiment_id": experiment_id, "status": "active"}),
        encoding="utf-8",
    )
    return Store(root=root, experiment_id=experiment_id)


def write_config(store: Store, name: str, config: dict) -> None:
    """Write one config file into the experiment configs dir."""

    (store.configs_dir / name).write_text(json.dumps(config), encoding="utf-8")


def write_journal(store: Store, lines: list[dict | str]) -> None:
    """Write journal.jsonl; str entries are written verbatim (for malformed-line tests)."""

    path = store.agentic_dir / "journal.jsonl"
    rendered = [line if isinstance(line, str) else json.dumps(line) for line in lines]
    path.write_text("\n".join(rendered) + "\n", encoding="utf-8")


def write_state(store: Store, state: dict) -> None:
    """Write agentic_research/state.json."""

    (store.agentic_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")


def journal_row(config: str, *, seed: int | None, metric: float, run_id: str, status: str = "completed") -> dict:
    """One completed journal line the resolver groups by recipe/seed."""

    return {"status": status, "config": config, "seed": seed, "metric": metric, "fnc": 0.02, "run_id": run_id}


# --------------------------------------------------------------------------- #
# Run materialization
# --------------------------------------------------------------------------- #


def build_run(
    store: Store,
    *,
    run_id: str,
    config: dict,
    bmc: float,
    experiment_id: str | None = None,
    profile: str = "purged_walk_forward",
    config_hash: str | None = None,
    contribution_target: str = DEFAULT_CONTRIBUTION_TARGET,
    benchmark_sha: str = DEFAULT_BENCHMARK_SHA,
    with_predictions: bool = True,
    era_ids: tuple[tuple[str, str], ...] = (("e1", "id1"), ("e1", "id2"), ("e2", "id3")),
    predictions: list[float] | None = None,
    targets: list[float] | None = None,
    target_col: str = DEFAULT_CONTRIBUTION_TARGET,
    incomplete: bool = False,
) -> Path:
    """Materialize one run directory the resolver reads (run/resolved/results/metrics + parquet)."""

    experiment_id = experiment_id if experiment_id is not None else store.experiment_id
    run_dir = store.root / "runs" / run_id
    run_dir.mkdir(parents=True)

    resolved_hash = config_hash if config_hash is not None else compute_config_hash(config)
    manifest = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "config": {"hash": resolved_hash},
        "training": {"engine": {"profile": profile}},
        "artifacts": {"predictions": "artifacts/predictions/pred_run.parquet"},
    }
    (run_dir / "run.json").write_text(json.dumps(manifest), encoding="utf-8")

    if incomplete:
        # Missing resolved/results/metrics -> classify_run_mode == "incomplete".
        return run_dir

    (run_dir / "resolved.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "results.json").write_text(json.dumps({"status": "FINISHED"}), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps({"bmc_last_200_eras": {"mean": bmc}, "fnc": {"mean": 0.02}}),
        encoding="utf-8",
    )
    (run_dir / "score_provenance.json").write_text(
        json.dumps(
            {
                "columns": {
                    "contribution_target_cols": [contribution_target],
                    "era_col": "era",
                    "id_col": "id",
                },
                "sources": {"benchmark": {"sha256": benchmark_sha}},
            }
        ),
        encoding="utf-8",
    )
    if with_predictions:
        _write_predictions(run_dir, era_ids, predictions=predictions, targets=targets, target_col=target_col)
    return run_dir


def _write_predictions(
    run_dir: Path,
    era_ids: tuple[tuple[str, str], ...],
    *,
    predictions: list[float] | None = None,
    targets: list[float] | None = None,
    target_col: str = DEFAULT_CONTRIBUTION_TARGET,
) -> None:
    import pandas as pd

    pred_dir = run_dir / "artifacts" / "predictions"
    pred_dir.mkdir(parents=True)
    columns: dict[str, list] = {
        "era": [era for era, _ in era_ids],
        "id": [pid for _, pid in era_ids],
        "prediction": list(predictions) if predictions is not None else [0.5 for _ in era_ids],
    }
    if targets is not None:
        columns[target_col] = list(targets)
    pd.DataFrame(columns).to_parquet(pred_dir / "pred_run.parquet")


def write_active_benchmark(
    store: Store,
    *,
    era_ids: tuple[tuple[str, str], ...],
    predictions: list[float],
) -> Path:
    """Write the canonical active-benchmark predictions parquet the diversity panel attaches."""

    import pandas as pd

    path = store.root / "datasets" / "baselines" / "active_benchmark" / "predictions.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "era": [era for era, _ in era_ids],
            "id": [pid for _, pid in era_ids],
            "prediction": list(predictions),
        }
    ).to_parquet(path)
    return path


# --------------------------------------------------------------------------- #
# Registry payloads
# --------------------------------------------------------------------------- #


def policy_block(*, filled: bool = True) -> dict:
    """Return a policy block; filled=False leaves the 8 gated params null."""

    base = {"policy_revision": 1, "policy_decision_record_id": "DR-001"}
    if not filled:
        return base
    base.update(
        {
            "scout_tranche_cap": 20,
            "scout_quality_floor": 0.001,
            "coverage_reserve": 4,
            "diversity_bmc_tolerance": 0.0003,
            "capacity_class_rule": "trees*depth",
            "live_review_min_resolved_rounds": 12,
            "combination_trial_cap": 8,
            "cross_lane_weight_cap": 0.5,
        }
    )
    return base


def registry_payload(
    *,
    store: Store,
    candidates: list[dict],
    policy_filled: bool = True,
    scale: str | None = None,
    superseded: list[dict] | None = None,
    expected_believed_best: str | None = None,
    live: dict | None = None,
) -> dict:
    """Assemble a valid registry.json payload with one lane."""

    lane: dict = {
        "lane_id": "medium_ender20",
        "axis": "feature_scope",
        "structural": True,
        "research_stage": "seed-confirmed",
        "deployment_stage": "unbound",
        "combination_stage": "not-ready",
        "constitution_revision": 1,
        "experiments": {
            "scale": scale if scale is not None else store.experiment_id,
            "superseded": superseded or [],
        },
        "envelope": {"max_rounds": 50, "approved_tranche_rounds": 20},
        "candidates": candidates,
    }
    if expected_believed_best is not None:
        lane["expected_believed_best"] = expected_believed_best
    if live is not None:
        lane["live"] = live
    return {"schema_version": 1, "policy": policy_block(filled=policy_filled), "lanes": [lane]}


def lane_block(
    *,
    lane_id: str,
    store: Store,
    candidates: list[dict],
    axis: str = "feature_scope",
    scale: str | None = None,
    live: dict | None = None,
) -> dict:
    """One valid registry lane dict; distinct axis per lane keeps a multi-lane registry valid."""

    lane: dict = {
        "lane_id": lane_id,
        "axis": axis,
        "structural": True,
        "research_stage": "seed-confirmed",
        "deployment_stage": "unbound",
        "combination_stage": "not-ready",
        "constitution_revision": 1,
        "experiments": {"scale": scale if scale is not None else store.experiment_id, "superseded": []},
        "envelope": {"max_rounds": 50, "approved_tranche_rounds": 20},
        "candidates": candidates,
    }
    if live is not None:
        lane["live"] = live
    return lane


def registry_with_lanes(*, lanes: list[dict], policy_filled: bool = True) -> dict:
    """Assemble a registry payload spanning multiple lanes (P2 cross-lane diversity)."""

    return {"schema_version": 1, "policy": policy_block(filled=policy_filled), "lanes": lanes}


def write_registry(store: Store, payload: dict) -> Path:
    """Write registry.json under the store portfolio root."""

    path = store.root / "portfolio" / "registry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Cross-lane diversity fixtures (P2)
# --------------------------------------------------------------------------- #


def diversity_era_ids(*, n_eras: int = 6, ids_per_era: int = 3) -> tuple[tuple[str, str], ...]:
    """A globally-unique (era, id) panel; unique ids are required for benchmark attach."""

    return tuple((f"e{era:02d}", f"id_{era:02d}_{row}") for era in range(n_eras) for row in range(ids_per_era))


def build_diversity_store(
    tmp_path: Path,
    *,
    n_eras: int = 6,
    ids_per_era: int = 3,
    lane_a_bmc: float = 0.0050,
    lane_b_bmc: float = 0.0050,
    lane_a_role: str = "believed_best",
    lane_b_role: str = "believed_best",
    policy_filled: bool = True,
    seed: int = 0,
) -> Store:
    """Two lanes (distinct recipes) sharing one comparison surface + an active benchmark.

    Both lanes are full trios (42/17/99) over the same (era, id) panel, target, benchmark
    sha, engine, and data scope, so every candidate resolves to one shared surface_id — the
    happy path for `portfolio_diversity`. Only ``model.params.max_depth`` differs between the
    two lanes, giving two distinct recipe keys with identical surfaces.
    """

    import numpy as np

    store = build_store(tmp_path)
    era_ids = diversity_era_ids(n_eras=n_eras, ids_per_era=ids_per_era)
    n_rows = len(era_ids)
    rng = np.random.default_rng(seed)

    target = [float(value) for value in rng.random(n_rows)]
    write_active_benchmark(store, era_ids=era_ids, predictions=[float(value) for value in rng.random(n_rows)])

    journal_rows: list[dict | str] = []
    lanes: list[dict] = []
    for role, bmc, depth, cand_id, lane_id, axis in (
        (lane_a_role, lane_a_bmc, 9, "cand_alpha", "lane_alpha", "feature_scope"),
        (lane_b_role, lane_b_bmc, 6, "cand_beta", "lane_beta", "target_family"),
    ):
        anchor = f"config_{lane_id}_s42.json"
        for seed_val in (42, 17, 99):
            name = f"config_{lane_id}_s{seed_val}.json"
            config = valid_config(
                random_state=seed_val, predictions_name=f"pred_{lane_id}_s{seed_val}", max_depth=depth
            )
            write_config(store, name, config)
            run_id = f"r_{lane_id}_s{seed_val}"
            build_run(
                store,
                run_id=run_id,
                config=config,
                bmc=bmc,
                era_ids=era_ids,
                predictions=[float(value) for value in rng.random(n_rows)],
                targets=target,
            )
            journal_rows.append(journal_row(name, seed=seed_val, metric=bmc, run_id=run_id))
        lanes.append(
            lane_block(
                lane_id=lane_id,
                store=store,
                axis=axis,
                candidates=[{"candidate_id": cand_id, "role": role, "anchor_config": anchor}],
            )
        )

    write_journal(store, journal_rows)
    write_state(store, {"total_rounds_completed": len(journal_rows)})
    write_registry(store, registry_with_lanes(lanes=lanes, policy_filled=policy_filled))
    return store


# --------------------------------------------------------------------------- #
# Combination study fixtures (P3)
# --------------------------------------------------------------------------- #

STUDY_DECISION_RECORD_ID = "DR-STUDY-1"
LANE_ALPHA_DEPTH = 9
LANE_BETA_DEPTH = 6


def build_study_store(
    tmp_path: Path,
    *,
    n_eras: int = 24,
    ids_per_era: int = 2,
    policy_filled: bool = True,
) -> Store:
    """Two-lane scale-confirmed store with enough eras for search folds + holdout.

    A superset of ``build_diversity_store`` sized for the combination study: both
    lanes are full trios over one shared surface, so `study freeze` resolves them
    as valid members.
    """

    return build_diversity_store(tmp_path, n_eras=n_eras, ids_per_era=ids_per_era, policy_filled=policy_filled)


def freeze_payload(
    store: Store,
    *,
    study_id: str = "S1",
    decision_record_id: str = STUDY_DECISION_RECORD_ID,
    members: list[dict] | None = None,
    baseline_candidate_id: str = "cand_alpha",
    holdout_n_eras: int = 6,
    era_gap: int = 2,
    meta_mode: str = "expanding",
    min_history_eras: int = 2,
    validation_width_eras: int = 4,
    step_eras: int = 4,
    gap_eras: int = 1,
    block_length_eras: int = 3,
    n_resamples: int = 200,
    rng_seed: int = 7,
    study_trial_cap: int = 4,
    neutralization: dict | None = None,
    exploratory: bool = False,
) -> dict:
    """Assemble a strictly-valid freeze.json payload for the two default lanes."""

    if members is None:
        members = [
            {"candidate_id": "cand_alpha", "lane_id": "lane_alpha", "anchor_config": "config_lane_alpha_s42.json"},
            {"candidate_id": "cand_beta", "lane_id": "lane_beta", "anchor_config": "config_lane_beta_s42.json"},
        ]
    payload: dict = {
        "schema_version": 1,
        "study_id": study_id,
        "experiment_id": store.experiment_id,
        "decision_record_id": decision_record_id,
        "baseline_candidate_id": baseline_candidate_id,
        "members": members,
        "split": {"mode": "chronological_suffix", "holdout_n_eras": holdout_n_eras, "era_gap": era_gap},
        "meta_validation": {
            "mode": meta_mode,
            "min_history_eras": min_history_eras,
            "validation_width_eras": validation_width_eras,
            "step_eras": step_eras,
            "gap_eras": gap_eras,
        },
        "inference": {"block_length_eras": block_length_eras, "n_resamples": n_resamples, "rng_seed": rng_seed},
        "study_trial_cap": study_trial_cap,
        "exploratory": exploratory,
    }
    if neutralization is not None:
        payload["neutralization"] = neutralization
    return payload


def study_trial(
    *,
    trial_id: str,
    alpha_weight: float = 0.5,
    beta_weight: float = 0.5,
    alpha_candidates: tuple[str, ...] = ("cand_alpha",),
    beta_candidates: tuple[str, ...] = ("cand_beta",),
    neutralization_p: float = 0.0,
) -> dict:
    """One trial dict weighting the two default lanes (weights stay under the 0.5 cap)."""

    return {
        "trial_id": trial_id,
        "selection": {"lane_alpha": list(alpha_candidates), "lane_beta": list(beta_candidates)},
        "lane_weights": {"lane_alpha": alpha_weight, "lane_beta": beta_weight},
        "neutralization_p": neutralization_p,
    }


def trials_payload(*, study_id: str = "S1", trials: list[dict] | None = None) -> dict:
    """Assemble a trials.json payload; defaults to a single even-weight trial."""

    if trials is None:
        trials = [study_trial(trial_id="trial_a")]
    return {"study_id": study_id, "trials": trials}


def write_json_file(tmp_path: Path, payload: dict, *, name: str) -> Path:
    """Write one JSON config (freeze/trials) next to the workspace and return its path."""

    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def append_journal(store: Store, row: dict) -> None:
    """Append one line to an existing journal.jsonl (build_* wrote it first)."""

    path = store.agentic_dir / "journal.jsonl"
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    path.write_text(existing + json.dumps(row) + "\n", encoding="utf-8")


def add_extra_seed_run(store: Store, *, lane_id: str, seed: int, max_depth: int, bmc: float = 0.005) -> str:
    """Add a same-recipe run at a non-trio seed so the candidate's trio is incomplete."""

    name = f"config_{lane_id}_s{seed}.json"
    config = valid_config(random_state=seed, predictions_name=f"pred_{lane_id}_s{seed}", max_depth=max_depth)
    write_config(store, name, config)
    run_id = f"r_{lane_id}_s{seed}"
    build_run(store, run_id=run_id, config=config, bmc=bmc)
    append_journal(store, journal_row(name, seed=seed, metric=bmc, run_id=run_id))
    return run_id


def set_run_profile(store: Store, run_id: str, profile: str) -> None:
    """Rewrite one run's training.engine.profile (used for the FHR-rejection fixture)."""

    path = store.root / "runs" / run_id / "run.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["training"] = {"engine": {"profile": profile}}
    path.write_text(json.dumps(manifest), encoding="utf-8")


def set_run_benchmark_sha(store: Store, run_id: str, sha: str) -> None:
    """Rewrite one run's benchmark provenance sha so its surface_id diverges."""

    path = store.root / "runs" / run_id / "score_provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    provenance.setdefault("sources", {}).setdefault("benchmark", {})["sha256"] = sha
    path.write_text(json.dumps(provenance), encoding="utf-8")


def tamper_run_predictions(store: Store, run_id: str) -> None:
    """Rewrite one run's prediction values (same row keys) to trip the frozen-input guard."""

    import pandas as pd

    path = store.root / "runs" / run_id / "artifacts" / "predictions" / "pred_run.parquet"
    frame = pd.read_parquet(path)
    frame["prediction"] = frame["prediction"].to_numpy() * 0.5 + 0.01
    frame.to_parquet(path)
