from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from numereng.features.ensemble import EnsembleSelectionRequest, EnsembleSelectionSourceRule
from numereng.features.ensemble.selection import _era_ranges, _score_weight_matrix, _weight_simplex, select_ensemble
from numereng.features.scoring._fastops import correlation_contribution_matrix, numerai_corr_matrix_vs_target


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _base_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    targets = {
        "era1": [0.1, 0.4, 0.7, 0.2],
        "era2": [0.3, 0.8, 0.6, 0.1],
        "era3": [0.2, 0.5, 0.9, 0.4],
    }
    for era, values in targets.items():
        for index, target_value in enumerate(values, start=1):
            rows.append(
                {
                    "id": f"{era}_{index}",
                    "era": era,
                    "target_ender_20": target_value,
                    "target_alpha_20": target_value,
                    "target_charlie_20": target_value * 0.9,
                    "target_delta_60": target_value * 1.1,
                    "target_echo_60": target_value * 0.8,
                }
            )
    return pd.DataFrame(rows)


def _prediction_series(frame: pd.DataFrame, *, flavor: str) -> np.ndarray:
    target = frame["target_ender_20"].to_numpy(dtype=float)
    if flavor == "alpha":
        return target + np.array([0.03, -0.01, 0.02, -0.02] * 3, dtype=float)
    if flavor == "charlie":
        return target + np.array([0.02, 0.01, -0.01, -0.03] * 3, dtype=float)
    if flavor == "delta":
        return target[::-1] + np.array([0.01, -0.02, 0.0, 0.03] * 3, dtype=float)
    if flavor == "echo":
        return (1.0 - target) + np.array([-0.01, 0.0, 0.02, -0.02] * 3, dtype=float)
    raise ValueError(flavor)


def _write_active_benchmark(store_root: Path, base_frame: pd.DataFrame) -> None:
    benchmark_dir = store_root / "datasets" / "baselines" / "active_benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    benchmark = base_frame[["id", "era"]].copy()
    benchmark["prediction"] = base_frame["target_ender_20"].to_numpy(dtype=float) * 0.75 + 0.05
    benchmark.to_parquet(benchmark_dir / "predictions.parquet", index=False)
    _write_json(
        benchmark_dir / "benchmark.json",
        {
            "name": "active_benchmark",
            "predictions_path": str((benchmark_dir / "predictions.parquet").resolve()),
        },
    )


def _write_run(
    store_root: Path,
    *,
    experiment_id: str,
    run_id: str,
    feature_set: str,
    target_col: str,
    seed: int,
    frame: pd.DataFrame,
    metrics_summary: dict[str, dict[str, float]],
) -> None:
    run_dir = store_root / "runs" / run_id
    predictions_dir = run_dir / "artifacts" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(predictions_dir / "predictions.parquet", index=False)
    _write_json(
        run_dir / "run.json",
        {
            "run_id": run_id,
            "status": "FINISHED",
            "experiment_id": experiment_id,
            "artifacts": {"predictions": "artifacts/predictions/predictions.parquet"},
            "data": {
                "feature_set": feature_set,
                "target_col": target_col,
                "version": "v5.2",
            },
            "training": {
                "data": {
                    "dataset_variant": "non_downsampled",
                    "dataset_scope": "train_plus_validation",
                }
            },
            "config": {"path": f"configs/r1_{target_col}_seed{seed}.json"},
            "metrics_summary": metrics_summary,
        },
    )
    _write_json(run_dir / "resolved.json", {"model": {"params": {"random_state": seed}}})
    _write_json(
        run_dir / "results.json",
        {
            "benchmark": {
                "mode": "active",
                "name": "active_benchmark",
                "file": str(
                    (store_root / "datasets" / "baselines" / "active_benchmark" / "predictions.parquet").resolve()
                ),
            }
        },
    )


def _write_experiment_manifest(store_root: Path, experiment_id: str, run_ids: list[str]) -> None:
    experiment_dir = store_root / "experiments" / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    _write_json(experiment_dir / "experiment.json", {"experiment_id": experiment_id, "runs": run_ids})


def _seeded_frame(base_frame: pd.DataFrame, *, flavor: str, seed_index: int) -> pd.DataFrame:
    frame = base_frame.copy()
    frame["prediction"] = _prediction_series(base_frame, flavor=flavor) + (seed_index * 0.0025)
    return frame


def _metric_summary(value: float, corr: float) -> dict[str, dict[str, float]]:
    return {
        "bmc_last_200_eras": {"mean": value},
        "bmc": {"mean": value - 0.01},
        "corr": {"mean": corr},
    }


def test_select_ensemble_uses_metric_rank_for_top3_overall_and_writes_artifacts(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    base_frame = _base_rows()
    _write_active_benchmark(store_root, base_frame)

    experiment_specs = {
        "medium-exp": [
            ("target_charlie_20", "charlie", 0.06, 0.04),
            ("target_alpha_20", "alpha", 0.08, 0.05),
        ],
        "small-exp": [
            ("target_echo_60", "echo", 0.02, 0.01),
            ("target_delta_60", "delta", 0.04, 0.03),
        ],
    }
    for experiment_id, specs in experiment_specs.items():
        run_ids: list[str] = []
        feature_set = "medium" if experiment_id == "medium-exp" else "small"
        for target_col, flavor, bmc_last_200, corr_mean in specs:
            for seed_index, seed in enumerate((42, 43, 44), start=1):
                run_id = f"{experiment_id}-{target_col}-{seed}"
                frame = _seeded_frame(base_frame, flavor=flavor, seed_index=seed_index)
                _write_run(
                    store_root,
                    experiment_id=experiment_id,
                    run_id=run_id,
                    feature_set=feature_set,
                    target_col=target_col,
                    seed=seed,
                    frame=frame,
                    metrics_summary=_metric_summary(bmc_last_200, corr_mean),
                )
                run_ids.append(run_id)
        _write_experiment_manifest(store_root, experiment_id, run_ids)

    _write_experiment_manifest(store_root, "selection-exp", [])

    result = select_ensemble(
        store_root=store_root,
        request=EnsembleSelectionRequest(
            experiment_id="selection-exp",
            selection_id="unit-test",
            correlation_threshold=0.999,
            source_experiment_ids=("medium-exp", "small-exp"),
            source_rules=(
                EnsembleSelectionSourceRule(
                    experiment_id="medium-exp",
                    selection_mode="explicit_targets",
                    explicit_targets=("target_charlie_20", "target_alpha_20"),
                ),
                EnsembleSelectionSourceRule(
                    experiment_id="small-exp",
                    selection_mode="explicit_targets",
                    explicit_targets=("target_echo_60", "target_delta_60"),
                ),
            ),
        ),
    )

    assert result.frozen_candidate_count == 4
    assert result.equal_weight_variant_count >= 1
    equal_weight_results = json.loads(
        (result.artifacts_path / "blends" / "equal_weight_results.json").read_text(encoding="utf-8")
    )
    top3 = next(row for row in equal_weight_results if row["blend_id"] == "top3_overall")
    assert top3["component_ids"] == [
        "medium_target_alpha_20",
        "small_target_delta_60",
        "small_target_echo_60",
    ]
    pruning_rows = pd.read_csv(result.artifacts_path / "correlation" / "pruning_recommendation.csv")
    assert "medium_target_charlie_20" in pruning_rows.loc[pruning_rows["status"] == "pruned", "candidate_id"].tolist()

    final_payload = json.loads((result.artifacts_path / "blends" / "final_selection.json").read_text(encoding="utf-8"))
    assert final_payload["winner"]["blend_id"]
    assert final_payload["winner"]["selection_mode"] in {"equal_weight", "weighted", "equal_weight_retained"}
    assert (result.artifacts_path / "blends" / "weighted_candidates.csv").is_file()
    assert (result.artifacts_path / "correlation" / "pruning_recommendation.csv").is_file()


def test_select_ensemble_allows_single_seed_candidates_by_default(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    base_frame = _base_rows()
    _write_active_benchmark(store_root, base_frame)

    medium_run_ids: list[str] = []
    for seed_index, seed in enumerate((42, 43, 44), start=1):
        run_id = f"medium-exp-target_alpha_20-{seed}"
        _write_run(
            store_root,
            experiment_id="medium-exp",
            run_id=run_id,
            feature_set="medium",
            target_col="target_alpha_20",
            seed=seed,
            frame=_seeded_frame(base_frame, flavor="alpha", seed_index=seed_index),
            metrics_summary=_metric_summary(0.08, 0.05),
        )
        medium_run_ids.append(run_id)
    _write_experiment_manifest(store_root, "medium-exp", medium_run_ids)

    single_run_id = "small-exp-target_delta_60-42"
    _write_run(
        store_root,
        experiment_id="small-exp",
        run_id=single_run_id,
        feature_set="small",
        target_col="target_delta_60",
        seed=42,
        frame=_seeded_frame(base_frame, flavor="delta", seed_index=1),
        metrics_summary=_metric_summary(0.04, 0.03),
    )
    _write_experiment_manifest(store_root, "small-exp", [single_run_id])
    _write_experiment_manifest(store_root, "selection-exp", [])

    result = select_ensemble(
        store_root=store_root,
        request=EnsembleSelectionRequest(
            experiment_id="selection-exp",
            selection_id="single-seed-ok",
            source_experiment_ids=("medium-exp", "small-exp"),
            source_rules=(
                EnsembleSelectionSourceRule(
                    experiment_id="medium-exp",
                    selection_mode="explicit_targets",
                    explicit_targets=("target_alpha_20",),
                ),
                EnsembleSelectionSourceRule(
                    experiment_id="small-exp",
                    selection_mode="explicit_targets",
                    explicit_targets=("target_delta_60",),
                ),
            ),
        ),
    )

    frozen_candidates = json.loads(
        (result.artifacts_path / "candidates" / "frozen_candidates.json").read_text(encoding="utf-8")
    )
    single_seed_row = next(row for row in frozen_candidates if row["candidate_id"] == "small_target_delta_60")
    assert single_seed_row["run_ids"] == [single_run_id]
    assert single_seed_row["seeds"] == [42]
    assert single_seed_row["bundle_row_count"] == len(base_frame)


def test_score_weight_matrix_matches_literal_bruteforce_reference() -> None:
    prediction_matrix = np.asarray(
        [
            [0.1, 0.8],
            [0.4, 0.3],
            [0.9, 0.2],
            [0.2, 0.7],
            [0.3, 0.5],
            [0.8, 0.1],
        ],
        dtype=np.float64,
    )
    target_vector = np.asarray([0.1, 0.5, 0.9, 0.2, 0.4, 0.8], dtype=np.float64)
    benchmark_vector = np.asarray([0.2, 0.4, 0.7, 0.3, 0.45, 0.75], dtype=np.float64)
    era_ranges = _era_ranges(["era1", "era1", "era1", "era2", "era2", "era2"])
    weight_matrix = np.asarray(_weight_simplex(2, 0.5), dtype=np.float64)

    fast = _score_weight_matrix(
        prediction_matrix=prediction_matrix,
        target_vector=target_vector,
        benchmark_vector=benchmark_vector,
        era_ranges=era_ranges,
        weight_matrix=weight_matrix,
    )

    corr_rows: list[list[float]] = []
    bmc_rows: list[list[float]] = []
    for _era, start, end in era_ranges:
        corr_values: list[float] = []
        bmc_values: list[float] = []
        pred_slice = prediction_matrix[start:end]
        target_slice = target_vector[start:end]
        benchmark_slice = benchmark_vector[start:end]
        for weights in weight_matrix:
            blended = pred_slice @ weights
            corr_values.append(float(numerai_corr_matrix_vs_target(blended.reshape(-1, 1), target_slice)[0]))
            bmc_values.append(
                float(correlation_contribution_matrix(blended.reshape(-1, 1), benchmark_slice, target_slice)[0])
            )
        corr_rows.append(corr_values)
        bmc_rows.append(bmc_values)

    corr_rows_np = np.asarray(corr_rows, dtype=np.float64)
    bmc_rows_np = np.asarray(bmc_rows, dtype=np.float64)
    np.testing.assert_allclose(fast["corr_mean"], np.nanmean(corr_rows_np, axis=0))
    np.testing.assert_allclose(fast["bmc_mean"], np.nanmean(bmc_rows_np, axis=0))
    np.testing.assert_allclose(fast["bmc_last_200_eras_mean"], np.nanmean(bmc_rows_np, axis=0))
