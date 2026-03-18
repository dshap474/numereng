from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow.parquet as pq
import pytest

import numereng.features.ensemble.service as service_module
from numereng.features.ensemble import EnsembleBuildRequest
from numereng.features.ensemble.weights import EnsembleWeightsError


def _write_run_predictions(store_root: Path, run_id: str, frame: pd.DataFrame) -> None:
    run_dir = store_root / "runs" / run_id
    predictions_dir = run_dir / "artifacts" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = predictions_dir / "predictions.parquet"
    frame.to_parquet(predictions_path, index=False)

    manifest = {
        "run_id": run_id,
        "artifacts": {
            "predictions": "artifacts/predictions/predictions.parquet",
        },
    }
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_build_ensemble_persists_components_metrics_and_artifacts(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"

    run_a = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["0001", "0001", "0002", "0002"],
            "target_ender_20": [0.1, 0.9, 0.2, 0.8],
            "prediction": [0.2, 0.7, 0.3, 0.6],
        }
    )
    run_b = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["0001", "0001", "0002", "0002"],
            "target_ender_20": [0.1, 0.9, 0.2, 0.8],
            "prediction": [0.3, 0.6, 0.25, 0.75],
        }
    )

    _write_run_predictions(store_root, "run-a", run_a)
    _write_run_predictions(store_root, "run-b", run_b)

    result = service_module.build_ensemble(
        store_root=store_root,
        request=EnsembleBuildRequest(
            experiment_id="2026-02-22_test-exp",
            run_ids=("run-a", "run-b"),
            metric="corr_sharpe",
            target="target_ender_20",
            optimize_weights=False,
        ),
    )

    assert result.status == "completed"
    assert len(result.components) == 2
    assert result.components[0].run_id == "run-a"
    assert result.components[1].run_id == "run-b"

    metrics = {item.name: item.value for item in result.metrics}
    assert "corr_mean" in metrics
    assert "corr_sharpe" in metrics

    artifacts_path = result.artifacts_path
    assert (artifacts_path / "predictions.parquet").is_file()
    assert (artifacts_path / "correlation_matrix.parquet").is_file()
    assert (artifacts_path / "metrics.json").is_file()
    assert (artifacts_path / "weights.parquet").is_file()
    assert (artifacts_path / "component_metrics.parquet").is_file()
    assert (artifacts_path / "era_metrics.parquet").is_file()
    assert (artifacts_path / "regime_metrics.parquet").is_file()
    assert (artifacts_path / "lineage.json").is_file()
    assert not (artifacts_path / "component_predictions.parquet").exists()
    assert not (artifacts_path / "bootstrap_metrics.json").exists()

    assert result.config["include_heavy_artifacts"] is False
    assert result.config["selection_note"] is None
    assert result.config["regime_buckets"] == 4
    assert result.config["artifacts"] == {
        "always": [
            "predictions.parquet",
            "correlation_matrix.parquet",
            "metrics.json",
            "weights.parquet",
            "component_metrics.parquet",
            "era_metrics.parquet",
            "regime_metrics.parquet",
            "lineage.json",
        ],
        "heavy": [],
    }
    for filename in (
        "predictions.parquet",
        "correlation_matrix.parquet",
        "weights.parquet",
        "component_metrics.parquet",
        "era_metrics.parquet",
        "regime_metrics.parquet",
    ):
        parquet_file = pq.ParquetFile(artifacts_path / filename)
        assert parquet_file.metadata.row_group(0).column(0).compression == "ZSTD"

    loaded = service_module.get_ensemble_view(store_root=store_root, ensemble_id=result.ensemble_id)
    assert loaded.ensemble_id == result.ensemble_id
    assert len(loaded.components) == 2

    listed = service_module.list_ensembles_view(store_root=store_root, experiment_id="2026-02-22_test-exp")
    assert len(listed) == 1
    assert listed[0].ensemble_id == result.ensemble_id


def test_build_ensemble_requires_two_run_ids(tmp_path: Path) -> None:
    with pytest.raises(service_module.EnsembleValidationError, match="ensemble_run_ids_insufficient"):
        service_module.build_ensemble(
            store_root=tmp_path / ".numereng",
            request=EnsembleBuildRequest(run_ids=("run-only",)),
        )


def test_build_ensemble_rejects_invalid_method(tmp_path: Path) -> None:
    with pytest.raises(service_module.EnsembleValidationError, match="ensemble_method_invalid"):
        service_module.build_ensemble(
            store_root=tmp_path / ".numereng",
            request=EnsembleBuildRequest(
                run_ids=("run-a", "run-b"),
                method=cast(Any, "forward"),
            ),
        )


def test_build_ensemble_rejects_invalid_experiment_id(tmp_path: Path) -> None:
    with pytest.raises(service_module.EnsembleValidationError, match="ensemble_experiment_id_invalid"):
        service_module.build_ensemble(
            store_root=tmp_path / ".numereng",
            request=EnsembleBuildRequest(
                run_ids=("run-a", "run-b"),
                experiment_id="bad/id",
            ),
        )


def test_build_ensemble_rejects_invalid_ensemble_id(tmp_path: Path) -> None:
    with pytest.raises(service_module.EnsembleValidationError, match="ensemble_id_invalid"):
        service_module.build_ensemble(
            store_root=tmp_path / ".numereng",
            request=EnsembleBuildRequest(
                run_ids=("run-a", "run-b"),
                ensemble_id="bad/id",
            ),
        )


def test_build_ensemble_dedupes_run_ids(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"

    run_a = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["0001", "0001", "0002", "0002"],
            "target_ender_20": [0.1, 0.9, 0.2, 0.8],
            "prediction": [0.2, 0.7, 0.3, 0.6],
        }
    )
    run_b = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["0001", "0001", "0002", "0002"],
            "target_ender_20": [0.1, 0.9, 0.2, 0.8],
            "prediction": [0.3, 0.6, 0.25, 0.75],
        }
    )
    _write_run_predictions(store_root, "run-a", run_a)
    _write_run_predictions(store_root, "run-b", run_b)

    result = service_module.build_ensemble(
        store_root=store_root,
        request=EnsembleBuildRequest(
            run_ids=("run-a", "run-a", "run-b"),
            experiment_id="exp-1",
        ),
    )

    assert [component.run_id for component in result.components] == ["run-a", "run-b"]


def test_build_ensemble_sets_warning_when_optimization_skipped_missing_target(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"

    run_a = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["0001", "0001", "0002", "0002"],
            "prediction": [0.2, 0.7, 0.3, 0.6],
        }
    )
    run_b = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["0001", "0001", "0002", "0002"],
            "prediction": [0.3, 0.6, 0.25, 0.75],
        }
    )
    _write_run_predictions(store_root, "run-a", run_a)
    _write_run_predictions(store_root, "run-b", run_b)

    result = service_module.build_ensemble(
        store_root=store_root,
        request=EnsembleBuildRequest(
            run_ids=("run-a", "run-b"),
            experiment_id="exp-1",
            optimize_weights=True,
        ),
    )

    assert result.config["optimization_warning"] == "ensemble_weight_optimization_skipped_missing_target"


def test_build_ensemble_sets_warning_when_optimization_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"

    run_a = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["0001", "0001", "0002", "0002"],
            "target_ender_20": [0.1, 0.9, 0.2, 0.8],
            "prediction": [0.2, 0.7, 0.3, 0.6],
        }
    )
    run_b = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["0001", "0001", "0002", "0002"],
            "target_ender_20": [0.1, 0.9, 0.2, 0.8],
            "prediction": [0.3, 0.6, 0.25, 0.75],
        }
    )
    _write_run_predictions(store_root, "run-a", run_a)
    _write_run_predictions(store_root, "run-b", run_b)

    def fake_optimize_weights(
        *,
        ranked_predictions: pd.DataFrame,
        era_series: pd.Series,
        target_series: pd.Series,
        metric: str,
        initial_weights: tuple[float, ...],
    ) -> tuple[float, ...]:
        _ = (ranked_predictions, era_series, target_series, metric, initial_weights)
        raise EnsembleWeightsError("optimization_failed")

    monkeypatch.setattr(service_module, "optimize_weights", fake_optimize_weights)

    result = service_module.build_ensemble(
        store_root=store_root,
        request=EnsembleBuildRequest(
            run_ids=("run-a", "run-b"),
            experiment_id="exp-1",
            optimize_weights=True,
        ),
    )

    assert result.config["optimization_warning"] == "optimization_failed"


def test_build_ensemble_writes_heavy_artifacts_when_enabled(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"

    run_a = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["0001", "0001", "0002", "0002"],
            "target_ender_20": [0.1, 0.9, 0.2, 0.8],
            "prediction": [0.2, 0.7, 0.3, 0.6],
        }
    )
    run_b = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["0001", "0001", "0002", "0002"],
            "target_ender_20": [0.1, 0.9, 0.2, 0.8],
            "prediction": [0.3, 0.6, 0.25, 0.75],
        }
    )
    _write_run_predictions(store_root, "run-a", run_a)
    _write_run_predictions(store_root, "run-b", run_b)

    result = service_module.build_ensemble(
        store_root=store_root,
        request=EnsembleBuildRequest(
            run_ids=("run-a", "run-b"),
            experiment_id="exp-1",
            include_heavy_artifacts=True,
            selection_note="  keep these two runs  ",
            regime_buckets=3,
        ),
    )

    artifacts_path = result.artifacts_path
    assert (artifacts_path / "component_predictions.parquet").is_file()
    assert (artifacts_path / "bootstrap_metrics.json").is_file()
    assert result.config["include_heavy_artifacts"] is True
    assert result.config["selection_note"] == "keep these two runs"
    assert result.config["regime_buckets"] == 3
    assert result.config["artifacts"]["heavy"] == [
        "component_predictions.parquet",
        "bootstrap_metrics.json",
    ]


def test_build_ensemble_removes_stale_heavy_artifacts_when_disabled(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"

    run_a = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["0001", "0001", "0002", "0002"],
            "target_ender_20": [0.1, 0.9, 0.2, 0.8],
            "prediction": [0.2, 0.7, 0.3, 0.6],
        }
    )
    run_b = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["0001", "0001", "0002", "0002"],
            "target_ender_20": [0.1, 0.9, 0.2, 0.8],
            "prediction": [0.3, 0.6, 0.25, 0.75],
        }
    )
    _write_run_predictions(store_root, "run-a", run_a)
    _write_run_predictions(store_root, "run-b", run_b)

    first = service_module.build_ensemble(
        store_root=store_root,
        request=EnsembleBuildRequest(
            run_ids=("run-a", "run-b"),
            experiment_id="exp-1",
            ensemble_id="ens-fixed",
            include_heavy_artifacts=True,
        ),
    )
    assert (first.artifacts_path / "component_predictions.parquet").is_file()
    assert (first.artifacts_path / "bootstrap_metrics.json").is_file()

    second = service_module.build_ensemble(
        store_root=store_root,
        request=EnsembleBuildRequest(
            run_ids=("run-a", "run-b"),
            experiment_id="exp-1",
            ensemble_id="ens-fixed",
            include_heavy_artifacts=False,
        ),
    )
    assert not (second.artifacts_path / "component_predictions.parquet").exists()
    assert not (second.artifacts_path / "bootstrap_metrics.json").exists()


def test_build_ensemble_cleans_new_artifacts_when_save_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"

    run_a = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["0001", "0001", "0002", "0002"],
            "target_ender_20": [0.1, 0.9, 0.2, 0.8],
            "prediction": [0.2, 0.7, 0.3, 0.6],
        }
    )
    run_b = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["0001", "0001", "0002", "0002"],
            "target_ender_20": [0.1, 0.9, 0.2, 0.8],
            "prediction": [0.3, 0.6, 0.25, 0.75],
        }
    )
    _write_run_predictions(store_root, "run-a", run_a)
    _write_run_predictions(store_root, "run-b", run_b)

    monkeypatch.setattr(
        service_module,
        "save_ensemble",
        lambda **_: (_ for _ in ()).throw(RuntimeError("ensemble_store_write_failed")),
    )

    with pytest.raises(RuntimeError, match="ensemble_store_write_failed"):
        service_module.build_ensemble(
            store_root=store_root,
            request=EnsembleBuildRequest(
                run_ids=("run-a", "run-b"),
                experiment_id="exp-1",
                ensemble_id="ens-save-fail",
            ),
        )

    artifacts_path = store_root / "experiments" / "exp-1" / "ensembles" / "ens-save-fail"
    assert not artifacts_path.exists()


def test_build_ensemble_rejects_invalid_regime_bucket_count(tmp_path: Path) -> None:
    with pytest.raises(service_module.EnsembleValidationError, match="ensemble_regime_buckets_invalid"):
        service_module.build_ensemble(
            store_root=tmp_path / ".numereng",
            request=EnsembleBuildRequest(
                run_ids=("run-a", "run-b"),
                regime_buckets=1,
            ),
        )
