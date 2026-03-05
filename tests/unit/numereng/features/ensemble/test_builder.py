from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import numereng.features.ensemble.builder as builder_module


def _write_manifest(run_dir: Path, *, predictions_relpath: str | None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"run_id": run_dir.name}
    if predictions_relpath is not None:
        payload["artifacts"] = {"predictions": predictions_relpath}
    (run_dir / "run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_predictions_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_load_ranked_components_requires_run_manifest(tmp_path: Path) -> None:
    with pytest.raises(builder_module.EnsembleBuildError, match="ensemble_run_manifest_not_found:run-a"):
        builder_module.load_ranked_components(
            store_root=tmp_path / ".numereng",
            run_ids=("run-a",),
            target_col="target_ender_20",
        )


def test_load_ranked_components_rejects_invalid_manifest_json(tmp_path: Path) -> None:
    run_dir = tmp_path / ".numereng" / "runs" / "run-a"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text("{bad-json", encoding="utf-8")

    with pytest.raises(builder_module.EnsembleBuildError, match="ensemble_run_manifest_invalid:run-a"):
        builder_module.load_ranked_components(
            store_root=tmp_path / ".numereng",
            run_ids=("run-a",),
            target_col="target_ender_20",
        )


def test_load_ranked_components_requires_predictions_file(tmp_path: Path) -> None:
    run_dir = tmp_path / ".numereng" / "runs" / "run-a"
    _write_manifest(run_dir, predictions_relpath="artifacts/predictions/predictions.csv")

    with pytest.raises(builder_module.EnsembleBuildError, match="ensemble_run_predictions_not_found:run-a"):
        builder_module.load_ranked_components(
            store_root=tmp_path / ".numereng",
            run_ids=("run-a",),
            target_col="target_ender_20",
        )


def test_load_ranked_components_rejects_missing_required_keys(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-a"
    _write_manifest(run_dir, predictions_relpath="artifacts/predictions/predictions.csv")
    _write_predictions_csv(
        run_dir / "artifacts" / "predictions" / "predictions.csv",
        pd.DataFrame({"id": ["a"], "prediction": [0.1]}),
    )

    with pytest.raises(builder_module.EnsembleBuildError, match="ensemble_predictions_missing_keys:run-a"):
        builder_module.load_ranked_components(
            store_root=store_root,
            run_ids=("run-a",),
            target_col="target_ender_20",
        )


def test_load_ranked_components_rejects_unsupported_prediction_format(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-a"
    _write_manifest(run_dir, predictions_relpath="artifacts/predictions/predictions.txt")
    predictions_path = run_dir / "artifacts" / "predictions" / "predictions.txt"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text("unsupported", encoding="utf-8")

    with pytest.raises(builder_module.EnsembleBuildError, match="ensemble_predictions_format_unsupported"):
        builder_module.load_ranked_components(
            store_root=store_root,
            run_ids=("run-a",),
            target_col="target_ender_20",
        )


def test_load_ranked_components_rejects_missing_prediction_column(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-a"
    _write_manifest(run_dir, predictions_relpath="artifacts/predictions/predictions.csv")
    _write_predictions_csv(
        run_dir / "artifacts" / "predictions" / "predictions.csv",
        pd.DataFrame({"era": ["0001"], "id": ["a"], "label": ["x"]}),
    )

    with pytest.raises(builder_module.EnsembleBuildError, match="ensemble_prediction_column_missing"):
        builder_module.load_ranked_components(
            store_root=store_root,
            run_ids=("run-a",),
            target_col="target_ender_20",
        )


def test_load_ranked_components_rejects_empty_alignment(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"

    run_a_dir = store_root / "runs" / "run-a"
    _write_manifest(run_a_dir, predictions_relpath="artifacts/predictions/predictions.csv")
    _write_predictions_csv(
        run_a_dir / "artifacts" / "predictions" / "predictions.csv",
        pd.DataFrame({"era": ["0001"], "id": ["a"], "prediction": [0.1]}),
    )

    run_b_dir = store_root / "runs" / "run-b"
    _write_manifest(run_b_dir, predictions_relpath="artifacts/predictions/predictions.csv")
    _write_predictions_csv(
        run_b_dir / "artifacts" / "predictions" / "predictions.csv",
        pd.DataFrame({"era": ["0001"], "id": ["b"], "prediction": [0.2]}),
    )

    with pytest.raises(builder_module.EnsembleBuildError, match="ensemble_component_alignment_empty"):
        builder_module.load_ranked_components(
            store_root=store_root,
            run_ids=("run-a", "run-b"),
            target_col="target_ender_20",
        )


def test_build_blended_predictions_rejects_weight_mismatch() -> None:
    ranked = pd.DataFrame({"pred_a": [0.1, 0.2], "pred_b": [0.3, 0.4]})
    with pytest.raises(builder_module.EnsembleBuildError, match="ensemble_weights_length_mismatch"):
        builder_module.build_blended_predictions(ranked_predictions=ranked, weights=(1.0,))
