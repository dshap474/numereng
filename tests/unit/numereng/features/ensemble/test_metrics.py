from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import numereng.features.ensemble.metrics as metrics_module


def test_summarize_metrics_returns_none_values_without_target_series() -> None:
    metrics = metrics_module.summarize_metrics(
        blended=np.array([0.1, 0.2], dtype=float),
        era_series=pd.Series(["0001", "0001"]),
        target_series=None,
    )
    by_name = {metric.name: metric.value for metric in metrics}
    assert by_name == {
        "corr_mean": None,
        "corr_sharpe": None,
        "max_drawdown": None,
    }


def test_summarize_metrics_sets_sharpe_none_when_per_era_std_zero() -> None:
    metrics = metrics_module.summarize_metrics(
        blended=np.array([0.0, 1.0, 0.0, 1.0], dtype=float),
        era_series=pd.Series(["0001", "0001", "0002", "0002"]),
        target_series=pd.Series([0.0, 1.0, 0.0, 1.0], dtype=float),
    )
    by_name = {metric.name: metric.value for metric in metrics}
    assert by_name["corr_mean"] == pytest.approx(1.0)
    assert by_name["corr_sharpe"] is None
    assert by_name["max_drawdown"] == pytest.approx(0.0)


def test_correlation_matrix_rejects_component_shape_mismatch() -> None:
    ranked = pd.DataFrame({"pred_a": [0.1, 0.2], "pred_b": [0.3, 0.4]})
    with pytest.raises(ValueError, match="ensemble_component_shape_mismatch"):
        metrics_module.correlation_matrix(ranked_predictions=ranked, run_ids=("run-a",))


def test_metric_dict_maps_metric_rows_by_name() -> None:
    metrics = metrics_module.summarize_metrics(
        blended=np.array([0.1, 0.2], dtype=float),
        era_series=pd.Series(["0001", "0001"]),
        target_series=None,
    )
    payload = metrics_module.metric_dict(metrics)
    assert set(payload.keys()) == {"corr_mean", "corr_sharpe", "max_drawdown"}


def test_regime_metrics_table_rejects_invalid_bucket_count() -> None:
    with pytest.raises(ValueError, match="ensemble_regime_buckets_invalid"):
        metrics_module.regime_metrics_table(
            per_era_corr=pd.Series([0.1, 0.2], index=["0001", "0002"], dtype=float),
            regime_buckets=1,
        )


def test_component_metrics_table_rejects_weight_length_mismatch() -> None:
    ranked = pd.DataFrame({"pred_a": [0.1, 0.2], "pred_b": [0.3, 0.4]})
    with pytest.raises(ValueError, match="ensemble_weights_length_mismatch"):
        metrics_module.component_metrics_table(
            ranked_predictions=ranked,
            run_ids=("run-a", "run-b"),
            era_series=pd.Series(["0001", "0002"]),
            target_series=pd.Series([0.0, 1.0], dtype=float),
            weights=(1.0,),
        )


def test_bootstrap_metric_summary_empty_input_returns_null_payload() -> None:
    payload = metrics_module.bootstrap_metric_summary(
        per_era_corr=pd.Series(dtype=float),
        n_resamples=8,
        seed=7,
    )
    assert payload["n_eras"] == 0
    assert payload["n_resamples"] == 8
    assert payload["seed"] == 7
    assert payload["metrics"]["corr_mean"]["mean"] is None
