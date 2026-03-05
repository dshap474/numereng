from __future__ import annotations

import builtins
from typing import Any

import numpy as np
import pandas as pd
import pytest

import numereng.features.ensemble.weights as weights_module


def test_normalize_weights_defaults_to_equal_weights() -> None:
    weights = weights_module.normalize_weights(raw_weights=None, n_components=3)
    assert weights == pytest.approx((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))


@pytest.mark.parametrize(
    ("raw_weights", "n_components", "error_fragment"),
    [
        ((0.5,), 2, "ensemble_weights_length_mismatch"),
        ((-1.0, 2.0), 2, "ensemble_weights_negative"),
        ((0.0, 0.0), 2, "ensemble_weights_sum_nonpositive"),
        ((float("nan"), 1.0), 2, "ensemble_weights_non_finite"),
        ((float("inf"), 1.0), 2, "ensemble_weights_non_finite"),
    ],
)
def test_normalize_weights_rejects_invalid_inputs(
    raw_weights: tuple[float, ...],
    n_components: int,
    error_fragment: str,
) -> None:
    with pytest.raises(weights_module.EnsembleWeightsError, match=error_fragment):
        weights_module.normalize_weights(raw_weights=raw_weights, n_components=n_components)


def test_optimize_weights_rejects_empty_prediction_frame() -> None:
    with pytest.raises(weights_module.EnsembleWeightsError, match="ensemble_predictions_empty"):
        weights_module.optimize_weights(
            ranked_predictions=pd.DataFrame(),
            era_series=pd.Series(dtype=str),
            target_series=pd.Series(dtype=float),
            metric="corr20v2_sharpe",
            initial_weights=(0.5, 0.5),
        )


def test_optimize_weights_rejects_optimizer_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResult:
        success = False
        message = "no-solution"
        x = np.array([0.5, 0.5], dtype=float)

    class _FakeOptimizeModule:
        @staticmethod
        def minimize(*args: Any, **kwargs: Any) -> _FakeResult:
            _ = (args, kwargs)
            return _FakeResult()

    monkeypatch.setattr(weights_module, "_load_scipy_optimize", lambda: _FakeOptimizeModule())

    ranked = pd.DataFrame({"pred_a": [0.1, 0.9], "pred_b": [0.2, 0.8]})
    eras = pd.Series(["0001", "0001"])
    targets = pd.Series([0.0, 1.0], dtype=float)
    with pytest.raises(weights_module.EnsembleWeightsError, match="ensemble_weight_optimization_failed"):
        weights_module.optimize_weights(
            ranked_predictions=ranked,
            era_series=eras,
            target_series=targets,
            metric="corr20v2_sharpe",
            initial_weights=(0.5, 0.5),
        )


def test_optimize_weights_returns_normalized_optimized_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResult:
        success = True
        message = "ok"
        x = np.array([0.4, 0.2], dtype=float)

    class _FakeOptimizeModule:
        @staticmethod
        def minimize(*args: Any, **kwargs: Any) -> _FakeResult:
            _ = (args, kwargs)
            return _FakeResult()

    monkeypatch.setattr(weights_module, "_load_scipy_optimize", lambda: _FakeOptimizeModule())

    ranked = pd.DataFrame({"pred_a": [0.1, 0.9], "pred_b": [0.2, 0.8]})
    eras = pd.Series(["0001", "0001"])
    targets = pd.Series([0.0, 1.0], dtype=float)
    optimized = weights_module.optimize_weights(
        ranked_predictions=ranked,
        era_series=eras,
        target_series=targets,
        metric="corr20v2_sharpe",
        initial_weights=(0.5, 0.5),
    )
    assert optimized == pytest.approx((2.0 / 3.0, 1.0 / 3.0))


def test_load_scipy_optimize_raises_when_dependency_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals_dict: dict[str, object] | None = None,
        locals_dict: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        _ = (globals_dict, locals_dict, fromlist, level)
        if name == "scipy":
            raise ImportError("missing")
        return real_import(name, globals_dict, locals_dict, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(weights_module.EnsembleWeightsError, match="ensemble_dependency_missing_scipy"):
        weights_module._load_scipy_optimize()
