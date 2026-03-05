from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from numereng.features.models.lgbm import LGBMRegressor


class _FakeLGBMError(Exception):
    pass


class _FakeSklearnModel:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def fit(self, *_: object, **__: object) -> _FakeSklearnModel:
        return self

    def predict(self, *_: object, **__: object) -> list[float]:
        return [0.0]


def _install_fake_lightgbm(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module: Any = types.ModuleType("lightgbm")
    fake_module.LGBMRegressor = _FakeSklearnModel
    fake_module.basic = types.SimpleNamespace(LightGBMError=_FakeLGBMError)
    monkeypatch.setitem(sys.modules, "lightgbm", fake_module)


def test_lgbm_wrapper_sets_quiet_default_verbosity(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_lightgbm(monkeypatch)

    model = LGBMRegressor()

    assert model._params["verbosity"] == -1
    assert model._model.kwargs["verbosity"] == -1


def test_lgbm_wrapper_keeps_explicit_verbosity(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_lightgbm(monkeypatch)

    model = LGBMRegressor(verbosity=2)

    assert model._params["verbosity"] == 2
    assert model._model.kwargs["verbosity"] == 2


def test_lgbm_wrapper_keeps_explicit_verbose_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_lightgbm(monkeypatch)

    model = LGBMRegressor(verbose=1)

    assert "verbosity" not in model._params
    assert model._params["verbose"] == 1
    assert "verbosity" not in model._model.kwargs
    assert model._model.kwargs["verbose"] == 1
