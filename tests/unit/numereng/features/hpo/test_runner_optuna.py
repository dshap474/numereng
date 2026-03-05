from __future__ import annotations

import builtins
from typing import Any, cast

import pytest

import numereng.features.hpo.runner_optuna as runner_module
from numereng.features.hpo import HpoParameterSpec


def test_load_optuna_raises_when_dependency_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals_dict: dict[str, object] | None = None,
        locals_dict: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        _ = (globals_dict, locals_dict, fromlist, level)
        if name == "optuna":
            raise ImportError("missing")
        return real_import(name, globals_dict, locals_dict, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(runner_module.HpoOptunaError, match="hpo_dependency_missing_optuna"):
        runner_module._load_optuna()


def test_build_sampler_rejects_invalid_sampler() -> None:
    with pytest.raises(runner_module.HpoOptunaError, match="hpo_sampler_invalid"):
        runner_module._build_sampler(
            optuna=object(),
            sampler=cast(Any, "unsupported"),
            seed=1337,
        )


def test_sample_params_rejects_empty_choices() -> None:
    class _FakeTrial:
        def suggest_categorical(self, name: str, choices: list[object]) -> object:
            _ = (name, choices)
            return "unused"

    with pytest.raises(runner_module.HpoOptunaError, match="hpo_choices_empty"):
        runner_module._sample_params(
            optuna_trial=_FakeTrial(),
            specs=(HpoParameterSpec(path="model.params.foo", kind="categorical", choices=()),),
        )


def test_sample_params_rejects_missing_bounds() -> None:
    class _FakeTrial:
        def suggest_float(
            self,
            name: str,
            low: float,
            high: float,
            *,
            step: float | None = None,
            log: bool = False,
        ) -> float:
            _ = (name, low, high, step, log)
            return 0.1

    with pytest.raises(runner_module.HpoOptunaError, match="hpo_bounds_missing"):
        runner_module._sample_params(
            optuna_trial=_FakeTrial(),
            specs=(HpoParameterSpec(path="model.params.lr", kind="float", low=None, high=1.0),),
        )


def test_run_optuna_study_returns_none_when_study_has_no_best(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeStudy:
        def optimize(
            self,
            objective: Any,
            *,
            n_trials: int,
            catch: tuple[type[Exception], ...],
        ) -> None:
            _ = (objective, n_trials, catch)

        @property
        def best_trial(self) -> Any:
            raise ValueError("no complete trials")

        @property
        def best_value(self) -> float:
            return 0.0

    class _FakeSamplers:
        class RandomSampler:
            def __init__(self, seed: int | None = None) -> None:
                _ = seed

        class TPESampler:
            def __init__(self, seed: int | None = None) -> None:
                _ = seed

    class _FakeOptuna:
        samplers = _FakeSamplers

        @staticmethod
        def create_study(*, direction: str, sampler: object) -> _FakeStudy:
            _ = (direction, sampler)
            return _FakeStudy()

    monkeypatch.setattr(runner_module, "_load_optuna", lambda: _FakeOptuna())
    best_trial_number, best_value = runner_module.run_optuna_study(
        direction="maximize",
        n_trials=2,
        sampler="tpe",
        seed=1337,
        specs=(),
        objective_callback=lambda _trial_number, _params: 0.0,
    )

    assert best_trial_number is None
    assert best_value is None
