from __future__ import annotations

import builtins
import time
from pathlib import Path
from typing import Any, cast

import pytest

import numereng.features.hpo.runner_optuna as runner_module
from numereng.features.hpo import HpoParameterSpec, HpoPlateauSpec, HpoSamplerSpec, HpoStoppingSpec


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
    class _BadSampler:
        kind = "unsupported"

    with pytest.raises(runner_module.HpoOptunaError, match="hpo_sampler_invalid"):
        runner_module._build_sampler(
            optuna=object(),
            sampler=cast(Any, _BadSampler()),
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


def test_run_optuna_study_returns_counts_when_study_has_no_best(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _FakeTrialState:
        COMPLETE = "COMPLETE"
        FAIL = "FAIL"

    class _FrozenTrial:
        def __init__(self, state: str, value: float | None) -> None:
            self.state = state
            self.value = value

    class _FakeStudy:
        def __init__(self) -> None:
            self.trials = [_FrozenTrial(_FakeTrialState.FAIL, None)]

        def optimize(
            self,
            objective: Any,
            *,
            n_trials: int,
            timeout: int | None,
            callbacks: list[Any] | None,
            catch: tuple[type[Exception], ...],
        ) -> None:
            _ = (objective, n_trials, timeout, callbacks, catch)

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
            def __init__(
                self,
                *,
                seed: int | None = None,
                n_startup_trials: int = 10,
                multivariate: bool = False,
                group: bool = False,
            ) -> None:
                _ = (seed, n_startup_trials, multivariate, group)

    class _FakeStudyNamespace:
        @staticmethod
        def MaxTrialsCallback(n_trials: int, *, states: tuple[str, ...]) -> object:
            _ = (n_trials, states)
            return object()

    class _FakeStorages:
        class JournalFileBackend:
            def __init__(self, path: str) -> None:
                self.path = path

        class JournalStorage:
            def __init__(self, backend: object) -> None:
                self.backend = backend

    class _FakeTrialNamespace:
        TrialState = _FakeTrialState

    class _FakeOptuna:
        samplers = _FakeSamplers
        study = _FakeStudyNamespace
        storages = _FakeStorages
        trial = _FakeTrialNamespace

        @staticmethod
        def create_study(
            *,
            storage: object,
            sampler: object,
            study_name: str,
            direction: str,
            load_if_exists: bool,
        ) -> _FakeStudy:
            _ = (storage, sampler, study_name, direction, load_if_exists)
            return _FakeStudy()

    monkeypatch.setattr(runner_module, "_load_optuna", lambda: _FakeOptuna())
    result = runner_module.run_optuna_study(
        study_id="study-a",
        storage_path=tmp_path / "study-a",
        direction="maximize",
        sampler=HpoSamplerSpec(kind="tpe"),
        stopping=HpoStoppingSpec(max_trials=2),
        specs=(),
        objective_callback=lambda _trial_number, _params: 0.0,
    )

    assert result.best_trial_number is None
    assert result.best_value is None
    assert result.attempted_trials == 1
    assert result.completed_trials == 0
    assert result.failed_trials == 1
    assert result.stop_reason == "all_trials_failed"


def test_plateau_callback_stops_after_completed_patience() -> None:
    class _FakeTrialState:
        COMPLETE = "COMPLETE"

    class _FrozenTrial:
        def __init__(self, value: float) -> None:
            self.state = _FakeTrialState.COMPLETE
            self.value = value

    class _FakeStudy:
        def __init__(self, values: list[float]) -> None:
            self.trials = [_FrozenTrial(value) for value in values]
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

    class _FakeTrialNamespace:
        TrialState = _FakeTrialState

    class _FakeOptuna:
        trial = _FakeTrialNamespace

    callback = runner_module._build_plateau_callback(
        optuna=_FakeOptuna(),
        direction="maximize",
        plateau=HpoPlateauSpec(
            enabled=True,
            min_completed_trials=3,
            patience_completed_trials=2,
            min_improvement_abs=0.01,
        ),
        stop_flags={"plateau_reached": False},
    )
    study = _FakeStudy([0.10, 0.12, 0.121, 0.1215, 0.1214])
    callback(study, study.trials[-1])

    assert study.stopped is True


def test_run_optuna_study_resumes_against_total_max_trials(tmp_path: Path) -> None:
    pytest.importorskip("optuna")
    storage_path = tmp_path / "study-total-budget"

    first = runner_module.run_optuna_study(
        study_id="study-total-budget",
        storage_path=storage_path,
        direction="maximize",
        sampler=HpoSamplerSpec(kind="tpe", seed=1337),
        stopping=HpoStoppingSpec(max_trials=1),
        specs=(),
        objective_callback=lambda trial_number, _params: float(trial_number + 1),
    )
    second = runner_module.run_optuna_study(
        study_id="study-total-budget",
        storage_path=storage_path,
        direction="maximize",
        sampler=HpoSamplerSpec(kind="tpe", seed=1337),
        stopping=HpoStoppingSpec(max_trials=2),
        specs=(),
        objective_callback=lambda trial_number, _params: float(trial_number + 1),
    )

    assert first.attempted_trials == 1
    assert second.attempted_trials == 2
    assert second.completed_trials == 2
    assert second.stop_reason == "max_trials_reached"


def test_run_optuna_study_resumes_against_total_completed_budget(tmp_path: Path) -> None:
    pytest.importorskip("optuna")
    storage_path = tmp_path / "study-completed-budget"

    first = runner_module.run_optuna_study(
        study_id="study-completed-budget",
        storage_path=storage_path,
        direction="maximize",
        sampler=HpoSamplerSpec(kind="tpe", seed=1337),
        stopping=HpoStoppingSpec(max_trials=2),
        specs=(),
        objective_callback=lambda trial_number, _params: (
            (_ for _ in ()).throw(RuntimeError("boom")) if trial_number == 0 else 1.0
        ),
    )
    second = runner_module.run_optuna_study(
        study_id="study-completed-budget",
        storage_path=storage_path,
        direction="maximize",
        sampler=HpoSamplerSpec(kind="tpe", seed=1337),
        stopping=HpoStoppingSpec(max_trials=4, max_completed_trials=2),
        specs=(),
        objective_callback=lambda trial_number, _params: float(trial_number + 1),
    )

    assert first.attempted_trials == 2
    assert first.completed_trials == 1
    assert first.failed_trials == 1
    assert second.attempted_trials == 3
    assert second.completed_trials == 2
    assert second.failed_trials == 1
    assert second.stop_reason == "max_completed_trials_reached"


def test_run_optuna_study_reports_timeout_stop_reason(tmp_path: Path) -> None:
    pytest.importorskip("optuna")
    storage_path = tmp_path / "study-timeout"

    result = runner_module.run_optuna_study(
        study_id="study-timeout",
        storage_path=storage_path,
        direction="maximize",
        sampler=HpoSamplerSpec(kind="random", seed=17),
        stopping=HpoStoppingSpec(max_trials=100, timeout_seconds=1),
        specs=(),
        objective_callback=lambda _trial_number, _params: (time.sleep(0.2), 1.0)[1],
    )

    assert result.attempted_trials > 0
    assert result.attempted_trials < 100
    assert result.completed_trials == result.attempted_trials
    assert result.failed_trials == 0
    assert result.stop_reason == "timeout_reached"


def test_run_optuna_study_reports_plateau_stop_reason(tmp_path: Path) -> None:
    pytest.importorskip("optuna")
    storage_path = tmp_path / "study-plateau"
    values = [0.10, 0.12, 0.121, 0.1215, 0.1214]

    result = runner_module.run_optuna_study(
        study_id="study-plateau",
        storage_path=storage_path,
        direction="maximize",
        sampler=HpoSamplerSpec(kind="random", seed=3),
        stopping=HpoStoppingSpec(
            max_trials=20,
            plateau=HpoPlateauSpec(
                enabled=True,
                min_completed_trials=3,
                patience_completed_trials=2,
                min_improvement_abs=0.01,
            ),
        ),
        specs=(),
        objective_callback=lambda trial_number, _params: values[trial_number],
    )

    assert result.attempted_trials == 4
    assert result.completed_trials == 4
    assert result.failed_trials == 0
    assert result.stop_reason == "plateau_reached"
