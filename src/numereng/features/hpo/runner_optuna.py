"""Optuna-backed study runner for HPO service orchestration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from numereng.features.hpo.contracts import HpoDirection, HpoParameterSpec, HpoSampler


class HpoOptunaError(RuntimeError):
    """Raised when Optuna dependency/runtime behavior fails."""


def run_optuna_study(
    *,
    direction: HpoDirection,
    n_trials: int,
    sampler: HpoSampler,
    seed: int | None,
    specs: tuple[HpoParameterSpec, ...],
    objective_callback: Callable[[int, dict[str, Any]], float],
) -> tuple[int | None, float | None]:
    """Run one Optuna study and return best trial index/value."""

    optuna = _load_optuna()
    sampler_obj = _build_sampler(optuna=optuna, sampler=sampler, seed=seed)

    study = optuna.create_study(direction=direction, sampler=sampler_obj)

    def objective(trial: Any) -> float:
        params = _sample_params(optuna_trial=trial, specs=specs)
        return float(objective_callback(int(trial.number), params))

    study.optimize(objective, n_trials=n_trials, catch=(Exception,))

    try:
        best_trial = study.best_trial
        best_value = study.best_value
    except ValueError:
        return None, None
    return int(best_trial.number), float(best_value)


def _load_optuna() -> Any:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise HpoOptunaError("hpo_dependency_missing_optuna") from exc
    return optuna


def _build_sampler(*, optuna: Any, sampler: HpoSampler, seed: int | None) -> Any:
    if sampler == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if sampler == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    raise HpoOptunaError(f"hpo_sampler_invalid:{sampler}")


def _sample_params(*, optuna_trial: Any, specs: tuple[HpoParameterSpec, ...]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for spec in specs:
        if spec.kind == "categorical":
            if not spec.choices:
                raise HpoOptunaError(f"hpo_choices_empty:{spec.path}")
            params[spec.path] = optuna_trial.suggest_categorical(spec.path, list(spec.choices))
            continue

        if spec.low is None or spec.high is None:
            raise HpoOptunaError(f"hpo_bounds_missing:{spec.path}")

        if spec.kind == "int":
            low = int(spec.low)
            high = int(spec.high)
            step = int(spec.step) if isinstance(spec.step, (int, float)) else 1
            params[spec.path] = optuna_trial.suggest_int(spec.path, low, high, step=step, log=spec.log)
            continue

        low_float = float(spec.low)
        high_float = float(spec.high)
        step_float = float(spec.step) if isinstance(spec.step, (int, float)) else None
        params[spec.path] = optuna_trial.suggest_float(
            spec.path,
            low_float,
            high_float,
            step=step_float,
            log=spec.log,
        )

    return params
