"""Optuna-backed study runner for HPO service orchestration."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from numereng.features.hpo.artifacts import optuna_journal_path
from numereng.features.hpo.contracts import (
    HpoDirection,
    HpoParameterSpec,
    HpoPlateauSpec,
    HpoSamplerSpec,
    HpoStoppingSpec,
    HpoStopReason,
)


class HpoOptunaError(RuntimeError):
    """Raised when Optuna dependency/runtime behavior fails."""


@dataclass(frozen=True)
class HpoOptunaStudyResult:
    """Summary returned by one Optuna study execution."""

    best_trial_number: int | None
    best_value: float | None
    attempted_trials: int
    completed_trials: int
    failed_trials: int
    stop_reason: HpoStopReason | None


def run_optuna_study(
    *,
    study_id: str,
    storage_path: Path,
    direction: HpoDirection,
    sampler: HpoSamplerSpec,
    stopping: HpoStoppingSpec,
    specs: tuple[HpoParameterSpec, ...],
    objective_callback: Callable[[int, dict[str, Any]], float],
    summary_callback: Callable[[], None] | None = None,
) -> HpoOptunaStudyResult:
    """Run or resume one Optuna study and return the best trial/value plus counts."""

    optuna = _load_optuna()
    sampler_obj = _build_sampler(optuna=optuna, sampler=sampler)
    storage = _build_storage(optuna=optuna, storage_path=storage_path)
    study = optuna.create_study(
        storage=storage,
        sampler=sampler_obj,
        study_name=study_id,
        direction=direction,
        load_if_exists=True,
    )
    existing_attempted_trials, existing_completed_trials, _ = _trial_counts(optuna=optuna, trials=study.trials)
    remaining_attempts = max(0, stopping.max_trials - existing_attempted_trials)
    remaining_completed = (
        None
        if stopping.max_completed_trials is None
        else max(0, stopping.max_completed_trials - existing_completed_trials)
    )

    stop_flags = {"plateau_reached": False}
    callbacks: list[Callable[[Any, Any], None]] = []
    if remaining_completed is not None and remaining_completed > 0:
        callbacks.append(
            optuna.study.MaxTrialsCallback(
                remaining_completed,
                states=(optuna.trial.TrialState.COMPLETE,),
            )
        )
    if stopping.plateau.enabled:
        callbacks.append(
            _build_plateau_callback(
                optuna=optuna,
                direction=direction,
                plateau=stopping.plateau,
                stop_flags=stop_flags,
            )
        )
    if summary_callback is not None:
        callbacks.append(lambda _study, _trial: summary_callback())

    def objective(trial: Any) -> float:
        params = _sample_params(optuna_trial=trial, specs=specs)
        return float(objective_callback(int(trial.number), params))

    elapsed_seconds = 0.0
    if remaining_attempts > 0 and (remaining_completed is None or remaining_completed > 0):
        started_at = monotonic()
        study.optimize(
            objective,
            n_trials=remaining_attempts,
            timeout=stopping.timeout_seconds,
            callbacks=callbacks or None,
            catch=(Exception,),
        )
        elapsed_seconds = monotonic() - started_at

    attempted_trials, completed_trials, failed_trials = _trial_counts(optuna=optuna, trials=study.trials)

    try:
        best_trial = study.best_trial
        best_value = study.best_value
    except ValueError:
        best_trial_number = None
        resolved_best_value = None
    else:
        best_trial_number = int(best_trial.number)
        resolved_best_value = float(best_value)

    return HpoOptunaStudyResult(
        best_trial_number=best_trial_number,
        best_value=resolved_best_value,
        attempted_trials=attempted_trials,
        completed_trials=completed_trials,
        failed_trials=failed_trials,
        stop_reason=_resolve_stop_reason(
            attempted_trials=attempted_trials,
            completed_trials=completed_trials,
            failed_trials=failed_trials,
            max_trials=stopping.max_trials,
            max_completed_trials=stopping.max_completed_trials,
            timeout_seconds=stopping.timeout_seconds,
            elapsed_seconds=elapsed_seconds,
            plateau_reached=bool(stop_flags["plateau_reached"]),
        ),
    )


def _load_optuna() -> Any:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise HpoOptunaError("hpo_dependency_missing_optuna") from exc
    return optuna


def _build_storage(*, optuna: Any, storage_path: Path) -> Any:
    journal_path = optuna_journal_path(storage_path=storage_path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    backend_factory = getattr(optuna.storages, "JournalFileBackend", None)
    if backend_factory is None:
        journal_module = getattr(optuna.storages, "journal", None)
        backend_factory = getattr(journal_module, "JournalFileBackend", None)
    if backend_factory is None:
        backend_factory = getattr(optuna.storages, "JournalFileStorage", None)
    if backend_factory is None:  # pragma: no cover - dependency gate
        raise HpoOptunaError("hpo_optuna_journal_backend_missing")
    return optuna.storages.JournalStorage(backend_factory(str(journal_path)))


def _build_sampler(*, optuna: Any, sampler: HpoSamplerSpec) -> Any:
    if sampler.kind == "random":
        return optuna.samplers.RandomSampler(seed=sampler.seed)
    if sampler.kind == "tpe":
        experimental_warning = getattr(getattr(optuna, "exceptions", None), "ExperimentalWarning", Warning)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=experimental_warning)
            return optuna.samplers.TPESampler(
                seed=sampler.seed,
                n_startup_trials=sampler.n_startup_trials,
                multivariate=sampler.multivariate,
                group=sampler.group,
            )
    raise HpoOptunaError(f"hpo_sampler_invalid:{sampler.kind}")


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


def _trial_counts(*, optuna: Any, trials: list[Any]) -> tuple[int, int, int]:
    attempted_trials = len(trials)
    completed_trials = sum(1 for item in trials if item.state == optuna.trial.TrialState.COMPLETE)
    failed_trials = sum(1 for item in trials if item.state == optuna.trial.TrialState.FAIL)
    return attempted_trials, completed_trials, failed_trials


def _build_plateau_callback(
    *,
    optuna: Any,
    direction: HpoDirection,
    plateau: HpoPlateauSpec,
    stop_flags: dict[str, bool],
) -> Callable[[Any, Any], None]:
    def callback(study: Any, trial: Any) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        completed_values = [
            float(item.value) for item in study.trials if item.state == optuna.trial.TrialState.COMPLETE
        ]
        if len(completed_values) < plateau.min_completed_trials:
            return
        last_improvement_index = _last_improvement_index(
            completed_values=completed_values,
            direction=direction,
            min_improvement_abs=plateau.min_improvement_abs,
        )
        if len(completed_values) - last_improvement_index >= plateau.patience_completed_trials:
            stop_flags["plateau_reached"] = True
            study.stop()

    return callback


def _last_improvement_index(
    *,
    completed_values: list[float],
    direction: HpoDirection,
    min_improvement_abs: float,
) -> int:
    best_value: float | None = None
    last_improvement_index = 0
    for index, value in enumerate(completed_values, start=1):
        if best_value is None:
            best_value = value
            last_improvement_index = index
            continue
        if _is_meaningful_improvement(
            candidate=value,
            incumbent=best_value,
            direction=direction,
            min_improvement_abs=min_improvement_abs,
        ):
            best_value = value
            last_improvement_index = index
    return last_improvement_index


def _is_meaningful_improvement(
    *,
    candidate: float,
    incumbent: float,
    direction: HpoDirection,
    min_improvement_abs: float,
) -> bool:
    if direction == "minimize":
        return candidate < (incumbent - min_improvement_abs)
    return candidate > (incumbent + min_improvement_abs)


def _resolve_stop_reason(
    *,
    attempted_trials: int,
    completed_trials: int,
    failed_trials: int,
    max_trials: int,
    max_completed_trials: int | None,
    timeout_seconds: int | None,
    elapsed_seconds: float,
    plateau_reached: bool,
) -> HpoStopReason | None:
    if attempted_trials > 0 and completed_trials == 0 and failed_trials == attempted_trials:
        return "all_trials_failed"
    if plateau_reached:
        return "plateau_reached"
    if max_completed_trials is not None and completed_trials >= max_completed_trials:
        return "max_completed_trials_reached"
    if attempted_trials >= max_trials:
        return "max_trials_reached"
    if timeout_seconds is not None and elapsed_seconds >= float(timeout_seconds):
        return "timeout_reached"
    return None
