"""Shared parsing and validation helpers for CLI commands."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

from pydantic import ValidationError

from numereng import api

_TOURNAMENT_MAP: dict[str, api.NumeraiTournament] = {
    "classic": "classic",
    "numerai": "classic",
    "signals": "signals",
    "crypto": "crypto",
    "8": "classic",
    "11": "signals",
    "12": "crypto",
}

_TOURNAMENT_USAGE = "classic|signals|crypto"
_TRAINING_PROFILE_MAP: dict[str, api.TrainingProfile] = {
    "simple": "simple",
    "purged_walk_forward": "purged_walk_forward",
    "full_history_refit": "full_history_refit",
}
_TRAINING_PROFILES = set(_TRAINING_PROFILE_MAP)
_TRAINING_PROFILE_USAGE = "simple|purged_walk_forward|full_history_refit"
_CLOUD_AWS_BACKENDS = {"sagemaker", "batch"}
_CLOUD_AWS_BACKENDS_USAGE = "sagemaker|batch"
_EXPERIMENT_STATUSES = {"draft", "active", "complete", "archived"}
CloudAwsBackend = Literal["sagemaker", "batch"]


def _parse_fail_flag(argv: Sequence[str]) -> tuple[bool, list[str]]:
    fail = False
    unknown: list[str] = []
    for arg in argv:
        if arg == "--fail":
            fail = True
        elif arg in {"-h", "--help"}:
            continue
        else:
            unknown.append(arg)
    return fail, unknown


def _parse_int_value(value: str, *, flag: str) -> tuple[int | None, str | None]:
    try:
        return int(value), None
    except ValueError:
        return None, f"invalid integer for {flag}: {value}"


def _parse_simple_options(
    argv: Sequence[str],
    *,
    value_flags: set[str],
    bool_flags: set[str] | None = None,
) -> tuple[dict[str, str], set[str], str | None]:
    values: dict[str, str] = {}
    toggles: set[str] = set()
    enabled_bools = bool_flags or set()

    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg in {"-h", "--help"}:
            return values, toggles, "__help__"
        if arg in enabled_bools:
            toggles.add(arg)
            idx += 1
            continue
        if arg in value_flags:
            if idx + 1 >= len(argv):
                return values, toggles, f"missing value for {arg}"
            values[arg] = argv[idx + 1]
            idx += 2
            continue
        return values, toggles, f"unknown arguments: {arg}"
    return values, toggles, None


def _parse_tournament_value(value: str) -> tuple[api.NumeraiTournament | None, str | None]:
    normalized = value.lower()
    parsed = _TOURNAMENT_MAP.get(normalized)
    if parsed is None:
        return None, f"invalid value for --tournament: {value} (expected {_TOURNAMENT_USAGE})"
    return parsed, None


def _parse_training_profile_value(value: str) -> tuple[api.TrainingProfile | None, str | None]:
    if value == "submission":
        return None, "invalid value for --profile: submission (renamed to full_history_refit)"
    if value not in _TRAINING_PROFILES:
        return None, f"invalid value for --profile: {value} (expected {_TRAINING_PROFILE_USAGE})"
    return _TRAINING_PROFILE_MAP[value], None


def _parse_cloud_backend_value(value: str) -> tuple[CloudAwsBackend | None, str | None]:
    if value not in _CLOUD_AWS_BACKENDS:
        return None, f"invalid value for --backend: {value} (expected {_CLOUD_AWS_BACKENDS_USAGE})"
    return cast(CloudAwsBackend, value), None


def _parse_experiment_status_value(value: str) -> tuple[api.ExperimentStatus | None, str | None]:
    if value not in _EXPERIMENT_STATUSES:
        return None, "invalid value for --status: expected draft|active|complete|archived"
    return cast(api.ExperimentStatus, value), None


def _validation_error_message(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return str(exc)
    return str(errors[0]["msg"])


__all__ = [
    "CloudAwsBackend",
    "_parse_cloud_backend_value",
    "_parse_experiment_status_value",
    "_parse_fail_flag",
    "_parse_int_value",
    "_parse_simple_options",
    "_parse_tournament_value",
    "_parse_training_profile_value",
    "_validation_error_message",
]
