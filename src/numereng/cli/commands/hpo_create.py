"""HPO create command helpers."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from pydantic import ValidationError

from numereng import api
from numereng.cli.common import _parse_int_value, _parse_simple_options, _validation_error_message
from numereng.cli.usage import USAGE
from numereng.config.hpo import HpoConfigLoaderError, load_hpo_study_config_json
from numereng.platform.errors import PackageError

HpoDirectionValue = Literal["maximize", "minimize"]
HpoSamplerValue = Literal["tpe", "random"]
NeutralizationModeValue = Literal["era", "global"]


def _parse_direction(value: str) -> tuple[HpoDirectionValue | None, str | None]:
    if value not in {"maximize", "minimize"}:
        return None, "invalid value for --direction: expected maximize|minimize"
    return cast(HpoDirectionValue, value), None


def _parse_sampler(value: str) -> tuple[HpoSamplerValue | None, str | None]:
    if value not in {"tpe", "random"}:
        return None, "invalid value for --sampler: expected tpe|random"
    return cast(HpoSamplerValue, value), None


def _parse_search_space(value: str) -> tuple[dict[str, dict[str, object]] | None, str | None]:
    candidate = value.strip()
    if not candidate:
        return None, "invalid value for --search-space"

    path = Path(candidate).expanduser()
    raw = candidate
    if path.exists() and path.is_file():
        raw = path.read_text(encoding="utf-8")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None, "invalid JSON for --search-space"
    if not isinstance(payload, dict):
        return None, "invalid value for --search-space: expected JSON object"

    normalized: dict[str, dict[str, object]] = {}
    for key, spec in payload.items():
        if not isinstance(key, str) or not isinstance(spec, dict):
            return None, "invalid value for --search-space: expected {path: spec} mapping"
        normalized[key] = {str(spec_key): spec_value for spec_key, spec_value in spec.items()}
    return normalized, None


def _parse_neutralization_mode(value: str) -> tuple[NeutralizationModeValue | None, str | None]:
    if value not in {"era", "global"}:
        return None, "invalid value for --neutralization-mode: expected era|global"
    return cast(NeutralizationModeValue, value), None


def _parse_neutralization_proportion(value: str) -> tuple[float | None, str | None]:
    try:
        parsed = float(value)
    except ValueError:
        return None, f"invalid float for --neutralization-proportion: {value}"
    if parsed < 0.0 or parsed > 1.0:
        return None, "invalid value for --neutralization-proportion: expected 0.0..1.0"
    return parsed, None


def _parse_neutralizer_cols(value: str) -> tuple[list[str] | None, str | None]:
    cols = [item.strip() for item in value.split(",") if item.strip()]
    if not cols:
        return None, "invalid value for --neutralizer-cols: expected comma-separated column names"
    return cols, None


def _as_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _as_search_space(value: object) -> dict[str, dict[str, object]] | None:
    if not isinstance(value, dict):
        return None
    parsed: dict[str, dict[str, object]] = {}
    for key, spec in value.items():
        if not isinstance(key, str) or not isinstance(spec, dict):
            return None
        parsed[key] = {str(spec_key): spec_value for spec_key, spec_value in spec.items()}
    return parsed


def handle_hpo_create(args: Sequence[str]) -> int:
    """Handle `numereng hpo create ...`."""

    values, toggles, parse_error = _parse_simple_options(
        args,
        value_flags={
            "--study-config",
            "--study-id",
            "--study-name",
            "--config",
            "--experiment-id",
            "--metric",
            "--direction",
            "--n-trials",
            "--timeout-seconds",
            "--max-completed-trials",
            "--sampler",
            "--seed",
            "--search-space",
            "--neutralizer-path",
            "--neutralization-proportion",
            "--neutralization-mode",
            "--neutralizer-cols",
            "--workspace",
        },
        bool_flags={"--neutralize", "--no-neutralization-rank"},
    )
    if parse_error == "__help__":
        print(USAGE)
        return 0
    if parse_error is not None:
        print(parse_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    study_config: dict[str, object] = {}
    if "--study-config" in values:
        try:
            study_config = load_hpo_study_config_json(Path(values["--study-config"]).expanduser().resolve())
        except HpoConfigLoaderError as exc:
            print(str(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

    objective_config = _as_mapping(study_config.get("objective"))
    sampler_config = _as_mapping(study_config.get("sampler"))
    stopping_config = _as_mapping(study_config.get("stopping"))
    plateau_config = _as_mapping(stopping_config.get("plateau"))
    neutralization_config = _as_mapping(objective_config.get("neutralization"))

    study_id = values.get("--study-id") or study_config.get("study_id")
    study_name = values.get("--study-name") or study_config.get("study_name")
    config_path = values.get("--config") or study_config.get("config_path")

    if not isinstance(study_id, str):
        print("missing required argument: --study-id", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    if not isinstance(study_name, str):
        print("missing required argument: --study-name", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    if not isinstance(config_path, str):
        print("missing required argument: --config", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    direction: HpoDirectionValue = "maximize"
    configured_direction = objective_config.get("direction")
    if isinstance(configured_direction, str):
        parsed_direction, direction_err = _parse_direction(configured_direction)
        if parsed_direction is None or direction_err is not None:
            print(direction_err or "invalid value for --direction", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        direction = parsed_direction
    if "--direction" in values:
        parsed_direction, direction_err = _parse_direction(values["--direction"])
        if parsed_direction is None or direction_err is not None:
            print(direction_err or "invalid value for --direction", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        direction = parsed_direction

    sampler: HpoSamplerValue = "tpe"
    configured_sampler = sampler_config.get("kind")
    if isinstance(configured_sampler, str):
        parsed_sampler, sampler_err = _parse_sampler(configured_sampler)
        if parsed_sampler is None or sampler_err is not None:
            print(sampler_err or "invalid value for --sampler", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        sampler = parsed_sampler
    if "--sampler" in values:
        parsed_sampler, sampler_err = _parse_sampler(values["--sampler"])
        if parsed_sampler is None or sampler_err is not None:
            print(sampler_err or "invalid value for --sampler", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        sampler = parsed_sampler

    metric = objective_config.get("metric")
    if not isinstance(metric, str):
        metric = "bmc_last_200_eras.mean"
    if "--metric" in values:
        metric = values["--metric"]

    max_trials = 100
    configured_max_trials = stopping_config.get("max_trials")
    if isinstance(configured_max_trials, int) and not isinstance(configured_max_trials, bool):
        max_trials = configured_max_trials
    if "--n-trials" in values:
        parsed_n_trials, n_trials_err = _parse_int_value(values["--n-trials"], flag="--n-trials")
        if parsed_n_trials is None or n_trials_err is not None:
            print(n_trials_err or "invalid integer for --n-trials", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        max_trials = parsed_n_trials

    timeout_seconds: int | None = None
    configured_timeout = stopping_config.get("timeout_seconds")
    if isinstance(configured_timeout, int) and not isinstance(configured_timeout, bool):
        timeout_seconds = configured_timeout
    elif configured_timeout is None:
        timeout_seconds = None
    if "--timeout-seconds" in values:
        parsed_timeout, timeout_err = _parse_int_value(values["--timeout-seconds"], flag="--timeout-seconds")
        if parsed_timeout is None or timeout_err is not None:
            print(timeout_err or "invalid integer for --timeout-seconds", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        timeout_seconds = parsed_timeout

    max_completed_trials: int | None = None
    configured_max_completed = stopping_config.get("max_completed_trials")
    if isinstance(configured_max_completed, int) and not isinstance(configured_max_completed, bool):
        max_completed_trials = configured_max_completed
    elif configured_max_completed is None:
        max_completed_trials = None
    if "--max-completed-trials" in values:
        parsed_max_completed, max_completed_err = _parse_int_value(
            values["--max-completed-trials"],
            flag="--max-completed-trials",
        )
        if parsed_max_completed is None or max_completed_err is not None:
            print(max_completed_err or "invalid integer for --max-completed-trials", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        max_completed_trials = parsed_max_completed

    seed: int | None = 1337
    if "seed" in sampler_config:
        configured_seed = sampler_config.get("seed")
        if configured_seed is None:
            seed = None
        elif isinstance(configured_seed, int) and not isinstance(configured_seed, bool):
            seed = configured_seed
    if "--seed" in values:
        parsed_seed, seed_err = _parse_int_value(values["--seed"], flag="--seed")
        if parsed_seed is None or seed_err is not None:
            print(seed_err or "invalid integer for --seed", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        seed = parsed_seed

    search_space = _as_search_space(study_config.get("search_space"))
    if "--search-space" in values:
        parsed_space, space_err = _parse_search_space(values["--search-space"])
        if space_err is not None:
            print(space_err, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        search_space = parsed_space
    if search_space is None:
        print("missing required argument: --search-space or --study-config", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    neutralization_enabled = bool(neutralization_config.get("enabled", False))
    if "--neutralize" in toggles:
        neutralization_enabled = True

    neutralizer_path = neutralization_config.get("neutralizer_path")
    if not isinstance(neutralizer_path, str):
        neutralizer_path = None
    if "--neutralizer-path" in values:
        neutralizer_path = values["--neutralizer-path"]

    neutralization_proportion = 0.5
    configured_proportion = neutralization_config.get("proportion")
    if isinstance(configured_proportion, (int, float)) and not isinstance(configured_proportion, bool):
        neutralization_proportion = float(configured_proportion)
    if "--neutralization-proportion" in values:
        parsed_proportion, proportion_err = _parse_neutralization_proportion(values["--neutralization-proportion"])
        if parsed_proportion is None or proportion_err is not None:
            print(proportion_err or "invalid value for --neutralization-proportion", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        neutralization_proportion = parsed_proportion

    neutralization_mode: NeutralizationModeValue = "era"
    configured_mode = neutralization_config.get("mode")
    if isinstance(configured_mode, str):
        parsed_mode, mode_err = _parse_neutralization_mode(configured_mode)
        if parsed_mode is None or mode_err is not None:
            print(mode_err or "invalid value for --neutralization-mode", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        neutralization_mode = parsed_mode
    if "--neutralization-mode" in values:
        parsed_mode, mode_err = _parse_neutralization_mode(values["--neutralization-mode"])
        if parsed_mode is None or mode_err is not None:
            print(mode_err or "invalid value for --neutralization-mode", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        neutralization_mode = parsed_mode

    neutralizer_cols = neutralization_config.get("neutralizer_cols")
    if not isinstance(neutralizer_cols, list):
        neutralizer_cols = None
    if "--neutralizer-cols" in values:
        parsed_cols, cols_err = _parse_neutralizer_cols(values["--neutralizer-cols"])
        if parsed_cols is None or cols_err is not None:
            print(cols_err or "invalid value for --neutralizer-cols", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        neutralizer_cols = parsed_cols

    rank_output = bool(neutralization_config.get("rank_output", True))
    if "--no-neutralization-rank" in toggles:
        rank_output = False

    try:
        sampler_payload: dict[str, object] = {
            "kind": sampler,
            "seed": seed,
        }
        if sampler == "tpe":
            sampler_payload["n_startup_trials"] = (
                int(sampler_config.get("n_startup_trials", 10))
                if isinstance(sampler_config.get("n_startup_trials", 10), int)
                else 10
            )
            sampler_payload["multivariate"] = bool(sampler_config.get("multivariate", True))
            sampler_payload["group"] = bool(sampler_config.get("group", False))
        request = api.HpoStudyCreateRequest(
            study_id=study_id,
            study_name=study_name,
            config_path=config_path,
            experiment_id=values.get("--experiment-id") or study_config.get("experiment_id"),
            objective=api.HpoObjectiveRequest(
                metric=metric,
                direction=direction,
                neutralization=api.HpoNeutralizationRequest(
                    enabled=neutralization_enabled,
                    neutralizer_path=neutralizer_path,
                    proportion=neutralization_proportion,
                    mode=neutralization_mode,
                    neutralizer_cols=neutralizer_cols,
                    rank_output=rank_output,
                ),
            ),
            search_space={
                path: api.HpoSearchSpaceSpecRequest.model_validate(spec) for path, spec in search_space.items()
            },
            sampler=api.HpoSamplerRequest(**sampler_payload),
            stopping=api.HpoStoppingRequest(
                max_trials=max_trials,
                max_completed_trials=max_completed_trials,
                timeout_seconds=timeout_seconds,
                plateau=api.HpoPlateauRequest(
                    enabled=bool(plateau_config.get("enabled", False)),
                    min_completed_trials=int(plateau_config.get("min_completed_trials", 15))
                    if isinstance(plateau_config.get("min_completed_trials", 15), int)
                    else 15,
                    patience_completed_trials=int(plateau_config.get("patience_completed_trials", 10))
                    if isinstance(plateau_config.get("patience_completed_trials", 10), int)
                    else 10,
                    min_improvement_abs=float(plateau_config.get("min_improvement_abs", 0.00025))
                    if isinstance(plateau_config.get("min_improvement_abs", 0.00025), (int, float))
                    else 0.00025,
                ),
            ),
            workspace_root=values.get("--workspace", "."),
        )
    except ValidationError as exc:
        print(_validation_error_message(exc), file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    try:
        payload = api.hpo_create(request)
    except PackageError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(payload.model_dump_json())
    return 0
