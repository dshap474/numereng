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
    if payload is None:
        return None, None
    if not isinstance(payload, dict):
        return None, "invalid value for --search-space: expected JSON object"

    normalized: dict[str, dict[str, object]] = {}
    for key, spec in payload.items():
        if not isinstance(key, str) or not isinstance(spec, dict):
            return None, "invalid value for --search-space: expected {path: spec} mapping"
        normalized[key] = spec
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
    if value is None:
        return None
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
            "--experiment-id",
            "--study-name",
            "--config",
            "--metric",
            "--direction",
            "--n-trials",
            "--sampler",
            "--seed",
            "--search-space",
            "--neutralizer-path",
            "--neutralization-proportion",
            "--neutralization-mode",
            "--neutralizer-cols",
            "--store-root",
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
            study_config = load_hpo_study_config_json(
                Path(values["--study-config"]).expanduser().resolve()
            )
        except HpoConfigLoaderError as exc:
            print(str(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

    study_name = values.get("--study-name")
    if study_name is None:
        resolved_study_name = study_config.get("study_name")
        if isinstance(resolved_study_name, str):
            study_name = resolved_study_name

    config_path = values.get("--config")
    if config_path is None:
        resolved_config_path = study_config.get("config_path")
        if isinstance(resolved_config_path, str):
            config_path = resolved_config_path

    if study_name is None:
        print("missing required argument: --study-name", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    if config_path is None:
        print("missing required argument: --config", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    direction: HpoDirectionValue = "maximize"
    if "--direction" not in values:
        configured_direction = study_config.get("direction")
        if isinstance(configured_direction, str):
            parsed_default_direction, default_direction_err = _parse_direction(configured_direction)
            if parsed_default_direction is None or default_direction_err is not None:
                print(default_direction_err or "invalid value for --direction", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            direction = parsed_default_direction
    if "--direction" in values:
        parsed, err = _parse_direction(values["--direction"])
        if parsed is None or err is not None:
            print(err or "invalid value for --direction", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        direction = parsed

    sampler: HpoSamplerValue = "tpe"
    if "--sampler" not in values:
        configured_sampler = study_config.get("sampler")
        if isinstance(configured_sampler, str):
            parsed_default_sampler, default_sampler_err = _parse_sampler(configured_sampler)
            if parsed_default_sampler is None or default_sampler_err is not None:
                print(default_sampler_err or "invalid value for --sampler", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            sampler = parsed_default_sampler
    if "--sampler" in values:
        parsed_sampler, sampler_err = _parse_sampler(values["--sampler"])
        if parsed_sampler is None or sampler_err is not None:
            print(sampler_err or "invalid value for --sampler", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        sampler = parsed_sampler

    n_trials = 100
    configured_n_trials = study_config.get("n_trials")
    if isinstance(configured_n_trials, int) and not isinstance(configured_n_trials, bool):
        n_trials = configured_n_trials
    if "--n-trials" in values:
        parsed_n_trials, n_trials_err = _parse_int_value(values["--n-trials"], flag="--n-trials")
        if parsed_n_trials is None or n_trials_err is not None:
            print(n_trials_err or "invalid integer for --n-trials", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        n_trials = parsed_n_trials

    seed: int | None = 1337
    if "seed" in study_config:
        configured_seed = study_config.get("seed")
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

    search_space: dict[str, dict[str, object]] | None = _as_search_space(study_config.get("search_space"))
    if "--search-space" in values:
        parsed_space, space_err = _parse_search_space(values["--search-space"])
        if space_err is not None:
            print(space_err, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        search_space = parsed_space

    neutralization_proportion = 0.5
    configured_neutralization = _as_mapping(study_config.get("neutralization"))
    configured_proportion = configured_neutralization.get("proportion")
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
    configured_mode = configured_neutralization.get("mode")
    if isinstance(configured_mode, str):
        parsed_default_mode, default_mode_err = _parse_neutralization_mode(configured_mode)
        if parsed_default_mode is None or default_mode_err is not None:
            print(default_mode_err or "invalid value for --neutralization-mode", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        neutralization_mode = parsed_default_mode
    if "--neutralization-mode" in values:
        parsed_mode, mode_err = _parse_neutralization_mode(values["--neutralization-mode"])
        if parsed_mode is None or mode_err is not None:
            print(mode_err or "invalid value for --neutralization-mode", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        neutralization_mode = parsed_mode

    neutralizer_cols: list[str] | None = None
    configured_cols = configured_neutralization.get("neutralizer_cols")
    if isinstance(configured_cols, list) and all(isinstance(item, str) for item in configured_cols):
        neutralizer_cols = [item.strip() for item in configured_cols]
    if "--neutralizer-cols" in values:
        parsed_cols, cols_err = _parse_neutralizer_cols(values["--neutralizer-cols"])
        if parsed_cols is None or cols_err is not None:
            print(cols_err or "invalid value for --neutralizer-cols", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        neutralizer_cols = parsed_cols

    neutralize = False
    if isinstance(configured_neutralization.get("enabled"), bool):
        neutralize = bool(configured_neutralization["enabled"])
    if "--neutralize" in toggles:
        neutralize = True

    neutralizer_path = values.get("--neutralizer-path")
    if neutralizer_path is None:
        configured_path = configured_neutralization.get("neutralizer_path")
        if isinstance(configured_path, str):
            neutralizer_path = configured_path

    neutralization_rank_output = True
    if isinstance(configured_neutralization.get("rank_output"), bool):
        neutralization_rank_output = bool(configured_neutralization["rank_output"])
    if "--no-neutralization-rank" in toggles:
        neutralization_rank_output = False

    experiment_id = values.get("--experiment-id")
    if experiment_id is None:
        configured_experiment_id = study_config.get("experiment_id")
        if isinstance(configured_experiment_id, str):
            experiment_id = configured_experiment_id

    metric = values.get("--metric")
    if metric is None:
        configured_metric = study_config.get("metric")
        if isinstance(configured_metric, str):
            metric = configured_metric
        else:
            metric = "post_fold_champion_objective"

    try:
        create_payload = api.hpo_create(
            api.HpoStudyCreateRequest(
                study_name=study_name,
                config_path=config_path,
                experiment_id=experiment_id,
                metric=metric,
                direction=direction,
                n_trials=n_trials,
                sampler=sampler,
                seed=seed,
                search_space=search_space,
                neutralize=neutralize,
                neutralizer_path=neutralizer_path,
                neutralization_proportion=neutralization_proportion,
                neutralization_mode=neutralization_mode,
                neutralizer_cols=neutralizer_cols,
                neutralization_rank_output=neutralization_rank_output,
                store_root=values.get("--store-root", ".numereng"),
            )
        )
    except ValidationError as exc:
        print(_validation_error_message(exc), file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    except PackageError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(create_payload.model_dump_json())
    return 0


__all__ = ["handle_hpo_create"]
