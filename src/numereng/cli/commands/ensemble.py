"""Ensemble command handlers."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Literal, cast

from pydantic import ValidationError

from numereng import api
from numereng.cli.commands.ensemble_select import handle_ensemble_select
from numereng.cli.common import _parse_int_value, _parse_simple_options, _validation_error_message
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError

OutputFormat = Literal["table", "json"]
EnsembleMethodValue = Literal["rank_avg"]
NeutralizationModeValue = Literal["era", "global"]


def _parse_output_format(value: str) -> tuple[OutputFormat | None, str | None]:
    if value not in {"table", "json"}:
        return None, "invalid value for --format: expected table|json"
    return cast(OutputFormat, value), None


def _parse_method(value: str) -> tuple[EnsembleMethodValue | None, str | None]:
    if value != "rank_avg":
        return None, "invalid value for --method: expected rank_avg"
    return cast(EnsembleMethodValue, value), None


def _parse_run_ids(value: str) -> tuple[list[str] | None, str | None]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if len(items) < 2:
        return None, "invalid value for --run-ids: expected at least two comma-separated run IDs"
    return items, None


def _parse_weights(value: str) -> tuple[list[float] | None, str | None]:
    raw_items = [item.strip() for item in value.split(",") if item.strip()]
    if not raw_items:
        return None, "invalid value for --weights"

    weights: list[float] = []
    for item in raw_items:
        try:
            weights.append(float(item))
        except ValueError:
            return None, "invalid value for --weights: expected comma-separated floats"
    return weights, None


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


def _print_ensembles_table(payload: api.EnsembleListResponse) -> None:
    if not payload.ensembles:
        print("No ensembles found")
        return

    header = f"{'Ensemble ID':<44} {'Status':<10} {'Method':<10} {'Components':<10} {'Metric'}"
    print(header)
    print("-" * len(header))
    for item in payload.ensembles:
        print(f"{item.ensemble_id:<44} {item.status:<10} {item.method:<10} {len(item.components):<10} {item.metric}")


def _print_ensemble_details_table(payload: api.EnsembleResponse) -> None:
    print(f"ensemble_id: {payload.ensemble_id}")
    print(f"name: {payload.name}")
    print(f"experiment_id: {payload.experiment_id or 'none'}")
    print(f"status: {payload.status}")
    print(f"method: {payload.method}")
    print(f"target: {payload.target}")
    print(f"metric: {payload.metric}")
    print(f"artifacts_path: {payload.artifacts_path}")

    print("components:")
    for component in payload.components:
        print(f"  - run_id={component.run_id} weight={component.weight:.6f} rank={component.rank}")

    print("metrics:")
    for metric in payload.metrics:
        print(f"  - {metric.name}={metric.value}")


def handle_ensemble_command(args: Sequence[str]) -> int:
    if not args:
        print(USAGE)
        return 0

    if args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    if args[0] == "select":
        return handle_ensemble_select(args[1:])

    if args[0] == "build":
        values, toggles, parse_error = _parse_simple_options(
            args[1:],
            value_flags={
                "--experiment-id",
                "--run-ids",
                "--method",
                "--metric",
                "--target",
                "--name",
                "--ensemble-id",
                "--weights",
                "--selection-note",
                "--regime-buckets",
                "--neutralizer-path",
                "--neutralization-proportion",
                "--neutralization-mode",
                "--neutralizer-cols",
                "--workspace",
            },
            bool_flags={
                "--optimize-weights",
                "--include-heavy-artifacts",
                "--neutralize-members",
                "--neutralize-final",
                "--no-neutralization-rank",
            },
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

        run_ids_value = values.get("--run-ids")
        if run_ids_value is None:
            print("missing required argument: --run-ids", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        run_ids, run_ids_err = _parse_run_ids(run_ids_value)
        if run_ids is None or run_ids_err is not None:
            print(run_ids_err or "invalid value for --run-ids", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

        method: EnsembleMethodValue = "rank_avg"
        if "--method" in values:
            parsed_method, method_err = _parse_method(values["--method"])
            if parsed_method is None or method_err is not None:
                print(method_err or "invalid value for --method", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            method = parsed_method

        weights: list[float] | None = None
        if "--weights" in values:
            parsed_weights, weights_err = _parse_weights(values["--weights"])
            if parsed_weights is None or weights_err is not None:
                print(weights_err or "invalid value for --weights", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            weights = parsed_weights

        regime_buckets = 4
        if "--regime-buckets" in values:
            parsed_regime_buckets, regime_buckets_err = _parse_int_value(
                values["--regime-buckets"],
                flag="--regime-buckets",
            )
            if parsed_regime_buckets is None or regime_buckets_err is not None:
                print(regime_buckets_err or "invalid integer for --regime-buckets", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            regime_buckets = parsed_regime_buckets

        neutralization_proportion = 0.5
        if "--neutralization-proportion" in values:
            parsed_proportion, proportion_err = _parse_neutralization_proportion(values["--neutralization-proportion"])
            if parsed_proportion is None or proportion_err is not None:
                print(proportion_err or "invalid value for --neutralization-proportion", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            neutralization_proportion = parsed_proportion

        neutralization_mode: NeutralizationModeValue = "era"
        if "--neutralization-mode" in values:
            parsed_mode, mode_err = _parse_neutralization_mode(values["--neutralization-mode"])
            if parsed_mode is None or mode_err is not None:
                print(mode_err or "invalid value for --neutralization-mode", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            neutralization_mode = parsed_mode

        neutralizer_cols: list[str] | None = None
        if "--neutralizer-cols" in values:
            parsed_cols, cols_err = _parse_neutralizer_cols(values["--neutralizer-cols"])
            if parsed_cols is None or cols_err is not None:
                print(cols_err or "invalid value for --neutralizer-cols", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            neutralizer_cols = parsed_cols

        try:
            build_payload = api.ensemble_build(
                api.EnsembleBuildRequest(
                    run_ids=run_ids,
                    experiment_id=values.get("--experiment-id"),
                    method=method,
                    metric=values.get("--metric", "corr_sharpe"),
                    target=values.get("--target", "target_ender_20"),
                    name=values.get("--name"),
                    ensemble_id=values.get("--ensemble-id"),
                    weights=weights,
                    optimize_weights="--optimize-weights" in toggles,
                    include_heavy_artifacts="--include-heavy-artifacts" in toggles,
                    selection_note=values.get("--selection-note"),
                    regime_buckets=regime_buckets,
                    neutralize_members="--neutralize-members" in toggles,
                    neutralize_final="--neutralize-final" in toggles,
                    neutralizer_path=values.get("--neutralizer-path"),
                    neutralization_proportion=neutralization_proportion,
                    neutralization_mode=neutralization_mode,
                    neutralizer_cols=neutralizer_cols,
                    neutralization_rank_output="--no-neutralization-rank" not in toggles,
                    workspace_root=values.get("--workspace", "."),
                )
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(build_payload.model_dump_json())
        return 0

    if args[0] == "list":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--experiment-id", "--limit", "--offset", "--format", "--workspace"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

        list_output_format: OutputFormat = "table"
        if "--format" in values:
            parsed_format, format_err = _parse_output_format(values["--format"])
            if parsed_format is None or format_err is not None:
                print(format_err or "invalid value for --format", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            list_output_format = parsed_format

        limit = 50
        if "--limit" in values:
            parsed_limit, limit_err = _parse_int_value(values["--limit"], flag="--limit")
            if parsed_limit is None or limit_err is not None:
                print(limit_err or "invalid integer for --limit", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            limit = parsed_limit

        offset = 0
        if "--offset" in values:
            parsed_offset, offset_err = _parse_int_value(values["--offset"], flag="--offset")
            if parsed_offset is None or offset_err is not None:
                print(offset_err or "invalid integer for --offset", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            offset = parsed_offset

        try:
            list_payload = api.ensemble_list(
                api.EnsembleListRequest(
                    experiment_id=values.get("--experiment-id"),
                    limit=limit,
                    offset=offset,
                    workspace_root=values.get("--workspace", "."),
                )
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        if list_output_format == "json":
            print(list_payload.model_dump_json())
        else:
            _print_ensembles_table(list_payload)
        return 0

    if args[0] == "details":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--ensemble-id", "--format", "--workspace"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

        ensemble_id = values.get("--ensemble-id")
        if ensemble_id is None:
            print("missing required argument: --ensemble-id", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

        details_output_format: OutputFormat = "table"
        if "--format" in values:
            parsed_format, format_err = _parse_output_format(values["--format"])
            if parsed_format is None or format_err is not None:
                print(format_err or "invalid value for --format", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            details_output_format = parsed_format

        try:
            details_payload = api.ensemble_get(
                api.EnsembleGetRequest(
                    ensemble_id=ensemble_id,
                    workspace_root=values.get("--workspace", "."),
                )
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        if details_output_format == "json":
            print(details_payload.model_dump_json())
        else:
            _print_ensemble_details_table(details_payload)
        return 0

    print(f"unknown arguments: ensemble {' '.join(args)}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


__all__ = ["handle_ensemble_command"]
