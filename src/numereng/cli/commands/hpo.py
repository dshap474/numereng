"""HPO command handlers."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Literal, cast

from pydantic import ValidationError

from numereng import api
from numereng.cli.commands.hpo_create import handle_hpo_create
from numereng.cli.common import _parse_int_value, _parse_simple_options, _validation_error_message
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError

OutputFormat = Literal["table", "json"]


def _parse_output_format(value: str) -> tuple[OutputFormat | None, str | None]:
    if value not in {"table", "json"}:
        return None, "invalid value for --format: expected table|json"
    return cast(OutputFormat, value), None


def _print_studies_table(payload: api.HpoStudyListResponse) -> None:
    if not payload.studies:
        print("No HPO studies found")
        return

    header = f"{'Study ID':<44} {'Status':<10} {'Trials':<6} {'Best Run':<14} {'Best Value':>10}"
    print(header)
    print("-" * len(header))
    for item in payload.studies:
        best_value = "n/a" if item.best_value is None else f"{item.best_value:.6f}"
        print(
            f"{item.study_id:<44} {item.status:<10} {item.n_trials:<6} "
            f"{(item.best_run_id or '-'): <14} {best_value:>10}"
        )


def _print_study_details_table(payload: api.HpoStudyResponse) -> None:
    print(f"study_id: {payload.study_id}")
    print(f"study_name: {payload.study_name}")
    print(f"experiment_id: {payload.experiment_id or 'none'}")
    print(f"status: {payload.status}")
    print(f"metric: {payload.metric}")
    print(f"direction: {payload.direction}")
    print(f"trials: {payload.n_trials}")
    print(f"sampler: {payload.sampler}")
    print(f"seed: {payload.seed}")
    print(f"best_trial_number: {payload.best_trial_number}")
    print(f"best_value: {payload.best_value}")
    print(f"best_run_id: {payload.best_run_id}")
    print(f"storage_path: {payload.storage_path}")
    if payload.error_message:
        print(f"error_message: {payload.error_message}")


def _print_trials_table(payload: api.HpoStudyTrialsResponse) -> None:
    if not payload.trials:
        print("No HPO trials found")
        return

    header = f"{'Trial':<8} {'Status':<10} {'Value':>12} {'Run ID':<14} {'Error'}"
    print(header)
    print("-" * len(header))
    for trial in payload.trials:
        value = "n/a" if trial.value is None else f"{trial.value:.6f}"
        print(
            f"{trial.trial_number:<8} {trial.status:<10} {value:>12} "
            f"{(trial.run_id or '-'): <14} {trial.error_message or ''}"
        )


def handle_hpo_command(args: Sequence[str]) -> int:
    if not args:
        print(USAGE)
        return 0

    if args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    if args[0] == "create":
        return handle_hpo_create(args[1:])

    if args[0] == "list":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--experiment-id", "--status", "--limit", "--offset", "--format", "--workspace"},
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
            list_payload = api.hpo_list(
                api.HpoStudyListRequest(
                    experiment_id=values.get("--experiment-id"),
                    status=values.get("--status"),
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
            _print_studies_table(list_payload)
        return 0

    if args[0] == "details":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--study-id", "--format", "--workspace"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

        study_id = values.get("--study-id")
        if study_id is None:
            print("missing required argument: --study-id", file=sys.stderr)
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
            study_payload = api.hpo_get(
                api.HpoStudyGetRequest(
                    study_id=study_id,
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
            print(study_payload.model_dump_json())
        else:
            _print_study_details_table(study_payload)
        return 0

    if args[0] == "trials":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--study-id", "--format", "--workspace"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

        study_id = values.get("--study-id")
        if study_id is None:
            print("missing required argument: --study-id", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

        trials_output_format: OutputFormat = "table"
        if "--format" in values:
            parsed_format, format_err = _parse_output_format(values["--format"])
            if parsed_format is None or format_err is not None:
                print(format_err or "invalid value for --format", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            trials_output_format = parsed_format

        try:
            trials_payload = api.hpo_trials(
                api.HpoStudyTrialsRequest(
                    study_id=study_id,
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

        if trials_output_format == "json":
            print(trials_payload.model_dump_json())
        else:
            _print_trials_table(trials_payload)
        return 0

    print(f"unknown arguments: hpo {' '.join(args)}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


__all__ = ["handle_hpo_command"]
