"""Feature-neutralization command handlers."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Literal, cast

from pydantic import ValidationError

from numereng import api
from numereng.cli.common import _parse_simple_options, _validation_error_message
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError

NeutralizationModeValue = Literal["era", "global"]


def _parse_mode(value: str) -> tuple[NeutralizationModeValue | None, str | None]:
    if value not in {"era", "global"}:
        return None, "invalid value for --neutralization-mode: expected era|global"
    return cast(NeutralizationModeValue, value), None


def _parse_proportion(value: str) -> tuple[float | None, str | None]:
    try:
        parsed = float(value)
    except ValueError:
        return None, f"invalid float for --neutralization-proportion: {value}"
    if parsed < 0.0 or parsed > 1.0:
        return None, "invalid value for --neutralization-proportion: expected 0.0..1.0"
    return parsed, None


def _parse_cols_csv(value: str) -> tuple[list[str] | None, str | None]:
    cols = [item.strip() for item in value.split(",") if item.strip()]
    if not cols:
        return None, "invalid value for --neutralizer-cols: expected comma-separated column names"
    return cols, None


def handle_neutralize_command(args: Sequence[str]) -> int:
    if not args:
        print(USAGE)
        return 0

    if args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    if args[0] == "apply":
        values, toggles, parse_error = _parse_simple_options(
            args[1:],
            value_flags={
                "--run-id",
                "--predictions",
                "--neutralizer-path",
                "--neutralization-proportion",
                "--neutralization-mode",
                "--neutralizer-cols",
                "--output-path",
                "--store-root",
            },
            bool_flags={"--no-neutralization-rank"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

        run_id = values.get("--run-id")
        predictions_path = values.get("--predictions")
        if (run_id is None) == (predictions_path is None):
            print("exactly one of --run-id or --predictions is required", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

        neutralizer_path = values.get("--neutralizer-path")
        if neutralizer_path is None:
            print("missing required argument: --neutralizer-path", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

        proportion = 0.5
        if "--neutralization-proportion" in values:
            parsed, err = _parse_proportion(values["--neutralization-proportion"])
            if parsed is None or err is not None:
                print(err or "invalid value for --neutralization-proportion", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            proportion = parsed

        mode: NeutralizationModeValue = "era"
        if "--neutralization-mode" in values:
            parsed_mode, mode_err = _parse_mode(values["--neutralization-mode"])
            if parsed_mode is None or mode_err is not None:
                print(mode_err or "invalid value for --neutralization-mode", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            mode = parsed_mode

        neutralizer_cols: list[str] | None = None
        if "--neutralizer-cols" in values:
            parsed_cols, cols_err = _parse_cols_csv(values["--neutralizer-cols"])
            if parsed_cols is None or cols_err is not None:
                print(cols_err or "invalid value for --neutralizer-cols", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            neutralizer_cols = parsed_cols

        try:
            payload = api.neutralize_apply(
                api.NeutralizeRequest(
                    run_id=run_id,
                    predictions_path=predictions_path,
                    neutralizer_path=neutralizer_path,
                    neutralization_proportion=proportion,
                    neutralization_mode=mode,
                    neutralizer_cols=neutralizer_cols,
                    neutralization_rank_output="--no-neutralization-rank" not in toggles,
                    output_path=values.get("--output-path"),
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

        print(payload.model_dump_json())
        return 0

    print(f"unknown arguments: neutralize {' '.join(args)}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


__all__ = ["handle_neutralize_command"]
