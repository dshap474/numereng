"""CLI handlers for baseline workflows."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from pydantic import ValidationError

from numereng import api
from numereng.cli.common import _parse_simple_options, _validation_error_message
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError


def handle_baseline_command(args: Sequence[str]) -> int:
    if not args:
        print(USAGE)
        return 0

    if args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    if args[0] == "build":
        values, toggles, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--run-ids", "--name", "--default-target", "--description", "--workspace"},
            bool_flags={"--promote-active"},
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

        name = values.get("--name")
        if name is None:
            print("missing required argument: --name", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

        try:
            payload = api.baseline_build(
                api.BaselineBuildRequest(
                    run_ids=_parse_run_ids(run_ids_value),
                    name=name,
                    default_target=values.get("--default-target", "target_ender_20"),
                    description=values.get("--description"),
                    promote_active="--promote-active" in toggles,
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

        print(payload.model_dump_json())
        return 0

    print(f"unknown arguments: baseline {' '.join(args)}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


def _parse_run_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


__all__ = ["handle_baseline_command"]
