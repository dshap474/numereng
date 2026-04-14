"""CLI helpers for `numereng workspace ...`."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence

from pydantic import ValidationError

import numereng.api as api
from numereng.cli.common import _parse_simple_options, _validation_error_message
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError


def handle_workspace_command(args: Sequence[str]) -> int:
    if not args or args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0
    if args[0] == "sync":
        return _handle_workspace_sync(args[1:])
    print(f"unknown workspace command: {args[0]}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


def _handle_workspace_sync(args: Sequence[str]) -> int:
    values, toggles, parse_error = _parse_simple_options(
        args,
        value_flags={"--workspace", "--runtime-source", "--runtime-path"},
        bool_flags={"--with-training", "--with-mlops"},
    )
    if parse_error == "__help__":
        print(USAGE)
        return 0
    if parse_error is not None:
        print(parse_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    try:
        response = api.workspace_sync(
            api.WorkspaceSyncRequest(
                workspace_root=values.get("--workspace", "."),
                runtime_source=values.get("--runtime-source"),
                runtime_path=values.get("--runtime-path"),
                with_training=True if "--with-training" in toggles else None,
                with_mlops=True if "--with-mlops" in toggles else None,
            )
        )
    except ValidationError as exc:
        print(_validation_error_message(exc), file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    except PackageError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(response.model_dump()))
    return 0


__all__ = ["handle_workspace_command"]
