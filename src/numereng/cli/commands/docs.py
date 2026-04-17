"""CLI helpers for `numereng docs ...`."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence

from pydantic import ValidationError

import numereng.api as api
from numereng.cli.common import _parse_simple_options, _validation_error_message
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError


def handle_docs_command(args: Sequence[str]) -> int:
    if not args or args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0
    if args[0] == "sync":
        return _handle_docs_sync(args[1:])
    print(f"unknown docs command: {args[0]}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


def _handle_docs_sync(args: Sequence[str]) -> int:
    if not args:
        print(USAGE)
        return 0
    if args[0] != "numerai":
        print(f"unknown docs sync target: {args[0]}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    values, _, parse_error = _parse_simple_options(args[1:], value_flags={"--workspace"})
    if parse_error == "__help__":
        print(USAGE)
        return 0
    if parse_error is not None:
        print(parse_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    try:
        response = api.sync_docs(
            api.DocsSyncRequest(
                workspace_root=values.get("--workspace", "."),
                domain="numerai",
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


__all__ = ["handle_docs_command"]
