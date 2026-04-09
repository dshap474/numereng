"""Workspace bootstrap command handler."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from pydantic import ValidationError

from numereng import api
from numereng.cli.common import _parse_simple_options, _validation_error_message
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError


def handle_init_command(args: Sequence[str]) -> int:
    """Initialize one canonical numereng workspace."""

    values, _, parse_error = _parse_simple_options(args, value_flags={"--workspace"})
    if parse_error == "__help__":
        print(USAGE)
        return 0
    if parse_error is not None:
        print(parse_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    try:
        payload = api.workspace_init(
            api.WorkspaceInitRequest(
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


__all__ = ["handle_init_command"]
