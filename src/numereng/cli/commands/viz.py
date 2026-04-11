"""Packaged viz runtime command handler."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from types import SimpleNamespace

import uvicorn

from numereng.cli.common import _parse_simple_options
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError


def create_viz_app(request: object) -> object:
    from numereng_viz import create_app as create_packaged_viz_app

    workspace_root = getattr(request, "workspace_root", ".")
    return create_packaged_viz_app(workspace_root=workspace_root)


def handle_viz_command(args: Sequence[str]) -> int:
    """Launch the packaged viz backend + frontend for one workspace."""

    values, _, parse_error = _parse_simple_options(
        args,
        value_flags={"--workspace", "--host", "--port"},
    )
    if parse_error == "__help__":
        print(USAGE)
        return 0
    if parse_error is not None:
        print(parse_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    host = values.get("--host", "127.0.0.1")
    port_raw = values.get("--port", "8502")
    try:
        port = int(port_raw)
    except ValueError:
        print("--port must be an integer", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    try:
        app = create_viz_app(SimpleNamespace(workspace_root=values.get("--workspace", ".")))
    except PackageError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    uvicorn.run(app, host=host, port=port)
    return 0


__all__ = ["handle_viz_command"]
