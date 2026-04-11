"""Workspace bootstrap command handler."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence

from numereng.cli.common import _parse_simple_options
from numereng.cli.usage import USAGE
from numereng.features.workspace.service import init_workspace
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
        payload = init_workspace(workspace_root=values.get("--workspace", "."))
    except PackageError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"workspace_init_failed:{exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "workspace_root": str(payload.workspace_root),
                "store_root": str(payload.store_root),
                "created_paths": [str(path) for path in payload.created_paths],
                "skipped_existing_paths": [str(path) for path in payload.skipped_existing_paths],
                "installed_skill_ids": list(payload.installed_skill_ids),
            }
        )
    )
    return 0


__all__ = ["handle_init_command"]
