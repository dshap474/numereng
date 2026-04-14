"""Workspace bootstrap command handler."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence

from pydantic import ValidationError

import numereng.api as api
from numereng.cli.common import _parse_simple_options, _validation_error_message
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError


def handle_init_command(args: Sequence[str]) -> int:
    """Initialize one canonical numereng workspace."""

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
        payload = api.workspace_init(
            api.WorkspaceInitRequest(
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
    except OSError as exc:
        print(f"workspace_init_failed:{exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "workspace_root": str(payload.workspace_root),
                "store_root": str(payload.store_root),
                "created_paths": payload.created_paths,
                "skipped_existing_paths": payload.skipped_existing_paths,
                "installed_skill_ids": list(payload.installed_skill_ids),
                "workspace_project_path": payload.workspace_project_path,
                "python_version_path": payload.python_version_path,
                "venv_path": payload.venv_path,
                "updated_paths": payload.updated_paths,
                "runtime_source": payload.runtime_source,
                "runtime_path": payload.runtime_path,
                "extras": payload.extras,
                "dependency_spec": payload.dependency_spec,
                "installed_numereng_version": payload.installed_numereng_version,
                "verified_dependencies": payload.verified_dependencies,
            }
        )
    )
    return 0


__all__ = ["handle_init_command"]
