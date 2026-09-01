"""`numereng research program check|resplice` — program CORE drift check and mechanical re-splice.

USAGE:
    numereng research program check --experiment-id <id> [--format <table|json>] [--workspace <path>]
    numereng research program resplice --experiment-id <id> [--format <table|json>] [--workspace <path>]

`check` exits 1 when the program's CORE has drifted from PROGRAM.md so it can gate scripts;
`resplice` rewrites the CORE sections in place (backup kept beside the program) and exits 0.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from pydantic import ValidationError

from numereng import api
from numereng.cli.common import _parse_simple_options, _validation_error_message
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError

# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def _print_program_table(payload: api.ResearchProgramResponse) -> None:
    print(f"experiment_id: {payload.experiment_id}")
    print(f"program_path: {payload.program_path}")
    print(f"base_program_path: {payload.base_program_path}")
    print(f"is_base_program: {payload.is_base_program}")
    print(f"in_sync: {payload.in_sync}")
    print(f"diverging_section: {payload.diverging_section or 'none'}")
    print(f"written: {payload.written}")
    print(f"backup_path: {payload.backup_path or 'none'}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def handle_research_program_command(args: Sequence[str]) -> int:
    if not args or args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0
    action = args[0]
    if action not in {"check", "resplice"}:
        print(f"unknown research program action: {action}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    values, _, parse_error = _parse_simple_options(args[1:], value_flags={"--experiment-id", "--workspace", "--format"})
    if parse_error == "__help__":
        print(USAGE)
        return 0
    if parse_error is not None:
        print(parse_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    experiment_id = values.get("--experiment-id")
    if experiment_id is None:
        print("missing required argument: --experiment-id", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    output_format = values.get("--format", "table")
    if output_format not in {"table", "json"}:
        print("invalid value for --format: expected table|json", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    try:
        request = api.ResearchProgramRequest(experiment_id=experiment_id, workspace_root=values.get("--workspace", "."))
        payload = api.research_program_check(request) if action == "check" else api.research_program_resplice(request)
    except ValidationError as exc:
        print(_validation_error_message(exc), file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    except PackageError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if output_format == "json":
        print(payload.model_dump_json())
    else:
        _print_program_table(payload)
    return 0 if payload.in_sync else 1
