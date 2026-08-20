"""Closeout-chain command handlers (`research closeout`, `research closeout-status`)."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from pydantic import ValidationError

from numereng import api
from numereng.cli.commands.research import _parse_output_format
from numereng.cli.common import _parse_simple_options, _validation_error_message
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError

_VALID_UNTIL = {"finalize", "classify", "extract", "synthesize"}


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _print_closeout_table(payload: api.ResearchCloseoutResponse) -> None:
    print(f"experiment_id: {payload.experiment_id}")
    print(f"stopped_at_phase: {payload.stopped_at_phase or 'none'}")
    print(f"error: {payload.error or 'none'}")
    for phase in payload.phases:
        duration = "n/a" if phase.duration_seconds is None else f"{phase.duration_seconds:.1f}s"
        outputs = ", ".join(sorted(phase.outputs)) or "none"
        print(f"{phase.name} | {phase.status} | {duration} | outputs={outputs}")


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def handle_research_closeout_command(args: Sequence[str], *, status_only: bool) -> int:
    value_flags = {"--experiment-id", "--workspace", "--format"}
    bool_flags: set[str] = set()
    if not status_only:
        value_flags |= {"--until", "--restart-from", "--memory-root"}
        bool_flags |= {"--accept-stale-running", "--allow-incomplete"}

    values, toggles, parse_error = _parse_simple_options(args, value_flags=value_flags, bool_flags=bool_flags)
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

    output_format = "table"
    if "--format" in values:
        output_format, format_error = _parse_output_format(values["--format"])
        if format_error is not None or output_format is None:
            print(format_error or "invalid value for --format", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

    for flag in ("--until", "--restart-from"):
        if flag in values and values[flag] not in _VALID_UNTIL:
            print(f"invalid value for {flag}: expected one of {sorted(_VALID_UNTIL)}", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

    try:
        if status_only:
            payload = api.research_closeout_status(
                api.ResearchCloseoutStatusRequest(
                    experiment_id=experiment_id,
                    workspace_root=values.get("--workspace", "."),
                )
            )
        else:
            payload = api.research_closeout(
                api.ResearchCloseoutRequest(
                    experiment_id=experiment_id,
                    until=values.get("--until"),
                    restart_from=values.get("--restart-from"),
                    memory_root=values.get("--memory-root"),
                    accept_stale_running="--accept-stale-running" in toggles,
                    allow_incomplete="--allow-incomplete" in toggles,
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

    if output_format == "json":
        print(payload.model_dump_json())
    else:
        _print_closeout_table(payload)
    return 1 if payload.error else 0


__all__ = ["handle_research_closeout_command"]
