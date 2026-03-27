"""CLI entrypoint and top-level command dispatch."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from numereng import api
from numereng.cli.commands import (
    handle_baseline_command,
    handle_cloud_command,
    handle_dataset_tools_command,
    handle_ensemble_command,
    handle_experiment_command,
    handle_hpo_command,
    handle_monitor_command,
    handle_neutralize_command,
    handle_numerai_command,
    handle_research_command,
    handle_run_command,
    handle_store_command,
)
from numereng.cli.common import _parse_fail_flag
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint returning process-style exit codes."""
    args = list(sys.argv[1:] if argv is None else argv)

    if args and args[0] == "run":
        return handle_run_command(args[1:])

    if args and args[0] == "baseline":
        return handle_baseline_command(args[1:])

    if args and args[0] == "dataset-tools":
        return handle_dataset_tools_command(args[1:])

    if args and args[0] == "experiment":
        return handle_experiment_command(args[1:])

    if args and args[0] == "hpo":
        return handle_hpo_command(args[1:])

    if args and args[0] == "neutralize":
        return handle_neutralize_command(args[1:])

    if args and args[0] == "monitor":
        return handle_monitor_command(args[1:])

    if args and args[0] == "ensemble":
        return handle_ensemble_command(args[1:])

    if args and args[0] == "store":
        return handle_store_command(args[1:])

    if args and args[0] == "cloud":
        return handle_cloud_command(args[1:])

    if args and args[0] == "numerai":
        return handle_numerai_command(args[1:])

    if args and args[0] == "research":
        return handle_research_command(args[1:])

    if any(arg in {"-h", "--help"} for arg in args):
        print(USAGE)
        return 0

    fail, unknown = _parse_fail_flag(args)
    if unknown:
        print(f"unknown arguments: {' '.join(unknown)}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    try:
        health_payload = api.run_bootstrap_check(fail=fail)
    except PackageError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(health_payload.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
