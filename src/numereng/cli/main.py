"""CLI entrypoint and top-level command dispatch."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from numereng.cli.common import _parse_fail_flag
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint returning process-style exit codes."""
    args = list(sys.argv[1:] if argv is None else argv)

    if args and args[0] == "run":
        from numereng.cli.commands.run import handle_run_command

        return handle_run_command(args[1:])

    if args and args[0] == "baseline":
        from numereng.cli.commands.baseline import handle_baseline_command

        return handle_baseline_command(args[1:])

    if args and args[0] == "dataset-tools":
        from numereng.cli.commands.dataset_tools import handle_dataset_tools_command

        return handle_dataset_tools_command(args[1:])

    if args and args[0] == "init":
        from numereng.cli.commands.init import handle_init_command

        return handle_init_command(args[1:])

    if args and args[0] == "experiment":
        from numereng.cli.commands.experiment import handle_experiment_command

        return handle_experiment_command(args[1:])

    if args and args[0] == "hpo":
        from numereng.cli.commands.hpo import handle_hpo_command

        return handle_hpo_command(args[1:])

    if args and args[0] == "neutralize":
        from numereng.cli.commands.neutralize import handle_neutralize_command

        return handle_neutralize_command(args[1:])

    if args and args[0] == "monitor":
        from numereng.cli.commands.monitor import handle_monitor_command

        return handle_monitor_command(args[1:])

    if args and args[0] == "ensemble":
        from numereng.cli.commands.ensemble import handle_ensemble_command

        return handle_ensemble_command(args[1:])

    if args and args[0] == "store":
        from numereng.cli.commands.store import handle_store_command

        return handle_store_command(args[1:])

    if args and args[0] == "cloud":
        from numereng.cli.commands.cloud import handle_cloud_command

        return handle_cloud_command(args[1:])

    if args and args[0] == "numerai":
        from numereng.cli.commands.numerai import handle_numerai_command

        return handle_numerai_command(args[1:])

    if args and args[0] == "serve":
        from numereng.cli.commands.serve import handle_serve_command

        return handle_serve_command(args[1:])

    if args and args[0] == "research":
        from numereng.cli.commands.research import handle_research_command

        return handle_research_command(args[1:])

    if args and args[0] == "remote":
        from numereng.cli.commands.remote import handle_remote_command

        return handle_remote_command(args[1:])

    if args and args[0] == "viz":
        from numereng.cli.commands.viz import handle_viz_command

        return handle_viz_command(args[1:])

    if any(arg in {"-h", "--help"} for arg in args):
        print(USAGE)
        return 0

    fail, unknown = _parse_fail_flag(args)
    if unknown:
        print(f"unknown arguments: {' '.join(unknown)}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    try:
        from numereng.api._health import run_bootstrap_check

        health_payload = run_bootstrap_check(fail=fail)
    except PackageError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(health_payload.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
