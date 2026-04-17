"""CLI entrypoint and top-level command dispatch."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from numereng.cli.common import _parse_fail_flag
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError


def _dispatch_command(
    *,
    module_name: str,
    handler_name: str,
    args: list[str],
) -> int:
    try:
        module = __import__(module_name, fromlist=[handler_name])
        handler = getattr(module, handler_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        if missing.startswith("numereng"):
            raise
        raise PackageError(f"runtime_dependency_missing:{missing}:run_uv_sync_extra_dev") from exc
    return handler(args)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint returning process-style exit codes."""
    args = list(sys.argv[1:] if argv is None else argv)

    try:
        if args and args[0] == "run":
            return _dispatch_command(
                module_name="numereng.cli.commands.run",
                handler_name="handle_run_command",
                args=args[1:],
            )

        if args and args[0] == "baseline":
            return _dispatch_command(
                module_name="numereng.cli.commands.baseline",
                handler_name="handle_baseline_command",
                args=args[1:],
            )

        if args and args[0] == "dataset-tools":
            return _dispatch_command(
                module_name="numereng.cli.commands.dataset_tools",
                handler_name="handle_dataset_tools_command",
                args=args[1:],
            )

        if args and args[0] == "docs":
            return _dispatch_command(
                module_name="numereng.cli.commands.docs",
                handler_name="handle_docs_command",
                args=args[1:],
            )

        if args and args[0] == "experiment":
            return _dispatch_command(
                module_name="numereng.cli.commands.experiment",
                handler_name="handle_experiment_command",
                args=args[1:],
            )

        if args and args[0] == "hpo":
            return _dispatch_command(
                module_name="numereng.cli.commands.hpo",
                handler_name="handle_hpo_command",
                args=args[1:],
            )

        if args and args[0] == "neutralize":
            return _dispatch_command(
                module_name="numereng.cli.commands.neutralize",
                handler_name="handle_neutralize_command",
                args=args[1:],
            )

        if args and args[0] == "monitor":
            return _dispatch_command(
                module_name="numereng.cli.commands.monitor",
                handler_name="handle_monitor_command",
                args=args[1:],
            )

        if args and args[0] == "ensemble":
            return _dispatch_command(
                module_name="numereng.cli.commands.ensemble",
                handler_name="handle_ensemble_command",
                args=args[1:],
            )

        if args and args[0] == "store":
            return _dispatch_command(
                module_name="numereng.cli.commands.store",
                handler_name="handle_store_command",
                args=args[1:],
            )

        if args and args[0] == "cloud":
            return _dispatch_command(
                module_name="numereng.cli.commands.cloud",
                handler_name="handle_cloud_command",
                args=args[1:],
            )

        if args and args[0] == "numerai":
            return _dispatch_command(
                module_name="numereng.cli.commands.numerai",
                handler_name="handle_numerai_command",
                args=args[1:],
            )

        if args and args[0] == "serve":
            return _dispatch_command(
                module_name="numereng.cli.commands.serve",
                handler_name="handle_serve_command",
                args=args[1:],
            )

        if args and args[0] == "research":
            return _dispatch_command(
                module_name="numereng.cli.commands.research",
                handler_name="handle_research_command",
                args=args[1:],
            )

        if args and args[0] == "remote":
            return _dispatch_command(
                module_name="numereng.cli.commands.remote",
                handler_name="handle_remote_command",
                args=args[1:],
            )

        if args and args[0] == "viz":
            return _dispatch_command(
                module_name="numereng.cli.commands.viz",
                handler_name="handle_viz_command",
                args=args[1:],
            )
    except PackageError as exc:
        print(str(exc), file=sys.stderr)
        return 1

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
