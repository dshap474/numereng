"""Agentic config-research command handlers."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from pydantic import ValidationError

from numereng import api
from numereng.cli.common import _parse_int_value, _parse_simple_options, _validation_error_message
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError


def _parse_output_format(value: str) -> tuple[str | None, str | None]:
    if value not in {"table", "json"}:
        return None, "invalid value for --format: expected table|json"
    return value, None


def _print_research_status_table(payload: api.ResearchStatusResponse) -> None:
    print(f"experiment_id: {payload.experiment_id}")
    print(f"status: {payload.status}")
    print(f"next_round_number: {payload.next_round_number}")
    print(f"total_rounds_completed: {payload.total_rounds_completed}")
    print(f"last_checkpoint: {payload.last_checkpoint}")
    print(f"last_round_label: {payload.last_round_label or 'none'}")
    print(f"last_run_id: {payload.last_run_id or 'none'}")
    print(f"stop_reason: {payload.stop_reason or 'none'}")
    champion = payload.champion or {}
    metric = champion.get("metric")
    print(f"champion_config: {champion.get('config') or 'none'}")
    print(f"champion_run_id: {champion.get('run_id') or 'none'}")
    print(f"champion_metric: {metric:.6f}" if isinstance(metric, float) else "champion_metric: none")
    print(f"agentic_research_dir: {payload.agentic_research_dir}")
    print(f"closeout_memo: {payload.closeout_memo}")


def _print_research_run_table(payload: api.ResearchRunResponse) -> None:
    print(f"experiment_id: {payload.experiment_id}")
    print(f"status: {payload.status}")
    print(f"rounds_completed: {len(payload.rounds)}")
    print(f"next_round_number: {payload.next_round_number}")
    print(f"stop_reason: {payload.stop_reason or 'none'}")
    for item in payload.rounds:
        metric = "n/a" if item.metric_value is None else f"{item.metric_value:.6f}"
        print(f"{item.round_label} | {item.action} | run_id={item.run_id or 'none'} | metric={metric}")


def _print_closeout_table(payload: api.ResearchCloseoutResponse) -> None:
    print(f"experiment_id: {payload.experiment_id}")
    print(f"evidence_path: {payload.evidence_path}")
    print(f"memo_path: {payload.memo_path}")
    holdout = payload.holdout_summary or {}
    print(f"holdout: {holdout.get('holdout_primary_mean', 'none')}")


def _handle_closeout(args: Sequence[str]) -> int:
    values, toggles, parse_error = _parse_simple_options(
        args,
        value_flags={"--experiment-id", "--workspace", "--format"},
        bool_flags={"--allow-incomplete"},
    )
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
    try:
        payload = api.research_closeout(
            api.ResearchCloseoutRequest(
                experiment_id=experiment_id,
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
    return 0


def handle_research_command(args: Sequence[str]) -> int:
    if not args or args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    if args[0] == "closeout":
        return _handle_closeout(args[1:])

    if args[0] == "status":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--experiment-id", "--workspace", "--format"},
        )
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
        try:
            payload = api.research_status(
                api.ResearchStatusRequest(
                    experiment_id=experiment_id,
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
            _print_research_status_table(payload)
        return 0

    if args[0] == "run":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--experiment-id", "--workspace", "--max-rounds"},
        )
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
        max_rounds = 1
        if "--max-rounds" in values:
            parsed, int_error = _parse_int_value(values["--max-rounds"], flag="--max-rounds")
            if int_error is not None or parsed is None:
                print(int_error or "invalid value for --max-rounds", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            max_rounds = parsed
        try:
            payload = api.research_run(
                api.ResearchRunRequest(
                    experiment_id=experiment_id,
                    max_rounds=max_rounds,
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
        _print_research_run_table(payload)
        return 0

    print(USAGE, file=sys.stderr)
    return 2


__all__ = ["handle_research_command"]
