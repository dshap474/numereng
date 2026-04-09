"""Agentic research command handlers."""

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


def _print_program_list_table(payload: api.ResearchProgramListResponse) -> None:
    for item in payload.programs:
        phase_text = "phase-aware" if item.phase_aware else "single-phase"
        print(
            f"{item.program_id} | {item.title} | source={item.source} | contract={item.planner_contract} | {phase_text}"
        )


def _print_program_show_table(payload: api.ResearchProgramShowResponse) -> None:
    print(f"program_id: {payload.program_id}")
    print(f"title: {payload.title}")
    print(f"description: {payload.description}")
    print(f"source: {payload.source}")
    print(f"planner_contract: {payload.planner_contract}")
    print(f"scoring_stage: {payload.scoring_stage}")
    print(f"improvement_threshold_default: {payload.improvement_threshold_default}")
    print(f"source_path: {payload.source_path or 'none'}")
    print(f"allowed_paths: {len(payload.config_policy.allowed_paths)}")
    print(f"max_candidate_configs: {payload.config_policy.max_candidate_configs}")
    print(f"phase_count: {len(payload.phases)}")
    print()
    print(payload.raw_markdown.rstrip())


def _print_research_status_table(payload: api.ResearchStatusResponse) -> None:
    print(f"root_experiment_id: {payload.root_experiment_id}")
    print(f"program_id: {payload.program_id}")
    print(f"program_title: {payload.program_title}")
    print(f"status: {payload.status}")
    print(f"active_experiment_id: {payload.active_experiment_id}")
    print(f"active_path_id: {payload.active_path_id}")
    print(f"next_round_number: {payload.next_round_number}")
    print(f"total_rounds_completed: {payload.total_rounds_completed}")
    print(f"total_paths_created: {payload.total_paths_created}")
    print(f"improvement_threshold: {payload.improvement_threshold}")
    print(f"last_checkpoint: {payload.last_checkpoint}")
    print(f"stop_reason: {payload.stop_reason or 'none'}")
    print(f"best_run_id: {payload.best_overall.run_id or 'none'}")
    if payload.current_phase is not None:
        print(f"current_phase: {payload.current_phase.phase_id} ({payload.current_phase.status})")
        print(f"current_phase_round_count: {payload.current_phase.round_count}")
    if payload.best_overall.bmc_last_200_eras_mean is not None:
        print(f"best_bmc_last_200_eras_mean: {payload.best_overall.bmc_last_200_eras_mean:.6f}")
    if payload.current_round is not None:
        print(f"current_round: {payload.current_round.round_label} ({payload.current_round.status})")


def handle_research_command(args: Sequence[str]) -> int:
    if not args or args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    if args[0] == "program":
        if len(args) < 2 or args[1] in {"-h", "--help"}:
            print(USAGE)
            return 0
        if args[1] == "list":
            values, _, parse_error = _parse_simple_options(args[2:], value_flags={"--format", "--workspace"})
            if parse_error == "__help__":
                print(USAGE)
                return 0
            if parse_error is not None:
                print(parse_error, file=sys.stderr)
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
                payload = api.research_program_list(
                    api.ResearchProgramListRequest(workspace_root=values.get("--workspace", "."))
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
                _print_program_list_table(payload)
            return 0
        if args[1] == "show":
            values, _, parse_error = _parse_simple_options(
                args[2:],
                value_flags={"--program", "--format", "--workspace"},
            )
            if parse_error == "__help__":
                print(USAGE)
                return 0
            if parse_error is not None:
                print(parse_error, file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            program_id = values.get("--program")
            if program_id is None:
                print("missing required argument: --program", file=sys.stderr)
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
                payload = api.research_program_show(
                    api.ResearchProgramShowRequest(
                        program_id=program_id,
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
                _print_program_show_table(payload)
            return 0
        print(USAGE, file=sys.stderr)
        return 2

    if args[0] == "init":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--experiment-id", "--program", "--workspace"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        experiment_id = values.get("--experiment-id")
        program_id = values.get("--program")
        if experiment_id is None:
            print("missing required argument: --experiment-id", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        if program_id is None:
            print("missing required argument: --program", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.research_init(
                api.ResearchInitRequest(
                    experiment_id=experiment_id,
                    program_id=program_id,
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
            value_flags={"--experiment-id", "--workspace", "--max-rounds", "--max-paths"},
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
        max_rounds: int | None = None
        if "--max-rounds" in values:
            max_rounds, int_error = _parse_int_value(values["--max-rounds"], flag="--max-rounds")
            if int_error is not None:
                print(int_error, file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
        max_paths: int | None = None
        if "--max-paths" in values:
            max_paths, int_error = _parse_int_value(values["--max-paths"], flag="--max-paths")
            if int_error is not None:
                print(int_error, file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
        try:
            payload = api.research_run(
                api.ResearchRunRequest(
                    experiment_id=experiment_id,
                    max_rounds=max_rounds,
                    max_paths=max_paths,
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

    print(USAGE, file=sys.stderr)
    return 2


__all__ = ["handle_research_command"]
