"""Run command handlers."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Literal, cast

from pydantic import ValidationError

from numereng import api
from numereng.cli.common import (
    _parse_tournament_value,
    _parse_training_profile_value,
)
from numereng.cli.usage import USAGE
from numereng.features.telemetry import bind_launch_metadata
from numereng.platform.errors import PackageError

NeutralizationModeValue = Literal["era", "global"]


def _parse_neutralization_mode(value: str) -> tuple[NeutralizationModeValue | None, str | None]:
    if value not in {"era", "global"}:
        return None, "invalid value for --neutralization-mode: expected era|global"
    return cast(NeutralizationModeValue, value), None


def _parse_neutralization_proportion(value: str) -> tuple[float | None, str | None]:
    try:
        parsed = float(value)
    except ValueError:
        return None, f"invalid float for --neutralization-proportion: {value}"
    if parsed < 0.0 or parsed > 1.0:
        return None, "invalid value for --neutralization-proportion: expected 0.0..1.0"
    return parsed, None


def _parse_neutralizer_cols(value: str) -> tuple[list[str] | None, str | None]:
    cols = [item.strip() for item in value.split(",") if item.strip()]
    if not cols:
        return None, "invalid value for --neutralizer-cols: expected comma-separated column names"
    return cols, None


def _parse_submit_request(argv: Sequence[str]) -> tuple[api.SubmissionRequest | None, str | None]:
    run_id: str | None = None
    predictions_path: str | None = None
    model_name: str | None = None
    store_root = ".numereng"
    tournament: api.NumeraiTournament = "classic"
    allow_non_live_artifact = False
    neutralize = False
    neutralizer_path: str | None = None
    neutralization_proportion = 0.5
    neutralization_mode: NeutralizationModeValue = "era"
    neutralizer_cols: list[str] | None = None
    neutralization_rank_output = True

    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg in {"-h", "--help"}:
            return None, "__help__"
        if arg == "--neutralize":
            neutralize = True
            idx += 1
            continue
        if arg == "--allow-non-live-artifact":
            allow_non_live_artifact = True
            idx += 1
            continue
        if arg == "--no-neutralization-rank":
            neutralization_rank_output = False
            idx += 1
            continue
        if arg in {
            "--run-id",
            "--predictions",
            "--model-name",
            "--store-root",
            "--tournament",
            "--neutralizer-path",
            "--neutralization-proportion",
            "--neutralization-mode",
            "--neutralizer-cols",
        }:
            if idx + 1 >= len(argv):
                return None, f"missing value for {arg}"
            value = argv[idx + 1]
            if arg == "--run-id":
                run_id = value
            elif arg == "--predictions":
                predictions_path = value
            elif arg == "--model-name":
                model_name = value
            elif arg == "--store-root":
                store_root = value
            elif arg == "--neutralizer-path":
                neutralizer_path = value
            elif arg == "--neutralization-proportion":
                parsed_proportion, parse_error = _parse_neutralization_proportion(value)
                if parse_error is not None:
                    return None, parse_error
                if parsed_proportion is None:  # pragma: no cover - parse_error branch guards this
                    return None, "invalid value for --neutralization-proportion"
                neutralization_proportion = parsed_proportion
            elif arg == "--neutralization-mode":
                parsed_mode, parse_error = _parse_neutralization_mode(value)
                if parse_error is not None:
                    return None, parse_error
                if parsed_mode is None:  # pragma: no cover - parse_error branch guards this
                    return None, "invalid value for --neutralization-mode"
                neutralization_mode = parsed_mode
            elif arg == "--neutralizer-cols":
                parsed_cols, parse_error = _parse_neutralizer_cols(value)
                if parse_error is not None:
                    return None, parse_error
                neutralizer_cols = parsed_cols
            else:
                parsed_tournament, parse_error = _parse_tournament_value(value)
                if parse_error is not None:
                    return None, parse_error
                if parsed_tournament is None:  # pragma: no cover - parse_error branch guards this
                    return None, "invalid value for --tournament"
                tournament = parsed_tournament
            idx += 2
            continue
        return None, f"unknown arguments: {arg}"

    if model_name is None:
        return None, "missing required argument: --model-name"

    try:
        request = api.SubmissionRequest(
            model_name=model_name,
            tournament=tournament,
            run_id=run_id,
            predictions_path=predictions_path,
            allow_non_live_artifact=allow_non_live_artifact,
            neutralize=neutralize,
            neutralizer_path=neutralizer_path,
            neutralization_proportion=neutralization_proportion,
            neutralization_mode=neutralization_mode,
            neutralizer_cols=neutralizer_cols,
            neutralization_rank_output=neutralization_rank_output,
            store_root=store_root,
        )
    except ValidationError as exc:
        errors = exc.errors()
        if errors:
            return None, str(errors[0]["msg"])
        return None, str(exc)

    return request, None


def _parse_train_request(argv: Sequence[str]) -> tuple[api.TrainRunRequest | None, str | None]:
    config_path: str | None = None
    output_dir: str | None = None
    profile: api.TrainingProfile | None = None
    window_size_eras: int | None = None
    embargo_eras: int | None = None
    experiment_id: str | None = None

    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg in {"-h", "--help"}:
            return None, "__help__"
        if arg in {"--method", "--method-overrides-json"}:
            return (
                None,
                f"legacy training option is no longer supported: {arg}; "
                "use --profile",
            )
        if arg in {"--engine-mode", "--window-size-eras", "--embargo-eras"}:
            return (
                None,
                f"legacy training option is no longer supported: {arg}; use --profile",
            )
        if arg in {
            "--config",
            "--output-dir",
            "--profile",
            "--experiment-id",
        }:
            if idx + 1 >= len(argv):
                return None, f"missing value for {arg}"
            value = argv[idx + 1]
            if arg == "--config":
                config_path = value
            elif arg == "--output-dir":
                output_dir = value
            elif arg == "--profile":
                parsed_profile, parse_error = _parse_training_profile_value(value)
                if parse_error is not None:
                    return None, parse_error
                profile = parsed_profile
            else:
                experiment_id = value
            idx += 2
            continue
        return None, f"unknown arguments: {arg}"

    if config_path is None:
        return None, "missing required argument: --config"

    try:
        request = api.TrainRunRequest(
            config_path=config_path,
            output_dir=output_dir,
            profile=profile,
            window_size_eras=window_size_eras,
            embargo_eras=embargo_eras,
            experiment_id=experiment_id,
        )
    except ValidationError as exc:
        errors = exc.errors()
        if errors:
            return None, str(errors[0]["msg"])
        return None, str(exc)

    return request, None


def _parse_score_request(argv: Sequence[str]) -> tuple[api.ScoreRunRequest | None, str | None]:
    run_id: str | None = None
    store_root = ".numereng"

    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg in {"-h", "--help"}:
            return None, "__help__"
        if arg in {"--run-id", "--store-root"}:
            if idx + 1 >= len(argv):
                return None, f"missing value for {arg}"
            value = argv[idx + 1]
            if arg == "--run-id":
                run_id = value
            else:
                store_root = value
            idx += 2
            continue
        return None, f"unknown arguments: {arg}"

    if run_id is None:
        return None, "missing required argument: --run-id"

    try:
        request = api.ScoreRunRequest(run_id=run_id, store_root=store_root)
    except ValidationError as exc:
        errors = exc.errors()
        if errors:
            return None, str(errors[0]["msg"])
        return None, str(exc)

    return request, None


def handle_run_command(args: Sequence[str]) -> int:
    if not args:
        print(USAGE)
        return 0

    if args[0] == "submit":
        submit_request, parse_error = _parse_submit_request(args[1:])
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

        if submit_request is None:  # pragma: no cover - parse_error branch guards this
            print("submission_request_invalid", file=sys.stderr)
            return 2

        try:
            submission_payload = api.submit_predictions(submit_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(submission_payload.model_dump_json())
        return 0

    if args[0] == "train":
        train_request, parse_error = _parse_train_request(args[1:])
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        if train_request is None:  # pragma: no cover - parse_error branch guards this
            print("training_request_invalid", file=sys.stderr)
            return 2

        try:
            with bind_launch_metadata(source="cli.run.train", operation_type="run", job_type="run"):
                training_payload = api.run_training(train_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(training_payload.model_dump_json())
        return 0

    if args[0] == "score":
        score_request, parse_error = _parse_score_request(args[1:])
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        if score_request is None:  # pragma: no cover - parse_error branch guards this
            print("score_request_invalid", file=sys.stderr)
            return 2

        try:
            with bind_launch_metadata(source="cli.run.score", operation_type="run", job_type="run"):
                score_payload = api.score_run(score_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(score_payload.model_dump_json())
        return 0

    if args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    print(f"unknown arguments: run {' '.join(args)}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


__all__ = ["handle_run_command"]
