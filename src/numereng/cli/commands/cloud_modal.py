"""Cloud Modal managed command handlers."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from pydantic import ValidationError

from numereng import api
from numereng.cli.commands.cloud_modal_data import handle_cloud_modal_data_command
from numereng.cli.common import (
    _parse_int_value,
    _parse_simple_options,
    _parse_training_profile_value,
    _validation_error_message,
)
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError


def _parse_float_value(value: str, *, flag: str) -> tuple[float | None, str | None]:
    try:
        return float(value), None
    except ValueError:
        return None, f"invalid number for {flag}: {value}"


def _parse_metadata_value(raw: str) -> tuple[dict[str, str] | None, str | None]:
    stripped = raw.strip()
    if not stripped:
        return {}, None

    metadata: dict[str, str] = {}
    for entry in stripped.split(","):
        chunk = entry.strip()
        if not chunk:
            return None, "invalid value for --metadata: empty entry"
        key, sep, value = chunk.partition("=")
        key = key.strip()
        value = value.strip()
        if sep != "=" or not key:
            return None, f"invalid value for --metadata: {chunk} (expected key=value)"
        metadata[key] = value
    return metadata, None


def _handle_cloud_modal_deploy_command(args: Sequence[str]) -> int:
    values, _, parse_error = _parse_simple_options(
        args,
        value_flags={
            "--app-name",
            "--function-name",
            "--ecr-image-uri",
            "--environment-name",
            "--aws-profile",
            "--timeout-seconds",
            "--gpu",
            "--cpu",
            "--memory-mb",
            "--data-volume-name",
            "--metadata",
            "--state-path",
        },
    )
    if parse_error == "__help__":
        print(USAGE)
        return 0
    if parse_error is not None:
        print(parse_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    ecr_image_uri = values.get("--ecr-image-uri")
    if ecr_image_uri is None:
        print("missing required argument: --ecr-image-uri", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    timeout_seconds: int | None = None
    if "--timeout-seconds" in values:
        parsed_timeout, parse_err = _parse_int_value(values["--timeout-seconds"], flag="--timeout-seconds")
        if parse_err is not None or parsed_timeout is None:
            print(parse_err or "invalid integer for --timeout-seconds", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        timeout_seconds = parsed_timeout

    cpu: float | None = None
    if "--cpu" in values:
        parsed_cpu, parse_err = _parse_float_value(values["--cpu"], flag="--cpu")
        if parse_err is not None or parsed_cpu is None:
            print(parse_err or "invalid number for --cpu", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        cpu = parsed_cpu

    memory_mb: int | None = None
    if "--memory-mb" in values:
        parsed_memory, parse_err = _parse_int_value(values["--memory-mb"], flag="--memory-mb")
        if parse_err is not None or parsed_memory is None:
            print(parse_err or "invalid integer for --memory-mb", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        memory_mb = parsed_memory

    metadata: dict[str, str] = {}
    if "--metadata" in values:
        parsed_metadata, metadata_error = _parse_metadata_value(values["--metadata"])
        if metadata_error is not None or parsed_metadata is None:
            print(metadata_error or "invalid value for --metadata", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        metadata = parsed_metadata

    try:
        request = api.ModalDeployRequest(
            app_name=values.get("--app-name", "numereng-train"),
            function_name=values.get("--function-name", "train_remote"),
            ecr_image_uri=ecr_image_uri,
            environment_name=values.get("--environment-name"),
            aws_profile=values.get("--aws-profile"),
            timeout_seconds=timeout_seconds,
            gpu=values.get("--gpu"),
            cpu=cpu,
            memory_mb=memory_mb,
            data_volume_name=values.get("--data-volume-name"),
            metadata=metadata,
            state_path=values.get("--state-path"),
        )
    except ValidationError as exc:
        print(_validation_error_message(exc), file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    try:
        payload = api.cloud_modal_deploy(request)
    except PackageError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(payload.model_dump_json())
    return 0


def _handle_cloud_modal_train_command(args: Sequence[str]) -> int:
    if not args:
        print(USAGE)
        return 0
    if args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    if args[0] == "submit":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={
                "--config",
                "--output-dir",
                "--profile",
                "--engine-mode",
                "--window-size-eras",
                "--embargo-eras",
                "--app-name",
                "--function-name",
                "--environment-name",
                "--metadata",
                "--state-path",
            },
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

        config_path = values.get("--config")
        if config_path is None:
            print("missing required argument: --config", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

        for legacy_flag in ("--engine-mode", "--window-size-eras", "--embargo-eras"):
            if legacy_flag in values:
                print(f"legacy training option is no longer supported: {legacy_flag}; use --profile", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2

        profile: api.TrainingProfile | None = None
        if "--profile" in values:
            parsed_profile, mode_error = _parse_training_profile_value(values["--profile"])
            if mode_error is not None:
                print(mode_error, file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            profile = parsed_profile

        metadata: dict[str, str] = {}
        if "--metadata" in values:
            parsed_metadata, metadata_error = _parse_metadata_value(values["--metadata"])
            if metadata_error is not None or parsed_metadata is None:
                print(metadata_error or "invalid value for --metadata", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            metadata = parsed_metadata

        try:
            submit_request = api.ModalTrainSubmitRequest(
                config_path=config_path,
                output_dir=values.get("--output-dir"),
                profile=profile,
                app_name=values.get("--app-name", "numereng-train"),
                function_name=values.get("--function-name", "train_remote"),
                environment_name=values.get("--environment-name"),
                metadata=metadata,
                state_path=values.get("--state-path"),
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.cloud_modal_train_submit(submit_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(payload.model_dump_json())
        return 0

    if args[0] == "status":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--call-id", "--timeout-seconds", "--state-path"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

        timeout_seconds = 0.0
        if "--timeout-seconds" in values:
            parsed_timeout, parse_err = _parse_float_value(values["--timeout-seconds"], flag="--timeout-seconds")
            if parse_err is not None or parsed_timeout is None:
                print(parse_err or "invalid number for --timeout-seconds", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            timeout_seconds = parsed_timeout

        try:
            status_request = api.ModalTrainStatusRequest(
                call_id=values.get("--call-id"),
                timeout_seconds=timeout_seconds,
                state_path=values.get("--state-path"),
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.cloud_modal_train_status(status_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(payload.model_dump_json())
        return 0

    if args[0] == "logs":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--call-id", "--state-path", "--lines"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

        lines = 200
        if "--lines" in values:
            parsed_lines, parse_err = _parse_int_value(values["--lines"], flag="--lines")
            if parse_err is not None or parsed_lines is None:
                print(parse_err or "invalid integer for --lines", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            lines = parsed_lines

        try:
            logs_request = api.ModalTrainLogsRequest(
                call_id=values.get("--call-id"),
                lines=lines,
                state_path=values.get("--state-path"),
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.cloud_modal_train_logs(logs_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(payload.model_dump_json())
        return 0

    if args[0] == "cancel":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--call-id", "--state-path"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            cancel_request = api.ModalTrainCancelRequest(
                call_id=values.get("--call-id"),
                state_path=values.get("--state-path"),
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.cloud_modal_train_cancel(cancel_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(payload.model_dump_json())
        return 0

    if args[0] == "pull":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--call-id", "--output-dir", "--timeout-seconds", "--state-path"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

        timeout_seconds = 0.0
        if "--timeout-seconds" in values:
            parsed_timeout, parse_err = _parse_float_value(values["--timeout-seconds"], flag="--timeout-seconds")
            if parse_err is not None or parsed_timeout is None:
                print(parse_err or "invalid number for --timeout-seconds", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            timeout_seconds = parsed_timeout

        try:
            pull_request = api.ModalTrainPullRequest(
                call_id=values.get("--call-id"),
                output_dir=values.get("--output-dir"),
                timeout_seconds=timeout_seconds,
                state_path=values.get("--state-path"),
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.cloud_modal_train_pull(pull_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(payload.model_dump_json())
        return 0

    print(f"unknown arguments: cloud modal train {' '.join(args)}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


def handle_cloud_modal_command(args: Sequence[str]) -> int:
    if not args:
        print(USAGE)
        return 0
    if args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0
    if args[0] == "deploy":
        return _handle_cloud_modal_deploy_command(args[1:])
    if args[0] == "data":
        return handle_cloud_modal_data_command(args[1:])
    if args[0] == "train":
        return _handle_cloud_modal_train_command(args[1:])
    print(f"unknown arguments: cloud modal {' '.join(args)}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


__all__ = ["handle_cloud_modal_command"]
