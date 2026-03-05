"""Cloud EC2 command handlers."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from pydantic import ValidationError

from numereng import api
from numereng.cli.common import (
    _parse_int_value,
    _parse_simple_options,
    _validation_error_message,
)
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError


def _handle_cloud_ec2_package_command(args: Sequence[str]) -> int:
    if not args:
        print(USAGE)
        return 0
    if args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0
    if args[0] != "build-upload":
        print(f"unknown arguments: cloud ec2 package {' '.join(args)}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    values, _, parse_error = _parse_simple_options(
        args[1:],
        value_flags={"--run-id", "--region", "--bucket", "--state-path"},
    )
    if parse_error == "__help__":
        print(USAGE)
        return 0
    if parse_error is not None:
        print(parse_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    try:
        request = api.Ec2PackageBuildUploadRequest(
            run_id=values.get("--run-id"),
            region=values.get("--region"),
            bucket=values.get("--bucket"),
            state_path=values.get("--state-path"),
        )
    except ValidationError as exc:
        print(_validation_error_message(exc), file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    try:
        payload = api.cloud_ec2_package_build_upload(request)
    except PackageError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(payload.model_dump_json())
    return 0


def _handle_cloud_ec2_config_command(args: Sequence[str]) -> int:
    if not args:
        print(USAGE)
        return 0
    if args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0
    if args[0] != "upload":
        print(f"unknown arguments: cloud ec2 config {' '.join(args)}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    values, _, parse_error = _parse_simple_options(
        args[1:],
        value_flags={"--config", "--run-id", "--region", "--bucket", "--state-path"},
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

    try:
        request = api.Ec2ConfigUploadRequest(
            run_id=values.get("--run-id"),
            config_path=config_path,
            region=values.get("--region"),
            bucket=values.get("--bucket"),
            state_path=values.get("--state-path"),
        )
    except ValidationError as exc:
        print(_validation_error_message(exc), file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    try:
        payload = api.cloud_ec2_config_upload(request)
    except PackageError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(payload.model_dump_json())
    return 0


def _handle_cloud_ec2_train_command(args: Sequence[str]) -> int:
    if not args:
        print(USAGE)
        return 0
    if args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    if args[0] == "start":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--instance-id", "--run-id", "--region", "--state-path"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            start_request = api.Ec2TrainStartRequest(
                instance_id=values.get("--instance-id"),
                run_id=values.get("--run-id"),
                region=values.get("--region"),
                state_path=values.get("--state-path"),
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.cloud_ec2_train_start(start_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(payload.model_dump_json())
        return 0

    if args[0] == "poll":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={
                "--instance-id",
                "--run-id",
                "--region",
                "--state-path",
                "--timeout-seconds",
                "--interval-seconds",
            },
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        timeout_seconds = 7200
        interval_seconds = 20
        if "--timeout-seconds" in values:
            parsed_timeout, parse_err = _parse_int_value(values["--timeout-seconds"], flag="--timeout-seconds")
            if parse_err is not None or parsed_timeout is None:
                print(parse_err or "invalid integer for --timeout-seconds", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            timeout_seconds = parsed_timeout
        if "--interval-seconds" in values:
            parsed_interval, parse_err = _parse_int_value(
                values["--interval-seconds"],
                flag="--interval-seconds",
            )
            if parse_err is not None or parsed_interval is None:
                print(parse_err or "invalid integer for --interval-seconds", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            interval_seconds = parsed_interval

        try:
            poll_request = api.Ec2TrainPollRequest(
                instance_id=values.get("--instance-id"),
                run_id=values.get("--run-id"),
                region=values.get("--region"),
                state_path=values.get("--state-path"),
                timeout_seconds=timeout_seconds,
                interval_seconds=interval_seconds,
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.cloud_ec2_train_poll(poll_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(payload.model_dump_json())
        return 0

    print(f"unknown arguments: cloud ec2 train {' '.join(args)}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


def _handle_cloud_ec2_s3_command(args: Sequence[str]) -> int:
    if not args:
        print(USAGE)
        return 0
    if args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    if args[0] == "ls":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--prefix", "--region", "--bucket"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        prefix = values.get("--prefix")
        if prefix is None:
            print("missing required argument: --prefix", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            list_request = api.Ec2S3ListRequest(
                prefix=prefix,
                region=values.get("--region"),
                bucket=values.get("--bucket"),
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.cloud_ec2_s3_list(list_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(payload.model_dump_json())
        return 0

    if args[0] == "cp":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--src", "--dst", "--region", "--bucket"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        src = values.get("--src")
        dst = values.get("--dst")
        if src is None:
            print("missing required argument: --src", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        if dst is None:
            print("missing required argument: --dst", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            copy_request = api.Ec2S3CopyRequest(
                src=src,
                dst=dst,
                region=values.get("--region"),
                bucket=values.get("--bucket"),
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.cloud_ec2_s3_copy(copy_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(payload.model_dump_json())
        return 0

    if args[0] == "rm":
        values, toggles, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--uri", "--region", "--bucket"},
            bool_flags={"--recursive"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        uri = values.get("--uri")
        if uri is None:
            print("missing required argument: --uri", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            remove_request = api.Ec2S3RemoveRequest(
                uri=uri,
                recursive=("--recursive" in toggles),
                region=values.get("--region"),
                bucket=values.get("--bucket"),
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.cloud_ec2_s3_remove(remove_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(payload.model_dump_json())
        return 0

    print(f"unknown arguments: cloud ec2 s3 {' '.join(args)}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


def handle_cloud_ec2_command(args: Sequence[str]) -> int:
    if not args:
        print(USAGE)
        return 0

    if args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    if args[0] == "init-iam":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--region", "--bucket", "--role-name", "--security-group-name"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            init_request = api.Ec2InitIamRequest(
                region=values.get("--region"),
                bucket=values.get("--bucket"),
                role_name=values.get("--role-name"),
                security_group_name=values.get("--security-group-name"),
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.cloud_ec2_init_iam(init_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(payload.model_dump_json())
        return 0

    if args[0] == "setup-data":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--cache-dir", "--data-version", "--region", "--bucket"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            setup_request = api.Ec2SetupDataRequest(
                cache_dir=values.get("--cache-dir"),
                data_version=values.get("--data-version", "v5.2"),
                region=values.get("--region"),
                bucket=values.get("--bucket"),
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.cloud_ec2_setup_data(setup_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(payload.model_dump_json())
        return 0

    if args[0] == "provision":
        values, toggles, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--tier", "--run-id", "--region", "--bucket", "--data-version", "--state-path"},
            bool_flags={"--spot", "--on-demand"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        if "--spot" in toggles and "--on-demand" in toggles:
            print("cannot use both --spot and --on-demand", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        use_spot = True
        if "--on-demand" in toggles:
            use_spot = False
        if "--spot" in toggles:
            use_spot = True

        try:
            provision_request = api.Ec2ProvisionRequest(
                tier=values.get("--tier", "large"),
                run_id=values.get("--run-id"),
                region=values.get("--region"),
                bucket=values.get("--bucket"),
                data_version=values.get("--data-version", "v5.2"),
                use_spot=use_spot,
                state_path=values.get("--state-path"),
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.cloud_ec2_provision(provision_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(payload.model_dump_json())
        return 0

    if args[0] == "package":
        return _handle_cloud_ec2_package_command(args[1:])

    if args[0] == "config":
        return _handle_cloud_ec2_config_command(args[1:])

    if args[0] == "push":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={
                "--instance-id",
                "--run-id",
                "--region",
                "--bucket",
                "--data-version",
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
        try:
            push_request = api.Ec2PushRequest(
                instance_id=values.get("--instance-id"),
                run_id=values.get("--run-id"),
                region=values.get("--region"),
                bucket=values.get("--bucket"),
                data_version=values.get("--data-version"),
                state_path=values.get("--state-path"),
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.cloud_ec2_push(push_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(payload.model_dump_json())
        return 0

    if args[0] == "install":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--instance-id", "--run-id", "--region", "--state-path"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            install_request = api.Ec2InstallRequest(
                instance_id=values.get("--instance-id"),
                run_id=values.get("--run-id"),
                region=values.get("--region"),
                state_path=values.get("--state-path"),
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.cloud_ec2_install(install_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(payload.model_dump_json())
        return 0

    if args[0] == "train":
        return _handle_cloud_ec2_train_command(args[1:])

    if args[0] == "logs":
        values, toggles, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--instance-id", "--region", "--state-path", "--lines"},
            bool_flags={"--follow"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        lines = 100
        if "--lines" in values:
            parsed_lines, parse_err = _parse_int_value(values["--lines"], flag="--lines")
            if parse_err is not None or parsed_lines is None:
                print(parse_err or "invalid integer for --lines", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            lines = parsed_lines

        try:
            logs_request = api.Ec2LogsRequest(
                instance_id=values.get("--instance-id"),
                region=values.get("--region"),
                state_path=values.get("--state-path"),
                lines=lines,
                follow=("--follow" in toggles),
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.cloud_ec2_logs(logs_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(payload.model_dump_json())
        return 0

    if args[0] == "pull":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={
                "--instance-id",
                "--run-id",
                "--output-dir",
                "--region",
                "--bucket",
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
        try:
            pull_request = api.Ec2PullRequest(
                instance_id=values.get("--instance-id"),
                run_id=values.get("--run-id"),
                output_dir=values.get("--output-dir"),
                region=values.get("--region"),
                bucket=values.get("--bucket"),
                state_path=values.get("--state-path"),
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.cloud_ec2_pull(pull_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(payload.model_dump_json())
        return 0

    if args[0] == "terminate":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--instance-id", "--region", "--state-path"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            terminate_request = api.Ec2TerminateRequest(
                instance_id=values.get("--instance-id"),
                region=values.get("--region"),
                state_path=values.get("--state-path"),
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.cloud_ec2_terminate(terminate_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(payload.model_dump_json())
        return 0

    if args[0] == "status":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--run-id", "--region", "--state-path"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            status_request = api.Ec2StatusRequest(
                run_id=values.get("--run-id"),
                region=values.get("--region"),
                state_path=values.get("--state-path"),
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.cloud_ec2_status(status_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(payload.model_dump_json())
        return 0

    if args[0] == "s3":
        return _handle_cloud_ec2_s3_command(args[1:])

    print(f"unknown arguments: cloud ec2 {' '.join(args)}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


__all__ = ["handle_cloud_ec2_command"]
