"""Cloud AWS managed command handlers."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from pydantic import ValidationError

from numereng import api
from numereng.cli.common import (
    CloudAwsBackend,
    _parse_cloud_backend_value,
    _parse_int_value,
    _parse_simple_options,
    _validation_error_message,
)
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError


def _handle_cloud_aws_image_command(args: Sequence[str]) -> int:
    if not args:
        print(USAGE)
        return 0
    if args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0
    if args[0] != "build-push":
        print(f"unknown arguments: cloud aws image {' '.join(args)}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    values, _, parse_error = _parse_simple_options(
        args[1:],
        value_flags={
            "--run-id",
            "--region",
            "--bucket",
            "--repository",
            "--image-tag",
            "--context-dir",
            "--dockerfile",
            "--runtime-profile",
            "--platform",
            "--state-path",
            "--workspace",
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
        request = api.AwsImageBuildPushRequest(
            run_id=values.get("--run-id"),
            region=values.get("--region"),
            bucket=values.get("--bucket"),
            repository=values.get("--repository"),
            image_tag=values.get("--image-tag"),
            context_dir=values.get("--context-dir", "."),
            dockerfile=values.get("--dockerfile"),
            runtime_profile=values.get("--runtime-profile", "standard"),
            platform=values.get("--platform"),
            state_path=values.get("--state-path"),
            workspace_root=values.get("--workspace", "."),
        )
    except ValidationError as exc:
        print(_validation_error_message(exc), file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    try:
        payload = api.cloud_aws_image_build_push(request)
    except PackageError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(payload.model_dump_json())
    return 0


def _handle_cloud_aws_train_command(args: Sequence[str]) -> int:
    if not args:
        print(USAGE)
        return 0
    if args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    if args[0] == "submit":
        values, toggles, parse_error = _parse_simple_options(
            args[1:],
            value_flags={
                "--run-id",
                "--backend",
                "--region",
                "--bucket",
                "--config",
                "--config-s3-uri",
                "--image-uri",
                "--runtime-profile",
                "--role-arn",
                "--instance-type",
                "--instance-count",
                "--volume-size-gb",
                "--max-runtime-seconds",
                "--max-wait-seconds",
                "--checkpoint-s3-uri",
                "--output-s3-uri",
                "--batch-job-queue",
                "--batch-job-definition",
                "--state-path",
                "--workspace",
            },
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

        submit_backend: CloudAwsBackend = "sagemaker"
        if "--backend" in values:
            parsed_backend, backend_err = _parse_cloud_backend_value(values["--backend"])
            if backend_err is not None or parsed_backend is None:
                print(backend_err or "invalid value for --backend", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            submit_backend = parsed_backend

        instance_count = 1
        volume_size_gb = 100
        max_runtime_seconds = 14400
        max_wait_seconds: int | None = None
        if "--instance-count" in values:
            parsed, err = _parse_int_value(values["--instance-count"], flag="--instance-count")
            if err is not None or parsed is None:
                print(err or "invalid integer for --instance-count", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            instance_count = parsed
        if "--volume-size-gb" in values:
            parsed, err = _parse_int_value(values["--volume-size-gb"], flag="--volume-size-gb")
            if err is not None or parsed is None:
                print(err or "invalid integer for --volume-size-gb", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            volume_size_gb = parsed
        if "--max-runtime-seconds" in values:
            parsed, err = _parse_int_value(values["--max-runtime-seconds"], flag="--max-runtime-seconds")
            if err is not None or parsed is None:
                print(err or "invalid integer for --max-runtime-seconds", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            max_runtime_seconds = parsed
        if "--max-wait-seconds" in values:
            parsed, err = _parse_int_value(values["--max-wait-seconds"], flag="--max-wait-seconds")
            if err is not None or parsed is None:
                print(err or "invalid integer for --max-wait-seconds", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            max_wait_seconds = parsed

        use_spot = True
        if "--on-demand" in toggles:
            use_spot = False
        if "--spot" in toggles:
            use_spot = True

        try:
            submit_request = api.AwsTrainSubmitRequest(
                run_id=values.get("--run-id"),
                backend=submit_backend,
                region=values.get("--region"),
                bucket=values.get("--bucket"),
                config_path=values.get("--config"),
                config_s3_uri=values.get("--config-s3-uri"),
                image_uri=values.get("--image-uri"),
                runtime_profile=values.get("--runtime-profile"),
                role_arn=values.get("--role-arn"),
                instance_type=values.get("--instance-type", "ml.m5.2xlarge"),
                instance_count=instance_count,
                volume_size_gb=volume_size_gb,
                max_runtime_seconds=max_runtime_seconds,
                max_wait_seconds=max_wait_seconds,
                use_spot=use_spot,
                checkpoint_s3_uri=values.get("--checkpoint-s3-uri"),
                output_s3_uri=values.get("--output-s3-uri"),
                batch_job_queue=values.get("--batch-job-queue"),
                batch_job_definition=values.get("--batch-job-definition"),
                state_path=values.get("--state-path"),
                workspace_root=values.get("--workspace", "."),
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

        try:
            payload = api.cloud_aws_train_submit(submit_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(payload.model_dump_json())
        return 0

    if args[0] == "status":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={
                "--backend",
                "--run-id",
                "--training-job-name",
                "--batch-job-id",
                "--region",
                "--state-path",
                "--workspace",
            },
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        status_backend: CloudAwsBackend | None = None
        if "--backend" in values:
            parsed_backend, backend_err = _parse_cloud_backend_value(values["--backend"])
            if backend_err is not None:
                print(backend_err, file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            status_backend = parsed_backend
        try:
            status_request = api.AwsTrainStatusRequest(
                backend=status_backend,
                run_id=values.get("--run-id"),
                training_job_name=values.get("--training-job-name"),
                batch_job_id=values.get("--batch-job-id"),
                region=values.get("--region"),
                state_path=values.get("--state-path"),
                workspace_root=values.get("--workspace", "."),
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.cloud_aws_train_status(status_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(payload.model_dump_json())
        return 0

    if args[0] == "logs":
        values, toggles, parse_error = _parse_simple_options(
            args[1:],
            value_flags={
                "--backend",
                "--run-id",
                "--training-job-name",
                "--batch-job-id",
                "--region",
                "--state-path",
                "--workspace",
                "--lines",
            },
            bool_flags={"--follow"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

        logs_backend: CloudAwsBackend | None = None
        if "--backend" in values:
            parsed_backend, backend_err = _parse_cloud_backend_value(values["--backend"])
            if backend_err is not None:
                print(backend_err, file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            logs_backend = parsed_backend

        lines = 100
        if "--lines" in values:
            parsed_lines, parse_err = _parse_int_value(values["--lines"], flag="--lines")
            if parse_err is not None or parsed_lines is None:
                print(parse_err or "invalid integer for --lines", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            lines = parsed_lines

        try:
            logs_request = api.AwsTrainLogsRequest(
                backend=logs_backend,
                run_id=values.get("--run-id"),
                training_job_name=values.get("--training-job-name"),
                batch_job_id=values.get("--batch-job-id"),
                region=values.get("--region"),
                lines=lines,
                follow=("--follow" in toggles),
                state_path=values.get("--state-path"),
                workspace_root=values.get("--workspace", "."),
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.cloud_aws_train_logs(logs_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(payload.model_dump_json())
        return 0

    if args[0] == "cancel":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={
                "--backend",
                "--run-id",
                "--training-job-name",
                "--batch-job-id",
                "--region",
                "--state-path",
                "--workspace",
            },
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        cancel_backend: CloudAwsBackend | None = None
        if "--backend" in values:
            parsed_backend, backend_err = _parse_cloud_backend_value(values["--backend"])
            if backend_err is not None:
                print(backend_err, file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            cancel_backend = parsed_backend
        try:
            cancel_request = api.AwsTrainCancelRequest(
                backend=cancel_backend,
                run_id=values.get("--run-id"),
                training_job_name=values.get("--training-job-name"),
                batch_job_id=values.get("--batch-job-id"),
                region=values.get("--region"),
                state_path=values.get("--state-path"),
                workspace_root=values.get("--workspace", "."),
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.cloud_aws_train_cancel(cancel_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(payload.model_dump_json())
        return 0

    if args[0] == "pull":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={
                "--run-id",
                "--region",
                "--bucket",
                "--output-s3-uri",
                "--output-dir",
                "--state-path",
                "--workspace",
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
            pull_request = api.AwsTrainPullRequest(
                run_id=values.get("--run-id"),
                region=values.get("--region"),
                bucket=values.get("--bucket"),
                output_s3_uri=values.get("--output-s3-uri"),
                output_dir=values.get("--output-dir"),
                state_path=values.get("--state-path"),
                workspace_root=values.get("--workspace", "."),
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.cloud_aws_train_pull(pull_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(payload.model_dump_json())
        return 0

    if args[0] == "extract":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={
                "--run-id",
                "--region",
                "--bucket",
                "--output-dir",
                "--state-path",
                "--workspace",
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
            extract_request = api.AwsTrainExtractRequest(
                run_id=values.get("--run-id"),
                region=values.get("--region"),
                bucket=values.get("--bucket"),
                output_dir=values.get("--output-dir"),
                state_path=values.get("--state-path"),
                workspace_root=values.get("--workspace", "."),
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.cloud_aws_train_extract(extract_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(payload.model_dump_json())
        return 0

    print(f"unknown arguments: cloud aws train {' '.join(args)}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


def handle_cloud_aws_command(args: Sequence[str]) -> int:
    if not args:
        print(USAGE)
        return 0
    if args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0
    if args[0] == "image":
        return _handle_cloud_aws_image_command(args[1:])
    if args[0] == "train":
        return _handle_cloud_aws_train_command(args[1:])
    print(f"unknown arguments: cloud aws {' '.join(args)}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


__all__ = ["handle_cloud_aws_command"]
