"""CLI helpers for `numereng serve ...`."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from pydantic import ValidationError

from numereng import api
from numereng.cli.common import _parse_simple_options, _validation_error_message
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError

OutputFormat = Literal["table", "json"]


def handle_serve_command(args: Sequence[str]) -> int:
    if not args or args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    if args[0] == "package":
        return _handle_package_command(args[1:])
    if args[0] == "live":
        return _handle_live_command(args[1:])
    if args[0] == "pickle":
        return _handle_pickle_command(args[1:])

    print(f"unknown serve command: {args[0]}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


def _handle_package_command(args: Sequence[str]) -> int:
    if not args:
        print("missing serve package subcommand", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    if args[0] == "create":
        return _handle_package_create(args[1:])
    if args[0] == "inspect":
        return _handle_package_inspect(args[1:])
    if args[0] == "list":
        return _handle_package_list(args[1:])
    print(f"unknown serve package command: {args[0]}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


def _handle_live_command(args: Sequence[str]) -> int:
    if not args:
        print("missing serve live subcommand", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    if args[0] == "build":
        return _handle_live_build(args[1:])
    if args[0] == "submit":
        return _handle_live_submit(args[1:])
    print(f"unknown serve live command: {args[0]}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


def _handle_pickle_command(args: Sequence[str]) -> int:
    if not args:
        print("missing serve pickle subcommand", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    if args[0] == "build":
        return _handle_pickle_build(args[1:])
    if args[0] == "upload":
        return _handle_pickle_upload(args[1:])
    print(f"unknown serve pickle command: {args[0]}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


def _handle_package_create(args: Sequence[str]) -> int:
    values, _, parse_error = _parse_simple_options(
        args,
        value_flags={
            "--experiment-id",
            "--package-id",
            "--components",
            "--data-version",
            "--blend-rule",
            "--neutralization",
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

    required = {"--experiment-id", "--package-id", "--components"}
    missing = [flag for flag in required if flag not in values]
    if missing:
        print(f"missing required argument: {missing[0]}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    components, components_err = _parse_json_value(values["--components"], expect="list", default=[])
    if components is None:
        print(components_err or "invalid value for --components", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    blend_rule_payload, blend_rule_err = _parse_json_value(values.get("--blend-rule"), expect="dict", default={})
    if blend_rule_payload is None:
        print(blend_rule_err or "invalid value for --blend-rule", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    neutralization_payload, neutralization_err = _parse_json_value(
        values.get("--neutralization"),
        expect="dict",
        default=None,
    )
    if neutralization_payload is None and neutralization_err is not None:
        print(neutralization_err, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    try:
        response = api.serve_package_create(
            api.ServePackageCreateRequest(
                experiment_id=values["--experiment-id"],
                package_id=values["--package-id"],
                data_version=values.get("--data-version", "v5.2"),
                components=[api.ServeComponentRequest.model_validate(item) for item in components],
                blend_rule=api.ServeBlendRuleRequest.model_validate(blend_rule_payload),
                neutralization=None
                if neutralization_payload is None
                else api.ServeNeutralizationRequest.model_validate(neutralization_payload),
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

    print(response.model_dump_json())
    return 0


def _handle_package_list(args: Sequence[str]) -> int:
    values, _, parse_error = _parse_simple_options(
        args,
        value_flags={"--experiment-id", "--format", "--workspace"},
    )
    if parse_error == "__help__":
        print(USAGE)
        return 0
    if parse_error is not None:
        print(parse_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    output_format = _parse_output_format(values.get("--format", "table"))
    if output_format is None:
        print("invalid value for --format: expected table|json", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    try:
        response = api.serve_package_list(
            api.ServePackageListRequest(
                experiment_id=values.get("--experiment-id"),
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
        print(response.model_dump_json())
    else:
        for item in response.packages:
            print(f"{item.experiment_id}/{item.package_id}\t{item.status}\t{item.package_path}")
    return 0


def _handle_package_inspect(args: Sequence[str]) -> int:
    return _handle_simple_request(
        args,
        request_cls=api.ServePackageInspectRequest,
        call=api.serve_package_inspect,
        required={"--experiment-id", "--package-id"},
    )


def _handle_live_build(args: Sequence[str]) -> int:
    return _handle_simple_request(
        args,
        request_cls=api.ServeLiveBuildRequest,
        call=api.serve_live_build,
        required={"--experiment-id", "--package-id"},
    )


def _handle_live_submit(args: Sequence[str]) -> int:
    return _handle_simple_request(
        args,
        request_cls=api.ServeLiveSubmitRequest,
        call=api.serve_live_submit,
        required={"--experiment-id", "--package-id", "--model-name"},
    )


def _handle_pickle_build(args: Sequence[str]) -> int:
    return _handle_simple_request(
        args,
        request_cls=api.ServePickleBuildRequest,
        call=api.serve_pickle_build,
        required={"--experiment-id", "--package-id"},
    )


def _handle_pickle_upload(args: Sequence[str]) -> int:
    return _handle_simple_request(
        args,
        request_cls=api.ServePickleUploadRequest,
        call=api.serve_pickle_upload,
        required={"--experiment-id", "--package-id", "--model-name"},
        optional={"--data-version", "--docker-image"},
    )


def _handle_simple_request(
    args: Sequence[str],
    *,
    request_cls: type[object],
    call: object,
    required: set[str],
    optional: set[str] | None = None,
) -> int:
    value_flags = {"--experiment-id", "--package-id", "--model-name", "--data-version", "--docker-image", "--workspace"}
    values, _, parse_error = _parse_simple_options(args, value_flags=value_flags)
    if parse_error == "__help__":
        print(USAGE)
        return 0
    if parse_error is not None:
        print(parse_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    missing = [flag for flag in required if flag not in values]
    if missing:
        print(f"missing required argument: {missing[0]}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    payload = {
        "experiment_id": values.get("--experiment-id"),
        "package_id": values.get("--package-id"),
        "model_name": values.get("--model-name"),
        "data_version": values.get("--data-version"),
        "docker_image": values.get("--docker-image"),
        "workspace_root": values.get("--workspace", "."),
    }
    if optional is None:
        optional = set()
    allowed_keys = {
        "--experiment-id": "experiment_id",
        "--package-id": "package_id",
        "--model-name": "model_name",
        "--data-version": "data_version",
        "--docker-image": "docker_image",
    }
    request_payload = {
        field: payload[field]
        for flag, field in allowed_keys.items()
        if flag in values or flag in required or flag in optional
    }
    request_payload["workspace_root"] = values.get("--workspace", ".")
    try:
        request = cast(object, request_cls).model_validate(request_payload)  # type: ignore[union-attr]
        response = cast(object, call)(request)  # type: ignore[misc]
    except ValidationError as exc:
        print(_validation_error_message(exc), file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    except PackageError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(cast(object, response).model_dump_json())  # type: ignore[union-attr]
    return 0


def _parse_output_format(value: str) -> OutputFormat | None:
    if value not in {"table", "json"}:
        return None
    return cast(OutputFormat, value)


def _parse_json_value(
    raw: str | None,
    *,
    expect: Literal["dict", "list"],
    default: object,
) -> tuple[dict[str, object] | list[object] | None, str | None]:
    if raw is None:
        return cast(dict[str, object] | list[object] | None, default), None
    candidate = raw.strip()
    path = Path(candidate).expanduser()
    text = path.read_text(encoding="utf-8") if path.is_file() else candidate
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None, "invalid JSON payload"
    if expect == "dict" and not isinstance(payload, dict):
        return None, "expected JSON object"
    if expect == "list" and not isinstance(payload, list):
        return None, "expected JSON list"
    return cast(dict[str, object] | list[object], payload), None


__all__ = ["handle_serve_command"]
