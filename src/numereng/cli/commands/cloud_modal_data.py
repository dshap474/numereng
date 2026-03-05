"""Cloud Modal data subcommand handlers."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from pydantic import ValidationError

from numereng import api
from numereng.cli.common import _parse_simple_options, _validation_error_message
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError


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


def handle_cloud_modal_data_command(args: Sequence[str]) -> int:
    if not args:
        print(USAGE)
        return 0
    if args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    if args[0] == "sync":
        values, toggles, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--config", "--volume-name", "--metadata", "--state-path"},
            bool_flags={"--force", "--no-create-if-missing"},
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

        volume_name = values.get("--volume-name")
        if volume_name is None:
            print("missing required argument: --volume-name", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

        metadata: dict[str, str] = {}
        if "--metadata" in values:
            parsed_metadata, metadata_error = _parse_metadata_value(values["--metadata"])
            if metadata_error is not None or parsed_metadata is None:
                print(metadata_error or "invalid value for --metadata", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            metadata = parsed_metadata

        try:
            request = api.ModalDataSyncRequest(
                config_path=config_path,
                volume_name=volume_name,
                create_if_missing="--no-create-if-missing" not in toggles,
                force="--force" in toggles,
                metadata=metadata,
                state_path=values.get("--state-path"),
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.cloud_modal_data_sync(request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(payload.model_dump_json())
        return 0

    print(f"unknown arguments: cloud modal data {' '.join(args)}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


__all__ = ["handle_cloud_modal_data_command"]
