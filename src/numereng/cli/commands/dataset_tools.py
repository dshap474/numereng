"""Dataset-tools command handlers."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from pydantic import ValidationError

from numereng import api
from numereng.cli.common import _parse_simple_options, _validation_error_message
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError

_BUILD_VALUE_FLAGS = {
    "--data-version",
    "--data-dir",
    "--downsample-eras-step",
    "--downsample-eras-offset",
}
_BUILD_BOOL_FLAGS = {"--rebuild", "--skip-downsample"}


def _handle_build_downsampled_full_subcommand(args: Sequence[str]) -> int:
    values, toggles, parse_error = _parse_simple_options(
        args,
        value_flags=_BUILD_VALUE_FLAGS,
        bool_flags=_BUILD_BOOL_FLAGS,
    )
    if parse_error == "__help__":
        print(USAGE)
        return 0
    if parse_error is not None:
        print(parse_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    request_kwargs: dict[str, object] = {}
    if "--data-version" in values:
        request_kwargs["data_version"] = values["--data-version"]
    if "--data-dir" in values:
        request_kwargs["data_dir"] = values["--data-dir"]
    if "--downsample-eras-step" in values:
        request_kwargs["downsample_eras_step"] = values["--downsample-eras-step"]
    if "--downsample-eras-offset" in values:
        request_kwargs["downsample_eras_offset"] = values["--downsample-eras-offset"]
    if "--skip-downsample" in toggles:
        request_kwargs["skip_downsample"] = True
    if "--rebuild" in toggles:
        request_kwargs["rebuild"] = True

    try:
        payload = api.dataset_tools_build_downsampled_full(
            api.DatasetToolsBuildDownsampleRequest(**request_kwargs)  # type: ignore[arg-type]
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


def handle_dataset_tools_command(args: Sequence[str]) -> int:
    if not args:
        print(USAGE)
        return 0

    if args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    if args[0] in {"quantize", "quantize-lossless"}:
        print(
            "dataset-tools quantization commands are removed; use dataset-tools build-full-datasets",
            file=sys.stderr,
        )
        print(USAGE, file=sys.stderr)
        return 2

    if args[0] in {"build-full-datasets", "build-downsampled-full"}:
        return _handle_build_downsampled_full_subcommand(args[1:])

    print(f"unknown arguments: dataset-tools {' '.join(args)}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


__all__ = ["handle_dataset_tools_command"]
