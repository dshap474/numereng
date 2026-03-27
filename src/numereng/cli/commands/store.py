"""Store command handlers."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from pydantic import ValidationError

from numereng import api
from numereng.cli.common import _parse_simple_options, _validation_error_message
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError


def handle_store_command(args: Sequence[str]) -> int:
    if not args:
        print(USAGE)
        return 0

    if args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    if args[0] == "init":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--store-root"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            init_payload = api.store_init(api.StoreInitRequest(store_root=values.get("--store-root", ".numereng")))
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(init_payload.model_dump_json())
        return 0

    if args[0] == "index":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--run-id", "--store-root"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

        run_id = values.get("--run-id")
        if run_id is None:
            print("missing required argument: --run-id", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

        try:
            index_payload = api.store_index_run(
                api.StoreIndexRequest(
                    run_id=run_id,
                    store_root=values.get("--store-root", ".numereng"),
                )
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(index_payload.model_dump_json())
        return 0

    if args[0] == "rebuild":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--store-root"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            rebuild_payload = api.store_rebuild(
                api.StoreRebuildRequest(store_root=values.get("--store-root", ".numereng"))
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(rebuild_payload.model_dump_json())
        return 0

    if args[0] == "doctor":
        values, toggles, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--store-root"},
            bool_flags={"--fix-strays"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            doctor_payload = api.store_doctor(
                api.StoreDoctorRequest(
                    store_root=values.get("--store-root", ".numereng"),
                    fix_strays="--fix-strays" in toggles,
                )
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(doctor_payload.model_dump_json())
        return 0

    if args[0] == "repair-run-lifecycles":
        values, toggles, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--run-id", "--store-root"},
            bool_flags={"--all"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.store_repair_run_lifecycles(
                api.StoreRunLifecycleRepairRequest(
                    store_root=values.get("--store-root", ".numereng"),
                    run_id=values.get("--run-id"),
                    active_only="--all" not in toggles,
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

    if args[0] == "materialize-viz-artifacts":
        values, toggles, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--store-root", "--kind", "--run-id", "--experiment-id"},
            bool_flags={"--all"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        if values.get("--kind") is None:
            print("missing required argument: --kind", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.store_materialize_viz_artifacts(
                api.StoreMaterializeVizArtifactsRequest(
                    store_root=values.get("--store-root", ".numereng"),
                    kind=values["--kind"],
                    run_id=values.get("--run-id"),
                    experiment_id=values.get("--experiment-id"),
                    all="--all" in toggles,
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

    print(f"unknown arguments: store {' '.join(args)}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


__all__ = ["handle_store_command"]
