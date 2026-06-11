"""Store command handlers."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from numereng import api
from numereng.cli.common import _parse_simple_options, _validation_error_message
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError


def _parse_output_format(value: str) -> tuple[str | None, str | None]:
    if value not in {"table", "json"}:
        return None, "invalid value for --format: expected table|json"
    return value, None


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
            value_flags={"--workspace"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            init_payload = api.store_init(api.StoreInitRequest(workspace_root=values.get("--workspace", ".")))
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
            value_flags={"--run-id", "--workspace"},
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
        print(index_payload.model_dump_json())
        return 0

    if args[0] == "rebuild":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--workspace"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            rebuild_payload = api.store_rebuild(api.StoreRebuildRequest(workspace_root=values.get("--workspace", ".")))
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
            value_flags={"--workspace"},
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
                    workspace_root=values.get("--workspace", "."),
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

    if args[0] == "backfill-run-execution":
        values, toggles, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--run-id", "--workspace"},
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
            payload = api.store_backfill_run_execution(
                api.StoreRunExecutionBackfillRequest(
                    workspace_root=values.get("--workspace", "."),
                    run_id=values.get("--run-id"),
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

    if args[0] == "repair-run-lifecycles":
        values, toggles, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--run-id", "--workspace"},
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
                    workspace_root=values.get("--workspace", "."),
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
            value_flags={"--workspace", "--kind", "--run-id", "--experiment-id"},
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
                    workspace_root=values.get("--workspace", "."),
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

    if args[0] == "prune-predictions":
        parsed, parse_error = _parse_prune_predictions_options(args[1:])
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            payload = api.store_prune_predictions(
                api.StorePrunePredictionsRequest(
                    workspace_root=str(parsed["workspace"]),
                    run_ids=list(parsed["run_ids"]),
                    experiment_id=parsed["experiment_id"],
                    all=bool(parsed["all"]),
                    apply=bool(parsed["apply"]),
                )
            )
        except ValidationError as exc:
            print(_validation_error_message(exc), file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        _print_prune_predictions(payload, output_format=str(parsed["format"]))
        return 0

    print(f"unknown arguments: store {' '.join(args)}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


def _parse_prune_predictions_options(argv: Sequence[str]) -> tuple[dict[str, Any], str | None]:
    parsed: dict[str, Any] = {
        "workspace": ".",
        "run_ids": [],
        "experiment_id": None,
        "all": False,
        "apply": False,
        "format": "table",
    }
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg in {"-h", "--help"}:
            return parsed, "__help__"
        if arg == "--apply":
            parsed["apply"] = True
            idx += 1
            continue
        if arg == "--all":
            parsed["all"] = True
            idx += 1
            continue
        if arg in {"--run-id", "--experiment-id", "--workspace", "--format"}:
            if idx + 1 >= len(argv):
                return parsed, f"missing value for {arg}"
            value = argv[idx + 1]
            if arg == "--run-id":
                parsed["run_ids"].append(value)
            elif arg == "--experiment-id":
                parsed["experiment_id"] = value
            elif arg == "--workspace":
                parsed["workspace"] = value
            else:
                output_format, format_error = _parse_output_format(value)
                if format_error is not None or output_format is None:
                    return parsed, format_error or "invalid value for --format"
                parsed["format"] = output_format
            idx += 2
            continue
        return parsed, f"unknown arguments: {arg}"

    scope_flags = int(bool(parsed["run_ids"])) + int(parsed["experiment_id"] is not None) + int(bool(parsed["all"]))
    if scope_flags != 1:
        return parsed, "exactly one of --run-id, --experiment-id, or --all is required"
    return parsed, None


def _print_prune_predictions(payload: api.StorePrunePredictionsResponse, *, output_format: str) -> None:
    if output_format == "json":
        print(payload.model_dump_json())
        return

    mode = "dry-run" if payload.dry_run else "apply"
    print(
        f"{mode}: candidates={payload.candidate_count} pruned={payload.pruned_count} "
        f"excluded={payload.excluded_count} reclaimable_bytes={payload.reclaimable_bytes} "
        f"reclaimed_bytes={payload.reclaimed_bytes}"
    )
    if payload.pruned:
        label = "would-prune" if payload.dry_run else "pruned"
        print(f"{label}:")
        print(f"{'run_id':<14} {'bytes':>12} predictions_dir")
        for item in payload.pruned:
            print(f"{item.run_id:<14} {item.bytes:>12} {item.predictions_dir}")
    if payload.excluded:
        print("excluded:")
        print(f"{'run_id':<14} reason")
        for item in payload.excluded:
            print(f"{item.run_id:<14} {item.reason}")


__all__ = ["handle_store_command"]
