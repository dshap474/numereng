"""Experiment command handlers."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from pydantic import ValidationError

from numereng import api
from numereng.cli.common import (
    _parse_experiment_status_value,
    _parse_int_value,
    _parse_simple_options,
    _parse_training_profile_value,
    _validation_error_message,
)
from numereng.cli.usage import USAGE
from numereng.features.telemetry import bind_launch_metadata
from numereng.platform.errors import PackageError


def _parse_experiment_output_format(value: str) -> tuple[str | None, str | None]:
    if value not in {"table", "json"}:
        return None, "invalid value for --format: expected table|json"
    return value, None


def _print_experiment_list_table(payload: api.ExperimentListResponse) -> None:
    if not payload.experiments:
        print("No experiments found")
        return

    header = f"{'ID':<32} {'Status':<10} {'Runs':<5} {'Champion':<12} {'Updated'}"
    print(header)
    print("-" * len(header))
    for item in payload.experiments:
        champion = item.champion_run_id or "-"
        print(f"{item.experiment_id:<32} {item.status:<10} {len(item.runs):<5} {champion:<12} {item.updated_at}")


def _print_experiment_details_table(payload: api.ExperimentResponse) -> None:
    print(f"experiment_id: {payload.experiment_id}")
    print(f"name: {payload.name}")
    print(f"status: {payload.status}")
    print(f"hypothesis: {payload.hypothesis or 'n/a'}")
    print(f"tags: {', '.join(payload.tags) if payload.tags else 'none'}")
    print(f"created_at: {payload.created_at}")
    print(f"updated_at: {payload.updated_at}")
    print(f"champion_run_id: {payload.champion_run_id or 'none'}")
    print(f"runs: {len(payload.runs)}")
    for run_id in payload.runs:
        print(f"  - {run_id}")


def _print_experiment_report_table(payload: api.ExperimentReportResponse) -> None:
    if not payload.rows:
        print("No experiment runs found")
        return

    print(
        f"experiment={payload.experiment_id} metric={payload.metric} "
        f"total_runs={payload.total_runs} champion={payload.champion_run_id or 'none'}"
    )
    header = (
        f"{'Run ID':<14} {'Metric':>10} {'CORR':>10} {'MMC':>10} {'CWMM':>10} "
        f"{'BMC':>10} {'BMC200':>10} {'Champion':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in payload.rows:
        metric_value = "n/a" if row.metric_value is None else f"{row.metric_value:.6f}"
        corr_mean = "n/a" if row.corr_mean is None else f"{row.corr_mean:.6f}"
        mmc_mean = "n/a" if row.mmc_mean is None else f"{row.mmc_mean:.6f}"
        cwmm_mean = "n/a" if row.cwmm_mean is None else f"{row.cwmm_mean:.6f}"
        bmc_mean = "n/a" if row.bmc_mean is None else f"{row.bmc_mean:.6f}"
        bmc_200_mean = "n/a" if row.bmc_last_200_eras_mean is None else f"{row.bmc_last_200_eras_mean:.6f}"
        champion = "yes" if row.is_champion else "no"
        print(
            f"{row.run_id:<14} {metric_value:>10} {corr_mean:>10} {mmc_mean:>10} "
            f"{cwmm_mean:>10} {bmc_mean:>10} {bmc_200_mean:>10} {champion:>9}"
        )


def handle_experiment_command(args: Sequence[str]) -> int:
    if not args:
        print(USAGE)
        return 0

    if args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    if args[0] == "create":
        values, toggles, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--id", "--name", "--hypothesis", "--tags", "--store-root"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        experiment_id = values.get("--id")
        if experiment_id is None:
            print("missing required argument: --id", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        tags = [item.strip() for item in values.get("--tags", "").split(",") if item.strip()]
        try:
            create_payload = api.experiment_create(
                api.ExperimentCreateRequest(
                    experiment_id=experiment_id,
                    name=values.get("--name"),
                    hypothesis=values.get("--hypothesis"),
                    tags=tags,
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
        print(create_payload.model_dump_json())
        return 0

    if args[0] == "list":
        values, toggles, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--status", "--format", "--store-root"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        output_format = "table"
        if "--format" in values:
            parsed_format, format_err = _parse_experiment_output_format(values["--format"])
            if format_err is not None or parsed_format is None:
                print(format_err or "invalid value for --format", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            output_format = parsed_format
        status: api.ExperimentStatus | None = None
        if "--status" in values:
            parsed_status, status_err = _parse_experiment_status_value(values["--status"])
            if status_err is not None or parsed_status is None:
                print(status_err or "invalid value for --status", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            status = parsed_status
        try:
            list_payload = api.experiment_list(
                api.ExperimentListRequest(
                    status=status,
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
        if output_format == "json":
            print(list_payload.model_dump_json())
        else:
            _print_experiment_list_table(list_payload)
        return 0

    if args[0] == "details":
        values, toggles, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--id", "--format", "--store-root"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        experiment_id = values.get("--id")
        if experiment_id is None:
            print("missing required argument: --id", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        output_format = "table"
        if "--format" in values:
            parsed_format, format_err = _parse_experiment_output_format(values["--format"])
            if format_err is not None or parsed_format is None:
                print(format_err or "invalid value for --format", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            output_format = parsed_format
        try:
            details_payload = api.experiment_get(
                api.ExperimentGetRequest(
                    experiment_id=experiment_id,
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
        if output_format == "json":
            print(details_payload.model_dump_json())
        else:
            _print_experiment_details_table(details_payload)
        return 0

    if args[0] in {"archive", "unarchive"}:
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--id", "--store-root"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        experiment_id = values.get("--id")
        if experiment_id is None:
            print("missing required argument: --id", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            if args[0] == "archive":
                payload = api.experiment_archive(
                    api.ExperimentArchiveRequest(
                        experiment_id=experiment_id,
                        store_root=values.get("--store-root", ".numereng"),
                    )
                )
            else:
                payload = api.experiment_unarchive(
                    api.ExperimentArchiveRequest(
                        experiment_id=experiment_id,
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
        print(payload.model_dump_json())
        return 0

    if args[0] == "train":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={
                "--id",
                "--config",
                "--output-dir",
                "--profile",
                "--engine-mode",
                "--window-size-eras",
                "--embargo-eras",
                "--store-root",
            },
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        experiment_id = values.get("--id")
        config_path = values.get("--config")
        if experiment_id is None:
            print("missing required argument: --id", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
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
            parsed_mode, parse_mode_err = _parse_training_profile_value(values["--profile"])
            if parse_mode_err is not None:
                print(parse_mode_err, file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            profile = parsed_mode

        try:
            with bind_launch_metadata(source="cli.experiment.train", operation_type="run", job_type="run"):
                train_payload = api.experiment_train(
                    api.ExperimentTrainRequest(
                        experiment_id=experiment_id,
                        config_path=config_path,
                        output_dir=values.get("--output-dir"),
                        profile=profile,
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
        print(train_payload.model_dump_json())
        return 0

    if args[0] == "score-round":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--id", "--round", "--stage", "--store-root"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        experiment_id = values.get("--id")
        round_label = values.get("--round")
        stage = values.get("--stage")
        if experiment_id is None:
            print("missing required argument: --id", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        if round_label is None:
            print("missing required argument: --round", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        if stage is None:
            print("missing required argument: --stage", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        if stage not in {"post_training_core", "post_training_full"}:
            print(
                "invalid value for --stage: expected post_training_core|post_training_full",
                file=sys.stderr,
            )
            print(USAGE, file=sys.stderr)
            return 2
        try:
            with bind_launch_metadata(source="cli.experiment.score-round", operation_type="run", job_type="run"):
                payload = api.experiment_score_round(
                    api.ExperimentScoreRoundRequest(
                        experiment_id=experiment_id,
                        round=round_label,
                        stage=stage,
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
        print(payload.model_dump_json())
        return 0

    if args[0] == "promote":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--id", "--run", "--metric", "--store-root"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        experiment_id = values.get("--id")
        if experiment_id is None:
            print("missing required argument: --id", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            promote_payload = api.experiment_promote(
                api.ExperimentPromoteRequest(
                    experiment_id=experiment_id,
                    run_id=values.get("--run"),
                    metric=values.get("--metric", "bmc_last_200_eras.mean"),
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
        print(promote_payload.model_dump_json())
        return 0

    if args[0] == "report":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--id", "--metric", "--limit", "--format", "--store-root"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        experiment_id = values.get("--id")
        if experiment_id is None:
            print("missing required argument: --id", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        output_format = "table"
        if "--format" in values:
            parsed_format, format_err = _parse_experiment_output_format(values["--format"])
            if format_err is not None or parsed_format is None:
                print(format_err or "invalid value for --format", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            output_format = parsed_format

        limit = 10
        if "--limit" in values:
            parsed_limit, limit_err = _parse_int_value(values["--limit"], flag="--limit")
            if limit_err is not None or parsed_limit is None:
                print(limit_err or "invalid integer for --limit", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            limit = parsed_limit

        try:
            report_payload = api.experiment_report(
                api.ExperimentReportRequest(
                    experiment_id=experiment_id,
                    metric=values.get("--metric", "bmc_last_200_eras.mean"),
                    limit=limit,
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
        if output_format == "json":
            print(report_payload.model_dump_json())
        else:
            _print_experiment_report_table(report_payload)
        return 0

    if args[0] == "pack":
        values, _, parse_error = _parse_simple_options(
            args[1:],
            value_flags={"--id", "--store-root"},
        )
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        experiment_id = values.get("--id")
        if experiment_id is None:
            print("missing required argument: --id", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        try:
            pack_payload = api.experiment_pack(
                api.ExperimentPackRequest(
                    experiment_id=experiment_id,
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
        print(pack_payload.model_dump_json())
        return 0

    print(f"unknown arguments: experiment {' '.join(args)}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


__all__ = ["handle_experiment_command"]
