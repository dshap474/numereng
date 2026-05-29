"""Submitted-model snapshot and calibration command handlers."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any, Literal

from numereng import api
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError

_Format = Literal["table", "json"]


def handle_submissions_command(args: Sequence[str]) -> int:
    """Dispatch `numereng submissions ...` commands."""

    if not args or args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    if args[0] == "refresh":
        parsed, parse_error = _parse_refresh_options(args[1:])
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            return _usage_error(parse_error)
        try:
            response = api.submissions_refresh(
                api.SubmissionRefreshRequest(
                    workspace_root=str(parsed["workspace"]),
                    model_names=list(parsed["models"]),
                    dry_run=bool(parsed["dry_run"]),
                )
            )
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return _print_refresh(response, output_format=parsed["format"])

    if args[0] == "calibration":
        return _handle_calibration_command(args[1:])

    return _usage_error(f"unknown arguments: submissions {' '.join(args)}")


def _handle_calibration_command(args: Sequence[str]) -> int:
    if not args or args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    if args[0] == "materialize":
        parsed, parse_error = _parse_workspace_format_options(args[1:])
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            return _usage_error(parse_error)
        try:
            response = api.submissions_calibration_materialize(
                api.SubmissionCalibrationMaterializeRequest(workspace_root=str(parsed["workspace"]))
            )
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return _print_materialize(response, output_format=parsed["format"])

    if args[0] == "report":
        parsed, parse_error = _parse_report_options(args[1:])
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            return _usage_error(parse_error)
        try:
            response = api.submissions_calibration_report(
                api.SubmissionCalibrationReportRequest(
                    workspace_root=str(parsed["workspace"]),
                    resolved_only=bool(parsed["resolved_only"]),
                )
            )
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return _print_report(response, output_format=parsed["format"])

    if args[0] == "update":
        parsed, parse_error = _parse_update_options(args[1:])
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            return _usage_error(parse_error)
        try:
            response = api.submissions_calibration_update(
                api.SubmissionCalibrationUpdateRequest(
                    workspace_root=str(parsed["workspace"]),
                    model_names=list(parsed["models"]),
                    dry_run=bool(parsed["dry_run"]),
                    resolved_only=bool(parsed["resolved_only"]),
                )
            )
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return _print_update(response, output_format=parsed["format"])

    return _usage_error(f"unknown arguments: submissions calibration {' '.join(args)}")


def _parse_refresh_options(argv: Sequence[str]) -> tuple[dict[str, Any], str | None]:
    parsed = {"workspace": ".", "models": [], "dry_run": False, "format": "table"}
    return _parse_common(argv, parsed, allow_models=True, allow_dry_run=True, allow_resolved_only=False)


def _parse_workspace_format_options(argv: Sequence[str]) -> tuple[dict[str, Any], str | None]:
    parsed = {"workspace": ".", "format": "table"}
    return _parse_common(argv, parsed, allow_models=False, allow_dry_run=False, allow_resolved_only=False)


def _parse_report_options(argv: Sequence[str]) -> tuple[dict[str, Any], str | None]:
    parsed = {"workspace": ".", "format": "table", "resolved_only": False}
    return _parse_common(argv, parsed, allow_models=False, allow_dry_run=False, allow_resolved_only=True)


def _parse_update_options(argv: Sequence[str]) -> tuple[dict[str, Any], str | None]:
    parsed = {"workspace": ".", "models": [], "dry_run": False, "format": "table", "resolved_only": False}
    return _parse_common(argv, parsed, allow_models=True, allow_dry_run=True, allow_resolved_only=True)


def _parse_common(
    argv: Sequence[str],
    parsed: dict[str, Any],
    *,
    allow_models: bool,
    allow_dry_run: bool,
    allow_resolved_only: bool,
) -> tuple[dict[str, Any], str | None]:
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg in {"-h", "--help"}:
            return parsed, "__help__"
        if arg == "--dry-run" and allow_dry_run:
            parsed["dry_run"] = True
            idx += 1
            continue
        if arg == "--resolved-only" and allow_resolved_only:
            parsed["resolved_only"] = True
            idx += 1
            continue
        if arg in {"--workspace", "--format"} or (arg == "--model" and allow_models):
            if idx + 1 >= len(argv):
                return parsed, f"missing value for {arg}"
            value = argv[idx + 1]
            if arg == "--workspace":
                parsed["workspace"] = value
            elif arg == "--model":
                parsed["models"].append(value)
            else:
                if value not in {"table", "json"}:
                    return parsed, "invalid value for --format: expected table or json"
                parsed["format"] = value
            idx += 2
            continue
        return parsed, f"unknown arguments: {arg}"
    return parsed, None


def _print_refresh(response: api.SubmissionRefreshResponse, *, output_format: _Format) -> int:
    if output_format == "json":
        print(response.model_dump_json(indent=2))
        return 0
    mode = "dry-run" if response.dry_run else "refresh"
    print(
        f"{mode}: refreshed={response.refreshed_count} skipped={response.skipped_count} "
        f"workspace={response.workspace_root}"
    )
    for item in response.items:
        status = "skipped" if item.skipped else "would_refresh" if response.dry_run else "refreshed"
        warning = f" warning={item.warning}" if item.warning else ""
        print(
            f"{status} {item.model_name}: rounds={item.round_count} scored={item.scored_round_count} "
            f"resolved={item.resolved_round_count} latest_scored={_format_value(item.latest_scored_round)}"
            f"{warning}"
        )
    return 0


def _print_materialize(
    response: api.SubmissionCalibrationMaterializeResponse,
    *,
    output_format: _Format,
) -> int:
    if output_format == "json":
        print(response.model_dump_json(indent=2))
        return 0
    print(f"materialized rows={response.row_count} scored={response.scored_row_count} models={response.model_count}")
    print(f"rows: {response.rows_path}")
    print(f"report: {response.report_path}")
    for warning in response.warnings:
        print(f"warning: {warning}")
    return 0


def _print_report(response: api.SubmissionCalibrationReportResponse, *, output_format: _Format) -> int:
    if output_format == "json":
        print(response.model_dump_json(indent=2))
        return 0
    report = response.report
    scope = _report_scope(report, resolved_only=response.scope == "resolved_only")
    stats = scope.get("correlations", {}).get("local_bmc200_mean", {}).get("live_mmc20", {})
    print(
        f"calibration scope={response.scope} rows={scope.get('row_count', 0)} "
        f"models={scope.get('model_count', 0)} artifact_rows={response.row_count}"
    )
    print(
        "local_bmc200_mean vs live_mmc20: "
        f"n={stats.get('n', 0)} r={_format_value(stats.get('pearson_r'), digits=3)} "
        f"rho={_format_value(stats.get('spearman_rho'), digits=3)} "
        f"r2={_format_value(stats.get('r2'), digits=3)}"
    )
    rank_rows = scope.get("rank_deltas") if isinstance(scope.get("rank_deltas"), list) else []
    for item in rank_rows[:10]:
        print(
            f"{item.get('model_name')}: local_rank={_format_rank(item.get('local_rank'))} "
            f"live_rank={_format_rank(item.get('live_rank'))} "
            f"delta={_format_value(item.get('rank_delta'), digits=0)} "
            f"live_mmc20={_format_value(item.get('live_mmc20_mean'))}"
        )
    return 0


def _print_update(response: api.SubmissionCalibrationUpdateResponse, *, output_format: _Format) -> int:
    if output_format == "json":
        print(response.model_dump_json(indent=2))
        return 0
    _print_refresh(response.refresh, output_format="table")
    _print_materialize(response.materialize, output_format="table")
    _print_report(response.report, output_format="table")
    return 0


def _report_scope(report: dict[str, Any], *, resolved_only: bool) -> dict[str, Any]:
    if "correlations" in report:
        return report
    scopes = report.get("scopes") if isinstance(report.get("scopes"), dict) else {}
    key = "resolved_only" if resolved_only else "all_scored"
    value = scopes.get(key)
    return value if isinstance(value, dict) else {}


def _format_value(value: Any, *, digits: int = 4) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return "n/a"
    if isinstance(value, float) and not value == value:
        return "n/a"
    if digits == 0:
        return str(int(value))
    return f"{float(value):.{digits}f}"


def _format_rank(value: Any) -> str:
    return f"#{value}" if isinstance(value, int) else "n/a"


def _usage_error(message: str) -> int:
    print(message, file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


__all__ = ["handle_submissions_command"]
