"""CLI helper for `numereng ensemble select ...`."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from pydantic import ValidationError

from numereng import api
from numereng.cli.common import _parse_int_value, _parse_simple_options, _validation_error_message
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError

OutputFormat = Literal["table", "json"]


def handle_ensemble_select(args: Sequence[str]) -> int:
    values, toggles, parse_error = _parse_simple_options(
        args,
        value_flags={
            "--experiment-id",
            "--source-experiment-ids",
            "--source-rules",
            "--selection-id",
            "--target",
            "--primary-metric",
            "--tie-break-metric",
            "--correlation-threshold",
            "--top-weighted-variants",
            "--weight-step",
            "--required-seed-count",
            "--blend-variants",
            "--weighted-promotion-min-gain",
            "--format",
            "--workspace",
        },
        bool_flags={"--require-full-seed-bundle"},
    )
    if parse_error == "__help__":
        print(USAGE)
        return 0
    if parse_error is not None:
        print(parse_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    experiment_id = values.get("--experiment-id")
    if experiment_id is None:
        print("missing required argument: --experiment-id", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    source_experiment_ids_raw = values.get("--source-experiment-ids")
    if source_experiment_ids_raw is None:
        print("missing required argument: --source-experiment-ids", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    source_rules_raw = values.get("--source-rules")
    if source_rules_raw is None:
        print("missing required argument: --source-rules", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    source_experiment_ids = [item.strip() for item in source_experiment_ids_raw.split(",") if item.strip()]
    source_rules, source_rules_err = _parse_source_rules(source_rules_raw)
    if source_rules is None or source_rules_err is not None:
        print(source_rules_err or "invalid value for --source-rules", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    output_format: OutputFormat = "json"
    if "--format" in values:
        parsed_format, format_err = _parse_output_format(values["--format"])
        if parsed_format is None or format_err is not None:
            print(format_err or "invalid value for --format", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        output_format = parsed_format

    top_weighted_variants = 2
    if "--top-weighted-variants" in values:
        parsed_value, parse_err = _parse_int_value(values["--top-weighted-variants"], flag="--top-weighted-variants")
        if parsed_value is None or parse_err is not None:
            print(parse_err or "invalid integer for --top-weighted-variants", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        top_weighted_variants = parsed_value

    required_seed_count = 1
    if "--required-seed-count" in values:
        parsed_value, parse_err = _parse_int_value(values["--required-seed-count"], flag="--required-seed-count")
        if parsed_value is None or parse_err is not None:
            print(parse_err or "invalid integer for --required-seed-count", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        required_seed_count = parsed_value

    blend_variants = _parse_csv(values.get("--blend-variants")) or [
        "all_surviving",
        "medium_only",
        "small_only",
        "top2_medium_top2_small",
        "top3_overall",
    ]

    try:
        response = api.ensemble_select(
            api.EnsembleSelectRequest(
                experiment_id=experiment_id,
                source_experiment_ids=source_experiment_ids,
                source_rules=source_rules,
                selection_id=values.get("--selection-id"),
                target=values.get("--target", "target_ender_20"),
                primary_metric=values.get("--primary-metric", "bmc_last_200_eras.mean"),
                tie_break_metric=values.get("--tie-break-metric", "bmc.mean"),
                correlation_threshold=float(values.get("--correlation-threshold", "0.85")),
                top_weighted_variants=top_weighted_variants,
                weight_step=float(values.get("--weight-step", "0.05")),
                required_seed_count=required_seed_count,
                require_full_seed_bundle="--require-full-seed-bundle" in toggles,
                blend_variants=blend_variants,
                weighted_promotion_min_gain=float(values.get("--weighted-promotion-min-gain", "0.0005")),
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
        _print_selection_table(response)
    return 0


def _parse_output_format(value: str) -> tuple[OutputFormat | None, str | None]:
    if value not in {"table", "json"}:
        return None, "invalid value for --format: expected table|json"
    return cast(OutputFormat, value), None


def _parse_source_rules(value: str) -> tuple[list[api.EnsembleSelectionSourceRuleRequest] | None, str | None]:
    candidate = value.strip()
    if not candidate:
        return None, "invalid value for --source-rules"
    path = Path(candidate).expanduser()
    raw = path.read_text(encoding="utf-8") if path.is_file() else candidate
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None, "invalid JSON for --source-rules"
    if not isinstance(payload, list):
        return None, "invalid value for --source-rules: expected JSON list"
    try:
        return [api.EnsembleSelectionSourceRuleRequest.model_validate(item) for item in payload], None
    except ValidationError as exc:
        return None, _validation_error_message(exc)


def _parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _print_selection_table(payload: api.EnsembleSelectResponse) -> None:
    print(f"selection_id: {payload.selection_id}")
    print(f"experiment_id: {payload.experiment_id}")
    print(f"status: {payload.status}")
    print(f"artifacts_path: {payload.artifacts_path}")
    print(f"frozen_candidate_count: {payload.frozen_candidate_count}")
    print(f"surviving_candidate_count: {payload.surviving_candidate_count}")
    print(f"winner_blend_id: {payload.winner.blend_id}")
    print(f"winner_selection_mode: {payload.winner.selection_mode}")
    print(f"winner_components: {', '.join(payload.winner.component_ids)}")
    print(f"winner_weights: {', '.join(f'{weight:.2f}' for weight in payload.winner.weights)}")
    print("winner_metrics:")
    for key, value in payload.winner.metrics.items():
        print(f"  - {key}={value}")


__all__ = ["handle_ensemble_select"]
