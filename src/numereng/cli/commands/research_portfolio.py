"""`numereng research portfolio ...` command handlers (P1 status/report, P2 diversity)."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from pydantic import ValidationError

from numereng import api
from numereng.cli.common import _parse_simple_options, _validation_error_message
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError

_VALUE_FLAGS = {"--workspace", "--format", "--lanes"}
_BOOL_FLAGS = {"--write"}


def handle_research_portfolio_command(args: Sequence[str]) -> int:
    if not args or args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    subcommand = args[0]
    if subcommand not in {"status", "report", "diversity"}:
        print(USAGE, file=sys.stderr)
        return 2

    values, toggles, parse_error = _parse_simple_options(
        args[1:],
        value_flags=_VALUE_FLAGS,
        bool_flags=_BOOL_FLAGS,
    )
    if parse_error == "__help__":
        print(USAGE)
        return 0
    if parse_error is not None:
        print(parse_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    output_format, format_error = _resolve_format(values.get("--format", "table"))
    if format_error is not None:
        print(format_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    if subcommand == "diversity":
        return _run_diversity(values, output_format)

    write = subcommand == "report" or "--write" in toggles
    try:
        payload = api.portfolio_status(
            api.PortfolioStatusRequest(
                workspace_root=values.get("--workspace", "."),
                write=write,
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
        print(payload.model_dump_json())
    else:
        _print_status_table(payload)
    return 0


def _run_diversity(values: dict[str, str], output_format: str | None) -> int:
    lanes_value = values.get("--lanes")
    lanes = [item.strip() for item in lanes_value.split(",") if item.strip()] if lanes_value else None
    try:
        payload = api.portfolio_diversity(
            api.PortfolioDiversityRequest(
                workspace_root=values.get("--workspace", "."),
                lanes=lanes,
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
        print(payload.model_dump_json())
    else:
        _print_diversity_table(payload)
    return 0


def _resolve_format(value: str) -> tuple[str | None, str | None]:
    if value not in {"table", "json"}:
        return None, "invalid value for --format: expected table|json"
    return value, None


def _print_status_table(payload: api.PortfolioStatusResponse) -> None:
    print(f"portfolio_present: {payload.portfolio_present}")
    print(f"schema_version: {payload.schema_version}")
    print(f"policy_hash: {payload.policy_hash or 'none'}")
    print(f"registry_path: {payload.registry_path}")
    if payload.report_path is not None:
        print(f"report_path: {payload.report_path}")
    if payload.policy_gaps:
        print(f"policy_gaps: {', '.join(payload.policy_gaps)}")
    print(f"lanes: {len(payload.lanes)}")
    for lane in payload.lanes:
        print(
            f"- {lane.lane_id} | research={lane.research_stage_asserted}"
            f"(seen:{lane.research_stage_evidenced})"
            f" | combination={lane.combination_stage_asserted}"
            f"(seen:{lane.combination_stage_evidenced})"
            f" | surface_match={lane.surface_match}"
        )
        for candidate in lane.candidates:
            mean = "n/a" if candidate.trio_bmc_mean is None else f"{candidate.trio_bmc_mean:.6f}"
            print(
                f"    {candidate.candidate_id} | role={candidate.role}"
                f" | tier={candidate.evidence_tier} | trio={candidate.trio_complete}"
                f" | bmc200_mean={mean}"
            )
        if lane.blockers:
            print(f"    blockers: {', '.join(lane.blockers)}")
    if payload.blockers:
        print(f"blockers: {len(payload.blockers)}")


def _print_diversity_table(payload: api.PortfolioDiversityResponse) -> None:
    blend = "n/a" if payload.blend_bmc_mean is None else f"{payload.blend_bmc_mean:.6f}"
    print(f"report_id: {payload.report_id}")
    print(f"report_dir: {payload.report_dir}")
    print(f"surface_id: {payload.surface_id or 'none'}")
    print(f"diversity_bmc_tolerance: {payload.diversity_bmc_tolerance}")
    print(f"n_eras: {payload.n_eras}")
    print(f"included_lanes: {', '.join(payload.included_lanes)}")
    print(f"blend_bmc_mean: {blend}")
    print(
        "inference:"
        f" block={payload.inference.block_length_eras}"
        f" resamples={payload.inference.n_resamples}"
        f" seed={payload.inference.rng_seed}"
    )
    print(f"members: {len(payload.members)}")
    for member in payload.members:
        trio = "n/a" if member.trio_bmc200 is None else f"{member.trio_bmc200:.6f}"
        print(f"- {member.candidate_id} | lane={member.lane_id} | trio_bmc200={trio}")
    for pair in payload.pairwise:
        rho = "n/a" if pair.spearman_mean is None else f"{pair.spearman_mean:.4f}"
        corr = "n/a" if pair.bmc_series_corr is None else f"{pair.bmc_series_corr:.4f}"
        jdd = "n/a" if pair.joint_drawdown_fraction is None else f"{pair.joint_drawdown_fraction:.4f}"
        print(f"    {pair.left} ~ {pair.right} | spearman_mean={rho} | bmc_corr={corr} | joint_dd={jdd}")
    for loo in payload.leave_one_out:
        diff = "n/a" if loo.mean_diff is None else f"{loo.mean_diff:.6f}"
        prob = "n/a" if loo.prob_positive is None else f"{loo.prob_positive:.3f}"
        print(f"    drop {loo.lane_id} | mean_diff={diff} | prob_positive={prob}")
    if payload.excluded_candidates:
        print(f"excluded: {', '.join(f'{cid}={reason}' for cid, reason in payload.excluded_candidates)}")


__all__ = ["handle_research_portfolio_command"]
