"""`numereng ensemble study freeze|run|finalize|status` command handlers (P3).

Thin dispatch over the combination-study api. Freeze/run take an immutable
config path; finalize/status address a study by id. Exit codes: 0 success,
1 domain failure (PackageError), 2 usage/validation error.

USAGE:
    numereng ensemble study freeze --config freeze.json
    numereng ensemble study run --trials trials.json
    numereng ensemble study finalize --study-id S1 --select trial_a
    numereng ensemble study status --study-id S1
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from pydantic import ValidationError

from numereng import api
from numereng.cli.common import _parse_simple_options, _validation_error_message
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError

_VALUE_FLAGS = {"--config", "--trials", "--study-id", "--select", "--experiment-id", "--format", "--workspace"}
_SUBCOMMANDS = {"freeze", "run", "finalize", "status"}


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #


def handle_ensemble_study(args: Sequence[str]) -> int:
    if not args or args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    subcommand = args[0]
    if subcommand not in _SUBCOMMANDS:
        print(f"unknown arguments: ensemble study {' '.join(args)}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    values, _toggles, parse_error = _parse_simple_options(args[1:], value_flags=_VALUE_FLAGS)
    if parse_error == "__help__":
        print(USAGE)
        return 0
    if parse_error is not None:
        print(parse_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    output_format, format_error = _resolve_format(values.get("--format", "json"))
    if format_error is not None:
        print(format_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    handlers = {
        "freeze": _run_freeze,
        "run": _run_run,
        "finalize": _run_finalize,
        "status": _run_status,
    }
    return handlers[subcommand](values, output_format)


# --------------------------------------------------------------------------- #
# Subcommand handlers
# --------------------------------------------------------------------------- #


def _run_freeze(values: dict[str, str], output_format: str) -> int:
    config_path = values.get("--config")
    if config_path is None:
        return _missing("--config")
    return _call(
        lambda: api.study_freeze(
            api.StudyFreezeRequest(workspace_root=values.get("--workspace", "."), config_path=config_path)
        ),
        output_format,
        _print_freeze,
    )


def _run_run(values: dict[str, str], output_format: str) -> int:
    trials_path = values.get("--trials")
    if trials_path is None:
        return _missing("--trials")
    return _call(
        lambda: api.study_run(
            api.StudyRunRequest(
                workspace_root=values.get("--workspace", "."),
                trials_path=trials_path,
                experiment_id=values.get("--experiment-id"),
            )
        ),
        output_format,
        _print_run,
    )


def _run_finalize(values: dict[str, str], output_format: str) -> int:
    study_id = values.get("--study-id")
    select = values.get("--select")
    if study_id is None:
        return _missing("--study-id")
    if select is None:
        return _missing("--select")
    return _call(
        lambda: api.study_finalize(
            api.StudyFinalizeRequest(
                workspace_root=values.get("--workspace", "."),
                study_id=study_id,
                select=select,
                experiment_id=values.get("--experiment-id"),
            )
        ),
        output_format,
        _print_finalize,
    )


def _run_status(values: dict[str, str], output_format: str) -> int:
    study_id = values.get("--study-id")
    if study_id is None:
        return _missing("--study-id")
    return _call(
        lambda: api.study_status(
            api.StudyStatusRequest(
                workspace_root=values.get("--workspace", "."),
                study_id=study_id,
                experiment_id=values.get("--experiment-id"),
            )
        ),
        output_format,
        _print_status,
    )


# --------------------------------------------------------------------------- #
# Shared execution + printers
# --------------------------------------------------------------------------- #


def _call(invoke, output_format: str, table_printer) -> int:
    try:
        payload = invoke()
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
        table_printer(payload)
    return 0


def _missing(flag: str) -> int:
    print(f"missing required argument: {flag}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


def _resolve_format(value: str) -> tuple[str | None, str | None]:
    if value not in {"table", "json"}:
        return None, "invalid value for --format: expected table|json"
    return value, None


def _print_freeze(payload: api.StudyFreezeResponse) -> None:
    print(f"study_id: {payload.study_id}")
    print(f"study_dir: {payload.study_dir}")
    print(f"frozen: {payload.frozen}")
    print(f"members: {payload.n_members} | lanes: {payload.n_lanes} | search_folds: {payload.n_search_folds}")
    print(f"holdout_n_eras: {payload.holdout_n_eras}")
    print(f"surface_id: {payload.surface_id or 'none'}")
    print(f"holdout_fingerprint: {payload.holdout_fingerprint}")
    print(f"exploratory: {payload.exploratory}")


def _print_run(payload: api.StudyRunResponse) -> None:
    print(f"study_id: {payload.study_id}")
    print(f"executed: {payload.executed} | skipped: {payload.skipped} | superseded: {payload.superseded}")
    print(f"trial_cap: {payload.trial_cap}")
    print(f"ledger_path: {payload.ledger_path}")
    for trial in payload.trials:
        pooled = "n/a" if trial.pooled_search_bmc is None else f"{trial.pooled_search_bmc:.6f}"
        diff = "n/a" if trial.diff_mean is None else f"{trial.diff_mean:.6f}"
        prob = "n/a" if trial.diff_prob_positive is None else f"{trial.diff_prob_positive:.3f}"
        print(f"- {trial.trial_id} | pooled={pooled} | diff={diff} | P(diff>0)={prob} | status={trial.status}")


def _print_finalize(payload: api.StudyFinalizeResponse) -> None:
    holdout = "n/a" if payload.holdout_bmc is None else f"{payload.holdout_bmc:.6f}"
    base = "n/a" if payload.baseline_holdout_bmc is None else f"{payload.baseline_holdout_bmc:.6f}"
    degr = "n/a" if payload.degradation is None else f"{payload.degradation:.6f}"
    print(f"study_id: {payload.study_id}")
    print(f"selected_trial: {payload.selected_trial} | is_baseline: {payload.is_baseline}")
    print(f"holdout_bmc: {holdout} | baseline_holdout_bmc: {base}")
    print(f"degradation_vs_search: {degr}")
    print(f"sealed: {payload.sealed}")
    print(f"artifacts_dir: {payload.artifacts_dir}")


def _print_status(payload: api.StudyStatusResponse) -> None:
    print(f"study_id: {payload.study_id}")
    print(f"study_dir: {payload.study_dir}")
    print(f"frozen: {payload.frozen} | sealed: {payload.sealed}")
    print(f"trials_executed: {payload.trials_executed} / {payload.trial_cap}")
    print(f"selected_trial: {payload.selected_trial or 'none'}")


__all__ = ["handle_ensemble_study"]
