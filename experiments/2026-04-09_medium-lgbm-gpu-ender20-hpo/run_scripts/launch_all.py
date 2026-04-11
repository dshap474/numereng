#!/usr/bin/env python3
"""Thin wrapper over the source-owned `numereng experiment run-plan` command."""

from __future__ import annotations

import argparse
from pathlib import Path

import numereng.api as api


def find_workspace_root(start: Path) -> Path:
    checked: set[Path] = set()
    for root in (start.resolve(), Path.cwd().resolve()):
        for candidate in (root, *root.parents):
            if candidate in checked:
                continue
            checked.add(candidate)
            if (candidate / ".numereng").is_dir() or (candidate / "experiments").is_dir():
                return candidate
    return start.resolve().parents[2]


SCRIPT_PATH = Path(__file__).resolve()
RUN_SCRIPTS_DIR = SCRIPT_PATH.parent
WORKSPACE_ROOT = find_workspace_root(RUN_SCRIPTS_DIR)
EXPERIMENT_ID = "2026-04-09_medium-lgbm-gpu-ender20-hpo"
VALID_SCORE_STAGES = ("post_training_core", "post_training_full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the experiment run_plan through the source-owned numereng API."
    )
    parser.add_argument("--start-index", type=int, default=1, help="1-based run_plan row index to start from.")
    parser.add_argument("--end-index", type=int, default=None, help="Optional 1-based run_plan row index to stop at.")
    parser.add_argument(
        "--score-stage",
        choices=VALID_SCORE_STAGES,
        default="post_training_core",
        help="Round scoring stage to materialize after the last planned config in a round.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume an existing run-plan state window.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = api.experiment_run_plan(
        api.ExperimentRunPlanRequest(
            experiment_id=EXPERIMENT_ID,
            start_index=args.start_index,
            end_index=args.end_index,
            score_stage=args.score_stage,
            resume=args.resume,
            workspace_root=str(WORKSPACE_ROOT),
        )
    )
    print(result.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
