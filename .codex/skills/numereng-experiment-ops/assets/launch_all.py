#!/usr/bin/env python3
"""Template launcher for one experiment run_plan with script-owned round scoring."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

EXPERIMENT_ID = "<YYYY-MM-DD_slug>"
ROUND_RE = re.compile(r"^(r\d+)_")
VALID_SCORE_STAGES = ("post_training_core", "post_training_full")


@dataclass(frozen=True)
class PlanRow:
    index: int
    config_path: Path
    round_label: str


def find_repo_root(start: Path) -> Path:
    checked: set[Path] = set()
    for root in (start.resolve(), Path.cwd().resolve()):
        for candidate in (root, *root.parents):
            if candidate in checked:
                continue
            checked.add(candidate)
            if (candidate / "pyproject.toml").is_file():
                return candidate
    raise SystemExit(f"repo_root_not_found_from:{start}")


SCRIPT_PATH = Path(__file__).resolve()
RUN_SCRIPTS_DIR = SCRIPT_PATH.parent
EXPERIMENT_DIR = RUN_SCRIPTS_DIR.parent
REPO_ROOT = find_repo_root(RUN_SCRIPTS_DIR)
RUN_PLAN_PATH = EXPERIMENT_DIR / "run_plan.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one experiment run_plan and batch-score each completed round.")
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based row index in run_plan.csv to start from.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Optional 1-based row index in run_plan.csv to stop at.",
    )
    parser.add_argument(
        "--score-stage",
        choices=VALID_SCORE_STAGES,
        default="post_training_core",
        help="Round batch scoring stage to materialize after the last planned row in each round.",
    )
    return parser.parse_args()


def resolve_config_path(raw_config_path: str) -> Path:
    path = Path(raw_config_path)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


def resolve_round_label(raw_round: str | None, *, config_path: Path) -> str:
    if raw_round is not None and raw_round.strip():
        stripped = raw_round.strip()
        if re.fullmatch(r"r\d+", stripped) is None:
            raise SystemExit(f"invalid_round_label:{stripped}:{config_path.name}")
        return stripped
    match = ROUND_RE.match(config_path.stem)
    if match is None:
        raise SystemExit(f"round_label_missing:{config_path.name}")
    return match.group(1)


def load_run_plan() -> list[PlanRow]:
    if not RUN_PLAN_PATH.is_file():
        raise SystemExit(f"run_plan_missing:{RUN_PLAN_PATH}")
    with RUN_PLAN_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[PlanRow] = []
        for index, row in enumerate(reader, start=1):
            raw_config_path = row.get("config_path")
            if not isinstance(raw_config_path, str) or not raw_config_path.strip():
                raise SystemExit(f"run_plan_config_path_missing:index={index}")
            config_path = resolve_config_path(raw_config_path)
            if not config_path.is_file():
                raise SystemExit(f"config_missing:{config_path}")
            round_label = resolve_round_label(row.get("round"), config_path=config_path)
            rows.append(PlanRow(index=index, config_path=config_path, round_label=round_label))
    if not rows:
        raise SystemExit(f"run_plan_empty:{RUN_PLAN_PATH}")
    return rows


def run_command(command: list[str]) -> None:
    print(f"+ {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    args = parse_args()
    rows = load_run_plan()
    total = len(rows)

    if args.start_index < 1 or args.start_index > total:
        raise SystemExit(f"start_index_out_of_range:{args.start_index}:1..{total}")
    if args.end_index is not None and (args.end_index < args.start_index or args.end_index > total):
        raise SystemExit(f"end_index_out_of_range:{args.end_index}:{args.start_index}..{total}")

    selected_rows = rows[args.start_index - 1 : args.end_index]
    last_index_by_round: dict[str, int] = {}
    for row in rows:
        last_index_by_round[row.round_label] = row.index

    print(f"Experiment: {EXPERIMENT_ID}")
    print(f"Run plan: {RUN_PLAN_PATH}")
    print(f"Selected rows: {args.start_index}..{args.end_index or total}")
    print(f"Round score stage: {args.score_stage}")

    scored_rounds: set[str] = set()
    for row in selected_rows:
        print()
        print(f">>> [{row.index}/{total}] Training: {row.config_path}")
        run_command(
            [
                "uv",
                "run",
                "numereng",
                "experiment",
                "train",
                "--id",
                EXPERIMENT_ID,
                "--config",
                str(row.config_path),
                "--post-training-scoring",
                "none",
            ]
        )
        if row.index == last_index_by_round[row.round_label] and row.round_label not in scored_rounds:
            print(f">>> [{row.index}/{total}] Batch scoring round {row.round_label}: {args.score_stage}")
            run_command(
                [
                    "uv",
                    "run",
                    "numereng",
                    "experiment",
                    "score-round",
                    "--id",
                    EXPERIMENT_ID,
                    "--round",
                    row.round_label,
                    "--stage",
                    args.score_stage,
                ]
            )
            scored_rounds.add(row.round_label)

    print()
    print(f"Completed rows {args.start_index}..{args.end_index or total}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
