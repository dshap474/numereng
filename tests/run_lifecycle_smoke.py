#!/usr/bin/env python3
"""Repeatable local lifecycle smoke runner for the testing experiment.

Run from the repo root with:

    uv run python tests/run_lifecycle_smoke.py

This script drives the public CLI entrypoints end-to-end and verifies:
- normal completion
- cooperative cancel
- orphan/stale reconciliation
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

POLL_INTERVAL_SECONDS = 1.0


@dataclass(frozen=True)
class CaseSpec:
    name: str
    template_name: str
    expected_status: str
    expected_manifest_status: str
    expected_terminal_reason: str
    reconciled: bool


@dataclass(frozen=True)
class CaseResult:
    name: str
    run_id: str
    status: str
    manifest_status: str
    terminal_reason: str
    runtime_path: Path
    manifest_path: Path


CASE_SPECS: dict[str, CaseSpec] = {
    "complete": CaseSpec(
        name="complete",
        template_name="ender20_small_lgbm_lifecycle_complete_20260324.json",
        expected_status="completed",
        expected_manifest_status="FINISHED",
        expected_terminal_reason="completed",
        reconciled=False,
    ),
    "cancel": CaseSpec(
        name="cancel",
        template_name="ender20_small_lgbm_lifecycle_cancel_20260324.json",
        expected_status="canceled",
        expected_manifest_status="CANCELED",
        expected_terminal_reason="cancel_requested",
        reconciled=False,
    ),
    "stale": CaseSpec(
        name="stale",
        template_name="ender20_small_lgbm_lifecycle_stale_20260324.json",
        expected_status="stale",
        expected_manifest_status="STALE",
        expected_terminal_reason="reconciled_stale",
        reconciled=True,
    ),
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        default=str(repo_root),
        help="Numereng workspace root. Defaults to the repo root.",
    )
    parser.add_argument(
        "--experiment-id",
        default="testing",
        help="Experiment ID to run against. Defaults to 'testing'.",
    )
    parser.add_argument(
        "--cases",
        default="complete,cancel,stale",
        help="Comma-separated subset of cases to run: complete,cancel,stale",
    )
    parser.add_argument(
        "--generated-root",
        default=str(repo_root / ".numereng" / "tmp" / "lifecycle_smoke"),
        help="Directory for generated per-run configs.",
    )
    parser.add_argument(
        "--case-timeout-seconds",
        type=float,
        default=420.0,
        help="Per-case timeout while waiting for terminal state.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=POLL_INTERVAL_SECONDS,
        help="Polling interval for lifecycle state checks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = Path(args.workspace).expanduser().resolve()
    store_root = workspace_root / ".numereng"
    generated_root = Path(args.generated_root).expanduser().resolve()
    template_root = workspace_root / "experiments" / args.experiment_id / "configs"

    selected_cases = [item.strip() for item in args.cases.split(",") if item.strip()]
    if not selected_cases:
        raise SystemExit("No cases selected.")
    unknown = [item for item in selected_cases if item not in CASE_SPECS]
    if unknown:
        raise SystemExit(f"Unknown cases: {', '.join(unknown)}")

    session_token = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    generated_dir = generated_root / session_token
    generated_dir.mkdir(parents=True, exist_ok=True)

    print(f"[smoke] repo_root={repo_root}")
    print(f"[smoke] workspace_root={workspace_root}")
    print(f"[smoke] store_root={store_root}")
    print(f"[smoke] experiment_id={args.experiment_id}")
    print(f"[smoke] generated_dir={generated_dir}")

    cleanup_success = False
    try:
        run_cli(
            ["uv", "run", "numereng", "store", "init", "--workspace", str(workspace_root)],
            cwd=repo_root,
            expect_returncodes={0},
            capture=True,
        )

        results: list[CaseResult] = []
        for index, case_name in enumerate(selected_cases, start=1):
            spec = CASE_SPECS[case_name]
            generated_config = build_generated_config(
                template_root=template_root,
                template_name=spec.template_name,
                generated_dir=generated_dir,
                session_token=session_token,
                seed=1000 + index,
            )
            print(f"[smoke] case={spec.name} config={generated_config}")
            if spec.name == "complete":
                result = run_complete_case(
                    repo_root=repo_root,
                    workspace_root=workspace_root,
                    store_root=store_root,
                    experiment_id=args.experiment_id,
                    config_path=generated_config,
                    spec=spec,
                    timeout_seconds=args.case_timeout_seconds,
                    poll_seconds=args.poll_seconds,
                )
            elif spec.name == "cancel":
                result = run_cancel_case(
                    repo_root=repo_root,
                    workspace_root=workspace_root,
                    store_root=store_root,
                    experiment_id=args.experiment_id,
                    config_path=generated_config,
                    spec=spec,
                    timeout_seconds=args.case_timeout_seconds,
                    poll_seconds=args.poll_seconds,
                )
            else:
                result = run_stale_case(
                    repo_root=repo_root,
                    workspace_root=workspace_root,
                    store_root=store_root,
                    experiment_id=args.experiment_id,
                    config_path=generated_config,
                    spec=spec,
                    timeout_seconds=args.case_timeout_seconds,
                    poll_seconds=args.poll_seconds,
                )
            results.append(result)
            print(
                f"[smoke] case={result.name} run_id={result.run_id} status={result.status} "
                f"manifest_status={result.manifest_status} terminal_reason={result.terminal_reason}"
            )

        print("[smoke] summary")
        for result in results:
            print(
                f"  - {result.name}: run_id={result.run_id} status={result.status} "
                f"manifest={result.manifest_status} runtime={result.runtime_path}"
            )
        cleanup_success = True
        return 0
    finally:
        finalize_generated_session_dir(generated_dir=generated_dir, success=cleanup_success)


def run_complete_case(
    *,
    repo_root: Path,
    workspace_root: Path,
    store_root: Path,
    experiment_id: str,
    config_path: Path,
    spec: CaseSpec,
    timeout_seconds: float,
    poll_seconds: float,
) -> CaseResult:
    process = start_train_process(
        repo_root=repo_root,
        workspace_root=workspace_root,
        store_root=store_root,
        experiment_id=experiment_id,
        config_path=config_path,
    )
    try:
        running_row = wait_for_lifecycle_row(
            store_root=store_root,
            config_path=config_path,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
            predicate=lambda row: row["status"] in {"starting", "running", "completed"},
        )
        run_id = str(running_row["run_id"])
        assert_runtime_snapshot_exists(store_root=store_root, run_id=run_id)
        completed = process.communicate(timeout=timeout_seconds)
        if process.returncode != 0:
            raise AssertionError(
                f"complete case failed: returncode={process.returncode} stdout={completed[0]!r} stderr={completed[1]!r}"
            )
        payload = parse_last_json_object(completed[0])
        if payload.get("run_id") != run_id:
            raise AssertionError(f"unexpected run_id in complete case output: {payload}")
        wait_for_lifecycle_row(
            store_root=store_root,
            run_id=run_id,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
            predicate=lambda row: row["status"] == spec.expected_status,
        )
        return assert_terminal_artifacts(store_root=store_root, run_id=run_id, spec=spec)
    finally:
        stop_process(process)


def run_cancel_case(
    *,
    repo_root: Path,
    workspace_root: Path,
    store_root: Path,
    experiment_id: str,
    config_path: Path,
    spec: CaseSpec,
    timeout_seconds: float,
    poll_seconds: float,
) -> CaseResult:
    process = start_train_process(
        repo_root=repo_root,
        workspace_root=workspace_root,
        store_root=store_root,
        experiment_id=experiment_id,
        config_path=config_path,
    )
    try:
        running_row = wait_for_lifecycle_row(
            store_root=store_root,
            config_path=config_path,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
            predicate=lambda row: row["status"] in {"starting", "running"} and row["pid"] is not None,
        )
        run_id = str(running_row["run_id"])
        cancel_completed = run_cli(
            ["uv", "run", "numereng", "run", "cancel", "--run-id", run_id, "--workspace", str(workspace_root)],
            cwd=repo_root,
            expect_returncodes={0},
            capture=True,
        )
        cancel_payload = parse_last_json_object(cancel_completed.stdout)
        if not cancel_payload.get("accepted"):
            raise AssertionError(f"cancel request was not accepted: {cancel_payload}")
        wait_for_lifecycle_row(
            store_root=store_root,
            run_id=run_id,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
            predicate=lambda row: bool(row["cancel_requested"]),
        )
        completed = process.communicate(timeout=timeout_seconds)
        if process.returncode == 0:
            raise AssertionError("cancel case unexpectedly exited with code 0")
        combined = (completed[0] or "") + (completed[1] or "")
        if "training_run_canceled" not in combined:
            raise AssertionError(f"cancel case missing cancellation marker: {combined!r}")
        wait_for_lifecycle_row(
            store_root=store_root,
            run_id=run_id,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
            predicate=lambda row: row["status"] == spec.expected_status,
        )
        return assert_terminal_artifacts(store_root=store_root, run_id=run_id, spec=spec)
    finally:
        stop_process(process)


def run_stale_case(
    *,
    repo_root: Path,
    workspace_root: Path,
    store_root: Path,
    experiment_id: str,
    config_path: Path,
    spec: CaseSpec,
    timeout_seconds: float,
    poll_seconds: float,
) -> CaseResult:
    process = start_train_process(
        repo_root=repo_root,
        workspace_root=workspace_root,
        store_root=store_root,
        experiment_id=experiment_id,
        config_path=config_path,
    )
    try:
        running_row = wait_for_lifecycle_row(
            store_root=store_root,
            config_path=config_path,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
            predicate=lambda row: row["status"] in {"starting", "running"} and row["pid"] is not None,
        )
        run_id = str(running_row["run_id"])
        worker_pid = int(running_row["pid"])
        os.kill(worker_pid, signal.SIGKILL)
        try:
            process.communicate(timeout=30.0)
        except subprocess.TimeoutExpired:
            stop_process(process)
        stranded = wait_for_lifecycle_row(
            store_root=store_root,
            run_id=run_id,
            timeout_seconds=30.0,
            poll_seconds=poll_seconds,
            predicate=lambda row: row["status"] == "running",
        )
        if stranded["finished_at"] is not None:
            raise AssertionError(f"stale case unexpectedly terminalized before repair: {dict(stranded)}")
        repair_completed = run_cli(
            [
                "uv",
                "run",
                "numereng",
                "store",
                "repair-run-lifecycles",
                "--run-id",
                run_id,
                "--workspace",
                str(workspace_root),
            ],
            cwd=repo_root,
            expect_returncodes={0},
            capture=True,
        )
        repair_payload = parse_last_json_object(repair_completed.stdout)
        if run_id not in repair_payload.get("run_ids", []):
            raise AssertionError(f"repair response did not include run_id {run_id}: {repair_payload}")
        if repair_payload.get("reconciled_stale_count", 0) < 1:
            raise AssertionError(f"repair response did not reconcile stale row: {repair_payload}")
        wait_for_lifecycle_row(
            store_root=store_root,
            run_id=run_id,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
            predicate=lambda row: row["status"] == spec.expected_status,
        )
        return assert_terminal_artifacts(store_root=store_root, run_id=run_id, spec=spec)
    finally:
        stop_process(process)


def build_generated_config(
    *,
    template_root: Path,
    template_name: str,
    generated_dir: Path,
    session_token: str,
    seed: int,
) -> Path:
    template_path = template_root / template_name
    payload = json.loads(template_path.read_text(encoding="utf-8"))
    model = expect_mapping(payload.get("model"), key="model")
    params = expect_mapping(model.get("params"), key="model.params")
    params["random_state"] = seed
    output = expect_mapping(payload.setdefault("output", {}), key="output")
    predictions_name = str(output.get("predictions_name") or template_path.stem)
    output["predictions_name"] = f"{predictions_name}_{session_token}_{seed}"
    generated_path = generated_dir / f"{template_path.stem}_{session_token}_{seed}.json"
    generated_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return generated_path.resolve()


def start_train_process(
    *,
    repo_root: Path,
    workspace_root: Path,
    store_root: Path,
    experiment_id: str,
    config_path: Path,
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            "uv",
            "run",
            "numereng",
            "experiment",
            "train",
            "--id",
            experiment_id,
            "--config",
            str(config_path),
            "--workspace",
            str(workspace_root),
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def assert_runtime_snapshot_exists(*, store_root: Path, run_id: str) -> None:
    runtime_path = store_root / "runs" / run_id / "runtime.json"
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if runtime_path.is_file():
            return
        time.sleep(0.5)
    raise AssertionError(f"runtime snapshot not found for run {run_id}: {runtime_path}")


def assert_terminal_artifacts(*, store_root: Path, run_id: str, spec: CaseSpec) -> CaseResult:
    import numereng.api as api

    lifecycle = api.get_run_lifecycle(api.RunLifecycleRequest(run_id=run_id, workspace_root=str(store_root.parent)))
    if lifecycle.status != spec.expected_status:
        raise AssertionError(f"unexpected API lifecycle status for {run_id}: {lifecycle.status}")
    if lifecycle.terminal_reason != spec.expected_terminal_reason:
        raise AssertionError(f"unexpected terminal reason for {run_id}: {lifecycle.terminal_reason}")
    if lifecycle.reconciled != spec.reconciled:
        raise AssertionError(f"unexpected reconciled flag for {run_id}: {lifecycle.reconciled}")

    run_dir = store_root / "runs" / run_id
    runtime_path = run_dir / "runtime.json"
    manifest_path = run_dir / "run.json"
    runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    runtime_status = str(runtime_payload.get("status"))
    if runtime_status != spec.expected_status:
        raise AssertionError(f"unexpected runtime status for {run_id}: {runtime_status}")
    runtime_terminal_reason = str(runtime_payload["runtime"].get("terminal_reason"))
    if runtime_terminal_reason != spec.expected_terminal_reason:
        raise AssertionError(f"unexpected runtime terminal reason for {run_id}: {runtime_terminal_reason}")

    manifest_status = str(manifest_payload.get("status"))
    if manifest_status != spec.expected_manifest_status:
        raise AssertionError(f"unexpected manifest status for {run_id}: {manifest_status}")
    manifest_lifecycle = expect_mapping(manifest_payload.get("lifecycle"), key="run.json.lifecycle")
    manifest_terminal_reason = str(manifest_lifecycle.get("terminal_reason"))
    if manifest_terminal_reason != spec.expected_terminal_reason:
        raise AssertionError(f"unexpected manifest terminal reason for {run_id}: {manifest_terminal_reason}")
    if bool(manifest_lifecycle.get("reconciled")) != spec.reconciled:
        raise AssertionError(
            f"unexpected manifest reconciled flag for {run_id}: {manifest_lifecycle.get('reconciled')}"
        )

    return CaseResult(
        name=spec.name,
        run_id=run_id,
        status=lifecycle.status,
        manifest_status=manifest_status,
        terminal_reason=lifecycle.terminal_reason or "",
        runtime_path=runtime_path,
        manifest_path=manifest_path,
    )


def wait_for_lifecycle_row(
    *,
    store_root: Path,
    timeout_seconds: float,
    poll_seconds: float,
    predicate: Callable[[sqlite3.Row], bool],
    run_id: str | None = None,
    config_path: Path | None = None,
) -> sqlite3.Row:
    deadline = time.monotonic() + timeout_seconds
    last_row: sqlite3.Row | None = None
    while time.monotonic() < deadline:
        row = load_lifecycle_row(store_root=store_root, run_id=run_id, config_path=config_path)
        if row is not None:
            last_row = row
            if predicate(row):
                return row
        time.sleep(poll_seconds)
    raise AssertionError(f"timed out waiting for lifecycle row. last_row={row_to_dict(last_row)}")


def load_lifecycle_row(
    *,
    store_root: Path,
    run_id: str | None = None,
    config_path: Path | None = None,
) -> sqlite3.Row | None:
    if (run_id is None) == (config_path is None):
        raise AssertionError("load_lifecycle_row requires exactly one of run_id or config_path")
    db_path = store_root / "numereng.db"
    if not db_path.is_file():
        return None
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if run_id is not None:
            return conn.execute(
                "SELECT * FROM run_lifecycles WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return conn.execute(
            "SELECT * FROM run_lifecycles WHERE config_path = ? ORDER BY created_at DESC LIMIT 1",
            (str(config_path.resolve()),),
        ).fetchone()


def run_cli(
    command: list[str],
    *,
    cwd: Path,
    expect_returncodes: set[int],
    capture: bool,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if completed.returncode not in expect_returncodes:
        raise AssertionError(
            f"command failed returncode={completed.returncode} command={' '.join(command)} "
            f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )
    return completed


def parse_last_json_object(text: str | None) -> dict[str, Any]:
    if not text:
        raise AssertionError("expected JSON output, got empty text")
    for line in reversed([item.strip() for item in text.splitlines() if item.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise AssertionError(f"no JSON object found in output: {text!r}")


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.communicate(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5.0)


def finalize_generated_session_dir(*, generated_dir: Path, success: bool) -> None:
    if not generated_dir.exists():
        return
    if success:
        shutil.rmtree(generated_dir)
        print(f"[smoke] cleaned generated_dir={generated_dir}")
        return
    print(f"[smoke] retained generated_dir={generated_dir}")


def expect_mapping(value: object, *, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AssertionError(f"expected mapping for {key}, got {type(value).__name__}")
    return value


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[smoke] interrupted", file=sys.stderr)
        raise SystemExit(130)
