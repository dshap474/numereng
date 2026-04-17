"""Platform helpers for cloning upstream docs repositories."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def ensure_git_available() -> str:
    """Return the git executable path or fail clearly if unavailable."""

    git_path = shutil.which("git")
    if git_path is None:
        raise RuntimeError("docs_sync_git_missing")
    return git_path


def clone_shallow(*, repo_url: str, destination: Path) -> str:
    """Clone one repository shallowly and return the checked-out commit SHA."""

    git_path = ensure_git_available()
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
    }
    clone_result = subprocess.run(
        [git_path, "clone", "--depth", "1", repo_url, str(destination)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if clone_result.returncode != 0:
        raise RuntimeError(f"docs_sync_clone_failed:{_best_effort_message(clone_result)}")

    sha_result = subprocess.run(
        [git_path, "-C", str(destination), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if sha_result.returncode != 0:
        raise RuntimeError(f"docs_sync_rev_parse_failed:{_best_effort_message(sha_result)}")

    commit_sha = sha_result.stdout.strip()
    if not commit_sha:
        raise RuntimeError("docs_sync_rev_parse_failed:missing_commit_sha")
    return commit_sha


def _best_effort_message(result: subprocess.CompletedProcess[str]) -> str:
    for candidate in (result.stderr, result.stdout):
        if candidate:
            for line in candidate.splitlines():
                stripped = line.strip()
                if stripped:
                    return stripped
    return f"exit_code_{result.returncode}"


__all__ = ["clone_shallow", "ensure_git_available"]
