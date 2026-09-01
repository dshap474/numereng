#!/usr/bin/env python3
"""Survey heavy numereng store artifacts and emit a JSON inventory.

Read-only: never deletes or mutates anything.

USAGE:
    uv run python .agents/skills/utility-cleanup-report/scripts/survey_artifacts.py
    uv run python .agents/skills/utility-cleanup-report/scripts/survey_artifacts.py --workspace .

Remote (copy to the target machine first, then run from its repo root):
    scp .agents/skills/utility-cleanup-report/scripts/survey_artifacts.py <host>:<repo>/.numereng/tmp/
    ssh <host> powershell -Command "cd <repo>; uv run python .numereng/tmp/survey_artifacts.py"

Pure stdlib, Python 3.12+, works on macOS and Windows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

HEAVY_FILE_BYTES = 10 * 1024 * 1024  # files >= 10MB count as heavy artifacts


# --------------------------------------------------------------------------- #
# Size helpers
# --------------------------------------------------------------------------- #


def tree_sizes(root: Path) -> tuple[int, int]:
    """Return (total_bytes, heavy_bytes) for all files under root."""
    total = heavy = 0
    if not root.is_dir():
        return 0, 0
    for f in root.rglob("*"):
        try:
            if not f.is_file():
                continue
            size = f.stat().st_size
        except OSError:
            continue
        total += size
        if size >= HEAVY_FILE_BYTES:
            heavy += size
    return total, heavy


def load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


# --------------------------------------------------------------------------- #
# Runs survey
# --------------------------------------------------------------------------- #


def survey_runs(runs_dir: Path) -> dict:
    by_exp: dict[str, dict] = {}
    total_bytes = dir_count = 0
    for run_dir in sorted(runs_dir.iterdir()) if runs_dir.is_dir() else []:
        if not run_dir.is_dir():
            continue
        dir_count += 1
        manifest = load_json(run_dir / "run.json")
        exp = manifest.get("experiment_id") or "(no experiment id)"
        if not (run_dir / "run.json").is_file():
            exp = "(no run.json)"
        total, heavy = tree_sizes(run_dir)
        total_bytes += total
        bucket = by_exp.setdefault(
            exp, {"run_count": 0, "total_bytes": 0, "heavy_bytes": 0, "run_ids": [], "statuses": {}}
        )
        bucket["run_count"] += 1
        bucket["total_bytes"] += total
        bucket["heavy_bytes"] += heavy
        bucket["run_ids"].append(run_dir.name)
        status = str(manifest.get("status", "unknown"))
        bucket["statuses"][status] = bucket["statuses"].get(status, 0) + 1
    return {"dir_count": dir_count, "total_bytes": total_bytes, "by_experiment": by_exp}


# --------------------------------------------------------------------------- #
# Experiments + package pickles survey
# --------------------------------------------------------------------------- #


def survey_packages(exp_dir: Path, runs_dir: Path) -> list[dict]:
    packages = []
    for pkg_json in sorted(exp_dir.glob("submission_packages/*/package.json")):
        pkg = load_json(pkg_json)
        pkg_dir = pkg_json.parent
        pickle_path = pkg_dir / "artifacts" / "pickle" / "model.pkl"
        component_run_ids = [
            c.get("run_id") for c in pkg.get("components", []) if isinstance(c, dict) and c.get("run_id")
        ]
        packages.append(
            {
                "package_id": pkg_dir.name,
                "status": pkg.get("status"),
                "last_pickle_model_name": pkg.get("last_pickle_model_name"),
                "last_pickle_upload_id": pkg.get("last_pickle_upload_id"),
                # Older packages record hosting only via status == "pickle_uploaded".
                "hosted": bool(pkg.get("last_pickle_upload_id")) or pkg.get("status") == "pickle_uploaded",
                "pickle_bytes": pickle_path.stat().st_size if pickle_path.is_file() else 0,
                "component_run_ids": component_run_ids,
                "components_with_local_model_artifact": [
                    rid
                    for rid in component_run_ids
                    if (runs_dir / rid / "artifacts" / "model" / "model.pkl").is_file()
                ],
                "package_dir_bytes": tree_sizes(pkg_dir)[0],
            }
        )
    return packages


def survey_experiments(experiments_dir: Path, runs_dir: Path) -> dict:
    out: dict[str, dict] = {}
    if not experiments_dir.is_dir():
        return out
    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        total, heavy = tree_sizes(exp_dir)
        entry: dict = {"total_bytes": total, "heavy_bytes": heavy, "archived": exp_dir.parent.name == "_archive"}
        packages = survey_packages(exp_dir, runs_dir)
        if packages:
            entry["packages"] = packages
        out[exp_dir.name] = entry
    return out


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only survey of heavy numereng artifacts.")
    parser.add_argument("--workspace", default=".", help="Workspace root (default: current directory)")
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    store = workspace / ".numereng"
    runs_dir = store / "runs"
    experiments_dir = store / "experiments"

    experiments = survey_experiments(experiments_dir, runs_dir)
    archive_dir = experiments_dir / "_archive"
    if archive_dir.is_dir():
        experiments.pop("_archive", None)
        for name, entry in survey_experiments(archive_dir, runs_dir).items():
            experiments[f"_archive/{name}"] = entry

    report = {
        "workspace": str(workspace),
        "heavy_file_threshold_bytes": HEAVY_FILE_BYTES,
        "runs": survey_runs(runs_dir),
        "experiments": experiments,
        "datasets_bytes": tree_sizes(store / "datasets")[0],
        "cache_bytes": tree_sizes(store / "cache")[0],
        "tmp_bytes": tree_sizes(store / "tmp")[0],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
