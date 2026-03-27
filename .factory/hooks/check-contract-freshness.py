from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CRITICAL_PREFIXES = (
    "src/numereng/api/",
    "src/numereng/cli/",
    "src/numereng/config/training/",
    "src/numereng/features/training/",
    "src/numereng/features/experiments/",
    "src/numereng/features/telemetry/",
    "src/numereng/features/viz/",
    "viz/api/numereng_viz/",
)

REQUIRED_FRESHNESS_FILES = {
    "AGENTS.md",
    ".factory/memories.md",
    ".factory/rules/architecture.md",
    ".factory/rules/observability.md",
    "docs/llms.txt",
    "docs/ARCHITECTURE.md",
}


def _changed_files(repo_dir: Path, base_ref: str, head_ref: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "diff", "--name-only", f"{base_ref}...{head_ref}"],
        capture_output=True,
        check=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    repo_dir = Path(__file__).resolve().parents[2]
    base_ref = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    head_ref = sys.argv[2] if len(sys.argv) > 2 else "HEAD"
    changed_files = _changed_files(repo_dir, base_ref, head_ref)
    touched_critical = [path for path in changed_files if path.startswith(CRITICAL_PREFIXES)]
    if not touched_critical:
        print("contract freshness: no critical contract paths changed")
        return 0
    touched_freshness = REQUIRED_FRESHNESS_FILES.intersection(changed_files)
    if touched_freshness:
        print("contract freshness: supporting docs/memory files changed alongside critical paths")
        return 0
    print("::warning::Critical contract paths changed without AGENTS/memory/rule/docs freshness updates.")
    print("Changed critical files:")
    for path in touched_critical:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
