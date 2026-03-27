from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    status = os.environ.get("NUMERENG_VALIDATION_STATUS", "").strip().lower()
    if status in {"ok", "passed"}:
        return 0
    status_file = os.environ.get("NUMERENG_VALIDATION_STATUS_FILE", "").strip()
    if status_file:
        path = Path(status_file)
        if path.is_file():
            file_status = path.read_text(encoding="utf-8").strip().lower()
            if file_status in {"ok", "passed"}:
                return 0
    sys.stderr.write(
        "numereng stop hook: validation evidence missing; run the canonical repo checks before declaring done.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
