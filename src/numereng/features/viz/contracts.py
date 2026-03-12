"""Request/response contracts for the viz API."""

from __future__ import annotations

def capabilities_payload() -> dict[str, bool]:
    """Return capability flags consumed by the frontend."""

    return {
        "read_only": True,
        "write_controls": False,
    }
