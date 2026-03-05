"""Request/response contracts for the viz compatibility API."""

from __future__ import annotations

from typing import Any


def numerai_classic_compat_payload() -> dict[str, Any]:
    """Return the static Numerai classic compat payload used by the dashboard."""

    return {
        "spec_id": "numerai-classic-v5.2-faith-ii",
        "version": "2026-01-01",
        "effective_date": "2026-01-01",
        "verified_at": "2026-02-13",
        "source_precedence": "numerai-docs/forum",
        "targets": {
            "payout_20d": "target_ender_20",
            "payout_60d": "target_ender_60",
        },
        "payout": {
            "corr_weight": 0.75,
            "mmc_weight": 2.25,
            "clip": 0.05,
        },
        "validation": {
            "walk_forward": {
                "scheme": "era_chunked_forward",
                "chunk_size_eras": 20,
                "purge_20d": 4,
                "purge_60d": 12,
            },
        },
    }


def capabilities_payload() -> dict[str, Any]:
    """Return capability flags consumed by the frontend."""

    return {
        "read_only": True,
        "write_controls": False,
    }
