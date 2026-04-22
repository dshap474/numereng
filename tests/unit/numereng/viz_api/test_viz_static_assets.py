"""Focused tests for shipped viz frontend static assets."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from numereng_viz.app import create_app


def test_packaged_viz_favicon_is_shipped_and_served(tmp_path: Path) -> None:
    """The packaged viz frontend should expose a real favicon for browser tabs."""

    client = TestClient(create_app(workspace_root=tmp_path, store_root=tmp_path / ".numereng"))

    response = client.get("/favicon.svg")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert "<svg" in response.text
