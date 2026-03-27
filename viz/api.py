"""Uvicorn entrypoint for the local viz backend."""

import sys
from pathlib import Path


def _load_create_app():
    backend_root = Path(__file__).resolve().parent / "api"
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))
    from numereng_viz import create_app

    return create_app


app = _load_create_app()()
