"""Uvicorn entrypoint for the local viz backend."""

from numereng.features.viz import create_app

app = create_app()
