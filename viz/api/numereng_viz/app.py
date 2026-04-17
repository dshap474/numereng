"""Application factory for the viz compatibility API."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from numereng.assets import viz_static_root
from numereng_viz.routes import create_router
from numereng_viz.services import VizService
from numereng_viz.store_adapter import VizStoreAdapter, VizStoreConfig

logger = logging.getLogger(__name__)


def create_app(
    *,
    workspace_root: str | Path | None = None,
    store_root: str | Path | None = None,
) -> FastAPI:
    """Create configured FastAPI app for the dashboard backend."""

    config = VizStoreConfig.from_env(workspace_root=workspace_root, store_root=store_root)
    adapter = VizStoreAdapter(config)
    service = VizService(adapter)
    frontend_root = viz_static_root()

    app = FastAPI(
        title="Numereng Viz API",
        version="0.3.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url="/api/redoc",
    )
    app.state.viz_service = service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def timing_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        response.headers["X-Request-Duration-Ms"] = f"{elapsed_ms:.2f}"
        if elapsed_ms > 1000.0:
            logger.warning("Slow API request: %s %s %.1fms", request.method, request.url.path, elapsed_ms)
        return response

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "service": "numereng-viz-api",
                "workspace_root": str(config.workspace_root),
                "store_root": str(config.store_root),
                "db_exists": adapter.db_path.exists(),
                "read_only": True,
            }
        )

    app.include_router(create_router(service))

    if frontend_root.is_dir():

        @app.get("/{path:path}")
        def serve_frontend(path: str) -> Response:
            normalized = path.strip("/")
            if not normalized:
                return FileResponse(frontend_root / "index.html")

            candidate = (frontend_root / normalized).resolve()
            try:
                candidate.relative_to(frontend_root.resolve())
            except ValueError:
                return JSONResponse({"detail": "Not found"}, status_code=404)
            if candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(frontend_root / "index.html")

    else:

        @app.get("/")
        def root() -> JSONResponse:
            return JSONResponse(
                {
                    "service": "numereng-viz-api",
                    "health": "/healthz",
                    "api": "/api",
                }
            )

    return app
