set shell := ["sh", "-cu"]

bootstrap:
    uv sync --extra dev

fmt:
    make fmt

lint:
    uv run ruff format --check .
    uv run ruff check .

type:
    uv run ty check

test:
    make test

test-all:
    make test-all

hpo-smoke:
    make hpo-smoke

readiness:
    make readiness

security:
    make security

build:
    uv build

kill-viz:
    ./scripts/viz-stop.sh

viz:
    ./scripts/viz-start.sh

ci:
    make ci
