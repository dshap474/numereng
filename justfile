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

readiness:
    make readiness

security:
    make security

build:
    uv build

ci:
    make ci
