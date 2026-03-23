# Contributing to numereng

Thanks for contributing.

## Prerequisites

- Python 3.11+
- `uv`
- Node.js 20+ (only needed for `make viz`)

## Setup

```bash
uv sync --extra dev
```

Optional extras:

- `uv sync --extra training`
- `uv sync --extra mlops`

## Development workflow

1. Branch from `main`.
2. Keep each change scoped to one behavior or documentation update.
3. Add or update tests when behavior changes.
4. Open a pull request back to `main`.

`main` is intended to stay merge-only. Even solo-maintainer changes should land through a PR so CI, docs, and release notes stay reviewable. Direct pushes to `main` are blocked by branch protection.

## Common checks

```bash
# auto-fix lint issues and normalize formatting
make fmt

# repo hygiene and high-confidence secret scan
make oss-preflight

# fast gate: format check + lint + scoped ty + non-slow tests
make test

# full gate: includes slow tests
make test-all

# build package artifacts
uv build
```

For targeted work, prefer direct `uv run` commands:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest -q tests/path/to/test_file.py
```

`ty` is the canonical repo type gate. Its first enforced surface is intentionally scoped via
`[tool.ty.src].include` in `pyproject.toml` while broader repo cleanup is deferred.

## Pull requests

- Use a clear title and short problem/solution summary.
- Include testing notes (what you ran and what passed).
- Avoid unrelated refactors in the same PR.
- Update docs when public CLI, API, or workflow behavior changes.
- Update `docs/llms.txt` and `docs/ARCHITECTURE.md` in the same PR when contracts or flows change.

## Security

Do not report vulnerabilities in public issues. Follow `SECURITY.md`.
