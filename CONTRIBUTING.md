# Contributing to numereng

Thanks for contributing.

## Prerequisites

- Python 3.11+
- `uv`
- Node.js 20+ (only needed for `make viz`)

## Setup

Numereng now lives in its own repository directory.

```bash
uv sync
```

Then work from `numereng`.

## Development workflow

1. Keep changes scoped to the issue you are fixing.
2. Add or update tests for behavior changes.
3. Run relevant checks before opening a PR.

## Common checks

```bash
# unit + integration
make test

# targeted suites
make test-unit
make test-integration

# OSS hygiene/public snapshot checks
make check-oss-hygiene
make verify-public-snapshot
```

## Type and lint ratchet

```bash
# run full typecheck/lint
make hygiene

# capture current baseline (writes dev/hygiene/*.json)
make hygiene-capture-baseline

# fail only if mypy/ruff counts increase vs baseline
make hygiene-ratchet

# collect tests without coverage gate
make test-collect
```

## Pull requests

- Use a clear title and include a short problem/solution summary.
- Include testing notes (what you ran and what passed).
- Avoid unrelated refactors in the same PR.

## Security

For vulnerabilities, do not open a public issue first. Follow `SECURITY.md`.
