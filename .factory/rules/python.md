# Python rules

- Use the project-managed environment only: `uv sync --extra dev`
- Prefer `uv run <tool>` for Ruff, `ty`, `pytest`, and `numereng`
- Python baseline is `3.12+`
- Match the existing typed dataclass / Pydantic / request-response style already used in `api/contracts.py`
- Follow current import style and module layout; do not introduce alternate package structures casually
- Do not bypass the public CLI/API boundary for user-facing flows unless the task explicitly targets internals
- Keep comments minimal and only when they clarify a non-obvious invariant
- Preserve deterministic filesystem and store behavior; avoid hidden environment-dependent behavior
