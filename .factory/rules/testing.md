# Testing rules

- Run the canonical fast gate before concluding work: `make test`
- Run `make test-all` when changes affect broad behavior, slow tests, or release confidence
- Prefer targeted `uv run pytest ...` during iteration, then return to the canonical gate
- Keep pytest markers registered and strict
- Add or update tests when behavior changes at the CLI, API, feature, or store contract boundary
- Preserve smoke coverage for the public CLI/API surface
- Do not mark work done if validators fail
