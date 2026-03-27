# numereng

## Bootstrap and validation

- Bootstrap: `just bootstrap`
- Format: `just fmt`
- Lint: `just lint`
- Type check: `just type`
- Fast tests: `just test`
- Full tests: `just test-all`
- Security and hygiene: `just security`
- Package build: `just build`

Use the project-managed environment only:

- Python baseline: `3.12+`
- Preferred runner: `uv run ...`
- Install deps with: `uv sync --extra dev`

## Architecture

- Dependency direction is strict: `config -> platform -> features -> api -> cli`
- `platform/*` must not import `features/*`
- Keep CLI thin; business logic belongs in `features/*` behind `api/*`
- Public import surfaces are `src/numereng/api/` and `src/numereng/cli/`
- No source file in `src/numereng/api/` or `src/numereng/cli/` may exceed 500 LOC

Read before changing contracts or flows:

1. `docs/llms.txt`
2. `docs/ARCHITECTURE.md`
3. `.factory/memories.md`
4. `.factory/rules/*.md`

## Repo memory and rules

- Project memory: `.factory/memories.md`
- Architecture rules: `.factory/rules/architecture.md`
- Python editing rules: `.factory/rules/python.md`
- Testing rules: `.factory/rules/testing.md`
- Security rules: `.factory/rules/security.md`
- Observability rules: `.factory/rules/observability.md`

## Factory-native workflows

- Factory skills live in `.factory/skills/*`
- Existing repo skills remain available in `.agents/skills/*`
- Factory hook scaffolding lives in `.factory/hooks/*`
- Project droid prompts live in `.factory/droids/*`

## Change-critical reminders

- Preserve API/CLI compatibility unless intentionally versioned
- Translate boundary errors to `PackageError`
- Training and HPO configs are JSON-only and reject unknown keys
- When contracts or execution flows change, update `docs/llms.txt` and `docs/ARCHITECTURE.md` in the same PR
