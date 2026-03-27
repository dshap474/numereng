# Architecture rules

- Preserve dependency direction: `config -> platform -> features -> api -> cli`
- `platform/*` must never import `features/*`
- Keep CLI thin: parsing, dispatch, help text, and output only
- Put business logic in `features/*` and expose it through `api/*`
- Keep public API and CLI compatibility unless the change is intentionally versioned
- Translate internal failures to `PackageError` at the API boundary
- Keep source files in `src/numereng/api/` and `src/numereng/cli/` at or under 500 LOC
- Prefer extending existing feature slices over creating cross-layer shortcuts
- When execution flows or contracts change, update `docs/llms.txt` and `docs/ARCHITECTURE.md` in the same change
