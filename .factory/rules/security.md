# Security rules

- Never commit `.env`, credentials, tokens, or generated local state under `.numereng/`
- Prefer environment variables and local ignored files for secrets
- Preserve source XOR invariants for submission and neutralization requests
- Treat `.github/`, `.factory/`, cloud workflow code, and auth-related paths as sensitive
- Keep secret exposure checks in the validation path via `make oss-preflight`
- Do not add networked automation or cloud-side behavior without explicit config and clear failure modes
- Fail fast on unsafe archive extraction, path traversal, or invalid run/store identities
