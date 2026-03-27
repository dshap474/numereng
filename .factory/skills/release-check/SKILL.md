---
name: release-check
description: "Run numereng's canonical repo validation flow before handoff or release-oriented changes."
---

# Release Check

Use this skill when the task needs a final proof pass for repo-local changes.

## Canonical commands

```bash
just security
just test
just test-all
just build
```

## Notes

- `make test-all` and `just build` are the full-confidence gates.
- If the change does not affect packaging metadata or build outputs, `just build` may be skipped with an explicit reason.
- If any contract or flow changed, confirm `docs/llms.txt` and `docs/ARCHITECTURE.md` were updated.
