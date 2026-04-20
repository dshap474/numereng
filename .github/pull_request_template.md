## Summary

- What changed?
- Why is this change needed?

## Testing

- [ ] `just oss-preflight`
- [ ] `just readiness`
- [ ] `just test`
- [ ] `just test-all` (if needed)
- [ ] `uv build --package numereng --wheel --no-build-logs` (only if the internal cloud packaging flow changed)

## Checklist

- [ ] Scope is limited to one behavior or documentation change
- [ ] Public docs were updated if user-facing behavior changed
- [ ] Security-sensitive changes were reviewed for secret exposure
