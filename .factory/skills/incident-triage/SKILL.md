---
name: incident-triage
description: "Factory-native mirror for diagnosing numereng run, store, and dashboard failures."
---

# Incident Triage

Use this skill when diagnosing failed local runs, stale lifecycle state, or dashboard/read-model mismatches.

## Start here

- Runbook: `runbooks/local-training-failures.md`
- Runbook: `runbooks/viz-monitoring.md`
- Existing repo skill: `.agents/skills/store-ops/SKILL.md`
- Deep lifecycle contract: `docs/llms.txt`

## Focus areas

- lifecycle bootstrap failures
- store/index drift
- missing `metrics.json` or scoring artifacts
- stale monitor state vs filesystem truth
- read-only viz surface issues

## Rule

- Prefer read-only diagnosis first.
- Use store repair or destructive cleanup workflows only through the existing store operations contract.
