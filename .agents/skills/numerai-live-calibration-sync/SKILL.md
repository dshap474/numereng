---
name: numerai-live-calibration-sync
description: Sync submitted Numerai models and calibrate local offline metrics against live performance. Use when Codex needs to refresh live rounds, rebuild local-vs-live calibration artifacts, distinguish provisional from resolved evidence, interpret calibration regressions or lane behavior, or update the research-memory live-calibration stance.
---

# Live Calibration Sync

Run from the numereng repo root. Use this skill for calibration; use `numerai-submissions-ops` only for individual submission-folder snapshots and `numerai-api-ops` for account or hosted-upload operations.

Read [references/calibration-contract.md](references/calibration-contract.md) before interpreting metrics, diagnosing low observation counts, or editing research memory.

## Modes

- **Full** (default): refresh, materialize, report, and integrate material changes into research memory.
- **Pull only**: refresh live state and rebuild calibration artifacts; do not edit research memory.
- **Interpret only**: use current artifacts without a network refresh. State their timestamps and do not describe them as fresh.

## Phase 1: Mechanical Sync

Preview the Numerai pull when useful:

```bash
uv run numereng submissions calibration update --dry-run --format json
```

Run the canonical idempotent update:

```bash
uv run numereng submissions calibration update --format json
```

The update refreshes every selected `.numereng/submissions/<model>/` snapshot and rebuilds `.numereng/analysis/live_calibration/`.

Important: update dry-run previews only the remote refresh. It intentionally does not materialize or report correlations because the on-disk round data remains unchanged.

## Phase 2: Validate The Evidence

Confirm:

1. the command exited zero;
2. `manifest.json.generated_at` is current;
3. model counts and excluded models are explicit;
4. resolved-round counts are separated from provisional scored counts;
5. each calibration observation represents one uploaded artifact per scope, not one round;
6. local metrics are matched to the correct upload provenance.

Treat low resolved counts on recent uploads as timing evidence, not automatically as a pipeline defect.

## Phase 3: Interpret

Use `resolved_only` for final claims. Use `all_scored` only as an explicitly provisional directional read.

Compute from the fresh artifacts:

- regressions for local BMC200 and FNC against live BMC, MMC20, and CORR20;
- resolved observations per uploaded artifact;
- target-family and feature-scope lane means;
- within-lane rank correlation when sample size supports it;
- resolved MMC20 hit rates from per-round snapshots;
- the local-BMC break-even point `-intercept / slope`, only within the observed local-metric domain.

Never mix package-level local BMC200 values with agentic per-era research metrics. Do not imply causality from a small cross-sectional correlation.

## Phase 4: Integrate Material Changes

Compare the fresh report and manifest with the **Data basis** line in `.numereng/notes/__RESEARCH_MEMORY__/scoring/live-local-calibration.md`.

Treat a change as material when it adds meaningfully more resolved evidence, adds a new live model or upload cohort, flips a lane verdict, changes the break-even conclusion, or resolves a previously open calibration question.

If nothing material changed, report the comparison and stop. Otherwise edit exactly:

1. `.numereng/notes/__RESEARCH_MEMORY__/scoring/live-local-calibration.md`
2. the `## Live Calibration Stance` section and related scale anchor in `.numereng/notes/__RESEARCH_MEMORY__/CURRENT.md`

Preserve the package-scale warning, measured scale mapping, evidence dates, and interpretation-pass history. Do not touch `topics/*.md` or experiment branches.

## Safety

- This skill is read-only against Numerai. Do not create slots, upload or assign pickles, submit predictions, replace models, or stake.
- Do not duplicate per-round data into markdown.
- Do not interpret stale artifacts as a current refresh.
- Do not extrapolate regressions beyond their observed local-metric range.
- Name excluded models and missing provenance instead of silently shrinking the sample.

## Done Criteria

Report:

- refresh timestamp and API source;
- models refreshed and excluded;
- provisional and resolved counts per model or upload cohort;
- resolved lane verdict and within-lane evidence strength;
- break-even estimate with its observed domain;
- changes versus the prior interpretation;
- exact artifacts and memory files updated.

If the pull fails, retry once. If it still fails, stop without interpreting stale state as fresh.
