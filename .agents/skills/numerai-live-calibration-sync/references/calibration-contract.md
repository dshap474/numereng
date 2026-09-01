<!--
Purpose: Durable metric, timing, and artifact contract for live calibration.
Usage: Read before interpreting local-versus-live results or diagnosing calibration counts.
-->

# Live Calibration Contract

## Artifact Map

Mechanical update writes:

```text
.numereng/submissions/<model>/
  submission.json
  live_rounds.parquet

.numereng/analysis/live_calibration/
  calibration_rows.parquet
  calibration_observations.parquet
  report.json
  manifest.json
```

`calibration_rows.parquet` is the per-model, per-round audit trail. Use it for dates, state transitions, metric histories, and upload attribution. Do not use raw rows as independent regression observations because long-running uploads would be overweighted.

`calibration_observations.parquet` is the comparison surface. It aggregates each uploaded model artifact into one observation per scope:

- `all_scored`: includes provisional scored rounds;
- `resolved_only`: includes only final resolved scored rounds.

`report.json` contains the derived regression/correlation readout. `manifest.json` records generation time, inputs, coverage, and provenance.

## Evidence Classes

Keep three classes separate:

1. **Local offline**: validation or package metrics such as local BMC200, CORR, MMC, and FNC.
2. **Live provisional**: Numerai scores available before final round resolution; useful only for early direction.
3. **Live resolved**: final round outcomes; the primary evidence for calibration conclusions.

Hosted validation or diagnostics proves executable compatibility. It is not resolved live performance.

## Numerai Timing

Classic rounds open on a near-daily weekday/weekend cadence. Provisional scores appear after submission and change over time. Final 20D2L scores usually resolve about 31-33 days after close; 60D2L resolves later.

Once an upload has passed the initial lag, expect resolved 20D2L observations to accumulate steadily. A recent upload with zero resolved rounds is normally too young, not broken.

## Interpretation Rules

- Use `resolved_only` for durable claims.
- Label `all_scored` conclusions provisional.
- Prefer at least 8-10 independent upload-level observations for an early correlation read; more cohorts are better.
- Report `n`, scope, observed metric range, and excluded models with every regression conclusion.
- Compare target-family and feature-scope lanes before comparing individual siblings.
- Treat within-lane correlation as weak when the lane has only a few uploads.
- Calculate break-even as `-intercept / slope` only when slope is meaningful and the result lies near the observed local-metric domain.
- Preserve sign conflicts between live BMC, MMC, CORR, and FNC instead of collapsing them into one score.

## Common Misreads

- More live rounds from one upload do not create more independent local-vs-live observations.
- A strong provisional correlation can reverse after resolution.
- Package-local BMC and agentic per-era BMC use different scales and must not share thresholds.
- A high hosted diagnostic score from `trainedOnVal=true` is in-sample compatibility evidence, not an expected live score.
- Missing local provenance can exclude an upload from calibration even when live rounds exist.

## Commands

```bash
# Full idempotent pull, rebuild, and report
uv run numereng submissions calibration update --format json

# Pull preview only; does not rebuild correlations
uv run numereng submissions calibration update --dry-run --format json

# Rebuild from existing local snapshots
uv run numereng submissions calibration materialize --format json

# Read current calibration report
uv run numereng submissions calibration report --format json

# Restrict the canonical update to resolved evidence in its report
uv run numereng submissions calibration update --resolved-only --format json
```
