## Research Strategy (Current CLI)

Use this reference for hypothesis design, round planning, stop criteria, and experiment reporting.

### Objective

Deliver a defensible experiment outcome with:
- clear hypothesis,
- multiple rounds (not one run),
- BMC-first ranking,
- explicit stop rationale,
- reproducible commands and configs.

### Non-Negotiable Defaults

1. Use rounds of 4-5 configs by default.
- One base config plus single-variable variants.
- Only run fewer variants when explicitly requested.

2. Keep baseline alignment stable within a round.
- Same dataset/feature set unless that is the tested variable.

3. Use payout-oriented metric ranking.
- Primary: `bmc_last_200_eras.mean`
- Tie-break: `bmc.mean`
- Sanity checks: `corr.mean`, `mmc.mean`, `cwmm.mean`

4. Stop on plateau.
- Do not finalize after a single strong run.

### Planning Checklist

- State novelty and expected mechanism.
- Define baseline config path.
- Pick sweep dimension for the next round.
- Define stop signal and risk checks.

### Ambiguity Handling (Fast Disambiguation)

If the request is underspecified:
1. List 2-4 plausible interpretations.
2. Run quick scout variants for each interpretation.
3. Use lower-cost scout settings (for example downsampled dataset variant and smaller model capacity).
4. Compare interpretation winners on primary/tie-break metrics and sanity checks.
5. Record chosen direction and rationale in `EXPERIMENT.md`.

### Scout -> Scale Progression

1. Scout phase:
- lower-cost, single-variable rounds.
2. Scale phase:
- run only for top candidates after repeatable scout wins.

Before concluding the experiment, run at least one scaled confirmatory round.

### Core Workflow

1. Create experiment:

```bash
uv run numereng experiment create --id <YYYY-MM-DD_slug> --hypothesis "..." --tags "tag1,tag2"
```

2. Prepare configs for this round in:
- `.numereng/experiments/<id>/configs/`

3. Train each config:

```bash
uv run numereng experiment train --id <id> --config <config.json>
```

4. Rank results:

```bash
uv run numereng experiment report --id <id> --metric bmc_last_200_eras.mean --format table
```

5. Inspect experiment state:

```bash
uv run numereng experiment details --id <id> --format json
```

6. Update `EXPERIMENT.md` with round outcomes and the next decision.

### Plateau Stop Criteria

Stop when both hold:
1. Two consecutive rounds do not exceed the best `bmc_last_200_eras.mean` by a meaningful margin.
2. Remaining untried knobs are likely redundant or likely to increase overfit risk.

Default meaningful margin threshold:
- `1e-4` to `3e-4` on `bmc_last_200_eras.mean`.

### Reporting Minimum

Each round should capture:
- config changes,
- ranked runs,
- round-best delta vs prior-best,
- chosen winner,
- risks,
- next round plan.

### Notes on Removed Commands

This package currently does not provide `experiment summarize/show/compare` or `orchestrator` command families.
Use `experiment report` + `experiment details` as the canonical replacement.
