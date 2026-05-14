"""Agent guidance for the agentic research prompt system."""

# Agentic Research Programs

This folder contains the tracked default prompt program and the local-only custom program area for
autonomous Numerai config research.

Important files:

- `PROGRAM.md`: the tracked base/default program contract.
- `run.py`: the deterministic operator that renders context, calls the LLM, validates decisions,
  writes configs, trains, scores, and records artifacts.
- `custom_programs/`: ignored local custom prompt programs for focused experiments.
- `custom_programs/README.md`: the only tracked file inside `custom_programs/`.

## Core Mental Model

Agentic research is split into two parts:

- LLM: reads the program plus experiment context, updates the rolling memo, and proposes one
  structured `decision_form`.
- Python: validates the proposal, materializes the strict machine decision, writes the next config,
  executes training/scoring, and appends exact results.

The LLM does not edit Python code and does not hand-author the machine log. It returns research
intent. Python turns that intent into reproducible artifacts.

## Program Resolution

By default, experiments use:

```text
src/numereng/features/agentic_research/PROGRAM.md
```

To use a custom program, create a local Markdown file under:

```text
src/numereng/features/agentic_research/custom_programs/MY-PROGRAM.md
```

Then point an experiment at it with `metadata.agentic_research_program`:

```json
{
  "metadata": {
    "agentic_research_program": "MY-PROGRAM.md"
  }
}
```

Use only the filename. The runner resolves names from `custom_programs/` and rejects paths.

Inspect resolution:

```bash
uv run numereng research status --experiment-id <experiment_id>
```

Run the loop:

```bash
uv run numereng research run --experiment-id <experiment_id> --max-rounds <n>
```

## Custom Program Rule

Every custom program must be fully self-contained.

Do not assume the runner loads `PROGRAM.md` in addition to the custom file. If a custom program needs
the evidence doctrine, output contract, metric priorities, or focused-program structure, copy those
instructions into the custom program.

## Required Output Contract

Every program must require exactly one JSON object with:

```json
{
  "decision_form": {
    "action": "run",
    "learning": "What the prior evidence taught us.",
    "belief_update": "What you now believe about this hypothesis.",
    "next_hypothesis": "The specific hypothesis tested by the next config.",
    "parent_config": "existing_config_filename.json",
    "changes": [
      {
        "path": "model.params.learning_rate",
        "value": 0.01,
        "reason": "Why this exact change is worth testing."
      }
    ],
    "stop_reason": null
  },
  "round_markdown": "# rNNN Research State\n\n..."
}
```

For stop decisions:

- `decision_form.action = "stop"`
- `next_hypothesis = null`
- `parent_config = null`
- `changes = []`
- `stop_reason` explains why no next run is justified.

`round_markdown` is the cumulative human-readable research state. Python appends an
`Execution Result` section after the deterministic run or stop is recorded.

## Focused Program Requirements

Every serious custom program should define:

| Section | Purpose |
| --- | --- |
| Research Hypothesis | The exact idea being tested. |
| Fixed Surface | What must not change during this experiment. |
| Only Vary | The single axis or small related family of config paths to explore. |
| Baseline And Comparison Rule | Which parent is fair and how to compare variants. |
| Evidence Rules | What counts as discovery, confirmation, plateau, or failure. |
| Sweep Discipline | How to change one variable at a time without mixing hypotheses. |
| Stop, Confirmation, And Handoff | When to stop, seed-confirm, or move to a new program. |
| Rolling Memo Contract | What must be carried forward in `round_markdown`. |

Good focused programs:

- target shortlist while holding model and feature set fixed
- LGBM regularization after target route is fixed
- feature-scope comparison while holding target/model recipe fixed
- Ender20 versus Ender60 branch comparison
- neutralization scout after a stable prediction source exists

Poor focused programs:

- "try anything that improves BMC"
- switching target, feature set, model family, and regularization in one chain
- tuning hyperparameters before the target or feature-scope question is resolved
- declaring convergence from single-seed discovery

## Complete Custom Program Template

```markdown
# Program: <specific hypothesis name>

You are the research brain for one focused Numereng experiment. Python is the deterministic
operator: it validates your `decision_form`, writes the machine decision log, mutates one config,
trains, scores, and records exact results.

## Research Hypothesis

<State the exact hypothesis. Example: On medium features, target route explains more BMC variance
than LGBM hyperparameter tuning.>

## Fixed Surface

These must not change:

- `data.feature_set = "<feature_set>"`
- `model.type = "<model_type>"`
- `<other fixed config paths and values>`

If a useful next experiment requires changing the fixed surface, return `action: "stop"` and name
the handoff program or next experiment type.

## Only Vary

The only intended variation axis is:

- `<allowed path or related path family>`

Do not tune unrelated knobs. Do not combine two unrelated hypotheses in one decision.

## Baseline And Comparison Rule

- Baseline config: `<filename or rule>`
- Use the best comparable parent, not automatically the previous round.
- Comparable means same fixed surface, evaluation surface, and evidence tier.

## Evidence Rules

- Single-seed discovery is directional only.
- Seed-averaged confirmation is required before claiming a true winner or convergence.
- Treat improvements below roughly `1e-4` to `3e-4` on `bmc_last_200_eras_mean` as provisional.
- Primary metric: `bmc_last_200_eras_mean`
- Tie-break: `bmc_mean`
- Sanity checks: `corr_mean`, `mmc_mean`, `cwmm_mean`

## Sweep Discipline

- Change 1 to 3 values per round.
- Prefer one-variable-at-a-time changes.
- Keep a clear table of tried configs in `round_markdown`.
- Record what should not be repeated.

## Stop, Confirmation, And Handoff

Stop when:

- the focused axis is exhausted,
- useful next moves require leaving the fixed surface,
- repeated confirmed challengers fail to beat the incumbent,
- or the next step should be seed confirmation rather than more discovery.

## Round Markdown

Return `round_markdown` as the cumulative research state after your decision. It should carry forward
the important prior-round learnings and include:

- current best config and why it is best
- confirmed versus unconfirmed beliefs
- what has been tried and should not be repeated
- what this decision is testing
- pending confirmation needs
- the next open question

Python will append an `Execution Result` section after the deterministic run or stop is recorded.

## Output

Return exactly one JSON object conforming to the provided schema:

```json
{
  "decision_form": {
    "action": "run",
    "learning": "What the prior evidence taught us.",
    "belief_update": "What you now believe about this hypothesis.",
    "next_hypothesis": "The specific hypothesis tested by the next config.",
    "parent_config": "existing_config_filename.json",
    "changes": [
      {
        "path": "model.params.learning_rate",
        "value": 0.01,
        "reason": "Why this exact change is worth testing."
      }
    ],
    "stop_reason": null
  },
  "round_markdown": "# rNNN Research State\n\n..."
}
```

## Context

```json
{{CONTEXT_JSON}}
```
```
