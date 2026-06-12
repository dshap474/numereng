"""Agent guidance for the agentic research prompt system."""

# Agentic Research Programs

This folder holds the tracked default program and the local-only custom-program area for autonomous
Numerai config research. After the 7-invariants rebuild, the harness is small and stateless about
strategy: it makes the research possible, the program file does the research.

Important files:

- `PROGRAM.md`: the tracked base/default program contract.
- `custom_programs/`: ignored local custom programs for focused experiments.
- `custom_programs/README.md`: the only tracked file inside `custom_programs/`.
- `STATE_DIAGRAM.md`: the session-lifecycle and per-round state machine.

## Core Mental Model

Two parts, one boundary between them:

- **LLM proposes.** It reads the program plus bounded experiment context, updates its memo, and
  returns one structured `decision_form` of research intent.
- **Harness validates, executes, and records — it never edits the proposal.** Python checks the
  decision against fixed boundaries, materializes one config (rejecting out-of-bounds proposals
  whole, never clamping), trains, scores, appends durable memory, and updates state.

The harness makes no research-strategy decision: what to try, when to confirm, when to diversify, and
what to believe all live in the program file and the model. The only things the harness rejects are
boundary violations.

## Experiment Isolation And Knowledge Flow

Each program run is an isolated experiment. The harness injects no cross-experiment memory into the
prompt: the single program file is the only cross-experiment knowledge input. After a run completes,
a human-driven distillation pass writes durable learnings to `.numereng/notes/__RESEARCH_MEMORY__/`.
The next run gets a freshly authored program file that encodes the selected learnings. This keeps
the run auditable: "what did run X know?" is answered by reading that run's program file.

## Module Layout (6 small modules)

- `types.py` — exceptions, the public result dataclasses, decision/response types, shared constants
  and small utils.
- `memory.py` — `state.json` load/save (+ defaults), `journal.jsonl` append/tail, `rounds/rN.md`
  writer, `EXPERIMENT.md` passthrough, failure debug dumps, heartbeat.
- `boundary.py` — decision → config materialization (path allowlist, value caps, horizon/target
  match, strict `TrainingConfig` validation, hash dedup), baseline config, stale-run-reuse guard,
  run-plan recording, frozen-scoring assertion.
- `llm.py` — prompt render, codex-exec + openrouter transport, static response schema, parse/validate
  (rejects non-`run` action), failure debug dumps.
- `context.py` — bounded context assembly (champion, capped report rows, recent journal, last memo,
  EXPERIMENT.md, last error); no term grows with round count.
- `loop.py` — the seam anchor: session lifecycle (`run_research`, `get_research_status`,
  `program_markdown`), round driver, keep/discard, failure counting.

`__init__.py` re-exports the public surface from `loop` and `types`.

## Program Resolution

By default, experiments use:

```text
src/numereng/features/agentic_research/PROGRAM.md
```

To use a custom program, create a local Markdown file under `custom_programs/` and point an
experiment at it via `metadata.agentic_research_program`:

```json
{
  "metadata": {
    "agentic_research_program": "MY-PROGRAM.md"
  }
}
```

Use only the filename. The runner resolves names from `custom_programs/` and rejects paths.

Inspect resolution and run the loop:

```bash
uv run numereng research status --experiment-id <experiment_id>
uv run numereng research run --experiment-id <experiment_id> --max-rounds <n>
```

## Custom Program Rule

Every custom program must be fully self-contained. The runner loads exactly one program file — it
does **not** also load `PROGRAM.md`. If a custom program needs the frozen evaluator, evidence
doctrine, output contract, or the known-traps list, copy those sections in from `PROGRAM.md`.

## Required Output Contract

Every program must require exactly one JSON object. The only action is `"run"`; there is no `stop`
action and no ensemble fields. `stop_reason` is kept in the schema for shape stability and ignored.

```json
{
  "decision_form": {
    "action": "run",
    "learning": "What the prior evidence taught us.",
    "belief_update": "What you now believe about this hypothesis.",
    "next_hypothesis": "The specific hypothesis tested by the next config.",
    "parent_config": "config_007.json",
    "changes": [
      {
        "path": "model.params.learning_rate",
        "value": 0.02,
        "reason": "Why this exact change is worth testing."
      }
    ],
    "stop_reason": null
  },
  "round_markdown": "# rNNN Research State\n\n...",
  "experiment_markdown": "# Champion State\n...\n"
}
```

- `changes` carries 1 to 5 `{path, value, reason}` entries on allowed paths within the value caps.
- `round_markdown` is the model's verbatim round memo; the harness appends a `## Machine Result`
  block below it.
- `experiment_markdown` overwrites `EXPERIMENT.md`, or `null` to preserve the prior file.

## Boundary Rejections

The harness rejects only boundary violations, with a stable error token surfaced in the next round's
`context.last_error`. A rejection fails the round and counts toward the 5-consecutive-failure bail;
a duplicate is the one exception (soft skip, no count):

- disallowed change path (`agentic_research_change_path_not_allowed:`)
- out-of-cap value (`agentic_research_change_value_out_of_cap:`) — not clamped
- `data.target_horizon` not matching the `data.target_col` suffix
  (`agentic_research_horizon_target_mismatch:`)
- invalid `TrainingConfig` (unknown keys forbidden, JSON-only)
- non-`run` action (`agentic_research_action_invalid`)
- cross-experiment stale-run reuse (`agentic_research_stale_run_reuse_blocked:`)
- duplicate-by-hash with a recorded run — soft skip

## Artifact Map

| Artifact | Role |
| --- | --- |
| `agentic_research/state.json` (schema_version 2) | small session state: status, counters, champion, heartbeat |
| `agentic_research/journal.jsonl` | one append-only line per round attempt (machine-readable) |
| `agentic_research/rounds/rN.md` | model memo verbatim + one `## Machine Result` block |
| `EXPERIMENT.md` | model-curated working set (passthrough write; `null` preserves prior) |

Durable history lives in `journal.jsonl` + `rounds/*.md`; `EXPERIMENT.md` and `state.json` are the
working set. The context shown to the model is bounded by construction — no term grows with round
count.

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
| Confirmation And Handoff | When to seed-confirm or note a handoff candidate for a future program. |
| Rolling Memo Contract | What must be carried forward in `round_markdown`. |

Good focused programs: a target shortlist with model and feature set fixed; LGBM regularization after
the target route is fixed; a feature-scope comparison with the target/model recipe fixed; an
Ender20-vs-Ender60 branch comparison. Poor ones: "try anything that improves BMC"; switching target,
feature set, model family, and regularization in one chain; declaring convergence from single-seed
discovery.

## Complete Custom Program Template

```markdown
# Program: <specific hypothesis name>

You are the research brain for one focused Numereng experiment. The harness validates your
`decision_form` against fixed boundaries, materializes one config, trains, scores, and records the
exact result. It never edits your config and never stops the run.

## Research Hypothesis

<State the exact hypothesis.>

## Fixed Surface

These must not change:

- `data.feature_set = "<feature_set>"`
- `model.type = "<model_type>"`
- `<other fixed config paths and values>`

## Only Vary

The only intended variation axis is:

- `<allowed path or related path family>`

## Baseline And Comparison Rule

- Baseline config: `<filename or rule>`
- Use the best comparable parent, not automatically the previous round.
- Comparable means same fixed surface, evaluation surface, and evidence tier.

## Frozen Evaluator

- Stage `post_training_core`; metric `bmc_last_200_eras_mean` = BMC against the payout target
  `target_ender_20`, orthogonal to the meta-model, last 200 eras. `data.target_col` is just the best
  training label for an `ender_20`-contributing signal; the metric never scores the trained target.

## Evidence Rules

- Single-seed discovery is directional only (noise ~`3e-4`).
- Confirm with the seed trio `42 / 17 / 99`; compute the trio mean yourself from your memo ledger
  (`recent_journal` carries seed+metric for only the last 12 rounds).
- Treat improvements below ~`1e-4`–`3e-4` as provisional until trio-confirmed.
- Primary metric `bmc_last_200_eras_mean`; tie-break `bmc_mean`; sanity `corr_mean`, `mmc_mean`,
  `cwmm_mean`.

## Known Traps

- LGBM `num_leaves` above `2 ** max_depth` is a no-op — raise `max_depth` first.
- `data.target_horizon` must match the `data.target_col` suffix (`_20`→`20d`, `_60`→`60d`) or the
  round is rejected; set it yourself when changing `target_col`.
- Switching `model.type` requires nulling the prior family's params yourself.
- Duplicate-by-hash wastes a round; check your tried-configs ledger first.

## Sweep Discipline

- Change 1 to 5 values per round; prefer one variable at a time.
- Keep a tried-configs table in `round_markdown`. Record what should not be repeated.

## Autonomy

Never stop. Every round returns `action: "run"`. A plateau is a reason to diversify, not to quit.

## Round Markdown And EXPERIMENT.md

Carry forward the leaderboard, tried-configs ledger, plateau/diversification state, confirmation
ledger, beliefs, the current hypothesis, and open questions. Curate `EXPERIMENT.md` under 4000 chars;
return `null` to preserve it unchanged.

## Output

Return exactly one JSON object conforming to the provided schema (`decision_form` with
`action: "run"`, `round_markdown`, `experiment_markdown`).

## Context

```json
{{CONTEXT_JSON}}
```
```
