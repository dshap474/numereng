<!--
README.md — operator guide to the agentic research loop, its prompt, and its closeout.
USAGE: read this before running `uv run numereng research run` or
`uv run numereng research closeout` from the repository root.
-->

# Agentic Research

Agentic research is a bounded LLM-guided experiment loop. Each round the model reads a bounded
context, proposes one config change, and numereng validates it, trains and scores it, and records
the exact result. The model carries the research strategy; the harness carries the boundaries.

- The model proposes research intent and writes the round memo and `EXPERIMENT.md`.
- Python validates, executes, and records. It never edits a proposal and never stops the run.
- A human creates the experiment, launches the loop, halts it, and approves anything downstream.

# The Prompt

Two files compose the prompt every round:

```text
programs/PROGRAM.md   {{STRATEGY}}      <- the experiment brief
                      {{CONTEXT_JSON}}  <- the bounded context
```

`PROGRAM.md` is tracked and holds the harness contract and the generic research doctrine: the
metric table and the BMC-versus-FNC rule, the frozen evaluator, the mechanical champion and the
seed trio, search discipline, the memo and `EXPERIMENT.md` contracts, the output schema, and the
context glossary. Editing it reaches every run at its next round. There is no per-experiment copy
of it and no re-splice step.

The brief holds what differs per experiment and lives at
`.numereng/experiments/<id>/agentic_research/STRATEGY.md`, a fixed filename that remote experiment
sync carries. An experiment with no brief falls back to the tracked generic
[programs/STRATEGY.md](programs/STRATEGY.md), whose headings a real brief follows:

- `## This Experiment`, opening with the hypothesis in one falsifiable sentence.
- `### Lane` — the fixed surface (feature set, target, dataset variant, profile) and what the
  allowed change paths leave mutable.
- `### Prior Evidence` — closed lanes, retired claims, inert axes, anchors, and the calibration
  stance, written as standalone prose. The harness injects no research memory, so everything the
  model should know from earlier work goes here.
- `### Sweep Plan` — which knob families to probe, in what order, at what step size.
- `### Confirmation And Handoff` — how a candidate is confirmed and what the run hands forward.

Substrate facts for the model family — legal shapes, enum values, host limits — belong in the brief
too. `research status` reports the resolved `strategy_path`.

# Running The Loop

```bash
uv run numereng research status --experiment-id <experiment_id> --format json
uv run numereng research run --experiment-id <experiment_id> --max-rounds <n>
```

`research run` initializes its own state on first use. With no scored primary metric yet, the first
round trains the seed config as the baseline; every later round goes through the model. Each round:

1. Builds bounded context: state, the recent journal, the recipe leaderboard, coverage, binding
   caps, observed seed noise, the previous memo, and the current `EXPERIMENT.md`. No term grows
   with round count.
2. Asks the model for one `decision_form` carrying one to five config changes and optionally one to
   three seeds.
3. Materializes the config, rejecting whole any proposal that leaves the allowed paths, exceeds a
   value cap, mismatches horizon and target, or fails the training-config schema. Nothing is
   clamped or repaired.
4. Trains and scores, appends the journal, writes the round memo, and updates the champion and
   `believed_best`.

The session is resumable. Five consecutive failed rounds end the invocation with
`stop_reason=consecutive_failures:5`; calling `research run` again continues from durable state. A
run ends on the round budget, a human stop, or that bail — there is no stop action.

## The retry

A boundary rejection and a duplicate config are both errors the model can fix in seconds. The
rejection token comes back once as `context.last_error`, and the model re-proposes inside the same
round. If the second proposal also fails, the round is recorded as failed (or skipped, for a
duplicate) and counted as before. The first token lands in the round memo's `## Machine Result`
block as `retry: <token>`.

In a multi-seed round, seeds that fail to materialize on their own are skipped and the round
proceeds with the rest.

For the per-round state machine, see
[docs/numereng/reference/agentic-research-state-diagram.md](../../../docs/numereng/reference/agentic-research-state-diagram.md).

# Reading Results

```text
.numereng/experiments/<experiment_id>/
|-- configs/config_NNN.json
|-- run_plan.csv
|-- EXPERIMENT.md
`-- agentic_research/
    |-- STRATEGY.md
    |-- state.json
    |-- journal.jsonl
    |-- rounds/rNNN.md
    `-- closeout/
```

`journal.jsonl` is append-only, one line per round attempt, carrying status, parent and child
config, seed, `metric` (BMC200), `fnc`, `benchmark_corr`, run id, and wall time. `state.json`
(schema v2) is the resumable session: status, counters, champion, `believed_best`, heartbeat, last
error. `rounds/rNNN.md` is the model's verbatim memo with the harness's `## Machine Result` block
appended, and `rounds/rNNN.debug.*` appears only after an LLM transport or parse failure.
`EXPERIMENT.md` is the model's curated working set, not a verdict.

`aggregate_recipes()` in `engine/aggregate.py` rebuilds the recipe-trio groups from the journal when
you want runners-up rather than only `believed_best`.

## The loop optimizes a proxy

`champion` and `believed_best` advance on within-lane `bmc_last_200_eras_mean`, BMC against the
payout target `target_ender_20` over the last 200 eras. That scalar ranks candidates inside one
lane. Live calibration (`.numereng/analysis/live_calibration/`) shows local metrics predict live
performance between lanes and stay flat to inverted within one — the altitude where the loop
searches. A within-lane champion is the best candidate the search found, not a deploy decision.

## Scout and scale are two experiments

A scout seeds `data.dataset_variant = "downsampled"`, runs cheap wide sweeps, and produces
candidates. A scale run is a separate experiment seeded `non_downsampled`, whose brief opens with
the scout's winners; full-data seed-trio confirmation happens there. `data.dataset_variant` is not
an allowed change path, so one experiment can never mix the two scales.

# Closeout

One command, once the run is down:

```bash
uv run numereng research closeout --experiment-id <experiment_id> --format json
```

It refuses an experiment whose state is still `running` or whose round budget is unspent;
`--allow-incomplete` waives both gates. It then builds the deterministic evidence bundle from the
journal, state, leaderboard, and referenced configs — including the one-time sealed-holdout
opening, which scores the believed-best runs on the frozen holdout eras once and seals them — and
asks the model for one decision memo bounded by that evidence.

```text
agentic_research/closeout/
|-- evidence_summary.json
|-- EXPERIMENT.closeout.md
|-- holdout_result.json     # only when a holdout was frozen
|-- finalize_response.md
`-- debug/
```

The memo carries a verdict, evidence and gaps, candidates against the mechanical champion, metric
conflicts rated by severity, a search audit, design-space roles, implications for the next
experiment, and recommendations for research memory. A response without `## Verdict`, or shorter
than 1,500 characters, is rejected and the raw text is left beside the evidence. Re-running
overwrites both artifacts.

Evidence construction fails if the believed-best or a leaderboard config cannot be resolved, so pull
the experiment back from the training host before closing out. Closeout itself writes no research
memory, trains nothing, and launches nothing.

# After Closeout

**Research memory.** Run the `research-memory-update` skill with the experiment id. It reads the
memo and the evidence, decides whether the experiment reaches master, its own branch, or nothing,
writes the branch under `.numereng/notes/__RESEARCH_MEMORY__/experiments/<id>/`, appends to the
topic ledgers it materially evidenced, and rewrites `CURRENT.md`.

**Deploy gate.** A candidate recipe earns a live slot through package evidence, not within-lane
rank: build and score a package for `believed_best` and one or two diverse runners-up, place them on
the live regression in `.numereng/analysis/live_calibration/report.json`, and deploy unstaked only
when the candidate is credible and adds coverage the fleet lacks. The
`numerai-package-validation` skill runs this gate end to end.

**The next experiment.** [prompts/INIT-PROGRAM.md](prompts/INIT-PROGRAM.md) is the pre-run playbook.
An agent working it compresses research memory into a dossier, writes one proposal, has a critic on
a different vendor's model attack it in one exchange, then creates the experiment folder with its
brief, seed config, manifest metadata, and design record. It stops at a human launch gate and never
trains, launches, or deploys.

# Operational Rules

- Training compute belongs on the configured remote GPU machine when the Mac is orchestration only.
  Closeout runs one LLM call and local file writes; it does not train.
- Confirm which experiment is live before launching a round; `research run` mutates durable state.
- Do not treat the within-lane champion as a deployment winner.
- Do not hand-edit `state.json` or `journal.jsonl`. The journal is the audit trail.
