<!--
README.md — the agentic research loop: its prompt, its round, its closeout, and the rules for
changing its code.
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
too. A brief carries neither placeholder.

`programs/` tracks only `PROGRAM.md`, `STRATEGY.md`, and this repository's gitignore rules for
them. Everything else there, `archive/` included, is local-only history and nothing in it is loaded
at run time.

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
run ends on the round budget, a human stop, or that bail — there is no stop action. No status is
terminal across invocations: `run_research` writes `status=running` on entry, so only a supervisor
that stops calling it actually ends the run. `last_heartbeat` is written every round, which is how
`research status` tells a live session from one whose host died still marked `running`.

## Round outcomes

A round's status comes from its seed outcomes. The primary outcome — the best completed seed, else
the last — is what the memo, the state, and the returned result speak for.

- **completed** — at least one seed trained and scored. The champion advances per completed run when
  `metric > champion.metric`, one mechanical comparison with no margin. Resets the failure counter.
- **skipped** — no seed completed and at least one was a duplicate-by-hash soft skip. Resets the
  failure counter and does not count toward the bail.
- **failed** — everything else: an LLM transport failure, an unreadable or wrongly shaped response, a
  non-`run` action, a boundary rejection, or a training or scoring failure. Increments the counter.

In a multi-seed round, seeds that fail to materialize on their own are skipped and the round
proceeds with the rest.

## The retry

A boundary rejection, a duplicate config, and a response the parser cannot read are all errors the
model can fix in seconds. The token comes back once as `context.last_error` — the boundary's token,
or `llm_response_invalid:<reason>` for an unparseable response — and the model re-proposes inside
the same round. If the second attempt also fails, the round is recorded as failed (or skipped, for a
duplicate) and counted as before. The first token lands in the round memo's `## Machine Result`
block as `retry: <token>`.

# Reading Results

Everything below is written under `.numereng/experiments/<experiment_id>/`.

| Path | Writer | Trigger |
| --- | --- | --- |
| `agentic_research/state.json` | `save_state` | each round |
| `agentic_research/journal.jsonl` | `_finalize_round` | one line per seed outcome, at least one per round attempt (append-only) |
| `agentic_research/rounds/rNNN.md` | `_finalize_round` | each round: the model's memo verbatim plus the `## Machine Result` block, with `retry:` and per-seed lines when present |
| `agentic_research/rounds/rNNN.debug.*` | failure debug dump | LLM transport or parse failures only |
| `agentic_research/closeout/` | `research closeout` | once the run is down |
| `EXPERIMENT.md` | passthrough write | each round the model returns a non-null `experiment_markdown` |
| `configs/config_NNN.json` | `materialize_config` (`baseline_config` on the baseline round) | each accepted seed; `config_NNN_s<seed>.json` in a multi-seed round |
| `run_plan.csv` | run-plan recorder | each round that trains a run |

A journal line carries status, parent and child config, seed, `metric` (BMC200), `fnc`,
`benchmark_corr`, run id, and wall time. `state.json` is the resumable session at
`schema_version: 2`: `experiment_id`, `status`, `next_round_number`, `total_rounds_completed`,
`failed_rounds_counter`, `stop_reason`, `champion {config, run_id, metric, round} | null`,
`believed_best`, `believed_best_changed_round`, `last_round_label`, `last_run_id`,
`last_checkpoint`, `last_error`, `last_heartbeat`, `created_at`, `updated_at`. An older state loads
through `apply_state_defaults`, which fills missing keys and drops a stale `best_overall`.
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

# Changing This Code

## Layout

- `engine/` — all harness code: `types.py`, `memory.py`, `aggregate.py`, `boundary.py`, `llm.py`,
  `context.py`, `loop.py`, plus `engine/closeout/` (`evidence.py`, `runner.py`, `types.py`), which
  turns a finished run into an evidence bundle and one decision memo.
- `prompts/` — `closeout-finalize.md`, the one closeout LLM call, plus the pre-run
  `INIT-PROGRAM.md` playbook that designs and stages the next experiment from research memory.
- `programs/` — the round prompt and the generic brief, as described above.

## The one boundary

The LLM proposes one `decision_form` per round; Python validates, executes, and records — it never
edits a proposal. The harness holds no research strategy: what to try, when to confirm, when to
diversify all live in `PROGRAM.md` and the experiment's brief. If a change adds strategy to Python,
it is in the wrong place.

## Invariants to preserve

- **Bounded context.** No term assembled by `engine/context.py` may grow with round count. This
  killed a 500-round run once (prompts grew to ~890 KB and the API stream-disconnected).
- **No auto-stop.** The only LLM action is `"run"`. Runs end on CLI budget, human stop, or the
  5-consecutive-failure bail (resumable).
- **Reject whole, never clamp.** Boundary violations (path allowlist, value caps, horizon/target
  match, strict `TrainingConfig`) fail the round with a stable error token; see `engine/boundary.py`.
  The one in-round retry hands that token back as `last_error` and re-asks; it never repairs a
  proposal, and a second failure is recorded and counted exactly as a single failure was.
- **The prompt is `PROGRAM.md` plus the experiment's `STRATEGY.md`**, composed at run time by two
  placeholder substitutions in `engine/llm.py`. Editing `PROGRAM.md` reaches every run, live ones
  included, at its next round. Keep both placeholders present exactly once and keep briefs free of
  them.
- **`data.dataset_variant` and `training.engine.*` are not LLM-mutable** — deliberately absent
  from `ALLOWED_CHANGE_PATHS` so one experiment can never mix downsampled and full-data metrics or
  move its own evaluator (profile, window, embargo). Manifests may only narrow the allowlist.
- **The seed path is data, not a name.** `agentic_research_seed_path` (manifest) drives seed
  injection, journal seed extraction (`loop._config_seed`), and recipe grouping
  (`aggregate.recipe_key`); do not hard-code `random_state`/`seed` anywhere new.
- **The payout target is owned by the scoring layer.** Context mirrors
  `features.scoring.metrics.DEFAULT_PAYOUT_TARGET_COL`; do not reintroduce a shadow constant here.
- **Dedup versus orphan.** A config hash that already has a recorded run in the journal is a true
  duplicate and soft-skips. A hash with no recorded run is a crash orphan, rewritten under this
  round's filename and run, so a mid-round crash cannot poison the hash and dead-end the search.
- **Stale-run reuse is same-experiment only.** Linking a FINISHED run on a hash collision is allowed
  within one experiment; a cross-experiment reuse hard-fails with
  `agentic_research_stale_run_reuse_blocked:`.
- **Scored or failed.** A round that links a FINISHED run must end with that run scored; a reused run
  with no primary metric on disk is rescored. Never complete a round with an unscored run.
- **Closeout never launches anything.** It ends at the evidence bundle and the memo; research-memory
  writes belong to the `research-memory-update` skill and next-experiment design to
  `prompts/INIT-PROGRAM.md`, both behind a human gate. Do not add auto-launch.
- **BMC200 is the search objective, not a deploy signal.** Never wire deploy automation to the
  within-lane champion.

## Verify

```bash
uv run pytest tests/unit/numereng -k agentic_research -q
uv run pytest tests/unit/numereng/agentic_research -q
```
