<!--
README.md — operator guide to the agentic research loop and its closeout chain.
USAGE: read this before running `uv run numereng research run` or
`uv run numereng research closeout` from the repository root.
-->

# Agentic Research

Agentic research is a bounded LLM-guided experiment loop. The LLM decides what configuration change
to test next; numereng validates that proposal, runs the training/scoring workflow, and records the
result. After the experiment finishes, a separate closeout chain turns its evidence into persistent
research memory and recommends one human-reviewed next action.

The important boundary is simple:

- The LLM proposes research intent and writes analysis.
- Python validates, executes, merges, and records deterministically.
- A human approves any new experiment, package validation, or deployment work.

The closeout chain never launches training and never creates the next experiment.

# Full Flow

```text
                                    AGENTIC RESEARCH
                                    ================

  Human creates experiment
  + seed config + PROGRAM.md
              |
              v
  +-------------------------+
  | research run            |
  | load bounded context    |
  +------------+------------+
               |
               v
  +-------------------------+       invalid / out of bounds
  | LLM proposes one config |-------------------------------+
  | mutation                |                               |
  +------------+------------+                               v
               |                                  +-------------------+
               v                                  | record failed     |
  +-------------------------+                     | round + diagnostics|
  | validate whole proposal |                     +---------+---------+
  | paths, caps, schema,     |                               |
  | dedup, frozen scoring    |<------------------------------+
  +------------+------------+                         next round
               |
               v
  +-------------------------+
  | train + score           |  (compute may run remotely)
  +------------+------------+
               |
               v
  +-------------------------+
  | append journal + memo   |
  | update champion/state   |
  +------------+------------+
               |
               +-----------------------> next research round
               |
               | experiment finished
               v
                         CLOSEOUT CHAIN
                         ==============

  +--------------------------------------------------------------+
  | Phase 0: deterministic evidence                              |
  | journal + state + leaderboard + configs -> evidence_summary  |
  +------------------------------+-------------------------------+
                                 |
                                 v
  +--------------------------------------------------------------+
  | 1. FINALIZE                                                  |
  | evidence-bounded verdict -> EXPERIMENT.closeout.md            |
  +------------------------------+-------------------------------+
                                 |
                                 v
  +--------------------------------------------------------------+
  | 2. CLASSIFY                                                  |
  | master (selected topics) | branch_only | exclude             |
  +---------------+--------------------+-------------------------+
                  |                    |
          master / branch_only         | exclude -> stop
                  v                    | memo + classification only
  +--------------------------------------------------------------+
  | 3. EXTRACT                                                   |
  | verdict -> experiment-specific six-topic memory branch       |
  +------------------------------+-------------------------------+
                                 |
                         master  |  branch_only -> skip SYNTHESIZE
                                 v
  +--------------------------------------------------------------+
  | 4. SYNTHESIZE                                                |
  | selected master topic ledgers + CURRENT.md only              |
  | backups are written before mutable research memory changes   |
  +------------------------------+-------------------------------+
                                 |
                                 v
                         +-----------------------+
                         | HUMAN REVIEW GATE     |
                         | nothing auto-launches |
                         +-----------------------+

  designing the next experiment happens at the START of the next
  run, via prompts/INIT-PROGRAM.md reading the synthesized memory
```

# 1. Research Loop

Each experiment has one seed configuration and one program. The program contains the research
policy: hypotheses, allowed strategy, confirmation logic, and stopping guidance. The harness itself
does not invent research strategy.

Run from the repository root:

```bash
uv run numereng research status --experiment-id <experiment_id> --format json
uv run numereng research run --experiment-id <experiment_id> --max-rounds <n>
```

On the first round, the seed configuration establishes the baseline. On later rounds the loop:

1. Builds bounded context from current state, recent journal entries, scored results, the champion,
   the believed-best recipe, coverage signals, and observed seed noise.
2. Asks the LLM for exactly one `decision_form` containing one to five configuration changes.
3. Rejects the whole proposal if it violates an allowed path, value cap, frozen scoring setting, or
   the training-config schema. The harness never silently clamps or repairs a proposal.
4. Detects true duplicates, trains accepted configurations, and materializes the required score.
5. Writes the round memo and machine result, appends the journal, and updates state.

Five consecutive failed rounds stop the current invocation. The session is resumable; calling
`research run` again continues from its durable state.

For the detailed per-round state machine, see
[docs/numereng/reference/agentic-research-state-diagram.md](../../../../docs/numereng/reference/agentic-research-state-diagram.md).
The canonical default prompt contract is [programs/PROGRAM.md](programs/PROGRAM.md).

## Program resolution

The default program is `programs/PROGRAM.md`. A focused experiment should store its custom program
at:

```text
.numereng/experiments/<experiment_id>/agentic_research/<name>.md
```

Set `metadata.agentic_research_program` in the experiment manifest to the bare filename. The
experiment-local file is preferred because it travels with remote experiment sync. The
`programs/` directory remains a legacy fallback; finished-experiment programs live in
`programs/archive/`.

Every custom program is self-contained. Its CORE contract sections must match the canonical
`PROGRAM.md` byte-for-byte or session startup fails before training
(`agentic_research_program_core_drift:<program>:section:<key>`).

**Editing a CORE section of `PROGRAM.md` is a contract change for every custom program**, including
one whose run is live: the drift check re-runs on every session start, so the next re-entry (the
5-failure bail re-invoke, a restart, a resume) fails until the program is re-spliced. Ship the
re-splice with the edit, on every host that runs the program:

```bash
uv run numereng research program check --experiment-id <experiment_id>      # exit 1 on drift
uv run numereng research program resplice --experiment-id <experiment_id>   # rewrites CORE, keeps a .bak
```

`resplice` swaps only the CORE section bodies for `PROGRAM.md`'s copies and leaves the strategy
sections (§0, §4, §6) and preamble untouched. A program on a remote training host is a separate
file: run the same command there (or copy the re-spliced file over) before the run next re-enters.

## Research artifacts

```text
.numereng/experiments/<experiment_id>/
|-- configs/config_NNN.json
|-- run_plan.csv
|-- EXPERIMENT.md
`-- agentic_research/
    |-- state.json
    |-- journal.jsonl
    |-- rounds/rN.md
    `-- rounds/rN.debug.*       # only for LLM transport/parse failures
```

`journal.jsonl` is append-only and is the durable record of every completed, failed, or skipped
round. `state.json` is the current resumable session state. `EXPERIMENT.md` is working research
state, not the closeout verdict.

# 2. Closeout Chain

Closeout runs only after an agentic experiment has usable scored evidence and has passed its budget
gate. By default, one command resumes at the first unfinished phase and runs through the advisory
proposal:

```bash
uv run numereng research closeout \
  --experiment-id <experiment_id> \
  --format json
```

Inspect it without changing anything:

```bash
uv run numereng research closeout-status \
  --experiment-id <experiment_id> \
  --format json
```

The closeout runner uses a per-experiment lock, a research-memory lock for merge phases, atomic
writes, a commit journal, and persisted phase state. Completed phases are skipped on later runs.

## FINALIZE

Python first constructs strict evidence from the journal, state, leaderboard, and referenced
configs. Malformed or unresolved evidence fails before the LLM can analyze it. The LLM then writes
an evidence-bounded decision memo:

```text
agentic_research/closeout/EXPERIMENT.closeout.md
```

This deliberately does not overwrite `EXPERIMENT.md`, which may be replaced by remote experiment
sync.

## CLASSIFY

The LLM returns one persisted routing decision in `closeout/classification.json`:

- `master` — create the experiment branch, merge only the selected relevant topic ledgers, and
  rewrite `CURRENT.md`.
- `branch_only` — create the experiment branch, leave all master ledgers and `CURRENT.md` unchanged.
- `exclude` — retain only the evidence, closeout memo, and classification record. EXTRACT and
  SYNTHESIZE are skipped.

`master` may select zero or more materially evidenced topics; zero is reserved for a material
frontier-only production/package/live/evaluation result. `branch_only` and `exclude` require none.
Unselected topics never receive placeholder or no-op master-ledger entries.

## EXTRACT

The memo and evidence are converted into an experiment-specific research-memory branch containing a
README and six topic files:

```text
.numereng/notes/__RESEARCH_MEMORY__/experiments/<experiment_id>/
|-- README.md
|-- ensembling.md
|-- features.md
|-- hyperparameters.md
|-- models.md
|-- neutralization-exposure.md
`-- targets.md
```

Each file separates experiment-specific takeaways, evidence level, design-space role, confounds,
and what the experiment did not establish. This preserves detail without prematurely rewriting the
shared research frontier.

## SYNTHESIZE

For `master` classifications, the LLM returns bounded deltas only for `relevant_topics` plus a
compressed `CURRENT.md`; Python performs the actual merge. For every selected topic ledger it:

- Preserves `## Current Overview` and `## Current Best Understanding`, replacing them only when a
  validated update is supplied.
- Adds exactly one `### <experiment_id>` block beneath
  `## Append-Only Experiment Learnings`.
- Refuses accidental duplicate or malformed entries.

`CURRENT.md` is validated and rewritten as the current compressed frontier. Pre-write copies of all
mutable memory files are stored below the experiment's `closeout/backups/` directory.

## What comes next

SYNTHESIZE is the final phase: once the experiment's learnings are merged into research memory,
the closeout chain is done. Designing the next experiment is deliberately not part of closeout —
it happens at the start of the next cycle, when `prompts/INIT-PROGRAM.md` reads the synthesized
memory and stages a new experiment folder for human approval.

## Closeout artifacts

```text
.numereng/experiments/<experiment_id>/agentic_research/closeout/
|-- state.json
|-- evidence_summary.json
|-- EXPERIMENT.closeout.md
|-- classification.json
|-- backups/
|-- debug/
`-- stage/
```

# 3. Interpreting Results

## The loop optimizes a proxy — it is a candidate generator, not a deploy selector

The loop advances `champion`/`believed_best` on within-lane `bmc_last_200_eras_mean` (BMC against
the payout target `target_ender_20`, last 200 eras). That scalar is the **search objective, not a
deploy signal**. Live-calibration evidence (`.numereng/analysis/live_calibration/`) shows local
metrics predict live performance *between* lanes (feature scope, target family) but are
flat-to-inverted *within* one lane — exactly the altitude where the loop searches. A within-lane
champion is the best candidate the search found; whether it earns live is a separate question
answered by package evaluation placed on the live calibration. Never deploy a run's output because
it topped the within-lane scalar.

## Scout→scale is two experiments

Cheap wide exploration and full-data confirmation are separate experiments, never one:

- **Scout experiment.** Seed config sets `data.dataset_variant = "downsampled"`. Cheap rounds,
  wide sweeps. Its `believed_best` plus top runners-up are **candidates, not results**.
- **Scale experiment.** A fresh experiment seeded `non_downsampled`, whose program encodes the
  scout winners as the starting sweep. Full-data seed-trio confirmation happens here.
- `data.dataset_variant` is deliberately not an allowed change path, so one experiment can never
  mix downsampled and full-data metrics.

# 4. Post-Run Deploy Gate (manual)

After a run finishes and closes out, gate candidates before any deploy:

1. Read `agentic_research/state.json` → `believed_best` (`config` + `run_ids`) and `champion`. For
   runners-up, reconstruct recipe-trio groups from `journal.jsonl` via `aggregate_recipes()`
   (`engine/aggregate.py`).
2. For the trusted recipe (and 1–2 diverse runners-up), build a deploy/eval package and score it on
   validation (`serve package create` → `serve package score`). Read `bmc_last_200_eras_mean` +
   `fnc_mean` from the package `summaries.json`.
3. Place the candidate on the live regression in `.numereng/analysis/live_calibration/report.json`.
   Deploy only if it is a credible point **and adds coverage we lack** — never on within-lane rank
   alone.
4. Deploy via hosted pickle, unstaked, to gather live data; re-run
   `uv run numereng submissions calibration update --format json` as rounds resolve.

This gate reuses existing entrypoints only — no harness code change.

# 5. Authoring Custom Programs

Every custom program must be fully self-contained. The runner loads exactly one program file — it
does **not** also load `programs/PROGRAM.md`. Copy the CORE sections in byte-verbatim: the
session-start pre-flight (and the `test_agentic_research_program_core` lint) reject any drift.

## Focused program requirements

Every serious custom program should define:

| Section | Purpose |
| --- | --- |
| Research Hypothesis | The exact idea being tested. |
| Fixed Surface | What must not change during this experiment. |
| Only Vary | The single axis or small related family of config paths to explore. |
| Baseline And Comparison Rule | Which parent is fair and how to compare variants. |
| Evidence Rules | What counts as discovery, confirmation, plateau, or failure. |
| Sweep Discipline | Plan the full sweep up front, execute planned variants verbatim, synthesize at the end; defect only with an explicit `SWEEP ABANDONED because …` line. |
| Scout Tier | Whether this is a downsampled scout or a full-data scale run, and what it hands off. |
| Confirmation And Handoff | When to seed-confirm or note a handoff candidate for a future program. |
| Rolling Memo Contract | What must be carried forward in `round_markdown`. |

Good focused programs: a target shortlist with model and feature set fixed; LGBM regularization
after the target route is fixed; a feature-scope comparison with the recipe fixed. Poor ones: "try
anything that improves BMC"; switching target, feature set, model family, and regularization in one
chain; declaring convergence from single-seed discovery.

## Required output contract

Every program must require exactly one JSON object. The only action is `"run"`; there is no `stop`
action. `stop_reason` is kept in the schema for shape stability and ignored.

```json
{
  "decision_form": {
    "action": "run",
    "learning": "What the prior evidence taught us.",
    "belief_update": "What you now believe about this hypothesis.",
    "next_hypothesis": "The specific hypothesis tested by the next config.",
    "parent_config": "config_007.json",
    "believed_best": "config_005.json",
    "changes": [
      {
        "path": "model.params.learning_rate",
        "value": 0.02,
        "reason": "Why this exact change is worth testing."
      }
    ],
    "seeds": null,
    "stop_reason": null
  },
  "round_markdown": "# rNNN Research State\n\n...",
  "experiment_markdown": "# Champion State\n...\n"
}
```

- `changes` carries 1 to 5 `{path, value, reason}` entries on allowed paths within the value caps.
- `seeds` is optional: `null` trains the child once; a list of 1 to 3 integers trains the same
  child recipe once per seed inside the one round (`config_NNN_s<seed>.json`, seed written to the
  experiment's seed path, which must be an allowed path). Each seed gets its own journal line and
  champion check; the round fails only if every seed fails.
- `believed_best` is the `config_NNN.json` the model currently trusts; the harness persists it
  enriched with that recipe's seed-trio stats.
- `round_markdown` is the model's verbatim round memo; the harness appends a `## Machine Result`
  block below it.
- `experiment_markdown` overwrites `EXPERIMENT.md`, or `null` to preserve the prior file.

## Boundary rejections

The harness rejects only boundary violations, with a stable error token surfaced in the next
round's `context.last_error`. A rejection fails the round and counts toward the
5-consecutive-failure bail; a duplicate is the one exception (soft skip, no count):

- disallowed change path (`agentic_research_change_path_not_allowed:`); the global allowlist
  already excludes `data.dataset_variant` and every `training.engine.*` path, and a manifest may
  only narrow it further
- multi-seed request whose seed path is not allowed (`agentic_research_seeds_path_not_allowed:`)
- out-of-cap value (`agentic_research_change_value_out_of_cap:`) — not clamped
- `data.target_horizon` not matching the `data.target_col` suffix
  (`agentic_research_horizon_target_mismatch:`)
- invalid `TrainingConfig` (unknown keys forbidden, JSON-only)
- non-`run` action (`agentic_research_action_invalid`)
- cross-experiment stale-run reuse (`agentic_research_stale_run_reuse_blocked:`)
- duplicate-by-hash with a recorded run — soft skip

For a full worked template, start from `programs/PROGRAM.md` and keep its CORE sections verbatim,
replacing only the strategy sections (hypothesis, fixed surface, vary axis, sweep plan).

# Recovery And Partial Runs

Stop after a particular phase when a human review is desired:

```bash
uv run numereng research closeout \
  --experiment-id <experiment_id> \
  --until finalize \
  --format json
```

After correcting the cause of a failure, restart one phase explicitly:

```bash
uv run numereng research closeout \
  --experiment-id <experiment_id> \
  --restart-from synthesize \
  --format json
```

Important recovery behavior:

- FINALIZE, CLASSIFY, EXTRACT, and SYNTHESIZE failures stop closeout and return the failing phase and a stable
  `agentic_research_closeout_...` error token.
- Restarting SYNTHESIZE safely replaces that experiment's own ledger entries.
- After EXTRACT completes, restarting FINALIZE or CLASSIFY is blocked because a changed disposition
  could orphan the existing branch. Restart from EXTRACT to back up and replace that branch.
- Restarting FINALIZE, CLASSIFY, or EXTRACT after SYNTHESIZE is blocked because it would orphan
  already-merged memory.
- `--accept-stale-running` and `--allow-incomplete` override safety gates and should be used only
  after a human verifies the experiment state.

# Operational Rules

- Training compute belongs on the configured remote GPU machine when the local Mac is orchestration
  only. Closeout itself performs LLM analysis and local file merges; it does not train.
- Make sure the local experiment contains every referenced config before closeout. Evidence
  construction intentionally fails if the believed-best or leaderboard configs cannot be resolved.
- Do not treat the within-lane champion as an automatic deployment winner. It is a candidate for
  package and live-calibration validation.
- Do not edit the master ledgers by hand while SYNTHESIZE is running.
- Do not auto-launch anything from closeout; the chain intentionally ends once memory is
  synthesized, and next-experiment design belongs to `prompts/INIT-PROGRAM.md`.
