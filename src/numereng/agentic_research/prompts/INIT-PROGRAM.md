<!--
INIT-PROGRAM.md — operator-agent playbook for opening a NEW agentic research experiment.
Purpose: turn the repo's accumulated research memory into one designed next experiment — a new
`.numereng/experiments/<slug>/` folder holding the program file, seed config, manifest metadata, and
design record — via cross-model divergent proposal, synthesis, and adversarial debate, stopping at a
human launch gate.
Usage: read and execute this file end to end as an interactive coding agent with full tool access
(Claude Code / Codex) from the repository root. This is NOT a runner-substituted prompt: it carries no
runner context placeholder, no bounded context bundle, and no strict output schema — you gather your
own inputs.
Stage E creates the experiment folder and authors files into it; nothing in this playbook launches
training.
-->

# INIT-PROGRAM — Design The Next Agentic Research Experiment

You are the design operator for the next bounded agentic research run. Prior experiments left a
research memory; a closeout chain compressed it; nothing in that chain decides what to do next. That
is your job. You will assemble one bounded dossier, farm divergent proposals to four cross-vendor
frontier models, score and dedupe them against a fixed rubric, subject the winner to an adversarial
cross-model critique, and then create the experiment folder and author the program file, seed config,
manifest metadata, and design record inside it, fully staged for a human to launch. You do not launch
it.

Work the stages in order. Do not skip forward: Stage B is worthless without Stage A's dossier, and
Stage E is worthless without Stage D's dissent record. Every stage produces a durable artifact.

Stage working directory (managed scratch, safe to write): `.numereng/tmp/init-program/`. Persist the
dossier, each subagent's raw JSON reply, the synthesis memo, and the debate transcript there so
Stage E can quote them verbatim instead of reconstructing them from memory.

## Hard Constraints

These are facts about the harness and the machine, not preferences. Any proposal or seed config that
violates one is **invalid** — reject it in Stage C rather than repairing it, and say so in the memo.

- **Training profile must be `purged_walk_forward` in the seed config.** `training.engine.*` is not
  LLM-mutable (absent from the harness allowlist, so it cannot appear in
  `metadata.agentic_research_allowed_change_paths` either); the profile the seed config declares is
  the profile for the whole run. A `simple` profile leaks through a no-embargo holdout and
  systematically favors overfit large-capacity models.
- **The embargo must match the training-target horizon.** Eras are weekly. The profile derives the
  embargo from `data.target_horizon` (explicit, or inferred from the target name): 8 eras for a
  20-day target (~4-era overlap), 16 for a 60-day target (~12-era overlap).
  `training.engine.embargo_eras` is hard-rejected by the profile
  (`training_profile_disallows_custom_parameters`), and the boundary rejects a `data.target_col`
  whose horizon contradicts the lane's `data.target_horizon` — set the horizon explicitly in the
  seed config and never mix horizons within one lane.
- **The lane is fixed by the seed config.** `data.dataset_variant`, `data.feature_set`, and
  `data.target_col` are not LLM-mutable during the run. Choosing the lane is the single highest-stakes
  decision in this playbook; treat it as such.
- **Configs are strict JSON-only and reject unknown keys.** No YAML, no comments, no legacy fields.
- **LGBM trains on the GPU by default.** Leave `model.device`, `model.params.device_type`, and
  `model.params.tree_method` absent in the seed config: the harness injects `device_type = "gpu"`
  (the OpenCL LightGBM build) for `LGBMRegressor` and falls back to CPU only on hosts without a
  GPU-enabled LightGBM. Never set `"cuda"` (a different, crash-prone build), and keep the device
  keys out of the allowed change paths.
- **Seed trio is `42 / 17 / 99`; seed `42` is the discovery seed.** Confirmation is by trio mean,
  reached either one seed per round at the seed path or several seeds in one round via
  `decision_form.seeds`.
- **Primary metric is `bmc_last_200_eras.mean`; FNC is co-primary directional.** Within-lane BMC200
  is a candidate ranker. It is never a live-viability signal and never a deploy signal.
- **Package-scale local BMC200 and agentic per-era BMC200 are non-comparable surfaces.** There is no
  conversion between them. Do not import a package break-even number into a per-era target, and do not
  compare an agentic trio mean against a package validation figure. This is the single most common
  category error in this workflow.
- **The run never auto-stops.** It is budget-bounded and runs to `budget_rounds`; a human stops it
  manually on convergence. Plateau means diversify within the lane, never quit.
- **No staking, no incumbent replacement, no manual submission, no upload, and no experiment launch
  without explicit human approval in the current session.**
- **CORE program sections must match `programs/PROGRAM.md` byte-verbatim** — enforced at session start
  and by a drift-lint test. Copy them mechanically; never retype or paraphrase them.

## Stage A — Dossier Assembly

Primary agent only. Deterministic, no delegation, no judgment calls about what to try yet.

Read, in this order, all of:

- `.numereng/notes/__RESEARCH_MEMORY__/CURRENT.md` — the compressed frontier and the authoritative
  `## Current Constraints` block.
- `.numereng/notes/__RESEARCH_MEMORY__/topics/features.md`
- `.numereng/notes/__RESEARCH_MEMORY__/topics/targets.md`
- `.numereng/notes/__RESEARCH_MEMORY__/topics/models.md`
- `.numereng/notes/__RESEARCH_MEMORY__/topics/hyperparameters.md`
- `.numereng/notes/__RESEARCH_MEMORY__/topics/ensembling.md`
- `.numereng/notes/__RESEARCH_MEMORY__/topics/neutralization-exposure.md`
- `.numereng/notes/__RESEARCH_MEMORY__/scoring/live-local-calibration.md`
- `README.md` of the 2–3 most recently dated experiment branches under
  `.numereng/notes/__RESEARCH_MEMORY__/experiments/` (sort by the `YYYY-MM-DD_` prefix, newest first).

Compress all of it into ONE research dossier at `.numereng/tmp/init-program/DOSSIER.md`, using exactly
this template. Hard ceiling **8,000 words**; aim for 6,000. Compression is the deliverable — a dossier
that merely concatenates the sources has failed.

```markdown
# Research Dossier — <UTC date>

## 1. Frontier State
What we currently believe, what the standing champion/candidate situation is, which lanes are live
and which are dormant. Name the source file for each non-obvious claim.

## 2. Open Questions By Topic
One subsection per topic ledger (features, targets, models, hyperparameters, ensembling,
neutralization-exposure). For each: the unresolved question, why it is still open, and the concrete
next test the ledger itself proposes. Mark a topic `no open question` rather than inventing one.

## 3. Closed Lanes And Retired Hypotheses
Directions closed by evidence, each with the disconfirming evidence and the experiment id that closed
it. Also list retired claims — beliefs a later experiment overturned — so no proposal resurrects a
belief that memory has already withdrawn.

## 4. Live-Calibration Stance And Scale Anchors
The current local-vs-live fit, its observed domain, its between-lane vs within-lane behavior, and the
resolved-vs-provisional split. MANDATORY: state that package-scale local BMC200 and agentic per-era
BMC200 are non-comparable surfaces with no conversion, and that a fleet break-even figure is a
package-scale anchor that must never be set as a per-era objective.

## 5. Current Constraints (verbatim)
The `## Current Constraints` bullets from CURRENT.md, copied verbatim, unabridged, unreordered.

## 6. Candidate Inventory
Every named candidate config/recipe still in play: experiment id, config filename, recipe summary,
trio mean and FNC where known, evidence class (validation / package / live), and what gate it is
waiting on.
```

Stage A ends when `DOSSIER.md` exists and is internally consistent. Every subagent in Stage B receives
this identical dossier text and **must not independently re-read research memory** — divergence in
inputs would make their proposals incomparable. A proposal citing a fact absent from the dossier is
invalid.

## Stage B — Divergent Proposals

Spawn **four parallel subagents**, one per model, cross-vendor by design:

| Slot | Model | Reasoning effort |
| --- | --- | --- |
| B1 | Claude Fable | medium |
| B2 | Claude Opus | medium |
| B3 | GPT-5.6 Sol | xhigh |
| B4 | Grok 4.5 | high |

Use whatever multi-model harness is available in this session. If a model cannot be reached,
substitute the nearest available frontier model **from a different vendor than the other three** and
record the substitution in `DESIGN.md`. Never fill all four slots from one vendor — cross-vendor
disagreement is the mechanism this stage exists for.

Each subagent prompt contains exactly three payloads, in this order, and nothing else that could bias
the direction: (1) the Stage A dossier verbatim, (2) the doctrine digest in §B.1 verbatim, (3) the
proposal schema in §B.2. Instruct each subagent to return **one JSON object only**, no prose. Save each
raw reply to `.numereng/tmp/init-program/proposal-<slot>.json`.

### B.1 Doctrine Digest (paste verbatim into every subagent prompt)

This is the standing experiment-design doctrine for numereng agentic research. It is self-contained;
do not seek other sources.

**Metric hierarchy.** Primary: `bmc_last_200_eras.mean` (surfaced in-loop as
`bmc_last_200_eras_mean`). Tie-break: `bmc.mean`. Sanity checks, guidance only: `corr.mean`,
`mmc.mean`, `cwmm.mean`. FNC (`fnc_mean`) is a **co-primary directional** signal: it tracks live
durability better than raw BMC, so a change that raises BMC200 while materially dropping FNC is
suspect, and among BMC200 ties inside the seed-noise floor the higher-FNC recipe wins. FNC never
changes the harness's mechanical champion; it changes what you believe and what you confirm.

**Scout → scale is two experiments, never one.** A scout experiment seeds
`data.dataset_variant = "downsampled"`: cheap rounds, wide sweeps, and its outputs are **candidates,
not results**. A scale experiment is a fresh experiment seeded `non_downsampled` whose program encodes
the scout's winners as the opening sweep; full-data seed-trio confirmation happens only there.
`data.dataset_variant` is deliberately not mutable in-run, so one experiment can never mix downsampled
and full-data metrics. Scale only winners. Before concluding any line of inquiry, run at least one
full-data confirmatory step.

**Round shape and one-variable-at-a-time.** Manual rounds are 4–5 configs: one base plus
single-variable variants, each named for the variable it moves. The in-loop equivalent is
plan-then-execute: design one base plus 3–4 variants up front, queue them in the memo, emit the next
planned variant verbatim each round without re-deciding on a single result, and synthesize only after
the last variant lands. Abandoning an open sweep requires an explicit `SWEEP ABANDONED because …` line.
Confirmation is a separate tier from exploration: a sweep explores, a seed trio confirms.

**Seed discipline.** Trio is `42 / 17 / 99`; `42` is the discovery seed. A single seed beating the
best identifies a candidate, not a winner. Confirm by trio mean, not the luckiest seed. The seed-noise
floor prior is **~3e-4** on BMC200 until the harness measures the pooled per-seed SD; treat any gap
below the floor as noise. Trio-mean standard error is roughly `floor/√3`.

**Sweep selection by research type** (guidelines, not templates — judge what actually answers the
question):

| research_type | Sweep the… |
| --- | --- |
| `new_target_or_feature_engineering` | target variants or preprocessing settings; skip broad hyperparameter sweeps unless results are unstable |
| `new_architecture` | capacity, learning rate, regularization, and related controls |
| `ensemble_or_blend` | component selection, combination weights, blend rules, stacker settings |
| `training_procedure` | procedure controls (scoring stage, prediction-stage transforms, neutralization settings) |
| `data_change` | feature scope and dataset variant |
| `hyperparameter_frontier` | the specific knob family that prior evidence left unresolved |

**Sweep design guidance.** Take steps large enough to plausibly move the primary metric by more than
the seed-noise floor; a search made of sub-noise nudges only maps the noise floor and promotes lucky
seeds. Open with one coarse probe per knob family before refining anything, and refine only families
that already showed a supra-noise, FNC-clean effect. When scaling depth/width/leaves/estimators,
consider whether learning rate or regularization must move with them. Respect the LGBM leaf cap: with
`max_depth > 0`, `num_leaves` above `2 ** max_depth` is a no-op and usually collides as a duplicate —
raise `max_depth` first. Stop re-probing axes measured as inert.

**Baseline alignment.** Declare which baseline the experiment aims to improve on, keep every
comparison aligned to it, and include a baseline row in every results table. Change the baseline only
when the round is explicitly about changing it.

**Plateau and stop criteria.** A plateau is two consecutive rounds failing to beat the best
`bmc_last_200_eras.mean` by a meaningful margin — rule of thumb **~1e-4 to 3e-4** — *and* the remaining
untried knobs being either redundant with what was already swept or likely to raise overfit or
benchmark-correlation. In an agentic run a plateau is a mandate to **diversify within the lane**, never
to stop: the harness surfaces `rounds_since_new_believed_best`, and at `≥ 5` the next change must move
into a region absent from `coverage`. The run itself never auto-stops; it is budget-bounded and a human
halts it.

**Attributability and organization.** One experiment folder is one line of inquiry. One hypothesis per
experiment. Prefer simple, attributable changes (1–5 config values per round, one hypothesis); an
unattributable result wastes the round even when it scores well. Branch from the best comparable
parent, not automatically the previous round. When two recipes tie inside the noise floor, prefer the
cheaper one — round wall-clock is part of the budget.

**Evidence-class separation.** Validation-surface metrics, package-scale metrics, and live results are
three classes and never interchangeable. Agentic per-era BMC200 (order `1e-3`–`1e-2`) cannot be placed
on a package-scale calibration. Local metrics choose lanes and reject weak packages; within a lane they
do not reliably choose live winners among close siblings.

### B.2 Proposal Schema

Return exactly one JSON object with keys `primary` (required) and `second_choice` (an object of the
same shape, or `null`). Both proposals must satisfy every Hard Constraint.

| Field | Type | Meaning |
| --- | --- | --- |
| `title` | string | Short human title; will become the experiment slug. |
| `hypothesis` | string | One falsifiable sentence: what this run tests. |
| `mechanism` | string | Why it should move the primary metric — the causal story, not the restated hypothesis. |
| `lane` | object | `{"feature_set": string, "target_col": string, "model_family": string}` — fixed for the whole run by the seed config. |
| `research_type` | enum | One of `new_target_or_feature_engineering`, `new_architecture`, `ensemble_or_blend`, `training_procedure`, `data_change`, `hyperparameter_frontier`. |
| `varied_axis` | array[string] | The config paths this run actually explores; the single axis or one tight family. |
| `seed_recipe` | object | Concrete `{config path: value}` params for the seed config, including every param the run will vary. |
| `allowed_change_paths` | array[string] | Exact config paths the in-run LLM may mutate. |
| `value_caps` | object | `{config path: {"min": number, "max": number}}` bounds the harness will enforce. |
| `budget_rounds` | integer | Positive round budget, justified against per-round wall time. |
| `scout_or_scale` | object | `{"tier": "scout"\|"scale", "rung": string, "why": string}` — which rung of the ladder and why this rung now. |
| `success_criteria` | string | The observation that would make this run a success, expressed in supra-noise terms. |
| `kill_criteria` | string | The observation that would make a human stop the run early. |
| `evidence_basis` | array[string] | MANDATORY. Dossier facts motivating the proposal, each citing its dossier section (e.g. `"§2 targets: …"`). |
| `constraints_checked` | array[string] | MANDATORY. Each closed lane, retired hypothesis, and Current Constraints bullet verified against, with the verdict. |

```json
{
  "primary": {
    "title": "small xerxes20 subsample frontier",
    "hypothesis": "Raising row subsampling above the seeded 0.5 regime unlocks a supra-noise BMC200 gain in the small/xerxes_20 lane without costing FNC.",
    "mechanism": "At subsample=0.5 each tree fits on half the rows, so per-tree variance is high and the ensemble spends capacity averaging noise; feeding trees more rows should let regularized capacity express, and row-diverse trees plausibly hold FNC.",
    "lane": {"feature_set": "small", "target_col": "target_xerxes_20", "model_family": "LGBMRegressor"},
    "research_type": "hyperparameter_frontier",
    "varied_axis": ["model.params.subsample", "model.params.min_child_samples", "model.params.reg_lambda"],
    "seed_recipe": {
      "model.params.subsample": 0.5,
      "model.params.learning_rate": 0.01,
      "model.params.max_depth": 6,
      "model.params.num_leaves": 64,
      "model.params.min_child_samples": 500,
      "model.params.n_estimators": 2000,
      "model.params.reg_alpha": 0.0,
      "model.params.reg_lambda": 1.0,
      "model.params.random_state": 42
    },
    "allowed_change_paths": [
      "model.params.subsample",
      "model.params.min_child_samples",
      "model.params.reg_alpha",
      "model.params.reg_lambda",
      "model.params.max_depth",
      "model.params.num_leaves",
      "model.params.n_estimators",
      "model.params.learning_rate",
      "model.params.random_state"
    ],
    "value_caps": {
      "model.params.subsample": {"min": 0.3, "max": 1.0},
      "model.params.min_child_samples": {"min": 100, "max": 20000},
      "model.params.reg_lambda": {"min": 0.0, "max": 50.0}
    },
    "budget_rounds": 45,
    "scout_or_scale": {"tier": "scale", "rung": "full-data confirmation of a knob family a prior scout left unresolved", "why": "A downsampled scout would answer a question the prior run already localized; the open question is whether the effect survives full data."},
    "success_criteria": "At least one trio-confirmed recipe beats the seeded baseline trio mean by more than the observed seed-noise floor with FNC held or raised.",
    "kill_criteria": "Every knob family probes inert within the noise floor across the first coarse pass, indicating the lane is saturated at this substrate.",
    "evidence_basis": [
      "§2 hyperparameters: the subsample family is recorded as never probed above 0.5 in this lane.",
      "§3: the bare capacity ladder is closed by evidence at subsample 0.5, which is exactly the confound this proposal removes.",
      "§4: FNC is the stronger live predictor, so a knob family that moves FNC is worth the budget."
    ],
    "constraints_checked": [
      "Closed lane 'wide/deep-trees': not entered — lane is small/xerxes_20.",
      "Retired claim 'row subsampling always hurts small lanes': not relied on.",
      "Current Constraints 'no incumbent replacement': this run proposes no deploy action.",
      "Hard Constraint 'horizon-matched embargo': target_xerxes_20 at the inferred 20d horizon."
    ]
  },
  "second_choice": null
}
```

## Stage C — Synthesis

Primary agent. Score every proposal (primaries and second choices) against this fixed rubric.

1. **Constraint compliance — gate, not score.** Any Hard Constraint violation, any resurrection of a
   closed lane, or any conflict with a Current Constraints bullet **disqualifies** the proposal. Record
   the disqualification and the exact clause; do not repair it.
2. **Frontier fit (0–3).** Does it attack a real open question named in dossier §2, or invent one?
3. **Rung correctness (0–3).** Is scout-vs-scale right for the evidence state? A scale run on an
   unlocalized question burns budget; a scout run on an already-localized question wastes a cycle.
4. **Noise-floor awareness (0–3).** Are the proposed step sizes plausibly supra-noise against the
   ~3e-4 seed-noise prior, or is this a sub-noise nudge search?
5. **Attributability (0–3).** One hypothesis, one axis or one tight family, confounds not bundled. A
   proposal moving target, feature scope, model family, and regularization together scores 0.
6. **Cost/budget realism (0–3).** Does `budget_rounds` × plausible per-round wall time fit a bounded
   run on the available machine, and does the recipe's cost match its expected information?

Dedupe before ranking: proposals sharing a lane and an axis are one candidate. Merge them, keep the
stronger framing, and attribute both authors. Do not let repetition across models function as a vote —
four models converging on the same idea is weak evidence when they all read one dossier.

Write `.numereng/tmp/init-program/SYNTHESIS.md` containing:

- **Scoring table** — every proposal, its six rubric marks (or `DISQUALIFIED` + clause), total.
- **Dedupe ledger** — which proposals merged into which candidate.
- **Ranked shortlist** — top 3 candidates with one line each on why they rank there.
- **Recommended pick** — one candidate, with the decisive reason.
- **Grafts** — explicitly what was taken from runners-up into the pick (a better kill criterion, a
  tighter cap, a sharper mechanism), each attributed to its source proposal.
- **Rejected-but-notable** — ideas worth a future cycle, so they are not lost.

## Stage D — Adversarial Cross-Model Debate

Spawn one **critic** subagent from a different vendor than the recommended pick's author:

| Recommended pick authored by | Critic |
| --- | --- |
| Claude Fable / Claude Opus | GPT-5.6 Sol |
| GPT-5.6 Sol | Claude Fable |
| Grok 4.5 | Claude Fable |

The critic receives the dossier, all proposals, and the synthesis memo. Its mandate is **to attack, not
to agree.** An agreeable critique is a failed critique; if the critic finds nothing, it must say which
attack vectors it tried and why each failed. Required attack vectors:

1. **Memory misreads** — does the pick's `evidence_basis` actually say what it claims the dossier says?
2. **Category errors** — the #1 named trap: conflating package-scale BMC200 with agentic per-era
   BMC200, or importing a fleet break-even into a per-era objective. Also validation-vs-live conflation
   and treating a within-lane champion as a deploy signal.
3. **Confound bundling** — will the result be attributable, or does the axis move two things at once?
4. **Redundancy with closed hypotheses** — is this a re-run of something memory already closed or
   retired, wearing new vocabulary?
5. **Sub-noise sweep design** — are the step sizes and caps capable of producing a supra-noise result
   at all, given the ~3e-4 floor and the trio-mean standard error?
6. **A better neglected alternative** — name the strongest proposal or open question the synthesis
   under-weighted, and argue for it concretely.

Protocol, bounded at **two exchanges** — do not extend it:

1. **Critic attack.** Findings ranked most-severe first, each naming the specific clause or number it
   attacks.
2. **Primary rebuttal or revision.** For each finding: accept and revise, or reject with a reason. State
   revisions concretely (changed cap, changed axis, changed rung, changed pick).
3. **Critic final position.** `consensus` or `dissent`, with the residual objection stated in full.

Save the whole transcript to `.numereng/tmp/init-program/DEBATE.md`. The debate ends in consensus or in
documented dissent. **The primary agent decides either way**, and a dissent is carried **verbatim** into
`DESIGN.md`. Never paraphrase a dissent, never silently drop one, and never resolve one by fiat without
recording that you did. If the critic changes the pick, re-run Stage C scoring for the new pick before
proceeding; do not enter Stage E on an unscored candidate.

## Stage E — Experiment Creation And Program Authoring

Primary agent. Execute this checklist in order. Every deliverable lands inside a **new experiment
folder** under `.numereng/experiments/` — nothing is staged in `programs/` and nothing is left in
`.numereng/tmp/` as the artifact of record.

**1. Create the experiment.** Slug the pick as `<YYYY-MM-DD>_<short-slug>` and create it:

```bash
uv run numereng experiment create --id <experiment-slug> --hypothesis "<one-sentence H1>" --tags "agentic,<lane>"
```

This yields `.numereng/experiments/<experiment-slug>/` with `experiment.json` and `configs/`. Creating
the experiment folder is in-scope; **launching anything from it is not** — that stays behind the human
gate in step 6.

**2. Author the program** at the experiment-local path — the preferred location, since it travels with
remote experiment sync. Copy, do not write:

```bash
mkdir -p .numereng/experiments/<experiment-slug>/agentic_research
cp src/numereng/agentic_research/programs/PROGRAM.md \
   .numereng/experiments/<experiment-slug>/agentic_research/<experiment-slug>.md
```

Now edit **only** the title line, §0, §4, and §6 of the copy. Everything else stays byte-identical
because you never touched it. Precision points:

- CORE sections — `1.`, `2.`, `2.1`, `3.`, `5.`, `7.`, `8.`, `9.`, `10.`, and `Context` — are
  off-limits. A numbered level-3 heading (`### 2.1 …`) is its own CORE section. An un-numbered level-3
  heading stays inside its parent section, so `### Sweep-Plan-Then-Execute` is editable (it lives in
  §6) while the `### round_markdown` subsections are not (they live in CORE §7).
- **§0 is where the dossier gets baked in.** The harness injects **no** research memory at runtime:
  this program file is the run's only cross-experiment knowledge. Whatever the in-run model must know
  about closed lanes, retired claims, the scale non-comparability warning, prior inert axes, and the
  live-calibration stance has to be written into §0 or it does not exist for the run. Write §0 as
  standalone prose, not as citations to files the run cannot read.
- **§4 (Substrate And Budget)** states the pinned lane, that `data.feature_set` / `data.target_col` /
  `data.dataset_variant` are deliberately not mutable, and how to spend the round budget.
- **§6 (Search Discipline)** encodes the sweep plan: which knob families to probe coarsely first, the
  supra-noise step sizes, the plateau-diversify rule, the parsimony tie-break, and the axes already
  measured inert.
- Keep the CORE runner context placeholder intact — the double-brace token in the fenced JSON block at
  the end of the `Context` section, which the run harness string-replaces with the bounded round
  context. (This playbook carries no such token; the program you author must.)

**3. Author the seed config** at `.numereng/experiments/<experiment-slug>/configs/config_001.json`.
Obey every Hard Constraint: `training.engine.profile` = `purged_walk_forward`, a target whose
horizon matches the lane's explicit `data.target_horizon`, no `device`/`device_type`/`tree_method`
keys for LGBM lanes (the harness injects the GPU default), strict JSON with no unknown keys, and
the discovery seed 42 at the experiment's seed path (`model.params.random_state` for LGBM;
`metadata.agentic_research_seed_path` overrides it for model families that name the seed
differently, e.g. `model.params.seed`). Include every param the run will vary so `value_caps` have
something to bound.

**4. Set the manifest metadata.** There is no CLI flag for these keys; edit the `metadata` object of
`.numereng/experiments/<experiment-slug>/experiment.json` directly (the established mechanism) and set:

- `agentic_research_program`: the **bare filename** `<experiment-slug>.md` — the resolver prefers the
  experiment-local `agentic_research/` copy over `programs/`.
- `agentic_research_allowed_change_paths`, `agentic_research_value_caps`,
  `agentic_research_budget_rounds` — exactly as the design specifies.

Confirm with `uv run numereng experiment details --id <experiment-slug> --format json` that the
manifest still parses and the four keys are present.

**5. Write `DESIGN.md`** at the experiment root, `.numereng/experiments/<experiment-slug>/DESIGN.md`,
containing, in order: the dossier digest; all four subagents' proposals (including second choices and
any substituted model); the synthesis memo with the scoring table and grafts; the debate summary with
any dissent **verbatim**; and the final rationale naming the lane, the axis, the rung, and the budget.

**6. Verify.** Both checks must pass before the gate. The repo's CORE-drift pytest only lints
`programs/*.md`, so it cannot see an experiment-local program — verify it inline instead:

```bash
uv run python -c "
from pathlib import Path
from numereng.agentic_research.engine.types import first_diverging_core_section
base = Path('src/numereng/agentic_research/programs/PROGRAM.md').read_text()
prog = Path('.numereng/experiments/<experiment-slug>/agentic_research/<experiment-slug>.md').read_text()
diverged = first_diverging_core_section(prog, base)
print('CORE OK' if diverged is None else f'CORE DRIFT in section {diverged!r}')
"
uv run python -c "
from pathlib import Path
from numereng.config.training import load_training_config_json
load_training_config_json(Path('.numereng/experiments/<experiment-slug>/configs/config_001.json'))
print('seed config OK')
"
```

A CORE-drift failure means you edited a CORE section — re-copy `PROGRAM.md` and redo step 2 rather than
hand-patching the diff. A loader failure names a stable error token; fix the config, not the loader.

**7. Stop at the human gate.** Present the design in chat: the pick, its hypothesis and mechanism, the
lane, the varied axis, the rung, `budget_rounds`, the `allowed_change_paths` and `value_caps` now set in
the manifest, the success and kill criteria, any dissent, and the exact files you wrote. The experiment
is fully staged; the only thing left for the human to authorize is **launching the run**.

Do **not** run `numereng research init`, `numereng research run`, any `remote` or `cloud` command, or
any training, submission, upload, or staking command. Creating the experiment, authoring its files,
setting its manifest metadata, and running the two verification commands above is the entire extent of
this playbook's authority.

## Done Criteria

- `.numereng/tmp/init-program/` holds `DOSSIER.md`, four `proposal-*.json`, `SYNTHESIS.md`, and
  `DEBATE.md` (working artifacts; the experiment folder is the artifact of record).
- `.numereng/experiments/<experiment-slug>/` exists, created via `experiment create`.
- The program exists at `agentic_research/<experiment-slug>.md` inside it, CORE sections byte-verbatim
  and only the title, §0, §4, and §6 authored.
- `configs/config_001.json` exists, parses strictly, and violates no Hard Constraint.
- `experiment.json` carries the four `agentic_research_*` metadata keys and still parses.
- `DESIGN.md` exists at the experiment root and carries any dissent verbatim.
- Both verification commands pass.
- The session ends with the design presented and nothing launched.
