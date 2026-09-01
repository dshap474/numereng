# Design The Next Agentic Research Experiment

You are designing the next bounded agentic research run from the repo's research memory. Work from
the repository root with full tool access. The deliverable is one new experiment folder under
`.numereng/experiments/`, staged with its brief, seed config, manifest metadata, and design record,
and stopped at a human launch gate. You create the folder. You never launch the run, train, submit,
upload, or stake.

Work the four stages in order and persist each stage's artifact under
`.numereng/tmp/init-program/` so later stages quote it instead of reconstructing it.

## Hard Constraints

A proposal that violates any of these is invalid. Reject it rather than repairing it, and say so in
the design record.

- The seed config's training profile is `purged_walk_forward`. `training.engine.*` is not mutable
  during the run, so the seed config sets the profile for the whole run. A `simple` profile leaks
  through a no-embargo holdout and favors overfit large models.
- `data.target_horizon` is set explicitly and matches the target. The harness derives the embargo
  from it: 8 eras for a 20-day target, 16 for a 60-day target. Never mix horizons in one lane.
- The seed config fixes the lane. `data.dataset_variant`, `data.feature_set`, and `data.target_col`
  cannot change during the run, so choosing the lane is the highest-stakes decision here.
- Configs are strict JSON and reject unknown keys.
- LightGBM lanes leave `model.device`, `model.params.device_type`, and `model.params.tree_method`
  absent; the harness injects the GPU default. Never set `"cuda"`.
- The primary metric is `bmc_last_200_eras.mean` with FNC co-primary. Within-lane BMC200 ranks
  candidates and is never a live-viability or deploy signal.
- Package-scale local BMC200 and agentic per-era BMC200 are different surfaces with no conversion.
  Never import a package break-even into a per-era objective or compare an agentic trio mean with a
  package validation figure.
- The run never auto-stops. It runs to `budget_rounds` and a human halts it.
- No staking, incumbent replacement, submission, upload, or launch without explicit human approval
  in the current session.

## Stage 1: Dossier

Read, in order, under `.numereng/notes/__RESEARCH_MEMORY__/`: `CURRENT.md`, the six ledgers under
`topics/`, `scoring/live-local-calibration.md`, and the `README.md` of the two or three newest
branches under `experiments/`.

Compress them into `.numereng/tmp/init-program/DOSSIER.md`, at most 8,000 words, with these
sections:

1. **Frontier State**: current beliefs, the champion and candidate situation, live and dormant
   lanes. Name the source file for each non-obvious claim.
2. **Open Questions By Topic**: one subsection per ledger with the unresolved question, why it is
   open, and the next test the ledger itself proposes. Write `no open question` rather than
   inventing one.
3. **Closed Lanes And Retired Claims**: each with its disconfirming evidence and the experiment that
   closed it.
4. **Calibration Stance And Scale Anchors**: the local-versus-live fit, its domain, its
   between-lane versus within-lane behavior, and the statement that package-scale and per-era
   BMC200 are non-comparable.
5. **Current Constraints**: the `## Current Constraints` bullets from `CURRENT.md`, verbatim.
6. **Candidate Inventory**: every recipe still in play with experiment id, config, recipe summary,
   trio mean and FNC where known, evidence class, and the gate it waits on.

A dossier that concatenates the sources has failed. Compression is the deliverable.

## Stage 2: Proposal

Write one proposal, and a second choice only if a genuinely different direction is close, to
`.numereng/tmp/init-program/PROPOSAL.md`. Every fact it relies on must be in the dossier.

Each proposal carries:

- a title, a one-sentence falsifiable hypothesis, and the mechanism: why it should move the primary
  metric
- the lane: feature set, target, model family, dataset variant
- the research type: `new_target_or_feature_engineering`, `new_architecture`,
  `ensemble_or_blend`, `training_procedure`, `data_change`, or `hyperparameter_frontier`
- the varied axis: the config paths the run explores, one axis or one tight family
- the seed recipe: every param the run will vary, with its starting value
- the allowed change paths and value caps
- the round budget, justified against per-round wall time on the training host
- the tier, scout or scale, and why this rung now. A scout seeds `downsampled`, runs cheap wide
  sweeps, and produces candidates only. A scale run seeds `non_downsampled`, opens with the scout's
  winners, and is the only place a trio confirms on full data.
- success criteria and kill criteria in terms larger than the seed-noise floor
- the evidence basis: the dossier facts motivating it, each citing its dossier section
- the constraints checked: each closed lane, retired claim, and Current Constraints bullet, with a
  verdict

Before moving on, check the proposal against the hard constraints and against these questions: does
it attack a real open question from the dossier; is scout-versus-scale right for the evidence state;
can the step sizes and caps produce a result above the roughly 3e-4 seed-noise floor; will the
result be attributable to one axis; and does the budget fit the machine.

## Stage 3: Critique

Spawn one critic subagent on a frontier model from a different vendor than your own. Give it the
dossier and the proposal. Its mandate is to attack: memory misreads, category errors between
package-scale and per-era metrics, confound bundling, redundancy with closed hypotheses, sub-noise
sweep design, and a stronger neglected alternative. If it finds nothing, it must say which attacks it
tried and why each failed.

One exchange: the critic's findings ranked by severity, your reply accepting and revising or
rejecting with a reason, and the critic's final position, consensus or dissent. Save the transcript
to `.numereng/tmp/init-program/DEBATE.md`. You decide either way. Carry a dissent verbatim into the
design record; never paraphrase or drop one. If the critique changes the pick, re-check the new pick
against Stage 2 before continuing.

## Stage 4: Create

1. Create the experiment:

   ```bash
   uv run numereng experiment create --id <YYYY-MM-DD>_<slug> --hypothesis "<H1>" --tags "agentic,<lane>"
   ```

2. Write the brief at `.numereng/experiments/<id>/agentic_research/STRATEGY.md`, using the headings
   of `src/numereng/agentic_research/programs/STRATEGY.md`. The harness inserts it into the tracked
   program at run time and injects no other research memory, so everything the in-run model must
   know from prior work goes here as standalone prose: closed lanes, retired claims, inert axes, the
   calibration stance, the sweep plan with families, order, and step sizes, the confirmation plan,
   and any substrate facts for the model family.

3. Write the seed config at `configs/config_001.json`, obeying every hard constraint, with seed 42
   at the seed path and every param the run will vary present.

4. Set the manifest metadata in `experiment.json` directly: `agentic_research_allowed_change_paths`,
   `agentic_research_value_caps`, `agentic_research_budget_rounds`, and `agentic_research_seed_path`
   when the model family's seed is not `model.params.random_state`.

5. Write `DESIGN.md` at the experiment root: the dossier digest, the proposal and any second choice,
   the critique with any dissent verbatim, and the final rationale naming lane, axis, tier, and
   budget.

6. Verify:

   ```bash
   uv run python -c "from pathlib import Path; from numereng.config.training import load_training_config_json; load_training_config_json(Path('.numereng/experiments/<id>/configs/config_001.json')); print('seed config OK')"
   uv run numereng experiment details --id <id> --format json
   ```

7. Stop and present the design: the pick, its hypothesis and mechanism, the lane, the axis, the
   tier, the budget, the allowed paths and caps, the success and kill criteria, any dissent, and the
   files you wrote. The human authorizes the launch.
