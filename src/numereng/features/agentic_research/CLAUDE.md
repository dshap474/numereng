# Agentic Research Supervisor Contract

`features.agentic_research` is the foreground supervisor that spans multiple experiment folders while
anchoring its own state under one root experiment. It must stay above `features.experiments`
instead of replacing it.

## Core contract

- The root experiment owns the supervisor ledger under `agentic_research/`.
- Child paths are regular experiment manifests created through `features.experiments.create_experiment`.
- The supervisor never lets Codex edit repo files directly. Headless Codex returns structured
  planning output only; Python validates and writes configs.
- Training is sequential in v1 and resumes from persisted state after interruption.

## Persisted artifacts

- `.numereng/experiments/<root>/agentic_research/program.json`
- `.numereng/experiments/<root>/agentic_research/lineage.json`
- `.numereng/experiments/<root>/agentic_research/rounds/rN/*`

Round folders persist the prompt, Codex response, planned config filenames, report snapshot, and
round summary so resume/debugging does not depend on volatile process state.

## Plateau logic

- Primary metric: `bmc_last_200_eras.mean`
- Tie-break: `bmc.mean`
- Sanity checks: `corr.mean`, `mmc.mean`, `cwmm.mean`
- Two consecutive non-improving rounds trigger one forced scale-confirmation round.
- If the path still does not improve after that confirmation, the next planning step pivots into a
  fresh child experiment.

## Metadata backlinks

Every experiment touched by the supervisor must write `metadata.agentic_research` with:

- `root_experiment_id`
- `program_experiment_id`
- `parent_experiment_id`
- `path_id`
- `pivot_reason`
- `source_round`
- `generation`

Store indexing must stay in sync after metadata writes so viz and experiment listings see the
lineage updates.
