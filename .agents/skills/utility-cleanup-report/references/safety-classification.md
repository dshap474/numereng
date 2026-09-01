# Cleanup Safety Classification Rules

How to classify each surveyed artifact group into SAFE / SAFE WITH PREREQUISITE /
PROTECTED / NEEDS HUMAN JUDGMENT. Derived from the 2026-07-31 cleanup session.

## Protected (never recommend deletion)

- `.numereng/datasets/` on any machine: inputs for future training, not experiment results.
- Run dirs that are the **canonical rebuild source for a hosted/live model**: any run ID that
  appears in `components_with_local_model_artifact` of a package with survey `hosted: true`
  is protected on the machine that holds the only full copy. Grid-lane packages count too,
  not just champions.
- Runs backing the active benchmark/baseline (`baseline` references).
- Experiment design + results metadata: `configs/`, `EXPERIMENT.md`, reports, round
  history, diagnostics artifacts, `experiment.json`.
- `.numereng/notes/__RESEARCH_MEMORY__/` and `.numereng/submissions/`.

## Safe to delete

- **Package pickles (`submission_packages/*/artifacts/pickle/model.pkl`)** when BOTH:
  1. the package was uploaded (survey `hosted: true` — `last_pickle_upload_id` set, or
     `status == "pickle_uploaded"` on older packages; Numerai hosts the live copy), and
  2. every `component_run_ids` entry appears in `components_with_local_model_artifact`
     on this machine (rebuildable via `serve pickle build` in minutes).
- **Cross-machine duplicate run dirs**: a run dir fully copied to another machine is safe to
  delete on the non-canonical machine. Convention: the Mac is canonical for package-eval /
  deploy runs; the PC is canonical for nothing (it is the compute box).

## Safe with prerequisite

- **Run dirs of closed + distilled experiments**: experiment has a branch under
  `__RESEARCH_MEMORY__/experiments/<id>/` AND its EXPERIMENT.md marks it
  complete/closed/stopped. Prerequisites before recommending deletion:
  1. Archive all small run metadata first (files < 10MB: run.json, summaries, scoring
     metrics, logs) into a per-experiment zip pulled to the Mac at
     `.numereng/experiments/_archive/pc_run_metadata/`. Do not trust
     `remote experiment pull --mode scoring` alone — runs with stale non-FINISHED
     status (e.g. after a PC reboot) materialize nothing.
  2. Deletion must go through utility-store-ops (`reset_experiment_runs.py` /
     `reset_runs.py` dry-run → execute), never raw `rm`, then `store doctor`.
- Exclude from the recommendation any run IDs that are protected per the rules above,
  even inside an otherwise-deletable experiment.

## Needs human judgment (report, do not recommend)

- Large experiment dirs whose heavy content is not run dirs or hosted pickles
  (e.g. bundled ensemble artifacts, ad-hoc exports) — list size and contents summary only.
- `_archive/` experiment dirs: archived does not imply distilled.
- Run dirs with no `run.json` (index husks): usually tiny; report count + size.
- Anything whose experiment lacks a research-memory branch.

## Cross-referencing checklist

1. Distilled experiments: `ls .numereng/notes/__RESEARCH_MEMORY__/experiments/`.
2. Hosted packages: every `package.json` with `last_pickle_model_name` set, across ALL
   experiments including `_archive/` — collect their component run IDs into the
   protected set for the machine holding the artifacts.
3. Live slots sanity check: `.numereng/submissions/<model>/` dirs enumerate models that
   have actually submitted; a hosted package without a submissions dir is still protected.
4. Duplicates: compare run IDs present in both machines' surveys.
