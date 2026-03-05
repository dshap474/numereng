# Numereng Refactor Plan

## Scope (Phase 0)

This phase creates planning and contracts only.

1. Populate this file (`PLAN.md`) as the implementation artifact for phase 0.
2. Do not migrate core training/runtime code yet.
3. Keep `numereng-old` read-only.

---

## Source Analysis Summary

### Which codebases are best for what

1. `numereng-old`: best reference for run lifecycle, artifact contract, orchestration behavior, and operational telemetry.
2. `vendor/example-scripts`: best reference for modular decomposition of the training pipeline (data prep, CV, model factory, pipeline composition).
3. `vendor/numerai-tools`: best reference for scoring/submission primitives (corr/BMC/MMC/churn/turnover/submission validation).
4. `vendor/numerai-cli` and `vendor/numerai-predict`: deployment/inference/runtime references, not primary architecture reference for core training.

### Design decision

Use `numereng-old` for behavioral contract parity, but implement the new package using `example-scripts`-style modularity and the repo architecture rules in `MASTER-PYTHON-ARCHITECTURE.md`.

---

## Current Runtime Store Learnings (`numereng-old/.numereng`)

`numereng-old` uses a hybrid store:

1. Filesystem artifacts under `.numereng/runs/<run_id>/...`, plus `.numereng/experiments`, `.numereng/datasets`, `.numereng/cloud`, `.numereng/archive`, `.numereng/notes`.
2. SQLite index/telemetry in `.numereng/numereng.db` (with `-wal` and `-shm`, WAL mode).

### Root config

Observed in `.numereng/config.yaml`:

1. `store_root: <repo-root>/.numereng`
2. `backend: local`

### Active table families in `numereng.db`

1. Run metadata: `runs`, `metrics`, `events`, `resource_samples`, `experiments`.
2. Operation/job lifecycle: `logical_runs`, `run_jobs`, `run_attempts`, `run_job_events`, `run_job_logs`, `run_job_samples`.
3. Registry layer: `exp_registry_experiments`, `exp_registry_runs`, `exp_registry_metrics`, `exp_registry_models`, `exp_registry_submissions`, `exp_registry_schema_info`.
4. Other domains: `hpo_studies`, `ensembles`, `ensemble_components`, `ensemble_metrics`, `campaigns`, `baselines`.
5. Schema management: `schema_migrations`.

### High-value operational patterns to preserve

1. Logical operation key and status (`logical_runs`) separate from physical attempts (`run_attempts`).
2. Retry/staleness support across attempts for same logical operation.
3. Rich job telemetry/event streams (`run_job_events`, `run_job_logs`, `run_job_samples`).
4. Per-run files always include canonical metadata and outputs (`run.json`, `metrics.json`, `results.json`, configs, artifacts).

---

## Target Package Architecture (`src/numereng`)

Follow dependency direction:

`config -> platform -> features -> api.py` and `cli.py -> api.py`

MVP package layout (intentionally minimal):

```text
src/numereng/
├── __init__.py
├── api.py
├── cli.py
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── training_schema.py
├── platform/
│   ├── __init__.py
│   ├── errors.py
│   ├── numerai_official/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── constants.py
│   │   ├── errors.py
│   │   ├── graphql.py
│   │   ├── mcp.py
│   │   └── operations.py
│   ├── ids.py
│   └── clock.py
└── features/
    ├── __init__.py
    ├── store/
    │   ├── __init__.py
    │   ├── service.py
    │   └── repo.py
    ├── training/
    │   ├── __init__.py
    │   ├── service.py
    │   ├── models.py
    │   ├── repo.py
    │   └── client.py
    ├── submission/
    │   ├── __init__.py
    │   ├── service.py
    │   └── client.py
    └── inference/
        ├── __init__.py
        └── service.py
```

### MVP slice responsibility boundaries

1. `store`: owns `.numereng` bootstrapping, schema lifecycle, and fs/db consistency helpers.
2. `training`: owns data loading, CV/OOF, model fit/predict, scoring (`corr/mmc/bmc`), and artifact writing for MVP.
3. `submission`: owns prediction submission workflow to Numerai (from produced run artifacts).
4. `inference`: prediction-only path (deferred implementation detail).
5. `platform`: shared infra only (no business logic).

### Deferred slices (post-MVP)

1. `experiments`
2. `jobs`
3. `model_publish`
4. `docs_notes`
5. `viz_api`
6. separate `data_prep` / `validation` / `scoring` / `artifacts` slices (only if/when training slice grows enough to justify extraction)

---

## Public API and CLI Contract (Target)

`api.py` is stable consumer surface.

MVP API entrypoints:

1. `run_training(config: TrainingRunRequest) -> TrainingRunResult`
2. `submit_predictions(request: SubmissionRequest) -> SubmissionResult`
3. `run_training_and_submit(config: TrainAndSubmitRequest) -> TrainAndSubmitResult`
4. `get_run(run_id: str) -> RunRecord`
5. `list_runs(...) -> list[RunRecord]`
6. `validate_submission(...) -> SubmissionValidationResult`

CLI contract:

1. `cli.py` remains thin wrapper over `api.py`.
2. CLI maps typed errors to stable non-zero exit codes.
3. No business logic in `cli.py`.
4. Natural-language intent must map to one canonical command path for agents.

### CLI-first workflow contract (agent-facing)

Primary operator command shape:

1. `numereng run train --target ender --model lgbm [--submit --model-name <numerai_model>]`
2. Optional explicit submit path: `numereng run submit --run-id <run_id> --model-name <numerai_model>`

Required orchestration behavior for the train command:

1. Resolve config/default profile for `target/model` combo.
2. Load/cache Numerai data and build feature/target matrices.
3. Train model and generate predictions.
4. Compute evaluation suite (`corr`, `mmc`, `bmc`, and required summaries).
5. Persist canonical run artifacts + DB rows.
6. If `--submit` is set, execute prediction submission and persist submission ID/status.

Required defaults to avoid agent ambiguity:

1. Default data version, feature set, and validation profile.
2. Default metric set and artifact bundle.
3. Explicit credential source precedence (CLI flags > env > config).
4. Stable failure semantics: training may succeed while submission fails; both statuses must be recorded.

---

## Official Numerai Integration (`platform/numerai_official`)

The existing `numereng-old/src/numereng/numerai_official` implementation is adopted as protocol-layer reference.

Decisions:

1. Keep official Numerai protocol clients in `platform` (not in feature business logic).
2. Use `graphql.py` as the primary low-level API transport for deterministic CLI workflows.
3. Keep `mcp.py` as optional advanced transport and discovery utility (not required for MVP).
4. Keep typed error taxonomy and map to package-level errors in `api.py`.
5. Keep auth helpers and environment contract:
   - `NUMERAI_PUBLIC_ID`
   - `NUMERAI_SECRET_KEY`
   - `NUMERAI_API_AUTH`
   - `NUMERAI_MCP_AUTH`

Integration boundaries:

1. `features/submission` calls `platform/numerai_official/graphql.py` for prediction submission/status.
2. Model upload/assignment integration is deferred until after MVP training+submission is stable.
3. MCP is not required for the core train/evaluate/submit happy path; it remains optional tooling.

---

## `.numereng` + `viz` Integration Findings (Applied)

Strengths retained:

1. Hybrid persistence (filesystem artifacts + SQLite index/telemetry).
2. `logical_runs` + attempts lifecycle with retry/stale semantics.
3. Rich telemetry stream model (events/logs/samples) and SSE-friendly consumption.
4. Config catalog + runnable validation + run linkage.
5. Control-plane flows (launch/retry/cancel/docs/notes) as first-class capability.

MVP scope decision:

1. Decompose monolithic API/client surfaces into feature-owned modules.
2. Remove duplicated registry responsibilities by consolidating around the primary store tables.
3. Defer `viz_api` integration until after CLI-first training+submission path is proven.
4. Keep a clear extension path for jobs/viz without making them launch blockers.

---

## Runtime Store Contract (v1)

Required filesystem shape:

```text
.numereng/
├── config.yaml
├── numereng.db
├── runs/
│   └── <run_id>/
│       ├── run.json
│       ├── config.yaml
│       ├── input.yaml
│       ├── resolved.yaml
│       ├── metrics.json
│       ├── results.json
│       ├── artifacts/
│       │   ├── data/
│       │   ├── eval/
│       │   ├── model/
│       │   └── predictions/
├── experiments/              # optional in MVP, required when experiment flows are added
├── datasets/
└── notes/                    # optional in MVP
```

### Required `run.json` fields (minimum)

1. `schema_version`
2. `run_id`
3. `run_hash`
4. `external_run_id` (nullable, optional in MVP)
5. `created_at`
6. `status`
7. `config_hash`
8. `data` (version/feature_set/targets)
9. `data_fingerprint` (digest + file signatures)
10. `model` (type + params)
11. `validation` and profile/scope fields
12. `experiment_id` (nullable, optional in MVP)
13. `run_type`
14. `artifacts` map
15. optional embedded `metrics` snapshot

---

## Database Model (v1)

### Required tables at MVP launch

1. `schema_migrations`
2. `runs`
3. `metrics`
4. `experiments` (only if experiment IDs are used by the CLI in MVP)

### Deferred tables (not launch blockers)

1. `events`, `resource_samples`
2. `logical_runs`, `run_jobs`, `run_attempts`, `run_job_events`, `run_job_logs`, `run_job_samples`
3. `exp_registry_*`
4. `ensembles`, `ensemble_components`, `ensemble_metrics`
5. `hpo_studies`
6. `campaigns`, `baselines`

### DB invariants (MVP)

1. `runs.run_id` is unique and maps to exactly one canonical `runs/<run_id>/run.json`.
2. Run-level metrics are key/value upserts keyed by `(run_id, name)`.
3. DB rows and filesystem manifests must be cross-consistent for status and run path.

---

## Migration Mapping (Old -> New)

1. `runs/engine.py` monolith becomes:
   - `features/training/service.py` (single vertical slice in MVP)
2. `runs/writer.py` and `runs/loader.py` behavior folds into `features/training/repo.py` + `features/store/repo.py` initially.
3. `data/numerai_data.py` behavior retained, with I/O adapters in `platform` and orchestration in `features/training`.
4. `models/lgbm.py` starts as first model implementation in `features/training`.
5. `store/*` and run indexing behavior move into `features/store`.
6. `submission/uploader.py` behavior moves into `features/submission`.
7. `numerai_official/*` protocol clients move to `platform/numerai_official`.
8. `experiments`, `jobs`, and `viz` flows are explicitly deferred until after MVP parity.

---

## Implementation Phases

### Phase 1: MVP CLI Training + Evaluation

1. Implement store bootstrap (`config.yaml`, minimal DB schema, run root layout).
2. Implement one vertical training path in `features/training`: load data, CV/OOF, LGBM fit/predict.
3. Implement scoring outputs (`corr`, `mmc`, `bmc`) and write canonical run artifacts.
4. Implement minimal API + thin CLI command:
   - `numereng run train --target ender --model lgbm`

Exit criteria:

1. A smoke run writes canonical run files + DB rows.
2. CLI training command succeeds end-to-end and produces expected metrics payload shape.
3. `get_run`/`list_runs` work for MVP artifacts.

### Phase 2: MVP CLI Submission

1. Implement `features/submission` using `platform/numerai_official/graphql.py`.
2. Add submit path to CLI:
   - inline: `numereng run train ... --submit --model-name <name>`
   - explicit: `numereng run submit --run-id <run_id> --model-name <name>`
3. Persist submission result metadata and failure details in run artifacts/DB.

Exit criteria:

1. Train+submit happy path works from CLI with deterministic artifact contract.
2. Submission failure is represented cleanly without corrupting training success state.

### Phase 3: Hardening + Selective Extraction (Only If Needed)

1. Add fs/db consistency checks and rebuild/hydration tooling.
2. Split training internals into dedicated slices (`data_prep`, `validation`, `scoring`, `artifacts`) only when code size/complexity requires it.
3. Add `inference` command path if needed by active workflows.

Exit criteria:

1. No premature abstraction; extra slices are added only with clear second/third reuse.
2. Core CLI workflow remains stable during internal refactors.

### Phase 4: Deferred Advanced Operations (Not Launch Blockers)

1. Job queue + retries + telemetry streaming.
2. Viz API + frontend compatibility work.
3. Model upload/assignment (Numerai Compute).
4. Extended experiment/docs/notes management.

Exit criteria:

1. Explicit decision to promote one or more deferred domains into active roadmap.

---

## Test and Quality Gates

Required gates for each implementation phase:

1. `make test` (ruff + mypy --strict + required unit/integration-smoke suites).
2. Deterministic hermetic tests only in gating path.

Required scenarios:

1. DB schema initialization and migration idempotence.
2. Run artifact file contract creation.
3. Run metadata upsert + metrics upsert behavior.
4. Train command happy path: target/model intent -> run artifacts + scoring outputs.
5. Submit path happy/failure behavior with typed error mapping.
6. Numerai credential resolution precedence and validation.

---

## Risks and Controls

1. Risk: carrying over too much complexity from old DB schema early.
   - Control: launch with required table subset only; defer registry/ensemble/hpo tables.
2. Risk: path coupling to absolute store roots.
   - Control: enforce configurable store root and relative artifact references.
3. Risk: monolith regression in new code.
   - Control: enforce file-size thresholds and split only when justified by growth/reuse.
4. Risk: protocol-layer coupling leaks into feature business logic.
   - Control: confine Numerai protocol clients to `platform/numerai_official`; feature services consume typed interfaces only.
5. Risk: CLI ambiguity for agent-driven tasks.
   - Control: single canonical train/evaluate/submit command path with explicit defaults.
6. Risk: deferred domains (jobs/viz/model-upload) create distraction before MVP works.
   - Control: keep them explicitly non-blocking until CLI MVP is proven.
7. Risk: fs/db drift during submission and future async flows.
   - Control: explicit consistency checks + rebuild/hydration commands + tests.

---

## Decision Log

1. Keep hybrid persistence model (filesystem artifacts + SQLite index).
2. Use `numerai-tools` as external primitive library for scoring/submission contracts.
3. Prioritize one MVP vertical slice (`train -> evaluate -> submit`) before adding advanced domains.
4. Use `example-scripts` as modular decomposition reference.
5. Treat `numerai-cli` and `numerai-predict` as deployment/inference references, not core training architecture.
6. Adopt `numerai_official` clients as the protocol layer for official Numerai GraphQL/MCP access.
7. Make CLI-first intent execution a non-negotiable design goal with a single canonical train command path.
8. Defer model upload/assignment, jobs, and viz integration until after MVP CLI workflow is stable.
9. Keep MCP support optional for advanced workflows; GraphQL path is the default deterministic runtime path.
