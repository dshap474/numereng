# Numereng Architecture

## Navigation
- [1. Purpose](#1-purpose)
- [2. Design Principles](#2-design-principles)
- [3. Layered Topology](#3-layered-topology)
- [4. Source Layout](#4-source-layout)
- [5. Public Contracts](#5-public-contracts)
- [6. Runtime Persistence Model](#6-runtime-persistence-model)
- [7. Core Execution Flows](#7-core-execution-flows)
- [8. Telemetry + Run Monitoring Architecture](#8-telemetry--run-monitoring-architecture)
- [9. Viz Frontend/Backend Topology](#9-viz-frontendbackend-topology)
- [10. Boundary Error Model](#10-boundary-error-model)
- [11. Contract-Critical Invariants](#11-contract-critical-invariants)
- [12. Testing + Verification](#12-testing--verification)
- [13. Tradeoffs](#13-tradeoffs)
- [14. Change Guidance](#14-change-guidance)

## 1. Purpose
`numereng` is a package-first Numerai workflow system with two stable public interfaces:
- CLI: `numereng`
- Python facade: `import numereng.api`

It orchestrates:
- training
- submission
- experiments
- HPO
- ensemble construction
- prediction-stage neutralization
- store indexing/rebuild/doctor
- cloud workflows (EC2, AWS managed, Modal)
- read-only dashboard APIs for monitoring and inspection

## 2. Design Principles
1. Dependency direction is strict: `config -> platform -> features -> api -> cli`.
2. Platform isolation: `platform/*` never imports `features/*`.
3. Thin boundaries: CLI parses/dispatches; API translates errors; features own behavior.
4. Deterministic identity: run IDs derive from resolved training identity hash.
5. Artifact + index coherence: filesystem artifacts and sqlite index stay aligned.
6. Explicit contracts over implicit behavior: strict JSON schemas, typed request/response models.
7. Monitor-only UI philosophy: dashboard reads state; CLI/API performs control actions.

## 3. Layered Topology
```text
                    +------------------------------+
                    | Users / Automation           |
                    | - CLI commands               |
                    | - Python callers             |
                    +---------------+--------------+
                                    |
                 +------------------+------------------+
                 |                                     |
                 v                                     v
      +----------+-----------+              +----------+-----------+
      | CLI Boundary         |              | Python API Boundary  |
      | src/numereng/cli/*   |              | src/numereng/api/*   |
      +----------+-----------+              +----------+-----------+
                 |                                     |
                 +------------------+------------------+
                                    v
                       +------------+------------+
                       | Feature Slices          |
                       | src/numereng/features/* |
                       +------------+------------+
                                    |
                     +--------------+---------------+
                     |                              |
                     v                              v
          +----------+-----------+       +----------+-----------+
          | config contracts     |       | platform adapters    |
          | src/numereng/config/*|       | src/numereng/platform|
          +----------------------+       +----------------------+

Persistence root (default): `.numereng/`
```

## 4. Source Layout
```text
src/numereng/
  config/
    cloud/                     # checked-in runtime-image catalogs for cloud defaults
    training/{contracts.py,loader.py,schema/*}
    hpo/{contracts.py,loader.py,schema/*}

  platform/
    errors.py
    forum_scraper.py
    numerai_client.py

  features/
    training/                  # training pipeline, run manifests/artifacts
      scoring/                 # modular post-training scoring service + metric engines
    submission/                # run/file artifact resolution + Numerai upload
    baseline/                  # shared benchmark baseline construction from persisted runs
    feature_neutralization/    # prediction neutralization
    experiments/               # experiment lifecycle + run linkage
    agentic_research/          # program-driven continuous research supervisor with persisted program snapshots
    hpo/                       # Optuna study execution + trial persistence
    ensemble/                  # rank-average build + diagnostics
    dataset-tools/             # local dataset downsampling tools
    store/                     # sqlite schema/index/rebuild/doctor/layout
    telemetry/                 # run job/event/log/sample persistence
    viz/                       # read-only API adapter over store
    cloud/
      aws/                     # EC2 + managed AWS services + state
      modal/                   # deploy/data/train orchestration + state
      lambda/, runpod/, vast/  # placeholders
    models/
      custom_models/           # model plugin path; numereng may normalize backend asset paths into `<effective_store_root>/cache/*`

  api/
    contracts.py
    __init__.py              # curated public facade; research exports lazy-load
    pipeline.py              # public workflow entrypoints
    _run.py
    _experiment.py
    _hpo.py
    _ensemble.py
    _neutralization.py
    _store.py
    _numerai.py
    _dataset_tools.py
    _health.py
    _factories.py
    cloud/{_ec2.py,_aws.py,_modal.py}

  cli/
    main.py
    usage.py
    common.py
    commands/{run,experiment,baseline,hpo,ensemble,neutralize,store,monitor,numerai,cloud*}.py
```

## 5. Public Contracts
Stable contract surfaces:
- CLI entrypoint: `numereng = "numereng.cli:main"`
- Python facade: `import numereng.api as api_module`
- Public workflow module: `import numereng.api.pipeline`
- Typed request/response definitions: `src/numereng/api/contracts.py`

API module convention:
- `src/numereng/api/__init__.py`, `src/numereng/api/pipeline.py`, and `src/numereng/api/contracts.py` are public import surfaces.
- underscore-prefixed modules under `src/numereng/api/` are internal implementation modules and are not stable import paths.

Boundary behavior:
- API modules map feature/internal exceptions to `PackageError`.
- CLI exit codes are fixed:
  - `0`: success/help
  - `1`: runtime/boundary failure
  - `2`: parse/usage failure

Command families:
- `run`: `train`, `score`, `submit`, `cancel`
- `experiment`: `create`, `list`, `details`, `archive`, `unarchive`, `train`, `score-round`, `promote`, `report`, `pack`
- `baseline`: `build`
- `research`: `init`, `status`, `run`
- `hpo`: `create`, `list`, `details`, `trials`
- `ensemble`: `build`, `list`, `details`
- `neutralize`: `apply`
- `dataset-tools`: `build-downsampled-full`
- `store`: `init`, `index`, `rebuild`, `doctor`, `materialize-viz-artifacts`
- `monitor`: `snapshot`
- `cloud`: `ec2`, `aws`, `modal`
- `numerai`: `datasets` (`list`, `download`), `models`, `round`, `forum scrape`

## 6. Runtime Persistence Model
Default store root: `.numereng`

Canonical top-level dirs:
- `runs`
- `datasets`
- `cloud`
- `experiments`
- `notes`
- `cache`
- `tmp`
- `remote_ops`

Canonical top-level sqlite files:
- `numereng.db`
- `numereng.db-shm`
- `numereng.db-wal`

Dynamic top-level dirs may also appear:
- `hpo/`
- `ensembles/`

```text
.numereng/
  numereng.db
  numereng.db-shm
  numereng.db-wal

  runs/
    <run_id>/
      run.json                      # includes durable execution provenance
      runtime.json
      run.log
      resolved.json
      results.json
      metrics.json
      score_provenance.json
      artifacts/predictions/*.parquet
        numereng-managed parquet artifacts use ZSTD level 3

  datasets/
    baselines/
      <baseline_name>/
        baseline.json
        pred_<baseline_name>.parquet

  experiments/
    <experiment_id>/
      experiment.json
      EXPERIMENT.md
      EXPERIMENT.pack.md
      configs/*.json
      agentic_research/
        program.json
        session_program.md
        lineage.json
        rounds/rN/
          round.json
          round.md
          llm_trace.jsonl
          llm_trace.md
    _archive/
      <experiment_id>/
        experiment.json
        EXPERIMENT.md
        EXPERIMENT.pack.md
        configs/*.json
      hpo/<study_id>/...
      ensembles/<ensemble_id>/...

  hpo/<study_id>/...                  # global studies
  ensembles/<ensemble_id>/...         # global ensembles
  datasets/
  cache/
    cloud/
      <provider>/
        runs/<run_id>/state.json
        runs/<run_id>/pull/
        ops/<op_id>/state.json
  tmp/
    remote-configs/*.json            # retention-managed staging for detached remote launches
    lifecycle_smoke/<session>/*.json # generated smoke-test configs
  notes/
  remote_ops/
  cloud/                              # legacy compatibility-only cloud state during migration
```

Data model split:
- Filesystem holds canonical run/experiment artifacts; `runs/<run_id>/run.json -> execution` is the durable provenance record for materialized runs.
- sqlite indexes query surfaces for runs/metrics/experiments/HPO/ensembles/telemetry.
- `cloud_jobs` is the active/in-flight cloud job control plane for jobs that do not yet have a materialized run directory.
- `cache/cloud/*` holds transient cloud state and pull staging; it is not the durable run store.
- `tmp/*` is store-owned scratch space; `store doctor --fix-strays` prunes old `tmp/remote-configs/*.json` files only when they are older than 30 days and not referenced by active run lifecycles.
- `remote_ops/*` is canonical operational state for SSH bootstrap/sync/detached launch workflows.
- `store rebuild` re-derives index state from filesystem artifacts.
- Archived experiments keep the same experiment ID, but their experiment-local files move under `.numereng/experiments/_archive/<id>` while run artifacts remain under `.numereng/runs/<run_id>`.

## 7. Core Execution Flows

### 7.1 Bootstrap
```text
numereng [--fail]
  -> cli.main
  -> api.run_bootstrap_check
  -> HealthResponse JSON
```

### 7.2 Training
Entry points:
- CLI: `numereng run train`
- API: `api.run_training(TrainRunRequest)`
- Public workflow module: `api.pipeline.run_training_pipeline(TrainRunRequest)`
- Feature: `features.training.run_training`

```text
cli run train
  -> parse/validate args
  -> bind launch metadata source=cli.run.train
  -> api.run_training
      (compat facade; default metadata source=api.run.train only when unbound)
  -> api.run_training_pipeline / api.pipeline.run_training_pipeline
  -> internal API stages
  -> features.training._pipeline.run_training_pipeline
      1) load strict JSON config
      2) prepare deterministic run identity, output paths, lock, required lifecycle bootstrap (`run_lifecycles` + `runtime.json`)
      3) load training data + prepare model data loader
      4) train model + persist predictions/results/metrics placeholders
      5) append `post_fold` pruning artifacts during CV when fold outputs exist
      6) apply resolved post-training scoring policy (`none|core|full|round_core|round_full`)
      7) index run (mandatory pre-finalization)
      8) write terminal manifest (`FINISHED|FAILED|CANCELED|STALE`) with lifecycle metadata and release run lock
```

### 7.2b Run Re-Scoring
Entry points:
- CLI: `numereng run score --run-id <id> [--stage <all|run_metric_series|post_fold|post_training_core|post_training_full>]`
- API: `api.score_run(ScoreRunRequest)`
- Feature: `features.scoring.run_service.score_run`

```text
cli run score
  -> parse/validate args
  -> bind launch metadata source=cli.run.score
  -> api.score_run
      (default metadata source=api.run.score only when unbound)
  -> features.scoring.run_service.score_run
      1) load persisted run artifacts (`run.json`, `resolved.json`, `results.json`, predictions parquet)
      2) execute the canonical persisted-run scorer from saved predictions
      3) persist refreshed `results.json`, `metrics.json`, the staged `artifacts/scoring/` bundle, and `score_provenance.json`
      4) refresh run manifest metrics/artifact links
      5) re-index run in store
```

### 7.2c Shared Benchmark Baselines
Entry points:
- CLI: `numereng baseline build --run-ids <id1,id2,...> --name <baseline_name> [--default-target <target_col>] [--promote-active]`
- API: `api.baseline_build(BaselineBuildRequest)`
- Feature: `features.baseline.build_baseline`

```text
cli baseline build
  -> parse/validate args
  -> api.baseline_build
  -> features.baseline.build_baseline
      1) validate at least two persisted source runs and one safe baseline name
      2) load aligned persisted predictions from those runs
      3) build equal-weight blended predictions and attach all shared targets
      4) persist `.numereng/datasets/baselines/<name>/pred_<name>.parquet`
      5) persist `.numereng/datasets/baselines/<name>/baseline.json` plus per-target CORR artifacts
      6) optionally seed the canonical active-benchmark artifacts used by payout-target scoring
```

Data loading and CV rules:
- `data.dataset_variant=non_downsampled` resolves canonical defaults from `.numereng/datasets/<data_version>/`.
- `data.dataset_variant=downsampled` resolves `full.parquet -> downsampled_full.parquet` and `full_benchmark_models.parquet -> downsampled_full_benchmark_models.parquet`.
- `data.dataset_variant` is required and allowed only: `non_downsampled|downsampled`.
- Canonical non-downsampled storage uses split sources under `.numereng/datasets/<data_version>/`:
  `train.parquet`, `validation.parquet`, plus split benchmark files.
- Stored derived downsampled artifacts (built by `dataset-tools build-downsampled-full`) are:
  `downsampled_full.parquet`, `downsampled_full_benchmark_models.parquet`.
- `dataset-tools build-downsampled-full` uses default era filtering of every 4th era (`offset=0`).
- `data.dataset_scope=train_only` uses `train.parquet` only.
- `data.dataset_scope=train_plus_validation` uses `train.parquet` plus `validation.parquet`, and applies `data_type=validation` filtering only on validation sources.
- `purged_walk_forward` uses walk-forward defaults of `chunk_size=156`; embargo is horizon-based (`20d -> 8`, `60d -> 16`).
- `training.resources.max_threads_per_worker` accepts integer >= 1 or `"default"`; `"default"` (and omitted/`null` for backward compatibility) resolves to `max(1, floor(available_cpus / parallel_folds))`.
- Legacy `data.loading` is removed from the training schema. Config loading now hard-fails if it is present, and training always uses the materialized loader. Run hashing strips that removed subtree so historical identities remain stable.
- `model.device` is the canonical explicit device contract for training and is allowed only: `cpu|cuda`.
- `model.device` is valid only for `LGBMRegressor`; legacy `model.params.device_type` remains supported but must exactly match when both are present.
- Windows LightGBM GPU/OpenCL targets should use legacy `model.params.device_type=gpu` without top-level `model.device`; `model.device=gpu` is not a valid contract.
- Canonical `model.x_groups` / `model.data_needed` are features-only by default; `era` and `id` are never auto-included and are rejected as input groups.
- FNC diagnostics are computed in post-run scoring by neutralizing predictions to dataset feature set `fncv3_features`, independent of the run's training feature set, then correlating against the scoring target being evaluated (`fnc` for the native target, `fnc_<alias>` for extra scoring targets).
- `corr`, `fnc`, and `mmc` are delegated to `numerai_tools`; `cwmm` is computed locally using the official diagnostic semantics of Pearson correlation between the Numerai-transformed submission and the raw meta-model series.
- Scoring implementation is optimized behind the existing boundary: `features.scoring.metrics` orchestrates cached parquet reads and canonical artifact/provenance assembly, while `features.scoring._fastops` owns the NumPy/Numba per-era kernels that replace the older pandas callback hot path.
- Canonical scoring artifacts are materialized under `artifacts/scoring/` in two ways: training always writes `post_fold` artifacts during CV and may also materialize `post_training_core` or inclusive `post_training_full` depending on `training.post_training_scoring`, while later `run score` / `experiment score-round` can refresh any stage from saved predictions. The bundle includes `run_metric_series.parquet` plus staged parquet artifacts: `post_fold_per_era`, `post_fold_snapshots`, `post_training_core_summary`, and `post_training_full_summary` when materialized. Historical runs may still expose legacy `post_training_summary` / `post_training_features_summary` files, which remain read-compatible. Training-time scoring and partial rescoring both rebuild recorded artifact metadata from the persisted scoring manifest so `score_provenance.json`, `results.json -> training.scoring.emitted_stage_files`, and `run.json -> training.scoring.emitted_stage_files` reflect the actual materialized bundle rather than the pre-persist in-memory stage map.
- `post_training_core_summary` is the flattened decision scorecard: it carries native/payout CORR summaries plus payout-backed `mmc`, `bmc`, `bmc_last_200_eras`, scalar `avg_corr_with_benchmark`, and `corr_delta_vs_baseline` summary stats when the aligned benchmark/meta inputs exist.
- When the requested stage is only `post_training_core`, the canonical stage omissions record `post_training_full=not_requested`. Availability-based omission reasons for `post_training_full` are reserved for requests that actually include feature-heavy diagnostics (`all` or `post_training_full`).
- `baseline_corr` is no longer persisted as a run metric; instead, provenance records whether payout-target baseline CORR came from the shared active-benchmark artifact or from transient fallback computation.
- Training scoring does not emit payout estimate fields because Numereng does not implement an official expected-payout estimator from validation metrics.
- Deferred or failed post-training scoring still leaves `results.json` and `metrics.json` on disk with `training.scoring.policy`, `status`, `requested_stage`, `refreshed_stages`, and optional `reason` / `error` metadata. `score_provenance.json` appears once scoring materializes successfully.
- Feature exposure diagnostics are computed during post-run scoring using the same `fncv3_features` join path as FNC. Training persists nested summaries for `feature_exposure` (RMS exposure) and `max_feature_exposure` (max absolute exposure), while viz exposes `feature_exposure_mean` and scalar `max_feature_exposure` from those nested payloads.
- `score_provenance.json` captures the fixed scoring policy (`fnc_feature_set=fncv3_features`, `fnc_target_policy=scoring_target`, `benchmark_min_overlap_ratio=0.0`), benchmark source metadata (`active` or explicit path), join row/era counts, benchmark/meta missing-row/era counts, and the persisted scoring-artifact manifest summary.
- Benchmark post-run scoring joins require strict era alignment, but `bmc` / `bmc_last_200_eras` are computed on the maximum available overlapping benchmark window whenever any overlap exists.
- Meta-model post-run scoring joins require strict era alignment, but `mmc` / `cwmm` are computed on the maximum available overlapping meta-model window whenever any overlap exists.
- Horizon resolution prefers explicit `data.target_horizon` (`20d|60d`), then falls back to `target_col` name inference.
- If `data.target_horizon` is omitted and target-name inference is ambiguous, training fails with `training_engine_target_horizon_ambiguous`.
- Purged walk-forward CV requires at least `chunk_size + 1` unique eras; insufficient eras fail with an explicit configuration error.

Failure semantics:
- Training writes `FAILED` manifest on exception.
- Pre-finalization indexing failure fails the command.
- Post-training scoring is best-effort; failures do not flip a completed training run to `FAILED`.
- Post-`FINISHED` index refresh is best-effort.

### 7.3 Experiment
```text
cli experiment create
  -> api.experiment_create
  -> features.experiments.create_experiment
      - validate `YYYY-MM-DD_slug` experiment id format
      - create experiment dir and manifest
      - scaffold deterministic experiment-local files:
          EXPERIMENT.md
          configs/
          run_plan.csv
          run_scripts/launch_all.py|.sh|.ps1
      - upsert experiment index row
```

```text
cli experiment train
  -> bind launch metadata source=cli.experiment.train
  -> api.experiment_train
      (default metadata source=api.experiment.train only when unbound)
  -> features.experiments.train_experiment
      - validate experiment manifest
      - reject archived experiments as read-only
      - enforce output_dir == store_root (or default)
      - run training with experiment_id
      - link run into experiment manifest/store
      - when `training.post_training_scoring` resolves to `round_core|round_full`, trigger one best-effort round batch scoring pass after manifest linkage
```

```text
cli experiment score-round
  -> bind launch metadata source=cli.experiment.score-round
  -> api.experiment_score_round
  -> features.experiments.score_experiment_round
      - resolve round label (`rN`) to config stems from `run_plan.csv` or `configs/rN_*.json`
      - map those config stems back to eligible experiment run ids (`FINISHED` with persisted predictions); if multiple eligible runs share one config stem, prefer the newest run
      - call `features.scoring.batch_service.score_run_batch`
      - persist `post_training_core` or inclusive `post_training_full` artifacts for the selected round
```

### 7.4 Experiment Archive / Unarchive
```text
cli experiment archive|unarchive
  -> parse/validate args
  -> api.experiment_archive|api.experiment_unarchive
  -> features.experiments.archive_experiment|unarchive_experiment
      - resolve experiment from live or archived root
      - mutate manifest status (`archived` or restored pre-archive status)
      - move experiment dir between:
          .numereng/experiments/<id>
          .numereng/experiments/_archive/<id>
      - upsert experiment index row so viz reflects the change immediately
```

### 7.5 Agentic Research
```text
cli research program list|show
  -> api.research_program_list|api.research_program_show
  -> features.agentic_research.list_research_programs|get_research_program

cli research init|status|run
  -> api.research_init|api.research_status|api.research_run
  -> features.agentic_research.init_research|get_research_status|run_research
      - persist supervisor state under:
          .numereng/experiments/<root>/agentic_research/program.json
          .numereng/experiments/<root>/agentic_research/session_program.md
          .numereng/experiments/<root>/agentic_research/lineage.json
          .numereng/experiments/<root>/agentic_research/rounds/rN/*
      - `research init` requires one persisted `program_id`
      - the default tracked program lives under `src/numereng/features/agentic_research/programs/numerai-experiment-loop.md`
      - local custom programs can be dropped into `src/numereng/features/agentic_research/programs/*.md`; that folder includes a README and `.gitignore` so extra programs stay untracked by default
      - each program markdown file contains YAML frontmatter plus the full planner prompt body
      - runtime layout is localized into `run.py` plus `utils/planning.py`, `utils/mutation.py`, `utils/programs.py`, `utils/store.py`, `utils/llm.py`, and `utils/types.py`
      - `program.json` stores the canonical `program_snapshot`; runtime resume uses that snapshot, not the live program catalog
      - `session_program.md` stores the exact markdown selected at init time for auditability
      - legacy sessions that only stored `strategy` are auto-migrated to a normalized program snapshot on load/save
      - phase-aware programs persist `current_phase` in program state and surface it in API/CLI status
      - select planner compute source from `src/numereng/config/openrouter/active-model.py`
      - checked-in default: `ACTIVE_MODEL_SOURCE=codex-exec`
      - `ACTIVE_MODEL_SOURCE=codex-exec` uses headless `codex exec`
      - `ACTIVE_MODEL_SOURCE=openrouter` uses the configured OpenRouter `ACTIVE_MODEL`
      - `codex-exec` inherits the user’s normal Codex configuration and environment; agentic research does not create a feature-specific `CODEX_HOME`
      - programs define policy and prompt surface only; Python still validates configs, writes files, trains, scores, persists lineage, and resumes sessions
      - `numerai-experiment-loop` is config-centric:
        - Python selects one parent config from the active-path lineage using the program metric policy
        - the planner sees one parent mutable-config snapshot built only from allowed mutation paths, the effective scoring stage, one compact lineage summary, and only the current program context
        - the planner returns a minimal `RATIONALE:` + `CHANGES:` text block instead of a dense planner JSON object
        - Python parses dotted `config.path = <json-literal>` edits, clones the parent config, validates the child `TrainingConfig`, names it deterministically, and retries once on invalid or duplicate mutations
        - each numerai autonomous iteration writes and trains at most one child config; the config file is the unit of evolution
      - phase-aware custom programs can still use the structured planner JSON contract
        - the JSON schema is generated from the persisted program definition instead of loaded from strategy assets
      - persist planner trace entries per round at `.numereng/experiments/<root>/agentic_research/rounds/rN/llm_trace.jsonl` and render the same chronological stream into `.numereng/experiments/<root>/agentic_research/rounds/rN/llm_trace.md`
      - persist one canonical round bundle per round at `rounds/rN/round.json` and `rounds/rN/round.md`
      - round bundles and planner traces record `program_id`, `program_sha256`, and `session_program_path`
      - the round markdown is deliberately compact: round status, parent selection, mutation lineage, scored outcome, and links back to the full planner trace for that round
      - numerai rounds persist mutation lineage in the round bundle (`parent_run_id`, `parent_config_filename`, `parent_selection_reason`, `change_set`, `llm_rationale`)
      - numerai mutation rounds now write one child config into the active experiment, train it, deferred-score the round, and persist one bundled round record instead of many transport-specific files
      - legacy per-round `codex_*`, `planned_configs.json`, `report.json`, `round_summary.json`, and per-round trace copies are removed when the bundle is written
      - track plateau streak on the program primary metric using the configured tie-break metric
      - let program phase policy decide whether the next round stays in the current phase, advances phase, or completes the campaign
      - after the configured non-improving round threshold, optionally force the configured scale-confirmation round(s); if the path still fails, create a fresh child experiment and continue there
      - write lineage backlinks into experiment manifest metadata so root and child paths remain auditable
```

### 7.6 HPO
```text
cli hpo create
  -> api.hpo_create
  -> features.hpo.create_study
      - validate config + search space
      - execute Optuna trials
      - each trial runs training + index_run
      - extract the HPO objective from `post_fold_snapshots.parquet` by default
      - optional neutralization + custom metric extraction only for explicit non-default metrics
      - persist trial rows + study summary
```

Metadata behavior in HPO:
- If launch metadata is already bound, HPO reuses it.
- Otherwise HPO sets default source `api.hpo.create` for trial runs.
- Default champion-run HPO objective:
  `0.25 * mean(corr_ender20_fold_mean) + 2.25 * mean(bmc_fold_mean)`
  with read-time legacy fallback to `bmc_ender20_fold_mean` for older `post_fold_snapshots.parquet` artifacts.

### 7.6 Ensemble
```text
cli ensemble build
  -> api.ensemble_build
  -> features.ensemble.build_ensemble
      - validate method and inputs
      - blend predictions (rank_avg)
      - compute diagnostics
      - persist artifacts + sqlite rows
```

### 7.6 Submission + Neutralize
```text
cli run submit
  -> api.submit_predictions
  -> features.submission
      - resolve source (run_id XOR predictions_path)
      - validate classic live eligibility with `numerai_tools` against downloaded `live.parquet` ids (strict by default)
      - optional neutralization
      - upload via Numerai client

cli neutralize apply
  -> api.neutralize_apply
  -> features.feature_neutralization
```

### 7.7 Store
```text
cli store init|index|rebuild|doctor
  -> api.store_*
  -> features.store service

cli store materialize-viz-artifacts --kind <per-era-corr|scoring-artifacts> (--run-id <id> | --experiment-id <id> | --all)
  -> api.store_materialize_viz_artifacts
  -> features.store.materialize_viz_artifacts
      - rerun canonical scoring for historical runs missing `artifacts/scoring/manifest.json`
      - explicit `--run-id` hard-fails with `store_run_not_found:<id>` when the target run does not exist
      - `--experiment-id` scope skips unreadable unrelated run manifests while continuing to backfill matching readable runs
      - refresh `artifacts/scoring/*`, `results.json`, `metrics.json`, `score_provenance.json`, and run manifest artifact links
```

### 7.8 Cloud
```text
cli cloud ec2 ...
  -> api.cloud_ec2_*
  -> features.cloud.aws.CloudEc2Service

cli cloud aws image|train ...
  -> api.cloud_aws_*
  -> features.cloud.aws.managed_service

cli cloud modal deploy|data sync|train ...
  -> api.cloud_modal_*
  -> features.cloud.modal.service
```

### 7.9 Numerai Forum Scrape
```text
cli numerai forum scrape
  -> api.scrape_numerai_forum
  -> platform.scrape_forum_posts
  -> docs/numerai/forum/{posts/YYYY/MM/*.md, posts/YYYY[/MM]/INDEX.md, INDEX.md, .forum_scraper_manifest.json, .forum_scraper_state.json}
  -> viz docs API appends "Forum Archive" under `/api/docs/numerai/tree` with year -> month index navigation
```

## 8. Telemetry + Run Monitoring Architecture
Telemetry is the required lifecycle control plane for local public training entrypoints and the append-only history plane for run monitoring.

### 8.1 Source binding model
```text
caller binds metadata
  -> bind_launch_metadata(source=...)
  -> training reads get_launch_metadata()

if metadata absent, local public training fails before data/model work:
  api.run_training      => source=api.run.train
  api.experiment_train  => source=api.experiment.train
  api.pipeline.train    => source=api.pipeline.train
  hpo.create_study      => source=api.hpo.create (trial fallback)
  modal runtime         => source=cloud.modal.runtime (fallback)
```

Boundary rule:
- Local public training entrypoints must arrive with launch metadata.
- Lower-level training execution treats missing lifecycle context as an error, not an opt-in omission.

### 8.2 Session lifecycle
```text
begin_local_training_session
  -> insert queued rows into:
       logical_runs
       run_jobs
       run_attempts
       run_lifecycles
  -> persist runs/<run_id>/runtime.json
  -> emit job_queued event
  -> append initial log + sample

mark_job_starting -> mark_job_running
  -> update run_jobs/run_attempts/logical_runs/run_lifecycles
  -> rewrite runtime.json
  -> emit job_starting/job_running

during training:
  -> emit stage_update events
  -> emit metric_update events
  -> append logs and resource samples
  -> refresh run_lifecycles current_stage/latest_metrics/latest_sample/progress_*
  -> rewrite runtime.json

cancel path:
  -> cli/api request cancel by run_id
  -> mark cancel_requested on run_lifecycles/run_jobs/run_attempts
  -> emit cancel_requested event
  -> training checks safe checkpoints and exits through canonical canceled finalization

terminal:
  -> mark_job_completed OR mark_job_failed OR mark_job_canceled
  -> persist terminal reason/detail and canonical run linkage
  -> rewrite runtime.json as final lifecycle snapshot
  -> write run.json lifecycle block
```

Current-truth split:
- `run_lifecycles`: canonical sqlite read model for current lifecycle state.
- `runs/<run_id>/runtime.json`: canonical filesystem snapshot for current lifecycle state.
- `runs/<run_id>/run.json -> execution`: canonical durable provenance for where the run executed (`local`, `remote_host`, `cloud`) and which backend/provider produced it.
- `run_job_events`, `run_job_logs`, `run_job_samples`: append-only history for monitors/debugging.
- Both current-truth surfaces carry backend-owned progress fields:
  `progress_percent`, `progress_label`, `progress_current`, `progress_total`.

Stable lifecycle statuses:
- `queued`, `starting`, `running`, `completed`, `failed`, `canceled`, `stale`
- nuance such as user cancel, reconciliation, or crash reason lives in `terminal_reason` + `terminal_detail`, not in a larger status enum
- Progress semantics are also owned by the backend. Training writes a fixed pipeline plan:
  `queued=0`, `initializing=5`, `load_data=15`, `train_model=15..80`,
  `score_predictions=80..85`, `persist_artifacts=85..92`, `index_run=92..95`,
  `score_predictions_post_run=95..99`, `finalize_manifest=99..100`.
- `train_model` progress is fold-aware when CV fold boundaries exist; single-fit paths expose `progress_total=1` and advance at stage boundaries only.

### 8.3 Experiment scoping behavior
Experiment id for telemetry job rows is resolved as:
1. explicit `experiment_id` argument (preferred)
2. fallback inference from config path under:
   `.numereng/experiments/<experiment_id>/configs/*`

Operational impact:
- `numereng run train --experiment-id <id>` guarantees experiment-scoped monitor visibility.
- Running directly from experiment config paths can still auto-scope via path inference.

### 8.4 Reliability contract
- Lifecycle bootstrap is required for local public training; if it cannot initialize, the training command fails.
- After bootstrap, telemetry event/log/sample writes remain fail-open so observability degradation does not silently kill an otherwise running job.
- Active lifecycle reads may auto-reconcile orphaned local runs to terminal `canceled` or `stale`.

## 9. Viz Frontend/Backend Topology

### 9.1 Backend API shape (`viz/api/numereng_viz`)
Read-only routes include:
- `/api/system/capabilities`
- `/api/experiments*`
- `/api/experiments/overview`
- `/api/configs*`
- `/api/run-jobs*`
- `/api/runs/*`
- `/api/runs/{run_id}/lifecycle`
- `/api/studies*`
- `/api/ensembles*`
- `/api/docs/*`, `/api/notes/*`

Backend packaging/layout:
- `viz/api.py` is the uvicorn launcher shim.
- `viz/api/numereng_viz/*` is the authoritative backend package.
- `src/numereng/features/viz/*` is compatibility-only and should not accumulate new business logic.
- The wheel ships both `src/numereng` and `viz/api/numereng_viz`.

Store/remote monitor contract:
- `numereng.api.build_monitor_snapshot(request)` and `numereng monitor snapshot --store-root <path> --json` build one normalized read-only monitor snapshot for a single numereng store.
- `numereng.api.remote_*` and `numereng remote ...` provide SSH-backed repo sync, experiment authoring sync, ad hoc config push, and detached remote launch.
- Mission control overview merges:
  - the local store snapshot
  - zero or more SSH remote store snapshots loaded from `src/numereng/platform/remotes/profiles/*.yaml` or `NUMERENG_REMOTE_PROFILES_DIR`
- SSH monitoring is numereng-owned and read-only: the local viz backend runs `numereng monitor snapshot --json` on the remote machine over SSH and merges the returned snapshot.
- SSH remote profiles declare `shell: posix|powershell`; `posix` keeps the existing `cd ... && ...` command path, while `powershell` wraps the snapshot command in `powershell -NoProfile -Command "Set-Location ...; ..."` for Windows SSH targets.
- Remote ops use the same target profiles plus `python_cmd` for helper scripts and optional `runner_cmd` for detached CLI launches. Sync is local-driven and archive-based over SSH:
  - repo sync includes tracked files plus untracked nonignored files from the local working tree
  - repo sync excludes `.git`, `.numereng`, gitignored machine profiles, envs, caches, and build outputs
  - experiment sync includes only `.numereng/experiments/<id>/experiment.json|EXPERIMENT.md|run_plan.csv|configs/*|run_scripts/*`
  - ad hoc remote launch configs are staged under `.numereng/tmp/remote-configs/*` on the remote repo
  - no command mirrors the full `.numereng` store
- Sync metadata is stored under each machine's `.numereng/remote_ops/sync/<target_id>/*.json`, and detached remote launch metadata/logs live under `.numereng/remote_ops/launches/*` on the remote machine.
- Windows detached targets should use direct remote-venv executables (`python.exe`, `python.exe -m numereng.cli`) instead of `uv run ...`; detached launch performs short startup verification and uses Windows job breakaway so the child survives the SSH parent/session boundary.
- Snapshot normalization prefers `runs/<run_id>/run.json -> execution` for any materialized run; `cloud_jobs` is only the live control plane for cloud work that has not yet materialized into `runs/<run_id>`.
- Managed SageMaker/Batch `pull/extract` is authoritative for cloud provenance: if extracted artifacts match an already-materialized deterministic run hash, numereng skips moving files but still refreshes that run's `execution` block from managed state.
- SageMaker/Batch monitoring is store-local: snapshot refresh reads tracked `cloud_jobs` and asks AWS for current job truth before normalization.
- Live run payloads carry `progress_mode`:
  - `exact` for lifecycle-backed local percentages
  - `estimated` for cloud phase/status mappings
  - `indeterminate` when no honest percentage exists
- Cloud mission-control percentages are intentionally coarse phase estimates rather than log-derived model progress:
  `queued/submitted/pending/created/validating=0`, `starting=8`, `downloading=22`,
  `training=68`, `uploading=92`, `stopping=96`, `completed=100`.
  Terminal cloud rows reuse `last_progress_percent` when the latest estimate was previously known.

Live monitor streaming route:
- `/api/run-jobs/{job_id}/stream` (SSE)
- multiplexes events/logs/samples with periodic heartbeat

Viz scoring contract:
- Public viz metrics are canonical-only: `corr_*`, `fnc_*`, `mmc_*`, `bmc_*`, `bmc_last_200_eras_mean`, `cwmm_*`, `feature_exposure_*`, `max_feature_exposure`, `max_drawdown`, and `mmc_coverage_ratio_rows`.
- Viz also publishes generic payout-target aliases `corr_payout_mean` and `mmc_payout_mean`; `corr_payout_mean` comes from `corr_ender20`, while `mmc_mean` / `mmc_payout_mean` normalize to the payout-backed MMC surface with legacy `mmc_ender20` fallback for older runs.
- Viz does not expose payout-derived metrics or payout-specific routes.
- Per-era correlation payloads use `{ era, corr }`.
- Run-detail section routes are independent (`manifest`, `metrics`, `per-era-corr`, `events`, `resources`, `trials`, `best-params`, `config`, `diagnostics-sources`); `/api/runs/{run_id}/bundle` remains compatibility-only.
- The experiment-page payout proxy scatter uses payout-target metrics (`corr_payout_mean`, `mmc_payout_mean`) rather than native-target metrics, so runs trained on different targets are compared on one common payout basis.
- The experiment-page analysis panel also includes a target-scoped scatter (`Target Analysis`) that filters to actual runs for one selected training target and compares native `corr_mean` vs payout-backed `mmc_mean`.
- Run-detail scoring charts prefer persisted `artifacts/scoring/run_metric_series.parquet` and use read-only legacy-file composition only for older runs that predate the canonical chart artifact. Legacy `bmc_ender20` / `mmc_ender20` / `corr_delta_vs_baseline_ender20` charts are normalized into `bmc` / `mmc` / `corr_delta_vs_baseline`, while `baseline_corr_*` and native contribution duplicates are hidden from the main performance panel.

### 9.2 Frontend route topology (`viz/web`)
Current frontend routes:
- `/docs`
- `/docs/numerai`
- `/notes`
- `/experiments`
- `/experiments/[id]`

Important UI contract:
- There are no standalone `/run-ops` or `/configs` frontend pages.
- `/experiments` is a mission-control dashboard, not a launch surface. It route-loads `/api/experiments/overview`, polls that endpoint every 3 seconds while visible, preserves the last successful snapshot on transient failures, and renders a left-rail experiment navigator with a right-pane live/attention/recent-activity dashboard.
- The mission-control overview is federated across the local store plus all enabled SSH remotes. The local viz backend remains the only UI/API process; remote hosts are polled over SSH via `numereng monitor snapshot --json` and are surfaced in `overview.sources` with `live|cached|unavailable` source state plus persisted bootstrap metadata.
- `make viz` runs `numereng remote bootstrap-viz --store-root <repo>/.numereng` before local API/frontend startup. That bootstrap step repo-syncs each enabled remote in `auto` mode, runs `remote doctor`, persists `ready|degraded` source state under `.numereng/remote_ops/bootstrap/viz.json`, and never starts a remote viz server on the target host.
- Mission-control live cards render one canonical progress instrument per live experiment. The frontend selects a `primary` run by `updated_at desc`, then `exact > estimated > indeterminate`, then `run_id`, and renders only that run's bar plus adjacent percent readout.
- Experiment detail exposes Analysis + Progress + Run Ops tabs; launch/control remains monitor-only.
- Launch/control actions are CLI/API-only by design.
- `experiment pack` writes `EXPERIMENT.pack.md` into the experiment directory and includes `EXPERIMENT.md` plus a dashboard-aligned scalar run summary table only.
- Experiment ranking defaults to `bmc_last_200_eras_mean`.
- Run detail loading is progressive: the shell renders immediately from the selected run list row, and tab sections fetch lazily with localized loading states.
- Run Ops and the dedicated run page no longer block on `/api/runs/{run_id}/bundle`; they share the same section-based loader and request cache/abort behavior.
- Run detail/chart reads are artifact-backed first; older runs that only have legacy scoring files are adapted in memory to the canonical dashboard payload, and rescoring remains the canonical backfill path.
- Normal experiment/config catalogs exclude archived experiments by default.
- Direct experiment lookups remain archive-aware, so `/experiments/[id]` can still render an archived experiment when addressed directly.
- The Progress tab is the only experiment-detail UI surface that live-polls lifecycle state. It keeps a page-local cache keyed by `run_id`, polls `/api/run-jobs?experiment_id=<id>` plus per-run `/api/runs/{run_id}/lifecycle`, and leaves Analysis + Run Ops behavior otherwise unchanged.
- Progress UI labels are a frontend mapping over canonical lifecycle state:
  `NOT_STARTED`, `QUEUED`, `TRAINING`, `COMPLETED`, `FAILED`, `CANCELED`, `STALE`.

### 9.3 Run monitor data flow
```text
CLI/API training launch
  -> lifecycle bootstrap writes run_lifecycles + runtime.json
  -> telemetry rows written to sqlite
  -> numereng monitor snapshot reads one store
  -> snapshot normalizes local run_lifecycles + cloud_jobs
  -> viz backend merges local snapshot + configured SSH remote snapshots
  -> viz auto-reconciles active local lifecycle rows on read
  -> /experiments route polls /api/experiments/overview
  -> mission-control UI renders left rail + live experiment cards + recent terminal feed
  -> Progress tab polls /api/run-jobs?experiment_id=<id>
  -> Progress tab polls /api/runs/<run_id>/lifecycle for active run_ids
  -> UI renders lifecycle-driven status + progress bars
  -> selected Run Ops monitor still polls /api/run-jobs?experiment_id=<id>
  -> frontend selects active job
  -> frontend opens SSE /api/run-jobs/<job_id>/stream
  -> UI updates events/logs/resource samples live
```

### 9.4 Notes copy behavior
- Notes copy buttons copy file contents (fetched from `/api/notes/content`), not file path labels.

## 10. Boundary Error Model
```text
feature/internal error
  -> api/* catches and maps to PackageError(<stable_message>)
  -> cli prints stderr + exit 1

parse/usage issue
  -> cli prints usage + exit 2

success/help
  -> JSON payload or help text + exit 0
```

## 11. Contract-Critical Invariants
1. Submission source XOR: exactly one of `run_id` or `predictions_path`.
2. Classic submissions use `numerai_tools.submissions.validate_submission_numerai` against the current classic `live.parquet` id universe unless `allow_non_live_artifact=true`.
3. Neutralization source XOR: exactly one of `run_id` or `predictions_path`.
4. Training/HPO config files are JSON-only and reject unknown keys.
5. Training profile allowed only: `simple|purged_walk_forward|full_history_refit`.
6. Legacy training flags `--method` and `--method-overrides-json` are hard-fail.
7. `baseline build` requires at least two persisted run IDs, a safe filesystem baseline name, and a `default_target` shared by every source run.
8. Run IDs are deterministic hash-derived IDs (12-char prefix).
9. Training success requires pre-finalization `index_run` success.
10. Experiment train defaults `output_dir` to `store_root`; when provided it must equal `store_root`.
11. Ensemble method is `rank_avg` only; at least 2 deduped runs are required.
12. Cloud AWS train backend is only `sagemaker|batch`; `--spot` and `--on-demand` are mutually exclusive.
13. `cloud aws train pull` is download-only and writes artifacts to `.numereng/cache/cloud/aws/runs/<run_id>/pull`.
14. `cloud aws train extract` is the separate step that unpacks tarballs into `.numereng/runs/*` and indexes runs.
15. Cloud Modal deploy requires full ECR URI `<registry>/<repository>:<tag>`.
14. Cloud state files (`--state-path`) must parse into valid typed state objects.
15. Telemetry is fail-open and must never block training completion.
16. Launch metadata precedence is outermost binding first; API defaults apply only when unbound.
17. Experiment-scoped monitor queries depend on `run_jobs.experiment_id`; pass `--experiment-id` when launching ad-hoc run training.
18. Viz API is read-only; run control is intentionally not wired into frontend controls.
19. `train_plus_validation` filtering must never apply `data_type=validation` to `train.parquet`.
20. Purged walk-forward defaults are fixed at `chunk_size=156` and horizon-based embargo (`20d -> 8`, `60d -> 16`).
21. Horizon resolution uses `data.target_horizon` first and target-name inference second; unresolved ambiguity is a hard error.
22. Purged walk-forward CV requires `chunk_size + 1` or more unique eras.
23. Benchmark model parquet data is metrics-only (BMC/correlation diagnostics) and is not included as training features.
24. Canonical `model.x_groups` / `model.data_needed` are features-only; identifier groups (`era`, `id`) and benchmark aliases (`benchmark`, `benchmarks`, `benchmark_models`) are rejected.
25. `TabPFNRegressor` bare checkpoint names resolve under the effective run `store_root` at `cache/tabpfn` unless `TABPFN_MODEL_CACHE_DIR` is already set, so experiment-scoped or alternate store roots keep model cache state local to that root.
26. Canonical FNC uses dataset feature set `fncv3_features` and correlates against the scoring target being evaluated; `post_training_core` omits those feature-heavy diagnostics, while `post_training_full` and `all` include them. If `features.json` does not define `fncv3_features` and a feature-heavy stage is requested, scoring fails.
27. Benchmark post-run scoring requires nonzero `(id, era)` overlap with strict era alignment and emits `bmc` / `bmc_last_200_eras` on the maximum available overlapping benchmark window; meta-model scoring also requires strict era alignment but emits `mmc` / `cwmm` on the maximum available overlapping meta-model window whenever any overlap exists.
28. `full_history_refit` is final-fit only for training and does not emit `post_fold`; post-training scoring stays policy-driven just like other profiles.
29. Training writes run-local live logs to `runs/<run_id>/run.log` and records the file in `run.json -> artifacts.log`.
30. Training never applies row-level subsampling; any size reduction must happen at dataset construction/downsampling time.
31. Managed AWS extract only promotes `runs/<run_id>/*` archive members; non-`runs/*` members are skipped and recorded as warnings.
32. Managed AWS extract hard-fails on unsafe archive members (absolute/traversal paths, links, invalid run IDs) and on hash-mismatched run-dir collisions.
33. Managed AWS extract indexes only extracted run IDs and does not fallback-index the outer cloud run ID.
34. SageMaker managed entrypoint removes store DB sidecars (`numereng.db*`) from managed output before artifact packaging.
35. Managed AWS `--state-path` defaults under `<store_root>/cache/cloud/aws/...`; explicit overrides must resolve under `<store_root>/cache/cloud/**/*.json` or legacy `<store_root>/cloud/*.json` during the migration window. Submit-time `NUMERENG_RUN_EXECUTION_JSON` is intentionally trimmed to the minimum provider/runtime fields needed inside the container so long descriptive run IDs and canonical state paths stay under SageMaker env-size limits; full cloud provenance is recovered from managed state during pull/extract.
36. Archived experiments are read-only: `experiment train` and `experiment promote` must fail until the experiment is unarchived.
37. Experiment archive moves affect only `.numereng/experiments/*`; run artifacts remain canonical under `.numereng/runs/*`.
38. Managed AWS and EC2 `runtime_profile` selects packaging only (`standard|lgbm-cuda`) and never overrides the training config device.
39. SageMaker submit may omit `--image-uri`; when omitted, numereng resolves the checked-in default alias for the selected `runtime_profile`, persists the resolved `image_uri`, and records the tag digest in managed state/cloud metadata for reproducibility.
40. SageMaker CUDA submit requires config device `cuda`, `runtime_profile=lgbm-cuda`, and a GPU instance type (`ml.g*` or `ml.p*`); mismatches fail before submit.
41. EC2 CUDA runtime install requires persisted/requested GPU instance state plus `runtime_profile=lgbm-cuda`; mismatches fail before remote install.
42. CUDA training is fail-fast; numereng does not silently fall back from `cuda` to CPU.
43. EC2 and Modal `--state-path` follow the same cache-first rule: default under `<store_root>/cache/cloud/...`, explicit override under `<store_root>/cache/cloud/**/*.json`, legacy `<store_root>/cloud/*.json` accepted for compatibility only, and all state paths must use `.json`.
44. EC2 pull and S3-prefix copy reject traversal keys by skipping unsafe paths and reporting them in `skipped_unsafe_keys`.
45. `baseline build --promote-active` refreshes the shared active benchmark artifacts consumed by payout-target scoring fallback paths.

## 12. Testing + Verification
Primary checks:
- Install `just` if you want to use the wrapper commands below.
- `just fmt`
- `just test` (fast gate)
- `just test-all` (full gate)
- `just build`

Tooling contract:
- `uv` is the project manager and command runner.
- Ruff is the canonical formatter and linter with the minimal repo baseline (`E`, `F`, `I`, `UP`).
- `ty` is the canonical enforced type gate.
- The initial `ty` surface is intentionally scoped through `[tool.ty.src].include` in `pyproject.toml` while backlog-heavy areas are migrated later.

Contract anchors:
- `tests/integration/test_smoke_structure.py`: validates public surface presence.
- `tests/unit/numereng/test_cli.py`: exit codes/dispatch behavior.
- `tests/unit/numereng/test_api.py`: API boundary translation and contracts.
- `tests/unit/numereng/features/telemetry/test_service.py`: telemetry lifecycle + inference.

## 13. Tradeoffs
1. Strict boundaries vs convenience
- Benefit: predictable ownership and safer refactors.
- Cost: repeated boundary plumbing across CLI/API wrappers.

2. Deterministic IDs and strict schemas vs flexibility
- Benefit: reproducibility and debuggable runs.
- Cost: less tolerance for ad-hoc or partially specified configs.

3. Mandatory indexing in hot path
- Benefit: dashboard/query consistency immediately after runs.
- Cost: sqlite/index failures can fail training commands.

4. Monitor-only dashboard
- Benefit: single control plane (CLI/API) and less frontend mutation risk.
- Cost: no browser-side run launch convenience.

5. Required lifecycle bootstrap + fail-open updates
- Benefit: every public local run gets authoritative lifecycle identity/current truth without sacrificing resilience for later telemetry updates.
- Cost: startup now depends on successful lifecycle bootstrap, while later observability writes may still be partial under transient failures.

## 14. Change Guidance
When changing behavior/contracts:
1. Preserve dependency direction and layering.
2. Preserve CLI/API compatibility unless intentionally versioned.
3. Keep API and CLI files <= 500 LOC.
4. Keep API boundary translation to `PackageError`.
5. If training/telemetry/viz flows change, update both `docs/llms.txt` and this file in the same PR.
