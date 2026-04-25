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
`numereng` is a repo-local Numerai workflow system with two stable public interfaces:
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

Canonical runtime store: `.numereng/` inside the repo checkout
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
    docs_sync.py
    forum_scraper.py
    numerai_client.py

  features/
    docs_sync.py                 # repo-local Numerai docs mirror sync
    training/                  # training pipeline, run manifests/artifacts
      scoring/                 # modular post-training scoring service + metric engines
    serving/                   # submission packages, live build, package validation scoring, diagnostics sync, and model-upload pickle assembly
    submission/                # run/file artifact resolution + Numerai upload
    baseline/                  # shared benchmark baseline construction from persisted runs
    feature_neutralization/    # prediction neutralization; vectorized least-squares engine with numerai-tools-style intercept parity
    experiments/               # experiment lifecycle + run linkage
    agentic_research/          # program-driven continuous research supervisor with persisted program snapshots
    hpo/                       # Optuna study execution + trial persistence
    ensemble/                  # rank-average build + experiment-aware selection workflow
    dataset-tools/             # local dataset downsampling tools
    store/                     # sqlite schema/index/rebuild/doctor/layout
    telemetry/                 # run job/event/log/sample persistence
    viz/                       # read-only API adapter over store
    cloud/
      aws/                     # EC2 + managed AWS services + state
      modal/                   # deploy/data/train orchestration + state
      lambda/, runpod/, vast/  # placeholders
    models/
      custom_models/           # tracked model plugin path; numereng may normalize backend asset paths into `<effective_store_root>/cache/*`

  api/
    contracts.py
    __init__.py              # curated public facade; research exports lazy-load
    pipeline.py              # public workflow entrypoints
    _run.py
    _experiment.py
    _hpo.py
    _ensemble.py
    _neutralization.py
    _serving.py
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
    commands/{run,experiment,baseline,hpo,ensemble,serve,neutralize,store,monitor,numerai,docs,cloud*}.py
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
- `docs`: `sync numerai`
- `run`: `train`, `score`, `submit`, `cancel`
- `experiment`: `create`, `list`, `details`, `archive`, `unarchive`, `train`, `run-plan`, `score-round`, `promote`, `report`, `pack`
- `baseline`: `build`
- `research`: `program list`, `program show`, `init`, `status`, `run`
- `hpo`: `create`, `list`, `details`, `trials`
- `ensemble`: `build`, `select`, `list`, `details`
- `serve`: `package create|list|inspect|score|sync-diagnostics`, `live build|submit`, `pickle build|upload`
- `neutralize`: `apply`
- `dataset-tools`: `build-downsampled-full`
- `store`: `init`, `index`, `rebuild`, `doctor`, `backfill-run-execution`, `repair-run-lifecycles`, `materialize-viz-artifacts`
- `monitor`: `snapshot`
- `viz`
- `remote`: `list`, `bootstrap-viz`, `doctor`, `repo sync`, `experiment sync`, `experiment pull`, `experiment launch`, `experiment status`, `experiment maintain`, `experiment stop`, `config push`, `run train`
- `cloud`: `ec2`, `aws`, `modal`
- `numerai`: `datasets` (`list`, `download`), `models`, `round`, `forum scrape`

## 6. Runtime Persistence Model
Numereng has two path surfaces:

- source-tree extension paths: `src/numereng/features/models/custom_models/`, `src/numereng/features/agentic_research/PROGRAM.md`
- local ignored custom-skill path: `.agents/skills/`
- optional synced vendor docs path: `docs/numerai/` (manual local mirror)
- repo-root dev files: `pyproject.toml`, `.python-version`, `.venv/`
- hidden runtime store: `.numereng/`

Canonical runtime-store dirs under `.numereng/`:
- `runs`
- `datasets`
- `cloud`
- `cache`
- `tmp`
- `remote_ops`

Canonical sqlite files under `.numereng/`:
- `numereng.db`
- `numereng.db-shm`
- `numereng.db-wal`

Dynamic runtime-store dirs may also appear under `.numereng/`:
- `hpo/`
- `ensembles/`

```text
<repo-root>/
  pyproject.toml
  .python-version
  .venv/
  docs/
    numerai/                     # optional local mirror from `numereng docs sync numerai`
  src/numereng/features/models/custom_models/
  src/numereng/features/agentic_research/
    PROGRAM.md
    run.py
  .agents/
    skills/
  .numereng/
    experiments/
      <experiment_id>/
        experiment.json
        EXPERIMENT.md
        EXPERIMENT.pack.md
      run_scripts/*
      ensemble_selection/
        <selection_id>/
          status.json
          candidates/*
          correlation/*
          blends/*
      submission_packages/
        <package_id>/
          package.json
          artifacts/
            datasets/<round_token>/*
            live/<round_token>/*
            pickle/model.pkl
      configs/*.json
      agentic_research/
        state.json
        ledger.jsonl
        rounds/
          decision.json
          rNNN.md
          rNNN.debug.*     # only on LLM/Codex failure
        hpo/<study_id>/...
        ensembles/<ensemble_id>/...
      _archive/
        <experiment_id>/
          experiment.json
          EXPERIMENT.md
          EXPERIMENT.pack.md
          configs/*.json
        hpo/<study_id>/...
        ensembles/<ensemble_id>/...

    notes/
      __RESEARCH_MEMORY__/
        CURRENT.md
        experiments/<experiment_id>.md
        topics/*.md
        decisions/*.md
        legacy-progression/...

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
    hpo/<study_id>/...                # global studies
    ensembles/<ensemble_id>/...       # global ensembles
    cache/
      cloud/
        <provider>/
          runs/<run_id>/state.json
          runs/<run_id>/pull/
          ops/<op_id>/state.json
      remote_ops/
        pulls/
          <target_id>/
            experiments/<experiment_id>/{experiment.json,pull.json}
            runs/<run_id>/*
    tmp/
      remote-configs/*.json           # retention-managed staging for detached remote launches
      lifecycle_smoke/<session>/*.json # generated smoke-test configs
    remote_ops/
      experiment_run_plan/
        <experiment_id>__<start>_<end>.json
    cloud/                            # legacy compatibility-only cloud state during migration
```

Data model split:
- Filesystem holds canonical run/experiment artifacts; `runs/<run_id>/run.json -> execution` is the durable provenance record for materialized runs.
- sqlite indexes query surfaces for runs/metrics/experiments/HPO/ensembles/telemetry.
- `cloud_jobs` is the active/in-flight cloud job control plane for jobs that do not yet have a materialized run directory.
- `cache/cloud/*` holds transient cloud state and pull staging; it is not the durable run store.
- `cache/remote_ops/pulls/*` is legacy compatibility-only state for older lightweight remote pullbacks; new `remote experiment pull` runs materialize finished remote runs directly into canonical local `runs/<run_id>` storage and skip already-materialized valid runs on rerun instead of failing on local conflicts.
- `tmp/*` is store-owned scratch space; `store doctor --fix-strays` prunes old `tmp/remote-configs/*.json` files only when they are older than 30 days and not referenced by active run lifecycles.
- `remote_ops/*` is canonical operational state for SSH bootstrap/sync/detached launch workflows.
- `remote_ops/experiment_run_plan/*.json` is the durable execution-state contract for source-owned experiment windows and is the only restart/reconciliation target for `remote experiment maintain`.
- `store rebuild` re-derives index state from filesystem artifacts.
- Viz detail reads treat local artifacts as authoritative hot-path data: local experiment/run detail requests do not fetch remote overlay metadata unless the caller explicitly supplies a remote source or the local artifact is missing.
- Archived experiments keep the same experiment ID, but their experiment-local files move under `.numereng/experiments/_archive/<id>` while run artifacts remain under `.numereng/runs/<run_id>`.

## 7. Core Execution Flows

### 7.1 Bootstrap
```text
numereng [--fail]
  -> cli.main
  -> api.run_bootstrap_check
  -> HealthResponse JSON
```

Repo bootstrap:
```text
uv sync --extra dev

numereng docs sync numerai
  -> cli.commands.docs
  -> api.sync_docs
  -> features.docs_sync.sync_numerai_docs
      -> platform.docs_sync.clone_shallow
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

### 7.2b Experiment Run-Plan Execution
Entry points:
- CLI: `numereng experiment run-plan`
- API: `api.experiment_run_plan(ExperimentRunPlanRequest)`
- Feature: `features.experiments.run_experiment_plan`

```text
cli experiment run-plan
  -> parse/validate experiment id + row window + score stage
  -> api.experiment_run_plan
  -> features.experiments.run_experiment_plan
      1) load `.numereng/experiments/<id>/run_plan.csv`
      2) create/update `.numereng/remote_ops/experiment_run_plan/<id>__<start>_<end>.json`
      3) train each selected config with `post_training_scoring=none`
      4) score each completed round once via `experiment score-round`
      5) repair missing manifest links before batch scoring when a finished run exists but is not linked
      6) retry only classified infra failures (for example dead stale lock owners)
      7) mark terminal completion only after the requested round scoring succeeds
```

### 7.2c Run Re-Scoring
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

### 7.2d Shared Benchmark Baselines
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
- Rows with null or non-finite values in the active `target_col` are treated as unlabeled for that target: train/full-history fits drop them before `model.fit`, validation prediction/scoring artifacts include only labeled rows for that target, and folds that become fully unlabeled fail with `training_target_rows_all_unlabeled:split=<split>:target=<target_col>[:fold=<n>]`.
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
- Canonical scoring artifacts are materialized under `artifacts/scoring/` in two ways: training always writes `post_fold` artifacts during CV and may also materialize `post_training_core` or inclusive `post_training_full` depending on `training.post_training_scoring`, while later `run score` / `experiment score-round` can refresh any stage from saved predictions. The bundle includes `run_metric_series.parquet` plus staged parquet artifacts: `post_fold_per_era`, `post_fold_snapshots`, `post_training_core_summary`, and `post_training_full_summary` when materialized. Stage-limited `post_training_core` and `post_training_full` refreshes also persist `run_metric_series.parquet` so viz keeps the standard per-era and cumulative run-detail charts after summary-only rescoring. Historical runs may still expose legacy `post_training_summary` / `post_training_features_summary` files, which remain read-compatible. Training-time scoring and partial rescoring both rebuild recorded artifact metadata from the persisted scoring manifest so `score_provenance.json`, `results.json -> training.scoring.emitted_stage_files`, and `run.json -> training.scoring.emitted_stage_files` reflect the actual materialized bundle rather than the pre-persist in-memory stage map.
- `post_training_core_summary` is the flattened decision scorecard: it carries native/payout CORR summaries plus payout-backed `mmc`, `bmc`, `bmc_last_200_eras`, scalar `avg_corr_with_benchmark`, and `corr_delta_vs_baseline` summary stats when the aligned benchmark/meta inputs exist.
- When the requested stage is only `post_training_core`, the canonical stage omissions record `post_training_full=not_requested`. Availability-based omission reasons for `post_training_full` are reserved for requests that actually include feature-heavy diagnostics (`all` or `post_training_full`).
- `baseline_corr` is no longer persisted as a run metric; instead, provenance records whether payout-target baseline CORR came from the shared active-benchmark artifact or from transient fallback computation.
- Training scoring does not emit payout estimate fields because Numereng does not implement an official expected-payout estimator from validation metrics.
- Deferred or failed post-training scoring still leaves `results.json` and `metrics.json` on disk with `training.scoring.policy`, `status`, `requested_stage`, `refreshed_stages`, and optional `reason` / `error` metadata. `score_provenance.json` appears once scoring materializes successfully.
- Canonical post-run feature diagnostics are FNC-only. Training neutralizes to `fncv3_features`, persists `fnc` summaries in `post_training_full_summary`, and no longer emits feature-exposure summaries as canonical scoring outputs.
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
cli research status|run
  -> api.research_status|api.research_run
  -> features.agentic_research.get_research_status|run_research
      - persist supervisor state under:
          .numereng/experiments/<root>/agentic_research/state.json
          .numereng/experiments/<root>/agentic_research/ledger.jsonl
          .numereng/experiments/<root>/agentic_research/rounds/decision.json
          .numereng/experiments/<root>/agentic_research/rounds/rNNN.md
      - `research run` initializes state on first use
      - the prompt policy lives in `src/numereng/features/agentic_research/PROGRAM.md`
      - runtime layout is localized into `PROGRAM.md` plus `run.py`
      - select planner compute source from `src/numereng/config/openrouter/active-model.py`
      - checked-in default: `ACTIVE_MODEL_SOURCE=codex-exec`
      - `ACTIVE_MODEL_SOURCE=codex-exec` uses headless `codex exec`
      - `ACTIVE_MODEL_SOURCE=openrouter` uses the configured OpenRouter `ACTIVE_MODEL`
      - `codex-exec` inherits the user’s normal Codex configuration and environment; agentic research does not create a feature-specific `CODEX_HOME`
      - the LLM sees configs, report rows, experiment notes, recent ledger rows, and research memory
      - the LLM returns one JSON decision: run or stop
      - Python validates allowed config paths, validates the resulting `TrainingConfig`, rejects duplicates, writes one child config, trains, scores, and records the round
      - if no scored primary-metric row exists yet, the first round is a deterministic baseline copy before any LLM mutation
```

### 7.6 HPO
```text
cli hpo create
  -> api.hpo_create
  -> features.hpo.create_study
      - validate the v2 HPO study spec (`study_id`, `objective`, `search_space`, `sampler`, `stopping`)
      - persist immutable `study_spec.json`
      - create/resume one JournalStorage-backed Optuna study keyed by `study_id`
      - execute Optuna trials
      - each trial runs training + index_run
      - default objective is `bmc_last_200_eras.mean`
      - legacy `post_fold_champion_objective` still extracts from `post_fold_snapshots.parquet`
      - optional neutralization + custom metric extraction applies when scoring an alternate metric path
      - persist `study_summary.json`, `trials_live.parquet`, and sqlite study/trial rows after every trial
```

Metadata behavior in HPO:
- If launch metadata is already bound, HPO reuses it.
- Otherwise HPO sets default source `api.hpo.create` for trial runs.
- HPO studies are resumable by explicit `study_id`; immutable spec drift raises `hpo_study_spec_mismatch:<study_id>`.
- Mutable resume controls are in the `stopping` block (`max_trials`, `max_completed_trials`, `timeout_seconds`, `plateau`).
  `max_trials` and `max_completed_trials` are total-study caps across resumes; `timeout_seconds` is per invocation.
- Optuna pruning and multi-worker execution are intentionally out of scope for HPO v2.
- Default HPO metric: `bmc_last_200_eras.mean`.
- Legacy `post_fold_champion_objective` remains available explicitly:
  `0.25 * mean(corr_ender20_fold_mean | corr_native_fold_mean) + 2.25 * mean(bmc_fold_mean | bmc_ender20_fold_mean)`
  with `results.json` fallback when the snapshot artifact is missing or unusable.
- Duplicate deterministic trial params first reuse a completed study trial when possible, then reuse a finished deterministic run only when its scoring artifacts are intact; nonterminal pre-existing run dirs fail loudly and are never reset automatically.

### 7.6 Ensemble
```text
cli ensemble build
  -> api.ensemble_build
  -> features.ensemble.build_ensemble
      - validate method and inputs
      - blend predictions (rank_avg)
      - compute diagnostics
      - persist artifacts + sqlite rows

cli ensemble select
  -> api.ensemble_select
  -> features.ensemble.select_ensemble
      - validate source experiment rules
      - freeze target-level averaged candidates from all available runs by default
      - score bundle candidates on the active benchmark
      - prune redundant candidates by correlation
      - score named equal-weight variants
      - run exact weighted search with cached arrays and fast scoring kernels
      - persist selection artifacts under the experiment root
```

### 7.6 Submission + Neutralize
```text
cli serve package create|inspect|list|score|sync-diagnostics
  -> api.serve_package_*
  -> features.serving
      - persist one submission package under `.numereng/experiments/<id>/submission_packages/<package_id>/`
      - freeze explicit component sources, weights, blend rule, and optional post-processing
      - inspect compatibility separately for local live builds, artifact-backed live readiness, and Numerai-hosted model uploads
      - persist a stable preflight report under `artifacts/preflight/report.json`
      - `score` executes the final package artifact on local validation data (`runtime=auto|pickle|local`)
      - `score` persists package-native predictions, summaries, provenance, and metric series under `artifacts/eval/validation/<runtime>/`
      - `score` reuses `features.scoring.metrics.score_prediction_file_with_details(...)` directly instead of the run scorer and emits explicit target-labeled metrics
      - `sync-diagnostics` polls the exact uploaded compute-pickle id, then persists upload-scoped compute status/logs plus the latest available Numerai model diagnostics snapshot under `artifacts/diagnostics/<upload_id>/`

cli serve live build|submit
  -> api.serve_live_*
  -> features.serving
      - resolve current classic `live.parquet` and `live_benchmark_models.parquet`
      - inspect compatibility before build
      - prefer persisted `full_history_refit` model artifacts for run-backed components
      - use config-backed retraining only as a local/dev fallback
      - prepare historical data once per compatible feature/data context only when retraining is required
      - rank-blend and optionally neutralize the final vector
      - write per-component live artifacts plus a submit-ready parquet
      - `submit` then hands the parquet to `features.submission`

cli serve pickle build|upload
  -> api.serve_pickle_*
  -> features.serving
      - inspect compatibility before build or upload
      - reject local-only packages conservatively for hosted inference
      - require persisted run-backed model artifacts for every component
      - classify `model_upload_compatible` separately from verified `pickle_upload_ready`
      - validate requested data version / docker image against the Numerai API on upload
      - load persisted fitted models without retraining
      - serialize a self-contained Numerai-compatible `predict(live_features, live_benchmark_models)` callable
      - run an isolated hosted-runtime smoke before marking the pickle upload-ready
      - `upload` then hands the pickle to `features.submission`
      - optional `--wait-diagnostics` chains into package diagnostics sync after a successful upload without changing the package deployment status

cli run submit
  -> api.submit_predictions
  -> features.submission
      - resolve source (run_id XOR predictions_path)
      - validate classic live eligibility with `numerai_tools` against downloaded `live.parquet` ids (strict by default)
      - optional neutralization
      - upload predictions via Numerai client

features.submission model-upload path
  - validate `.pkl` source
  - resolve model id from model name
  - upload pickle through the Numerai client `model_upload(...)` wrapper

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
  -> local export output chosen by the operator (commonly `docs/numerai/forum/`); this is generated user data, not packaged product docs
  -> viz docs API appends "Forum Archive" under `/api/docs/numerai/tree` with year -> month index navigation
```

### 7.10 Numerai Docs Browsing
```text
viz docs numerai
  -> prefer repo docs/numerai/
  -> if no local mirror exists, docs page points the user at `uv run numereng docs sync numerai`
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
- Internal wheel builds used by cloud package-transfer flows include both `src/numereng` and `viz/api/numereng_viz`; public OSS support is repo-clone first, not wheel-first.

Store/remote monitor contract:
- `numereng.api.build_monitor_snapshot(request)` and `numereng monitor snapshot --workspace <path> --json` build one normalized read-only monitor snapshot for a single numereng store.
- `numereng.api.remote_*` and `numereng remote ...` provide SSH-backed repo sync, experiment authoring sync, experiment artifact pullback, ad hoc config push, and detached remote launch.
- Mission control overview merges:
  - the local store snapshot
  - zero or more SSH remote store snapshots loaded from gitignored profile YAMLs under `src/numereng/platform/remotes/profiles/*.yaml` or `NUMERENG_REMOTE_PROFILES_DIR`
- SSH monitoring is numereng-owned and read-only: the local viz backend runs `numereng monitor snapshot --json` on the remote machine over SSH and merges the returned snapshot.
- Remote experiment execution is also numereng-owned: detached remote windows launch `numereng experiment run-plan ...`, and `remote experiment status|maintain|stop` operate only on the durable run-plan state file for that exact row window.
- Remote detail reads stay local-server-owned too: experiment/run/study/ensemble detail routes keep the same frontend paths and add optional `source_kind` + `source_id` query params, while the local viz backend resolves canonical local run artifacts first, keeps a narrow read-only compatibility fallback for older pulled-cache artifacts under `.numereng/cache/remote_ops/pulls`, and only then dispatches remaining misses over SSH with remote Python helpers instead of starting a remote viz API.
- SSH remote profiles declare `shell: posix|powershell`; `posix` keeps the existing `cd ... && ...` command path, while `powershell` wraps the snapshot command in `powershell -NoProfile -Command "Set-Location ...; ..."` for Windows SSH targets.
- Remote ops use the same target profiles plus `python_cmd` for helper scripts and optional `runner_cmd` for detached CLI launches. Sync is local-driven and archive-based over SSH:
  - repo sync includes tracked files plus untracked nonignored files from the local working tree
  - repo sync excludes `.git`, `.numereng`, gitignored machine profiles, envs, caches, and build outputs
  - experiment sync includes only `.numereng/experiments/<id>/EXPERIMENT.md|run_plan.csv|configs/*|run_scripts/*`
  - remote experiment creation uses the target store root directly and does not overwrite an existing remote manifest
  - experiment pull preflights one remote experiment, selects only `FINISHED` runs, materializes full remote run directories into `.numereng/runs/<run_id>`, and reconciles the local experiment manifest from remote runtime truth while leaving active/incomplete remote runs to SSH fallback
  - experiment maintain is idempotent: terminal windows noop, live supervisors noop, and dead nonterminal windows relaunch the same row window with `--resume`
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
- Public viz metrics are canonical-only: `corr_*`, `fnc_*`, `mmc_*`, `bmc_*`, `bmc_last_200_eras_mean`, `cwmm_*`, `max_drawdown`, and `mmc_coverage_ratio_rows`.
- Viz also publishes generic payout-target aliases `corr_payout_mean` and `mmc_payout_mean`; `corr_payout_mean` comes from `corr_ender20`, while `mmc_mean` / `mmc_payout_mean` normalize to the payout-backed MMC surface with legacy `mmc_ender20` fallback for older runs.
- Viz does not expose payout-derived metrics or payout-specific routes.
- Per-era correlation payloads use `{ era, corr }`.
- Run-detail section routes remain independent (`manifest`, `metrics`, `per-era-corr`, `events`, `resources`, `trials`, `best-params`, `config`, `diagnostics-sources`), and `/api/runs/{run_id}/bundle` now also accepts an optional `sections=` query so the frontend can request only the first-paint subset or one lazy tab payload at a time. Omitting `sections` preserves the full compatibility bundle.
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

Viz backend API-doc routes:
- `/api/docs`
- `/api/openapi.json`
- `/api/redoc`

Important UI contract:
- There are no standalone `/run-ops` or `/configs` frontend pages.
- The frontend is SSR-first again. Server-side route loads use `http://127.0.0.1:8502/api` by default and may be overridden with `VIZ_API_BASE`; browser requests still use `/api`.
- The root layout is intentionally local-fast. It loads only the local experiment list plus system capabilities and does not fetch remote-aware mission-control data.
- `/experiments` is a mission-control dashboard, not a launch surface. It route-loads `/api/experiments/overview?include_remote=false`, then refreshes `/api/experiments/overview?include_remote=true` after hydration and polls that remote-aware view every 10 seconds while visible, preserving the last successful snapshot on transient failures.
- The frontend docs pages keep exclusive ownership of `/docs`; generated FastAPI Swagger/ReDoc endpoints are namespaced under `/api/docs` and `/api/redoc` to avoid route collisions on direct reloads and deep links.
- The mission-control overview is federated across the local store plus all enabled SSH remotes. The local viz backend remains the only UI/API process; remote hosts are polled over SSH via `numereng monitor snapshot --json` and are surfaced in `overview.sources` with `live|cached|unavailable` source state plus persisted bootstrap metadata.
- The mission-control overview is canonicalized by `experiment_id`: when an experiment exists locally and on one remote, the overview keeps one local-primary row and annotates it with remote freshness overlay metadata instead of rendering duplicate local/remote rows.
- The global left-rail experiment navigator now uses the local experiment list from the root layout, so unrelated routes stay fast and remote-free. Remote-aware canonicalized experiment rows remain on `/experiments`.
- `just viz` is the canonical dashboard launcher for the repo-root workflow.
- Mission-control live cards render one canonical progress instrument per live experiment. The frontend selects a `primary` run by `updated_at desc`, then `exact > estimated > indeterminate`, then `run_id`, and renders only that run's bar plus adjacent percent readout.
- Experiment detail exposes Analysis + Progress + Run Ops tabs; launch/control remains monitor-only.
- Launch/control actions are CLI/API-only by design.
- `experiment pack` writes `EXPERIMENT.pack.md` into the experiment directory and includes `EXPERIMENT.md` plus a dashboard-aligned scalar run summary table only.
- Experiment ranking defaults to `bmc_last_200_eras_mean`.
- Run detail loading is progressive and bundle-first: the shell renders from one initial partial bundle (`manifest`, `metrics`, `scoring_dashboard`), and diagnostics/artifacts/timeline sections lazy-load their own bundle sections with localized loading states.
- Remote experiment, run, study, and ensemble detail pages are read-only parity surfaces. Explicit source context is still preserved through query params for remote-only and ambiguous cases, but the default local route now uses local-first artifact resolution: canonical local store -> pulled remote cache compatibility -> unique-source SSH fallback.
- Run Ops and the dedicated run page now share a partial-bundle loader and request cache/abort behavior rather than a many-endpoint first-paint fan-out.
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
- `.numereng/notes/__RESEARCH_MEMORY__/CURRENT.md` is the canonical rolling research-memory surface.
- `.numereng/notes/__RESEARCH_MEMORY__/experiments/*.md` stores one durable review per completed experiment.
- `.numereng/notes/__RESEARCH_MEMORY__/topics/*.md` stores hybrid topic ledgers.
- `.numereng/notes/__RESEARCH_MEMORY__/legacy-progression/` preserves the prior `__PROGRESSION__` monthly summaries as historical inputs only.

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
31. Null or non-finite labels in the active `target_col` are treated as unlabeled rows for that target, so supervised training and target-aware validation/scoring exclude them instead of sending them to the model backend.
32. If label filtering leaves zero rows in a train, validation, or full-history batch, training fails with `training_target_rows_all_unlabeled:...` rather than a backend-specific fit error.
33. Managed AWS extract only promotes `runs/<run_id>/*` archive members; non-`runs/*` members are skipped and recorded as warnings.
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
