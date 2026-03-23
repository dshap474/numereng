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
    feature_neutralization/    # prediction neutralization
    experiments/               # experiment lifecycle + run linkage
    agentic_research/          # strategy-driven continuous research supervisor + planner assets
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
      custom_models/           # model plugin path

  api/
    contracts.py
    __init__.py              # curated public facade
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
    commands/{run,experiment,hpo,ensemble,neutralize,store,numerai,cloud*}.py
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
- `run`: `train`, `score`, `submit`
- `experiment`: `create`, `list`, `details`, `archive`, `unarchive`, `train`, `score-round`, `promote`, `report`, `pack`
- `research`: `init`, `status`, `run`
- `hpo`: `create`, `list`, `details`, `trials`
- `ensemble`: `build`, `list`, `details`
- `neutralize`: `apply`
- `dataset-tools`: `build-downsampled-full`
- `store`: `init`, `index`, `rebuild`, `doctor`, `materialize-viz-artifacts`
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
      run.json
      run.log
      resolved.json
      results.json
      metrics.json
      score_provenance.json
      artifacts/predictions/*.parquet
        numereng-managed parquet artifacts use ZSTD level 3

  experiments/
    <experiment_id>/
      experiment.json
      EXPERIMENT.md
      EXPERIMENT.pack.md
      configs/*.json
      agentic_research/
        program.json
        lineage.json
        rounds/rN/
          codex_prompt.txt
          codex_usage.json
          codex_stdout.jsonl
          codex_stderr.txt
          codex_last_message.json
          codex_decision.json
          planned_configs.json
          report.json
          round_summary.json
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
  cloud/
  notes/
```

Data model split:
- Filesystem holds canonical run/experiment artifacts.
- sqlite indexes query surfaces for runs/metrics/experiments/HPO/ensembles/telemetry.
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
      2) prepare run identity, output paths, lock, manifest bootstrap, telemetry
      3) load training data + prepare model data loader
      4) train model + persist predictions/results/metrics placeholders
      5) append `post_fold` pruning artifacts during CV when fold outputs exist
      6) index run (mandatory pre-finalization)
      7) write FINISHED/FAILED manifest and release run lock
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
- `model.device` is the canonical explicit device contract for training and is allowed only: `cpu|cuda`.
- `model.device` is valid only for `LGBMRegressor`; legacy `model.params.device_type` remains supported but must exactly match when both are present.
- Canonical `model.x_groups` / `model.data_needed` are features-only by default; `era` and `id` are never auto-included and are rejected as input groups.
- FNC diagnostics are computed in post-run scoring by neutralizing predictions to dataset feature set `fncv3_features`, independent of the run's training feature set, then correlating against the scoring target being evaluated (`fnc` for the native target, `fnc_<alias>` for extra scoring targets).
- `corr`, `fnc`, and `mmc` are delegated to `numerai_tools`; `cwmm` is computed locally using the official diagnostic semantics of Pearson correlation between the Numerai-transformed submission and the raw meta-model series.
- Scoring implementation is optimized behind the existing boundary: `features.scoring.metrics` orchestrates cached parquet reads and canonical artifact/provenance assembly, while `features.scoring._fastops` owns the NumPy/Numba per-era kernels that replace the older pandas callback hot path.
- Canonical scoring artifacts are materialized under `artifacts/scoring/` in two ways: training writes `post_fold` artifacts during CV, and later `run score` / `experiment score-round` writes deferred post-training artifacts from saved predictions. The bundle includes `run_metric_series.parquet` plus staged parquet artifacts: `post_fold_per_era`, `post_fold_snapshots`, `post_training_core_summary`, and `post_training_full_summary` when enabled. Historical runs may still expose legacy `post_training_summary` / `post_training_features_summary` files, which remain read-compatible. Partial rescoring overwrites only the selected stage artifacts while still refreshing manifest/provenance/results/metrics metadata, and the refreshed provenance artifact block is rebuilt from the persisted scoring manifest so it matches the materialized bundle.
- `post_training_core_summary` is the flattened decision scorecard: it carries native/payout CORR summaries plus payout-backed `mmc`, `bmc`, `bmc_last_200_eras`, scalar `avg_corr_with_benchmark`, and `corr_delta_vs_baseline` summary stats when the aligned benchmark/meta inputs exist.
- `baseline_corr` is no longer persisted as a run metric; instead, provenance records whether payout-target baseline CORR came from the shared active-benchmark artifact or from transient fallback computation.
- Training scoring does not emit payout estimate fields because Numereng does not implement an official expected-payout estimator from validation metrics.
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
- Post-`FINISHED` index refresh is best-effort.

### 7.3 Experiment
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
cli research init|status|run
  -> api.research_init|api.research_status|api.research_run
  -> features.agentic_research.init_research|get_research_status|run_research
      - persist supervisor state under:
          .numereng/experiments/<root>/agentic_research/program.json
          .numereng/experiments/<root>/agentic_research/lineage.json
          .numereng/experiments/<root>/agentic_research/rounds/rN/*
      - `research init` requires one persisted strategy id
      - supported strategies:
          - `numerai-experiment-loop`
          - `kaggle-gm-loop`
      - phase-aware strategies persist `current_phase` in program state and surface it in API/CLI status
      - select planner compute source from `src/numereng/config/openrouter/active-model.py`
      - checked-in default: `ACTIVE_MODEL_SOURCE=codex-exec`
      - `ACTIVE_MODEL_SOURCE=codex-exec` uses headless `codex exec`
      - `ACTIVE_MODEL_SOURCE=openrouter` uses the configured OpenRouter `ACTIVE_MODEL`
      - isolate planner runtime under a dedicated learner `CODEX_HOME` with lean headless defaults (`gpt-5.4`, low reasoning, read-only sandbox, shell tool disabled, no apps/multi-agent/js_repl) when the source is `codex-exec`
      - load prompt/schema assets from the selected strategy package under `features/agentic_research/assets/<strategy-id>/`
      - feed the selected planner backend a closed-world context bundle: metric policy, strategy context, best/prior round summaries, one authoritative base config snapshot, allowed override paths, and validated config examples
      - validate planner output against the checked-in strategy schema, then materialize full configs locally by applying override-only deltas onto the authoritative base config before `TrainingConfig` validation
      - persist planner telemetry per round (`codex_usage.json`, `codex_stdout.jsonl`, `codex_stderr.txt`, `codex_last_message.json`) so prompt cost and latency are measurable; artifact names remain `codex_*` for compatibility even when the planner source is OpenRouter
      - write 4-5 configs into the active experiment, train sequentially, deferred-score the round, and persist summary artifacts
      - track plateau streak on `bmc_last_200_eras.mean` with `bmc.mean` tie-break
      - let strategy policy decide whether the next round stays in the current phase, advances phase, or completes the campaign
      - after two failed rounds, force one scale-confirmation round; if that also fails, create a fresh child experiment and continue there
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
Telemetry is an additive local instrumentation layer attached to training sessions.

### 8.1 Source binding model
```text
caller binds metadata (optional)
  -> bind_launch_metadata(source=...)
  -> training reads get_launch_metadata()

if metadata absent:
  api.run_training      => source=api.run.train
  api.experiment_train  => source=api.experiment.train
  hpo.create_study      => source=api.hpo.create (trial fallback)
  modal runtime         => source=cloud.modal.runtime (fallback)
```

### 8.2 Session lifecycle
```text
begin_local_training_session
  -> insert queued rows into:
       logical_runs
       run_jobs
       run_attempts
  -> emit job_queued event
  -> append initial log + sample

mark_job_starting -> mark_job_running
  -> update run_jobs/run_attempts/logical_runs
  -> emit job_starting/job_running

during training:
  -> emit stage_update events
  -> emit metric_update events
  -> append logs and resource samples

terminal:
  -> mark_job_completed OR mark_job_failed
  -> persist canonical_run_id/run_dir/error payload
```

### 8.3 Experiment scoping behavior
Experiment id for telemetry job rows is resolved as:
1. explicit `experiment_id` argument (preferred)
2. fallback inference from config path under:
   `.numereng/experiments/<experiment_id>/configs/*`

Operational impact:
- `numereng run train --experiment-id <id>` guarantees experiment-scoped monitor visibility.
- Running directly from experiment config paths can still auto-scope via path inference.

### 8.4 Reliability contract
- Telemetry writes are fail-open.
- Telemetry failures must not fail training commands.

## 9. Viz Frontend/Backend Topology

### 9.1 Backend API shape (`features.viz`)
Read-only routes include:
- `/api/system/capabilities`
- `/api/experiments*`
- `/api/configs*`
- `/api/run-jobs*`
- `/api/runs/*`
- `/api/studies*`
- `/api/ensembles*`
- `/api/docs/*`, `/api/notes/*`

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
- Run Ops UI is embedded on experiment detail (`/experiments/[id]`) and is monitor-only.
- Launch/control actions are CLI/API-only by design.
- `experiment pack` writes `EXPERIMENT.pack.md` into the experiment directory and includes `EXPERIMENT.md` plus a dashboard-aligned scalar run summary table only.
- Experiment ranking defaults to `bmc_last_200_eras_mean`.
- Run detail loading is progressive: the shell renders immediately from the selected run list row, and tab sections fetch lazily with localized loading states.
- Run Ops and the dedicated run page no longer block on `/api/runs/{run_id}/bundle`; they share the same section-based loader and request cache/abort behavior.
- Run detail/chart reads are artifact-backed first; older runs that only have legacy scoring files are adapted in memory to the canonical dashboard payload, and rescoring remains the canonical backfill path.
- Normal experiment/config catalogs exclude archived experiments by default.
- Direct experiment lookups remain archive-aware, so `/experiments/[id]` can still render an archived experiment when addressed directly.

### 9.3 Run monitor data flow
```text
CLI/API training launch
  -> telemetry rows written to sqlite
  -> viz backend reads sqlite/filesystem
  -> frontend polls /api/run-jobs?experiment_id=<id>
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
8. Run IDs are deterministic hash-derived IDs (12-char prefix).
9. Training success requires pre-finalization `index_run` success.
10. Experiment train defaults `output_dir` to `store_root`; when provided it must equal `store_root`.
11. Ensemble method is `rank_avg` only; at least 2 deduped runs are required.
12. Cloud AWS train backend is only `sagemaker|batch`; `--spot` and `--on-demand` are mutually exclusive.
13. `cloud aws train pull` is download-only and writes artifacts to `.numereng/cloud/<run_id>/pull`.
14. `cloud aws train extract` is the separate step that unpacks tarballs into `.numereng/runs/*` and indexes runs.
15. Cloud Modal deploy requires full ECR URI `<registry>/<repository>:<tag>`.
14. Cloud state files (`--state-path`) must parse into valid typed state objects.
15. Telemetry is fail-open and must never block training completion.
16. Launch metadata precedence is outermost binding first; API defaults apply only when unbound.
17. Experiment-scoped monitor queries depend on `run_jobs.experiment_id`; pass `--experiment-id` when launching ad-hoc run training.
18. Viz API is read-only; run control is intentionally not wired into frontend controls.
19. `train_plus_validation` filtering (materialized and fold-lazy) must never apply `data_type=validation` to `train.parquet`.
20. Purged walk-forward defaults are fixed at `chunk_size=156` and horizon-based embargo (`20d -> 8`, `60d -> 16`).
21. Horizon resolution uses `data.target_horizon` first and target-name inference second; unresolved ambiguity is a hard error.
22. Purged walk-forward CV requires `chunk_size + 1` or more unique eras.
23. Benchmark model parquet data is metrics-only (BMC/correlation diagnostics) and is not included as training features.
24. Canonical `model.x_groups` / `model.data_needed` are features-only; identifier groups (`era`, `id`) and benchmark aliases (`benchmark`, `benchmarks`, `benchmark_models`) are rejected.
25. Canonical FNC uses dataset feature set `fncv3_features` and correlates against the scoring target being evaluated; if feature-neutral metrics are enabled and `features.json` does not define `fncv3_features`, post-run scoring fails.
26. Benchmark post-run scoring requires nonzero `(id, era)` overlap with strict era alignment and emits `bmc` / `bmc_last_200_eras` on the maximum available overlapping benchmark window; meta-model scoring also requires strict era alignment but emits `mmc` / `cwmm` on the maximum available overlapping meta-model window whenever any overlap exists.
27. `full_history_refit` is final-fit only for training and does not emit `post_fold`; deferred `post_training_core` / `post_training_full` scoring can still be materialized later from saved predictions.
28. Training writes run-local live logs to `runs/<run_id>/run.log` and records the file in `run.json -> artifacts.log`.
29. Training never applies row-level subsampling; any size reduction must happen at dataset construction/downsampling time.
30. Managed AWS extract only promotes `runs/<run_id>/*` archive members; non-`runs/*` members are skipped and recorded as warnings.
31. Managed AWS extract hard-fails on unsafe archive members (absolute/traversal paths, links, invalid run IDs) and on hash-mismatched run-dir collisions.
32. Managed AWS extract indexes only extracted run IDs and does not fallback-index the outer cloud run ID.
33. SageMaker managed entrypoint removes store DB sidecars (`numereng.db*`) from managed output before artifact packaging.
34. Managed AWS `--state-path` must resolve under `<store_root>/cloud/*.json`.
35. Archived experiments are read-only: `experiment train` and `experiment promote` must fail until the experiment is unarchived.
36. Experiment archive moves affect only `.numereng/experiments/*`; run artifacts remain canonical under `.numereng/runs/*`.
37. Managed AWS and EC2 `runtime_profile` selects packaging only (`standard|lgbm-cuda`) and never overrides the training config device.
38. SageMaker CUDA submit requires config device `cuda`, `runtime_profile=lgbm-cuda`, and a GPU instance type (`ml.g*` or `ml.p*`); mismatches fail before submit.
39. EC2 CUDA runtime install requires persisted/requested GPU instance state plus `runtime_profile=lgbm-cuda`; mismatches fail before remote install.
40. CUDA training is fail-fast; numereng does not silently fall back from `cuda` to CPU.
41. EC2 and Modal `--state-path` must resolve to `.numereng/cloud/*.json` and must use `.json` extension.
42. EC2 pull and S3-prefix copy reject traversal keys by skipping unsafe paths and reporting them in `skipped_unsafe_keys`.

## 12. Testing + Verification
Primary checks:
- `make fmt`
- `make test` (fast gate)
- `make test-all` (full gate)
- `uv build`

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

5. Fail-open telemetry
- Benefit: observability failures do not break critical pipeline execution.
- Cost: partial observability during transient db/write issues.

## 14. Change Guidance
When changing behavior/contracts:
1. Preserve dependency direction and layering.
2. Preserve CLI/API compatibility unless intentionally versioned.
3. Keep API and CLI files <= 500 LOC.
4. Keep API boundary translation to `PackageError`.
5. If training/telemetry/viz flows change, update both `docs/llms.txt` and this file in the same PR.
