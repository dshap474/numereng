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
    run.py
    experiment.py
    hpo.py
    ensemble.py
    neutralization.py
    submission.py (via run.py handlers)
    store.py
    numerai.py
    cloud/{ec2.py,aws.py,modal.py}

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
- Typed request/response definitions: `src/numereng/api/contracts.py`

Boundary behavior:
- API modules map feature/internal exceptions to `PackageError`.
- CLI exit codes are fixed:
  - `0`: success/help
  - `1`: runtime/boundary failure
  - `2`: parse/usage failure

Command families:
- `run`: `train`, `submit`
- `experiment`: `create`, `list`, `details`, `train`, `promote`, `report`
- `hpo`: `create`, `list`, `details`, `trials`
- `ensemble`: `build`, `list`, `details`
- `neutralize`: `apply`
- `store`: `init`, `index`, `rebuild`, `doctor`
- `cloud`: `ec2`, `aws`, `modal`
- `numerai`: `datasets` (`list`, `download`), `models`, `round`, `forum scrape`

## 6. Runtime Persistence Model
Default store root: `.numereng`

Canonical top-level dirs:
- `runs`
- `datasets`
- `cloud`
- `experiments`
- `logs`
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

  experiments/
    <experiment_id>/
      experiment.json
      EXPERIMENT.md
      configs/*.json
      hpo/<study_id>/...
      ensembles/<ensemble_id>/...

  hpo/<study_id>/...                  # global studies
  ensembles/<ensemble_id>/...         # global ensembles
  datasets/
  cloud/
  logs/
  notes/
```

Data model split:
- Filesystem holds canonical run/experiment artifacts.
- sqlite indexes query surfaces for runs/metrics/experiments/HPO/ensembles/telemetry.
- `store rebuild` re-derives index state from filesystem artifacts.

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
- Feature: `features.training.run_training`

```text
cli run train
  -> parse/validate args
  -> bind launch metadata source=cli.run.train
  -> api.run_training
      (default metadata source=api.run.train only when unbound)
  -> features.training.run_training
      1) load strict JSON config
      2) resolve training profile (`simple|purged_walk_forward|submission`)
      3) resolve loading/scoring modes
      4) compute deterministic run_hash -> run_id
      5) resolve output root + run dir
      6) write RUNNING manifest + resolved config
      7) write run-local live stage/status log (`run.log`) during execution
      8) train + persist predictions/results/metrics placeholders
      9) index run (mandatory pre-finalization)
      10) write FINISHED/FAILED manifest
      11) for non-`submission`, call `features.training.scoring.run_post_training_scoring` from saved predictions and refresh run artifacts (`corr`, `fnc`, `mmc`, `cwmm`, `bmc`, `bmc_last_200_eras`)
```

Data loading and CV rules:
- `data.dataset_variant=non_downsampled` resolves canonical defaults from `.numereng/datasets/<data_version>/`.
- `data.dataset_variant=downsampled` resolves `full.parquet -> downsampled_full.parquet` and `full_benchmark_models.parquet -> downsampled_full_benchmark_models.parquet`.
- `data.dataset_variant` is required and allowed only: `non_downsampled|downsampled`.
- Official downsampling artifacts (built by `dataset-tools build-full-datasets`) are:
  `full.parquet`, `full_benchmark_models.parquet`, `downsampled_full.parquet`, `downsampled_full_benchmark_models.parquet`.
- `dataset-tools build-full-datasets` uses default era filtering of every 4th era (`offset=0`) and supports `--skip-downsample` to build only full datasets.
- `data.dataset_scope=train_only` uses `train.parquet` only.
- `data.dataset_scope=train_plus_validation` uses `train.parquet` plus `validation.parquet`, and applies `data_type=validation` filtering only on validation sources.
- `purged_walk_forward` uses walk-forward defaults of `chunk_size=156`; embargo is horizon-based (`20d -> 8`, `60d -> 16`).
- `training.resources.max_threads_per_worker` accepts integer >= 1 or `"default"`; `"default"` (and omitted/`null` for backward compatibility) resolves to `max(1, floor(available_cpus / parallel_folds))`.
- FNC diagnostics are computed in post-run scoring by neutralizing predictions to the run's feature set (`data.feature_set`) rather than during model fitting.
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
      - enforce output_dir == store_root (or default)
      - run training with experiment_id
      - link run into experiment manifest/store
```

### 7.4 HPO
```text
cli hpo create
  -> api.hpo_create
  -> features.hpo.create_study
      - validate config + search space
      - execute Optuna trials
      - each trial runs training + index_run
      - optional neutralization + metric extraction
      - persist trial rows + study summary
```

Metadata behavior in HPO:
- If launch metadata is already bound, HPO reuses it.
- Otherwise HPO sets default source `api.hpo.create` for trial runs.

### 7.5 Ensemble
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
      - validate live eligibility (strict by default)
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
2. Neutralization source XOR: exactly one of `run_id` or `predictions_path`.
3. Training/HPO config files are JSON-only and reject unknown keys.
4. Training profile allowed only: `simple|purged_walk_forward|submission`.
5. Legacy training flags `--method` and `--method-overrides-json` are hard-fail.
6. Run IDs are deterministic hash-derived IDs (12-char prefix).
7. Training success requires pre-finalization `index_run` success.
8. Experiment train defaults `output_dir` to `store_root`; when provided it must equal `store_root`.
9. Ensemble method is `rank_avg` only; at least 2 deduped runs are required.
10. Cloud AWS train backend is only `sagemaker|batch`; `--spot` and `--on-demand` are mutually exclusive.
11. `cloud aws train pull` is download-only and writes artifacts to `.numereng/cloud/<run_id>/pull`.
12. `cloud aws train extract` is the separate step that unpacks tarballs into `.numereng/runs/*` and indexes runs.
13. Cloud Modal deploy requires full ECR URI `<registry>/<repository>:<tag>`.
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
24. `model.x_groups` benchmark aliases (`benchmark`, `benchmarks`, `benchmark_models`) are invalid and hard-fail with `training_model_x_groups_benchmark_not_supported`.
25. Training writes run-local live logs to `runs/<run_id>/run.log` and records the file in `run.json -> artifacts.log`.
26. Training never applies row-level subsampling; any size reduction must happen at dataset construction/downsampling time.
27. Managed AWS extract only promotes `runs/<run_id>/*` archive members; non-`runs/*` members are skipped and recorded as warnings.
28. Managed AWS extract hard-fails on unsafe archive members (absolute/traversal paths, links, invalid run IDs) and on hash-mismatched run-dir collisions.
29. Managed AWS extract indexes only extracted run IDs and does not fallback-index the outer cloud run ID.
30. SageMaker managed entrypoint removes store DB sidecars (`numereng.db*`) from managed output before artifact packaging.
31. Managed AWS `--state-path` must resolve under `<store_root>/cloud/*.json`.
32. EC2 and Modal `--state-path` must resolve to `.numereng/cloud/*.json` and must use `.json` extension.
33. EC2 pull and S3-prefix copy reject traversal keys by skipping unsafe paths and reporting them in `skipped_unsafe_keys`.

## 12. Testing + Verification
Primary checks:
- `make test` (fast gate)
- `make test-all` (full gate)
- `uv build`

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
