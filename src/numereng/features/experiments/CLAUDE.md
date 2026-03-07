# Experiments Visualization Contract

Explorations driven by `features.experiments` land directly in the viz read path, so this note keeps downstream viz owners and agents aligned on the expectations that experimentation metadata, manifests, and metrics feed.

## Surface points into viz

- `VizStoreAdapter` (via `features.viz`) is the single point that touches `.numereng/experiments/*`, `.numereng/runs/*`, and the SQLite store (`experiments`, `runs`, `metrics`, `run_jobs`, etc.). It always tries the indexed tables first and falls back to the filesystem tree, so experiment/manifest operations must call `features.store.index_run` / `upsert_experiment` whenever metadata changes so viz sees the update.
- `list_experiment_configs` and `list_all_configs` sanitize every config by reading the YAML, hashing it, and synthesizing the summary that the dashboard shows (`run_id`, `model_type`, `target`, `feature_set`, `stages`). Viz expects `data.target` (or `target_col`/`target_train`) inside the config map even if the experiment only logically targets one column.
- `list_experiment_runs` and `list_experiment_round_results` combine manifest metadata, run metrics, and metrics files to surface run timelines. Champion linkage comes from the experiment manifest (`champion_run_id`) and is honored even during the filesystem fallback path.
- `linked_runs_for_configs` ties experiment configs to the canonical `run_id` (via `run_jobs.canonical_run_id`) so viz can open the latest qualifying run.

## Metric canonicalization & queries

- `_normalize_round_metrics` flattens nested metrics (e.g. `{"corr": {"mean": ...}}`) and normalizes aliases to canonical keys: `corr20v2_mean`, `corr20v2_sharpe`, `mmc_mean`, `bmc_mean`, `cwmm_mean`, `payout_estimate_mean`, `max_drawdown`, etc. This keeps the viz schema stable regardless of how training writes the row.
- `_expand_metric_query_names` lets viz requests use canonical names while the adapter expands down to all known persisted aliases before hitting the `metrics` table (`corr.mean`, `corr_sharpe`, `payout_score`, …). That means experiments must keep writing at least one variant of each metric that downstream features rely on.
- `get_metrics_for_runs` pulls the normalized metrics map and can filter to a subset of canonical names, so experiment promotions with new metrics must either persist them in `metrics` or provide translated keys that `_normalize_round_metrics` handles.

## Fallback computations viz relies on

- When `payout_estimate_mean` is missing, viz computes `clip(0.75 * corr20v2_mean + 2.25 * mmc_mean, -0.05, +0.05)` to mimic Numerai Classic payout scoring. This fallback applies to `target_ender_20` contexts; non-Ender targets intentionally keep payout estimate unset/null.
- Per-era CORR (val_per_era_corr20v2) and payout maps are expected artifacts (`artifacts/predictions/val_per_era_*.parquet|csv`), but viz can derive them from the canonical predictors/meta model if they are absent. The derivation path requires `numerai_tools` in the environment and `score_provenance.json` plus the predictions/meta files to merge on `era` + `id`. Experiments must ensure those base artifacts exist for viz derivation.
- `mmc_coverage_ratio_rows` is computed on demand from `score_provenance.json --> joins.meta_overlap_rows / joins.predictions_rows` when the metric isn’t persisted. That ratio is surfaced within `get_run_metrics` so the dashboard shows MMC coverage even if the training run never stored that field.

## Provenance, diagnostics, and artefact resolution

- `score_provenance.json` (or `manifest.artifacts.score_provenance`) is the trust source for `columns`, `joins`, and `sources`. Viz exposes that information through `diagnostics_sources`, reporting column names, join counts, artifact paths, and existence/sha256 metadata so users can trace how a metric was computed.
- Predictions and meta-model artifacts are resolved by checking, in order: the score provenance `sources` map, explicit `artifacts` entries in `run.json`, the `results.json` output, and then the `artifacts/predictions/` directory for parquet/csv blobs. Losing any link in that chain means viz cannot build the per-era fallback or show coverage diagnostics, so experiment runs must annotate artifacts consistently.
- Meta-model resolution also accepts dataset-level pointers (`datasets/<version>/meta_model.parquet`) when the run manifest supplies `data.version`/`data.data_version`. Experiments that upgrade dataset snapshots must still write a deterministic pointer so viz can find the meta model without exploring the entire dataset tree.

## Read-path resilience

- `get_run_manifest` and `get_run_metrics` prefer SQLite-backed rows but gracefully read `run.json`/`metrics.json` when the DB is missing. That means offline experiments (without the index) still show up in viz if their directories contain the canonical files.
- Derivation functions (`_scoring_context`, `_derive_per_era_corr_fallback`, `_derive_per_era_payout_map_fallback`) aggressively fallback to column heuristics (infer `era`, `target`, `prediction` columns by case/substring) so experiments that rename columns still work as long as a sensible guess exists.

## Maintenance & anti-drift

Any experiment workflow change that touches the store manifest, run artifact names, experiment config metadata, or the viz schema must update:

1. `src/numereng/features/experiments/CLAUDE.md` (this file)
2. `docs/llms.txt` and `docs/ARCHITECTURE.md` so viz flows stay documented
3. `src/numereng/features/viz/store_adapter.py` where the flattening/alias logic lives
4. `docs/numereng/*` docs referenced by viz if there are new user-visible behaviours

Keep the experiments store index in lockstep with `.numereng` so viz’s SQLite-first reads always see the latest champion/promoted runs and experiment metadata.
