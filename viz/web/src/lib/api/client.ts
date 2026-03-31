import { browser } from '$app/environment';
import { normalizedSource, sourceQueryParams, type SourceContext } from '$lib/source';

const serverProcess = globalThis as typeof globalThis & {
	process?: {
		env?: Record<string, string | undefined>;
	};
};
const BASE = browser ? '/api' : (serverProcess.process?.env?.VIZ_API_BASE ?? 'http://127.0.0.1:8502/api');

type FetchFn = typeof globalThis.fetch;

async function get<T>(path: string, fetchFn: FetchFn = globalThis.fetch, signal?: AbortSignal): Promise<T> {
	const res = await fetchFn(`${BASE}${path}`, signal ? { signal } : undefined);
	if (!res.ok) {
		throw new Error(`API ${res.status}: ${path}`);
	}
	return res.json();
}

export interface Experiment {
	experiment_id: string;
	name: string;
	status: string;
	preset: string | null;
	hypothesis: string | null;
	created_at: string;
	updated_at: string;
	champion_run_id: string | null;
	runs?: string[];
	run_count?: number;
	tags: string[];
}

export interface ExperimentOverviewSummary {
	total_experiments: number;
	active_experiments: number;
	completed_experiments: number;
	live_experiments: number;
	live_runs: number;
	queued_runs: number;
	attention_count: number;
}

export interface ExperimentOverviewItem {
	experiment_id: string;
	name: string;
	status: string;
	created_at: string;
	updated_at: string;
	run_count?: number | null;
	tags: string[];
	has_live: boolean;
	live_run_count: number;
	attention_state: 'failed' | 'stale' | 'canceled' | 'none' | string;
	latest_activity_at: string | null;
	source_kind?: 'local' | 'ssh' | string;
	source_id?: string;
	source_label?: string;
	detail_href?: string | null;
}

export interface LiveRunOverview {
	run_id: string;
	experiment_id?: string | null;
	experiment_name?: string | null;
	job_id: string | null;
	config_id: string | null;
	config_label: string;
	status: string;
	current_stage: string | null;
	progress_percent: number | null;
	progress_mode?: 'exact' | 'estimated' | 'indeterminate' | string;
	progress_label: string | null;
	updated_at: string | null;
	terminal_reason: string | null;
	source_kind?: 'local' | 'ssh' | string;
	source_id?: string;
	source_label?: string;
	backend?: string | null;
	provider_run_id?: string | null;
	detail_href?: string | null;
}

export interface LiveExperimentOverview {
	experiment_id: string;
	name: string;
	status: string;
	tags: string[];
	live_run_count: number;
	queued_run_count: number;
	attention_state: 'failed' | 'stale' | 'canceled' | 'none' | string;
	latest_activity_at: string | null;
	aggregate_progress_percent: number | null;
	runs: LiveRunOverview[];
	source_kind?: 'local' | 'ssh' | string;
	source_id?: string;
	source_label?: string;
	detail_href?: string | null;
}

export interface RecentExperimentActivityItem {
	experiment_id: string;
	experiment_name: string;
	run_id: string | null;
	job_id: string | null;
	config_id: string | null;
	config_label: string;
	status: string;
	current_stage: string | null;
	progress_percent: number | null;
	progress_mode?: 'exact' | 'estimated' | 'indeterminate' | string;
	progress_label: string | null;
	updated_at: string | null;
	finished_at: string | null;
	terminal_reason: string | null;
	source_kind?: 'local' | 'ssh' | string;
	source_id?: string;
	source_label?: string;
	backend?: string | null;
	provider_run_id?: string | null;
}

export interface ExperimentOverviewResponse {
	generated_at?: string | null;
	summary: ExperimentOverviewSummary;
	experiments: ExperimentOverviewItem[];
	live_experiments: LiveExperimentOverview[];
	recent_activity: RecentExperimentActivityItem[];
	sources?: Array<{
		kind: string;
		id: string;
		label: string;
		host?: string | null;
		store_root?: string;
		state?: string;
		bootstrap_status?: 'ready' | 'degraded' | string;
		last_bootstrap_at?: string | null;
		last_bootstrap_error?: string | null;
	}>;
}

export interface ExperimentRun {
	run_id: string;
	experiment_id: string;
	status: string;
	is_champion: boolean;
	metrics: Record<string, number>;
	created_at: string;
	run_name?: string | null;
	model_type?: string | null;
	seed?: number | null;
	target?: string | null;
	target_train?: string | null;
	target_payout?: string | null;
	target_col?: string | null;
	feature_set?: string | null;
}

export interface ExperimentRoundResult {
	result_id: string;
	run_id: string;
	name: string;
	round_index: number | null;
	created_at: string;
	metrics: Record<string, number>;
	model_type?: string | null;
	target?: string | null;
	target_train?: string | null;
	target_payout?: string | null;
	feature_set?: string | null;
	source_file?: string | null;
}

export interface RunManifest {
	run_id?: string;
	status?: string;
	model_type?: string;
	model?:
		| string
		| {
				type?: string;
				[key: string]: unknown;
		  }
		| null;
	target?: string;
	data?:
		| {
				target_train?: string;
				target_payout?: string;
				target_col?: string;
				feature_set?: string;
				[key: string]: unknown;
		  }
		| null;
	created_at?: string;
	[key: string]: unknown;
}

export interface RunEvent {
	id: number;
	run_id: string | null;
	event_id: string | null;
	event_type: string;
	created_at: string;
	payload: Record<string, unknown>;
}

export interface ResourceSample {
	id: number;
	run_id: string | null;
	cpu: number | null;
	ram: number | null;
	gpu: number | null;
	created_at: string;
}

export interface PerEraRow {
	[key: string]: string | number;
}

export interface ScoringSeriesRow {
	run_id?: string;
	config_hash?: string;
	seed?: number | null;
	target_col?: string | null;
	payout_target_col?: string | null;
	prediction_col?: string | null;
	era: string | number;
	metric_key: string;
	series_type: 'per_era' | 'cumulative';
	value: number | null;
}

export interface FoldSnapshotRow {
	[key: string]: string | number | null;
}

export interface ScoringDashboardMeta {
	target_col?: string | null;
	payout_target_col?: string | null;
	available_metric_keys: string[];
	source: 'canonical' | 'legacy_fallback';
	omissions: Record<string, unknown>;
}

export interface ScoringDashboard {
	series: ScoringSeriesRow[];
	fold_snapshots: FoldSnapshotRow[] | null;
	summary: Record<string, unknown> | null;
	feature_summary: Record<string, unknown> | null;
	meta: ScoringDashboardMeta;
}

export interface DiagnosticsSourceItem {
	path?: string | null;
	exists?: boolean | null;
	sha256?: string | null;
	size_bytes?: number | null;
}

export interface DiagnosticsSources {
	columns?: Record<string, unknown> | null;
	joins?: Record<string, unknown> | null;
	sources?: Record<string, DiagnosticsSourceItem> | null;
	score_provenance_path?: string | null;
}

export interface RunBundle {
	metrics?: Record<string, number> | null;
	manifest?: RunManifest | null;
	scoring_dashboard?: ScoringDashboard | null;
	trials?: Record<string, unknown>[] | null;
	best_params?: Record<string, unknown> | null;
	resolved_config?: { yaml: string } | null;
	events?: RunEvent[] | null;
	resources?: ResourceSample[] | null;
	diagnostics_sources?: DiagnosticsSources | null;
}

export type RunBundleSection =
	| 'manifest'
	| 'metrics'
	| 'scoring_dashboard'
	| 'events'
	| 'resources'
	| 'resolved_config'
	| 'trials'
	| 'best_params'
	| 'diagnostics_sources';

export interface ExperimentConfigSummary {
	run_id?: string | null;
	backend?: string | null;
	tier?: string | null;
	model_type?: string | null;
	target?: string | null;
	target_payout?: string | null;
	feature_set?: string | null;
	stages?: string[] | null;
}

export interface ExperimentConfig {
	config_id: string;
	relative_path: string;
	abs_path: string;
	sha256: string;
	mtime: string;
	summary: ExperimentConfigSummary;
	is_runnable?: boolean;
	runnable_reason?: string | null;
}

export interface DocResponse {
	content: string;
	exists: boolean;
}

export interface ConfigListResponse {
	experiment_id: string;
	items: ExperimentConfig[];
	total: number;
}

export interface GlobalConfig {
	config_id: string;
	relative_path: string;
	experiment_id: string;
	sha256: string;
	mtime: string;
	summary: ExperimentConfigSummary;
	linked_run_id: string | null;
	linked_metrics: {
		bmc_last_200_eras_mean?: number | null;
		bmc_mean?: number | null;
		corr_sharpe?: number | null;
		corr_mean?: number | null;
		mmc_mean?: number | null;
	} | null;
}

export interface GlobalConfigListResponse {
	items: GlobalConfig[];
	total: number;
}

export interface ConfigCompareItem {
	config_id: string;
	yaml: string;
}

export interface ConfigCompareResponse {
	configs: ConfigCompareItem[];
}

export interface RunJob {
	job_id: string;
	batch_id: string;
	experiment_id: string | null;
	logical_run_id?: string | null;
	operation_type?: string | null;
	attempt_no?: number | null;
	attempt_id?: string | null;
	config_id: string;
	config_source: string;
	config_path: string;
	config_sha256: string;
	request: Record<string, unknown> | null;
	job_type: string;
	status: string;
	queue_name: string;
	priority: number;
	created_at: string;
	queued_at: string;
	started_at: string | null;
	finished_at: string | null;
	updated_at: string;
	worker_id: string | null;
	pid: number | null;
	exit_code: number | null;
	signal: number | null;
	backend: string | null;
	tier: string | null;
	budget: number | null;
	timeout_seconds: number | null;
	canonical_run_id: string | null;
	external_run_id: string | null;
	run_dir: string | null;
	cancel_requested: number;
	error: Record<string, unknown> | null;
	queue_position: number | null;
}

export interface RunLifecycle {
	run_id: string;
	run_hash: string;
	config_hash: string;
	job_id: string;
	logical_run_id: string;
	attempt_id: string;
	attempt_no: number;
	source: string;
	operation_type: string;
	job_type: string;
	status: 'queued' | 'starting' | 'running' | 'completed' | 'failed' | 'canceled' | 'stale' | string;
	experiment_id: string | null;
	config_id: string;
	config_source: string;
	config_path: string;
	config_sha256: string;
	run_dir: string;
	runtime_path: string;
	backend: string | null;
	worker_id: string | null;
	pid: number | null;
	host: string | null;
	current_stage: string | null;
	completed_stages: string[];
	progress_percent: number | null;
	progress_label: string | null;
	progress_current: number | null;
	progress_total: number | null;
	cancel_requested: boolean;
	cancel_requested_at: string | null;
	created_at: string;
	queued_at: string | null;
	started_at: string | null;
	last_heartbeat_at: string | null;
	updated_at: string;
	finished_at: string | null;
	terminal_reason: string | null;
	terminal_detail: Record<string, unknown>;
	latest_metrics: Record<string, unknown>;
	latest_sample: Record<string, unknown>;
	reconciled: boolean;
}

export interface RunJobListResponse {
	items: RunJob[];
	total: number;
}

export interface RunJobBatchResponse {
	batch_id: string;
	items: RunJob[];
	total: number;
}

export interface RunJobEvent {
	id: number;
	job_id: string;
	sequence: number;
	event_type: string;
	source: string;
	payload: Record<string, unknown>;
	created_at: string;
}

export interface RunJobLog {
	id: number;
	job_id: string;
	line_no: number;
	stream: 'stdout' | 'stderr' | string;
	line: string;
	created_at: string;
}

export interface RunJobSample {
	id: number;
	job_id: string;
	process_cpu_percent: number | null;
	process_rss_gb: number | null;
	host_cpu_percent: number | null;
	host_ram_available_gb: number | null;
	host_ram_used_gb: number | null;
	host_gpu_percent: number | null;
	host_gpu_mem_used_gb: number | null;
	scope:
		| 'launcher_process_tree'
		| 'launcher_wrapper_only'
		| 'launcher_host_only'
		| 'unavailable'
		| string;
	status: 'ok' | 'partial' | 'unavailable' | string;
	cpu_percent: number | null;
	rss_gb: number | null;
	ram_available_gb: number | null;
	gpu_percent: number | null;
	gpu_mem_gb: number | null;
	created_at: string;
}

export interface RunpodPod {
	pod_id: string;
	name: string;
	status: string;
	gpu_type: string;
	gpu_count: number;
	cost_per_hour: number;
	created_at: string | null;
	volume_id: string | null;
	datacenter_id: string | null;
	ssh_command: string | null;
}

export interface RunpodPodListResponse {
	items: RunpodPod[];
	total: number;
	error?: string | null;
}

// ── HPO Study types ────────────────────────────────────────────────────

export interface HpoStudy {
	study_id: string;
	experiment_id: string | null;
	name: string;
	mode: string | null;
	preset: string | null;
	n_trials: number | null;
	n_completed: number | null;
	status: string | null;
	best_trial_number: number | null;
	best_value: number | null;
	best_run_id: string | null;
	storage_path: string | null;
	config: Record<string, unknown> | null;
	created_at: string | null;
	updated_at: string | null;
}

export interface HpoStudyListResponse {
	items: HpoStudy[];
	total: number;
}

export interface HpoTrial {
	[key: string]: unknown;
}

// ── Ensemble types ─────────────────────────────────────────────────────

export interface EnsembleComponent {
	ensemble_id: string;
	run_id: string;
	weight: number | null;
	rank: number | null;
}

export interface Ensemble {
	ensemble_id: string;
	experiment_id: string | null;
	name: string;
	method: string | null;
	status: string | null;
	artifacts_path: string | null;
	config: Record<string, unknown> | null;
	created_at: string | null;
	components?: EnsembleComponent[];
	metrics?: Record<string, number | null>;
}

export interface EnsembleListResponse {
	items: Ensemble[];
	total: number;
}

export interface CorrelationMatrix {
	labels: string[];
	matrix: (number | null)[][];
}

export type EnsembleArtifactRow = Record<string, string | number | boolean | null>;

export interface EnsembleArtifacts {
	weights: EnsembleArtifactRow[] | null;
	component_metrics: EnsembleArtifactRow[] | null;
	era_metrics: EnsembleArtifactRow[] | null;
	regime_metrics: EnsembleArtifactRow[] | null;
	lineage: Record<string, unknown> | null;
	bootstrap_metrics: Record<string, unknown> | null;
	heavy_component_predictions_available: boolean;
	available_files: string[];
}

// ── Numerai docs types ─────────────────────────────────────────────

export interface NumeraiDocNode {
	title: string;
	path: string | null;
	children?: NumeraiDocNode[];
}

export interface NumeraiDocTree {
	sections: Array<{ heading: string; items: NumeraiDocNode[] }>;
}

export interface SystemCapabilities {
	read_only: boolean;
	write_controls: boolean;
}

const bundleCache = new Map<string, RunBundle>();
const immutableRunSectionCache = new Map<string, unknown>();

const RUN_BUNDLE_SECTIONS: RunBundleSection[] = [
	'manifest',
	'metrics',
	'scoring_dashboard',
	'events',
	'resources',
	'resolved_config',
	'trials',
	'best_params',
	'diagnostics_sources'
];

function cachedKey(section: string, runId: string, source?: SourceContext): string {
	const normalized = normalizedSource(source);
	return `${section}:${normalized.source_kind}:${normalized.source_id}:${runId}`;
}

function cachedGet<T>(
	section: string,
	runId: string,
	path: string,
	fetchFn: FetchFn,
	signal?: AbortSignal,
	source?: SourceContext
): Promise<T> {
	const key = cachedKey(section, runId, source);
	const cached = immutableRunSectionCache.get(key) as T | undefined;
	if (cached !== undefined) {
		return Promise.resolve(cached);
	}
	return get<T>(path, fetchFn, signal).then((data) => {
		immutableRunSectionCache.set(key, data);
		return data;
	});
}

export function createApi(fetchFn: FetchFn = globalThis.fetch) {
	const query = (params: Record<string, string | number | boolean | undefined | null>) => {
		const values = Object.entries(params).filter(([, value]) => value !== undefined && value !== null);
		if (values.length === 0) return '';
		return `?${new URLSearchParams(values.map(([key, value]) => [key, String(value)])).toString()}`;
	};

	return {
		listConfigs: (params?: {
			q?: string;
			experiment_id?: string;
			model_type?: string;
			target?: string;
			limit?: number;
			offset?: number;
		}) =>
			get<GlobalConfigListResponse>(
				`/configs${query({
					q: params?.q,
					experiment_id: params?.experiment_id,
					model_type: params?.model_type,
					target: params?.target,
					limit: params?.limit,
					offset: params?.offset
				})}`,
				fetchFn
			),
		compareConfigs: (configIds: string[]) =>
			get<ConfigCompareResponse>(
				`/configs/compare${query({ config_ids: configIds.join(',') })}`,
				fetchFn
			),
		listExperiments: () => get<Experiment[]>('/experiments', fetchFn),
		getExperimentsOverview: (params?: { include_remote?: boolean }) =>
			get<ExperimentOverviewResponse>(
				`/experiments/overview${query({ include_remote: params?.include_remote })}`,
				fetchFn
			),
		getExperiment: (id: string, source?: SourceContext) =>
			get<Experiment>(`/experiments/${id}${query(sourceQueryParams(source))}`, fetchFn),
		getExperimentConfigs: (
			id: string,
			params?: { q?: string; limit?: number; offset?: number } & SourceContext
		) =>
			get<ConfigListResponse>(
				`/experiments/${id}/configs${query({
					q: params?.q,
					limit: params?.limit,
					offset: params?.offset,
					...sourceQueryParams(params)
				})}`,
				fetchFn
			),
		getExperimentRuns: (id: string, source?: SourceContext) =>
			get<ExperimentRun[]>(`/experiments/${id}/runs${query(sourceQueryParams(source))}`, fetchFn),
		getExperimentRoundResults: (id: string, source?: SourceContext) =>
			get<ExperimentRoundResult[]>(
				`/experiments/${id}/round-results${query(sourceQueryParams(source))}`,
				fetchFn
			),
		getRunManifest: (runId: string, signal?: AbortSignal, source?: SourceContext) =>
			cachedGet<RunManifest>(
				'manifest',
				runId,
				`/runs/${runId}/manifest${query(sourceQueryParams(source))}`,
				fetchFn,
				signal,
				source
			),
		getRunMetrics: (runId: string, signal?: AbortSignal, source?: SourceContext) =>
			cachedGet<Record<string, number>>(
				'metrics',
				runId,
				`/runs/${runId}/metrics${query(sourceQueryParams(source))}`,
				fetchFn,
				signal,
				source
			),
		getRunScoringDashboard: (runId: string, signal?: AbortSignal, source?: SourceContext) =>
			cachedGet<ScoringDashboard>(
				'scoring-dashboard',
				runId,
				`/runs/${runId}/scoring-dashboard${query(sourceQueryParams(source))}`,
				fetchFn,
				signal,
				source
			),
		getTrials: (runId: string, signal?: AbortSignal, source?: SourceContext) =>
			cachedGet<Record<string, unknown>[]>(
				'trials',
				runId,
				`/runs/${runId}/trials${query(sourceQueryParams(source))}`,
				fetchFn,
				signal,
				source
			),
		getBestParams: (runId: string, signal?: AbortSignal, source?: SourceContext) =>
			cachedGet<Record<string, unknown>>(
				'best-params',
				runId,
				`/runs/${runId}/best-params${query(sourceQueryParams(source))}`,
				fetchFn,
				signal,
				source
			),
		getResolvedConfig: (runId: string, signal?: AbortSignal, source?: SourceContext) =>
			cachedGet<{ yaml: string }>(
				'config',
				runId,
				`/runs/${runId}/config${query(sourceQueryParams(source))}`,
				fetchFn,
				signal,
				source
			),
		getRunDiagnosticsSources: (runId: string, signal?: AbortSignal, source?: SourceContext) =>
			cachedGet<DiagnosticsSources>(
				'diagnostics-sources',
				runId,
				`/runs/${runId}/diagnostics-sources${query(sourceQueryParams(source))}`,
				fetchFn,
				signal,
				source
			),
		getRunEvents: (runId: string, limit = 50, signal?: AbortSignal, source?: SourceContext) =>
			get<RunEvent[]>(
				`/runs/${runId}/events${query({ limit, ...sourceQueryParams(source) })}`,
				fetchFn,
				signal
			),
		getRunResources: (runId: string, limit = 50, signal?: AbortSignal, source?: SourceContext) =>
			get<ResourceSample[]>(
				`/runs/${runId}/resources${query({ limit, ...sourceQueryParams(source) })}`,
				fetchFn,
				signal
			),
		listRunJobs: (params?: {
			experiment_id?: string;
			status?: string;
			limit?: number;
			offset?: number;
			include_attempts?: boolean;
		} & SourceContext) =>
			get<RunJobListResponse>(
				`/run-jobs${query({
					experiment_id: params?.experiment_id,
					status: params?.status,
					limit: params?.limit,
					offset: params?.offset,
					include_attempts: params?.include_attempts,
					...sourceQueryParams(params)
				})}`,
				fetchFn
			),
		getRunLifecycle: (runId: string, source?: SourceContext) =>
			get<RunLifecycle>(`/runs/${runId}/lifecycle${query(sourceQueryParams(source))}`, fetchFn),
		listRunpodPods: () => get<RunpodPodListResponse>('/runpod/pods', fetchFn),
		getRunJob: (jobId: string, source?: SourceContext) =>
			get<RunJob>(`/run-jobs/${jobId}${query(sourceQueryParams(source))}`, fetchFn),
		getRunJobBatch: (batchId: string) => get<RunJobBatchResponse>(`/run-jobs/batches/${batchId}`, fetchFn),
		getRunJobEvents: (jobId: string, params?: { after_id?: number; limit?: number } & SourceContext) =>
			get<RunJobEvent[]>(
				`/run-jobs/${jobId}/events${query({
					after_id: params?.after_id,
					limit: params?.limit,
					...sourceQueryParams(params)
				})}`,
				fetchFn
			),
		getRunJobLogs: (
			jobId: string,
			params?: { after_id?: number; limit?: number; stream?: 'all' | 'stdout' | 'stderr' } & SourceContext
		) =>
			get<RunJobLog[]>(
				`/run-jobs/${jobId}/logs${query({
					after_id: params?.after_id,
					limit: params?.limit,
					stream: params?.stream,
					...sourceQueryParams(params)
				})}`,
				fetchFn
			),
		getRunJobSamples: (jobId: string, params?: { after_id?: number; limit?: number } & SourceContext) =>
			get<RunJobSample[]>(
				`/run-jobs/${jobId}/samples${query({
					after_id: params?.after_id,
					limit: params?.limit,
					...sourceQueryParams(params)
				})}`,
				fetchFn
			),
		runJobStreamUrl: (
			jobId: string,
			params?: { after_event_id?: number; after_log_id?: number; after_sample_id?: number } & SourceContext
		) =>
			`${BASE}/run-jobs/${jobId}/stream${query({
				after_event_id: params?.after_event_id,
				after_log_id: params?.after_log_id,
				after_sample_id: params?.after_sample_id,
				...sourceQueryParams(params)
			})}`,
		getRunBundle: (
			runId: string,
			params?: { sections?: RunBundleSection[]; signal?: AbortSignal } & SourceContext
		): Promise<RunBundle> => {
			const key = cachedKey('bundle', runId, params);
			const cached = bundleCache.get(key);
			const requestedSections = params?.sections?.length
				? Array.from(new Set(params.sections))
				: RUN_BUNDLE_SECTIONS;
			if (cached && requestedSections.every((section) => section in cached)) {
				return Promise.resolve(cached);
			}
			return get<RunBundle>(
				`/runs/${runId}/bundle${query({
					sections: params?.sections?.length ? requestedSections.join(',') : undefined,
					...sourceQueryParams(params)
				})}`,
				fetchFn,
				params?.signal
			).then((data) => {
				const next = { ...(cached ?? {}), ...data };
				bundleCache.set(key, next);
				return next;
			});
		},
		getSystemCapabilities: () => get<SystemCapabilities>('/system/capabilities', fetchFn),
		getExperimentDoc: (id: string, filename: string, source?: SourceContext) =>
			get<DocResponse>(
				`/experiments/${id}/docs/${filename}${query(sourceQueryParams(source))}`,
				fetchFn
			),
		getRunDoc: (runId: string, filename: string, source?: SourceContext) =>
			get<DocResponse>(`/runs/${runId}/docs/${filename}${query(sourceQueryParams(source))}`, fetchFn),

		// ── HPO Studies ────────────────────────────────────────────────
		listStudies: (params?: { experiment_id?: string; status?: string; limit?: number; offset?: number }) =>
			get<HpoStudyListResponse>(
				`/studies${query({ experiment_id: params?.experiment_id, status: params?.status, limit: params?.limit, offset: params?.offset })}`,
				fetchFn
			),
		getStudy: (studyId: string, source?: SourceContext) =>
			get<HpoStudy>(`/studies/${studyId}${query(sourceQueryParams(source))}`, fetchFn),
		getStudyTrials: (studyId: string, source?: SourceContext) =>
			get<HpoTrial[]>(`/studies/${studyId}/trials${query(sourceQueryParams(source))}`, fetchFn),
		getExperimentStudies: (experimentId: string, source?: SourceContext) =>
			get<HpoStudy[]>(
				`/experiments/${experimentId}/studies${query(sourceQueryParams(source))}`,
				fetchFn
			),

		// ── Ensembles ──────────────────────────────────────────────────
		listEnsembles: (params?: { experiment_id?: string; limit?: number; offset?: number }) =>
			get<EnsembleListResponse>(
				`/ensembles${query({ experiment_id: params?.experiment_id, limit: params?.limit, offset: params?.offset })}`,
				fetchFn
			),
		getEnsemble: (ensembleId: string, source?: SourceContext) =>
			get<Ensemble>(`/ensembles/${ensembleId}${query(sourceQueryParams(source))}`, fetchFn),
		getEnsembleCorrelations: (ensembleId: string, source?: SourceContext) =>
			get<CorrelationMatrix>(
				`/ensembles/${ensembleId}/correlations${query(sourceQueryParams(source))}`,
				fetchFn
			),
		getEnsembleArtifacts: (ensembleId: string, source?: SourceContext) =>
			get<EnsembleArtifacts>(
				`/ensembles/${ensembleId}/artifacts${query(sourceQueryParams(source))}`,
				fetchFn
			),
		getExperimentEnsembles: (experimentId: string, source?: SourceContext) =>
			get<Ensemble[]>(
				`/experiments/${experimentId}/ensembles${query(sourceQueryParams(source))}`,
				fetchFn
			),

		// ── Numerai docs ──────────────────────────────────────────────
		getNumeraiDocTree: () => get<NumeraiDocTree>('/docs/numerai/tree', fetchFn),
		getNumeraiDocContent: (path: string) =>
			get<DocResponse>(`/docs/numerai/content?path=${encodeURIComponent(path)}`, fetchFn),

		// ── Numereng docs ─────────────────────────────────────────────
		getNumerengDocTree: () => get<NumeraiDocTree>('/docs/numereng/tree', fetchFn),
		getNumerengDocContent: (path: string) =>
			get<DocResponse>(`/docs/numereng/content?path=${encodeURIComponent(path)}`, fetchFn),

		// ── Notes ─────────────────────────────────────────────────────
		getNotesTree: () => get<NumeraiDocTree>('/notes/tree', fetchFn),
		getNotesContent: (path: string) =>
			get<DocResponse>(`/notes/content?path=${encodeURIComponent(path)}`, fetchFn)
	};
}

/** Default client for use in components (browser-side) */
export const api = createApi();
