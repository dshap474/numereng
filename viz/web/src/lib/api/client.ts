const BASE = '/api';

type FetchFn = typeof globalThis.fetch;

async function get<T>(path: string, fetchFn: FetchFn = globalThis.fetch): Promise<T> {
	const res = await fetchFn(`${BASE}${path}`);
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

export interface ExperimentRun {
	run_id: string;
	experiment_id: string;
	status: string;
	is_champion: boolean;
	metrics: Record<string, number>;
	created_at: string;
	run_name?: string | null;
	model_type?: string | null;
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

export interface FeatureImportanceRow {
	[key: string]: string | number;
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
	metrics: Record<string, number> | null;
	manifest: RunManifest | null;
	per_era_corr: PerEraRow[] | null;
	feature_importance: FeatureImportanceRow[] | null;
	trials: Record<string, unknown>[] | null;
	best_params: Record<string, unknown> | null;
	resolved_config: { yaml: string } | null;
	events: RunEvent[] | null;
	resources: ResourceSample[] | null;
	diagnostics_sources: DiagnosticsSources | null;
}

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
		getExperiment: (id: string) => get<Experiment>(`/experiments/${id}`, fetchFn),
		getExperimentConfigs: (
			id: string,
			params?: { q?: string; limit?: number; offset?: number }
		) =>
			get<ConfigListResponse>(
				`/experiments/${id}/configs${query({
					q: params?.q,
					limit: params?.limit,
					offset: params?.offset
				})}`,
				fetchFn
			),
		getExperimentRuns: (id: string) => get<ExperimentRun[]>(`/experiments/${id}/runs`, fetchFn),
		getExperimentRoundResults: (id: string) =>
			get<ExperimentRoundResult[]>(`/experiments/${id}/round-results`, fetchFn),
		getRunManifest: (runId: string) => get<RunManifest>(`/runs/${runId}/manifest`, fetchFn),
		getRunMetrics: (runId: string) => get<Record<string, number>>(`/runs/${runId}/metrics`, fetchFn),
		getPerEraCorr: (runId: string) => get<PerEraRow[]>(`/runs/${runId}/per-era-corr`, fetchFn),
		getFeatureImportance: (runId: string, topN = 30) =>
			get<FeatureImportanceRow[]>(`/runs/${runId}/feature-importance?top_n=${topN}`, fetchFn),
		getTrials: (runId: string) => get<Record<string, unknown>[]>(`/runs/${runId}/trials`, fetchFn),
		getBestParams: (runId: string) =>
			get<Record<string, unknown>>(`/runs/${runId}/best-params`, fetchFn),
		getResolvedConfig: (runId: string) => get<{ yaml: string }>(`/runs/${runId}/config`, fetchFn),
		getRunEvents: (runId: string, limit = 50) =>
			get<RunEvent[]>(`/runs/${runId}/events?limit=${limit}`, fetchFn),
		getRunResources: (runId: string, limit = 50) =>
			get<ResourceSample[]>(`/runs/${runId}/resources?limit=${limit}`, fetchFn),
		listRunJobs: (params?: {
			experiment_id?: string;
			status?: string;
			limit?: number;
			offset?: number;
			include_attempts?: boolean;
		}) =>
			get<RunJobListResponse>(
				`/run-jobs${query({
					experiment_id: params?.experiment_id,
					status: params?.status,
					limit: params?.limit,
					offset: params?.offset,
					include_attempts: params?.include_attempts
				})}`,
				fetchFn
			),
		listRunpodPods: () => get<RunpodPodListResponse>('/runpod/pods', fetchFn),
		getRunJob: (jobId: string) => get<RunJob>(`/run-jobs/${jobId}`, fetchFn),
		getRunJobBatch: (batchId: string) => get<RunJobBatchResponse>(`/run-jobs/batches/${batchId}`, fetchFn),
		getRunJobEvents: (jobId: string, params?: { after_id?: number; limit?: number }) =>
			get<RunJobEvent[]>(
				`/run-jobs/${jobId}/events${query({ after_id: params?.after_id, limit: params?.limit })}`,
				fetchFn
			),
		getRunJobLogs: (
			jobId: string,
			params?: { after_id?: number; limit?: number; stream?: 'all' | 'stdout' | 'stderr' }
		) =>
			get<RunJobLog[]>(
				`/run-jobs/${jobId}/logs${query({
					after_id: params?.after_id,
					limit: params?.limit,
					stream: params?.stream
				})}`,
				fetchFn
			),
		getRunJobSamples: (jobId: string, params?: { after_id?: number; limit?: number }) =>
			get<RunJobSample[]>(
				`/run-jobs/${jobId}/samples${query({ after_id: params?.after_id, limit: params?.limit })}`,
				fetchFn
			),
		runJobStreamUrl: (
			jobId: string,
			params?: { after_event_id?: number; after_log_id?: number; after_sample_id?: number }
		) =>
			`${BASE}/run-jobs/${jobId}/stream${query({
				after_event_id: params?.after_event_id,
				after_log_id: params?.after_log_id,
				after_sample_id: params?.after_sample_id
			})}`,
		getRunBundle: (runId: string): Promise<RunBundle> => {
			const cached = bundleCache.get(runId);
			if (cached) return Promise.resolve(cached);
			return get<RunBundle>(`/runs/${runId}/bundle`, fetchFn).then((data) => {
				bundleCache.set(runId, data);
				return data;
			});
		},
		getSystemCapabilities: () => get<SystemCapabilities>('/system/capabilities', fetchFn),
		getExperimentDoc: (id: string, filename: string) =>
			get<DocResponse>(`/experiments/${id}/docs/${filename}`, fetchFn),
		getRunDoc: (runId: string, filename: string) =>
			get<DocResponse>(`/runs/${runId}/docs/${filename}`, fetchFn),

		// ── HPO Studies ────────────────────────────────────────────────
		listStudies: (params?: { experiment_id?: string; status?: string; limit?: number; offset?: number }) =>
			get<HpoStudyListResponse>(
				`/studies${query({ experiment_id: params?.experiment_id, status: params?.status, limit: params?.limit, offset: params?.offset })}`,
				fetchFn
			),
		getStudy: (studyId: string) => get<HpoStudy>(`/studies/${studyId}`, fetchFn),
		getStudyTrials: (studyId: string) => get<HpoTrial[]>(`/studies/${studyId}/trials`, fetchFn),
		getExperimentStudies: (experimentId: string) =>
			get<HpoStudy[]>(`/experiments/${experimentId}/studies`, fetchFn),

		// ── Ensembles ──────────────────────────────────────────────────
		listEnsembles: (params?: { experiment_id?: string; limit?: number; offset?: number }) =>
			get<EnsembleListResponse>(
				`/ensembles${query({ experiment_id: params?.experiment_id, limit: params?.limit, offset: params?.offset })}`,
				fetchFn
			),
		getEnsemble: (ensembleId: string) => get<Ensemble>(`/ensembles/${ensembleId}`, fetchFn),
		getEnsembleCorrelations: (ensembleId: string) =>
			get<CorrelationMatrix>(`/ensembles/${ensembleId}/correlations`, fetchFn),
		getEnsembleArtifacts: (ensembleId: string) =>
			get<EnsembleArtifacts>(`/ensembles/${ensembleId}/artifacts`, fetchFn),
		getExperimentEnsembles: (experimentId: string) =>
			get<Ensemble[]>(`/experiments/${experimentId}/ensembles`, fetchFn),

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
