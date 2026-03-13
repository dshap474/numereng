<script lang="ts">
	import {
		api,
		type ConfigListResponse,
		type Ensemble,
		type Experiment,
		type ExperimentConfig,
		type ExperimentRoundResult,
		type ExperimentRun,
		type HpoStudy,
		type RunJob,
		type RunJobEvent,
		type RunJobListResponse,
		type RunJobLog,
		type RunJobSample,
		type SystemCapabilities
	} from '$lib/api/client';
	import {
		DASHBOARD_KEY_METRICS,
		RUNOPS_ALL_SCORING_METRICS,
		RUNOPS_MAIN_METRICS,
		metricNumber,
		targetLabel
	} from '$lib/metrics/canonical';
	import {
		hostGpu,
		hostRamAvailableGb,
		latestSample as latestMonitorSample,
		processCpu,
		processRssGb,
		sampleAverages as monitorSampleAverages,
		sampleStatusMessage,
		scopeLabel
	} from '$lib/monitor/samples';
	import { fmt, fmtGb, fmtPercent } from '$lib/utils';
	import SharpeBarChart from '$lib/components/charts/SharpeBarChart.svelte';
	import ParetoChart from '$lib/components/charts/ParetoChart.svelte';
	import MarkdownDoc from '$lib/components/ui/MarkdownDoc.svelte';
	import RunDetailPanel from '$lib/components/ui/RunDetailPanel.svelte';
	import HpoDetailPanel from '$lib/components/ui/HpoDetailPanel.svelte';
	import EnsembleDetailPanel from '$lib/components/ui/EnsembleDetailPanel.svelte';

	let {
		data
	}: {
			data: {
				experiment: Experiment;
				runs: ExperimentRun[];
				roundResults: ExperimentRoundResult[];
				configs: ConfigListResponse;
				runJobs: RunJobListResponse;
				studies: HpoStudy[];
				ensembles: Ensemble[];
				capabilities: SystemCapabilities;
			};
		} = $props();

	const ACTIVE_JOB_STATUSES = new Set(['queued', 'starting', 'running', 'canceling']);
	const TERMINAL_JOB_STATUSES = new Set(['completed', 'failed', 'canceled', 'stale']);

	let sortColumn = $state('bmc_last_200_eras_mean');
	let sortDirection = $state<'asc' | 'desc'>('desc');
	let chartTab = $state<'composite' | 'charts'>('charts');
	let pageTab = $state<'analysis' | 'progress' | 'runops'>('analysis');
	let runOpsView = $state<'table' | 'chart'>('table');
	let launchSectionOpen = $state(true);
	let queueSectionOpen = $state(true);
	let monitorSectionOpen = $state(true);
	let progressQuery = $state('');
	const SHOW_RUNOPS_SIDEBAR = false;

	type ParetoColorMode = 'default' | 'model' | 'target';
	const PARETO_COLOR_MODES: ParetoColorMode[] = ['default', 'model', 'target'];
	const PARETO_COLOR_MODE_LABELS: Record<ParetoColorMode, string> = {
		default: 'Default',
		model: 'Op',
		target: 'Target'
	};
	let paretoColorMode = $state<ParetoColorMode>('default');

	let configItems = $state<ExperimentConfig[]>([]);
	let configTotal = $state<number>(0);
	let configQuery = $state('');
	let configsError = $state<string | null>(null);

	let runJobs = $state<RunJob[]>([]);
	let runJobsTotal = $state<number>(0);
	let jobsBusy = $state(false);
	let jobsError = $state<string | null>(null);

	let selectedJobId = $state<string | null>(null);
	let selectedJob = $state<RunJob | null>(null);

	type OpType = 'run' | 'hpo' | 'ensemble';
	let selectedOp = $state<{ id: string; type: OpType } | null>(null);
	let readOnly = $derived(Boolean(data.capabilities?.read_only));

	let monitorTab = $state<'events' | 'logs' | 'resources'>('events');
	let monitorLoading = $state(false);
	let monitorError = $state<string | null>(null);
	let streamState = $state<'idle' | 'connecting' | 'open' | 'reconnecting' | 'closed' | 'error'>('idle');
	let streamError = $state<string | null>(null);

	let jobEvents = $state<RunJobEvent[]>([]);
	let jobLogs = $state<RunJobLog[]>([]);
	let jobSamples = $state<RunJobSample[]>([]);
	let eventCursor = $state(0);
	let logCursor = $state(0);
	let sampleCursor = $state(0);

	let currentStream: EventSource | null = null;
	let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
	let monitorGeneration = 0;
	let configsRefreshGeneration = 0;
	let jobsRefreshGeneration = 0;

	let runOpsAvailableMetricKeys = $derived.by(() => {
		const keys = new Set<string>();
		for (const run of data.runs) {
			for (const key of Object.keys(run.metrics)) {
				keys.add(key);
			}
		}
		return keys;
	});

	let runOpsMainMetricColumns = $derived.by(() => {
		return RUNOPS_MAIN_METRICS.filter((key) => runOpsAvailableMetricKeys.has(key));
	});

	let metricColumns = $derived.by(() => {
		const ordered: string[] = [];
		const seen = new Set<string>();
		for (const key of [...RUNOPS_MAIN_METRICS, ...RUNOPS_ALL_SCORING_METRICS]) {
			if (!runOpsAvailableMetricKeys.has(key) || seen.has(key)) continue;
			ordered.push(key);
			seen.add(key);
		}
		return ordered;
	});

	let runOpsMetricDividerColumn = $derived.by(() => {
		if (runOpsMainMetricColumns.length === 0) return null;
		return runOpsMainMetricColumns[runOpsMainMetricColumns.length - 1];
	});

	interface Operation {
		op_id: string;
		op_type: OpType;
		name: string;
		model_type: string;
		target: string;
		feature_set: string;
		metrics: Record<string, number | null>;
		status: string | null;
	}

	let sortedOps = $derived.by(() => {
		const ops: Operation[] = [];
		for (const run of data.runs) {
			ops.push({
				op_id: run.run_id,
				op_type: 'run',
				name: run.run_name || run.run_id,
				model_type: run.model_type ?? '-',
				target: targetLabel(run),
				feature_set: run.feature_set ?? '-',
				metrics: run.metrics,
				status: run.status
			});
		}
		for (const study of data.studies) {
			ops.push({
				op_id: study.study_id,
				op_type: 'hpo',
				name: study.name || study.study_id,
				model_type: study.mode ?? '-',
				target: '-',
				feature_set: '-',
				metrics: study.best_value != null ? { best_value: study.best_value } : {},
				status: study.status
			});
		}
		for (const ens of data.ensembles) {
			ops.push({
				op_id: ens.ensemble_id,
				op_type: 'ensemble',
				name: ens.name || ens.ensemble_id,
				model_type: ens.method ?? '-',
				target: '-',
				feature_set: '-',
				metrics: ens.metrics ?? {},
				status: ens.status
			});
		}
		const dir = sortDirection === 'desc' ? 1 : -1;
		return ops.sort((a, b) => {
			const av = a.metrics[sortColumn] ?? -Infinity;
			const bv = b.metrics[sortColumn] ?? -Infinity;
			return (bv - av) * dir;
		});
	});

	$effect(() => {
		if (!metricColumns.some((column) => column === sortColumn)) {
			sortColumn = (metricColumns[0] ?? DASHBOARD_KEY_METRICS[0]) as string;
			sortDirection = 'desc';
		}
	});

	function opMetricValue(op: Operation, key: string): number | undefined {
		return metricNumber(op.metrics, key) ?? undefined;
	}

	let selectedOperationItem = $derived.by(() => {
		const current = selectedOp;
		if (!current) return null;
		return sortedOps.find((item) => item.op_id === current.id && item.op_type === current.type) ?? null;
	});

	let bmcRuns = $derived.by(() => {
		const runPoints = data.runs
			.filter((r) => metricNumber(r.metrics, 'bmc_last_200_eras_mean') != null)
			.map((r) => ({
				run_id: r.run_id,
				run_name: r.run_name ?? undefined,
				composite: metricNumber(r.metrics, 'bmc_last_200_eras_mean') ?? 0,
				is_champion: r.is_champion
			}));

		const roundPoints = data.roundResults
			.filter((r) => metricNumber(r.metrics, 'bmc_last_200_eras_mean') != null)
			.map((r) => ({
				run_id: r.run_id,
				run_name: r.name,
				composite: metricNumber(r.metrics, 'bmc_last_200_eras_mean') ?? 0,
				is_champion: false
			}));

		return [...runPoints, ...roundPoints];
	});

	let hpoBestRunIds = $derived(
		new Set(data.studies.map((s) => s.best_run_id).filter(Boolean) as string[])
	);

	let configNameByRunId = $derived.by(() => {
		const map = new Map<string, string>();
		const sourceItems = configItems.length > 0 ? configItems : (data.configs.items ?? []);
		for (const item of sourceItems) {
			const runId = item.summary.run_id ?? null;
			if (!runId) continue;
			const label = fileName(item.relative_path || item.config_id);
			if (!map.has(runId)) map.set(runId, label);
		}
		return map;
	});

	type ProgressState = 'not_started' | 'running' | 'finished' | 'failed';

	interface ProgressRow {
		config_id: string;
		relative_path: string;
		model_type: string;
		target: string;
		feature_set: string;
		status: ProgressState;
		run_id: string | null;
		job_id: string | null;
		job_status: string | null;
		finished_at: string | null;
		created_at: string | null;
	}

	let latestJobByConfigId = $derived.by(() => {
		const map = new Map<string, RunJob>();
		for (const job of runJobs) {
			if (!job.config_id) continue;
			const existing = map.get(job.config_id);
			if (!existing || (job.created_at ?? '') > (existing.created_at ?? '')) {
				map.set(job.config_id, job);
			}
		}
		return map;
	});

	let runById = $derived.by(() => {
		const map = new Map<string, ExperimentRun>();
		for (const run of data.runs) {
			map.set(run.run_id, run);
		}
		return map;
	});

	let progressRows = $derived.by(() => {
		const sourceItems = configItems.length > 0 ? configItems : (data.configs.items ?? []);
		return sourceItems.map((item): ProgressRow => {
			const latestJob = latestJobByConfigId.get(item.config_id) ?? null;
			const linkedRunId = item.summary.run_id ?? latestJob?.canonical_run_id ?? null;
			const linkedRun = linkedRunId ? runById.get(linkedRunId) ?? null : null;
			let status: ProgressState = 'not_started';
			if (latestJob && ACTIVE_JOB_STATUSES.has(latestJob.status)) {
				status = 'running';
			} else if (linkedRunId || linkedRun?.status === 'FINISHED') {
				status = 'finished';
			} else if (latestJob && TERMINAL_JOB_STATUSES.has(latestJob.status)) {
				status = latestJob.status === 'completed' ? 'finished' : 'failed';
			}
			return {
				config_id: item.config_id,
				relative_path: item.relative_path,
				model_type: item.summary.model_type ?? '-',
				target: item.summary.target ?? '-',
				feature_set: item.summary.feature_set ?? '-',
				status,
				run_id: linkedRunId,
				job_id: latestJob?.job_id ?? null,
				job_status: latestJob?.status ?? null,
				finished_at: latestJob?.finished_at ?? linkedRun?.created_at ?? null,
				created_at: latestJob?.created_at ?? linkedRun?.created_at ?? null
			};
		});
	});

	let progressCounts = $derived.by(() => {
		const counts = {
			total: progressRows.length,
			not_started: 0,
			running: 0,
			finished: 0,
			failed: 0
		};
		for (const row of progressRows) {
			counts[row.status] += 1;
		}
		return counts;
	});

	let nextPendingRow = $derived.by(
		() => progressRows.find((row) => row.status === 'not_started') ?? null
	);

	let nextAttentionRow = $derived.by(
		() => progressRows.find((row) => row.status === 'failed') ?? null
	);

	let filteredProgressRows = $derived.by(() => {
		const query = progressQuery.trim().toLowerCase();
		if (!query) return progressRows;
		return progressRows.filter((row) =>
			[
				row.relative_path,
				row.model_type,
				row.target,
				row.feature_set,
				row.status,
				row.run_id ?? '',
				row.job_id ?? ''
			]
				.join(' ')
				.toLowerCase()
				.includes(query)
		);
	});

	let paretoRuns = $derived.by(() => {
		const runPoints = data.runs
			.filter(
				(r) =>
					metricNumber(r.metrics, 'corr_mean') != null &&
					metricNumber(r.metrics, 'mmc_mean') != null
			)
				.map((r) => ({
					run_id: r.run_id,
					config_name:
						configNameByRunId.get(r.run_id) ??
						r.run_name ??
						`${r.model_type ?? 'unknown'} · ${targetLabel(r)}`,
					corr_mean: metricNumber(r.metrics, 'corr_mean') ?? 0,
					mmc_mean: metricNumber(r.metrics, 'mmc_mean') ?? 0,
					model_type: r.model_type ?? 'unknown',
				target: targetLabel(r),
				entity_type: hpoBestRunIds.has(r.run_id) ? ('hpo_best' as const) : ('run' as const),
				mmc_coverage_ratio_rows: metricNumber(r.metrics, 'mmc_coverage_ratio_rows'),
				bmc_mean: metricNumber(r.metrics, 'bmc_mean'),
				bmc_last_200_eras_mean: metricNumber(r.metrics, 'bmc_last_200_eras_mean'),
				corr_sharpe: metricNumber(r.metrics, 'corr_sharpe'),
				max_drawdown: metricNumber(r.metrics, 'max_drawdown')
			}));
		const roundPoints = data.roundResults
			.filter(
				(r) =>
					metricNumber(r.metrics, 'corr_mean') != null &&
					metricNumber(r.metrics, 'mmc_mean') != null
			)
				.map((r) => ({
					run_id: r.run_id,
					config_name:
						configNameByRunId.get(r.run_id) ??
						r.name ??
						`${r.model_type ?? 'derived'} · ${r.target ?? '-'}`,
					corr_mean: metricNumber(r.metrics, 'corr_mean') ?? 0,
					mmc_mean: metricNumber(r.metrics, 'mmc_mean') ?? 0,
					model_type: r.model_type ?? 'derived',
				target: r.target ?? '-',
				entity_type: 'run' as const,
				mmc_coverage_ratio_rows: metricNumber(r.metrics, 'mmc_coverage_ratio_rows'),
				bmc_mean: metricNumber(r.metrics, 'bmc_mean'),
				bmc_last_200_eras_mean: metricNumber(r.metrics, 'bmc_last_200_eras_mean'),
				corr_sharpe: metricNumber(r.metrics, 'corr_sharpe'),
				max_drawdown: metricNumber(r.metrics, 'max_drawdown')
			}));
		const ensemblePoints = data.ensembles
			.filter(
				(e) =>
					metricNumber(e.metrics, 'corr_mean') != null &&
					metricNumber(e.metrics, 'mmc_mean') != null
			)
				.map((e) => ({
					run_id: e.ensemble_id,
					config_name: e.name ?? 'ensemble',
					corr_mean: metricNumber(e.metrics, 'corr_mean') ?? 0,
					mmc_mean: metricNumber(e.metrics, 'mmc_mean') ?? 0,
				model_type: e.method ?? 'ensemble',
				target: 'ensemble',
				entity_type: 'ensemble' as const,
				mmc_coverage_ratio_rows: metricNumber(e.metrics, 'mmc_coverage_ratio_rows'),
				bmc_mean: metricNumber(e.metrics, 'bmc_mean'),
				bmc_last_200_eras_mean: metricNumber(e.metrics, 'bmc_last_200_eras_mean'),
				corr_sharpe: metricNumber(e.metrics, 'corr_sharpe'),
				max_drawdown: metricNumber(e.metrics, 'max_drawdown')
			}));
		return [...runPoints, ...roundPoints, ...ensemblePoints];
	});

	let filteredConfigs = $derived.by(() => {
		const query = configQuery.trim().toLowerCase();
		return configItems.filter((item) => {
			if (!query) return true;
			const haystack = [
				item.relative_path,
				item.summary.run_id ?? '',
				item.summary.model_type ?? '',
				item.summary.target ?? '',
				item.summary.target_payout ?? '',
				item.summary.feature_set ?? ''
			]
				.join(' ')
				.toLowerCase();
			return haystack.includes(query);
		});
	});

	let latestStageEvent = $derived.by(() => {
		for (let i = jobEvents.length - 1; i >= 0; i -= 1) {
			if (jobEvents[i].event_type === 'stage_update') return jobEvents[i];
		}
		return null;
	});

	let latestMetricEvent = $derived.by(() => {
		for (let i = jobEvents.length - 1; i >= 0; i -= 1) {
			if (jobEvents[i].event_type === 'metric_update') return jobEvents[i];
		}
		return null;
	});

	let latestSample = $derived.by(() => {
		return latestMonitorSample(jobSamples);
	});

	let sampleAverages = $derived.by(() => {
		return monitorSampleAverages(jobSamples, 30);
	});

	let telemetryMessage = $derived.by(() => sampleStatusMessage(latestSample));

	$effect(() => {
		const loadedConfigs = data.configs.items ?? [];
		const loadedJobs = data.runJobs.items ?? [];
		const sortedLoadedJobs = sortedJobs(loadedJobs);
		configItems = loadedConfigs;
		configTotal = data.configs.total ?? loadedConfigs.length;
		runJobs = sortedLoadedJobs;
		runJobsTotal = data.runJobs.total ?? loadedJobs.length;
		selectedJobId = pickInitialJob(sortedLoadedJobs);
		selectedJob = selectedJobId
			? sortedLoadedJobs.find((job) => job.job_id === selectedJobId) ?? null
			: null;
	});

	$effect(() => {
		if (pageTab !== 'runops') return;
		if (runOpsView !== 'chart') return;
		if (selectedOp != null) return;
		const first = sortedOps[0];
		if (!first) return;
		selectedOp = { id: first.op_id, type: first.op_type };
	});

	$effect(() => {
		const timer = setInterval(() => {
			if (streamState === 'open') return;
			void refreshRunJobs({ keepSelection: true });
		}, 4000);
		return () => clearInterval(timer);
	});

	$effect(() => {
		const jobId = selectedJobId;
		void activateMonitor(jobId);
		return () => {
			monitorGeneration += 1;
			closeMonitorStream();
		};
	});

	function jobSortKey(job: RunJob): number {
		if (job.status === 'running' || job.status === 'starting') return 0;
		if (job.status === 'canceling') return 1;
		if (job.status === 'queued') return 2;
		return 3; // terminal: completed, failed, canceled, stale
	}

	function sortedJobs(items: RunJob[]): RunJob[] {
		return [...items].sort((a, b) => jobSortKey(a) - jobSortKey(b));
	}

	let activeJobs = $derived(runJobs.filter((j) => !TERMINAL_JOB_STATUSES.has(j.status)));
	let terminalJobs = $derived(runJobs.filter((j) => TERMINAL_JOB_STATUSES.has(j.status)));

	function pickInitialJob(items: RunJob[]): string | null {
		const running = items.find((i) => i.status === 'running' || i.status === 'starting');
		if (running) return running.job_id;
		const queued = items.find((i) => i.status === 'queued');
		if (queued) return queued.job_id;
		return items.length > 0 ? items[0].job_id : null;
	}

	function toggleSort(col: string) {
		if (sortColumn === col) {
			sortDirection = sortDirection === 'desc' ? 'asc' : 'desc';
		} else {
			sortColumn = col;
			sortDirection = 'desc';
		}
	}

	function sortIndicator(col: string): string {
		if (sortColumn !== col) return '';
		return sortDirection === 'desc' ? ' ▼' : ' ▲';
	}

	function metricValue(run: ExperimentRun, key: string): number | undefined {
		return metricNumber(run.metrics, key) ?? undefined;
	}

	function runJobStatusClass(status: string): string {
		switch (status) {
			case 'queued':
				return 'bg-muted text-muted-foreground';
			case 'starting':
			case 'running':
				return 'bg-positive/15 text-positive';
			case 'canceling':
				return 'bg-amber-500/20 text-amber-300';
			case 'completed':
				return 'bg-blue-500/20 text-blue-300';
			case 'failed':
				return 'bg-negative/20 text-negative';
			case 'canceled':
			case 'stale':
				return 'bg-muted text-muted-foreground';
			default:
				return 'bg-muted text-muted-foreground';
		}
	}

	function progressStatusClass(status: ProgressState): string {
		switch (status) {
			case 'finished':
				return 'bg-positive/15 text-positive';
			case 'running':
				return 'bg-sky-500/15 text-sky-300';
			case 'failed':
				return 'bg-negative/15 text-negative';
			default:
				return 'bg-muted text-muted-foreground';
		}
	}

	function progressStatusLabel(status: ProgressState): string {
		switch (status) {
			case 'not_started':
				return 'not started';
			default:
				return status;
		}
	}

	function shortId(value: string | null | undefined, length = 8): string {
		if (!value) return '-';
		if (value.length <= length) return value;
		return value.slice(0, length);
	}

	function fileName(value: string | null | undefined): string {
		if (!value) return '-';
		const parts = value.split('/').filter(Boolean);
		return parts.length === 0 ? value : parts[parts.length - 1];
	}

	function fmtTime(value: string | null | undefined): string {
		if (!value) return '-';
		const date = new Date(value);
		if (Number.isNaN(date.getTime())) return value;
		return date.toLocaleTimeString();
	}

	function fmtDateTime(value: string | null | undefined): string {
		if (!value) return '-';
		const date = new Date(value);
		if (Number.isNaN(date.getTime())) return value;
		return date.toLocaleString();
	}

	async function refreshConfigs() {
		const generation = ++configsRefreshGeneration;
		configsError = null;
		try {
			const result = await api.getExperimentConfigs(data.experiment.experiment_id, {
				limit: 500,
				offset: 0
			});
			if (generation !== configsRefreshGeneration) return;
			configItems = result.items;
			configTotal = result.total;
		} catch (error) {
			if (generation !== configsRefreshGeneration) return;
			configsError = error instanceof Error ? error.message : 'Failed to refresh config catalog.';
		}
	}

	async function refreshRunJobs({ keepSelection = true }: { keepSelection: boolean }) {
		const generation = ++jobsRefreshGeneration;
		jobsBusy = true;
		jobsError = null;
		try {
			const result = await api.listRunJobs({
				experiment_id: data.experiment.experiment_id,
				limit: 200,
				offset: 0
			});
			if (generation !== jobsRefreshGeneration) return;
			runJobs = sortedJobs(result.items);
			runJobsTotal = result.total;
			if (!keepSelection || !selectedJobId || !runJobs.some((item) => item.job_id === selectedJobId)) {
				selectedJobId = pickInitialJob(runJobs);
			}
				selectedJob = selectedJobId
					? runJobs.find((item) => item.job_id === selectedJobId) ?? selectedJob
					: null;
		} catch (error) {
			if (generation !== jobsRefreshGeneration) return;
			jobsError = error instanceof Error ? error.message : 'Failed to refresh jobs.';
		} finally {
			if (generation !== jobsRefreshGeneration) return;
			jobsBusy = false;
		}
	}

	function closeMonitorStream() {
		if (reconnectTimer) {
			clearTimeout(reconnectTimer);
			reconnectTimer = null;
		}
		if (currentStream) {
			currentStream.close();
			currentStream = null;
		}
	}

	function parseSseData<T>(value: MessageEvent): T | null {
		try {
			return JSON.parse(value.data) as T;
		} catch {
			return null;
		}
	}

	function appendEvent(event: RunJobEvent) {
		jobEvents = [...jobEvents, event].slice(-400);
		eventCursor = Math.max(eventCursor, event.id);
	}

	function appendLog(log: RunJobLog) {
		jobLogs = [...jobLogs, log].slice(-1200);
		logCursor = Math.max(logCursor, log.id);
	}

	function appendSample(sample: RunJobSample) {
		jobSamples = [...jobSamples, sample].slice(-600);
		sampleCursor = Math.max(sampleCursor, sample.id);
	}

	function scheduleReconnect(jobId: string, generation: number) {
		if (reconnectTimer) clearTimeout(reconnectTimer);
		reconnectTimer = setTimeout(() => {
			if (generation !== monitorGeneration || selectedJobId !== jobId) return;
			startMonitorStream(jobId, generation);
		}, 1500);
	}

	function startMonitorStream(jobId: string, generation: number) {
		closeMonitorStream();
		streamState = 'connecting';
		streamError = null;
		const url = api.runJobStreamUrl(jobId, {
			after_event_id: eventCursor,
			after_log_id: logCursor,
			after_sample_id: sampleCursor
		});
		const stream = new EventSource(url);
		currentStream = stream;

		stream.onopen = () => {
			if (generation !== monitorGeneration) return;
			streamState = 'open';
			streamError = null;
		};

		stream.addEventListener('job_event', (raw) => {
			if (generation !== monitorGeneration || selectedJobId !== jobId) return;
			const event = parseSseData<RunJobEvent>(raw as MessageEvent);
			if (!event) return;
			appendEvent(event);
			if (event.event_type.startsWith('job_')) {
				void refreshRunJobs({ keepSelection: true });
			}
			if (
				event.event_type === 'job_completed' ||
				event.event_type === 'job_failed' ||
				event.event_type === 'job_canceled' ||
				event.event_type === 'job_stale'
			) {
				streamState = 'closed';
				closeMonitorStream();
			}
		});

		stream.addEventListener('log_line', (raw) => {
			if (generation !== monitorGeneration || selectedJobId !== jobId) return;
			const item = parseSseData<RunJobLog>(raw as MessageEvent);
			if (!item) return;
			appendLog(item);
		});

		stream.addEventListener('resource_sample', (raw) => {
			if (generation !== monitorGeneration || selectedJobId !== jobId) return;
			const item = parseSseData<RunJobSample>(raw as MessageEvent);
			if (!item) return;
			appendSample(item);
		});

		stream.onerror = () => {
			if (generation !== monitorGeneration || selectedJobId !== jobId) return;
			if (currentStream === stream) {
				stream.close();
				currentStream = null;
			}
			if (streamState === 'closed') return;
			const latest = runJobs.find((item) => item.job_id === jobId) ?? selectedJob;
			if (latest && !TERMINAL_JOB_STATUSES.has(latest.status)) {
				streamState = 'reconnecting';
				streamError = 'Connection dropped. Reconnecting...';
				scheduleReconnect(jobId, generation);
			} else {
				streamState = 'closed';
			}
		};
	}

	async function activateMonitor(jobId: string | null) {
		monitorGeneration += 1;
		const generation = monitorGeneration;
		closeMonitorStream();
		monitorError = null;
		streamError = null;

		if (!jobId) {
			selectedJob = null;
			jobEvents = [];
			jobLogs = [];
			jobSamples = [];
			streamState = 'idle';
			return;
		}

		monitorLoading = true;
		try {
			const [job, events, logs, samples] = await Promise.all([
				api.getRunJob(jobId),
				api.getRunJobEvents(jobId, { limit: 400 }),
				api.getRunJobLogs(jobId, { limit: 1200, stream: 'all' }),
				api.getRunJobSamples(jobId, { limit: 600 })
			]);
			if (generation !== monitorGeneration || selectedJobId !== jobId) return;

			selectedJob = job;
			jobEvents = events;
			jobLogs = logs;
			jobSamples = samples;
			eventCursor = events.length > 0 ? events[events.length - 1].id : 0;
			logCursor = logs.length > 0 ? logs[logs.length - 1].id : 0;
			sampleCursor = samples.length > 0 ? samples[samples.length - 1].id : 0;

			if (!TERMINAL_JOB_STATUSES.has(job.status)) {
				startMonitorStream(jobId, generation);
			} else {
				streamState = 'closed';
			}
		} catch (error) {
			if (generation !== monitorGeneration) return;
			monitorError = error instanceof Error ? error.message : 'Failed to load monitor data.';
			streamState = 'error';
		} finally {
			if (generation === monitorGeneration) {
				monitorLoading = false;
			}
		}
	}

	function selectJob(jobId: string) {
		selectedJobId = jobId;
	}

	function selectOperation(op: Operation) {
		selectedOp = { id: op.op_id, type: op.op_type };
	}

	function opTypeLabel(opType: OpType): string {
		if (opType === 'hpo') return 'HPO';
		if (opType === 'ensemble') return 'Ens';
		return 'Run';
	}

	function opTypeBadgeClass(opType: OpType): string {
		if (opType === 'hpo') return 'bg-violet-500/20 text-violet-300';
		if (opType === 'ensemble') return 'bg-amber-500/20 text-amber-300';
		return 'bg-blue-500/20 text-blue-300';
	}

	function prettyEventPayload(event: RunJobEvent): string {
		const value = JSON.stringify(event.payload);
		if (value.length <= 120) return value;
		return `${value.slice(0, 120)}...`;
	}

	function jobTypeBadge(job: RunJob): string {
		const jt = job.job_type;
		if (jt === 'hpo') return 'HPO';
		if (jt === 'ensemble') return 'Ens';
		return '';
	}

</script>

{#snippet opsTable()}
	<div class="border border-border rounded-lg overflow-hidden flex-1 min-h-0 flex flex-col" aria-label="Operations table">
		<div class="overflow-auto flex-1 min-h-0">
			<table class="w-max min-w-full text-sm leading-[1.35] border-separate border-spacing-0">
				<thead>
					<tr class="h-[92px] text-left align-middle">
						<th
							scope="col"
							class="sticky left-0 top-0 z-30 w-[320px] min-w-[320px] border-b border-r border-border bg-background px-4 py-3 align-middle"
						>
							<div class="flex items-start justify-between gap-3">
								<div>
									<h2 class="text-sm font-semibold">Ops</h2>
									<p class="text-xs text-muted-foreground">{sortedOps.length} total</p>
								</div>
								<div class="inline-flex rounded-md border border-border/60 bg-background/25 p-0.5">
									{#each ['table', 'chart'] as view (view)}
										<button
											type="button"
											class="px-2.5 py-1 rounded text-[11px] capitalize transition-colors {runOpsView === view ? 'bg-muted/70 text-foreground' : 'text-muted-foreground hover:text-foreground'}"
											onclick={() => (runOpsView = view as 'table' | 'chart')}
										>{view}</button>
									{/each}
								</div>
							</div>
						</th>
						{#each metricColumns as col (col)}
							<th
								scope="col"
								class="sticky top-0 z-20 min-w-[128px] border-b border-border bg-background px-3 py-2.5 align-middle font-medium text-xs uppercase tracking-wider text-muted-foreground text-right cursor-pointer hover:text-foreground select-none {runOpsMetricDividerColumn === col ? 'border-r border-border/80' : ''}"
								onclick={() => toggleSort(col)}
							>{col}{sortIndicator(col)}</th>
						{/each}
					</tr>
				</thead>
				<tbody class="divide-y divide-border">
					{#each sortedOps as op (op.op_id)}
						<tr
							class="group h-[88px] cursor-pointer transition-colors"
							onclick={() => selectOperation(op)}
						>
							<td
								class="sticky left-0 z-10 w-[320px] min-w-[320px] border-b border-r border-border px-4 py-3 text-left {selectedOp?.id === op.op_id && selectedOp?.type === op.op_type ? 'bg-primary/10 shadow-[inset_2px_0_0_0_var(--color-primary)]' : 'bg-background group-hover:bg-muted/20'}"
							>
								<div class="flex items-start gap-2">
									<span class="mt-0.5 inline-flex rounded px-1.5 py-0.5 text-[9px] uppercase {opTypeBadgeClass(op.op_type)}">
										{opTypeLabel(op.op_type)}
									</span>
									<div class="min-w-0 flex-1">
										<div class="truncate text-[12px] font-medium">{op.name}</div>
										<div class="mt-1 text-[10px] text-muted-foreground">
											{op.model_type} · {op.target} · {op.feature_set}
										</div>
										<div class="mt-1 font-mono text-[10px] text-muted-foreground">{shortId(op.op_id, 12)}</div>
									</div>
								</div>
							</td>
							{#each metricColumns as col (col)}
								<td
									class="min-w-[128px] border-b border-border px-3 py-0 text-right tabular-nums align-middle {selectedOp?.id === op.op_id && selectedOp?.type === op.op_type ? 'bg-primary/10' : 'bg-transparent group-hover:bg-muted/20'} {runOpsMetricDividerColumn === col ? 'border-r border-border/80' : ''}"
								>{fmt(opMetricValue(op, col))}</td>
							{/each}
						</tr>
					{:else}
						<tr>
							<td
								colspan={metricColumns.length + 1}
								class="px-4 py-6 text-center text-sm text-muted-foreground"
							>
								No ops available.
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	</div>
{/snippet}

{#snippet opsRail()}
	<div class="w-[320px] flex-shrink-0 flex flex-col rounded-lg border border-border bg-card overflow-hidden">
		<div class="h-[92px] border-b border-border px-4 py-3">
			<div class="flex items-center justify-between gap-3">
				<div>
					<h2 class="text-sm font-semibold">Ops</h2>
					<p class="text-xs text-muted-foreground">{sortedOps.length} total</p>
				</div>
				<div class="inline-flex rounded-md border border-border/60 bg-background/25 p-0.5">
					{#each ['table', 'chart'] as view (view)}
						<button
							type="button"
							class="px-2.5 py-1 rounded text-[11px] capitalize transition-colors {runOpsView === view ? 'bg-muted/70 text-foreground' : 'text-muted-foreground hover:text-foreground'}"
							onclick={() => (runOpsView = view as 'table' | 'chart')}
						>{view}</button>
					{/each}
				</div>
			</div>
			{#if runOpsView === 'chart' && selectedOperationItem}
				<p class="mt-2 text-[11px] text-muted-foreground">
					Selected: {selectedOperationItem.name}
				</p>
			{/if}
		</div>

		<div class="flex-1 min-h-0 overflow-auto divide-y divide-border/50">
			{#each sortedOps as op (op.op_id)}
				<button
					type="button"
					class="h-[88px] w-full border-l-2 px-4 py-3 text-left transition-colors hover:bg-muted/20 {selectedOp?.id === op.op_id && selectedOp?.type === op.op_type ? 'border-l-primary bg-primary/10 shadow-[inset_0_0_0_1px_rgba(255,255,255,0.05)]' : 'border-l-transparent'}"
					onclick={() => selectOperation(op)}
				>
					<div class="flex items-start gap-2">
						<span class="mt-0.5 inline-flex rounded px-1.5 py-0.5 text-[9px] uppercase {opTypeBadgeClass(op.op_type)}">
							{opTypeLabel(op.op_type)}
						</span>
						<div class="min-w-0 flex-1">
							<div class="truncate text-[12px] font-medium">{op.name}</div>
							<div class="mt-1 text-[10px] text-muted-foreground">
								{op.model_type} · {op.target} · {op.feature_set}
							</div>
							<div class="mt-1 font-mono text-[10px] text-muted-foreground">{shortId(op.op_id, 12)}</div>
						</div>
					</div>
				</button>
			{:else}
				<div class="px-4 py-6 text-sm text-muted-foreground">No ops available.</div>
			{/each}
		</div>
	</div>
{/snippet}

<div class="-mx-8 -mt-14 md:-mt-8 -mb-8 flex flex-col h-screen">
	<div class="flex gap-0 border-b border-border flex-shrink-0 px-8">
		{#each [['analysis', 'Analysis'], ['progress', 'Progress'], ['runops', 'Run Ops']] as [key, label] (key)}
			<button
				type="button"
				class="px-4 py-3.5 text-sm font-medium transition-colors {pageTab === key
					? 'border-b-2 border-primary text-foreground'
					: 'text-muted-foreground hover:text-foreground'}"
				onclick={() => (pageTab = key as 'analysis' | 'progress' | 'runops')}
			>{label}</button>
		{/each}
	</div>

	{#if pageTab === 'analysis'}
		<div class="flex flex-1 min-h-0">
				<MarkdownDoc
					borderless
					label="Experiment"
					load={() => api.getExperimentDoc(data.experiment.experiment_id, 'EXPERIMENT.md')}
					readOnly={readOnly}
					readOnlyMessage=""
				/>

			<div class="flex-1 min-h-0 p-4" aria-label="Charts wrapper">
			<div class="bg-card border border-border rounded-lg h-full flex flex-col" aria-label="Charts section">
				<div class="flex items-center justify-between border-b border-border px-5 pt-3 pb-0 flex-shrink-0">
					<div class="flex gap-0">
						{#each [['charts', 'BMC Ranking'], ['composite', 'Payout Proxy']] as [key, label] (key)}
							<button
								type="button"
								class="px-3 py-2 text-xs font-medium transition-colors {chartTab === key
									? 'border-b-2 border-primary text-foreground'
									: 'text-muted-foreground hover:text-foreground'}"
								onclick={() => (chartTab = key as 'composite' | 'charts')}
							>{label}</button>
						{/each}
					</div>
					{#if chartTab === 'composite'}
						<div class="flex rounded-md border border-border overflow-hidden text-[10px]">
							{#each PARETO_COLOR_MODES as mode (mode)}
								<button
									type="button"
									class="px-2 py-0.5 transition-colors {paretoColorMode === mode
										? 'bg-primary text-primary-foreground'
										: 'bg-muted/40 text-muted-foreground hover:text-foreground'}"
									onclick={() => (paretoColorMode = mode)}
								>{PARETO_COLOR_MODE_LABELS[mode]}</button>
							{/each}
						</div>
					{/if}
				</div>

				<div class="flex-1 min-h-0 p-5">
					{#if chartTab === 'charts'}
						<div class="flex flex-col h-full min-h-0">
							<h3 class="mb-3 text-xs uppercase tracking-wider text-muted-foreground font-medium flex-shrink-0">
								BMC Last 200 by Run
							</h3>
							{#if bmcRuns.length > 0}
								<div class="flex-1 min-h-0 overflow-y-auto">
									<SharpeBarChart runs={bmcRuns} championRunId={data.experiment.champion_run_id} />
								</div>
							{:else}
								<div class="flex h-full items-center justify-center text-muted-foreground text-sm">
									No BMC last-200 data available.
								</div>
							{/if}
						</div>
					{:else}
						<div class="flex h-full min-h-0 flex-col">
							<div class="mb-3 flex-shrink-0">
								<p class="text-xs text-muted-foreground">
									Historical proxy from CORR and MMC. Research-only; not a live payout forecast.
								</p>
								<p class="mt-1 text-xs text-muted-foreground font-mono">
									Iso-lines: payout proxy = 0.75*CORR + 2.25*MMC
								</p>
							</div>
							{#if paretoRuns.length > 0}
								<ParetoChart
									runs={paretoRuns}
									colorMode={paretoColorMode}
									showIsoLines={true}
									class="h-full"
								/>
							{:else}
								<div class="flex h-full items-center justify-center text-muted-foreground text-sm">
									No CORR/MMC data available for payout proxy view.
								</div>
							{/if}
						</div>
					{/if}
				</div>
			</div>
			</div>
		</div>

	{:else if pageTab === 'progress'}
		<div class="flex-1 min-h-0 px-6 py-4 overflow-auto">
			<div class="space-y-4">
				<div class="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
					<div class="rounded-lg border border-border bg-card px-4 py-3">
						<div class="text-[11px] uppercase tracking-wide text-muted-foreground">Total Configs</div>
						<div class="mt-1 text-2xl font-semibold tabular-nums">{progressCounts.total}</div>
					</div>
					<div class="rounded-lg border border-border bg-card px-4 py-3">
						<div class="text-[11px] uppercase tracking-wide text-muted-foreground">Finished</div>
						<div class="mt-1 text-2xl font-semibold tabular-nums text-positive">{progressCounts.finished}</div>
					</div>
					<div class="rounded-lg border border-border bg-card px-4 py-3">
						<div class="text-[11px] uppercase tracking-wide text-muted-foreground">Running</div>
						<div class="mt-1 text-2xl font-semibold tabular-nums text-sky-300">{progressCounts.running}</div>
					</div>
					<div class="rounded-lg border border-border bg-card px-4 py-3">
						<div class="text-[11px] uppercase tracking-wide text-muted-foreground">Needs Attention</div>
						<div class="mt-1 text-2xl font-semibold tabular-nums text-negative">{progressCounts.failed}</div>
					</div>
					<div class="rounded-lg border border-border bg-card px-4 py-3">
						<div class="text-[11px] uppercase tracking-wide text-muted-foreground">Not Started</div>
						<div class="mt-1 text-2xl font-semibold tabular-nums">{progressCounts.not_started}</div>
					</div>
				</div>

				<div class="grid gap-4 xl:grid-cols-2">
					<div class="rounded-lg border border-border bg-card px-4 py-3">
						<div class="text-[11px] uppercase tracking-wide text-muted-foreground">Next Pending</div>
						{#if nextPendingRow}
							<div class="mt-2 font-mono text-sm">{fileName(nextPendingRow.relative_path)}</div>
							<div class="mt-1 text-xs text-muted-foreground">
								{nextPendingRow.model_type} · {nextPendingRow.target} · {nextPendingRow.feature_set}
							</div>
						{:else}
							<div class="mt-2 text-sm text-muted-foreground">All configs have been attempted.</div>
						{/if}
					</div>

					<div class="rounded-lg border border-border bg-card px-4 py-3">
						<div class="text-[11px] uppercase tracking-wide text-muted-foreground">Needs Attention</div>
						{#if nextAttentionRow}
							<div class="mt-2 font-mono text-sm">{fileName(nextAttentionRow.relative_path)}</div>
							<div class="mt-1 text-xs text-muted-foreground">
								Last job {nextAttentionRow.job_id ? shortId(nextAttentionRow.job_id, 12) : '-'}
							</div>
						{:else}
							<div class="mt-2 text-sm text-muted-foreground">No failed configs currently tracked.</div>
						{/if}
					</div>
				</div>

				<div class="rounded-lg border border-border bg-card overflow-hidden">
					<div class="flex items-center gap-3 border-b border-border px-4 py-3">
						<div>
							<h2 class="text-sm font-semibold">Config Status</h2>
							<p class="text-xs text-muted-foreground">Durable execution state derived from configs, runs, and latest job outcomes.</p>
						</div>
						<input
							type="text"
							class="ml-auto w-full max-w-sm rounded-md border border-border/60 bg-background/60 px-3 py-1.5 text-[12px] placeholder:text-muted-foreground/80"
							placeholder="Search configs"
							bind:value={progressQuery}
						/>
					</div>

					<div class="overflow-auto">
						<table class="w-full text-sm">
							<thead class="sticky top-0 z-10 bg-card">
								<tr class="border-b border-border/60 text-left text-[11px] uppercase tracking-wide text-muted-foreground">
									<th class="px-4 py-2">Status</th>
									<th class="px-4 py-2">Config</th>
									<th class="px-4 py-2">Model</th>
									<th class="px-4 py-2">Target</th>
									<th class="px-4 py-2">Features</th>
									<th class="px-4 py-2">Run</th>
									<th class="px-4 py-2">Last Update</th>
								</tr>
							</thead>
							<tbody class="divide-y divide-border/50">
								{#each filteredProgressRows as row (row.config_id)}
									<tr class="hover:bg-muted/15">
										<td class="px-4 py-2.5">
											<span class="inline-flex rounded px-2 py-0.5 text-[10px] font-medium uppercase {progressStatusClass(row.status)}">
												{progressStatusLabel(row.status)}
											</span>
										</td>
										<td class="px-4 py-2.5">
											<div class="font-mono text-[12px]">{fileName(row.relative_path)}</div>
											<div class="text-[11px] text-muted-foreground">{row.relative_path}</div>
										</td>
										<td class="px-4 py-2.5">{row.model_type}</td>
										<td class="px-4 py-2.5">{row.target}</td>
										<td class="px-4 py-2.5">{row.feature_set}</td>
										<td class="px-4 py-2.5">
											{#if row.run_id}
												<a
													href="/experiments/{data.experiment.experiment_id}/runs/{row.run_id}"
													class="font-mono text-primary underline underline-offset-2"
												>
													{shortId(row.run_id, 12)}
												</a>
											{:else if row.job_id}
												<span class="font-mono text-muted-foreground">{shortId(row.job_id, 12)}</span>
											{:else}
												<span class="text-muted-foreground">-</span>
											{/if}
										</td>
										<td class="px-4 py-2.5 text-muted-foreground">{fmtTime(row.finished_at ?? row.created_at)}</td>
									</tr>
								{:else}
									<tr>
										<td colspan="7" class="px-4 py-6 text-center text-muted-foreground">No configs match the current filter.</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				</div>
			</div>
		</div>

	{:else}
		<div class="flex flex-1 min-h-0 px-6 py-4 {SHOW_RUNOPS_SIDEBAR ? 'gap-6' : ''}">
			{#if SHOW_RUNOPS_SIDEBAR}
			<div class="w-[420px] flex-shrink-0 flex flex-col border border-border rounded-lg overflow-hidden bg-background">
				<section class="flex flex-col min-h-0 {launchSectionOpen ? 'flex-1' : 'flex-shrink-0'}">
					<button
						type="button"
						class="flex-shrink-0 px-4 w-full flex items-center gap-2.5 py-2.5 bg-muted/40 hover:bg-muted/55 transition-colors cursor-pointer select-none text-left"
						onclick={() => (launchSectionOpen = !launchSectionOpen)}
						aria-label={launchSectionOpen ? 'Collapse configs section' : 'Expand configs section'}
					>
						<span class="h-1.5 w-1.5 rounded-full bg-emerald-400/80 flex-shrink-0"></span>
						<h2 class="text-[15px] font-semibold tracking-tight">Configs</h2>
						<svg class="ml-auto w-4 h-4 text-muted-foreground transition-transform {launchSectionOpen ? '' : '-rotate-90'}" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clip-rule="evenodd"/></svg>
					</button>

					{#if launchSectionOpen}
						<div class="flex-shrink-0 flex gap-2 px-4 pt-2.5">
							<input
								type="text"
								class="flex-1 min-w-0 rounded-md border border-border/60 bg-background/60 px-3 py-1.5 text-[12px] placeholder:text-muted-foreground/80"
								placeholder="Search configs"
								bind:value={configQuery}
							/>
						</div>

						<div class="flex-1 min-h-0 overflow-y-auto mx-4 my-2 rounded-md ring-1 ring-border/40 divide-y divide-border/40">
							{#each filteredConfigs as item (item.config_id)}
								<div
									class="w-full flex items-start gap-2 px-2.5 py-1.5 text-left text-[11px] hover:bg-muted/20 border-l-2 border-transparent"
								>
									<div class="flex-1 min-w-0">
										<div class="font-mono text-[10px] leading-tight truncate" title={item.relative_path}>
											{fileName(item.relative_path)}
										</div>
										{#if item.summary.run_id}
											<div class="mt-0.5 text-[10px] text-muted-foreground font-mono">{shortId(item.summary.run_id, 12)}</div>
										{/if}
									</div>
									<div class="text-right shrink-0">
										<div>{item.summary.model_type ?? '-'}</div>
										<div class="text-muted-foreground text-[10px] truncate max-w-[120px]" title={item.summary.target_payout ?? item.summary.target ?? '-'}>
											{item.summary.target_payout ?? item.summary.target ?? '-'}
										</div>
									</div>
								</div>
							{:else}
								<div class="px-3 py-3 text-center text-muted-foreground text-[11px]">
									No configs found.
								</div>
							{/each}
						</div>

						<div class="flex-shrink-0 px-4 pb-2.5 space-y-2">
							<p class="text-[11px] text-muted-foreground">{filteredConfigs.length} visible / {configTotal} total</p>
							{#if configsError}
								<p class="text-xs text-negative">{configsError}</p>
							{/if}
						</div>
					{/if}
				</section>

				<section class="flex flex-col min-h-0 {queueSectionOpen ? 'flex-1' : 'flex-shrink-0'}">
					<button
						type="button"
						class="flex-shrink-0 px-4 w-full flex items-center gap-2.5 py-2.5 bg-muted/40 hover:bg-muted/55 transition-colors cursor-pointer select-none text-left"
						onclick={() => (queueSectionOpen = !queueSectionOpen)}
						aria-label={queueSectionOpen ? 'Collapse run queue section' : 'Expand run queue section'}
					>
						<span class="h-1.5 w-1.5 rounded-full bg-sky-400/80 flex-shrink-0"></span>
						<h2 class="text-[15px] font-semibold tracking-tight">Run Queue</h2>
						<span class="text-xs text-muted-foreground ml-auto">{activeJobs.length} active</span>
						<svg class="w-4 h-4 text-muted-foreground transition-transform {queueSectionOpen ? '' : '-rotate-90'}" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clip-rule="evenodd"/></svg>
					</button>

					{#if queueSectionOpen}
						<div class="flex-1 min-h-0 overflow-y-auto px-4 py-2.5 space-y-2.5">
							<div class="rounded-md bg-background/30 ring-1 ring-border/40">
								<table class="w-full text-[11px]">
									<thead class="sticky top-0 z-10 bg-background">
										<tr class="border-b border-border/60 bg-muted/10 text-left">
											<th scope="col" class="px-2 py-1">Job</th>
											<th scope="col" class="px-2 py-1">Status</th>
											<th scope="col" class="px-2 py-1">Config</th>
										</tr>
									</thead>
									<tbody class="divide-y divide-border/50">
										{#each activeJobs as job (job.job_id)}
											<tr
												class="hover:bg-muted/20 cursor-pointer {selectedJobId === job.job_id ? 'bg-muted/30' : ''}"
												onclick={() => selectJob(job.job_id)}
											>
												<td class="px-2 py-1 font-mono">
													{shortId(job.job_id, 10)}
													{#if jobTypeBadge(job)}
														<span class="ml-1 inline-block rounded px-1 py-0 text-[9px] bg-violet-500/20 text-violet-300">{jobTypeBadge(job)}</span>
													{/if}
												</td>
												<td class="px-2 py-1">
													<span class="inline-block rounded px-1.5 py-0.5 text-[10px] uppercase {runJobStatusClass(job.status)}">
														{job.status}
													</span>
													{#if job.queue_position != null}
														<div class="mt-0.5 text-[10px] text-muted-foreground">q{job.queue_position}</div>
													{/if}
												</td>
												<td class="px-2 py-1">
													<div class="font-mono text-[10px] truncate max-w-[170px]" title={job.config_id}>
														{fileName(job.config_id)}
													</div>
													{#if job.canonical_run_id}
														<a
															href="/experiments/{data.experiment.experiment_id}/runs/{job.canonical_run_id}"
															class="text-primary underline underline-offset-2 text-[10px]"
															onclick={(event) => event.stopPropagation()}
															>run {shortId(job.canonical_run_id, 10)}</a>
													{/if}
												</td>
											</tr>
										{:else}
											<tr>
												<td colspan="3" class="px-3 py-3 text-center text-muted-foreground">No active jobs.</td>
											</tr>
										{/each}
									</tbody>
								</table>
							</div>

							{#if terminalJobs.length > 0}
								<details class="pt-1">
									<summary class="text-xs text-muted-foreground cursor-pointer hover:text-foreground select-none">
										Job History ({terminalJobs.length})
									</summary>
									<div class="mt-2 max-h-44 overflow-auto rounded-md bg-background/30 ring-1 ring-border/40">
										<table class="w-full text-[11px]">
											<thead class="sticky top-0 z-10 bg-background">
												<tr class="border-b border-border/60 bg-muted/10 text-left">
													<th scope="col" class="px-2 py-1">Job</th>
													<th scope="col" class="px-2 py-1">Status</th>
													<th scope="col" class="px-2 py-1">Config</th>
													<th scope="col" class="px-2 py-1">Finished</th>
												</tr>
											</thead>
											<tbody class="divide-y divide-border/50">
												{#each terminalJobs as job (job.job_id)}
													<tr
														class="hover:bg-muted/20 cursor-pointer {selectedJobId === job.job_id ? 'bg-muted/30' : ''}"
														onclick={() => selectJob(job.job_id)}
													>
														<td class="px-2 py-1 font-mono">
															{shortId(job.job_id, 10)}
															{#if jobTypeBadge(job)}
																<span class="ml-1 inline-block rounded px-1 py-0 text-[9px] bg-violet-500/20 text-violet-300">{jobTypeBadge(job)}</span>
															{/if}
														</td>
														<td class="px-2 py-1">
															<span class="inline-block rounded px-1.5 py-0.5 text-[10px] uppercase {runJobStatusClass(job.status)}">
																{job.status}
															</span>
														</td>
														<td class="px-2 py-1">
															<div class="font-mono text-[10px] truncate max-w-[160px]" title={job.config_id}>
																{fileName(job.config_id)}
															</div>
															{#if job.canonical_run_id}
																<a
																	href="/experiments/{data.experiment.experiment_id}/runs/{job.canonical_run_id}"
																	class="text-primary underline underline-offset-2 text-[10px]"
																	onclick={(event) => event.stopPropagation()}
																	>run {shortId(job.canonical_run_id, 10)}</a>
															{/if}
														</td>
														<td class="px-2 py-1 text-[10px] text-muted-foreground">{fmtTime(job.finished_at)}</td>
													</tr>
												{/each}
											</tbody>
										</table>
									</div>
								</details>
							{/if}

							{#if jobsError}
								<p class="text-xs text-negative mt-2">{jobsError}</p>
							{/if}
						</div>
					{/if}
				</section>

				<section class="flex flex-col min-h-0 {monitorSectionOpen ? 'flex-1' : 'flex-shrink-0'}">
					<button
						type="button"
						class="flex-shrink-0 px-4 w-full flex items-center gap-2.5 py-2.5 bg-muted/40 hover:bg-muted/55 transition-colors cursor-pointer select-none text-left"
						onclick={() => (monitorSectionOpen = !monitorSectionOpen)}
						aria-label={monitorSectionOpen ? 'Collapse live monitor section' : 'Expand live monitor section'}
					>
						<span class="h-1.5 w-1.5 rounded-full bg-violet-400/80 flex-shrink-0"></span>
						<h3 class="text-[15px] font-semibold tracking-tight whitespace-nowrap">Live Monitor</h3>
						{#if selectedJob}
							<div class="flex items-center gap-1.5 text-[10px] text-muted-foreground whitespace-nowrap shrink-0 ml-auto">
								<span class="font-mono text-muted-foreground/90">{shortId(selectedJob.job_id, 12)}</span>
								<span class="inline-block rounded px-1.5 py-0.5 uppercase {runJobStatusClass(selectedJob.status)}">
									{selectedJob.status}
								</span>
							</div>
						{/if}
						<svg class="w-4 h-4 text-muted-foreground transition-transform {monitorSectionOpen ? '' : '-rotate-90'} {selectedJob ? '' : 'ml-auto'}" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clip-rule="evenodd"/></svg>
					</button>

					{#if monitorSectionOpen}
						<div class="flex-1 min-h-0 overflow-y-auto px-4 py-2.5 space-y-3">
							{#if monitorLoading}
								<p class="text-sm text-muted-foreground">Loading monitor...</p>
							{:else if monitorError}
								<p class="text-sm text-negative">{monitorError}</p>
							{:else if !selectedJob}
								<p class="text-sm text-muted-foreground">Select a queued or running job to monitor live progress.</p>
							{:else}
								{#if latestSample}
									<p class="text-[11px] text-muted-foreground">
										Telemetry scope: {scopeLabel(latestSample.scope)}
									</p>
								{/if}
								{#if telemetryMessage}
									<p class="text-xs text-amber-300">{telemetryMessage}</p>
								{/if}
								{#if TERMINAL_JOB_STATUSES.has(selectedJob.status) && selectedJob.status !== 'completed'}
									<div class="rounded-md border border-negative/40 bg-negative/10 px-3 py-2.5">
										<div class="flex items-center gap-2 mb-1.5">
											<span class="inline-block rounded px-1.5 py-0.5 text-[10px] uppercase {runJobStatusClass(selectedJob.status)}">{selectedJob.status}</span>
											<span class="text-xs font-medium text-negative">Job {shortId(selectedJob.job_id, 10)}</span>
										</div>
										{#if selectedJob.error?.message}
											<p class="text-xs text-foreground">{selectedJob.error.message}</p>
										{:else if selectedJob.exit_code == null && selectedJob.signal == null}
											<p class="text-xs text-foreground">Process terminated unexpectedly (likely OOM kill)</p>
										{:else}
											<p class="text-xs text-foreground">
												Exit code: {selectedJob.exit_code ?? 'unknown'}
												{#if selectedJob.signal != null}, Signal: {selectedJob.signal}{/if}
											</p>
										{/if}
									</div>
								{/if}

								<div class="grid grid-cols-2 gap-2 text-xs">
									<div class="rounded-md border border-border/45 bg-background/25 px-2 py-1.5">
										<div class="text-muted-foreground uppercase">Current stage</div>
										<div class="font-medium mt-1">{latestStageEvent?.payload.current_stage ?? '-'}</div>
									</div>
									<div class="rounded-md border border-border/45 bg-background/25 px-2 py-1.5">
										<div class="text-muted-foreground uppercase">Completed stages</div>
										<div class="font-medium mt-1">
											{Array.isArray(latestStageEvent?.payload.completed_stages)
												? latestStageEvent?.payload.completed_stages.length
												: '-'}
										</div>
									</div>
									<div class="rounded-md border border-border/45 bg-background/25 px-2 py-1.5">
										<div class="text-muted-foreground uppercase">CPU (job tree)</div>
										<div class="font-medium mt-1 tabular-nums">{fmtPercent(processCpu(latestSample))}</div>
									</div>
									<div class="rounded-md border border-border/45 bg-background/25 px-2 py-1.5">
										<div class="text-muted-foreground uppercase">RAM (job tree)</div>
										<div class="font-medium mt-1 tabular-nums">{fmtGb(processRssGb(latestSample))}</div>
									</div>
									<div class="rounded-md border border-border/45 bg-background/25 px-2 py-1.5">
										<div class="text-muted-foreground uppercase">RAM free (host)</div>
										<div class="font-medium mt-1 tabular-nums">{fmtGb(hostRamAvailableGb(latestSample))}</div>
									</div>
									<div class="rounded-md border border-border/45 bg-background/25 px-2 py-1.5">
										<div class="text-muted-foreground uppercase">GPU (host)</div>
										<div class="font-medium mt-1 tabular-nums">{fmtPercent(hostGpu(latestSample))}</div>
									</div>
								</div>

								{#if latestMetricEvent}
									<div class="text-xs text-muted-foreground">
										Latest metrics:
										<span class="ml-2 tabular-nums">bmc_last_200_eras_mean {fmt((latestMetricEvent.payload.bmc_last_200_eras_mean as number | undefined) ?? null)}</span>
										<span class="ml-2 tabular-nums">corr_sharpe {fmt((latestMetricEvent.payload.corr_sharpe as number | undefined) ?? null)}</span>
										<span class="ml-2 tabular-nums">corr_mean {fmt((latestMetricEvent.payload.corr_mean as number | undefined) ?? null)}</span>
										<span class="ml-2 tabular-nums">mmc_mean {fmt((latestMetricEvent.payload.mmc_mean as number | undefined) ?? null)}</span>
										<span class="ml-2 tabular-nums">bmc_mean {fmt((latestMetricEvent.payload.bmc_mean as number | undefined) ?? null)}</span>
									</div>
								{/if}

								<div class="inline-flex rounded-md border border-border/60 bg-background/25 p-0.5">
									{#each ['events', 'logs', 'resources'] as tab (tab)}
										<button
											type="button"
											class="px-2.5 py-1 rounded text-[11px] capitalize transition-colors {monitorTab === tab ? 'bg-muted/70 text-foreground' : 'text-muted-foreground hover:text-foreground'}"
											onclick={() => (monitorTab = tab as 'events' | 'logs' | 'resources')}
										>{tab}</button>
									{/each}
								</div>

								{#if monitorTab === 'events'}
									<div class="max-h-56 overflow-auto space-y-1.5">
										{#each [...jobEvents].reverse().slice(0, 50) as event (event.id)}
											<div class="rounded-md border border-border/45 bg-background/25 px-2.5 py-1.5">
												<div class="flex items-center justify-between gap-2">
													<span class="font-medium text-xs">{event.event_type}</span>
													<span class="text-[10px] text-muted-foreground">{fmtTime(event.created_at)}</span>
												</div>
												{#if Object.keys(event.payload).length > 0}
													<p class="mt-1 text-[10px] text-muted-foreground font-mono break-all">{prettyEventPayload(event)}</p>
												{/if}
											</div>
										{:else}
											<p class="text-xs text-muted-foreground">No events yet.</p>
										{/each}
									</div>
								{:else if monitorTab === 'logs'}
									<div class="max-h-56 overflow-auto rounded-md border border-border/45 bg-background/25 p-2 font-mono text-[10px] leading-relaxed">
										{#each jobLogs.slice(-300) as log (log.id)}
											<div class="whitespace-pre-wrap break-all {log.stream === 'stderr' ? 'text-negative' : 'text-foreground'}">
												<span class="text-muted-foreground">[{fmtTime(log.created_at)}]</span>
												<span class="text-muted-foreground ml-1 uppercase">{log.stream}</span>
												<span class="ml-2">{log.line}</span>
											</div>
										{:else}
											<p class="text-xs text-muted-foreground">No logs yet.</p>
										{/each}
									</div>
								{:else}
									<div class="space-y-2.5">
										<div class="grid grid-cols-2 gap-2 text-xs">
											<div class="rounded-md border border-border/45 bg-background/25 px-2 py-1.5">
												<div class="text-muted-foreground uppercase">CPU avg (job tree)</div>
												<div class="font-medium tabular-nums mt-1">{fmtPercent(sampleAverages?.process_cpu_percent)}</div>
											</div>
											<div class="rounded-md border border-border/45 bg-background/25 px-2 py-1.5">
												<div class="text-muted-foreground uppercase">RAM avg (job tree)</div>
												<div class="font-medium tabular-nums mt-1">{fmtGb(sampleAverages?.process_rss_gb)}</div>
											</div>
											<div class="rounded-md border border-border/45 bg-background/25 px-2 py-1.5">
												<div class="text-muted-foreground uppercase">RAM free (host)</div>
												<div class="font-medium tabular-nums mt-1">{fmtGb(latestSample?.host_ram_available_gb ?? latestSample?.ram_available_gb)}</div>
											</div>
											<div class="rounded-md border border-border/45 bg-background/25 px-2 py-1.5">
												<div class="text-muted-foreground uppercase">GPU avg (host)</div>
												<div class="font-medium tabular-nums mt-1">{fmtPercent(sampleAverages?.host_gpu_percent)}</div>
											</div>
										</div>
										<div class="max-h-44 overflow-auto rounded-md bg-background/30 ring-1 ring-border/40">
											<table class="w-full text-xs">
												<thead class="sticky top-0 z-10 bg-background">
													<tr class="border-b border-border/60 bg-muted/10">
														<th scope="col" class="text-left px-2 py-1 text-muted-foreground uppercase">Time</th>
														<th scope="col" class="text-right px-2 py-1 text-muted-foreground uppercase">CPU (job tree)</th>
														<th scope="col" class="text-right px-2 py-1 text-muted-foreground uppercase">RAM (job tree)</th>
														<th scope="col" class="text-right px-2 py-1 text-muted-foreground uppercase">RAM free (host)</th>
														<th scope="col" class="text-right px-2 py-1 text-muted-foreground uppercase">GPU (host)</th>
													</tr>
												</thead>
												<tbody class="divide-y divide-border/50">
													{#each [...jobSamples].reverse().slice(0, 25) as sample (sample.id)}
														<tr>
															<td class="px-2 py-1">{fmtTime(sample.created_at)}</td>
															<td class="px-2 py-1 text-right tabular-nums">{fmtPercent(processCpu(sample))}</td>
															<td class="px-2 py-1 text-right tabular-nums">{fmtGb(processRssGb(sample))}</td>
															<td class="px-2 py-1 text-right tabular-nums">{fmtGb(hostRamAvailableGb(sample))}</td>
															<td class="px-2 py-1 text-right tabular-nums">{fmtPercent(hostGpu(sample))}</td>
														</tr>
													{:else}
														<tr>
															<td colspan="5" class="px-3 py-4 text-center text-muted-foreground">No resource samples yet.</td>
														</tr>
													{/each}
												</tbody>
											</table>
										</div>
									</div>
								{/if}
								{#if streamError}
									<p class="text-xs text-amber-300 mt-2">{streamError}</p>
								{/if}
							{/if}
						</div>
					{/if}
				</section>

				<div class="flex-shrink-0 flex justify-end px-4 py-2 border-t border-border">
					<button
						type="button"
						class="px-2 py-1 rounded-md border border-border/60 text-[11px] text-muted-foreground hover:text-foreground hover:bg-muted/20 transition-colors"
						onclick={() => { void refreshConfigs(); void refreshRunJobs({ keepSelection: true }); }}
						disabled={jobsBusy}
					>Refresh</button>
				</div>
			</div>
			{/if}

			{#if runOpsView === 'chart'}
				{@render opsRail()}

				<div class="flex-1 min-w-0 flex flex-col min-h-0">
					{#if selectedOp?.type === 'run'}
						<div class="overflow-y-auto overflow-x-hidden flex-1 min-h-0">
							<RunDetailPanel
								runId={selectedOp.id}
								experimentId={data.experiment.experiment_id}
								runs={data.runs}
								readOnly={readOnly}
								onClose={() => (selectedOp = null)}
							/>
						</div>
					{:else if selectedOp?.type === 'hpo'}
						<div class="overflow-y-auto overflow-x-hidden flex-1 min-h-0">
						<HpoDetailPanel
							studyId={selectedOp.id}
							experimentId={data.experiment.experiment_id}
							onClose={() => (selectedOp = null)}
						/>
						</div>
					{:else if selectedOp?.type === 'ensemble'}
						<div class="overflow-y-auto overflow-x-hidden flex-1 min-h-0">
						<EnsembleDetailPanel
							ensembleId={selectedOp.id}
							experimentId={data.experiment.experiment_id}
							onClose={() => (selectedOp = null)}
						/>
						</div>
					{:else}
						<div class="flex flex-1 items-center justify-center rounded-lg border border-border bg-card text-sm text-muted-foreground">
							Select an op from the left to inspect it.
						</div>
					{/if}
				</div>
			{:else}
				<div class="flex-1 min-w-0 flex flex-col min-h-0">
					{@render opsTable()}
				</div>
			{/if}
		</div>
	{/if}
</div>
