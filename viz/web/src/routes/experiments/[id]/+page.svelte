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
		type RunLifecycle,
		type RunJobListResponse,
		type RunJobLog,
		type RunJobSample,
		type SystemCapabilities
	} from '$lib/api/client';
	import {
		DASHBOARD_KEY_METRICS,
		RUNOPS_ALL_SCORING_METRICS,
		RUNOPS_MAIN_METRICS,
		metricLabel,
		metricNumber,
		metricShortLabel,
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
	import type { ExperimentDocContext } from '$lib/markdown/experiment';
	import { fmt, fmtGb, fmtPercent } from '$lib/utils';
	import BenchmarkSimilarityChart from '$lib/components/charts/BenchmarkSimilarityChart.svelte';
	import ParetoChart from '$lib/components/charts/ParetoChart.svelte';
	import MarkdownDoc from '$lib/components/ui/MarkdownDoc.svelte';
	import Select, { type SelectOption } from '$lib/components/ui/Select.svelte';
	import RunDetailPanel from '$lib/components/ui/RunDetailPanel.svelte';
	import HpoDetailPanel from '$lib/components/ui/HpoDetailPanel.svelte';
	import EnsembleDetailPanel from '$lib/components/ui/EnsembleDetailPanel.svelte';
	import { isLocalSource, withSourceHref, type SourceContext } from '$lib/source';

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
				source: SourceContext;
			};
		} = $props();

	const ACTIVE_JOB_STATUSES = new Set(['queued', 'starting', 'running', 'canceling']);
	const TERMINAL_JOB_STATUSES = new Set(['completed', 'failed', 'canceled', 'stale']);

	let sortColumn = $state('bmc_last_200_eras_mean');
	let sortDirection = $state<'asc' | 'desc'>('desc');
	let chartTab = $state<'charts' | 'composite' | 'target-analysis'>('charts');
	let benchmarkSimilarityBmcMode = $state<'last200' | 'full'>('last200');
	let pageTab = $state<'analysis' | 'progress' | 'runops'>('analysis');
	let runOpsView = $state<'table' | 'chart'>('table');
	let runOpsColumnPickerOpen = $state(false);
	let runOpsHiddenColumns = $state<Set<string>>(new Set());
	let runOpsHeatmapEnabled = $state(true);
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
	const PARETO_CORR_WEIGHT = 0.75;
	const PARETO_MMC_WEIGHT = 2.25;
	let paretoColorMode = $state<ParetoColorMode>('default');
	let selectedTargetAnalysisTarget = $state('');

	let configItems = $state<ExperimentConfig[]>([]);
	let configTotal = $state<number>(0);
	let configQuery = $state('');
	let configsError = $state<string | null>(null);

	let runJobs = $state<RunJob[]>([]);
	let runJobsTotal = $state<number>(0);
	let jobsBusy = $state(false);
	let jobsError = $state<string | null>(null);
	let runLifecycles = $state(new Map<string, RunLifecycle>());
	let progressPollGeneration = 0;
	let documentVisible = $state(true);

	let selectedJobId = $state<string | null>(null);
	let selectedJob = $state<RunJob | null>(null);

	type OpType = 'run' | 'hpo' | 'ensemble';
	let selectedOp = $state<{ id: string; type: OpType } | null>(null);
	let readOnly = $derived(Boolean(data.capabilities?.read_only));
	let sourceLabel = $derived.by(() => {
		if (isLocalSource(data.source)) return 'Local store';
		return data.source.source_label ?? data.source.source_id ?? data.source.source_kind ?? 'Remote source';
	});
	const COMPLETED_RUN_STATUSES = new Set(['FINISHED', 'COMPLETED', 'COMPLETE']);
	let experimentDocContext = $derived.by<ExperimentDocContext>(() => {
		const roundIndexes = new Set<number>();
		for (const result of data.roundResults) {
			if (typeof result.round_index === 'number' && Number.isFinite(result.round_index)) {
				roundIndexes.add(result.round_index);
			}
		}
		const completedRuns = data.runs.filter((run) =>
			COMPLETED_RUN_STATUSES.has((run.status ?? '').toUpperCase())
		).length;
		return {
			experimentId: data.experiment.experiment_id,
			name: data.experiment.name,
			status: data.experiment.status,
			createdAt: data.experiment.created_at,
			updatedAt: data.experiment.updated_at,
			championRunId: data.experiment.champion_run_id,
			tags: data.experiment.tags ?? [],
			stats: {
				totalRuns: data.runs.length,
				completedRuns,
				roundCount: roundIndexes.size,
				studyCount: data.studies.length,
				ensembleCount: data.ensembles.length
			}
		};
	});

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

	let metricColumns = $derived.by(() => allMetricColumns.filter((col) => !runOpsHiddenColumns.has(col)));

	let runOpsMetricDividerColumn = $derived.by(() => {
		if (runOpsMainMetricColumns.length === 0) return null;
		return runOpsMainMetricColumns[runOpsMainMetricColumns.length - 1];
	});

	// Metrics where a lower number is preferable — standard deviations and the
	// drawdown. Everything else defaults to "higher is better" in the tint scale.
	const METRIC_LOWER_IS_BETTER = new Set<string>([
		'bmc_std',
		'corr_std',
		'fnc_std',
		'mmc_std',
		'cwmm_std',
		'max_drawdown'
	]);

	let allMetricColumns = $derived.by(() => {
		const ordered: string[] = [];
		const seen = new Set<string>();
		for (const key of [...RUNOPS_MAIN_METRICS, ...RUNOPS_ALL_SCORING_METRICS]) {
			if (!runOpsAvailableMetricKeys.has(key) || seen.has(key)) continue;
			ordered.push(key);
			seen.add(key);
		}
		return ordered;
	});

	function runOpsColumnsStorageKey(experimentId: string): string {
		return `numereng:runops:hidden-columns:${experimentId}`;
	}

	function runOpsHeatmapStorageKey(experimentId: string): string {
		return `numereng:runops:heatmap:${experimentId}`;
	}

	$effect(() => {
		const experimentId = data.experiment.experiment_id;
		if (typeof window === 'undefined') return;
		try {
			const raw = window.localStorage.getItem(runOpsColumnsStorageKey(experimentId));
			if (raw) {
				const parsed = JSON.parse(raw);
				if (Array.isArray(parsed)) {
					runOpsHiddenColumns = new Set(parsed.filter((key) => typeof key === 'string'));
				}
			}
			const heatmapRaw = window.localStorage.getItem(runOpsHeatmapStorageKey(experimentId));
			if (heatmapRaw === '0' || heatmapRaw === 'false') {
				runOpsHeatmapEnabled = false;
			}
		} catch {
			// localStorage might be unavailable or corrupted — fall back to defaults.
		}
	});

	function toggleHeatmap() {
		runOpsHeatmapEnabled = !runOpsHeatmapEnabled;
		if (typeof window !== 'undefined') {
			try {
				window.localStorage.setItem(
					runOpsHeatmapStorageKey(data.experiment.experiment_id),
					runOpsHeatmapEnabled ? '1' : '0'
				);
			} catch {
				// ignore
			}
		}
	}

	function setColumnVisibility(col: string, visible: boolean) {
		const next = new Set(runOpsHiddenColumns);
		if (visible) next.delete(col);
		else next.add(col);
		runOpsHiddenColumns = next;
		if (typeof window !== 'undefined') {
			try {
				window.localStorage.setItem(
					runOpsColumnsStorageKey(data.experiment.experiment_id),
					JSON.stringify([...next])
				);
			} catch {
				// ignore
			}
		}
	}

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

	// Per-column min/max across the current visible ops, used to drive a
	// subtle monochrome luminance tint on each metric cell. Computed once per
	// (sortedOps, metricColumns) change rather than per cell.
	let metricColumnRanges = $derived.by(() => {
		const ranges = new Map<string, { min: number; max: number }>();
		for (const col of metricColumns) {
			let min = Number.POSITIVE_INFINITY;
			let max = Number.NEGATIVE_INFINITY;
			for (const op of sortedOps) {
				const value = opMetricValue(op, col);
				if (value == null || Number.isNaN(value)) continue;
				if (value < min) min = value;
				if (value > max) max = value;
			}
			if (Number.isFinite(min) && Number.isFinite(max)) {
				ranges.set(col, { min, max });
			}
		}
		return ranges;
	});

	// Cell tint is a subtle red-to-green gradient keyed on the value's rank
	// within its column. Alpha is a U-shape around the midpoint so median
	// values stay (nearly) untinted and the extremes speak the loudest.
	// Only reds/greens are used — the rest of the UI stays monochrome, so the
	// semantic colors register as a whisper rather than a branding choice.
	const HEATMAP_MAX_ALPHA = 0.09;
	const HEATMAP_POSITIVE_RGB = '34, 197, 94'; // emerald-500
	const HEATMAP_NEGATIVE_RGB = '239, 68, 68'; // red-500

	function metricCellTint(col: string, value: number | undefined | null): string {
		if (!runOpsHeatmapEnabled) return '';
		if (value == null || Number.isNaN(value)) return '';
		const range = metricColumnRanges.get(col);
		if (!range) return '';
		const span = range.max - range.min;
		if (span <= 0) return '';
		let ratio = (value - range.min) / span;
		if (METRIC_LOWER_IS_BETTER.has(col)) {
			ratio = 1 - ratio;
		}
		const signed = ratio * 2 - 1; // -1 (worst) ... 0 (median) ... 1 (best)
		const strength = Math.abs(signed);
		const alpha = strength * HEATMAP_MAX_ALPHA;
		if (alpha <= 0) return '';
		const rgb = signed >= 0 ? HEATMAP_POSITIVE_RGB : HEATMAP_NEGATIVE_RGB;
		return `rgba(${rgb}, ${alpha.toFixed(3)})`;
	}

	let selectedOperationItem = $derived.by(() => {
		const current = selectedOp;
		if (!current) return null;
		return sortedOps.find((item) => item.op_id === current.id && item.op_type === current.type) ?? null;
	});

	let benchmarkSimilarityPoints = $derived.by(() => {
		const metricKey =
			benchmarkSimilarityBmcMode === 'full' ? 'bmc_mean' : 'bmc_last_200_eras_mean';
		const runPoints = data.runs
			.filter(
				(r) =>
					metricNumber(r.metrics, metricKey) != null &&
					metricNumber(r.metrics, 'corr_with_benchmark') != null
			)
			.map((r) => ({
				id: r.run_id,
				name: r.run_name ?? configNameByRunId.get(r.run_id) ?? r.run_id,
				model_type: r.model_type ?? 'unknown',
				target: targetLabel(r),
				corr_with_benchmark: metricNumber(r.metrics, 'corr_with_benchmark') ?? 0,
				bmc_mean: metricNumber(r.metrics, 'bmc_mean'),
				bmc_last_200_eras_mean: metricNumber(r.metrics, 'bmc_last_200_eras_mean') ?? 0,
				is_champion: r.is_champion,
				source_type: 'run' as const
			}));

		const roundPoints = data.roundResults
			.filter(
				(r) =>
					metricNumber(r.metrics, metricKey) != null &&
					metricNumber(r.metrics, 'corr_with_benchmark') != null
			)
			.map((r) => ({
				id: r.result_id,
				name: r.name,
				model_type: r.model_type ?? 'derived',
				target: r.target ?? '-',
				corr_with_benchmark: metricNumber(r.metrics, 'corr_with_benchmark') ?? 0,
				bmc_mean: metricNumber(r.metrics, 'bmc_mean'),
				bmc_last_200_eras_mean: metricNumber(r.metrics, 'bmc_last_200_eras_mean') ?? 0,
				is_champion: false,
				source_type: 'round_result' as const
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

	type ProgressState =
		| 'not_started'
		| 'queued'
		| 'training'
		| 'completed'
		| 'failed'
		| 'canceled'
		| 'stale';

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
		current_stage: string | null;
		progress_percent: number | null;
		progress_label: string | null;
		updated_at: string | null;
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
			const linkedRunId = latestJob?.canonical_run_id ?? item.summary.run_id ?? null;
			const linkedRun = linkedRunId ? runById.get(linkedRunId) ?? null : null;
			const lifecycle = linkedRunId ? runLifecycles.get(linkedRunId) ?? null : null;
			const status = resolveProgressStatus({
				latestJob,
				linkedRun,
				lifecycle
			});
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
				current_stage: lifecycle?.current_stage ?? null,
				progress_percent:
					status === 'queued' || status === 'training' ? lifecycle?.progress_percent ?? null : null,
				progress_label:
					status === 'queued' || status === 'training' ? lifecycle?.progress_label ?? null : null,
				updated_at:
					lifecycle?.last_heartbeat_at ??
					lifecycle?.updated_at ??
					latestJob?.updated_at ??
					latestJob?.finished_at ??
					linkedRun?.created_at ??
					null,
				finished_at: lifecycle?.finished_at ?? latestJob?.finished_at ?? linkedRun?.created_at ?? null,
				created_at: latestJob?.created_at ?? linkedRun?.created_at ?? null
			};
		});
	});

	let progressCounts = $derived.by(() => {
		const counts = {
			total: progressRows.length,
			not_started: 0,
			queued: 0,
			training: 0,
			completed: 0,
			failed: 0,
			canceled: 0,
			stale: 0
		};
		for (const row of progressRows) {
			counts[row.status] += 1;
		}
		return counts;
	});

	let nextPendingRow = $derived.by(
		() => progressRows.find((row) => row.status === 'not_started') ?? null
	);

	let nextAttentionRow = $derived.by(() => {
		for (const status of ['failed', 'stale', 'canceled'] as const) {
			const match = progressRows.find((row) => row.status === status);
			if (match) return match;
		}
		return null;
	});

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
				row.job_id ?? '',
				row.current_stage ?? '',
				row.progress_label ?? ''
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
					metricNumber(r.metrics, 'corr_payout_mean') != null &&
					metricNumber(r.metrics, 'mmc_payout_mean') != null
			)
				.map((r) => ({
					run_id: r.run_id,
					config_name:
						configNameByRunId.get(r.run_id) ??
						r.run_name ??
						`${r.model_type ?? 'unknown'} · ${targetLabel(r)}`,
					corr_payout_mean: metricNumber(r.metrics, 'corr_payout_mean') ?? 0,
					mmc_payout_mean: metricNumber(r.metrics, 'mmc_payout_mean') ?? 0,
					model_type: r.model_type ?? 'unknown',
					seed: r.seed,
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
					metricNumber(r.metrics, 'corr_payout_mean') != null &&
					metricNumber(r.metrics, 'mmc_payout_mean') != null
			)
				.map((r) => ({
					run_id: r.run_id,
					config_name:
						configNameByRunId.get(r.run_id) ??
						r.name ??
						`${r.model_type ?? 'derived'} · ${r.target ?? '-'}`,
					corr_payout_mean: metricNumber(r.metrics, 'corr_payout_mean') ?? 0,
					mmc_payout_mean: metricNumber(r.metrics, 'mmc_payout_mean') ?? 0,
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
					metricNumber(e.metrics, 'corr_payout_mean') != null &&
					metricNumber(e.metrics, 'mmc_payout_mean') != null
			)
				.map((e) => ({
					run_id: e.ensemble_id,
					config_name: e.name ?? 'ensemble',
					corr_payout_mean: metricNumber(e.metrics, 'corr_payout_mean') ?? 0,
					mmc_payout_mean: metricNumber(e.metrics, 'mmc_payout_mean') ?? 0,
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

	function nativeTargetKey(run: ExperimentRun): string | null {
		return run.target_train ?? run.target_col ?? run.target ?? null;
	}

	function formatTargetOptionLabel(target: string): string {
		const stripped = target.replace(/^target_/, '');
		const parts = stripped.split('_').filter(Boolean);
		if (parts.length === 0) return target;
		const day = parts.length > 1 ? parts[parts.length - 1] : null;
		const nameParts = day ? parts.slice(0, -1) : parts;
		const name = nameParts
			.map((part) => part.charAt(0).toUpperCase() + part.slice(1))
			.join(' ');
		return day ? `${name} ${day}` : name;
	}

	let targetAnalysisOptions = $derived.by(() => {
		const counts = new Map<string, number>();
		for (const run of data.runs) {
			const target = nativeTargetKey(run);
			if (!target) continue;
			if (metricNumber(run.metrics, 'corr_mean') == null || metricNumber(run.metrics, 'mmc_mean') == null) {
				continue;
			}
			counts.set(target, (counts.get(target) ?? 0) + 1);
		}

		return [...counts.entries()]
			.sort((a, b) => {
				if (b[1] !== a[1]) return b[1] - a[1];
				return a[0].localeCompare(b[0]);
			})
			.map(
				([value]): SelectOption => ({
					value,
					label: formatTargetOptionLabel(value)
				})
			);
	});

	let targetAnalysisRuns = $derived.by(() => {
		if (!selectedTargetAnalysisTarget) return [];
		return data.runs
			.filter((run) => nativeTargetKey(run) === selectedTargetAnalysisTarget)
			.filter(
				(run) =>
					metricNumber(run.metrics, 'corr_mean') != null &&
					metricNumber(run.metrics, 'mmc_mean') != null
			)
			.map((run) => ({
				run_id: run.run_id,
				config_name:
					configNameByRunId.get(run.run_id) ??
					run.run_name ??
					`${run.model_type ?? 'unknown'} · ${formatTargetOptionLabel(selectedTargetAnalysisTarget)}`,
				corr_mean: metricNumber(run.metrics, 'corr_mean') ?? 0,
				mmc_mean: metricNumber(run.metrics, 'mmc_mean') ?? 0,
				model_type: run.model_type ?? 'unknown',
				seed: run.seed,
				target: selectedTargetAnalysisTarget,
				entity_type: hpoBestRunIds.has(run.run_id) ? ('hpo_best' as const) : ('run' as const),
				mmc_coverage_ratio_rows: metricNumber(run.metrics, 'mmc_coverage_ratio_rows'),
				bmc_mean: metricNumber(run.metrics, 'bmc_mean'),
				bmc_last_200_eras_mean: metricNumber(run.metrics, 'bmc_last_200_eras_mean'),
				corr_sharpe: metricNumber(run.metrics, 'corr_sharpe'),
				max_drawdown: metricNumber(run.metrics, 'max_drawdown')
			}));
	});

	$effect(() => {
		const options = targetAnalysisOptions;
		if (options.length === 0) {
			selectedTargetAnalysisTarget = '';
			return;
		}
		if (!options.some((option) => option.value === selectedTargetAnalysisTarget)) {
			selectedTargetAnalysisTarget = options[0]?.value ?? '';
		}
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
		if (typeof document === 'undefined') return;
		const updateVisibility = () => {
			documentVisible = document.visibilityState !== 'hidden';
		};
		updateVisibility();
		document.addEventListener('visibilitychange', updateVisibility);
		return () => document.removeEventListener('visibilitychange', updateVisibility);
	});

	$effect(() => {
		if (pageTab !== 'progress') return;
		if (!documentVisible) return;
		void refreshProgressLifecycles();
		const timer = setInterval(() => {
			if (!documentVisible) return;
			void refreshProgressLifecycles();
		}, 3000);
		return () => {
			progressPollGeneration += 1;
			clearInterval(timer);
		};
	});

	$effect(() => {
		const timer = setInterval(() => {
			if (pageTab === 'progress') return;
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

	function resolveProgressStatus({
		latestJob,
		linkedRun,
		lifecycle
	}: {
		latestJob: RunJob | null;
		linkedRun: ExperimentRun | null;
		lifecycle: RunLifecycle | null;
	}): ProgressState {
		if (lifecycle && !ACTIVE_JOB_STATUSES.has(lifecycle.status)) {
			return mapLifecycleStatus(lifecycle.status);
		}
		if (latestJob && TERMINAL_JOB_STATUSES.has(latestJob.status)) {
			return mapLifecycleStatus(latestJob.status);
		}
		if (lifecycle) {
			return mapLifecycleStatus(lifecycle.status);
		}
		if (latestJob) {
			return mapLifecycleStatus(latestJob.status);
		}
		if (linkedRun) {
			return 'completed';
		}
		return 'not_started';
	}

	function mapLifecycleStatus(status: string): ProgressState {
		switch (status) {
			case 'queued':
				return 'queued';
			case 'starting':
			case 'running':
				return 'training';
			case 'completed':
				return 'completed';
			case 'failed':
				return 'failed';
			case 'canceled':
				return 'canceled';
			case 'stale':
				return 'stale';
			default:
				return 'not_started';
		}
	}

	function progressStatusClass(status: ProgressState): string {
		switch (status) {
			case 'completed':
				return 'bg-positive/15 text-positive';
			case 'training':
				return 'bg-sky-500/15 text-sky-300';
			case 'queued':
				return 'bg-muted text-muted-foreground';
			case 'failed':
				return 'bg-negative/15 text-negative';
			case 'canceled':
				return 'bg-amber-500/20 text-amber-300';
			case 'stale':
				return 'bg-muted text-muted-foreground';
			default:
				return 'bg-muted text-muted-foreground';
		}
	}

	function progressStatusLabel(status: ProgressState): string {
		switch (status) {
			case 'not_started':
				return 'not started';
			case 'training':
				return 'training';
			default:
				return status;
		}
	}

	function progressBarWidth(percent: number | null | undefined): string {
		if (percent == null || Number.isNaN(percent)) return '0%';
		return `${Math.min(100, Math.max(0, percent))}%`;
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
				offset: 0,
				...data.source
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
				offset: 0,
				...data.source
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

	async function refreshProgressLifecycles() {
		const generation = ++progressPollGeneration;
		await refreshRunJobs({ keepSelection: true });
		if (generation !== progressPollGeneration) return;

		const runIds = new Set<string>();
		for (const job of runJobs) {
			const runId = job.canonical_run_id;
			if (!runId) continue;
			if (!TERMINAL_JOB_STATUSES.has(job.status) || runLifecycles.has(runId)) {
				runIds.add(runId);
			}
		}

		if (runIds.size === 0) {
			if (generation === progressPollGeneration) {
				runLifecycles = new Map();
			}
			return;
		}

		const results = await Promise.allSettled(
			[...runIds].map(async (runId) => {
				const lifecycle = await api.getRunLifecycle(runId, data.source);
				return [runId, lifecycle] as const;
			})
		);
		if (generation !== progressPollGeneration) return;

		const next = new Map<string, RunLifecycle>();
		const existing = runLifecycles;
		for (const [index, runId] of [...runIds].entries()) {
			const result = results[index];
			if (result?.status === 'fulfilled') {
				next.set(result.value[0], result.value[1]);
				continue;
			}
			const previous = existing.get(runId);
			if (previous) {
				next.set(runId, previous);
			}
		}
		runLifecycles = next;
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
			after_sample_id: sampleCursor,
			...data.source
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
				api.getRunJob(jobId, data.source),
				api.getRunJobEvents(jobId, { limit: 400, ...data.source }),
				api.getRunJobLogs(jobId, { limit: 1200, stream: 'all', ...data.source }),
				api.getRunJobSamples(jobId, { limit: 600, ...data.source })
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

	function selectOperationAndShowDetail(op: Operation) {
		selectOperation(op);
		runOpsView = 'chart';
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

{#snippet opsRail()}
	<div class="w-[280px] flex-shrink-0 flex flex-col border-r border-border bg-card overflow-hidden">
		<div class="flex items-center justify-between gap-3 border-b border-border px-4" style:height="56px">
			<div>
				<h2 class="text-sm font-semibold leading-tight">Ops</h2>
				<p class="text-[11px] text-muted-foreground tabular-nums">{sortedOps.length} total</p>
			</div>
			<div class="flex items-center gap-2">
				<div class="flex flex-col gap-[4px]">
					<button
						type="button"
						aria-label={runOpsHeatmapEnabled ? 'Disable heatmap tint' : 'Enable heatmap tint'}
						aria-pressed={runOpsHeatmapEnabled}
						title={runOpsHeatmapEnabled ? 'Heatmap on' : 'Heatmap off'}
						class="inline-flex h-[18px] w-[34px] items-center justify-center rounded-full border text-muted-foreground transition-colors {runOpsHeatmapEnabled ? 'border-white/25 text-foreground' : 'border-white/15 hover:text-foreground hover:bg-white/10'}"
						onclick={toggleHeatmap}
					>
						{#if runOpsHeatmapEnabled}
							<span class="flex items-center gap-[3px]">
								<span class="h-[6px] w-[6px] rounded-full" style:background-color="rgba(239, 68, 68, 0.7)"></span>
								<span class="h-[6px] w-[6px] rounded-full" style:background-color="rgba(34, 197, 94, 0.7)"></span>
							</span>
						{:else}
							<span class="flex items-center gap-[3px]">
								<span class="h-[6px] w-[6px] rounded-full border border-muted-foreground/40"></span>
								<span class="h-[6px] w-[6px] rounded-full border border-muted-foreground/40"></span>
							</span>
						{/if}
					</button>
					<div class="relative">
						<button
							type="button"
							aria-label="Toggle column visibility"
							aria-expanded={runOpsColumnPickerOpen}
							class="inline-flex h-[18px] w-[34px] items-center justify-center rounded-full border border-white/15 text-muted-foreground hover:text-foreground hover:bg-white/10 transition-colors"
							onclick={() => (runOpsColumnPickerOpen = !runOpsColumnPickerOpen)}
						>
							<svg viewBox="0 0 16 16" aria-hidden="true" class="h-[10px] w-[10px]"><circle cx="3" cy="8" r="1.4" fill="currentColor"/><circle cx="8" cy="8" r="1.4" fill="currentColor"/><circle cx="13" cy="8" r="1.4" fill="currentColor"/></svg>
						</button>
						{#if runOpsColumnPickerOpen}
							<div
								class="absolute right-0 top-full z-40 mt-1.5 w-56 max-h-80 overflow-auto rounded-lg border border-border bg-card shadow-lg p-2 text-xs"
								role="menu"
							>
								<div class="px-2 pt-1 pb-2 text-[10px] uppercase tracking-wider text-muted-foreground">Columns</div>
								{#each allMetricColumns as col (col)}
									{@const visible = !runOpsHiddenColumns.has(col)}
									<label class="flex items-center gap-2 rounded px-2 py-1 cursor-pointer hover:bg-white/5">
										<input
											type="checkbox"
											class="h-3 w-3 accent-white"
											checked={visible}
											onchange={(e) => setColumnVisibility(col, (e.currentTarget as HTMLInputElement).checked)}
										/>
										<span class="tabular-nums">{metricShortLabel(col)}</span>
										<span class="ml-auto truncate text-[10px] text-muted-foreground/60">{col}</span>
									</label>
								{/each}
							</div>
						{/if}
					</div>
				</div>
				<div class="pill-tabs" aria-label="Ops view">
					{#each ['table', 'chart'] as view (view)}
						<button
							type="button"
							class={`pill-tab pill-tab-sm ${runOpsView === view ? 'pill-tab-active' : ''}`}
							onclick={() => (runOpsView = view as 'table' | 'chart')}
						>{view}</button>
					{/each}
				</div>
			</div>
		</div>

		<div class="flex-1 min-h-0 overflow-auto divide-y divide-border/50">
			{#each sortedOps as op (op.op_id)}
				{@const isSelected = selectedOp?.id === op.op_id && selectedOp?.type === op.op_type}
				<button
					type="button"
					class="w-full border-l-2 px-4 text-left transition-colors hover:bg-muted/20 {isSelected ? 'border-l-primary bg-primary/10 shadow-[inset_0_0_0_1px_rgba(255,255,255,0.05)]' : 'border-l-transparent'}"
					style:height="56px"
					onclick={() => selectOperationAndShowDetail(op)}
				>
					<div class="flex items-center gap-2">
						<span class="inline-flex rounded px-1.5 py-0.5 text-[9px] uppercase tracking-wide {opTypeBadgeClass(op.op_type)}">
							{opTypeLabel(op.op_type)}
						</span>
						<div class="min-w-0 flex-1">
							<div class="truncate font-mono text-[12px] font-medium tabular-nums">{shortId(op.op_id, 12)}</div>
							<div class="mt-0.5 truncate text-[10px] text-muted-foreground">
								{op.model_type} · {op.target} · {op.feature_set}
							</div>
						</div>
					</div>
				</button>
			{:else}
				<div class="px-4 py-6 text-sm text-muted-foreground">No ops available.</div>
			{/each}
		</div>
	</div>
{/snippet}

	<div class="experiment-detail-theme -mx-8 -mt-14 md:-mt-8 -mb-8 flex flex-col h-screen">
		<div class="analysis-top-toolbar flex items-center justify-between gap-4 border-b border-border flex-shrink-0 px-8">
		<div class="min-w-0 py-2">
			<div class="truncate text-sm font-semibold text-foreground">{data.experiment.name}</div>
			<div class="mt-1 flex items-center gap-2 text-[10px] uppercase tracking-[0.24em] text-muted-foreground">
				<span>{data.experiment.experiment_id}</span>
				<span class="rounded border border-border/70 px-1.5 py-0.5 tracking-[0.18em]">{sourceLabel}</span>
			</div>
		</div>
		<div class="pill-tabs" aria-label="Experiment section">
		{#each [['analysis', 'Analysis'], ['progress', 'Progress'], ['runops', 'Run Ops']] as [key, label] (key)}
			<button
				type="button"
				class={`pill-tab ${pageTab === key ? 'pill-tab-active' : ''}`}
				onclick={() => (pageTab = key as 'analysis' | 'progress' | 'runops')}
			>{label}</button>
		{/each}
		</div>
	</div>

		{#if pageTab === 'analysis'}
			<div class="analysis-shell flex flex-1 min-h-0 min-w-0">
				<div class="analysis-reader-pane flex-1 min-h-0 min-w-0 overflow-x-hidden">
				<MarkdownDoc
					borderless
					label="Experiment"
					load={() => api.getExperimentDoc(data.experiment.experiment_id, 'EXPERIMENT.md', data.source)}
					variant="experiment"
					experimentContext={experimentDocContext}
					readOnly={readOnly}
					readOnlyMessage=""
				/>
			</div>

				<div class="flex-1 min-h-0 min-w-0 p-4" aria-label="Charts wrapper">
			<div class="bg-card border border-border rounded-lg h-full flex flex-col" aria-label="Charts section">
					<div class="flex items-center justify-between border-b border-border px-5 pt-3 pb-0 flex-shrink-0">
						<div class="pill-tabs" aria-label="Chart view">
							{#each [['charts', 'Benchmark Similarity'], ['composite', 'Payout Proxy'], ['target-analysis', 'Target Analysis']] as [key, label] (key)}
							<button
								type="button"
								class={`pill-tab pill-tab-sm ${chartTab === key ? 'pill-tab-active' : ''}`}
								onclick={() => (chartTab = key as 'charts' | 'composite' | 'target-analysis')}
							>{label}</button>
						{/each}
					</div>
						{#if chartTab === 'charts'}
							<div class="flex rounded-md border border-border overflow-hidden text-[10px]">
								{#each [
									['last200', 'BMC Last 200'],
									['full', 'BMC Mean']
								] as [mode, label] (mode)}
									<button
										type="button"
										class="px-2 py-0.5 transition-colors {benchmarkSimilarityBmcMode === mode
											? 'bg-primary text-primary-foreground'
											: 'bg-muted/40 text-muted-foreground hover:text-foreground'}"
										onclick={() => (benchmarkSimilarityBmcMode = mode as 'last200' | 'full')}
									>{label}</button>
								{/each}
							</div>
						{:else if chartTab === 'composite'}
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
					{:else if chartTab === 'target-analysis'}
						<div class="w-48">
							<Select
								options={targetAnalysisOptions}
								bind:value={selectedTargetAnalysisTarget}
								placeholder="Select target"
								size="xs"
								align="right"
								ariaLabel="Target analysis target"
							/>
						</div>
					{/if}
				</div>

				<div class="flex-1 min-h-0 p-5">
						{#if chartTab === 'charts'}
							<div class="flex flex-col h-full min-h-0">
								<div class="mb-3 flex-shrink-0">
									<h3 class="text-xs uppercase tracking-wider text-muted-foreground font-medium">
										{benchmarkSimilarityBmcMode === 'full'
											? 'BMC Mean vs Corr With Benchmark'
											: 'BMC Last 200 vs Corr With Benchmark'}
									</h3>
								</div>
								{#if benchmarkSimilarityPoints.length > 0}
									<BenchmarkSimilarityChart
										points={benchmarkSimilarityPoints}
										bmcMode={benchmarkSimilarityBmcMode}
										class="h-full"
									/>
								{:else}
									<div class="flex h-full items-center justify-center text-muted-foreground text-sm">
										No benchmark-similarity data available.
								</div>
							{/if}
						</div>
					{:else if chartTab === 'composite'}
						<div class="flex h-full min-h-0 flex-col">
							<div class="mb-3 flex-shrink-0">
								<p class="text-xs text-muted-foreground">
									Historical proxy from CORR (Payout Target) and MMC (Payout Target). Research-only; not a live payout forecast.
								</p>
								<p class="mt-1 text-xs text-muted-foreground font-mono">
									Iso-lines: payout proxy = {PARETO_CORR_WEIGHT.toFixed(2)}*CORR_payout + {PARETO_MMC_WEIGHT.toFixed(2)}*MMC_payout
								</p>
							</div>
							{#if paretoRuns.length > 0}
								<ParetoChart
									runs={paretoRuns}
									colorMode={paretoColorMode}
									corrWeight={PARETO_CORR_WEIGHT}
									mmcWeight={PARETO_MMC_WEIGHT}
									showIsoLines={true}
									class="h-full"
								/>
							{:else}
								<div class="flex h-full items-center justify-center text-muted-foreground text-sm">
									No payout-target CORR/MMC data available for payout proxy view.
								</div>
							{/if}
						</div>
					{:else}
						<div class="flex h-full min-h-0 flex-col">
							<div class="mb-3 flex-shrink-0">
								<p class="text-xs text-muted-foreground">
									Native CORR vs MMC for runs sharing the selected target.
								</p>
							</div>
							{#if targetAnalysisOptions.length === 0}
								<div class="flex h-full items-center justify-center text-muted-foreground text-sm">
									No native CORR/MMC target data available.
								</div>
							{:else if targetAnalysisRuns.length > 0}
								<ParetoChart
									runs={targetAnalysisRuns}
									metricMode="native"
									showIsoLines={false}
									class="h-full"
								/>
							{:else}
								<div class="flex h-full items-center justify-center text-muted-foreground text-sm">
									No native CORR/MMC runs available for this target.
								</div>
							{/if}
						</div>
					{/if}
				</div>
			</div>
			</div>
		</div>

	{:else if pageTab === 'progress'}
		{@const inFlightCount = progressCounts.queued + progressCounts.training}
		{@const completionPercent =
			progressCounts.total > 0
				? Math.round((progressCounts.completed / progressCounts.total) * 100)
				: 0}
		<div class="flex-1 min-h-0 px-6 py-4 overflow-auto">
			<div class="space-y-4">
				<div class="rounded-lg border border-border bg-card px-5 py-4 space-y-3">
					<div class="flex flex-wrap items-center gap-4">
						<div class="flex items-baseline gap-2 tabular-nums">
							<span class="text-2xl font-semibold">{progressCounts.completed}</span>
							<span class="text-sm text-muted-foreground">/ {progressCounts.total} completed</span>
						</div>
						<div class="flex-1 min-w-[12rem] h-2 rounded-full bg-white/5 overflow-hidden">
							<div
								class="h-full bg-primary transition-all"
								style:width="{completionPercent}%"
							></div>
						</div>
						<div class="text-sm tabular-nums text-muted-foreground">{completionPercent}%</div>
					</div>

					<div class="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
						<span class={progressCounts.not_started > 0 ? 'text-foreground' : 'text-muted-foreground/60'}>
							<span class="tabular-nums">{progressCounts.not_started}</span> not started
						</span>
						<span class={inFlightCount > 0 ? 'text-sky-300' : 'text-muted-foreground/60'}>
							<span class="tabular-nums">{inFlightCount}</span> in flight
						</span>
						<span class={progressCounts.failed > 0 ? 'text-negative' : 'text-muted-foreground/60'}>
							<span class="tabular-nums">{progressCounts.failed}</span> failed
						</span>
						{#if progressCounts.canceled > 0}
							<span class="text-amber-300">
								<span class="tabular-nums">{progressCounts.canceled}</span> canceled
							</span>
						{/if}
						{#if progressCounts.stale > 0}
							<span class="text-muted-foreground">
								<span class="tabular-nums">{progressCounts.stale}</span> stale
							</span>
						{/if}
					</div>

					{#if nextPendingRow}
						<div class="text-xs text-muted-foreground">
							<span class="uppercase tracking-wide">Next pending</span>
							<span class="mx-2 text-muted-foreground/50">·</span>
							<span class="font-mono text-foreground">{fileName(nextPendingRow.relative_path)}</span>
							<span class="mx-2 text-muted-foreground/50">·</span>
							<span>{nextPendingRow.model_type} · {nextPendingRow.target} · {nextPendingRow.feature_set}</span>
						</div>
					{/if}
				</div>

				{#if nextAttentionRow}
					<div class="rounded-lg border border-negative/40 bg-negative/5 px-4 py-3">
						<div class="text-[11px] uppercase tracking-wide text-negative">Needs attention</div>
						<div class="mt-1 font-mono text-sm">{fileName(nextAttentionRow.relative_path)}</div>
						<div class="mt-0.5 text-xs text-muted-foreground">
							{progressStatusLabel(nextAttentionRow.status)} · last job {nextAttentionRow.job_id ? shortId(nextAttentionRow.job_id, 12) : '-'}
						</div>
					</div>
				{/if}

				<div class="rounded-lg border border-border bg-card overflow-hidden">
					<div class="flex items-center gap-3 border-b border-border px-4 py-3">
						<div>
							<h2 class="text-sm font-semibold">Config Status</h2>
							<p class="text-xs text-muted-foreground">Live lifecycle state for the latest known attempt per config, with progress for queued and training runs.</p>
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
										<td class="px-4 py-2.5 min-w-[220px]">
											<span class="inline-flex rounded px-2 py-0.5 text-[10px] font-medium uppercase {progressStatusClass(row.status)}">
												{progressStatusLabel(row.status)}
											</span>
											{#if row.status === 'queued' || row.status === 'training'}
												<div class="mt-2 h-2 overflow-hidden rounded-full bg-muted/60">
													<div
														class="h-full rounded-full bg-sky-400 transition-[width] duration-500"
														style={`width: ${progressBarWidth(row.progress_percent)}`}
													></div>
												</div>
												<div class="mt-1 text-[11px] text-muted-foreground">
													{row.progress_label ?? 'Waiting for lifecycle progress...'}
													{#if row.progress_percent != null}
														<span class="ml-1 tabular-nums">{Math.round(row.progress_percent)}%</span>
													{/if}
												</div>
											{/if}
											{#if row.current_stage}
												<div class="mt-1 text-[11px] text-muted-foreground font-mono">{row.current_stage}</div>
											{/if}
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
													href={withSourceHref(`/experiments/${data.experiment.experiment_id}/runs/${row.run_id}`, data.source)}
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
										<td class="px-4 py-2.5 text-muted-foreground">{fmtTime(row.updated_at ?? row.finished_at ?? row.created_at)}</td>
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
		<div class="flex flex-1 min-h-0 {SHOW_RUNOPS_SIDEBAR ? 'gap-6 px-6 py-4' : ''}">
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
															href={withSourceHref(`/experiments/${data.experiment.experiment_id}/runs/${job.canonical_run_id}`, data.source)}
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
																	href={withSourceHref(`/experiments/${data.experiment.experiment_id}/runs/${job.canonical_run_id}`, data.source)}
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

			{@render opsRail()}

			<div class="flex-1 min-w-0 flex flex-col min-h-0 relative bg-card overflow-hidden">
				{#if runOpsView === 'table'}
					<div class="overflow-auto flex-1 min-h-0">
						<table class="w-max min-w-full text-sm leading-[1.35] border-separate border-spacing-0">
							<thead>
								<tr>
									{#each metricColumns as col (col)}
										<th
											scope="col"
											title={metricLabel(col)}
											class="sticky top-0 z-20 min-w-[88px] border-b border-border/70 bg-card px-3 align-middle font-medium text-[11px] tracking-wide text-muted-foreground text-right cursor-pointer hover:text-foreground select-none {runOpsMetricDividerColumn === col ? 'border-r border-border/60' : ''}"
											style:height="56px"
											onclick={() => toggleSort(col)}
										>{metricShortLabel(col)}{sortIndicator(col)}</th>
									{/each}
								</tr>
							</thead>
							<tbody>
								{#each sortedOps as op (op.op_id)}
									{@const isSelected = selectedOp?.id === op.op_id && selectedOp?.type === op.op_type}
									<tr
										class="group cursor-pointer transition-colors hover:bg-white/[0.03] {isSelected ? '' : 'odd:bg-white/[0.015]'}"
										style:height="56px"
										onclick={() => selectOperationAndShowDetail(op)}
									>
										{#each metricColumns as col (col)}
											{@const value = opMetricValue(op, col)}
											{@const tint = isSelected ? '' : metricCellTint(col, value)}
											<td
												class="min-w-[88px] border-b border-border/40 px-3 text-right tabular-nums align-middle {isSelected ? 'bg-primary/10' : ''} {runOpsMetricDividerColumn === col ? 'border-r border-border/60' : ''}"
												style:background-color={tint || undefined}
											>{#if value == null || Number.isNaN(value)}<span class="text-muted-foreground/40">—</span>{:else}{value.toFixed(4)}{/if}</td>
										{/each}
									</tr>
								{:else}
									<tr>
										<td colspan={metricColumns.length} class="px-4 py-6 text-center text-sm text-muted-foreground">
											No ops available.
										</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				{:else if selectedOp?.type === 'run'}
					<div class="overflow-y-auto overflow-x-hidden flex-1 min-h-0">
						{#key selectedOp.id}
							<RunDetailPanel
								runId={selectedOp.id}
								experimentId={data.experiment.experiment_id}
								experimentName={data.experiment.name}
								runs={data.runs}
								source={data.source}
								readOnly={readOnly}
								onClose={() => (selectedOp = null)}
							/>
						{/key}
					</div>
				{:else if selectedOp?.type === 'hpo'}
					<div class="overflow-y-auto overflow-x-hidden flex-1 min-h-0">
						<HpoDetailPanel
							studyId={selectedOp.id}
							experimentId={data.experiment.experiment_id}
							source={data.source}
							onClose={() => (selectedOp = null)}
						/>
					</div>
				{:else if selectedOp?.type === 'ensemble'}
					<div class="overflow-y-auto overflow-x-hidden flex-1 min-h-0">
						<EnsembleDetailPanel
							ensembleId={selectedOp.id}
							experimentId={data.experiment.experiment_id}
							source={data.source}
							onClose={() => (selectedOp = null)}
						/>
					</div>
				{:else}
					<div class="flex flex-1 items-center justify-center text-sm text-muted-foreground">
						Select an op from the left to inspect it.
					</div>
				{/if}
			</div>
		</div>
	{/if}
</div>

<style>
	.experiment-detail-theme {
		background: #0f0f11;
		--color-background: #0f0f11;
		--color-card: #151519;
		--color-popover: #151519;
		--color-surface-100: #151519;
		--color-secondary: #17171b;
		--color-muted: #19191d;
		--color-accent: #1d1d23;
	}

	.analysis-reader-pane {
		background: #111115;
		--color-background: #111115;
		--color-card: #151519;
		--color-popover: #151519;
		--color-surface-100: #151519;
		--color-secondary: #17171b;
		--color-muted: #19191d;
		--color-accent: #1d1d23;
	}

	.analysis-top-toolbar {
		background: #111115;
	}

	.analysis-shell {
		background: #111115;
	}
</style>
