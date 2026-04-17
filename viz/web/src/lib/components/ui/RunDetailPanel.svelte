<script lang="ts">
	import { tick, untrack } from 'svelte';

	import {
		api,
		type DiagnosticsSources,
		type ExperimentRun,
		type ResourceSample,
		type RunBundle,
		type RunBundleSection,
		type RunEvent,
		type RunManifest,
		type ScoringDashboard
	} from '$lib/api/client';
	import { ensureClientPerfObservers, mark, measure } from '$lib/perf';
	import { isLocalSource, type SourceContext } from '$lib/source';
	import { fmt } from '$lib/utils';

	let {
		runId,
		experimentId,
		experimentName = null,
		runs = [],
		source,
		readOnly = false,
		onClose = undefined
	}: {
		runId: string;
		experimentId: string;
		experimentName?: string | null;
		runs?: ExperimentRun[];
		source?: SourceContext;
		readOnly?: boolean;
		onClose?: (() => void) | undefined;
	} = $props();

	type MainTab = 'performance' | 'diagnostics' | 'artifacts' | 'timeline';

	type AsyncState<T> = {
		runId: string | null;
		loaded: boolean;
		loading: boolean;
		error: string | null;
		value: T | null;
	};

	type PerformanceData = {
		scoringDashboard: ScoringDashboard | null;
	};

	type DiagnosticsData = {
		resources: ResourceSample[] | null;
		diagnosticsSources: DiagnosticsSources | null;
	};

	type ArtifactsData = {
		resolvedConfig: { yaml: string } | null;
		trials: Record<string, unknown>[] | null;
		bestParams: Record<string, unknown> | null;
	};

	const sectionControllers = new Map<string, AbortController>();
	const chartModulesPromise = Promise.all([
		import('$lib/components/charts/CumulativeCorrChart.svelte'),
		import('$lib/components/charts/PerEraLineChart.svelte')
	]);

	let mainTab = $state<MainTab>('performance');
	let copyState = $state<'idle' | 'copied' | 'error'>('idle');
	let diagnosticsPanelModule = $state<Promise<typeof import('$lib/components/ui/run-detail/RunDetailDiagnosticsSection.svelte')> | null>(null);
	let timelinePanelModule = $state<Promise<typeof import('$lib/components/ui/run-detail/RunDetailTimelineSection.svelte')> | null>(null);
	let artifactsPanelModule = $state<Promise<typeof import('$lib/components/ui/run-detail/RunDetailArtifactsSection.svelte')> | null>(null);

	let manifestState = $state<AsyncState<RunManifest>>({
		runId: null,
		loaded: false,
		loading: false,
		error: null,
		value: null
	});
	let metricsState = $state<AsyncState<Record<string, number>>>({
		runId: null,
		loaded: false,
		loading: false,
		error: null,
		value: null
	});
	let performanceState = $state<AsyncState<PerformanceData>>({
		runId: null,
		loaded: false,
		loading: false,
		error: null,
		value: null
	});
	let diagnosticsState = $state<AsyncState<DiagnosticsData>>({
		runId: null,
		loaded: false,
		loading: false,
		error: null,
		value: null
	});
	let artifactsState = $state<AsyncState<ArtifactsData>>({
		runId: null,
		loaded: false,
		loading: false,
		error: null,
		value: null
	});
	let timelineState = $state<AsyncState<RunEvent[]>>({
		runId: null,
		loaded: false,
		loading: false,
		error: null,
		value: null
	});

	let currentRun = $derived.by(() => runs.find((run) => run.run_id === runId) ?? null);
	let manifest = $derived(manifestState.value);
	let scoringDashboard = $derived(performanceState.value?.scoringDashboard ?? null);
	let resources = $derived(diagnosticsState.value?.resources ?? []);
	let diagnosticsSources = $derived(diagnosticsState.value?.diagnosticsSources ?? null);
	let resolvedConfig = $derived(artifactsState.value?.resolvedConfig ?? null);
	let trials = $derived(artifactsState.value?.trials ?? null);
	let bestParams = $derived(artifactsState.value?.bestParams ?? null);
	let events = $derived(timelineState.value ?? []);
	let performanceTransitioning = $derived(performanceState.loading && performanceState.runId !== runId && performanceState.value != null);
	let showPerformancePlaceholder = $derived(performanceState.loading && performanceState.value == null);

	let headerTitle = $derived(manifest?.run_name ?? currentRun?.run_name ?? runId);
	let headerStatus = $derived(manifest?.status ?? currentRun?.status ?? null);
	let headerCreatedAt = $derived(manifest?.created_at ?? currentRun?.created_at ?? null);
	let experimentLabel = $derived(experimentName ?? experimentId);
	let sourceLabel = $derived.by(() => {
		if (isLocalSource(source)) return 'Local store';
		return source?.source_label ?? source?.source_id ?? source?.source_kind ?? 'Remote source';
	});

	let resourceStats = $derived.by(() => {
		if (resources.length === 0) return null;
		const cpu = resources.map((sample) => sample.cpu).filter((value): value is number => typeof value === 'number');
		const ram = resources.map((sample) => sample.ram).filter((value): value is number => typeof value === 'number');
		const gpu = resources.map((sample) => sample.gpu).filter((value): value is number => typeof value === 'number');
		const avg = (vals: number[]) => (vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null);
		const max = (vals: number[]) => (vals.length ? Math.max(...vals) : null);
		return {
			cpu_avg: avg(cpu),
			cpu_max: max(cpu),
			ram_avg: avg(ram),
			ram_max: max(ram),
			gpu_avg: avg(gpu),
			gpu_max: max(gpu)
		};
	});

	let latestResource = $derived.by(() => {
		if (resources.length === 0) return null;
		return resources[resources.length - 1];
	});

	const performanceMetricOrder = [
		'corr_native',
		'corr_ender20',
		'bmc',
		'mmc',
		'cwmm',
		'corr_with_benchmark',
		'corr_delta_vs_baseline',
		'fnc_native',
		'fnc_ender20'
	];

	let visiblePerformanceMetrics = $derived.by(() => {
		const available = new Set(availableMetricKeys());
		return performanceMetricOrder.filter((key) => available.has(key) && !shouldHideMetric(key));
	});
	let scalarScorecardCards = $derived.by(() => {
		const summary = scoringDashboard?.summary;
		if (!summary) return [];
		const cards: Array<{
			key: string;
			label: string;
			chips: Array<{ label: string; value: unknown }>;
		}> = [];
		const bmcLast200Chips = metricSummaryChips('bmc_last_200_eras');
		if (bmcLast200Chips.length > 0) {
			cards.push({
				key: 'bmc_last_200_eras',
				label: 'BMC Last 200 Eras',
				chips: bmcLast200Chips
			});
		}
		if (typeof summary.avg_corr_with_benchmark === 'number') {
			cards.push({
				key: 'avg_corr_with_benchmark',
				label: 'Avg Corr with Benchmark',
				chips: [{ label: 'value', value: summary.avg_corr_with_benchmark }]
			});
		}
		return cards;
	});

	const foldMetricCharts: Array<{
		key: 'corr_native' | 'corr_ender20' | 'bmc';
		label: string;
	}> = [
		{ key: 'corr_native', label: 'CORR Native Fold Mean' },
		{ key: 'corr_ender20', label: 'CORR Ender20 Fold Mean' },
		{ key: 'bmc', label: 'BMC Fold Mean' }
	];

	function availableMetricKeys(): string[] {
		return scoringDashboard?.meta?.available_metric_keys ?? [];
	}

	function shouldHideMetric(metricKey: string): boolean {
		const meta = scoringDashboard?.meta;
		if (!meta) return false;
		if (metricKey.startsWith('fnc_')) {
			if (meta.omissions?.post_training_features === 'feature_neutral_metrics_disabled') {
				return true;
			}
		}
		if (meta.target_col && meta.payout_target_col && meta.target_col === meta.payout_target_col) {
			if (metricKey.endsWith('_ender20')) {
				const nativeKey = metricKey.replace(/_ender20$/, '_native');
				return availableMetricKeys().includes(nativeKey);
			}
		}
		return false;
	}

	function metricRows(metricKey: string, seriesType: 'per_era' | 'cumulative'): Array<Record<string, string | number>> {
		const rows = scoringDashboard?.series ?? [];
		return rows
			.filter((row) => row.metric_key === metricKey && row.series_type === seriesType)
			.map((row, index) => ({
				era: row.era ?? index + 1,
				value: typeof row.value === 'number' ? row.value : 0
			}));
	}

	function foldMetricRows(metricKey: 'corr_native' | 'corr_ender20' | 'bmc'): Array<Record<string, string | number>> {
		const rows = scoringDashboard?.fold_snapshots ?? [];
		const valueKey =
			metricKey === 'corr_native'
				? 'corr_native_fold_mean'
				: metricKey === 'corr_ender20'
					? 'corr_ender20_fold_mean'
					: 'bmc_fold_mean';
		return rows
			.filter((row) => typeof row[valueKey] === 'number')
			.map((row) => ({
				era: Number(row.cv_fold),
				value: Number(row[valueKey])
			}));
	}

	function metricDisplayLabel(metricKey: string): string {
		const labels: Record<string, string> = {
			corr_native: 'CORR Native',
			corr_ender20: 'CORR Ender20',
			bmc: 'BMC',
			mmc: 'MMC',
			fnc_native: 'FNC Native',
			fnc_ender20: 'FNC Ender20',
			cwmm: 'CWMM',
			corr_with_benchmark: 'Corr vs Benchmark',
			corr_delta_vs_baseline: 'Corr Delta vs Baseline'
		};
		const label = labels[metricKey] ?? metricKey;
		const meta = scoringDashboard?.meta;
		if (meta?.target_col && meta?.payout_target_col && meta.target_col === meta.payout_target_col) {
			if (metricKey.endsWith('_native')) {
				return `${label} (Native = Ender20)`;
			}
		}
		return label;
	}

	function metricSummaryRow(metricKey: string): Record<string, unknown> | null {
		if (!scoringDashboard) return null;
		if (metricKey.startsWith('fnc_')) {
			return scoringDashboard.feature_summary;
		}
		return scoringDashboard.summary;
	}

	function metricSummaryChips(metricKey: string): Array<{ label: string; value: unknown }> {
		const row = metricSummaryRow(metricKey);
		if (!row) return [];
		const chips: Array<{ label: string; value: unknown }> = [];
		for (const suffix of ['mean', 'std', 'sharpe', 'max_drawdown']) {
			const key = `${metricKey}_${suffix}`;
			if (key in row) {
				chips.push({ label: suffix.replace('_', ' '), value: row[key] });
			}
		}
		if (metricKey === 'bmc' && typeof row.avg_corr_with_benchmark === 'number') {
			chips.push({ label: 'avg corr vs bench', value: row.avg_corr_with_benchmark });
		}
		return chips;
	}

	function statusClass(status?: string | null): string {
		switch ((status ?? '').toUpperCase()) {
			case 'FINISHED':
			case 'COMPLETED':
				return 'bg-blue-500/20 text-blue-300 border border-blue-500/30';
			case 'RUNNING':
				return 'bg-positive/20 text-positive border border-positive/40';
			case 'FAILED':
				return 'bg-negative/20 text-negative border border-negative/40';
			case 'CANCELED':
				return 'bg-muted text-muted-foreground border border-border';
			default:
				return 'bg-muted text-muted-foreground border border-border';
		}
	}

	function formatTime(value: string | undefined | null): string {
		if (!value) return '—';
		const date = new Date(value);
		if (Number.isNaN(date.getTime())) return value;
		return date.toLocaleString();
	}

	function resetState<T>(state: AsyncState<T>, preserveValue = false): void {
		if (!preserveValue) {
			state.runId = null;
			state.value = null;
		}
		state.loaded = false;
		state.loading = false;
		state.error = null;
	}

	function abortSection(section: string): void {
		const controller = sectionControllers.get(section);
		if (!controller) return;
		controller.abort();
		sectionControllers.delete(section);
	}

	function abortAllSections(): void {
		for (const controller of sectionControllers.values()) {
			controller.abort();
		}
		sectionControllers.clear();
	}

	function beginSectionRequest(section: string): AbortController {
		abortSection(section);
		const controller = new AbortController();
		sectionControllers.set(section, controller);
		return controller;
	}

	function endSectionRequest(section: string, controller: AbortController): void {
		const active = sectionControllers.get(section);
		if (active === controller) {
			sectionControllers.delete(section);
		}
	}

	function measureShell(id: string): void {
		mark(`run-detail:${id}:shell`);
		measure(`run_click_to_shell:${id}`, `run-detail:${id}:start`, `run-detail:${id}:shell`);
	}

	function markTabFetchStart(id: string, tab: MainTab): void {
		mark(`run-detail:${id}:${tab}:start`);
	}

	function markTabFetchEnd(id: string, tab: MainTab): void {
		mark(`run-detail:${id}:${tab}:end`);
		measure(`run_tab_fetch:${id}:${tab}`, `run-detail:${id}:${tab}:start`, `run-detail:${id}:${tab}:end`);
		if (tab === 'performance') {
			measure(
				`run_click_to_performance_chart:${id}`,
				`run-detail:${id}:start`,
				`run-detail:${id}:${tab}:end`
			);
		}
	}

	async function loadBundleSection(id: string, section: MainTab, sections: RunBundleSection[]): Promise<RunBundle | null> {
		const controller = beginSectionRequest(section);
		markTabFetchStart(id, section);
		try {
			const payload = await api.getRunBundle(id, { sections, signal: controller.signal, ...source });
			if (controller.signal.aborted || runId !== id) return null;
			markTabFetchEnd(id, section);
			return payload;
		} catch (err) {
			if (controller.signal.aborted) return null;
			throw err;
		} finally {
			endSectionRequest(section, controller);
		}
	}

	async function loadPerformance(id: string): Promise<void> {
		if (performanceState.loading && performanceState.runId === id) return;
		if (performanceState.loaded && performanceState.runId === id) return;
		performanceState.loading = true;
		manifestState.loading = true;
		metricsState.loading = true;
		performanceState.error = null;
		manifestState.error = null;
		metricsState.error = null;
		try {
			const payload = await loadBundleSection(id, 'performance', ['manifest', 'metrics', 'scoring_dashboard']);
			if (payload == null) return;
			manifestState.runId = id;
			manifestState.value = payload.manifest ?? manifestState.value;
			manifestState.loaded = true;
			metricsState.runId = id;
			metricsState.value = payload.metrics ?? metricsState.value ?? currentRun?.metrics ?? null;
			metricsState.loaded = true;
			performanceState.runId = id;
			performanceState.value = {
				scoringDashboard: payload.scoring_dashboard ?? null
			};
			performanceState.loaded = true;
		} catch (err) {
			performanceState.runId = id;
			performanceState.value = null;
			performanceState.error = err instanceof Error ? err.message : 'Failed to load performance data.';
			performanceState.loaded = true;
			manifestState.runId = id;
			manifestState.error = performanceState.error;
			manifestState.loaded = true;
			metricsState.runId = id;
			metricsState.error = performanceState.error;
			metricsState.loaded = true;
		} finally {
			performanceState.loading = false;
			manifestState.loading = false;
			metricsState.loading = false;
		}
	}

	async function loadDiagnostics(id: string): Promise<void> {
		if (diagnosticsState.loading && diagnosticsState.runId === id) return;
		if (diagnosticsState.loaded && diagnosticsState.runId === id) return;
		diagnosticsState.loading = true;
		diagnosticsState.error = null;
		try {
			const payload = await loadBundleSection(id, 'diagnostics', ['resources', 'diagnostics_sources']);
			if (payload == null) return;
			diagnosticsState.runId = id;
			diagnosticsState.value = {
				resources: payload.resources ?? [],
				diagnosticsSources: payload.diagnostics_sources ?? null
			};
			diagnosticsState.loaded = true;
		} catch (err) {
			diagnosticsState.runId = id;
			diagnosticsState.value = null;
			diagnosticsState.error = err instanceof Error ? err.message : 'Failed to load diagnostics.';
			diagnosticsState.loaded = true;
		} finally {
			diagnosticsState.loading = false;
		}
	}

	async function loadArtifacts(id: string): Promise<void> {
		if (artifactsState.loading && artifactsState.runId === id) return;
		if (artifactsState.loaded && artifactsState.runId === id) return;
		artifactsState.loading = true;
		artifactsState.error = null;
		try {
			const payload = await loadBundleSection(id, 'artifacts', ['resolved_config', 'trials', 'best_params']);
			if (payload == null) return;
			artifactsState.runId = id;
			artifactsState.value = {
				resolvedConfig: payload.resolved_config ?? null,
				trials: payload.trials ?? null,
				bestParams: payload.best_params ?? null
			};
			artifactsState.loaded = true;
		} catch (err) {
			artifactsState.runId = id;
			artifactsState.value = null;
			artifactsState.error = err instanceof Error ? err.message : 'Failed to load artifacts.';
			artifactsState.loaded = true;
		} finally {
			artifactsState.loading = false;
		}
	}

	async function loadTimeline(id: string): Promise<void> {
		if (timelineState.loading && timelineState.runId === id) return;
		if (timelineState.loaded && timelineState.runId === id) return;
		timelineState.loading = true;
		timelineState.error = null;
		try {
			const payload = await loadBundleSection(id, 'timeline', ['events']);
			if (payload == null) return;
			timelineState.runId = id;
			timelineState.value = payload.events ?? [];
			timelineState.loaded = true;
		} catch (err) {
			timelineState.runId = id;
			timelineState.value = null;
			timelineState.error = err instanceof Error ? err.message : 'Failed to load timeline.';
			timelineState.loaded = true;
		} finally {
			timelineState.loading = false;
		}
	}

	async function copyRunId(): Promise<void> {
		try {
			await navigator.clipboard.writeText(runId);
			copyState = 'copied';
			setTimeout(() => {
				copyState = 'idle';
			}, 1500);
		} catch {
			copyState = 'error';
			setTimeout(() => {
				copyState = 'idle';
			}, 1500);
		}
	}

	function ensureLazyPanel(tab: Exclude<MainTab, 'performance'>): void {
		if (tab === 'diagnostics' && diagnosticsPanelModule == null) {
			diagnosticsPanelModule = import('$lib/components/ui/run-detail/RunDetailDiagnosticsSection.svelte');
			return;
		}
		if (tab === 'timeline' && timelinePanelModule == null) {
			timelinePanelModule = import('$lib/components/ui/run-detail/RunDetailTimelineSection.svelte');
			return;
		}
		if (tab === 'artifacts' && artifactsPanelModule == null) {
			artifactsPanelModule = import('$lib/components/ui/run-detail/RunDetailArtifactsSection.svelte');
		}
	}

	$effect(() => {
		ensureClientPerfObservers();
	});

	$effect(() => {
		const id = runId;
		abortAllSections();
		mainTab = 'performance';
		copyState = 'idle';
		untrack(() => {
			resetState(manifestState);
			resetState(metricsState);
			resetState(performanceState);
			resetState(diagnosticsState);
			resetState(artifactsState);
			resetState(timelineState);
		});
		mark(`run-detail:${id}:start`);
		void tick().then(() => measureShell(id));
		untrack(() => {
			void loadPerformance(id);
		});
		return () => {
			abortAllSections();
		};
	});

	$effect(() => {
		const id = runId;
		const tab = mainTab;
		untrack(() => {
			if (tab === 'performance') {
				void loadPerformance(id);
				return;
			}
			if (tab === 'diagnostics') {
				ensureLazyPanel('diagnostics');
				void loadDiagnostics(id);
				return;
			}
			if (tab === 'artifacts') {
				ensureLazyPanel('artifacts');
				void loadArtifacts(id);
				return;
			}
			ensureLazyPanel('timeline');
			void loadTimeline(id);
		});
	});
</script>

<div class="space-y-6 min-w-0 px-4 pb-6 lg:px-5">
	<header class="sticky top-0 z-20 py-3 bg-background/90 backdrop-blur border-b border-border/60">
		<div class="flex items-start justify-between gap-3">
			<div>
				<h1 class="text-xl font-semibold">
					{headerTitle}
				</h1>
				<div class="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
					<span class="font-mono">{runId}</span>
					<span>•</span>
					<span>{experimentLabel}</span>
					<span>•</span>
					<span class="rounded border border-border/70 px-1.5 py-0.5 text-[10px] uppercase tracking-[0.18em]">
						{sourceLabel}
					</span>
					{#if headerCreatedAt}
						<span>•</span>
						<span>{formatTime(headerCreatedAt)}</span>
					{/if}
				</div>
			</div>
			<div class="flex items-center gap-2">
				{#if headerStatus}
					<span class="inline-flex px-2 py-1 rounded-full text-[11px] uppercase tracking-wider font-medium {statusClass(headerStatus)}">
						{headerStatus}
					</span>
				{/if}
				{#if onClose}
					<button
						type="button"
						class="rounded-md border border-border px-2.5 py-1.5 text-xs text-muted-foreground hover:text-foreground hover:bg-muted/30"
						onclick={onClose}
					>
						Back to runs
					</button>
				{/if}
				<button
					type="button"
					class="rounded-md border border-border px-2.5 py-1.5 text-xs hover:bg-muted/30 {copyState === 'copied' ? 'text-positive border-positive/40' : copyState === 'error' ? 'text-negative border-negative/40' : 'text-muted-foreground border-border hover:text-foreground'}"
					onclick={() => void copyRunId()}
				>
					{copyState === 'copied' ? 'Copied' : copyState === 'error' ? 'Copy failed' : 'Copy Run ID'}
				</button>
			</div>
		</div>
	</header>

	<div class="flex flex-wrap items-center gap-2 border-b border-border">
		{#each [
			{ key: 'performance', label: 'Performance' },
			{ key: 'diagnostics', label: 'Diagnostics' },
			{ key: 'artifacts', label: 'Artifacts' },
			{ key: 'timeline', label: 'Timeline' }
		] as tab (tab.key)}
			<button
				type="button"
				class="px-3 py-2 text-sm font-medium transition-colors {mainTab === tab.key
					? 'border-b-2 border-primary text-foreground'
					: 'text-muted-foreground hover:text-foreground'}"
				onclick={() => (mainTab = tab.key as MainTab)}
			>
				{tab.label}
			</button>
		{/each}
	</div>

	{#if mainTab === 'performance'}
		<div class="space-y-5 transition-opacity duration-150 {performanceTransitioning ? 'opacity-90' : 'opacity-100'}">
			{#if showPerformancePlaceholder}
				<div class="bg-card border border-border rounded-lg p-6 text-center text-muted-foreground text-sm" aria-busy="true">
					Loading performance charts...
				</div>
			{:else if performanceState.error}
				<div class="bg-card border border-border rounded-lg p-6 text-center text-negative text-sm">
					{performanceState.error}
				</div>
			{:else}
				{#if performanceTransitioning}
					<div class="flex justify-end -mb-2">
						<p class="text-[11px] text-muted-foreground">Updating charts...</p>
					</div>
				{/if}
				<div class="space-y-5">
					{#if scoringDashboard?.meta?.source === 'legacy_fallback'}
						<div class="bg-card border border-border rounded-lg p-4">
							<p class="text-sm text-muted-foreground">
								This run predates canonical scoring series. Rescore the run to materialize MMC, FNC, and exposure charts.
							</p>
						</div>
					{/if}

					{#if scalarScorecardCards.length > 0}
						<div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
							{#each scalarScorecardCards as card (card.key)}
								<div class="rounded-lg border border-border bg-card p-4 space-y-3">
									<div>
										<h3 class="text-sm font-medium">{card.label}</h3>
									</div>
									<div class="flex flex-wrap gap-1.5">
										{#each card.chips as chip (`${card.key}-${chip.label}`)}
											<span class="rounded-full border border-border px-2 py-1 text-[11px] text-muted-foreground">
												{chip.label}: {fmt(chip.value as number | null)}
											</span>
										{/each}
									</div>
								</div>
							{/each}
						</div>
					{/if}

					<div class="grid grid-cols-1 2xl:grid-cols-2 gap-4">
						{#each visiblePerformanceMetrics as metricKey (metricKey)}
							<div class="rounded-lg border border-border bg-card p-4 space-y-3">
								<div class="flex items-start justify-between gap-3">
									<div>
										<h3 class="text-sm font-medium">{metricDisplayLabel(metricKey)}</h3>
										<p class="text-[11px] text-muted-foreground">{metricRows(metricKey, 'per_era').length} eras</p>
									</div>
									{#if metricSummaryChips(metricKey).length > 0}
										<div class="flex flex-wrap justify-end gap-1.5">
											{#each metricSummaryChips(metricKey) as chip (`${metricKey}-${chip.label}`)}
												<span class="rounded-full border border-border px-2 py-1 text-[11px] text-muted-foreground">
													{chip.label}: {fmt(chip.value as number | null)}
												</span>
											{/each}
										</div>
									{/if}
								</div>
								<div class="grid grid-cols-1 gap-3 3xl:grid-cols-2">
									{#await chartModulesPromise}
										<div class="grid grid-cols-1 gap-3 3xl:grid-cols-2">
											<div class="rounded-md border border-border/60 p-3 text-[11px] text-muted-foreground" aria-busy="true">Loading chart…</div>
											<div class="rounded-md border border-border/60 p-3 text-[11px] text-muted-foreground" aria-busy="true">Loading chart…</div>
										</div>
									{:then [cumulativeModule, perEraModule]}
										{@const CumulativeChart = cumulativeModule.default}
										{@const PerEraChart = perEraModule.default}
										<div class="rounded-md border border-border/60 p-2">
											<p class="mb-2 text-[11px] uppercase tracking-wider text-muted-foreground">Cumulative</p>
											<CumulativeChart data={metricRows(metricKey, 'cumulative')} height="220px" />
										</div>
										<div class="rounded-md border border-border/60 p-2">
											<p class="mb-2 text-[11px] uppercase tracking-wider text-muted-foreground">Per Era</p>
											<PerEraChart data={metricRows(metricKey, 'per_era')} height="220px" />
										</div>
									{/await}
								</div>
							</div>
						{/each}

						{#each foldMetricCharts as foldChart (`fold-${foldChart.key}`)}
							{#if foldMetricRows(foldChart.key).length > 0}
								<div class="rounded-lg border border-border bg-card p-4 space-y-3">
									<div>
										<h3 class="text-sm font-medium">{foldChart.label}</h3>
										<p class="text-[11px] text-muted-foreground">{foldMetricRows(foldChart.key).length} folds</p>
									</div>
									<div class="rounded-md border border-border/60 p-2">
										{#await chartModulesPromise}
											<div class="p-3 text-[11px] text-muted-foreground" aria-busy="true">Loading chart…</div>
										{:then [, perEraModule]}
											{@const FoldChart = perEraModule.default}
											<FoldChart data={foldMetricRows(foldChart.key)} height="220px" />
										{/await}
									</div>
								</div>
							{/if}
						{/each}

					</div>
				</div>
			{/if}
		</div>
	{:else if mainTab === 'diagnostics'}
		<svelte:boundary>
			{#snippet failed()}
				<div class="rounded-lg border border-border bg-card p-5 text-sm text-negative">Failed to render diagnostics panel.</div>
			{/snippet}
			{#if diagnosticsPanelModule}
				{#await diagnosticsPanelModule}
					<div class="rounded-lg border border-border bg-card p-5 text-sm text-muted-foreground" aria-busy="true">Loading diagnostics panel…</div>
				{:then module}
					{@const DiagnosticsPanel = module.default}
					<DiagnosticsPanel
						loading={diagnosticsState.loading}
						error={diagnosticsState.error}
						resources={resources}
						resourceStats={resourceStats}
						latestResource={latestResource}
						manifest={manifest}
						manifestLoading={manifestState.loading}
						manifestError={manifestState.error}
						diagnosticsSources={diagnosticsSources}
					/>
				{/await}
			{/if}
		</svelte:boundary>
	{:else if mainTab === 'timeline'}
		<svelte:boundary>
			{#snippet failed()}
				<div class="rounded-lg border border-border bg-card p-5 text-sm text-negative">Failed to render timeline panel.</div>
			{/snippet}
			{#if timelinePanelModule}
				{#await timelinePanelModule}
					<div class="rounded-lg border border-border bg-card p-5 text-sm text-muted-foreground" aria-busy="true">Loading timeline panel…</div>
				{:then module}
					{@const TimelinePanel = module.default}
					<TimelinePanel loading={timelineState.loading} error={timelineState.error} events={events} />
				{/await}
			{/if}
		</svelte:boundary>
	{:else if mainTab === 'artifacts'}
		<svelte:boundary>
			{#snippet failed()}
				<div class="rounded-lg border border-border bg-card p-5 text-sm text-negative">Failed to render artifacts panel.</div>
			{/snippet}
			{#if artifactsPanelModule}
				{#await artifactsPanelModule}
					<div class="rounded-lg border border-border bg-card p-5 text-sm text-muted-foreground" aria-busy="true">Loading artifacts panel…</div>
				{:then module}
					{@const ArtifactsPanel = module.default}
					<ArtifactsPanel
						runId={runId}
						source={source}
						readOnly={readOnly}
						loading={artifactsState.loading}
						resolvedConfig={resolvedConfig}
						trials={trials}
						bestParams={bestParams}
						manifest={manifest}
						manifestLoading={manifestState.loading}
					/>
				{/await}
			{/if}
		</svelte:boundary>
	{/if}
</div>
