<script lang="ts">
	import { tick, untrack } from 'svelte';

	import {
		api,
		type DiagnosticsSources,
		type ExperimentRun,
		type FeatureImportanceRow,
		type PerEraRow,
		type ResourceSample,
		type RunEvent,
		type RunManifest
	} from '$lib/api/client';
	import CumulativeCorrChart from '$lib/components/charts/CumulativeCorrChart.svelte';
	import FeatureImportanceChart from '$lib/components/charts/FeatureImportanceChart.svelte';
	import PerEraLineChart from '$lib/components/charts/PerEraLineChart.svelte';
	import MarkdownDoc from '$lib/components/ui/MarkdownDoc.svelte';
	import { fmt, fmtGb, fmtPercent } from '$lib/utils';

	let {
		runId,
		experimentId,
		experimentName = null,
		runs,
		readOnly = false,
		onClose = undefined
	}: {
		runId: string;
		experimentId: string;
		experimentName?: string | null;
		runs: ExperimentRun[];
		readOnly?: boolean;
		onClose?: (() => void) | undefined;
	} = $props();

	type MainTab = 'performance' | 'diagnostics' | 'artifacts' | 'timeline';
	type ArtifactTab = 'config' | 'trials' | 'data' | 'notes';
	type MetricKey =
		| 'bmc_last_200_eras_mean'
		| 'corr_sharpe'
		| 'corr_mean'
		| 'mmc_mean'
		| 'mmc_coverage_ratio_rows'
		| 'bmc_mean'
		| 'max_drawdown';

	type AsyncState<T> = {
		runId: string | null;
		loaded: boolean;
		loading: boolean;
		error: string | null;
		value: T | null;
	};

	type PerformanceData = {
		perEraCorr: PerEraRow[] | null;
		featureImportance: FeatureImportanceRow[] | null;
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

	const keyMetrics: MetricKey[] = [
		'bmc_last_200_eras_mean',
		'corr_sharpe',
		'corr_mean',
		'mmc_mean',
		'mmc_coverage_ratio_rows',
		'bmc_mean',
		'max_drawdown'
	];

	const lowerBetter = new Set<MetricKey>(['max_drawdown']);
	const sectionControllers = new Map<string, AbortController>();

	let mainTab = $state<MainTab>('performance');
	let artifactTab = $state<ArtifactTab>('config');
	let copyState = $state<'idle' | 'copied' | 'error'>('idle');

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
	let championRun = $derived.by(() => runs.find((run) => run.is_champion) ?? null);
	let shellMetrics = $derived.by(() => {
		if (currentRun?.metrics && Object.keys(currentRun.metrics).length > 0) {
			return currentRun.metrics;
		}
		return metricsState.value;
	});
	let manifest = $derived(manifestState.value);
	let perEraCorr = $derived(performanceState.value?.perEraCorr ?? null);
	let featureImportance = $derived(performanceState.value?.featureImportance ?? null);
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

	let metricCards = $derived.by(() => {
		return keyMetrics.map((key) => {
			const value = metricValue(currentRun, key) ?? shellMetrics?.[key] ?? null;
			const values = runs
				.map((run) => metricValue(run, key))
				.filter((candidate): candidate is number => candidate != null && Number.isFinite(candidate));
			const total = values.length;
			let rank: number | null = null;
			if (value != null && total > 0) {
				const sorted = [...values].sort((a, b) => (lowerBetter.has(key) ? a - b : b - a));
				rank = sorted.findIndex((candidate) => candidate === value) + 1;
				if (rank === 0) rank = null;
			}
			const championValue = metricValue(championRun, key);
			const improvement =
				value != null && championValue != null
					? lowerBetter.has(key)
						? championValue - value
						: value - championValue
					: null;
			return {
				key,
				label: metricLabel(key),
				value,
				rank,
				total,
				improvement,
				isChampion: championRun?.run_id === runId
			};
		});
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

	function metricValue(run: ExperimentRun | null | undefined, key: MetricKey): number | null {
		if (!run?.metrics) return null;
		const value = run.metrics[key];
		return typeof value === 'number' && Number.isFinite(value) ? value : null;
	}

	function metricLabel(key: MetricKey): string {
		switch (key) {
			case 'bmc_last_200_eras_mean':
				return 'BMC Mean (Last 200 Eras)';
			case 'corr_sharpe':
				return 'CORR Sharpe';
			case 'corr_mean':
				return 'CORR Mean';
			case 'mmc_mean':
				return 'MMC Mean';
			case 'mmc_coverage_ratio_rows':
				return 'MMC Coverage Ratio';
			case 'bmc_mean':
				return 'BMC Mean';
			case 'max_drawdown':
				return 'Max Drawdown';
			default:
				return key;
		}
	}

	function improvementClass(value: number | null): string {
		if (value == null || value === 0) return 'text-muted-foreground';
		return value > 0 ? 'text-positive' : 'text-negative';
	}

	function signed(value: number | null): string {
		if (value == null) return 'n/a';
		const abs = Math.abs(value).toFixed(4);
		if (value > 0) return `+${abs}`;
		if (value < 0) return `-${abs}`;
		return '0.0000';
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

	function formatPayload(payload: Record<string, unknown> | undefined): string {
		if (!payload) return '';
		const text = JSON.stringify(payload);
		if (text.length <= 200) return text;
		return `${text.slice(0, 200)}...`;
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
		performance.mark(`run-detail:${id}:shell`);
		performance.measure(`run_click_to_shell:${id}`, `run-detail:${id}:start`, `run-detail:${id}:shell`);
	}

	function markTabFetchStart(id: string, tab: MainTab): void {
		performance.mark(`run-detail:${id}:${tab}:start`);
	}

	function markTabFetchEnd(id: string, tab: MainTab): void {
		performance.mark(`run-detail:${id}:${tab}:end`);
		performance.measure(`run_tab_fetch:${id}:${tab}`, `run-detail:${id}:${tab}:start`, `run-detail:${id}:${tab}:end`);
		if (tab === 'performance') {
			performance.measure(
				`run_click_to_performance_chart:${id}`,
				`run-detail:${id}:start`,
				`run-detail:${id}:${tab}:end`
			);
		}
	}

	async function loadManifest(id: string): Promise<void> {
		if (manifestState.loading && manifestState.runId === id) return;
		if (manifestState.loaded && manifestState.runId === id) return;
		const controller = beginSectionRequest('manifest');
		manifestState.loading = true;
		manifestState.error = null;
		try {
			const payload = await api.getRunManifest(id, controller.signal);
			if (controller.signal.aborted || runId !== id) return;
			manifestState.runId = id;
			manifestState.value = payload;
			manifestState.loaded = true;
		} catch (err) {
			if (controller.signal.aborted) return;
			manifestState.runId = id;
			manifestState.value = null;
			manifestState.error = err instanceof Error ? err.message : 'Failed to load manifest.';
			manifestState.loaded = true;
		} finally {
			if (!controller.signal.aborted && runId === id) {
				manifestState.loading = false;
			}
			endSectionRequest('manifest', controller);
		}
	}

	async function loadMetrics(id: string): Promise<void> {
		if (metricsState.loading && metricsState.runId === id) return;
		if (metricsState.loaded && metricsState.runId === id) return;
		const controller = beginSectionRequest('metrics');
		metricsState.loading = true;
		metricsState.error = null;
		try {
			const payload = await api.getRunMetrics(id, controller.signal);
			if (controller.signal.aborted || runId !== id) return;
			metricsState.runId = id;
			metricsState.value = payload;
			metricsState.loaded = true;
		} catch (err) {
			if (controller.signal.aborted) return;
			metricsState.runId = id;
			metricsState.value = null;
			metricsState.error = err instanceof Error ? err.message : 'Failed to load metrics.';
			metricsState.loaded = true;
		} finally {
			if (!controller.signal.aborted && runId === id) {
				metricsState.loading = false;
			}
			endSectionRequest('metrics', controller);
		}
	}

	async function loadPerformance(id: string): Promise<void> {
		if (performanceState.loading && performanceState.runId === id) return;
		if (performanceState.loaded && performanceState.runId === id) return;
		const controller = beginSectionRequest('performance');
		performanceState.loading = true;
		performanceState.error = null;
		markTabFetchStart(id, 'performance');
		try {
			const [corr, importance] = await Promise.all([
				api.getPerEraCorr(id, controller.signal).catch(() => null),
				api.getFeatureImportance(id, 30, controller.signal).catch(() => null)
			]);
			if (controller.signal.aborted || runId !== id) return;
			performanceState.runId = id;
			performanceState.value = {
				perEraCorr: corr,
				featureImportance: importance
			};
			performanceState.loaded = true;
			markTabFetchEnd(id, 'performance');
		} catch (err) {
			if (controller.signal.aborted) return;
			performanceState.runId = id;
			performanceState.value = null;
			performanceState.error = err instanceof Error ? err.message : 'Failed to load performance data.';
			performanceState.loaded = true;
		} finally {
			if (!controller.signal.aborted && runId === id) {
				performanceState.loading = false;
			}
			endSectionRequest('performance', controller);
		}
	}

	async function loadDiagnostics(id: string): Promise<void> {
		if (diagnosticsState.loading && diagnosticsState.runId === id) return;
		if (diagnosticsState.loaded && diagnosticsState.runId === id) return;
		const controller = beginSectionRequest('diagnostics');
		diagnosticsState.loading = true;
		diagnosticsState.error = null;
		markTabFetchStart(id, 'diagnostics');
		try {
			const [resourcePayload, diagnosticsPayload] = await Promise.all([
				api.getRunResources(id, 50, controller.signal).catch(() => []),
				api.getRunDiagnosticsSources(id, controller.signal).catch(() => null)
			]);
			if (controller.signal.aborted || runId !== id) return;
			diagnosticsState.runId = id;
			diagnosticsState.value = {
				resources: resourcePayload,
				diagnosticsSources: diagnosticsPayload
			};
			diagnosticsState.loaded = true;
			markTabFetchEnd(id, 'diagnostics');
		} catch (err) {
			if (controller.signal.aborted) return;
			diagnosticsState.runId = id;
			diagnosticsState.value = null;
			diagnosticsState.error = err instanceof Error ? err.message : 'Failed to load diagnostics.';
			diagnosticsState.loaded = true;
		} finally {
			if (!controller.signal.aborted && runId === id) {
				diagnosticsState.loading = false;
			}
			endSectionRequest('diagnostics', controller);
		}
	}

	async function loadArtifacts(id: string): Promise<void> {
		if (artifactsState.loading && artifactsState.runId === id) return;
		if (artifactsState.loaded && artifactsState.runId === id) return;
		const controller = beginSectionRequest('artifacts');
		artifactsState.loading = true;
		artifactsState.error = null;
		markTabFetchStart(id, 'artifacts');
		try {
			const [configPayload, trialsPayload, bestParamsPayload] = await Promise.all([
				api.getResolvedConfig(id, controller.signal).catch(() => null),
				api.getTrials(id, controller.signal).catch(() => null),
				api.getBestParams(id, controller.signal).catch(() => null)
			]);
			if (controller.signal.aborted || runId !== id) return;
			artifactsState.runId = id;
			artifactsState.value = {
				resolvedConfig: configPayload,
				trials: trialsPayload,
				bestParams: bestParamsPayload
			};
			artifactsState.loaded = true;
			markTabFetchEnd(id, 'artifacts');
		} catch (err) {
			if (controller.signal.aborted) return;
			artifactsState.runId = id;
			artifactsState.value = null;
			artifactsState.error = err instanceof Error ? err.message : 'Failed to load artifacts.';
			artifactsState.loaded = true;
		} finally {
			if (!controller.signal.aborted && runId === id) {
				artifactsState.loading = false;
			}
			endSectionRequest('artifacts', controller);
		}
	}

	async function loadTimeline(id: string): Promise<void> {
		if (timelineState.loading && timelineState.runId === id) return;
		if (timelineState.loaded && timelineState.runId === id) return;
		const controller = beginSectionRequest('timeline');
		timelineState.loading = true;
		timelineState.error = null;
		markTabFetchStart(id, 'timeline');
		try {
			const payload = await api.getRunEvents(id, 50, controller.signal);
			if (controller.signal.aborted || runId !== id) return;
			timelineState.runId = id;
			timelineState.value = payload;
			timelineState.loaded = true;
			markTabFetchEnd(id, 'timeline');
		} catch (err) {
			if (controller.signal.aborted) return;
			timelineState.runId = id;
			timelineState.value = null;
			timelineState.error = err instanceof Error ? err.message : 'Failed to load timeline.';
			timelineState.loaded = true;
		} finally {
			if (!controller.signal.aborted && runId === id) {
				timelineState.loading = false;
			}
			endSectionRequest('timeline', controller);
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

	$effect(() => {
		const id = runId;
		const needsMetrics = !currentRun?.metrics || Object.keys(currentRun.metrics).length === 0;
		abortAllSections();
		mainTab = 'performance';
		artifactTab = 'config';
		copyState = 'idle';
		untrack(() => {
			resetState(manifestState);
			resetState(metricsState);
			resetState(performanceState, true);
			resetState(diagnosticsState);
			resetState(artifactsState);
			resetState(timelineState);
		});
		performance.mark(`run-detail:${id}:start`);
		void tick().then(() => measureShell(id));
		untrack(() => {
			void loadManifest(id);
			if (needsMetrics) {
				void loadMetrics(id);
			}
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
				void loadDiagnostics(id);
				return;
			}
			if (tab === 'artifacts') {
				void loadArtifacts(id);
				return;
			}
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

	{#if shellMetrics}
		<div class="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-7 gap-3" aria-label="Key metrics">
			{#each metricCards as card (card.key)}
				<div class="bg-card border border-border rounded-lg p-3.5">
					<p class="text-[11px] uppercase tracking-wider text-muted-foreground mb-1">{card.label}</p>
					<p class="text-2xl font-semibold tabular-nums">{fmt(card.value)}</p>
					<p class="mt-1 text-[11px] text-muted-foreground">
						{#if card.rank && card.total > 0}
							rank #{card.rank} / {card.total}
						{:else}
							rank n/a
						{/if}
					</p>
					<p class="text-[11px] {improvementClass(card.improvement)}">
						{#if card.isChampion}
							champion
						{:else}
							vs champion {signed(card.improvement)}
						{/if}
					</p>
				</div>
			{/each}
		</div>
	{:else if metricsState.loading}
		<div class="bg-card border border-border rounded-lg p-6 text-center text-muted-foreground text-sm">
			Loading metrics...
		</div>
	{:else}
		<div class="bg-card border border-border rounded-lg p-6 text-center text-muted-foreground text-sm">
			Metrics not available for this run.
		</div>
	{/if}

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
				{#if perEraCorr}
					<div class="bg-card border border-border rounded-lg p-5">
						<div class="flex items-center justify-between gap-2 mb-3">
							<h3 class="text-sm font-semibold">Cumulative Correlation</h3>
							<p class="text-xs text-muted-foreground">Primary trajectory view</p>
						</div>
						<CumulativeCorrChart data={perEraCorr} height="360px" />
					</div>
				{/if}

				<div class="grid grid-cols-1 xl:grid-cols-3 gap-5">
					<div class="xl:col-span-2 bg-card border border-border rounded-lg p-5">
						<div class="flex items-center justify-between gap-2 mb-3">
							<h3 class="text-sm font-semibold">Per-Era Correlation</h3>
							<p class="text-xs text-muted-foreground">Volatility + consistency</p>
						</div>
						{#if perEraCorr}
							<PerEraLineChart data={perEraCorr} height="300px" />
						{:else}
							<p class="text-sm text-muted-foreground">No per-era data available.</p>
						{/if}
					</div>

					<div class="bg-card border border-border rounded-lg p-5">
						<h3 class="text-sm font-semibold mb-3">Run Snapshot</h3>
						<dl class="space-y-2 text-sm">
							<div class="flex justify-between gap-3">
								<dt class="text-muted-foreground">Model</dt>
								<dd class="font-medium text-right">
									{manifest?.model_type ?? (typeof manifest?.model === 'object' ? manifest?.model?.type : manifest?.model) ?? currentRun?.model_type ?? 'N/A'}
								</dd>
							</div>
							<div class="flex justify-between gap-3">
								<dt class="text-muted-foreground">Target</dt>
								<dd class="font-medium text-right">
									{manifest?.data?.target_payout ?? manifest?.data?.target_train ?? manifest?.data?.target_col ?? manifest?.target ?? currentRun?.target_payout ?? currentRun?.target_train ?? currentRun?.target_col ?? currentRun?.target ?? 'N/A'}
								</dd>
							</div>
							<div class="flex justify-between gap-3">
								<dt class="text-muted-foreground">Feature Set</dt>
								<dd class="font-medium text-right">{manifest?.data?.feature_set ?? currentRun?.feature_set ?? 'N/A'}</dd>
							</div>
							<div class="flex justify-between gap-3">
								<dt class="text-muted-foreground">Created</dt>
								<dd class="font-medium text-right tabular-nums">{headerCreatedAt?.slice(0, 10) ?? 'N/A'}</dd>
							</div>
							<div class="flex justify-between gap-3">
								<dt class="text-muted-foreground">Champion</dt>
								<dd class="font-medium text-right font-mono text-xs">{championRun?.run_id ?? 'N/A'}</dd>
							</div>
							<div class="flex justify-between gap-3">
								<dt class="text-muted-foreground">Resource samples</dt>
								<dd class="font-medium text-right tabular-nums">{resources.length > 0 ? resources.length : '—'}</dd>
							</div>
						</dl>
					</div>
				</div>

				<div class="bg-card border border-border rounded-lg p-5">
					<div class="flex items-center justify-between gap-2 mb-3">
						<h3 class="text-sm font-semibold">Feature Importance</h3>
						<p class="text-xs text-muted-foreground">Top weighted features</p>
					</div>
					{#if featureImportance}
						<FeatureImportanceChart data={featureImportance} topN={15} />
					{:else}
						<p class="text-sm text-muted-foreground">Feature importance not available for this run.</p>
					{/if}
				</div>
			{/if}
		</div>
	{:else if mainTab === 'diagnostics'}
		<div class="grid grid-cols-1 xl:grid-cols-2 gap-6">
			<div class="bg-card border border-border rounded-lg p-5">
				<h3 class="text-sm font-semibold mb-3">Resource Utilization</h3>
				{#if diagnosticsState.loading}
					<p class="text-sm text-muted-foreground" aria-busy="true">Loading diagnostics...</p>
				{:else if diagnosticsState.error}
					<p class="text-sm text-negative">{diagnosticsState.error}</p>
				{:else if resources.length > 0}
					<div class="grid grid-cols-2 gap-3 text-xs mb-4">
						<div>
							<p class="text-muted-foreground">CPU Avg / Max</p>
							<p class="font-medium tabular-nums">{fmtPercent(resourceStats?.cpu_avg)} / {fmtPercent(resourceStats?.cpu_max)}</p>
						</div>
						<div>
							<p class="text-muted-foreground">GPU Avg / Max</p>
							<p class="font-medium tabular-nums">{fmtPercent(resourceStats?.gpu_avg)} / {fmtPercent(resourceStats?.gpu_max)}</p>
						</div>
						<div>
							<p class="text-muted-foreground">RAM Avg / Max</p>
							<p class="font-medium tabular-nums">{fmtGb(resourceStats?.ram_avg)} / {fmtGb(resourceStats?.ram_max)}</p>
						</div>
						<div>
							<p class="text-muted-foreground">Latest Sample</p>
							<p class="font-medium tabular-nums">{formatTime(latestResource?.created_at)}</p>
						</div>
					</div>
					<div class="overflow-x-auto max-h-[320px]" aria-label="Resource samples">
						<table class="w-full text-xs">
							<thead>
								<tr class="border-b border-border bg-muted/30">
									<th scope="col" class="text-left px-2 py-1 text-muted-foreground uppercase tracking-wider">Time</th>
									<th scope="col" class="text-right px-2 py-1 text-muted-foreground uppercase tracking-wider">CPU</th>
									<th scope="col" class="text-right px-2 py-1 text-muted-foreground uppercase tracking-wider">RAM</th>
									<th scope="col" class="text-right px-2 py-1 text-muted-foreground uppercase tracking-wider">GPU</th>
								</tr>
							</thead>
							<tbody class="divide-y divide-border">
								{#each resources.slice(-40).reverse() as sample (sample.id)}
									<tr>
										<td class="px-2 py-1 text-muted-foreground">{formatTime(sample.created_at)}</td>
										<td class="px-2 py-1 text-right tabular-nums">{fmtPercent(sample.cpu)}</td>
										<td class="px-2 py-1 text-right tabular-nums">{fmtGb(sample.ram)}</td>
										<td class="px-2 py-1 text-right tabular-nums">{fmtPercent(sample.gpu)}</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				{:else}
					<p class="text-sm text-muted-foreground">No resource samples recorded.</p>
				{/if}
			</div>

			<div class="bg-card border border-border rounded-lg p-5">
				<h3 class="text-sm font-semibold mb-3">Data + Manifest Snapshot</h3>
				{#if manifestState.loading}
					<p class="text-sm text-muted-foreground" aria-busy="true">Loading manifest...</p>
				{:else if manifestState.error}
					<p class="text-sm text-negative">{manifestState.error}</p>
				{:else if manifest}
					<pre class="bg-background rounded-lg p-4 overflow-x-auto text-xs font-mono max-h-[460px]"><code>{JSON.stringify(manifest, null, 2)}</code></pre>
				{:else}
					<p class="text-sm text-muted-foreground">Manifest not available.</p>
				{/if}
			</div>

			<div class="bg-card border border-border rounded-lg p-5 xl:col-span-2">
				<h3 class="text-sm font-semibold mb-3">Diagnostics Sources</h3>
				{#if diagnosticsState.loading}
					<p class="text-sm text-muted-foreground" aria-busy="true">Loading diagnostics source metadata...</p>
				{:else if diagnosticsSources}
					<pre class="bg-background rounded-lg p-4 overflow-x-auto text-xs font-mono max-h-[320px]"><code>{JSON.stringify(diagnosticsSources, null, 2)}</code></pre>
				{:else}
					<p class="text-sm text-muted-foreground">Diagnostics source metadata not available.</p>
				{/if}
			</div>
		</div>
	{:else if mainTab === 'timeline'}
		<div class="bg-card border border-border rounded-lg p-5">
			<h3 class="text-sm font-semibold mb-3">Run Timeline</h3>
			{#if timelineState.loading}
				<p class="text-sm text-muted-foreground" aria-busy="true">Loading timeline...</p>
			{:else if timelineState.error}
				<p class="text-sm text-negative">{timelineState.error}</p>
			{:else if events.length > 0}
				<div class="space-y-2 max-h-[540px] overflow-y-auto pr-1">
					{#each [...events].reverse() as event (event.id)}
						<div class="rounded-md border border-border bg-muted/20 px-3 py-2">
							<div class="flex items-center justify-between gap-3">
								<span class="text-xs font-medium text-foreground">{event.event_type}</span>
								<span class="text-[11px] text-muted-foreground">{formatTime(event.created_at)}</span>
							</div>
							{#if event.payload && Object.keys(event.payload).length > 0}
								<p class="mt-1 text-[11px] text-muted-foreground font-mono break-all">{formatPayload(event.payload)}</p>
							{/if}
						</div>
					{/each}
				</div>
			{:else}
				<p class="text-sm text-muted-foreground">No events recorded.</p>
			{/if}
		</div>
	{:else if mainTab === 'artifacts'}
		<div class="bg-card border border-border rounded-lg overflow-hidden">
			<div class="flex border-b border-border">
				{#each [
					{ key: 'config', label: 'Resolved Config' },
					{ key: 'trials', label: 'HPO Trials' },
					{ key: 'data', label: 'Manifest JSON' },
					{ key: 'notes', label: 'Run Notes' }
				] as tab (tab.key)}
					<button
						type="button"
						class="px-4 py-2.5 text-sm font-medium transition-colors {artifactTab === tab.key
							? 'border-b-2 border-primary text-foreground bg-muted/30'
							: 'text-muted-foreground hover:text-foreground hover:bg-muted/20'}"
						onclick={() => (artifactTab = tab.key as ArtifactTab)}
					>
						{tab.label}
					</button>
				{/each}
			</div>

			<div class="p-5">
				{#if artifactTab === 'config'}
					{#if artifactsState.loading}
						<p class="text-muted-foreground text-sm" aria-busy="true">Loading artifacts...</p>
					{:else if resolvedConfig?.yaml}
						<pre class="bg-background rounded-lg p-4 overflow-x-auto text-sm font-mono"><code>{resolvedConfig.yaml}</code></pre>
					{:else}
						<p class="text-muted-foreground text-sm">Resolved config not available.</p>
					{/if}
				{:else if artifactTab === 'trials'}
					{#if artifactsState.loading}
						<p class="text-muted-foreground text-sm" aria-busy="true">Loading artifacts...</p>
					{:else if trials && trials.length > 0}
						<div class="overflow-x-auto mb-4" aria-label="HPO trials table">
							<table class="w-full text-sm">
								<thead>
									<tr class="border-b border-border bg-muted/30">
										{#each Object.keys(trials[0]) as col (col)}
											<th scope="col" class="text-left px-4 py-2 font-medium text-xs uppercase tracking-wider text-muted-foreground">{col}</th>
										{/each}
									</tr>
								</thead>
								<tbody class="divide-y divide-border">
									{#each trials as trial, i (i)}
										<tr class="hover:bg-muted/20 transition-colors">
											{#each Object.values(trial) as val, j (j)}
												<td class="px-4 py-2 tabular-nums">{val}</td>
											{/each}
										</tr>
									{/each}
								</tbody>
							</table>
						</div>
						{#if bestParams}
							<h3 class="text-xs uppercase tracking-wider text-muted-foreground font-medium mb-2">Best Parameters</h3>
							<pre class="bg-background rounded-lg p-4 overflow-x-auto text-sm font-mono"><code>{JSON.stringify(bestParams, null, 2)}</code></pre>
						{/if}
					{:else}
						<p class="text-muted-foreground text-sm">No HPO trials available.</p>
					{/if}
				{:else if artifactTab === 'data'}
					{#if manifest}
						<pre class="bg-background rounded-lg p-4 overflow-x-auto text-sm font-mono"><code>{JSON.stringify(manifest, null, 2)}</code></pre>
					{:else if manifestState.loading}
						<p class="text-muted-foreground text-sm" aria-busy="true">Loading manifest...</p>
					{:else}
						<p class="text-muted-foreground text-sm">Manifest not available.</p>
					{/if}
				{:else if artifactTab === 'notes'}
					<MarkdownDoc
						label="RUN.md"
						load={() => api.getRunDoc(runId, 'RUN.md')}
						readOnly={readOnly}
						readOnlyMessage="Read-only mode: run notes editing is disabled."
					/>
				{/if}
			</div>
		</div>
	{/if}
</div>
