<script lang="ts">
	import { api, type ExperimentRun, type RunBundle } from '$lib/api/client';
	import PerEraLineChart from '$lib/components/charts/PerEraLineChart.svelte';
	import CumulativeCorrChart from '$lib/components/charts/CumulativeCorrChart.svelte';
	import FeatureImportanceChart from '$lib/components/charts/FeatureImportanceChart.svelte';
	import MarkdownDoc from '$lib/components/ui/MarkdownDoc.svelte';
	import { fmt, fmtPercent, fmtGb } from '$lib/utils';

	let {
		runId,
		experimentId,
		runs,
		readOnly = false,
		onClose
	}: {
		runId: string;
		experimentId: string;
		runs: ExperimentRun[];
		readOnly?: boolean;
		onClose: () => void;
	} = $props();

	type MainTab = 'performance' | 'diagnostics' | 'artifacts' | 'timeline';
	type ArtifactTab = 'config' | 'trials' | 'data' | 'notes';
	type MetricKey =
		| 'corr20v2_sharpe'
		| 'corr20v2_mean'
		| 'mmc_mean'
		| 'payout_estimate_mean'
		| 'mmc_coverage_ratio_rows'
		| 'bmc_mean'
		| 'max_drawdown';

	let mainTab = $state<MainTab>('performance');
	let artifactTab = $state<ArtifactTab>('config');
	let loading = $state(true);
	let error = $state<string | null>(null);
	let bundle = $state<RunBundle | null>(null);

	const keyMetrics: MetricKey[] = [
		'corr20v2_sharpe',
		'corr20v2_mean',
		'mmc_mean',
		'payout_estimate_mean',
		'mmc_coverage_ratio_rows',
		'bmc_mean',
		'max_drawdown'
	];

	const lowerBetter = new Set<MetricKey>(['max_drawdown']);

	$effect(() => {
		const id = runId;
		loading = true;
		error = null;
		bundle = null;
		mainTab = 'performance';
		api.getRunBundle(id).then(
			(data) => {
				if (runId !== id) return;
				bundle = data;
				loading = false;
			},
			(err) => {
				if (runId !== id) return;
				error = err instanceof Error ? err.message : 'Failed to load run bundle.';
				loading = false;
			}
		);
	});

	let metrics = $derived(bundle?.metrics ?? null);
	let manifest = $derived(bundle?.manifest ?? null);
	let perEraCorr = $derived(bundle?.per_era_corr ?? null);
	let featureImportance = $derived(bundle?.feature_importance ?? null);
	let trials = $derived(bundle?.trials ?? null);
	let bestParams = $derived(bundle?.best_params ?? null);
	let resolvedConfig = $derived(bundle?.resolved_config ?? null);
	let diagnosticsSources = $derived(bundle?.diagnostics_sources ?? null);
	let events = $derived(bundle?.events ?? []);
	let resources = $derived(bundle?.resources ?? []);

	let currentRun = $derived.by(() => runs.find((run) => run.run_id === runId) ?? null);
	let championRun = $derived.by(() => runs.find((run) => run.is_champion) ?? null);

	let metricCards = $derived.by(() => {
		return keyMetrics.map((key) => {
			const value = metricValue(currentRun, key) ?? metrics?.[key] ?? null;
			const values = runs
				.map((run) => metricValue(run, key))
				.filter((candidate): candidate is number => candidate != null && Number.isFinite(candidate));
			const total = values.length;
			let rank: number | null = null;
			if (value != null && total > 0) {
				const sorted = [...values].sort((a, b) =>
					lowerBetter.has(key) ? a - b : b - a
				);
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
		const cpu = resources.map((s) => s.cpu).filter((v): v is number => typeof v === 'number');
		const ram = resources.map((s) => s.ram).filter((v): v is number => typeof v === 'number');
		const gpu = resources.map((s) => s.gpu).filter((v): v is number => typeof v === 'number');
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
			case 'corr20v2_sharpe':
				return 'CORR20v2 Sharpe';
			case 'corr20v2_mean':
				return 'CORR20v2 Mean';
			case 'mmc_mean':
				return 'MMC Mean';
			case 'payout_estimate_mean':
				return 'Payout Estimate Mean';
			case 'mmc_coverage_ratio_rows':
				return 'MMC Coverage Ratio';
			case 'bmc_mean':
				return 'BMC Mean (Diagnostic)';
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

	function formatTime(value: string | undefined): string {
		if (!value) return '\u2014';
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

</script>

{#if loading}
	<div class="flex items-center justify-center h-full text-muted-foreground text-sm">
		Loading run details...
	</div>
{:else if error}
	<div class="flex flex-col items-center justify-center h-full gap-3">
		<p class="text-sm text-negative">{error}</p>
		<button
			type="button"
			class="text-sm text-muted-foreground hover:text-foreground underline underline-offset-2"
			onclick={onClose}
		>Back to runs</button>
	</div>
{:else}
	<div class="space-y-6 min-w-0 px-4 pb-6 lg:px-5">
		<header class="sticky top-0 z-20 py-3 bg-background/90 backdrop-blur border-b border-border/60">
			<div class="flex items-start gap-3">
				<div>
					<h1 class="text-xl font-semibold">
						{manifest?.run_name ?? runId}
					</h1>
					<div class="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
						<span class="font-mono">{runId}</span>
						<span>&bull;</span>
						<span>{experimentId}</span>
						{#if manifest?.created_at}
							<span>&bull;</span>
							<span>{formatTime(manifest.created_at)}</span>
						{/if}
					</div>
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
			<div class="space-y-5">
				{#if metrics}
					<div class="grid grid-cols-1 gap-3 xl:grid-cols-7" aria-label="Key metrics">
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
				{:else}
					<div class="bg-card border border-border rounded-lg p-6 text-center text-muted-foreground text-sm">
						Metrics not available for this run.
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
								<dd class="font-medium text-right">{manifest?.model_type ?? (typeof manifest?.model === 'object' ? manifest?.model?.type : manifest?.model) ?? 'N/A'}</dd>
							</div>
								<div class="flex justify-between gap-3">
									<dt class="text-muted-foreground">Target</dt>
									<dd class="font-medium text-right">{manifest?.data?.target_payout ?? manifest?.data?.target_train ?? manifest?.data?.target_col ?? manifest?.target ?? 'N/A'}</dd>
								</div>
							<div class="flex justify-between gap-3">
								<dt class="text-muted-foreground">Feature Set</dt>
								<dd class="font-medium text-right">{manifest?.data?.feature_set ?? 'N/A'}</dd>
							</div>
							<div class="flex justify-between gap-3">
								<dt class="text-muted-foreground">Created</dt>
								<dd class="font-medium text-right tabular-nums">{manifest?.created_at?.slice(0, 10) ?? 'N/A'}</dd>
							</div>
							<div class="flex justify-between gap-3">
								<dt class="text-muted-foreground">Champion</dt>
								<dd class="font-medium text-right font-mono text-xs">{championRun?.run_id ?? 'N/A'}</dd>
							</div>
							<div class="flex justify-between gap-3">
								<dt class="text-muted-foreground">Resource samples</dt>
								<dd class="font-medium text-right tabular-nums">{resources.length}</dd>
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
			</div>
	{:else if mainTab === 'diagnostics'}
		<div class="grid grid-cols-1 xl:grid-cols-2 gap-6">
				<div class="bg-card border border-border rounded-lg p-5">
					<h3 class="text-sm font-semibold mb-3">Resource Utilization</h3>
					{#if resources.length > 0}
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
				{#if manifest}
					<pre class="bg-background rounded-lg p-4 overflow-x-auto text-xs font-mono max-h-[460px]"><code>{JSON.stringify(manifest, null, 2)}</code></pre>
				{:else}
					<p class="text-sm text-muted-foreground">Manifest not available.</p>
				{/if}
			</div>

			<div class="bg-card border border-border rounded-lg p-5 xl:col-span-2">
				<h3 class="text-sm font-semibold mb-3">Diagnostics Sources</h3>
				{#if diagnosticsSources}
					<pre class="bg-background rounded-lg p-4 overflow-x-auto text-xs font-mono max-h-[320px]"><code>{JSON.stringify(diagnosticsSources, null, 2)}</code></pre>
				{:else}
					<p class="text-sm text-muted-foreground">Diagnostics source metadata not available.</p>
				{/if}
			</div>
		</div>
		{:else if mainTab === 'timeline'}
			<div class="bg-card border border-border rounded-lg p-5">
				<h3 class="text-sm font-semibold mb-3">Run Timeline</h3>
				{#if events.length > 0}
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
						{#if resolvedConfig?.yaml}
							<pre class="bg-background rounded-lg p-4 overflow-x-auto text-sm font-mono"><code>{resolvedConfig.yaml}</code></pre>
						{:else}
							<p class="text-muted-foreground text-sm">Resolved config not available.</p>
						{/if}
					{:else if artifactTab === 'trials'}
						{#if trials && trials.length > 0}
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
{/if}
