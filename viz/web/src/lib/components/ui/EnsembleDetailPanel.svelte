<script lang="ts">
	import {
		api,
		type Ensemble,
		type CorrelationMatrix,
		type EnsembleArtifactRow,
		type EnsembleArtifacts
	} from '$lib/api/client';
	import EnsembleWeightsChart from '$lib/components/charts/EnsembleWeightsChart.svelte';
	import CorrelationHeatmap from '$lib/components/charts/CorrelationHeatmap.svelte';
	import { fmt } from '$lib/utils';

	let {
		ensembleId,
		experimentId,
		onClose
	}: {
		ensembleId: string;
		experimentId: string;
		onClose: () => void;
	} = $props();

	let loading = $state(true);
	let error = $state<string | null>(null);
	let ensemble = $state<Ensemble | null>(null);
	let correlations = $state<CorrelationMatrix>({ labels: [], matrix: [] });
	let artifacts = $state<EnsembleArtifacts | null>(null);

	$effect(() => {
		const id = ensembleId;
		loading = true;
		error = null;
		ensemble = null;
		correlations = { labels: [], matrix: [] };
		artifacts = null;
		Promise.all([
			api.getEnsemble(id),
			api.getEnsembleCorrelations(id).catch(() => ({ labels: [], matrix: [] }) as CorrelationMatrix),
			api.getEnsembleArtifacts(id).catch(() => null)
		]).then(
			([e, c, a]) => {
				if (ensembleId !== id) return;
				ensemble = e;
				correlations = c;
				artifacts = a;
				loading = false;
			},
			(err) => {
				if (ensembleId !== id) return;
				error = err instanceof Error ? err.message : 'Failed to load ensemble.';
				loading = false;
			}
		);
	});

	let components = $derived(ensemble?.components ?? []);
	let metrics = $derived(ensemble?.metrics ?? {});
	let componentMetricsRows = $derived(artifacts?.component_metrics ?? []);
	let eraMetricsRows = $derived(artifacts?.era_metrics ?? []);
	let regimeMetricsRows = $derived(artifacts?.regime_metrics ?? []);

	function artifactColumns(rows: EnsembleArtifactRow[]): string[] {
		if (rows.length === 0) return [];
		const seen = new Set<string>();
		const ordered: string[] = [];
		for (const row of rows) {
			for (const key of Object.keys(row)) {
				if (seen.has(key)) continue;
				seen.add(key);
				ordered.push(key);
			}
		}
		return ordered;
	}

	function artifactValue(value: string | number | boolean | null): string {
		if (value === null) return '-';
		if (typeof value === 'number') return fmt(value);
		if (typeof value === 'boolean') return value ? 'true' : 'false';
		return value;
	}

	function statusClass(status: string | null): string {
		switch (status) {
			case 'completed': return 'bg-positive/15 text-positive';
			case 'building': return 'bg-blue-500/15 text-blue-400';
			case 'failed': return 'bg-negative/15 text-negative';
			default: return 'bg-muted text-muted-foreground';
		}
	}
</script>

{#if loading}
	<div class="flex items-center justify-center h-full text-muted-foreground text-sm">
		Loading ensemble details...
	</div>
{:else if error}
	<div class="flex flex-col items-center justify-center h-full gap-3">
		<p class="text-sm text-negative">{error}</p>
		<button
			type="button"
			class="text-sm text-muted-foreground hover:text-foreground underline underline-offset-2"
			onclick={onClose}
		>&larr; Back</button>
	</div>
{:else if ensemble}
	<div class="space-y-6 min-w-0">
		<header class="sticky top-0 z-20 -mx-2 px-2 py-3 bg-background/90 backdrop-blur border-b border-border/60">
			<div class="flex items-start justify-between gap-3">
				<div>
					<div class="flex items-center gap-2">
						<span class="inline-block rounded px-1.5 py-0.5 text-[10px] uppercase bg-amber-500/20 text-amber-300">Ensemble</span>
						<h1 class="text-xl font-semibold">{ensemble.name || ensemble.ensemble_id}</h1>
						<span class="inline-block rounded px-2 py-0.5 text-xs uppercase {statusClass(ensemble.status)}">
							{ensemble.status ?? 'unknown'}
						</span>
					</div>
					<div class="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
						<span class="font-mono">{ensemble.ensemble_id}</span>
					</div>
				</div>
				<button
					type="button"
					class="rounded-md border border-border px-2.5 py-1.5 text-xs text-muted-foreground hover:text-foreground hover:bg-muted/30 whitespace-nowrap"
					onclick={onClose}
				>&larr; Back</button>
			</div>
		</header>

		<!-- Summary cards -->
		<div class="grid grid-cols-2 md:grid-cols-4 gap-3">
			<div class="rounded-lg border border-border bg-card px-4 py-3">
				<div class="text-xs text-muted-foreground uppercase">Method</div>
				<div class="mt-1 font-medium">{ensemble.method ?? '-'}</div>
			</div>
			<div class="rounded-lg border border-border bg-card px-4 py-3">
				<div class="text-xs text-muted-foreground uppercase">Components</div>
				<div class="mt-1 font-medium">{components.length}</div>
			</div>
			{#each Object.entries(metrics).slice(0, 2) as [key, val] (key)}
				<div class="rounded-lg border border-border bg-card px-4 py-3">
					<div class="text-xs text-muted-foreground uppercase">{key}</div>
					<div class="mt-1 font-medium tabular-nums">{fmt(val)}</div>
				</div>
			{/each}
		</div>

		<!-- All metrics -->
		{#if Object.keys(metrics).length > 2}
			<div class="rounded-lg border border-border bg-card p-4">
				<h2 class="text-sm font-medium mb-3">Ensemble Metrics</h2>
				<div class="grid grid-cols-2 md:grid-cols-4 gap-2">
					{#each Object.entries(metrics) as [key, val] (key)}
						<div class="rounded-md border border-border/50 bg-background/50 px-3 py-2">
							<div class="text-[10px] text-muted-foreground">{key}</div>
							<div class="text-xs font-mono tabular-nums mt-0.5">{fmt(val)}</div>
						</div>
					{/each}
				</div>
			</div>
		{/if}

		<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
			<!-- Weights chart -->
			{#if components.length > 0}
				<div class="rounded-lg border border-border bg-card p-4">
					<h2 class="text-sm font-medium mb-3">Component Weights</h2>
					<EnsembleWeightsChart {components} />
				</div>
			{/if}

			<!-- Correlation heatmap -->
			{#if correlations.labels.length > 0}
				<div class="rounded-lg border border-border bg-card p-4">
					<h2 class="text-sm font-medium mb-3">Correlation Matrix</h2>
					<CorrelationHeatmap labels={correlations.labels} matrix={correlations.matrix} />
				</div>
			{/if}
		</div>

		<!-- Components table -->
		{#if components.length > 0}
			<div class="rounded-lg border border-border bg-card overflow-hidden">
				<h2 class="text-sm font-medium px-4 py-3 border-b border-border">Components ({components.length})</h2>
				<div class="overflow-auto">
					<table class="w-full text-xs">
						<thead>
							<tr class="border-b border-border text-left">
								<th class="px-3 py-2 text-muted-foreground font-medium">Rank</th>
								<th class="px-3 py-2 text-muted-foreground font-medium">Run ID</th>
								<th class="px-3 py-2 text-muted-foreground font-medium text-right">Weight</th>
							</tr>
						</thead>
						<tbody class="divide-y divide-border/50">
							{#each [...components].sort((a, b) => (a.rank ?? 99) - (b.rank ?? 99)) as comp (comp.run_id)}
								<tr class="hover:bg-muted/20">
									<td class="px-3 py-1.5 tabular-nums">{comp.rank ?? '-'}</td>
									<td class="px-3 py-1.5">
										<a
											href="/experiments/{experimentId}/runs/{comp.run_id}"
											class="text-primary underline underline-offset-2 font-mono text-[11px]"
										>{comp.run_id}</a>
									</td>
									<td class="px-3 py-1.5 text-right tabular-nums font-medium">
										{comp.weight != null ? `${(comp.weight * 100).toFixed(1)}%` : '-'}
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</div>
		{/if}

		{#if artifacts}
			<div class="rounded-lg border border-border bg-card p-4">
				<h2 class="text-sm font-medium mb-2">Artifact Inventory</h2>
				<p class="text-xs text-muted-foreground">
					Heavy component predictions: {artifacts.heavy_component_predictions_available ? 'available' : 'not present'}
				</p>
				{#if artifacts.available_files.length > 0}
					<div class="mt-2 flex flex-wrap gap-1.5">
						{#each artifacts.available_files as filename (filename)}
							<span class="rounded border border-border/60 px-2 py-0.5 text-[10px] font-mono">{filename}</span>
						{/each}
					</div>
				{/if}
			</div>
		{/if}

		{#if componentMetricsRows.length > 0}
			<div class="rounded-lg border border-border bg-card overflow-auto">
				<h2 class="text-sm font-medium px-4 py-3 border-b border-border">Component Metrics</h2>
				<table class="w-full text-xs">
					<thead>
						<tr class="border-b border-border text-left">
							{#each artifactColumns(componentMetricsRows) as col (col)}
								<th class="px-3 py-2 text-muted-foreground font-medium">{col}</th>
							{/each}
						</tr>
					</thead>
					<tbody class="divide-y divide-border/50">
						{#each componentMetricsRows as row, idx (`cmp-${idx}`)}
							<tr class="hover:bg-muted/20">
								{#each artifactColumns(componentMetricsRows) as col (`cmp-col-${idx}-${col}`)}
									<td class="px-3 py-1.5 tabular-nums">{artifactValue(row[col] ?? null)}</td>
								{/each}
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}

		<div class="grid grid-cols-1 xl:grid-cols-2 gap-6">
			{#if eraMetricsRows.length > 0}
				<div class="rounded-lg border border-border bg-card overflow-auto">
					<h2 class="text-sm font-medium px-4 py-3 border-b border-border">Era Metrics</h2>
					<table class="w-full text-xs">
						<thead>
							<tr class="border-b border-border text-left">
								{#each artifactColumns(eraMetricsRows) as col (col)}
									<th class="px-3 py-2 text-muted-foreground font-medium">{col}</th>
								{/each}
							</tr>
						</thead>
						<tbody class="divide-y divide-border/50">
							{#each eraMetricsRows as row, idx (`era-${idx}`)}
								<tr class="hover:bg-muted/20">
									{#each artifactColumns(eraMetricsRows) as col (`era-col-${idx}-${col}`)}
										<td class="px-3 py-1.5 tabular-nums">{artifactValue(row[col] ?? null)}</td>
									{/each}
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{/if}

			{#if regimeMetricsRows.length > 0}
				<div class="rounded-lg border border-border bg-card overflow-auto">
					<h2 class="text-sm font-medium px-4 py-3 border-b border-border">Regime Metrics</h2>
					<table class="w-full text-xs">
						<thead>
							<tr class="border-b border-border text-left">
								{#each artifactColumns(regimeMetricsRows) as col (col)}
									<th class="px-3 py-2 text-muted-foreground font-medium">{col}</th>
								{/each}
							</tr>
						</thead>
						<tbody class="divide-y divide-border/50">
							{#each regimeMetricsRows as row, idx (`regime-${idx}`)}
								<tr class="hover:bg-muted/20">
									{#each artifactColumns(regimeMetricsRows) as col (`regime-col-${idx}-${col}`)}
										<td class="px-3 py-1.5 tabular-nums">{artifactValue(row[col] ?? null)}</td>
									{/each}
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{/if}
		</div>

		<div class="grid grid-cols-1 xl:grid-cols-2 gap-6">
			{#if artifacts?.lineage}
				<div class="rounded-lg border border-border bg-card p-4 overflow-auto">
					<h2 class="text-sm font-medium mb-2">Lineage</h2>
					<pre class="text-[11px] whitespace-pre-wrap break-all font-mono">{JSON.stringify(artifacts.lineage, null, 2)}</pre>
				</div>
			{/if}

			{#if artifacts?.bootstrap_metrics}
				<div class="rounded-lg border border-border bg-card p-4 overflow-auto">
					<h2 class="text-sm font-medium mb-2">Bootstrap Metrics</h2>
					<pre class="text-[11px] whitespace-pre-wrap break-all font-mono">{JSON.stringify(artifacts.bootstrap_metrics, null, 2)}</pre>
				</div>
			{/if}
		</div>
	</div>
{/if}
