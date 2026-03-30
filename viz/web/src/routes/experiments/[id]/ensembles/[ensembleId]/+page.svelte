<script lang="ts">
	import type {
		Ensemble,
		CorrelationMatrix,
		EnsembleArtifactRow,
		EnsembleArtifacts
	} from '$lib/api/client';
	import EnsembleWeightsChart from '$lib/components/charts/EnsembleWeightsChart.svelte';
	import CorrelationHeatmap from '$lib/components/charts/CorrelationHeatmap.svelte';
	import { withSourceHref, type SourceContext } from '$lib/source';
	import { fmt } from '$lib/utils';

	let {
		data
	}: {
		data: {
			experimentId: string;
			ensemble: Ensemble;
			correlations: CorrelationMatrix;
			artifacts: EnsembleArtifacts | null;
			source: SourceContext;
		};
	} = $props();

	let components = $derived(data.ensemble.components ?? []);
	let metrics = $derived(data.ensemble.metrics ?? {});
	let componentMetricsRows = $derived(data.artifacts?.component_metrics ?? []);
	let eraMetricsRows = $derived(data.artifacts?.era_metrics ?? []);
	let regimeMetricsRows = $derived(data.artifacts?.regime_metrics ?? []);

	function statusClass(status: string | null): string {
		switch (status) {
			case 'completed': return 'bg-positive/15 text-positive';
			case 'building': return 'bg-blue-500/15 text-blue-400';
			case 'failed': return 'bg-negative/15 text-negative';
			default: return 'bg-muted text-muted-foreground';
		}
	}

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
</script>

<div class="space-y-6">
	<div class="flex items-center gap-3">
		<a
			href={withSourceHref(`/experiments/${data.experimentId}`, data.source)}
			class="text-sm text-muted-foreground hover:text-foreground transition-colors"
		>&larr; Experiment</a>
		<h1 class="text-xl font-semibold">{data.ensemble.name || data.ensemble.ensemble_id}</h1>
		<span class="inline-block rounded px-2 py-0.5 text-xs uppercase {statusClass(data.ensemble.status)}">
			{data.ensemble.status ?? 'unknown'}
		</span>
	</div>

	<div class="grid grid-cols-2 md:grid-cols-4 gap-3">
		<div class="rounded-lg border border-border bg-card px-4 py-3">
			<div class="text-xs text-muted-foreground uppercase">Method</div>
			<div class="mt-1 font-medium">{data.ensemble.method ?? '-'}</div>
		</div>
		<div class="rounded-lg border border-border bg-card px-4 py-3">
			<div class="text-xs text-muted-foreground uppercase">Components</div>
			<div class="mt-1 font-medium">{components.length}</div>
		</div>
		{#each Object.entries(metrics).slice(0, 2) as [key, val]}
			<div class="rounded-lg border border-border bg-card px-4 py-3">
				<div class="text-xs text-muted-foreground uppercase">{key}</div>
				<div class="mt-1 font-medium tabular-nums">{fmt(val)}</div>
			</div>
		{/each}
	</div>

	{#if Object.keys(metrics).length > 2}
		<div class="rounded-lg border border-border bg-card p-4">
			<h2 class="text-sm font-medium mb-3">Ensemble Metrics</h2>
			<div class="grid grid-cols-2 md:grid-cols-4 gap-2">
				{#each Object.entries(metrics) as [key, val]}
					<div class="rounded-md border border-border/50 bg-background/50 px-3 py-2">
						<div class="text-[10px] text-muted-foreground">{key}</div>
						<div class="text-xs font-mono tabular-nums mt-0.5">{fmt(val)}</div>
					</div>
				{/each}
			</div>
		</div>
	{/if}

	<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
		{#if components.length > 0}
			<div class="rounded-lg border border-border bg-card p-4">
				<h2 class="text-sm font-medium mb-3">Component Weights</h2>
				<EnsembleWeightsChart {components} />
			</div>
		{/if}

		{#if data.correlations.labels.length > 0}
			<div class="rounded-lg border border-border bg-card p-4">
				<h2 class="text-sm font-medium mb-3">Correlation Matrix</h2>
				<CorrelationHeatmap labels={data.correlations.labels} matrix={data.correlations.matrix} />
			</div>
		{/if}
	</div>

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
						{#each [...components].sort((a, b) => (a.rank ?? 99) - (b.rank ?? 99)) as comp}
							<tr class="hover:bg-muted/20">
								<td class="px-3 py-1.5 tabular-nums">{comp.rank ?? '-'}</td>
								<td class="px-3 py-1.5">
									<a
										href={withSourceHref(`/experiments/${data.experimentId}/runs/${comp.run_id}`, data.source)}
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

	{#if data.artifacts}
		<div class="rounded-lg border border-border bg-card p-4">
			<h2 class="text-sm font-medium mb-2">Artifact Inventory</h2>
			<p class="text-xs text-muted-foreground">
				Heavy component predictions: {data.artifacts.heavy_component_predictions_available ? 'available' : 'not present'}
			</p>
			{#if data.artifacts.available_files.length > 0}
				<div class="mt-2 flex flex-wrap gap-1.5">
					{#each data.artifacts.available_files as filename (filename)}
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
		{#if data.artifacts?.lineage}
			<div class="rounded-lg border border-border bg-card p-4 overflow-auto">
				<h2 class="text-sm font-medium mb-2">Lineage</h2>
				<pre class="text-[11px] whitespace-pre-wrap break-all font-mono">{JSON.stringify(data.artifacts.lineage, null, 2)}</pre>
			</div>
		{/if}

		{#if data.artifacts?.bootstrap_metrics}
			<div class="rounded-lg border border-border bg-card p-4 overflow-auto">
				<h2 class="text-sm font-medium mb-2">Bootstrap Metrics</h2>
				<pre class="text-[11px] whitespace-pre-wrap break-all font-mono">{JSON.stringify(data.artifacts.bootstrap_metrics, null, 2)}</pre>
			</div>
		{/if}
	</div>
</div>
