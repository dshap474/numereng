<script lang="ts">
	import type { DiagnosticsSources, ResourceSample, RunManifest } from '$lib/api/client';
	import { fmtGb, fmtPercent } from '$lib/utils';

	let {
		loading = false,
		error = null,
		resources = [],
		resourceStats = null,
		latestResource = null,
		manifest = null,
		manifestLoading = false,
		manifestError = null,
		diagnosticsSources = null
	}: {
		loading?: boolean;
		error?: string | null;
		resources?: ResourceSample[];
		resourceStats?: {
			cpu_avg: number | null;
			cpu_max: number | null;
			ram_avg: number | null;
			ram_max: number | null;
			gpu_avg: number | null;
			gpu_max: number | null;
		} | null;
		latestResource?: ResourceSample | null;
		manifest?: RunManifest | null;
		manifestLoading?: boolean;
		manifestError?: string | null;
		diagnosticsSources?: DiagnosticsSources | null;
	} = $props();

	function formatTime(value: string | undefined | null): string {
		if (!value) return '—';
		const date = new Date(value);
		if (Number.isNaN(date.getTime())) return value;
		return date.toLocaleString();
	}
</script>

<div class="grid grid-cols-1 xl:grid-cols-2 gap-6">
	<div class="bg-card border border-border rounded-lg p-5">
		<h3 class="text-sm font-semibold mb-3">Resource Utilization</h3>
		{#if loading}
			<p class="text-sm text-muted-foreground" aria-busy="true">Loading diagnostics...</p>
		{:else if error}
			<p class="text-sm text-negative">{error}</p>
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
		{#if manifestLoading}
			<p class="text-sm text-muted-foreground" aria-busy="true">Loading manifest...</p>
		{:else if manifestError}
			<p class="text-sm text-negative">{manifestError}</p>
		{:else if manifest}
			<pre class="bg-background rounded-lg p-4 overflow-x-auto text-xs font-mono max-h-[460px]"><code>{JSON.stringify(manifest, null, 2)}</code></pre>
		{:else}
			<p class="text-sm text-muted-foreground">Manifest not available.</p>
		{/if}
	</div>

	<div class="bg-card border border-border rounded-lg p-5 xl:col-span-2">
		<h3 class="text-sm font-semibold mb-3">Diagnostics Sources</h3>
		{#if loading}
			<p class="text-sm text-muted-foreground" aria-busy="true">Loading diagnostics source metadata...</p>
		{:else if diagnosticsSources}
			<pre class="bg-background rounded-lg p-4 overflow-x-auto text-xs font-mono max-h-[320px]"><code>{JSON.stringify(diagnosticsSources, null, 2)}</code></pre>
		{:else}
			<p class="text-sm text-muted-foreground">Diagnostics source metadata not available.</p>
		{/if}
	</div>
</div>
