<script lang="ts">
	import { api, type RunManifest } from '$lib/api/client';
	import MarkdownDoc from '$lib/components/ui/MarkdownDoc.svelte';
	import type { SourceContext } from '$lib/source';

	type ArtifactTab = 'config' | 'trials' | 'data' | 'notes';

	let {
		runId,
		source,
		readOnly = false,
		loading = false,
		resolvedConfig = null,
		trials = null,
		bestParams = null,
		manifest = null,
		manifestLoading = false
	}: {
		runId: string;
		source?: SourceContext;
		readOnly?: boolean;
		loading?: boolean;
		resolvedConfig?: { yaml: string } | null;
		trials?: Record<string, unknown>[] | null;
		bestParams?: Record<string, unknown> | null;
		manifest?: RunManifest | null;
		manifestLoading?: boolean;
	} = $props();

	let artifactTab = $state<ArtifactTab>('config');
</script>

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
			{#if loading}
				<p class="text-muted-foreground text-sm" aria-busy="true">Loading artifacts...</p>
			{:else if resolvedConfig?.yaml}
				<pre class="bg-background rounded-lg p-4 overflow-x-auto text-sm font-mono"><code>{resolvedConfig.yaml}</code></pre>
			{:else}
				<p class="text-muted-foreground text-sm">Resolved config not available.</p>
			{/if}
		{:else if artifactTab === 'trials'}
			{#if loading}
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
			{:else if manifestLoading}
				<p class="text-muted-foreground text-sm" aria-busy="true">Loading manifest...</p>
			{:else}
				<p class="text-muted-foreground text-sm">Manifest not available.</p>
			{/if}
		{:else if artifactTab === 'notes'}
			<MarkdownDoc
				label="RUN.md"
				load={() => api.getRunDoc(runId, 'RUN.md', source)}
				readOnly={readOnly}
				readOnlyMessage="Read-only mode: run notes editing is disabled."
			/>
		{/if}
	</div>
</div>
