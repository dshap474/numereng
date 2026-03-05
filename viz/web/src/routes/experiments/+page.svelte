<script lang="ts">
	import { badgeClass } from '$lib/utils';

	let { data } = $props();

	let total = $derived(data.experiments.length);
	let active = $derived(data.experiments.filter((e: any) => e.status === 'active').length);
	let complete = $derived(data.experiments.filter((e: any) => e.status === 'complete').length);

	function runCount(exp: any): number {
		if (typeof exp.run_count === 'number') return exp.run_count;
		if (Array.isArray(exp.runs)) return exp.runs.length;
		return 0;
	}
</script>

<div>
	<h1 class="text-xl font-semibold mb-6">Experiments</h1>

	<div class="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
		<div class="bg-card border border-border rounded-lg p-5">
			<p class="text-xs uppercase tracking-wider text-muted-foreground mb-1">Total</p>
			<p class="text-3xl font-medium tabular-nums">{total}</p>
		</div>
		<div class="bg-card border border-border rounded-lg p-5">
			<p class="text-xs uppercase tracking-wider text-muted-foreground mb-1">Active</p>
			<p class="text-3xl font-medium tabular-nums text-positive">{active}</p>
		</div>
		<div class="bg-card border border-border rounded-lg p-5">
			<p class="text-xs uppercase tracking-wider text-muted-foreground mb-1">Complete</p>
			<p class="text-3xl font-medium tabular-nums text-positive">{complete}</p>
		</div>
	</div>

	<div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4" aria-label="Experiments">
		{#each data.experiments as exp (exp.experiment_id)}
			<a
				href="/experiments/{exp.experiment_id}"
				class="bg-card border border-border rounded-lg p-5 min-h-[140px] flex flex-col hover:bg-muted/30 transition-colors"
			>
				<div class="flex items-start justify-between mb-3">
					<span class="text-[11px] text-muted-foreground tabular-nums">{exp.created_at.slice(0, 10)}</span>
					<span class="inline-block px-2 py-0.5 rounded-full text-[10px] font-medium {badgeClass(exp.status)}">
						{exp.status}
					</span>
				</div>
				<h3 class="text-sm font-semibold mb-auto">{exp.name}</h3>
				<div class="flex items-center gap-4 mt-3 text-[11px] text-muted-foreground">
					<span class="tabular-nums">{runCount(exp)} runs</span>
					{#if exp.tags.length > 0}
						<span>{exp.tags.join(', ')}</span>
					{/if}
				</div>
			</a>
		{/each}
	</div>
</div>
