<script lang="ts">
	interface Component {
		run_id: string;
		weight: number | null;
		rank: number | null;
	}

	interface Props {
		components: Component[];
		class?: string;
	}

	let { components, class: className = '' }: Props = $props();

	let sorted = $derived(
		[...components]
			.filter((c) => c.weight != null && c.weight > 0)
			.sort((a, b) => (b.weight ?? 0) - (a.weight ?? 0))
	);

	let maxWeight = $derived(Math.max(...sorted.map((c) => c.weight ?? 0), 0.01));
</script>

<div class="w-full {className}" aria-label="Ensemble weights chart">
	{#if sorted.length === 0}
		<div class="flex h-full items-center justify-center text-muted-foreground text-sm">No components</div>
	{:else}
		<div class="space-y-2">
			{#each sorted as comp (comp.run_id)}
				<div class="flex items-center gap-3 text-xs">
					<span class="w-28 truncate font-mono text-[10px] text-muted-foreground" title={comp.run_id}>
						{comp.run_id.length > 18 ? comp.run_id.slice(0, 18) + '...' : comp.run_id}
					</span>
					<div class="flex-1 h-5 bg-muted/30 rounded-sm overflow-hidden">
						<div
							class="h-full w-full bg-primary/70 rounded-sm origin-left transition-transform"
							style:transform="scaleX({(comp.weight ?? 0) / maxWeight})"
						></div>
					</div>
					<span class="w-12 text-right tabular-nums font-medium">
						{((comp.weight ?? 0) * 100).toFixed(1)}%
					</span>
				</div>
			{/each}
		</div>
	{/if}
</div>
