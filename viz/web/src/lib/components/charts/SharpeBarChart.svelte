<script lang="ts">
	import { BarChart } from 'layerchart';

	interface Run {
		run_id: string;
		run_name?: string;
		composite: number;
		is_champion: boolean;
	}

	interface Props {
		runs: Run[];
		championRunId: string | null;
	}

	let { runs, championRunId }: Props = $props();

	let sortedRuns = $derived(
		[...runs]
			.sort((a, b) => b.composite - a.composite)
			.map((r) => ({
				...r,
				label: r.run_name || r.run_id,
				color: r.is_champion ? 'var(--color-chart-bar-champion)' : 'var(--color-chart-bar)'
			}))
	);
	let chartHeight = $derived(Math.max(runs.length * 30, 100));
</script>

<div class="overflow-visible" style:height="{chartHeight}px" aria-label="Composite score bar chart">
	{#if runs.length > 0}
		<BarChart
			data={sortedRuns}
			x="composite"
			y="label"
			orientation="horizontal"
			axis
			tooltip
			grid={false}
			padding={{ left: 160 }}
			series={[{ key: 'default', value: 'composite', color: 'var(--color-chart-bar)' }]}
		/>
	{:else}
		<div class="flex h-full items-center justify-center text-muted-foreground text-sm">No score data</div>
	{/if}
</div>
