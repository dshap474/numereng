<script lang="ts">
	import { BarChart } from 'layerchart';

	interface Props {
		data: Record<string, unknown>[];
		topN?: number;
	}

	let { data, topN = 15 }: Props = $props();

	let impCol = $derived.by(() => {
		if (data.length === 0) return 'importance';
		return (
			Object.keys(data[0]).find(
				(k) => k.toLowerCase().includes('importance') || k.toLowerCase().includes('gain')
			) ?? Object.keys(data[0])[Object.keys(data[0]).length - 1]
		);
	});

	let featCol = $derived.by(() => {
		if (data.length === 0) return 'feature';
		return (
			Object.keys(data[0]).find((k) => k.toLowerCase().includes('feature')) ??
			Object.keys(data[0])[0]
		);
	});

	let chartData = $derived.by(() => {
		return [...data]
			.sort((a, b) => (Number(b[impCol]) || 0) - (Number(a[impCol]) || 0))
			.slice(0, topN)
			.map((d) => ({
				feature: truncateLabel(String(d[featCol])),
				feature_full: String(d[featCol]),
				importance: Number(d[impCol]) || 0
			}));
	});

	let chartHeight = $derived(Math.max(chartData.length * 26, 220));

	function truncateLabel(value: string): string {
		if (value.length <= 34) return value;
		return `${value.slice(0, 31)}...`;
	}
</script>

<div aria-label="Feature importance chart">
	<div style:height="{chartHeight}px">
		{#if chartData.length > 0}
			<BarChart
				data={chartData}
				x="importance"
				y="feature"
				orientation="horizontal"
				series={[{ key: 'default', value: 'importance', color: 'var(--color-chart-bar)' }]}
				props={{
					xAxis: {
						ticks: 5,
						tickLabelProps: { 'font-size': 11 }
					},
					yAxis: {
						tickLabelProps: { 'font-size': 11 }
					}
				}}
				axis
				tooltip
				grid={false}
				padding={{ left: 230 }}
			/>
		{:else}
			<div class="flex h-full items-center justify-center text-muted-foreground text-sm">No feature importance data</div>
		{/if}
	</div>
</div>
