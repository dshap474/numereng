<script lang="ts">
	import { AreaChart } from 'layerchart';
	import { scaleLinear } from 'd3-scale';

	interface Props {
		data: Record<string, string | number>[];
		height?: string;
	}

	let { data, height = '300px' }: Props = $props();

	let eraKey = $derived.by(() => {
		if (data.length === 0) return 'era';
		return Object.keys(data[0]).find((k) => k.toLowerCase().includes('era')) ?? Object.keys(data[0])[0];
	});

	let corrKey = $derived.by(() => {
		if (data.length === 0) return 'corr';
		return Object.keys(data[0]).find((k) => k !== eraKey) ?? Object.keys(data[0])[1];
	});

	let cumulativeData = $derived.by(() => {
		return data.map((d, index) => ({
			...d,
			__idx: index + 1,
			__era: String(d[eraKey] ?? index + 1),
			cumulative: Number(d[corrKey]) || 0
		}));
	});
</script>

<div aria-label="Cumulative correlation chart">
	<div style:height={height} style:min-height="200px">
		{#if data.length > 0}
			<AreaChart
				data={cumulativeData}
				x="__idx"
				xScale={scaleLinear()}
				y="cumulative"
				series={[{ key: 'default', value: 'cumulative', color: 'var(--color-chart-area)' }]}
				props={{
					xAxis: {
						ticks: 8,
						format: 'integer',
						tickSpacing: 120,
						tickLabelProps: { 'font-size': 11 }
					},
					yAxis: {
						ticks: 6,
						tickLabelProps: { 'font-size': 11 }
					}
				}}
				axis
				tooltip
			/>
		{:else}
			<div class="flex h-full items-center justify-center text-muted-foreground text-sm">No cumulative data</div>
		{/if}
	</div>
</div>
