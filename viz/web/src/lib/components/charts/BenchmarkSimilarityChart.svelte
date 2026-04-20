<script lang="ts">
	import { ScatterChart, Tooltip } from 'layerchart';

	interface Point {
		id: string;
		name: string;
		model_type: string;
		target: string;
		corr_with_benchmark: number;
		bmc_mean?: number | null;
		bmc_last_200_eras_mean: number;
		is_champion: boolean;
		source_type: 'run' | 'round_result';
	}

	interface Props {
		points: Point[];
		bmcMode?: 'last200' | 'full';
		class?: string;
	}

	let { points, bmcMode = 'last200', class: className }: Props = $props();

	function formatNumber(value: number | null | undefined, digits = 5): string {
		if (typeof value !== 'number' || !Number.isFinite(value)) return 'n/a';
		return value.toFixed(digits);
	}

	function formatTarget(value: string): string {
		const stripped = value.replace(/^target_/, '');
		const parts = stripped
			.split(/[_\s-]+/)
			.map((part) => part.trim())
			.filter(Boolean);
		if (parts.length === 0) return value;
		return parts.map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join(' ');
	}

	function bmcValue(point: Point): number | null {
		const value = bmcMode === 'full' ? point.bmc_mean : point.bmc_last_200_eras_mean;
		return typeof value === 'number' && Number.isFinite(value) ? value : null;
	}

	let chartPoints = $derived.by(() => {
		return points
			.map((point) => {
				const yValue = bmcValue(point);
				if (yValue == null) return null;
				return {
					...point,
					__bmc_value: yValue
				};
			})
			.filter((point): point is Point & { __bmc_value: number } => point != null);
	});

	const bmcAxisLabel = $derived(
		bmcMode === 'full' ? 'BMC Mean (Full Run)' : 'BMC Mean (Last 200 Eras)'
	);

	const bmcTooltipLabel = $derived(
		bmcMode === 'full' ? 'BMC Mean' : 'BMC Last 200'
	);

	let paddedDomains = $derived.by(() => {
		if (chartPoints.length === 0) return null;
		let xMin = Infinity;
		let xMax = -Infinity;
		let yMin = Infinity;
		let yMax = -Infinity;
		for (const point of chartPoints) {
			xMin = Math.min(xMin, point.corr_with_benchmark);
			xMax = Math.max(xMax, point.corr_with_benchmark);
			yMin = Math.min(yMin, point.__bmc_value);
			yMax = Math.max(yMax, point.__bmc_value);
		}
		if (!Number.isFinite(xMin) || !Number.isFinite(yMin)) return null;
		const xRange = xMax - xMin;
		const yRange = yMax - yMin;
		const xPad = Math.max(xRange * 0.08, 0.00035);
		const yPad = Math.max(yRange * 0.1, 0.00035);
		return {
			xDomain: [xMin - xPad, xMax + xPad] as [number, number],
			yDomain: [yMin - yPad, yMax + yPad] as [number, number]
		};
	});

	let chartAxisProps = $derived.by(() => ({
		xAxis: {
			label: 'Corr vs Benchmark',
			ticks: 8,
			tickSpacing: 110,
			tickLabelProps: { fontSize: 11 },
			labelProps: { fontSize: 12 }
		},
		yAxis: {
			label: bmcAxisLabel,
			ticks: 7,
			tickSpacing: 70,
			tickLabelProps: { fontSize: 11 },
			labelProps: { fontSize: 12 }
		}
	}));
</script>

{#snippet pointTooltip({ context }: { context: any })}
	{@const point = context.tooltip?.data as Point | undefined}
	{#if point}
		<Tooltip.Root {context}>
			{#snippet children()}
				<div class="min-w-[280px] max-w-[340px] space-y-2 text-xs">
					<div class="font-medium">{point.name}</div>
					<div class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11px]">
						<span class="text-muted-foreground">Type</span><span>{point.source_type === 'run' ? 'Run' : 'Round Result'}</span>
						<span class="text-muted-foreground">Model</span><span class="truncate">{point.model_type}</span>
						<span class="text-muted-foreground">Target</span><span class="truncate">{formatTarget(point.target)}</span>
						<span class="text-muted-foreground">Champion</span><span>{point.is_champion ? 'Yes' : 'No'}</span>
						</div>
						<div class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11px]">
							<span class="text-muted-foreground">Corr vs Benchmark</span><span class="tabular-nums">{formatNumber(point.corr_with_benchmark)}</span>
							<span class="text-muted-foreground">{bmcTooltipLabel}</span><span class="tabular-nums">{formatNumber(bmcValue(point))}</span>
						</div>
					</div>
				{/snippet}
		</Tooltip.Root>
	{/if}
{/snippet}

<div class={`flex min-h-0 flex-col ${className ?? ''}`} aria-label="Benchmark similarity chart">
		<div class="relative min-h-0 flex-1 min-w-0">
		{#if chartPoints.length > 0}
			{#key bmcMode}
				<ScatterChart
					data={chartPoints}
					x="corr_with_benchmark"
					y="__bmc_value"
					xDomain={paddedDomains?.xDomain}
					yDomain={paddedDomains?.yDomain}
					padding={{ top: 12, right: 18, bottom: 42, left: 64 }}
					props={chartAxisProps}
					tooltip={pointTooltip}
					axis
					grid={{ x: { style: 'stroke-dasharray: 4 3' }, y: { style: 'stroke-dasharray: 4 3' } }}
				>
					{#snippet marks({ context })}
						{#each chartPoints as point (point.id)}
							{@const cx = context.xScale(point.corr_with_benchmark)}
							{@const cy = context.yScale(point.__bmc_value)}
							<circle
								{cx}
								{cy}
								r={point.is_champion ? 7 : 5}
								fill={point.is_champion ? 'var(--color-chart-bar-champion)' : 'var(--color-primary)'}
								stroke={point.is_champion ? 'var(--color-chart-bar-champion)' : 'var(--color-border)'}
								stroke-width={point.is_champion ? 2 : 1}
								opacity={point.is_champion ? 1 : 0.9}
							/>
						{/each}
					{/snippet}
				</ScatterChart>
			{/key}
		{:else}
			<div class="flex h-full items-center justify-center text-sm text-muted-foreground">
				No benchmark-similarity data available.
			</div>
		{/if}
	</div>
</div>
