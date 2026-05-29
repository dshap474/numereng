<script lang="ts">
	import { ScatterChart, Tooltip } from 'layerchart';

	export interface CalibrationPoint {
		id: string;
		modelName: string;
		roundNumber?: number | null;
		target?: string | null;
		confidence?: string | null;
		liveStartedAt?: string | null;
		x: number;
		y: number;
	}

	interface Props {
		points: CalibrationPoint[];
		xLabel: string;
		yLabel: string;
		class?: string;
	}

	let { points, xLabel, yLabel, class: className }: Props = $props();

	const TARGET_COLORS = new Map([
		['ender20', '#7dd3fc'],
		['ender60', '#38bdf8'],
		['cyrusd20', '#c4b5fd'],
		['cyrusd60', '#a78bfa'],
		['cross_scope', '#f0abfc']
	]);
	const FALLBACK_COLOR = '#e5e7eb';

	function formatNumber(value: number | null | undefined, digits = 4): string {
		if (typeof value !== 'number' || !Number.isFinite(value)) return 'n/a';
		return value.toFixed(digits);
	}

	function formatDate(value: unknown): string {
		if (typeof value !== 'string' || !value.trim()) return 'n/a';
		const date = new Date(value);
		if (Number.isNaN(date.getTime())) return value;
		return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
	}

	function confidenceText(value: unknown): string {
		if (typeof value !== 'string' || !value.trim()) return 'waiting';
		return value.replaceAll('_', ' ');
	}

	function targetKey(point: CalibrationPoint): string {
		return point.target ?? 'unknown';
	}

	function targetLabel(value: string): string {
		return value.replaceAll('_', ' ');
	}

	function targetColor(value: string): string {
		return TARGET_COLORS.get(value) ?? FALLBACK_COLOR;
	}

	function paddedDomain(values: number[], includeZero = false): [number, number] {
		const min = Math.min(...values);
		const max = Math.max(...values);
		const baseMin = includeZero ? Math.min(min, 0) : min;
		const baseMax = includeZero ? Math.max(max, 0) : max;
		if (baseMin === baseMax) return [baseMin - 1, baseMax + 1];
		const padding = (baseMax - baseMin) * 0.12;
		return [baseMin - padding, baseMax + padding];
	}

	let paddedDomains = $derived.by(() => {
		if (points.length === 0) return null;
		return {
			xDomain: paddedDomain(points.map((point) => point.x)),
			yDomain: paddedDomain(points.map((point) => point.y), true)
		};
	});

	let targetLegend = $derived.by(() => {
		const values = [...new Set(points.map(targetKey))].sort();
		return values.map((value) => ({
			value,
			label: targetLabel(value),
			color: targetColor(value)
		}));
	});

	let regression = $derived.by(() => {
		if (points.length < 3) return null;
		const n = points.length;
		const meanX = points.reduce((sum, point) => sum + point.x, 0) / n;
		const meanY = points.reduce((sum, point) => sum + point.y, 0) / n;
		const sxx = points.reduce((sum, point) => sum + (point.x - meanX) ** 2, 0);
		const sxy = points.reduce((sum, point) => sum + (point.x - meanX) * (point.y - meanY), 0);
		if (sxx === 0) return null;
		const slope = sxy / sxx;
		const intercept = meanY - slope * meanX;
		const xValues = points.map((point) => point.x);
		const x1 = Math.min(...xValues);
		const x2 = Math.max(...xValues);
		return {
			x1,
			y1: slope * x1 + intercept,
			x2,
			y2: slope * x2 + intercept
		};
	});

	let chartAxisProps = $derived.by(() => ({
		xAxis: {
			label: xLabel,
			ticks: 7,
			tickSpacing: 96,
			tickLabelProps: { fontSize: 11 },
			labelProps: { fontSize: 12 }
		},
		yAxis: {
			label: yLabel,
			ticks: 7,
			tickSpacing: 58,
			tickLabelProps: { fontSize: 11 },
			labelProps: { fontSize: 12 }
		}
	}));
</script>

{#snippet pointTooltip({ context }: { context: any })}
	{@const point = context.tooltip?.data as CalibrationPoint | undefined}
	{#if point}
		<Tooltip.Root {context}>
			{#snippet children()}
					<div class="min-w-[280px] max-w-[340px] space-y-2 text-xs">
					<div class="font-mono font-medium">{point.modelName}</div>
					<div class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11px]">
						<span class="text-muted-foreground">{xLabel}</span><span class="tabular-nums">{formatNumber(point.x)}</span>
						<span class="text-muted-foreground">{yLabel}</span><span class="tabular-nums">{formatNumber(point.y)}</span>
						<span class="text-muted-foreground">Round</span><span>{point.roundNumber ?? 'n/a'}</span>
						<span class="text-muted-foreground">Confidence</span><span>{confidenceText(point.confidence)}</span>
						<span class="text-muted-foreground">Live Since</span><span>{formatDate(point.liveStartedAt)}</span>
					</div>
				</div>
			{/snippet}
		</Tooltip.Root>
	{/if}
{/snippet}

<div
	class={`flex h-[440px] min-w-0 flex-col overflow-hidden rounded-md border border-white/8 bg-white/[0.018] ${className ?? ''}`}
	role="img"
	aria-label="Local vs live calibration scatter plot"
>
	{#if points.length > 0}
		<div class="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-white/6 px-4 py-2">
			<span class="text-[11px] uppercase tracking-[0.16em] text-muted-foreground">Target</span>
			{#each targetLegend as target (target.value)}
				<span class="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
					<span class="h-2 w-2 rounded-full" style={`background:${target.color}`}></span>
					<span>{target.label}</span>
				</span>
			{/each}
			<span class="ml-auto hidden text-xs text-muted-foreground md:inline">zero line shown for live metric</span>
		</div>
		{#key `${xLabel}:${yLabel}:${points.length}`}
			<div class="min-h-0 flex-1">
				<ScatterChart
					data={points}
					x="x"
					y="y"
					xDomain={paddedDomains?.xDomain}
					yDomain={paddedDomains?.yDomain}
					padding={{ top: 18, right: 24, bottom: 42, left: 64 }}
					props={chartAxisProps}
					tooltip={pointTooltip}
					axis
					grid={{ x: { style: 'stroke-dasharray: 4 3' }, y: { style: 'stroke-dasharray: 4 3' } }}
				>
					{#snippet marks({ context })}
						<line
							x1={context.xScale(paddedDomains?.xDomain[0] ?? 0)}
							y1={context.yScale(0)}
							x2={context.xScale(paddedDomains?.xDomain[1] ?? 0)}
							y2={context.yScale(0)}
							stroke="rgba(255,255,255,0.2)"
							stroke-width="1"
							stroke-dasharray="3 3"
						/>
						{#if regression}
							<line
								x1={context.xScale(regression.x1)}
								y1={context.yScale(regression.y1)}
								x2={context.xScale(regression.x2)}
								y2={context.yScale(regression.y2)}
								stroke="rgba(56,189,248,0.52)"
								stroke-width="1.5"
								stroke-dasharray="6 4"
							/>
						{/if}
						{#each points as point (point.id)}
							<circle
								cx={context.xScale(point.x)}
								cy={context.yScale(point.y)}
								r="5.5"
								fill={targetColor(targetKey(point))}
								stroke="var(--color-background)"
								stroke-width="1.5"
								opacity="0.92"
							/>
						{/each}
					{/snippet}
				</ScatterChart>
			</div>
		{/key}
	{:else}
		<div class="flex h-full items-center justify-center text-sm text-muted-foreground">
			No models match the current filters.
		</div>
	{/if}
</div>
