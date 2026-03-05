<script lang="ts">
	import { ScatterChart, Tooltip } from 'layerchart';

	interface Row {
		era: string | number;
		corr20v2: number;
		mmc: number;
		payout_estimate: number;
	}

	interface Props {
		data: Row[];
		height?: string;
		corrWeight?: number;
		mmcWeight?: number;
		clip?: number;
		class?: string;
	}

	let {
		data,
		height = '360px',
		corrWeight = 0.75,
		mmcWeight = 2.25,
		clip = 0.05,
		class: className
	}: Props = $props();

	function clamp(value: number, min: number, max: number): number {
		return Math.max(min, Math.min(max, value));
	}

	function hexToRgb(hex: string): { r: number; g: number; b: number } {
		const normalized = hex.replace('#', '');
		const raw = normalized.length === 3
			? normalized.split('').map((c) => c + c).join('')
			: normalized;
		const n = Number.parseInt(raw, 16);
		return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
	}

	function rgbToHex(rgb: { r: number; g: number; b: number }): string {
		const to = (v: number) => clamp(Math.round(v), 0, 255).toString(16).padStart(2, '0');
		return `#${to(rgb.r)}${to(rgb.g)}${to(rgb.b)}`;
	}

	function mix(a: string, b: string, t: number): string {
		const ca = hexToRgb(a);
		const cb = hexToRgb(b);
		const tt = clamp(t, 0, 1);
		return rgbToHex({
			r: ca.r + (cb.r - ca.r) * tt,
			g: ca.g + (cb.g - ca.g) * tt,
			b: ca.b + (cb.b - ca.b) * tt
		});
	}

	function payoutColor(value: number, clipValue: number): string {
		// Diverging palette around 0 with saturation at +/- clip.
		const neutral = '#6b7280';
		const positive = '#2e9e5e';
		const negative = '#d44030';
		if (!Number.isFinite(value) || !Number.isFinite(clipValue) || clipValue <= 0) return neutral;
		const t = clamp(value / clipValue, -1, 1);
		if (t >= 0) return mix(neutral, positive, t);
		return mix(neutral, negative, -t);
	}

	let chartData = $derived.by(() => {
		return (data ?? []).map((row, index) => {
			const corr = Number(row.corr20v2) || 0;
			const mmc = Number(row.mmc) || 0;
			const payout = Number(row.payout_estimate) || 0;
			return {
				...row,
				__idx: index,
				__era: String(row.era ?? index + 1),
				__corr: corr,
				__mmc: mmc,
				__payout: payout,
				__raw: corrWeight * corr + mmcWeight * mmc
			};
		});
	});

	let paddedDomains = $derived.by(() => {
		if (chartData.length === 0) return null;
		let xMin = Infinity;
		let xMax = -Infinity;
		let yMin = Infinity;
		let yMax = -Infinity;
		for (const row of chartData) {
			xMin = Math.min(xMin, row.__corr);
			xMax = Math.max(xMax, row.__corr);
			yMin = Math.min(yMin, row.__mmc);
			yMax = Math.max(yMax, row.__mmc);
		}
		const xRange = xMax - xMin;
		const yRange = yMax - yMin;
		const xPad = Math.max(xRange * 0.08, 0.00035);
		const yPad = Math.max(yRange * 0.10, 0.00035);
		return {
			xDomain: [xMin - xPad, xMax + xPad] as [number, number],
			yDomain: [yMin - yPad, yMax + yPad] as [number, number]
		};
	});

	let isoValues = $derived.by(() => {
		if (chartData.length === 0) return [];
		let min = Infinity;
		let max = -Infinity;
		for (const row of chartData) {
			if (row.__raw < min) min = row.__raw;
			if (row.__raw > max) max = row.__raw;
		}
		const lo = Math.floor(min * 1000) / 1000;
		const hi = Math.ceil(max * 1000) / 1000;
		const step = (hi - lo) / 5;
		if (step <= 0) return [lo];
		const vals: number[] = [];
		for (let v = lo; v <= hi + step * 0.01; v += step) {
			vals.push(Math.round(v * 1000) / 1000);
		}
		return vals;
	});

	let clipIsoValues = $derived.by(() => {
		if (!Number.isFinite(clip) || clip <= 0) return [];
		return [-clip, clip];
	});

	const chartAxisProps = {
		xAxis: {
			label: 'CORR',
			ticks: 8,
			tickSpacing: 110,
			tickLabelProps: { fontSize: 11 },
			labelProps: { fontSize: 12 }
		},
		yAxis: {
			label: 'MMC',
			ticks: 7,
			tickSpacing: 70,
			tickLabelProps: { fontSize: 11 },
			labelProps: { fontSize: 12 }
		}
	};
</script>

{#snippet isoContent(context: any)}
	{@const xScale = context.xScale}
	{@const yScale = context.yScale}
	{@const xDom = context.xDomain}
	{@const yDom = context.yDomain}
	{@const left = xScale(xDom[0])}
	{@const right = xScale(xDom[1])}
	{@const top = yScale(yDom[1])}
	{@const bottom = yScale(yDom[0])}
	<defs>
		<clipPath id="per-era-payout-map-clip">
			<rect x={left} y={top} width={right - left} height={bottom - top} />
		</clipPath>
	</defs>
	<g clip-path="url(#per-era-payout-map-clip)">
		{#each isoValues as C, i (i)}
			{@const x0 = xDom[0]}
			{@const x1 = xDom[1]}
			{@const y0_at_x0 = (C - corrWeight * x0) / mmcWeight}
			{@const y0_at_x1 = (C - corrWeight * x1) / mmcWeight}
			{@const clampedStartX = y0_at_x0 > yDom[1] ? (C - mmcWeight * yDom[1]) / corrWeight : y0_at_x0 < yDom[0] ? (C - mmcWeight * yDom[0]) / corrWeight : x0}
			{@const clampedStartY = Math.max(yDom[0], Math.min(yDom[1], y0_at_x0))}
			{@const clampedEndX = y0_at_x1 > yDom[1] ? (C - mmcWeight * yDom[1]) / corrWeight : y0_at_x1 < yDom[0] ? (C - mmcWeight * yDom[0]) / corrWeight : x1}
			{@const clampedEndY = Math.max(yDom[0], Math.min(yDom[1], y0_at_x1))}
			{@const sx = xScale(clampedStartX)}
			{@const sy = yScale(clampedStartY)}
			{@const ex = xScale(clampedEndX)}
			{@const ey = yScale(clampedEndY)}
			<line x1={sx} y1={sy} x2={ex} y2={ey}
				stroke="currentColor" stroke-opacity="0.25" stroke-width="1"
				stroke-dasharray="6 4" />
		{/each}
		{#each clipIsoValues as C, i (i)}
			{@const x0 = xDom[0]}
			{@const x1 = xDom[1]}
			{@const y0_at_x0 = (C - corrWeight * x0) / mmcWeight}
			{@const y0_at_x1 = (C - corrWeight * x1) / mmcWeight}
			{@const clampedStartX = y0_at_x0 > yDom[1] ? (C - mmcWeight * yDom[1]) / corrWeight : y0_at_x0 < yDom[0] ? (C - mmcWeight * yDom[0]) / corrWeight : x0}
			{@const clampedStartY = Math.max(yDom[0], Math.min(yDom[1], y0_at_x0))}
			{@const clampedEndX = y0_at_x1 > yDom[1] ? (C - mmcWeight * yDom[1]) / corrWeight : y0_at_x1 < yDom[0] ? (C - mmcWeight * yDom[0]) / corrWeight : x1}
			{@const clampedEndY = Math.max(yDom[0], Math.min(yDom[1], y0_at_x1))}
			{@const sx = xScale(clampedStartX)}
			{@const sy = yScale(clampedStartY)}
			{@const ex = xScale(clampedEndX)}
			{@const ey = yScale(clampedEndY)}
			<line x1={sx} y1={sy} x2={ex} y2={ey}
				stroke="currentColor" stroke-opacity="0.55" stroke-width="1.25"
			/>
		{/each}
	</g>
	<rect x={left} y={top} width={right - left} height={bottom - top} fill="none" stroke="currentColor" stroke-opacity="0.3" stroke-width="1" />
{/snippet}

{#snippet rowTooltip({ context }: { context: any })}
	{@const d = context.tooltip?.data as (Row & { __era: string; __corr: number; __mmc: number; __payout: number; __raw: number }) | undefined}
	{#if d}
		{@const eps = 1e-12}
		{@const clipped = Number.isFinite(clip) && clip > 0 ? Math.abs(d.__payout) >= clip - eps : false}
		<Tooltip.Root {context}>
			{#snippet children({ data })}
				<div class="text-xs space-y-1">
					<div class="font-medium truncate max-w-48">Era {d.__era}</div>
					<div class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11px]">
						<span class="text-muted-foreground">CORR</span><span class="tabular-nums">{d.__corr.toFixed(5)}</span>
						<span class="text-muted-foreground">MMC</span><span class="tabular-nums">{d.__mmc.toFixed(5)}</span>
						<span class="text-muted-foreground">Raw</span><span class="tabular-nums">{d.__raw.toFixed(5)}</span>
						<span class="text-muted-foreground">Payout</span><span class="tabular-nums">{d.__payout.toFixed(5)}</span>
						<span class="text-muted-foreground">Clipped?</span><span class="tabular-nums">{clipped ? 'yes' : 'no'}</span>
					</div>
				</div>
			{/snippet}
		</Tooltip.Root>
	{/if}
{/snippet}

<div aria-label="Per-era payout map">
	<div style:height={height} style:min-height="260px" class={`relative ${className ?? ''}`}>
		{#if chartData.length > 0}
			<div class="pointer-events-none absolute left-3 top-2 z-10 rounded bg-background/80 px-2 py-1 text-[10px] text-muted-foreground">
				Iso-lines: raw score = {corrWeight.toFixed(2)}*CORR + {mmcWeight.toFixed(2)}*MMC (clip: ±{clip.toFixed(2)})
			</div>
			<ScatterChart
				data={chartData}
				x="__corr"
				y="__mmc"
				xDomain={paddedDomains?.xDomain}
				yDomain={paddedDomains?.yDomain}
				padding={{ top: 12, right: 42, bottom: 42, left: 56 }}
				props={chartAxisProps}
				tooltip={rowTooltip}
				axis
				grid={{ x: { style: 'stroke-dasharray: 4 3' }, y: { style: 'stroke-dasharray: 4 3' } }}
			>
				{#snippet belowMarks({ context })}
					{@render isoContent(context)}
				{/snippet}
				{#snippet marks({ context })}
					{#each chartData as d (d.__idx)}
						{@const cx = context.xScale(d.__corr)}
						{@const cy = context.yScale(d.__mmc)}
						{@const fill = payoutColor(d.__payout, clip)}
						<circle {cx} {cy} r="4" fill={fill} opacity="0.88" />
					{/each}
				{/snippet}
			</ScatterChart>
		{:else}
			<div class="flex h-full items-center justify-center text-muted-foreground text-sm">No per-era payout data</div>
		{/if}
	</div>
</div>
