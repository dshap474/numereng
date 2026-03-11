<script lang="ts">
	import { ScatterChart, Tooltip } from 'layerchart';

	type ColorMode = 'default' | 'model' | 'target';
	type EntityType = 'run' | 'hpo_best' | 'ensemble';

	interface Run {
		run_id: string;
		config_name?: string;
		corr20v2_mean: number;
		mmc_mean: number;
		model_type: string;
		target: string;
		entity_type?: EntityType;
		corr20v2_sharpe?: number | null;
		payout_estimate_mean?: number | null;
		mmc_coverage_ratio_rows?: number | null;
		bmc_mean?: number | null;
		max_drawdown?: number | null;
	}

	interface Props {
		runs: Run[];
		height?: number;
		colorMode?: ColorMode;
		showIsoLines?: boolean;
		corrWeight?: number;
		mmcWeight?: number;
		class?: string;
	}

	let {
		runs,
		height,
		colorMode = 'default',
		showIsoLines = false,
		corrWeight = 0.75,
		mmcWeight = 2.25,
		class: className
	}: Props = $props();

	const PALETTE = [
		'#d44030', '#5868b0', '#2e9e5e', '#a04aaf', '#c9a52e', '#3a9a8e',
		'#c04030', '#4a9ec2', '#7daa30', '#a03a6e', '#b88a24', '#2e5a7a',
		'#48a850', '#c46040', '#6060b8', '#38906a', '#c89430', '#4878a8',
		'#a8a030', '#3a8e88'
	];

	let highlightedSeries = $state<string | null>(null);

	function formatModelName(value: string): string {
		const normalized = value.replace(/[_-]+/g, ' ').trim();
		if (!normalized) return 'Unknown';
		return normalized
			.split(/\s+/)
			.map((part) => {
				const lower = part.toLowerCase();
				if (lower === 'lgbm') return 'LGBM';
				if (lower === 'xgboost') return 'XGBoost';
				if (lower === 'hpo') return 'HPO';
				return part.charAt(0).toUpperCase() + part.slice(1).toLowerCase();
			})
			.join(' ');
	}

	function opPrefix(entityType: EntityType | undefined): string {
		if (entityType === 'ensemble') return 'Ensemble';
		if (entityType === 'hpo_best') return 'HPO';
		return 'Run';
	}

	function formatLabel(key: string, mode: ColorMode): string {
		if (mode === 'target') {
			// "target_agnes_20" → "Agnes 20"
			const stripped = key.replace(/^target_/, '');
			const parts = stripped.split('_');
			const day = parts.pop();
			const name = parts.join('_');
			return `${name.charAt(0).toUpperCase()}${name.slice(1)} ${day}`;
		}
		return key;
	}

	let seriesConfig = $derived.by(() => {
		if (colorMode === 'default') return undefined;

		const groups = new Map<string, { label: string; data: Run[] }>();
		for (const run of runs) {
			const key =
				colorMode === 'model'
					? `${opPrefix(run.entity_type)}::${run.model_type}`
					: run.target;
			const label =
				colorMode === 'model'
					? `${opPrefix(run.entity_type)}: ${formatModelName(run.model_type)}`
					: formatLabel(run.target, colorMode);
			if (!groups.has(key)) groups.set(key, { label, data: [] });
			groups.get(key)!.data.push(run);
		}

		const sortedKeys = [...groups.keys()].sort();
		const hl = highlightedSeries;
		return sortedKeys.map((key, i) => ({
			key,
			label: groups.get(key)!.label,
			data: groups.get(key)!.data,
			color: PALETTE[i % PALETTE.length],
			props: hl
				? {
						opacity: key === hl ? 1 : 0.15,
						r: key === hl ? 7 : 4
					}
				: undefined
		}));
	});

	function computeBands(
		xMin: number, xMax: number, yMin: number, yMax: number,
		isos: number[], wX: number, wY: number,
		xScale: (v: number) => number, yScale: (v: number) => number
	): { points: string; even: boolean }[] {
		const cornerC = (x: number, y: number) => wX * x + wY * y;
		const cMin = Math.min(cornerC(xMin, yMin), cornerC(xMax, yMin), cornerC(xMin, yMax), cornerC(xMax, yMax));
		const cMax = Math.max(cornerC(xMin, yMin), cornerC(xMax, yMin), cornerC(xMin, yMax), cornerC(xMax, yMax));

		const sortedIso = [...isos]
			.filter((value) => Number.isFinite(value))
			.sort((a, b) => a - b)
			.filter((value) => value > cMin + 1e-9 && value < cMax - 1e-9);
		const extended = [cMin, ...sortedIso, cMax];

		function lineRectIntersections(C: number): [number, number][] {
			const pts: [number, number][] = [];
			const yAtXMin = (C - wX * xMin) / wY;
			if (yAtXMin >= yMin - 1e-9 && yAtXMin <= yMax + 1e-9)
				pts.push([xMin, Math.max(yMin, Math.min(yMax, yAtXMin))]);
			const yAtXMax = (C - wX * xMax) / wY;
			if (yAtXMax >= yMin - 1e-9 && yAtXMax <= yMax + 1e-9)
				pts.push([xMax, Math.max(yMin, Math.min(yMax, yAtXMax))]);
			const xAtYMin = (C - wY * yMin) / wX;
			if (xAtYMin > xMin + 1e-9 && xAtYMin < xMax - 1e-9)
				pts.push([xAtYMin, yMin]);
			const xAtYMax = (C - wY * yMax) / wX;
			if (xAtYMax > xMin + 1e-9 && xAtYMax < xMax - 1e-9)
				pts.push([xAtYMax, yMax]);
			return pts;
		}

		function side(C: number, x: number, y: number) {
			return wX * x + wY * y - C;
		}

		const corners: [number, number][] = [[xMin, yMin], [xMax, yMin], [xMax, yMax], [xMin, yMax]];
		const bands: { points: string; even: boolean }[] = [];

		for (let i = 0; i < extended.length - 1; i++) {
			const cLo = extended[i];
			const cHi = extended[i + 1];

			const verts: [number, number][] = [
				...lineRectIntersections(cLo),
				...lineRectIntersections(cHi)
			];
			for (const c of corners) {
				const s = side(cLo, c[0], c[1]);
				const sHi = side(cHi, c[0], c[1]);
				if (s >= -1e-9 && sHi <= 1e-9) verts.push(c);
			}

			if (verts.length < 3) continue;

			let cx = 0, cy = 0;
			for (const v of verts) { cx += v[0]; cy += v[1]; }
			cx /= verts.length; cy /= verts.length;
			verts.sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));

			const pts = verts.map(([x, y]) => `${xScale(x)},${yScale(y)}`).join(' ');
			bands.push({ points: pts, even: i % 2 === 0 });
		}

		return bands;
	}

	let isoValues = $derived.by(() => {
		if (!showIsoLines || runs.length === 0) return [];
		let xMin = Infinity;
		let xMax = -Infinity;
		let yMin = Infinity;
		let yMax = -Infinity;
		for (const run of runs) {
			xMin = Math.min(xMin, run.corr20v2_mean);
			xMax = Math.max(xMax, run.corr20v2_mean);
			yMin = Math.min(yMin, run.mmc_mean);
			yMax = Math.max(yMax, run.mmc_mean);
		}
		const xRange = xMax - xMin;
		const yRange = yMax - yMin;
		const xPad = Math.max(xRange * 0.08, 0.00035);
		const yPad = Math.max(yRange * 0.10, 0.00035);
		const domXMin = xMin - xPad;
		const domXMax = xMax + xPad;
		const domYMin = yMin - yPad;
		const domYMax = yMax + yPad;
		const cornerC = (x: number, y: number) => corrWeight * x + mmcWeight * y;
		const cMin = Math.min(
			cornerC(domXMin, domYMin),
			cornerC(domXMax, domYMin),
			cornerC(domXMin, domYMax),
			cornerC(domXMax, domYMax)
		);
		const cMax = Math.max(
			cornerC(domXMin, domYMin),
			cornerC(domXMax, domYMin),
			cornerC(domXMin, domYMax),
			cornerC(domXMax, domYMax)
		);
		if (!Number.isFinite(cMin) || !Number.isFinite(cMax)) return [];
		if (Math.abs(cMax - cMin) < 1e-12) return [cMin];
		const lineCount = 6;
		const step = (cMax - cMin) / (lineCount + 1);
		const values: number[] = [];
		for (let i = 1; i <= lineCount; i++) {
			values.push(cMin + i * step);
		}
		return values;
	});

	let paddedDomains = $derived.by(() => {
		if (runs.length === 0) return null;
		let xMin = Infinity;
		let xMax = -Infinity;
		let yMin = Infinity;
		let yMax = -Infinity;
		for (const run of runs) {
			xMin = Math.min(xMin, run.corr20v2_mean);
			xMax = Math.max(xMax, run.corr20v2_mean);
			yMin = Math.min(yMin, run.mmc_mean);
			yMax = Math.max(yMax, run.mmc_mean);
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
	{@const bands = computeBands(xDom[0], xDom[1], yDom[0], yDom[1], isoValues, corrWeight, mmcWeight, xScale, yScale)}
	<defs>
		<clipPath id="chart-clip">
			<rect x={left} y={top} width={right - left} height={bottom - top} />
		</clipPath>
	</defs>
		<g clip-path="url(#chart-clip)">
			{#each bands as band, i (i)}
				<polygon points={band.points} fill="#000000" fill-opacity={band.even ? 0.085 : 0.045} stroke="none" />
			{/each}
		</g>
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
		{@const labelY = Math.max(top + 9, Math.min(bottom - 9, ey))}
		<line x1={sx} y1={sy} x2={ex} y2={ey}
			stroke="currentColor" stroke-opacity="0.35" stroke-width="1"
			stroke-dasharray="6 4" />
		<text x={ex + 8} y={labelY} font-size="9" fill="currentColor" fill-opacity="0.58" dominant-baseline="middle">{C.toFixed(3)}</text>
	{/each}
	<!-- Solid border around chart area -->
	<rect x={left} y={top} width={right - left} height={bottom - top} fill="none" stroke="currentColor" stroke-opacity="0.3" stroke-width="1" />
{/snippet}

	{#snippet runTooltip({ context }: { context: any })}
		{@const d = context.tooltip?.data as Run | undefined}
		{#if d}
			{@const rawPayout = corrWeight * d.corr20v2_mean + mmcWeight * d.mmc_mean}
			<Tooltip.Root {context}>
				{#snippet children({ data })}
					<div class="text-xs space-y-1">
						<div class="font-medium truncate max-w-56" title={d.config_name ?? `${d.model_type} · ${d.target}`}>
							{d.config_name ?? `${d.model_type} · ${d.target}`}
						</div>
						<div class="text-[10px] text-muted-foreground font-mono truncate max-w-56" title={d.run_id}>{d.run_id}</div>
						<div class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11px]">
							<span class="text-muted-foreground">CORR</span><span class="tabular-nums">{d.corr20v2_mean?.toFixed(5)}</span>
							<span class="text-muted-foreground">MMC</span><span class="tabular-nums">{d.mmc_mean?.toFixed(5)}</span>
							<span class="text-muted-foreground">Raw Payout</span><span class="tabular-nums">{rawPayout.toFixed(5)}</span>
							{#if d.payout_estimate_mean != null}
								<span class="text-muted-foreground">Payout Est</span><span class="tabular-nums">{d.payout_estimate_mean.toFixed(5)}</span>
								<span class="text-muted-foreground">Raw - Est</span><span class="tabular-nums">{(rawPayout - d.payout_estimate_mean).toFixed(5)}</span>
							{/if}
							{#if d.mmc_coverage_ratio_rows != null}
								<span class="text-muted-foreground">MMC Coverage</span><span class="tabular-nums">{d.mmc_coverage_ratio_rows.toFixed(3)}</span>
							{/if}
							{#if d.corr20v2_sharpe != null}
							<span class="text-muted-foreground">CORR Sharpe</span><span class="tabular-nums">{d.corr20v2_sharpe.toFixed(3)}</span>
						{/if}
						{#if d.bmc_mean != null}
							<span class="text-muted-foreground">BMC (Diag)</span><span class="tabular-nums">{d.bmc_mean.toFixed(5)}</span>
						{/if}
						{#if d.max_drawdown != null}
							<span class="text-muted-foreground">Max DD</span><span class="tabular-nums">{d.max_drawdown.toFixed(4)}</span>
						{/if}
					</div>
				</div>
			{/snippet}
		</Tooltip.Root>
	{/if}
{/snippet}

	<div class={`flex min-h-0 flex-col ${className ?? ''}`} style:height={height ? `${height}px` : '100%'}>
		<div class="relative min-h-0 flex-1 min-w-0">
			{#if showIsoLines && runs.length > 0}
				<div class="pointer-events-none absolute left-3 top-2 z-10 rounded bg-background/80 px-2 py-1 text-[10px] text-muted-foreground">
					Iso-lines: raw payout score (unclipped) = {corrWeight.toFixed(2)}*CORR + {mmcWeight.toFixed(2)}*MMC
				</div>
			{/if}
		{#if runs.length > 0}
				{#if seriesConfig}
					<ScatterChart
						data={runs}
						x="corr20v2_mean"
						y="mmc_mean"
					xDomain={paddedDomains?.xDomain}
					yDomain={paddedDomains?.yDomain}
					series={seriesConfig}
					padding={{ top: 12, right: 42, bottom: 42, left: 56 }}
					props={chartAxisProps}
					tooltip={runTooltip}
					axis
					grid={{ x: { style: 'stroke-dasharray: 4 3' }, y: { style: 'stroke-dasharray: 4 3' } }}
				>
					{#snippet belowMarks({ context })}
						{#if showIsoLines}
							{@render isoContent(context)}
						{/if}
					{/snippet}
					{#snippet marks({ context, visibleSeries })}
						{#each visibleSeries as s, i (s.key)}
							{@const color = s.color ?? PALETTE[i % PALETTE.length]}
							{@const hl = highlightedSeries}
							{@const op = hl ? (s.key === hl ? 1 : 0.15) : 0.85}
							{@const r = hl ? (s.key === hl ? 7 : 4) : 5}
							{#each s.data ?? [] as d (d.run_id)}
								{@const cx = context.xScale(d.corr20v2_mean)}
								{@const cy = context.yScale(d.mmc_mean)}
								<circle {cx} {cy} {r} fill={color} opacity={op} />
							{/each}
						{/each}
					{/snippet}
				</ScatterChart>
				{:else}
					<ScatterChart
						data={runs}
						x="corr20v2_mean"
						y="mmc_mean"
					xDomain={paddedDomains?.xDomain}
					yDomain={paddedDomains?.yDomain}
					padding={{ top: 12, right: 42, bottom: 42, left: 56 }}
					props={chartAxisProps}
					tooltip={runTooltip}
					axis
					grid={{ x: { style: 'stroke-dasharray: 4 3' }, y: { style: 'stroke-dasharray: 4 3' } }}
				>
					{#snippet belowMarks({ context })}
						{#if showIsoLines}
							{@render isoContent(context)}
						{/if}
					{/snippet}
						{#snippet marks({ context })}
							{#each runs as d (d.run_id)}
								{@const cx = context.xScale(d.corr20v2_mean)}
								{@const cy = context.yScale(d.mmc_mean)}
								<circle {cx} {cy} r="5" fill="var(--color-primary)" opacity="0.85" />
							{/each}
						{/snippet}
				</ScatterChart>
			{/if}
		{:else}
			<div class="flex h-full items-center justify-center text-muted-foreground">No CORR/MMC data</div>
		{/if}
	</div>
	{#if seriesConfig}
		<div class="mt-4 max-h-28 shrink-0 overflow-y-auto border-t border-border pt-3">
			<div class="flex flex-wrap gap-x-4 gap-y-2">
				{#each seriesConfig as s (s.key)}
					<!-- svelte-ignore a11y_no_static_element_interactions -->
					<div
						class="flex min-w-0 items-center gap-1.5 cursor-pointer"
						title={s.label}
						onmouseenter={() => (highlightedSeries = s.key)}
						onmouseleave={() => (highlightedSeries = null)}
					>
						<span class="w-2 h-2 rounded-full shrink-0" style:background={s.color}></span>
						<span class="text-[10px] leading-tight text-muted-foreground">{s.label}</span>
					</div>
				{/each}
			</div>
		</div>
	{/if}
</div>
