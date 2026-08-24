<!--
	Round-by-round trajectory sparkline for one submitted model and one live metric.

	USAGE:
		<SubmissionTrajectoryChart rounds={rounds} metric="mmc20" label="MMC20" />

	Plain SVG on purpose: the existing chart components are per-era / scatter shaped, and the
	page must distinguish resolved from resolving (estimated) points without adding a dependency.
-->
<script lang="ts">
	import type { SubmissionRound } from '$lib/api/client';
	import {
		DASH,
		formatNumber,
		roundMetric,
		roundNumber,
		roundSortValue,
		roundState,
		type LiveRoundMetricKey
	} from './format';

	// ----------------------------------------------------------------------- //
	// Props
	// ----------------------------------------------------------------------- //

	interface Props {
		rounds: SubmissionRound[];
		metric: LiveRoundMetricKey;
		label: string;
	}

	let { rounds, metric, label }: Props = $props();

	const WIDTH = 640;
	const HEIGHT = 180;
	const PAD_X = 44;
	const PAD_Y = 16;

	// ----------------------------------------------------------------------- //
	// Geometry
	// ----------------------------------------------------------------------- //

	type Point = { x: number; y: number; value: number; roundLabel: string; resolved: boolean; estimate: boolean };

	let series = $derived.by(() =>
		[...rounds]
			.sort((a, b) => roundSortValue(a) - roundSortValue(b))
			.map((row) => ({
				row,
				value: roundMetric(row, metric)
			}))
			.filter((entry): entry is { row: SubmissionRound; value: number } => entry.value !== null)
	);

	let bounds = $derived.by(() => {
		const values = series.map((entry) => entry.value);
		if (values.length === 0) return { min: 0, max: 0 };
		const min = Math.min(0, ...values);
		const max = Math.max(0, ...values);
		return min === max ? { min: min - 0.01, max: max + 0.01 } : { min, max };
	});

	let points = $derived.by<Point[]>(() => {
		const count = series.length;
		if (count === 0) return [];
		const span = bounds.max - bounds.min || 1;
		return series.map((entry, index) => {
			const ratio = count === 1 ? 0.5 : index / (count - 1);
			const state = roundState(entry.row).toLowerCase();
			return {
				x: PAD_X + ratio * (WIDTH - PAD_X - PAD_Y),
				y: HEIGHT - PAD_Y - ((entry.value - bounds.min) / span) * (HEIGHT - PAD_Y * 2),
				value: entry.value,
				roundLabel: roundNumber(entry.row),
				resolved: state === 'resolved',
				estimate: entry.row.is_estimate === true
			};
		});
	});

	let linePath = $derived(
		points.map((point, index) => `${index === 0 ? 'M' : 'L'}${point.x.toFixed(1)} ${point.y.toFixed(1)}`).join(' ')
	);

	let zeroY = $derived.by(() => {
		const span = bounds.max - bounds.min || 1;
		return HEIGHT - PAD_Y - ((0 - bounds.min) / span) * (HEIGHT - PAD_Y * 2);
	});
</script>

<!-- ------------------------------------------------------------------------ -->
<!-- Markup                                                                    -->
<!-- ------------------------------------------------------------------------ -->

<div class="rounded-md border border-white/8 bg-white/[0.02] px-3 py-3">
	<div class="flex items-baseline justify-between gap-3">
		<p class="text-[11px] uppercase tracking-[0.16em] text-muted-foreground">{label} trajectory</p>
		<p class="font-mono text-[11px] text-muted-foreground">
			{points.length} scored {points.length === 1 ? 'round' : 'rounds'}
		</p>
	</div>

	{#if points.length === 0}
		<p class="mt-3 text-sm text-muted-foreground">No scored rounds for {label} yet.</p>
	{:else}
		<svg
			class="mt-2 w-full"
			viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
			role="img"
			aria-label={`${label} by round`}
		>
			<line x1={PAD_X} y1={zeroY} x2={WIDTH - PAD_Y} y2={zeroY} stroke="currentColor" stroke-width="1" class="text-white/15" />
			<text x="4" y={PAD_Y + 4} class="fill-current text-[10px] text-muted-foreground">{formatNumber(bounds.max, 3)}</text>
			<text x="4" y={HEIGHT - PAD_Y} class="fill-current text-[10px] text-muted-foreground">{formatNumber(bounds.min, 3)}</text>
			<path d={linePath} fill="none" stroke="var(--color-chart-line)" stroke-width="1.5" />
			{#each points as point (point.roundLabel)}
				<circle
					cx={point.x}
					cy={point.y}
					r={point.resolved ? 4 : 3.5}
					fill={point.resolved ? 'var(--color-positive)' : 'transparent'}
					stroke={point.resolved ? 'var(--color-positive)' : '#38bdf8'}
					stroke-width="1.5"
					stroke-dasharray={point.estimate ? '2 2' : undefined}
				>
					<title>
						{`round ${point.roundLabel}: ${formatNumber(point.value)}${point.resolved ? ' (resolved)' : ' (resolving)'}${point.estimate ? ' est.' : ''}`}
					</title>
				</circle>
			{/each}
		</svg>
		<div class="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
			<span><span class="mr-1 inline-block h-2 w-2 rounded-full bg-positive align-middle"></span>resolved</span>
			<span><span class="mr-1 inline-block h-2 w-2 rounded-full border border-sky-400 align-middle"></span>resolving</span>
			<span>dashed outline = estimate</span>
			<span class="font-mono">first {points[0]?.roundLabel ?? DASH} → latest {points[points.length - 1]?.roundLabel ?? DASH}</span>
		</div>
	{/if}
</div>
