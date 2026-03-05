<script lang="ts">
	interface Trial {
		number?: number;
		trial_number?: number;
		value?: number;
		state?: string;
		[key: string]: unknown;
	}

	interface Props {
		trials: Trial[];
		class?: string;
	}

	let { trials, class: className = '' }: Props = $props();

	let completedTrials = $derived(
		trials
			.filter((t) => (t.state === 'COMPLETE' || t.state == null) && t.value != null)
			.sort((a, b) => (a.number ?? a.trial_number ?? 0) - (b.number ?? b.trial_number ?? 0))
	);

	let bestSoFar = $derived.by(() => {
		let best = -Infinity;
		return completedTrials.map((t) => {
			if (t.value != null && t.value > best) best = t.value;
			return { ...t, best_so_far: best };
		});
	});

	let yMin = $derived(Math.min(...completedTrials.map((t) => t.value ?? 0)) * 0.95);
	let yMax = $derived(Math.max(...completedTrials.map((t) => t.value ?? 0)) * 1.05);
	let xMax = $derived(completedTrials.length);

	function scaleX(i: number, width: number): number {
		if (xMax <= 1) return width / 2;
		return (i / (xMax - 1)) * (width - 40) + 20;
	}

	function scaleY(v: number, height: number): number {
		const range = yMax - yMin || 1;
		return height - 20 - ((v - yMin) / range) * (height - 40);
	}
</script>

<div class="w-full h-full {className}" aria-label="Trial convergence chart">
	{#if completedTrials.length === 0}
		<div class="flex h-full items-center justify-center text-muted-foreground text-sm">No completed trials</div>
	{:else}
		<svg viewBox="0 0 600 300" preserveAspectRatio="xMidYMid meet" class="w-full h-full">
			<!-- Grid lines -->
			{#each [0.25, 0.5, 0.75] as frac}
				<line
					x1="20" y1={300 - 20 - frac * 260}
					x2="580" y2={300 - 20 - frac * 260}
					stroke="currentColor" stroke-opacity="0.1" stroke-width="0.5"
				/>
			{/each}

			<!-- Trial values (dots) -->
			{#each bestSoFar as t, i}
				<circle
					cx={scaleX(i, 600)}
					cy={scaleY(t.value ?? 0, 300)}
					r="2.5"
					fill="var(--color-primary)"
					opacity="0.4"
				/>
			{/each}

			<!-- Best-so-far line -->
			{#if bestSoFar.length > 1}
				<polyline
					points={bestSoFar.map((t, i) => `${scaleX(i, 600)},${scaleY(t.best_so_far, 300)}`).join(' ')}
					fill="none"
					stroke="var(--color-positive)"
					stroke-width="2"
				/>
			{/if}

			<!-- Y-axis labels -->
			<text x="16" y="20" text-anchor="end" class="fill-muted-foreground text-[9px]">{yMax.toFixed(3)}</text>
			<text x="16" y="290" text-anchor="end" class="fill-muted-foreground text-[9px]">{yMin.toFixed(3)}</text>

			<!-- X-axis label -->
			<text x="300" y="298" text-anchor="middle" class="fill-muted-foreground text-[9px]">Trial</text>
		</svg>
	{/if}
</div>
