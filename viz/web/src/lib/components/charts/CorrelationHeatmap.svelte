<script lang="ts">
	interface Props {
		labels: string[];
		matrix: (number | null)[][];
		class?: string;
	}

	let { labels, matrix, class: className = '' }: Props = $props();

	let n = $derived(labels.length);
	let cellSize = $derived(Math.min(40, 300 / Math.max(n, 1)));
	let labelWidth = $derived(Math.min(120, Math.max(...labels.map((l) => l.length * 6), 40)));

	function color(value: number | null): string {
		if (value === null) {
			return 'rgb(203,213,225)';
		}
		// Blue (-1) -> White (0) -> Red (+1)
		const clamped = Math.max(-1, Math.min(1, value));
		const t = (clamped + 1) / 2; // normalize to [0, 1]
		if (t < 0.5) {
			const s = t / 0.5;
			const r = Math.round(59 + s * (255 - 59));
			const g = Math.round(130 + s * (255 - 130));
			const b = Math.round(246 + s * (255 - 246));
			return `rgb(${r},${g},${b})`;
		} else {
			const s = (t - 0.5) / 0.5;
			const r = Math.round(255 - s * (255 - 239));
			const g = Math.round(255 - s * (255 - 68));
			const b = Math.round(255 - s * (255 - 68));
			return `rgb(${r},${g},${b})`;
		}
	}
</script>

<div class="w-full overflow-auto {className}" aria-label="Correlation heatmap">
	{#if n === 0}
		<div class="flex h-full items-center justify-center text-muted-foreground text-sm">No data</div>
	{:else}
		<svg
			width={labelWidth + n * cellSize + 10}
			height={labelWidth + n * cellSize + 10}
			class="mx-auto"
		>
			<!-- Column labels (top) -->
			{#each labels as label, j}
				<text
					x={labelWidth + j * cellSize + cellSize / 2}
					y={labelWidth - 4}
					text-anchor="end"
					transform="rotate(-45, {labelWidth + j * cellSize + cellSize / 2}, {labelWidth - 4})"
					class="fill-muted-foreground text-[9px]"
				>
					{label.length > 15 ? label.slice(0, 15) + '..' : label}
				</text>
			{/each}

			<!-- Row labels + cells -->
			{#each labels as rowLabel, i}
				<text
					x={labelWidth - 4}
					y={labelWidth + i * cellSize + cellSize / 2 + 3}
					text-anchor="end"
					class="fill-muted-foreground text-[9px]"
				>
					{rowLabel.length > 15 ? rowLabel.slice(0, 15) + '..' : rowLabel}
				</text>

				{#each matrix[i] ?? [] as value, j}
					<rect
						x={labelWidth + j * cellSize}
						y={labelWidth + i * cellSize}
						width={cellSize - 1}
						height={cellSize - 1}
						fill={color(value)}
						rx="2"
					>
						<title>{labels[i]} vs {labels[j]}: {value === null ? 'n/a' : value.toFixed(3)}</title>
					</rect>
					{#if cellSize >= 24}
						<text
							x={labelWidth + j * cellSize + cellSize / 2 - 0.5}
							y={labelWidth + i * cellSize + cellSize / 2 + 3}
							text-anchor="middle"
							class="text-[8px] {value !== null && Math.abs(value) > 0.7 ? 'fill-white' : 'fill-foreground'}"
						>
							{value === null ? '-' : value.toFixed(2)}
						</text>
					{/if}
				{/each}
			{/each}
		</svg>
	{/if}
</div>
