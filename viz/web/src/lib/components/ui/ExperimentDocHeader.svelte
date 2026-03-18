<script lang="ts">
	import { renderMarkdownInline } from '$lib/markdown/render';
	import {
		normalizedHeaderLabel,
		type ExperimentDocContext,
		type ExperimentDocHeaderEntry
	} from '$lib/markdown/experiment';

	let {
		title,
		headerEntries = [],
		context
	}: {
		title: string | null;
		headerEntries?: ExperimentDocHeaderEntry[];
		context: ExperimentDocContext;
	} = $props();

	const HEADER_FIELDS = new Set(['id', 'created', 'updated', 'status', 'champion run', 'champion', 'tags']);

	const entryMap = $derived.by(() => {
		const map = new Map<string, string>();
		for (const entry of headerEntries) {
			map.set(normalizedHeaderLabel(entry.label), entry.value);
		}
		return map;
	});

	const displayTitle = $derived(title?.trim() || context.name || context.experimentId);
	const displayCreated = $derived(formatDate(context.createdAt || entryMap.get('created') || null, false));
	const extraEntries = $derived(
		headerEntries.filter((entry) => {
			const normalized = normalizedHeaderLabel(entry.label);
			if (
				normalized === 'tags' ||
				normalized === 'champion' ||
				normalized === 'champion run' ||
				normalized === 'id' ||
				normalized === 'status'
			) {
				return false;
			}
			return !HEADER_FIELDS.has(normalized);
		})
	);
	const statItems = $derived([
		{ label: 'Runs', value: `${context.stats.completedRuns}/${context.stats.totalRuns}` },
		{ label: 'HPO Studies', value: String(context.stats.studyCount) },
		{ label: 'Ensembles', value: String(context.stats.ensembleCount) }
	]);

	function formatDate(value: string | null, includeTime: boolean): string {
		if (!value) return '—';
		const parsed = new Date(value);
		if (Number.isNaN(parsed.getTime())) return value;
		const formatter = new Intl.DateTimeFormat('en-US', {
			year: 'numeric',
			month: 'short',
			day: '2-digit',
			...(includeTime
				? {
						hour: '2-digit',
						minute: '2-digit',
						timeZoneName: 'short'
					}
				: {})
		});
		return formatter.format(parsed);
	}

</script>

<section class="mb-4 pb-0">
	<div class="space-y-3">
		<div>
			<h1 class="text-4xl leading-tight font-semibold tracking-tight text-foreground">{displayTitle}</h1>
		</div>

		<div class="space-y-2 text-sm">
			<div class="flex flex-wrap gap-x-8 gap-y-2">
				<div class="flex items-baseline gap-3">
					<span class="text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">Created</span>
					<span class="font-medium text-foreground">{displayCreated}</span>
				</div>
			</div>

			<div class="flex flex-wrap gap-x-8 gap-y-2">
				{#each statItems as item (item.label)}
					<div class="flex items-baseline gap-3">
						<span class="text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">{item.label}</span>
						<span class="tabular-nums font-medium text-foreground">{item.value}</span>
					</div>
				{/each}
			</div>
		</div>

		<div class="space-y-0">
			{#if extraEntries.length > 0}
				<div class="grid gap-2 py-3 md:grid-cols-[140px_minmax(0,1fr)] md:items-start">
					<div class="text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">Notes</div>
					<div class="space-y-2 text-sm leading-6 text-foreground experiment-header-inline">
						{#each extraEntries as entry (entry.label)}
							<div>
								<span class="mr-2 font-semibold text-foreground">{entry.label}:</span>
								{@html renderMarkdownInline(entry.value)}
							</div>
						{/each}
					</div>
				</div>
			{/if}
		</div>
	</div>
</section>
