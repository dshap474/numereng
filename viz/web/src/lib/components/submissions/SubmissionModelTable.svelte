<!--
	Left-hand list of submitted models on the Submissions page.

	USAGE:
		<SubmissionModelTable items={sortedItems} selected={selectedModel} onSelect={(name) => (selectedModel = name)} />

	Read-only: clicking a row selects the model for the detail panel and, when the snapshot
	records a source experiment, navigates to that experiment page.
-->
<script lang="ts">
	import { goto } from '$app/navigation';
	import type { SubmissionItem } from '$lib/api/client';
	import Select from '$lib/components/ui/Select.svelte';
	import {
		formatNumber,
		formatShortDate,
		formatStatusLabel,
		signToneClass,
		type SubmissionSortKey
	} from './format';

	// ----------------------------------------------------------------------- //
	// Props
	// ----------------------------------------------------------------------- //

	interface Props {
		items: SubmissionItem[];
		selected: string | null;
		sortKey: SubmissionSortKey;
		onSelect: (modelName: string) => void;
		onSortChange: (sortKey: SubmissionSortKey) => void;
	}

	let { items, selected, sortKey, onSelect, onSortChange }: Props = $props();

	const sortOptions = [
		{ value: 'live_since', label: 'live since' },
		{ value: 'name', label: 'name' },
		{ value: 'mmc20', label: 'latest MMC20' },
		{ value: 'corr20', label: 'latest CORR20' },
		{ value: 'rounds', label: 'rounds' }
	];

	// `Select` exposes a bindable string; keep one local mirror and push changes upward.
	let sortValue = $state<string>(sortKey);

	$effect(() => {
		onSortChange(sortValue as SubmissionSortKey);
	});

	// ----------------------------------------------------------------------- //
	// Aggregates
	// ----------------------------------------------------------------------- //

	let totals = $derived.by(() => ({
		models: items.length,
		resolved: items.reduce((sum, item) => sum + (item.summary.resolved_round_count ?? 0), 0),
		resolving: items.reduce((sum, item) => sum + (item.summary.resolving_round_count ?? 0), 0),
		missingMetadata: items.filter((item) => item.summary.has_upload_metadata === false).length
	}));

	function rowTone(item: SubmissionItem): string {
		if ((item.summary.resolving_round_count ?? 0) > 0) return 'border-l-sky-400/70';
		if ((item.summary.resolved_round_count ?? 0) > 0) return 'border-l-emerald-400/80';
		return 'border-l-transparent';
	}

	// ----------------------------------------------------------------------- //
	// Row navigation
	// ----------------------------------------------------------------------- //

	function sourceHref(item: SubmissionItem): string | null {
		const experimentId = item.summary.source_experiment_id;
		return experimentId ? `/experiments/${encodeURIComponent(experimentId)}` : null;
	}

	// Selection updates first so the back button returns to a sane detail panel.
	function activateRow(item: SubmissionItem): void {
		onSelect(item.model_name);
		const href = sourceHref(item);
		if (href) void goto(href);
	}

	function onRowKeydown(event: KeyboardEvent, item: SubmissionItem): void {
		if (event.key !== 'Enter' && event.key !== ' ') return;
		event.preventDefault();
		activateRow(item);
	}
</script>

<!-- ------------------------------------------------------------------------ -->
<!-- Markup                                                                    -->
<!-- ------------------------------------------------------------------------ -->

<section
	class="flex min-h-0 flex-col border-b-[1.5px] border-white/12 bg-[#111115] xl:border-r-[1.5px] xl:border-b-0"
>
	<div
		class="flex shrink-0 flex-wrap items-end justify-between gap-3 border-y-[1.5px] border-white/12 bg-card px-5 py-4"
	>
		<div>
			<p class="font-mono text-[11px] uppercase tracking-[0.28em] text-muted-foreground">Live submissions</p>
			<h2 class="mt-2 text-lg font-semibold text-foreground">Submitted models</h2>
			<p class="mt-1 flex flex-wrap gap-x-3 text-xs text-muted-foreground">
				<span class="font-mono tabular-nums">{totals.models} models</span>
				<span class="font-mono tabular-nums">{totals.resolved} resolved rounds</span>
				<span class="font-mono tabular-nums">{totals.resolving} resolving</span>
				{#if totals.missingMetadata > 0}
					<span class="text-amber-300/90">{totals.missingMetadata} without upload metadata</span>
				{/if}
			</p>
		</div>
		<label class="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
			Sort
			<div class="mt-1 w-40">
				<Select options={sortOptions} bind:value={sortValue} size="xs" ariaLabel="Sort submitted models" />
			</div>
		</label>
	</div>

	<div class="min-h-0 flex-1 overflow-x-hidden overflow-y-auto">
		<table class="w-full table-fixed text-left text-sm">
			<thead
				class="sticky top-0 z-10 bg-card text-[11px] uppercase tracking-[0.16em] text-muted-foreground shadow-[inset_0_-1px_0_rgba(255,255,255,0.08)]"
			>
				<tr>
					<th class="w-[34%] px-5 py-3 font-medium">Model</th>
					<th class="w-[16%] px-3 py-3 font-medium">Live Since</th>
					<th class="w-[14%] px-3 py-3 text-right font-medium">Rounds</th>
					<th class="w-[18%] px-3 py-3 text-right font-medium">MMC20</th>
					<th class="w-[18%] px-5 py-3 text-right font-medium">CORR20</th>
				</tr>
			</thead>
			<tbody class="divide-y divide-white/6">
				{#each items as item (item.model_name)}
					<tr
						class="border-l-2 transition-colors hover:bg-white/[0.04] {sourceHref(item)
							? 'cursor-pointer'
							: 'cursor-default'} {selected === item.model_name ? 'bg-white/[0.045]' : ''} {rowTone(item)}"
						role={sourceHref(item) ? 'link' : 'button'}
						tabindex="0"
						title={sourceHref(item)
							? `Open source experiment ${item.summary.source_experiment_id}`
							: 'No source experiment recorded on this machine'}
						onclick={() => activateRow(item)}
						onkeydown={(event) => onRowKeydown(event, item)}
					>
						<td class="px-5 py-3">
							<div class="flex items-center gap-2">
								<span class="truncate font-mono text-sm font-semibold text-foreground" title={item.model_name}>
									{item.model_name}
								</span>
								{#if item.summary.data_version}
									<span
										class="shrink-0 rounded border border-white/10 px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground"
									>
										{item.summary.data_version}
									</span>
								{/if}
								{#if item.summary.has_upload_metadata === false}
									<span
										class="shrink-0 text-[11px] text-amber-300/90"
										title="upload metadata not recorded on this machine"
										aria-label="upload metadata not recorded on this machine">!</span
									>
								{/if}
							</div>
							<div class="mt-1">
								<span class="rounded bg-white/[0.05] px-1.5 py-0.5 text-[11px] text-muted-foreground">
									{formatStatusLabel(item.summary.status ?? item.summary.refresh_status ?? item.summary.role)}
								</span>
							</div>
						</td>
						<td class="truncate px-3 py-3 font-mono text-xs text-muted-foreground">
							{formatShortDate(item.summary.live_started_at)}
						</td>
						<td class="px-3 py-3 text-right font-mono tabular-nums text-foreground">{item.summary.round_count}</td>
						<td
							class="px-3 py-3 text-right font-mono tabular-nums {signToneClass(item.summary.latest_scored_round?.mmc20)}"
						>
							{formatNumber(item.summary.latest_scored_round?.mmc20)}
						</td>
						<td
							class="px-5 py-3 text-right font-mono tabular-nums {signToneClass(item.summary.latest_scored_round?.corr20)}"
						>
							{formatNumber(item.summary.latest_scored_round?.corr20)}
						</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
</section>
