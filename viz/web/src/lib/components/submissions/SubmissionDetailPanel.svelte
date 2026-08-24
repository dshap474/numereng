<!--
	Right-hand detail panel for one submitted model: deployment metadata, upload history,
	scoring aggregates, a metric trajectory, and the per-round table.

	USAGE:
		<SubmissionDetailPanel item={selectedItem} detail={selectedDetail} loading={detailLoading} />

	Renders as one continuous pane: a fixed pane header plus scrolling sections divided by single
	shared borders. Every metadata row is omitted when its value is missing, so thin snapshots
	(uploaded from another machine) render an honest "upload metadata not recorded" note, not "null".
-->
<script lang="ts">
	import type { SubmissionDetail, SubmissionItem, SubmissionRound, SubmissionUpload } from '$lib/api/client';
	import Select from '$lib/components/ui/Select.svelte';
	import SubmissionTrajectoryChart from './SubmissionTrajectoryChart.svelte';
	import {
		DASH,
		formatDate,
		formatNumber,
		formatPercentile,
		formatShortDate,
		formatStatusLabel,
		isScoredRound,
		meanMetric,
		roundNumber,
		roundState,
		roundWindow,
		signToneClass,
		type LiveRoundMetricKey
	} from './format';

	// ----------------------------------------------------------------------- //
	// Props and local state
	// ----------------------------------------------------------------------- //

	interface Props {
		item: SubmissionItem | null;
		detail: SubmissionDetail | null;
		loading: boolean;
	}

	let { item, detail, loading }: Props = $props();

	const metricOptions = [
		{ value: 'mmc20', label: 'MMC20' },
		{ value: 'corr20', label: 'CORR20' },
		{ value: 'mmc60', label: 'MMC60' },
		{ value: 'corr60', label: 'CORR60' },
		{ value: 'mmc', label: 'MMC' },
		{ value: 'corr', label: 'CORR' },
		{ value: 'bmc', label: 'BMC' },
		{ value: 'fnc', label: 'FNC' }
	];
	const scoreColumns: { key: LiveRoundMetricKey; label: string }[] = [
		{ key: 'mmc20', label: 'MMC20' },
		{ key: 'corr20', label: 'CORR20' },
		{ key: 'mmc60', label: 'MMC60' },
		{ key: 'corr60', label: 'CORR60' },
		{ key: 'bmc', label: 'BMC' },
		{ key: 'corr', label: 'CORR' },
		{ key: 'fnc', label: 'FNC' }
	];

	let trajectoryMetric = $state<string>('mmc20');

	// ----------------------------------------------------------------------- //
	// Derived views
	// ----------------------------------------------------------------------- //

	let rounds = $derived<SubmissionRound[]>(detail?.rounds ?? []);
	let scoredRounds = $derived(rounds.filter((row) => isScoredRound(row)));
	let resolvedRounds = $derived(rounds.filter((row) => roundState(row).toLowerCase() === 'resolved'));
	let resolvingRounds = $derived(rounds.filter((row) => roundState(row).toLowerCase() === 'resolving'));

	let uploads = $derived.by<SubmissionUpload[]>(() => {
		const raw = detail?.metadata?.uploads;
		return Array.isArray(raw) ? (raw.filter((entry) => entry && typeof entry === 'object') as SubmissionUpload[]) : [];
	});

	let metadataRows = $derived.by(() => {
		const summary = item?.summary;
		if (!summary) return [] as { label: string; value: string; mono?: boolean }[];
		const candidates: { label: string; value: unknown; mono?: boolean }[] = [
			{ label: 'Model id', value: summary.model_id, mono: true },
			{ label: 'Upload id', value: summary.upload_id, mono: true },
			{ label: 'Uploaded at', value: summary.uploaded_at ? formatDate(summary.uploaded_at) : null },
			{ label: 'Live since', value: summary.live_started_at ? formatDate(summary.live_started_at) : null },
			{ label: 'Live ended', value: summary.live_ended_at ? formatDate(summary.live_ended_at) : null },
			{ label: 'Data version', value: summary.data_version },
			{ label: 'Docker image', value: summary.docker_image },
			{ label: 'Source experiment', value: summary.source_experiment_id, mono: true },
			{ label: 'Package id', value: summary.source_package_id, mono: true },
			{ label: 'Package path', value: summary.source_package_path, mono: true },
			{ label: 'Pulled at', value: summary.pulled_at ? formatDate(summary.pulled_at) : null },
			{ label: 'Refresh source', value: summary.refresh_source },
			{ label: 'Latest resolved round', value: summary.latest_resolved_round },
			{ label: 'Latest scored round', value: summary.latest_scored_round_number }
		];
		return candidates
			.filter((row) => row.value !== null && row.value !== undefined && row.value !== '')
			.map((row) => ({ label: row.label, value: String(row.value), mono: row.mono === true }));
	});

	let aggregates = $derived.by(() => [
		{ label: 'Resolved MMC20', value: meanMetric(resolvedRounds, 'mmc20') },
		{ label: 'Resolved CORR20', value: meanMetric(resolvedRounds, 'corr20') },
		{ label: 'Scored MMC20', value: meanMetric(scoredRounds, 'mmc20') },
		{ label: 'Scored CORR20', value: meanMetric(scoredRounds, 'corr20') }
	]);

	function experimentHref(experimentId: string): string {
		return `/experiments/${encodeURIComponent(experimentId)}`;
	}
</script>

<!-- ------------------------------------------------------------------------ -->
<!-- Markup                                                                    -->
<!-- ------------------------------------------------------------------------ -->

<section class="flex min-h-0 flex-col bg-[#111115]">
	{#if !item}
		<div class="flex flex-1 items-center justify-center px-6 py-10 text-center text-sm text-muted-foreground">
			Select a model to see its deployment metadata and live rounds.
		</div>
	{:else}
		<div class="shrink-0 border-y-[1.5px] border-white/12 bg-card px-5 py-4">
			<div class="flex items-start justify-between gap-3">
				<div class="min-w-0">
					<p class="font-mono text-[11px] uppercase tracking-[0.28em] text-muted-foreground">Selected model</p>
					<h2 class="mt-2 truncate font-mono text-lg font-semibold text-foreground">{item.model_name}</h2>
					<p class="mt-1 text-xs text-muted-foreground">
						{formatStatusLabel(item.summary.status ?? item.summary.refresh_status)}
					</p>
				</div>
				<span class="shrink-0 rounded-md border border-white/10 px-2 py-1 text-xs text-muted-foreground">
					{loading ? 'loading' : `${rounds.length} rounds`}
				</span>
			</div>
		</div>

		<div class="min-h-0 flex-1 overflow-y-auto">
			{#if item.summary.has_upload_metadata === false}
				<p class="border-b border-white/8 bg-amber-400/[0.06] px-5 py-3 text-xs text-amber-200/90">
					Upload metadata not recorded on this machine. Only the refreshed live-round snapshot is available.
				</p>
			{/if}

			{#if metadataRows.length > 0}
				<dl class="divide-y divide-white/6 border-b border-white/8">
					{#each metadataRows as row (row.label)}
						<div class="flex items-baseline justify-between gap-4 px-5 py-2 odd:bg-white/[0.015]">
							<dt class="shrink-0 text-[11px] uppercase tracking-[0.14em] text-muted-foreground">{row.label}</dt>
							<dd
								class="min-w-0 truncate text-right text-xs text-foreground {row.mono ? 'font-mono' : ''}"
								title={row.value}
							>
								{#if row.label === 'Source experiment'}
									<a
										class="underline decoration-white/20 underline-offset-2 hover:text-foreground"
										href={experimentHref(row.value)}
									>
										{row.value}
									</a>
								{:else}
									{row.value}
								{/if}
							</dd>
						</div>
					{/each}
				</dl>
			{/if}

			{#if uploads.length > 1}
				<div class="border-b border-white/8">
					<h3 class="px-5 pt-4 pb-2 text-sm font-semibold text-foreground">Upload history</h3>
					<table class="w-full table-fixed text-left text-sm">
						<thead class="bg-white/[0.025] text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
							<tr>
								<th class="w-[46%] px-5 py-2 font-medium">Upload</th>
								<th class="w-[27%] px-3 py-2 font-medium">Live started</th>
								<th class="w-[27%] px-5 py-2 font-medium">Live ended</th>
							</tr>
						</thead>
						<tbody class="divide-y divide-white/6">
							{#each uploads as upload, index (upload.upload_id ?? index)}
								<tr>
									<td class="truncate px-5 py-2 font-mono text-xs text-foreground" title={upload.upload_id ?? ''}>
										{upload.upload_id ?? DASH}
									</td>
									<td class="px-3 py-2 text-xs text-muted-foreground">{formatShortDate(upload.live_started_at)}</td>
									<td class="px-5 py-2 text-xs text-muted-foreground">{formatShortDate(upload.live_ended_at)}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{/if}

			{#if rounds.length === 0}
				<div class="px-5 py-5">
					<p class="text-sm font-medium text-foreground">Awaiting live scores.</p>
					<p class="mt-1 text-sm text-muted-foreground">
						{loading
							? 'Loading round snapshot.'
							: 'No live rounds have been pulled for this model yet. Refresh with `uv run numereng submissions refresh`.'}
					</p>
				</div>
			{:else}
				<div class="grid grid-cols-2 border-b border-white/8 sm:grid-cols-3 xl:grid-cols-6">
					<div class="border-r border-b border-white/6 px-4 py-3">
						<p class="text-[11px] text-muted-foreground">Resolved</p>
						<p class="mt-1 font-mono text-base tabular-nums text-foreground">{resolvedRounds.length}</p>
					</div>
					<div class="border-r border-b border-white/6 px-4 py-3">
						<p class="text-[11px] text-muted-foreground">Resolving</p>
						<p class="mt-1 font-mono text-base tabular-nums text-foreground">{resolvingRounds.length}</p>
					</div>
					{#each aggregates as aggregate (aggregate.label)}
						<div class="border-r border-b border-white/6 px-4 py-3">
							<p class="text-[11px] text-muted-foreground">{aggregate.label}</p>
							<p class="mt-1 font-mono text-base tabular-nums {signToneClass(aggregate.value)}">
								{formatNumber(aggregate.value)}
							</p>
						</div>
					{/each}
				</div>

				<div class="border-b border-white/8 px-5 py-4">
					<div class="flex items-end justify-end">
						<label class="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
							Trajectory metric
							<div class="mt-1 w-32">
								<Select options={metricOptions} bind:value={trajectoryMetric} size="xs" ariaLabel="Trajectory metric" />
							</div>
						</label>
					</div>
					<div class="mt-2">
						<SubmissionTrajectoryChart
							{rounds}
							metric={trajectoryMetric as LiveRoundMetricKey}
							label={metricOptions.find((option) => option.value === trajectoryMetric)?.label ?? 'MMC20'}
						/>
					</div>
				</div>

				<div class="overflow-x-auto">
				<table class="w-full text-left text-sm">
					<thead
						class="sticky top-0 z-10 bg-card text-[11px] uppercase tracking-[0.16em] text-muted-foreground shadow-[inset_0_-1px_0_rgba(255,255,255,0.08)]"
					>
						<tr>
							<th class="px-5 py-2 font-medium">Round</th>
							<th class="px-2 py-2 font-medium">State</th>
							{#each scoreColumns as column (column.key)}
								<th class="px-2 py-2 text-right font-medium">{column.label}</th>
							{/each}
						</tr>
					</thead>
					<tbody class="divide-y divide-white/6">
						{#each rounds as row (roundNumber(row))}
							<tr>
								<td class="px-5 py-2">
									<div class="font-mono tabular-nums text-foreground">{roundNumber(row)}</div>
									<div class="mt-0.5 text-[11px] text-muted-foreground">{roundWindow(row)}</div>
								</td>
								<td class="px-2 py-2">
									<span
										class="rounded px-1.5 py-0.5 text-[11px] {roundState(row).toLowerCase() === 'resolved'
											? 'bg-emerald-400/12 text-emerald-300'
											: 'bg-sky-400/12 text-sky-300'}"
									>
										{roundState(row)}
									</span>
									{#if row.is_estimate === true}
										<span class="ml-1 text-[10px] text-muted-foreground" title="estimated score">est.</span>
									{/if}
								</td>
								{#each scoreColumns as column (column.key)}
									<td class="px-2 py-2 text-right last:pr-5">
										<div class="font-mono tabular-nums {signToneClass(row[column.key])}">
											{formatNumber(row[column.key])}
										</div>
										{#if formatPercentile(row[`${column.key}_percentile`])}
											<div class="mt-0.5 font-mono text-[10px] text-muted-foreground">
												{formatPercentile(row[`${column.key}_percentile`])}
											</div>
										{/if}
									</td>
								{/each}
							</tr>
						{/each}
					</tbody>
				</table>
				</div>
			{/if}
		</div>
	{/if}
</section>
