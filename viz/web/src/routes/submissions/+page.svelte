<script lang="ts">
	import { api, type SubmissionDetail, type SubmissionItem, type SubmissionListResponse, type SubmissionRound } from '$lib/api/client';
	import AccentCard from '$lib/components/ui/AccentCard.svelte';

	let {
		data
	}: {
		data: {
			submissions: SubmissionListResponse;
		};
	} = $props();

	const routeSubmissions = () => data.submissions;
	let submissions = $state<SubmissionListResponse>(routeSubmissions());
	let selectedModel = $state<string | null>(routeSubmissions().items[0]?.model_name ?? null);
	let selectedDetail = $state<SubmissionDetail | null>(null);
	let detailLoading = $state(false);

	let selectedItem = $derived(submissions.items.find((item) => item.model_name === selectedModel) ?? null);
	let rounds = $derived(selectedDetail?.rounds ?? []);

	$effect(() => {
		submissions = routeSubmissions();
		if (selectedModel == null && submissions.items.length > 0) {
			selectedModel = submissions.items[0].model_name;
		}
	});

	$effect(() => {
		const modelName = selectedModel;
		if (!modelName) {
			selectedDetail = null;
			return;
		}

		let active = true;
		detailLoading = true;
		api
			.getSubmission(modelName)
			.then((payload) => {
				if (active) selectedDetail = payload;
			})
			.catch(() => {
				if (active) selectedDetail = null;
			})
			.finally(() => {
				if (active) detailLoading = false;
			});

		return () => {
			active = false;
		};
	});

	function formatNumber(value: unknown, digits = 4): string {
		if (typeof value !== 'number' || Number.isNaN(value)) return 'n/a';
		return value.toFixed(digits);
	}

	function formatText(value: unknown): string {
		if (typeof value === 'string' && value.trim()) return value;
		if (typeof value === 'number') return String(value);
		return 'n/a';
	}

	function roundNumber(row: SubmissionRound | null | undefined): string {
		return formatText(row?.round ?? row?.round_number);
	}

	function formatDate(value: unknown): string {
		if (typeof value !== 'string' || !value.trim()) return 'n/a';
		const calendarMatch = value.match(/^(\d{4})-(\d{2})-(\d{2})$/);
		const date = calendarMatch
			? new Date(Number(calendarMatch[1]), Number(calendarMatch[2]) - 1, Number(calendarMatch[3]))
			: new Date(value);
		if (Number.isNaN(date.getTime())) return value;
		return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
	}

	function roundDates(row: SubmissionRound | null | undefined): string {
		const closeDate = formatDate(row?.close_date);
		const resolveDate = formatDate(row?.resolve_date);
		if (closeDate === 'n/a' && resolveDate === 'n/a') return 'dates n/a';
		if (resolveDate === 'n/a') return `close ${closeDate}`;
		if (closeDate === 'n/a') return `resolve ${resolveDate}`;
		return `${closeDate} -> ${resolveDate}`;
	}

	function roundState(row: SubmissionRound | null | undefined): string {
		return formatText(row?.state ?? row?.status);
	}

	function rowTone(item: SubmissionItem): string {
		if (item.summary.resolving_round_count > 0) return 'border-l-sky-400/70';
		if (item.summary.resolved_round_count > 0) return 'border-l-emerald-400/80';
		return 'border-l-transparent';
	}
</script>

<svelte:head>
	<title>Submissions · Numereng</title>
</svelte:head>

<div class="space-y-6">
	<header class="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
		<div>
			<p class="text-[11px] uppercase tracking-[0.22em] text-muted-foreground">Live submission snapshots</p>
			<h1 class="mt-1 text-2xl font-semibold tracking-tight text-foreground">Submissions</h1>
		</div>
		<div class="text-sm text-muted-foreground">
			<span class="font-mono text-foreground">{submissions.total}</span>
			<span>{submissions.total === 1 ? 'model' : 'models'}</span>
		</div>
	</header>

	{#if submissions.items.length === 0}
		<AccentCard paddingClass="px-5 py-5" roundedClass="rounded-lg">
			<div class="flex flex-col gap-2">
				<p class="text-sm font-medium text-foreground">No submission snapshots found.</p>
				<p class="text-sm text-muted-foreground">
					Expected local folders under <code class="font-mono text-foreground">{submissions.root}</code>.
				</p>
			</div>
		</AccentCard>
	{:else}
		<div class="grid gap-4 lg:grid-cols-[minmax(0,1.05fr)_minmax(360px,0.95fr)]">
			<AccentCard paddingClass="p-0" roundedClass="rounded-lg" class="overflow-hidden">
				<div class="border-b border-white/8 px-4 py-3">
					<h2 class="text-sm font-semibold text-foreground">Submitted Models</h2>
				</div>
				<div class="overflow-x-auto">
					<table class="min-w-full text-left text-sm">
						<thead class="bg-white/[0.025] text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
							<tr>
								<th class="px-4 py-3 font-medium">Model</th>
								<th class="px-4 py-3 text-right font-medium">Rounds</th>
								<th class="px-4 py-3 text-right font-medium">MMC20</th>
								<th class="px-4 py-3 text-right font-medium">CORR20</th>
							</tr>
						</thead>
						<tbody class="divide-y divide-white/6">
							{#each submissions.items as item (item.model_name)}
								<tr
									class="cursor-pointer border-l transition-colors hover:bg-white/[0.04] {selectedModel === item.model_name ? 'bg-white/[0.045]' : ''} {rowTone(item)}"
									onclick={() => (selectedModel = item.model_name)}
								>
									<td class="px-4 py-3">
										<div class="font-mono text-sm font-semibold text-foreground">{item.model_name}</div>
										<div class="mt-1 text-xs text-muted-foreground">{formatText(item.summary.status ?? item.summary.role)}</div>
									</td>
									<td class="px-4 py-3 text-right font-mono text-foreground">{item.summary.round_count}</td>
									<td class="px-4 py-3 text-right font-mono text-foreground">
										{formatNumber(item.summary.latest_scored_round?.mmc20)}
									</td>
									<td class="px-4 py-3 text-right font-mono text-foreground">
										{formatNumber(item.summary.latest_scored_round?.corr20)}
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</AccentCard>

			<AccentCard paddingClass="px-4 py-4" roundedClass="rounded-lg">
				{#if selectedItem}
					<div class="flex items-start justify-between gap-3">
						<div>
							<p class="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">Selected model</p>
							<h2 class="mt-1 font-mono text-lg font-semibold text-foreground">{selectedItem.model_name}</h2>
						</div>
						<span class="rounded-md border border-white/10 px-2 py-1 text-xs text-muted-foreground">
							{detailLoading ? 'loading' : `${rounds.length} rows`}
						</span>
					</div>

					<div class="mt-4 grid grid-cols-2 gap-3">
						<div class="rounded-md border border-white/8 bg-white/[0.025] px-3 py-3">
							<p class="text-xs text-muted-foreground">Resolving</p>
							<p class="mt-1 font-mono text-lg text-foreground">{selectedItem.summary.resolving_round_count}</p>
						</div>
						<div class="rounded-md border border-white/8 bg-white/[0.025] px-3 py-3">
							<p class="text-xs text-muted-foreground">Resolved</p>
							<p class="mt-1 font-mono text-lg text-foreground">{selectedItem.summary.resolved_round_count}</p>
						</div>
					</div>

					<div class="mt-4 overflow-x-auto">
						<table class="min-w-full text-left text-sm">
							<thead class="text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
								<tr>
									<th class="py-2 pr-3 font-medium">Round / Dates</th>
									<th class="py-2 pr-3 font-medium">State</th>
									<th class="py-2 pr-3 text-right font-medium">MMC20</th>
									<th class="py-2 pr-3 text-right font-medium">CORR20</th>
								</tr>
							</thead>
							<tbody class="divide-y divide-white/6">
								{#each rounds.slice(0, 14) as row}
									<tr>
										<td class="py-2 pr-3">
											<div class="font-mono text-foreground">{roundNumber(row)}</div>
											<div class="mt-0.5 text-xs text-muted-foreground">{roundDates(row)}</div>
										</td>
										<td class="py-2 pr-3 text-muted-foreground">{roundState(row)}</td>
										<td class="py-2 pr-3 text-right font-mono text-foreground">{formatNumber(row.mmc20)}</td>
										<td class="py-2 pr-3 text-right font-mono text-foreground">{formatNumber(row.corr20)}</td>
									</tr>
								{/each}
								{#if rounds.length === 0}
									<tr>
										<td class="py-4 text-sm text-muted-foreground" colspan="4">No round rows found.</td>
									</tr>
								{/if}
							</tbody>
						</table>
					</div>
				{/if}
			</AccentCard>
		</div>
	{/if}
</div>
