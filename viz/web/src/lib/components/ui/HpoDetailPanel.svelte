<script lang="ts">
	import { api, type HpoStudy, type HpoTrial } from '$lib/api/client';
	import TrialConvergenceChart from '$lib/components/charts/TrialConvergenceChart.svelte';
	import { fmt } from '$lib/utils';

	let {
		studyId,
		experimentId,
		onClose
	}: {
		studyId: string;
		experimentId: string;
		onClose: () => void;
	} = $props();

	let loading = $state(true);
	let error = $state<string | null>(null);
	let study = $state<HpoStudy | null>(null);
	let trials = $state<HpoTrial[]>([]);

	$effect(() => {
		const id = studyId;
		loading = true;
		error = null;
		study = null;
		trials = [];
		Promise.all([api.getStudy(id), api.getStudyTrials(id)]).then(
			([s, t]) => {
				if (studyId !== id) return;
				study = s;
				trials = t;
				loading = false;
			},
			(err) => {
				if (studyId !== id) return;
				error = err instanceof Error ? err.message : 'Failed to load study.';
				loading = false;
			}
		);
	});

	let bestTrial = $derived.by(() => {
		if (study?.best_trial_number == null) return null;
		return trials.find(
			(t) => (t.number ?? t.trial_number) === study!.best_trial_number
		) ?? null;
	});

	let bestParams = $derived.by(() => {
		if (!bestTrial) return null;
		const params = (bestTrial.params ?? bestTrial.user_attrs ?? {}) as Record<string, unknown>;
		if (Object.keys(params).length === 0) return null;
		return params;
	});

	function statusClass(status: string | null): string {
		switch (status) {
			case 'completed': return 'bg-positive/15 text-positive';
			case 'running': return 'bg-blue-500/15 text-blue-400';
			case 'failed': return 'bg-negative/15 text-negative';
			default: return 'bg-muted text-muted-foreground';
		}
	}

	function trialStateClass(state: string | undefined): string {
		switch (state) {
			case 'COMPLETE': return 'text-positive';
			case 'FAIL': return 'text-negative';
			case 'PRUNED': return 'text-amber-400';
			default: return 'text-muted-foreground';
		}
	}

	function fmtDuration(seconds: number | undefined | null): string {
		if (seconds == null) return '-';
		if (seconds < 60) return `${seconds.toFixed(0)}s`;
		const m = Math.floor(seconds / 60);
		const s = seconds % 60;
		return `${m}m ${s.toFixed(0)}s`;
	}
</script>

{#if loading}
	<div class="flex items-center justify-center h-full text-muted-foreground text-sm">
		Loading study details...
	</div>
{:else if error}
	<div class="flex flex-col items-center justify-center h-full gap-3">
		<p class="text-sm text-negative">{error}</p>
		<button
			type="button"
			class="text-sm text-muted-foreground hover:text-foreground underline underline-offset-2"
			onclick={onClose}
		>&larr; Back</button>
	</div>
{:else if study}
	<div class="space-y-6 min-w-0">
		<header class="sticky top-0 z-20 -mx-2 px-2 py-3 bg-background/90 backdrop-blur border-b border-border/60">
			<div class="flex items-start justify-between gap-3">
				<div>
					<div class="flex items-center gap-2">
						<span class="inline-block rounded px-1.5 py-0.5 text-[10px] uppercase bg-violet-500/20 text-violet-300">HPO</span>
						<h1 class="text-xl font-semibold">{study.name || study.study_id}</h1>
						<span class="inline-block rounded px-2 py-0.5 text-xs uppercase {statusClass(study.status)}">
							{study.status ?? 'unknown'}
						</span>
					</div>
					<div class="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
						<span class="font-mono">{study.study_id}</span>
					</div>
				</div>
				<button
					type="button"
					class="rounded-md border border-border px-2.5 py-1.5 text-xs text-muted-foreground hover:text-foreground hover:bg-muted/30 whitespace-nowrap"
					onclick={onClose}
				>&larr; Back</button>
			</div>
		</header>

		<!-- Summary cards -->
		<div class="grid grid-cols-2 md:grid-cols-4 gap-3">
			<div class="rounded-lg border border-border bg-card px-4 py-3">
				<div class="text-xs text-muted-foreground uppercase">Mode</div>
				<div class="mt-1 font-medium">{study.mode ?? '-'}</div>
			</div>
			<div class="rounded-lg border border-border bg-card px-4 py-3">
				<div class="text-xs text-muted-foreground uppercase">Trials</div>
				<div class="mt-1 font-medium">{study.n_completed ?? 0} / {study.n_trials ?? '?'}</div>
			</div>
			<div class="rounded-lg border border-border bg-card px-4 py-3">
				<div class="text-xs text-muted-foreground uppercase">Best Value</div>
				<div class="mt-1 font-medium tabular-nums">{fmt(study.best_value)}</div>
			</div>
			<div class="rounded-lg border border-border bg-card px-4 py-3">
				<div class="text-xs text-muted-foreground uppercase">Best Run</div>
				<div class="mt-1 font-medium">
					{#if study.best_run_id}
						<a
							href="/experiments/{experimentId}/runs/{study.best_run_id}"
							class="text-primary underline underline-offset-2"
						>{study.best_run_id.slice(0, 12)}</a>
					{:else}
						-
					{/if}
				</div>
			</div>
		</div>

		<!-- Convergence chart -->
		{#if trials.length > 0}
			<div class="rounded-lg border border-border bg-card p-4">
				<h2 class="text-sm font-medium mb-3">Trial Convergence</h2>
				<div class="h-64">
					<TrialConvergenceChart {trials} class="h-full" />
				</div>
			</div>
		{/if}

		<!-- Best trial params -->
		{#if bestParams}
			<div class="rounded-lg border border-border bg-card p-4">
				<h2 class="text-sm font-medium mb-3">Best Trial #{study.best_trial_number} Parameters</h2>
				<div class="grid grid-cols-2 md:grid-cols-3 gap-2">
					{#each Object.entries(bestParams) as [key, val] (key)}
						<div class="rounded-md border border-border/50 bg-background/50 px-3 py-2">
							<div class="text-[10px] text-muted-foreground">{key}</div>
							<div class="text-xs font-mono tabular-nums mt-0.5">
								{typeof val === 'number' ? val.toPrecision(5) : String(val)}
							</div>
						</div>
					{/each}
				</div>
			</div>
		{/if}

		<!-- Trials table -->
		{#if trials.length > 0}
			<div class="rounded-lg border border-border bg-card overflow-hidden">
				<h2 class="text-sm font-medium px-4 py-3 border-b border-border">Trials ({trials.length})</h2>
				<div class="overflow-auto max-h-96">
					<table class="w-full text-xs">
						<thead class="sticky top-0 bg-card">
							<tr class="border-b border-border text-left">
								<th class="px-3 py-2 text-muted-foreground font-medium">#</th>
								<th class="px-3 py-2 text-muted-foreground font-medium">State</th>
								<th class="px-3 py-2 text-muted-foreground font-medium text-right">Value</th>
								<th class="px-3 py-2 text-muted-foreground font-medium text-right">Duration</th>
							</tr>
						</thead>
						<tbody class="divide-y divide-border/50">
							{#each trials as trial, i (i)}
								{@const num = (trial.number ?? trial.trial_number ?? i) as number}
								{@const state = (trial.state ?? 'COMPLETE') as string}
								{@const value = trial.value as number | undefined}
								{@const duration = trial.duration as number | undefined}
								<tr class="hover:bg-muted/20 {num === study.best_trial_number ? 'bg-primary/5' : ''}">
									<td class="px-3 py-1.5 tabular-nums">{num}</td>
									<td class="px-3 py-1.5 {trialStateClass(state)}">{state}</td>
									<td class="px-3 py-1.5 text-right tabular-nums">{value != null ? value.toFixed(5) : '-'}</td>
									<td class="px-3 py-1.5 text-right tabular-nums">{fmtDuration(duration)}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</div>
		{/if}
	</div>
{/if}
