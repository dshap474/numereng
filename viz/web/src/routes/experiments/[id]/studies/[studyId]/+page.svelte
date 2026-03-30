<script lang="ts">
	import type { HpoStudy, HpoTrial } from '$lib/api/client';
	import TrialConvergenceChart from '$lib/components/charts/TrialConvergenceChart.svelte';
	import { withSourceHref, type SourceContext } from '$lib/source';
	import { fmt } from '$lib/utils';

	let {
		data
	}: {
		data: {
			experimentId: string;
			study: HpoStudy;
			trials: HpoTrial[];
			source: SourceContext;
		};
	} = $props();

	let bestTrial = $derived.by(() => {
		if (data.study.best_trial_number == null) return null;
		return data.trials.find(
			(t) => (t.number ?? t.trial_number) === data.study.best_trial_number
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

<div class="space-y-6">
	<div class="flex items-center gap-3">
		<a
			href={withSourceHref(`/experiments/${data.experimentId}`, data.source)}
			class="text-sm text-muted-foreground hover:text-foreground transition-colors"
		>&larr; Experiment</a>
		<h1 class="text-xl font-semibold">{data.study.name || data.study.study_id}</h1>
		<span class="inline-block rounded px-2 py-0.5 text-xs uppercase {statusClass(data.study.status)}">
			{data.study.status ?? 'unknown'}
		</span>
	</div>

	<!-- Summary cards -->
	<div class="grid grid-cols-2 md:grid-cols-4 gap-3">
		<div class="rounded-lg border border-border bg-card px-4 py-3">
			<div class="text-xs text-muted-foreground uppercase">Mode</div>
			<div class="mt-1 font-medium">{data.study.mode ?? '-'}</div>
		</div>
		<div class="rounded-lg border border-border bg-card px-4 py-3">
			<div class="text-xs text-muted-foreground uppercase">Trials</div>
			<div class="mt-1 font-medium">{data.study.n_completed ?? 0} / {data.study.n_trials ?? '?'}</div>
		</div>
		<div class="rounded-lg border border-border bg-card px-4 py-3">
			<div class="text-xs text-muted-foreground uppercase">Best Value</div>
			<div class="mt-1 font-medium tabular-nums">{fmt(data.study.best_value)}</div>
		</div>
		<div class="rounded-lg border border-border bg-card px-4 py-3">
			<div class="text-xs text-muted-foreground uppercase">Best Run</div>
			<div class="mt-1 font-medium">
				{#if data.study.best_run_id}
					<a
						href={withSourceHref(`/experiments/${data.experimentId}/runs/${data.study.best_run_id}`, data.source)}
						class="text-primary underline underline-offset-2"
					>{data.study.best_run_id.slice(0, 12)}</a>
				{:else}
					-
				{/if}
			</div>
		</div>
	</div>

	<!-- Convergence chart -->
	{#if data.trials.length > 0}
		<div class="rounded-lg border border-border bg-card p-4">
			<h2 class="text-sm font-medium mb-3">Trial Convergence</h2>
			<div class="h-64">
				<TrialConvergenceChart trials={data.trials} class="h-full" />
			</div>
		</div>
	{/if}

	<!-- Best trial params -->
	{#if bestParams}
		<div class="rounded-lg border border-border bg-card p-4">
			<h2 class="text-sm font-medium mb-3">Best Trial #{data.study.best_trial_number} Parameters</h2>
			<div class="grid grid-cols-2 md:grid-cols-3 gap-2">
				{#each Object.entries(bestParams) as [key, val]}
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
	{#if data.trials.length > 0}
		<div class="rounded-lg border border-border bg-card overflow-hidden">
			<h2 class="text-sm font-medium px-4 py-3 border-b border-border">Trials ({data.trials.length})</h2>
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
						{#each data.trials as trial, i}
							{@const num = (trial.number ?? trial.trial_number ?? i) as number}
							{@const state = (trial.state ?? 'COMPLETE') as string}
							{@const value = trial.value as number | undefined}
							{@const duration = trial.duration as number | undefined}
							<tr class="hover:bg-muted/20 {num === data.study.best_trial_number ? 'bg-primary/5' : ''}">
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
