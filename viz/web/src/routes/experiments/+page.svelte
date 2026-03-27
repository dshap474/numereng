<script lang="ts">
	import type {
		ExperimentOverviewItem,
		ExperimentOverviewResponse,
		LiveExperimentOverview,
		LiveRunOverview,
		RecentExperimentActivityItem
	} from '$lib/api/client';
	import { api } from '$lib/api/client';
	import { fmtPercent } from '$lib/utils';

	let {
		data
	}: {
		data: {
			overview: ExperimentOverviewResponse;
		};
	} = $props();

	const routeOverview = () => data.overview;
	let overview = $state<ExperimentOverviewResponse>(routeOverview());
	let documentVisible = $state(true);
	let overviewGeneration = 0;
	let feedState = $state<'live' | 'holding'>('live');

	const timeFormatter = new Intl.DateTimeFormat(undefined, {
		hour: 'numeric',
		minute: '2-digit'
	});
	const dateFormatter = new Intl.DateTimeFormat(undefined, {
		month: 'short',
		day: 'numeric'
	});

	let experiments = $derived(overview.experiments ?? []);
	let liveExperiments = $derived(overview.live_experiments ?? []);
	let recentActivity = $derived(overview.recent_activity ?? []);
	let summary = $derived(overview.summary);
	let lastSurfacePulse = $derived(
		liveExperiments[0]?.latest_activity_at ??
			recentActivity[0]?.updated_at ??
			recentActivity[0]?.finished_at ??
			experiments[0]?.latest_activity_at ??
			null
	);

	$effect(() => {
		overview = routeOverview();
	});

	$effect(() => {
		if (typeof document === 'undefined') return;
		const handleVisibilityChange = () => {
			documentVisible = !document.hidden;
		};
		handleVisibilityChange();
		document.addEventListener('visibilitychange', handleVisibilityChange);
		return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
	});

	$effect(() => {
		if (!documentVisible) return;
		if (typeof window === 'undefined') return;
		void refreshOverview();
		const timer = window.setInterval(() => {
			void refreshOverview();
		}, 3000);
		return () => window.clearInterval(timer);
	});

	async function refreshOverview() {
		const generation = ++overviewGeneration;
		try {
			const next = await api.getExperimentsOverview();
			if (generation !== overviewGeneration) return;
			overview = next;
			feedState = 'live';
		} catch {
			if (generation !== overviewGeneration) return;
			feedState = 'holding';
		}
	}

	function formatStatusLabel(status: string | null | undefined): string {
		const normalized = (status ?? '').toLowerCase();
		switch (normalized) {
			case 'running':
			case 'starting':
			case 'canceling':
				return 'TRAINING';
			case 'queued':
				return 'QUEUED';
			case 'completed':
			case 'complete':
				return 'COMPLETED';
			case 'failed':
				return 'FAILED';
			case 'canceled':
				return 'CANCELED';
			case 'stale':
				return 'STALE';
			case 'active':
				return 'ACTIVE';
			case 'draft':
				return 'DRAFT';
			default:
				return normalized ? normalized.toUpperCase() : 'UNKNOWN';
		}
	}

	function formatAttentionLabel(state: string | null | undefined): string {
		const normalized = (state ?? '').toLowerCase();
		if (!normalized || normalized === 'none') return 'stable';
		return normalized;
	}

	function formatStageLabel(stage: string | null | undefined): string {
		if (!stage) return 'awaiting stage';
		return stage.replaceAll('_', ' ');
	}

	function experimentStatusClass(status: string | null | undefined): string {
		switch ((status ?? '').toLowerCase()) {
			case 'active':
				return 'bg-sky-400/8 text-sky-100 ring-1 ring-sky-400/14';
			case 'complete':
			case 'completed':
				return 'bg-emerald-400/10 text-emerald-200 ring-1 ring-emerald-400/20';
			case 'draft':
				return 'bg-white/[0.06] text-slate-300 ring-1 ring-white/8';
			default:
				return 'bg-white/[0.06] text-slate-300 ring-1 ring-white/8';
		}
	}

	function statusChipClass(status: string | null | undefined): string {
		switch ((status ?? '').toLowerCase()) {
			case 'running':
			case 'starting':
			case 'canceling':
				return 'bg-sky-400/10 text-sky-100 ring-1 ring-sky-400/16';
			case 'queued':
				return 'bg-white/[0.06] text-slate-200 ring-1 ring-white/10';
			case 'completed':
			case 'complete':
				return 'bg-emerald-400/12 text-emerald-200 ring-1 ring-emerald-400/20';
			case 'failed':
				return 'bg-red-400/12 text-red-200 ring-1 ring-red-400/20';
			case 'stale':
				return 'bg-amber-400/12 text-amber-200 ring-1 ring-amber-400/20';
			case 'canceled':
				return 'bg-yellow-400/12 text-yellow-100 ring-1 ring-yellow-400/20';
			default:
				return 'bg-white/[0.06] text-slate-300 ring-1 ring-white/8';
		}
	}

	function attentionClass(state: string | null | undefined): string {
		switch ((state ?? '').toLowerCase()) {
			case 'failed':
				return 'text-red-200 bg-red-500/[0.08] border-red-400/20';
			case 'stale':
				return 'text-amber-200 bg-amber-500/[0.08] border-amber-400/20';
			case 'canceled':
				return 'text-yellow-100 bg-yellow-500/[0.08] border-yellow-400/20';
			default:
				return 'text-slate-300 bg-transparent border-transparent';
		}
	}

	function experimentRowClass(item: ExperimentOverviewItem): string {
		if (item.has_live) {
			return 'border-l-sky-400/70 bg-white/[0.02] hover:bg-white/[0.035]';
		}
		switch (item.attention_state) {
			case 'failed':
				return 'border-l-red-400/80 bg-red-500/[0.025] hover:bg-red-500/[0.045]';
			case 'stale':
				return 'border-l-amber-400/80 bg-amber-500/[0.025] hover:bg-amber-500/[0.045]';
			case 'canceled':
				return 'border-l-yellow-400/80 bg-yellow-500/[0.025] hover:bg-yellow-500/[0.045]';
			default:
				return 'border-l-transparent hover:bg-white/[0.04]';
		}
	}

	function liveSectionClass(item: LiveExperimentOverview): string {
		switch (item.attention_state) {
			case 'failed':
				return 'border-white/8 bg-card';
			case 'stale':
				return 'border-white/8 bg-card';
			case 'canceled':
				return 'border-white/8 bg-card';
			default:
				return 'border-white/8 bg-card';
		}
	}

	function activityTone(item: RecentExperimentActivityItem): string {
		switch ((item.status ?? '').toLowerCase()) {
			case 'failed':
				return 'bg-card';
			case 'stale':
				return 'bg-card';
			case 'canceled':
				return 'bg-card';
			case 'completed':
			case 'complete':
				return 'bg-card';
			default:
				return 'bg-card';
		}
	}

	function activityBarClass(item: RecentExperimentActivityItem): string {
		switch ((item.status ?? '').toLowerCase()) {
			case 'failed':
				return 'bg-red-400';
			case 'stale':
				return 'bg-amber-400';
			case 'canceled':
				return 'bg-yellow-300';
			case 'completed':
			case 'complete':
				return 'bg-emerald-400';
			default:
				return 'bg-sky-400';
		}
	}

	function progressWidth(value: number | null | undefined): string {
		if (value == null || Number.isNaN(value)) return '0%';
		const clamped = Math.max(0, Math.min(100, value));
		return `${clamped}%`;
	}

	function formatSurfaceTime(value: string | null | undefined): string {
		if (!value) return 'No activity';
		const date = new Date(value);
		if (Number.isNaN(date.getTime())) return value;
		const now = new Date();
		const isSameDay =
			date.getFullYear() === now.getFullYear() &&
			date.getMonth() === now.getMonth() &&
			date.getDate() === now.getDate();
		if (isSameDay) return timeFormatter.format(date);
		return `${dateFormatter.format(date)} · ${timeFormatter.format(date)}`;
	}

	function formatSurfaceDate(value: string | null | undefined): string {
		if (!value) return 'Unknown';
		const date = new Date(value);
		if (Number.isNaN(date.getTime())) return value;
		return dateFormatter.format(date);
	}

	function experimentMeta(item: ExperimentOverviewItem): string {
		const parts: string[] = [];
		parts.push(`${item.run_count ?? 0} runs`);
		if (item.tags.length > 0) {
			parts.push(item.tags.slice(0, 3).join(' · '));
		}
		return parts.join(' · ');
	}

	function sourceMeta(sourceKind: string | null | undefined, sourceLabel: string | null | undefined): string | null {
		if (!sourceKind || sourceKind === 'local') return null;
		if (!sourceLabel) return sourceKind.toUpperCase();
		return `${sourceKind.toUpperCase()} · ${sourceLabel}`;
	}

	function liveRunHint(run: LiveRunOverview): string {
		if (run.progress_label) return run.progress_label;
		if (run.current_stage) return formatStageLabel(run.current_stage);
		return 'Awaiting lifecycle update';
	}

	function isIndeterminateRun(run: LiveRunOverview): boolean {
		return (
			(!run.current_stage || run.current_stage === 'awaiting_stage') &&
			(run.progress_percent == null || !run.progress_label)
		);
	}

	function liveStageChipLabel(run: LiveRunOverview): string {
		if (isIndeterminateRun(run)) return 'SYNCING · telemetry';
		return `${formatStatusLabel(run.status)} · ${formatStageLabel(run.current_stage)}`;
	}

	function runTelemetryLabel(run: LiveRunOverview): string {
		if (isIndeterminateRun(run)) return 'Telemetry sync pending';
		return liveRunHint(run);
	}

	function recentActivityHint(item: RecentExperimentActivityItem): string {
		if (item.terminal_reason) return item.terminal_reason.replaceAll('_', ' ');
		if (item.progress_label) return item.progress_label;
		if (item.current_stage) return formatStageLabel(item.current_stage);
		return 'No terminal detail recorded';
	}
</script>

<div class="mission-shell space-y-6">
	<section class="mission-header mission-panel rounded-[1.9rem] border border-white/8 px-5 py-6 shadow-[0_22px_70px_rgba(0,0,0,0.24)] sm:px-7">
		<div class="flex flex-col gap-6 xl:flex-row xl:items-end xl:justify-between">
			<div class="max-w-3xl">
				<p class="font-mono text-[11px] uppercase tracking-[0.36em] text-slate-400">Mission Control</p>
				<h1 class="mt-3 text-[clamp(2.2rem,4.2vw,3.5rem)] font-semibold tracking-[-0.05em] text-white">
					Workspace experiments
				</h1>
				<p class="mt-3 max-w-2xl text-sm leading-6 text-slate-300">
					Live monitor for every experiment, active run, and terminal event in the workspace.
				</p>
			</div>

			<div class="flex flex-wrap items-center gap-2 text-[11px] uppercase tracking-[0.22em]">
				<div class="signal-pill">
					<span class="live-dot rounded-full bg-sky-400"></span>
					<span>{feedState === 'live' ? 'Live feed' : 'Holding snapshot'}</span>
				</div>
				<div class="signal-pill signal-pill-muted">Last pulse {formatSurfaceTime(lastSurfacePulse)}</div>
			</div>
		</div>
	</section>

	<div class="grid gap-6 xl:grid-cols-[320px_minmax(0,1fr)] 2xl:grid-cols-[360px_minmax(0,1fr)]">
		<section class="mission-panel rail-panel rounded-[1.7rem] border border-white/8 px-0 py-0 shadow-[0_18px_54px_rgba(0,0,0,0.2)]">
			<div class="border-b border-white/8 px-5 py-4">
				<p class="font-mono text-[11px] uppercase tracking-[0.28em] text-slate-400">Experiment Index</p>
				<div class="mt-2 flex items-end justify-between gap-4">
					<h2 class="text-base font-semibold text-white">All experiments</h2>
					<p class="font-mono text-xs text-slate-300">{summary.total_experiments} tracked</p>
				</div>
			</div>

			{#if experiments.length === 0}
				<div class="px-5 py-10 text-center text-sm text-slate-400">No experiments indexed yet.</div>
			{:else}
				<div class="max-h-[calc(100vh-15rem)] overflow-y-auto px-3 py-3">
					<div class="space-y-3">
						{#each experiments as experiment (experiment.experiment_id)}
							<svelte:element
								this={experiment.detail_href ? 'a' : 'div'}
								href={experiment.detail_href ?? undefined}
								class="experiment-row block rounded-[1.2rem] border border-white/7 px-4 py-4 transition-all duration-200 {experimentRowClass(experiment)}"
							>
								<div class="flex items-start justify-between gap-4">
									<div class="min-w-0">
										<div class="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.18em] text-slate-400">
											<span>{formatSurfaceDate(experiment.created_at)}</span>
											{#if experiment.has_live}
												<span class="inline-flex items-center gap-1.5 text-slate-300">
													<span class="live-dot rounded-full bg-sky-400"></span>
													live
												</span>
											{/if}
										</div>
										<h3 class="mt-2 truncate text-sm font-semibold text-white">{experiment.name}</h3>
									</div>
									<span class="inline-flex shrink-0 rounded-full px-2.5 py-1 text-[10px] font-medium uppercase {experimentStatusClass(experiment.status)}">
										{formatStatusLabel(experiment.status)}
									</span>
								</div>

								<div class="mt-3 flex items-center justify-between gap-3">
									<p class="min-w-0 truncate font-mono text-[11px] text-slate-300/90">{experimentMeta(experiment)}</p>
									<span class="rounded-full border px-2 py-0.5 text-[10px] uppercase {attentionClass(experiment.attention_state)}">
										{formatAttentionLabel(experiment.attention_state)}
									</span>
								</div>

								<div class="mt-2 font-mono text-[11px] text-slate-400">
									Last activity {formatSurfaceTime(experiment.latest_activity_at)}
								</div>
								{#if sourceMeta(experiment.source_kind, experiment.source_label)}
									<div class="mt-2 font-mono text-[10px] uppercase tracking-[0.12em] text-slate-500">
										{sourceMeta(experiment.source_kind, experiment.source_label)}
									</div>
								{/if}
							</svelte:element>
						{/each}
					</div>
				</div>
			{/if}
		</section>

		<div class="space-y-6">
			<section class="mission-panel overflow-hidden rounded-[1.45rem] border border-white/8 shadow-[0_16px_44px_rgba(0,0,0,0.18)]">
				<div class="grid gap-0 md:grid-cols-2 xl:grid-cols-5">
					<div class="metric-cell border-b border-white/8 xl:border-r xl:border-b-0">
						<p class="metric-label">Live Experiments</p>
						<p class="metric-value text-white">{summary.live_experiments}</p>
						<p class="metric-meta">active walls</p>
					</div>
					<div class="metric-cell border-b border-white/8 xl:border-r xl:border-b-0">
						<p class="metric-label">Live Runs</p>
						<p class="metric-value text-white">{summary.live_runs}</p>
						<p class="metric-meta">training now</p>
					</div>
					<div class="metric-cell border-b border-white/8 xl:border-r xl:border-b-0">
						<p class="metric-label">Queued</p>
						<p class="metric-value text-slate-100">{summary.queued_runs}</p>
						<p class="metric-meta">waiting slots</p>
					</div>
					<div class="metric-cell border-b border-white/8 md:border-b-0 xl:border-r">
						<p class="metric-label">Attention</p>
						<p class="metric-value text-amber-100">{summary.attention_count}</p>
						<p class="metric-meta">failed stale canceled</p>
					</div>
					<div class="metric-cell">
						<p class="metric-label">Total</p>
						<p class="metric-value text-white">{summary.total_experiments}</p>
						<p class="metric-meta">
							{summary.active_experiments} active · {summary.completed_experiments} complete
						</p>
					</div>
				</div>
			</section>

			<div class="grid gap-6 2xl:grid-cols-[minmax(0,1.55fr)_minmax(320px,0.9fr)]">
				<section class="mission-panel overflow-hidden rounded-[1.6rem] border border-white/8 shadow-[0_18px_50px_rgba(0,0,0,0.18)] scroll-mt-20">
					<div class="border-b border-white/8 px-5 py-4">
						<div class="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
							<div>
								<p class="font-mono text-[11px] uppercase tracking-[0.28em] text-slate-400">Live Operations</p>
								<h2 class="mt-2 text-lg font-semibold text-white">Active systems</h2>
							</div>
							<div class="font-mono text-[11px] uppercase tracking-[0.22em] text-slate-300/80">
								3s cadence · workspace pulse
							</div>
						</div>
					</div>

					{#if liveExperiments.length === 0}
						<div class="px-6 py-12 text-center">
							<p class="text-sm font-medium text-white">No active experiments right now.</p>
							<p class="mt-2 text-sm text-slate-400">The live surface will populate as soon as new runs enter the queue.</p>
						</div>
					{:else}
						<div class="divide-y divide-white/8">
							{#each liveExperiments as item (item.experiment_id)}
								<article class="px-5 py-5">
									<div class="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
										<div class="min-w-0">
											<div class="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.2em] text-slate-300">
												<span class="live-dot rounded-full bg-sky-400"></span>
												{item.live_run_count} active
											</div>
											<h3 class="mt-2 truncate text-lg font-semibold text-white">
												<svelte:element
													this={item.detail_href ? 'a' : 'span'}
													href={item.detail_href ?? undefined}
													class={item.detail_href ? 'transition-colors hover:text-white' : undefined}
												>
													{item.name}
												</svelte:element>
											</h3>
										</div>

										<div class="flex flex-wrap items-center gap-2 font-mono text-[10px] uppercase tracking-[0.18em] text-slate-300/85">
											<span class="rounded-full border px-2.5 py-1 text-[10px] uppercase {attentionClass(item.attention_state)}">
												{formatAttentionLabel(item.attention_state)}
											</span>
											<span class="rounded-full border border-white/10 bg-white/[0.03] px-2.5 py-1 text-[10px] text-slate-300">
												{item.queued_run_count} queued
											</span>
										</div>
									</div>

									<div class="mt-5 grid gap-5 lg:grid-cols-[minmax(0,1fr)_170px]">
										<div>
											<div class="flex items-center justify-between gap-3 font-mono text-[11px] uppercase tracking-[0.18em] text-slate-400">
												<span>Aggregate progress</span>
												<span class="text-white">{fmtPercent(item.aggregate_progress_percent)}</span>
											</div>
											<div class="progress-rail mt-2 h-2 overflow-hidden rounded-full bg-white/[0.05]">
												<div
													class="progress-fill h-full rounded-full transition-[width] duration-700"
													style={`width: ${progressWidth(item.aggregate_progress_percent)}`}
												></div>
											</div>

											<div class="mt-4 space-y-4">
												{#each item.runs as run (run.run_id)}
													<div class="border-t border-white/8 pt-4 first:border-t-0 first:pt-0">
														<div class="flex items-start justify-between gap-4">
															<div class="min-w-0">
																<p class="truncate text-sm font-medium text-white">{run.config_label}</p>
																<div class="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px] text-slate-300/88">
																	<span>{formatStageLabel(run.current_stage)}</span>
																	<span>{runTelemetryLabel(run)}</span>
																</div>
															</div>

															<div class="text-right">
																<span class="rounded-full px-2 py-0.5 text-[10px] font-medium uppercase {statusChipClass(run.status)}">
																	{formatStatusLabel(run.status)}
																</span>
																<p class="mt-2 font-mono text-sm text-white">{fmtPercent(run.progress_percent)}</p>
															</div>
														</div>

														<div class="progress-rail mt-3 h-1.5 overflow-hidden rounded-full bg-white/[0.05]">
															<div
																class="progress-fill h-full rounded-full transition-[width] duration-700 {isIndeterminateRun(run)
																	? 'progress-fill-indeterminate'
																	: ''}"
																style={`width: ${isIndeterminateRun(run) ? '24%' : progressWidth(run.progress_percent)}`}
															></div>
														</div>

														<div class="mt-3 flex flex-wrap gap-2">
															<span class="rounded-full border border-white/10 bg-white/[0.03] px-2.5 py-1 font-mono text-[10px] uppercase tracking-[0.14em] text-slate-200">
																{liveStageChipLabel(run)}
															</span>
														</div>
													</div>
												{/each}
											</div>
										</div>

										<div class="grid gap-3 border-t border-white/8 pt-4 lg:border-t-0 lg:border-l lg:border-white/8 lg:pt-0 lg:pl-4">
											<div>
												<p class="font-mono text-[10px] uppercase tracking-[0.18em] text-slate-400">Live runs</p>
												<p class="mt-1 text-lg font-semibold text-white">{item.live_run_count}</p>
											</div>
											<div>
												<p class="font-mono text-[10px] uppercase tracking-[0.18em] text-slate-400">Queue</p>
												<p class="mt-1 text-lg font-semibold text-white">{item.queued_run_count}</p>
											</div>
											<div>
												<p class="font-mono text-[10px] uppercase tracking-[0.18em] text-slate-400">Last pulse</p>
												<p class="mt-1 font-mono text-sm text-slate-100">{formatSurfaceTime(item.latest_activity_at)}</p>
											</div>
										</div>
									</div>
								</article>
							{/each}
						</div>
					{/if}
				</section>

				<aside class="mission-panel rounded-[1.6rem] border border-white/8 shadow-[0_18px_50px_rgba(0,0,0,0.18)] scroll-mt-20">
					<div class="border-b border-white/8 px-5 py-4">
						<div class="flex items-end justify-between gap-4">
							<div>
								<p class="font-mono text-[11px] uppercase tracking-[0.28em] text-slate-400">Terminal Feed</p>
								<h2 class="mt-2 text-lg font-semibold text-white">Recent activity</h2>
							</div>
							<p class="font-mono text-[11px] uppercase tracking-[0.18em] text-slate-300/80">Newest 8</p>
						</div>
					</div>

					{#if recentActivity.length === 0}
						<div class="px-5 py-10 text-center">
							<p class="text-sm font-medium text-white">No terminal activity yet.</p>
							<p class="mt-2 text-sm text-slate-400">Completed, canceled, stale, and failed runs will appear here.</p>
						</div>
					{:else}
						<div class="space-y-3 px-3 py-3">
							{#each recentActivity as item, index (item.job_id ?? `${item.experiment_id}-${index}`)}
								<div class="recent-activity-card relative rounded-[1.15rem] border border-white/7 px-4 py-4 {activityTone(item)}">
									<span class="absolute left-0 top-4 bottom-4 w-[2px] rounded-full {activityBarClass(item)}"></span>

									<div class="flex items-start justify-between gap-4">
										<div class="min-w-0">
											<p class="truncate text-sm font-medium text-white">{item.experiment_name}</p>
											<p class="mt-1 truncate font-mono text-[11px] text-slate-300/88">{item.config_label}</p>
										</div>

										<div class="text-right">
											<span class="rounded-full px-2 py-0.5 text-[10px] font-medium uppercase {statusChipClass(item.status)}">
												{formatStatusLabel(item.status)}
											</span>
											<p class="mt-2 font-mono text-[11px] text-slate-300/80">
												{formatSurfaceTime(item.updated_at ?? item.finished_at)}
											</p>
										</div>
									</div>

									<div class="mt-3 flex items-center justify-between gap-3 font-mono text-[11px] text-slate-300/88">
										<span class="truncate">{recentActivityHint(item)}</span>
										<span class="shrink-0">{fmtPercent(item.progress_percent)}</span>
									</div>
								</div>
							{/each}
						</div>
					{/if}
				</aside>
			</div>
		</div>
	</div>
</div>

<style>
	.mission-shell {
		position: relative;
	}

	.mission-panel {
		background: oklch(0.18 0 0);
		backdrop-filter: blur(18px);
	}

	.mission-header {
		background: oklch(0.18 0 0);
	}

	.signal-pill {
		display: inline-flex;
		align-items: center;
		gap: 0.55rem;
		border-radius: 9999px;
		border: 1px solid rgba(255, 255, 255, 0.1);
		background: rgba(255, 255, 255, 0.035);
		padding: 0.5rem 0.9rem;
		font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace;
		color: rgb(226 232 240 / 0.92);
	}

	.signal-pill-muted {
		color: rgb(148 163 184 / 0.88);
		background: oklch(0.22 0 0);
	}

	.metric-label {
		font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace;
		font-size: 0.7rem;
		letter-spacing: 0.24em;
		text-transform: uppercase;
		color: rgba(203, 213, 225, 0.82);
	}

	.metric-value {
		margin-top: 0.5rem;
		font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace;
		font-size: clamp(1.65rem, 2vw, 2.25rem);
		line-height: 1;
		font-weight: 600;
	}

	.metric-meta {
		margin-top: 0.45rem;
		font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace;
		font-size: 0.72rem;
		letter-spacing: 0.1em;
		text-transform: uppercase;
		color: rgba(203, 213, 225, 0.7);
	}

	.live-dot {
		display: inline-flex;
		width: 0.5rem;
		height: 0.5rem;
		box-shadow: none;
		animation: telemetry-pulse 1.8s ease-in-out infinite;
	}

	.metric-cell {
		padding: 1rem 1.25rem;
		background: transparent;
	}

	.rail-panel {
		background: oklch(0.18 0 0);
	}

	.experiment-row {
		background: oklch(0.18 0 0);
		box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02);
	}

	.recent-activity-card {
		box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02);
	}

	.progress-rail {
		box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.04);
	}

	.progress-fill {
		background: rgb(56 189 248 / 0.7);
	}

	.progress-fill-indeterminate {
		background:
			repeating-linear-gradient(
				90deg,
				rgba(148, 163, 184, 0.42),
				rgba(148, 163, 184, 0.42) 10px,
				rgba(71, 85, 105, 0.18) 10px,
				rgba(71, 85, 105, 0.18) 20px
			);
	}

	@keyframes telemetry-pulse {
		0%,
		100% {
			transform: scale(1);
			opacity: 0.9;
		}
		50% {
			transform: scale(1.18);
			opacity: 0.45;
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.live-dot {
			animation: none !important;
		}
	}
</style>
