<script lang="ts">
	import AccentCard from '$lib/components/ui/AccentCard.svelte';
	import type {
		ExperimentOverviewItem,
		ExperimentOverviewResponse,
		LiveExperimentOverview,
		LiveRunOverview,
		RecentExperimentActivityItem
	} from '$lib/api/client';
	import { api } from '$lib/api/client';
	import { ensureClientPerfObservers, mark, measure } from '$lib/perf';
	import { fmtPercent } from '$lib/utils';

	let {
		data
	}: {
		data: {
			overview: ExperimentOverviewResponse;
			overviewPending: boolean;
		};
	} = $props();

	type OverviewSource = NonNullable<ExperimentOverviewResponse['sources']>[number];

	const routeOverview = () => data.overview;
	const routeOverviewPending = () => data.overviewPending;
	let overview = $state<ExperimentOverviewResponse>(routeOverview());
	let overviewBootstrapPending = $state(routeOverviewPending());
	let documentVisible = $state(true);
	let remoteSourcesViewOpen = $state(false);
	let overviewRefreshInFlight = false;
	let overviewPollSession = 0;

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
	let sources = $derived(overview.sources ?? []);
	let remoteSources = $derived(sources.filter((source) => source.kind === 'ssh'));
	let primaryRemoteSource = $derived(remoteSources[0] ?? null);
	let summary = $derived(overview.summary);
	let liveOperationsLoading = $derived(overviewBootstrapPending && liveExperiments.length === 0);
	let terminalFeedLoading = $derived(overviewBootstrapPending && recentActivity.length === 0);
	let remoteSourcesLoading = $derived(overviewBootstrapPending && remoteSources.length === 0);

	$effect(() => {
		ensureClientPerfObservers();
	});

	$effect(() => {
		mark('experiments:local-ready:start');
		overview = routeOverview();
		overviewBootstrapPending = routeOverviewPending();
		queueMicrotask(() => {
			mark('experiments:local-ready:end');
			measure('experiments_local_payload_ready', 'experiments:local-ready:start', 'experiments:local-ready:end');
		});
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
		const session = ++overviewPollSession;
		void refreshOverview(session);
		const timer = window.setInterval(() => {
			void refreshOverview(session);
		}, 10000);
		return () => window.clearInterval(timer);
	});

	async function refreshOverview(session: number) {
		if (overviewRefreshInFlight) return;
		overviewRefreshInFlight = true;
		mark(`experiments:remote-refresh:${session}:start`);
		try {
			const next = await api.getExperimentsOverview({ include_remote: true });
			if (session !== overviewPollSession) return;
			overview = next;
		} catch {
			if (session !== overviewPollSession) return;
		} finally {
			mark(`experiments:remote-refresh:${session}:end`);
			measure(
				`experiments_remote_refresh:${session}`,
				`experiments:remote-refresh:${session}:start`,
				`experiments:remote-refresh:${session}:end`
			);
			if (session === overviewPollSession) {
				overviewBootstrapPending = false;
			}
			overviewRefreshInFlight = false;
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

	function experimentRowToneClass(item: ExperimentOverviewItem): string {
		if (item.has_live) {
			return 'bg-white/[0.02] hover:bg-white/[0.035]';
		}
		switch (item.attention_state) {
			case 'failed':
				return 'bg-red-500/[0.025] hover:bg-red-500/[0.045]';
			case 'stale':
				return 'bg-amber-500/[0.025] hover:bg-amber-500/[0.045]';
			case 'canceled':
				return 'bg-yellow-500/[0.025] hover:bg-yellow-500/[0.045]';
			default:
				return 'hover:bg-white/[0.04]';
		}
	}

	function experimentAccentClass(item: ExperimentOverviewItem): string {
		if (item.has_live) return 'border-l-sky-400/70';
		switch ((item.attention_state ?? '').toLowerCase()) {
			case 'failed':
				return 'border-l-red-400/80';
			case 'stale':
				return 'border-l-amber-400/80';
			case 'canceled':
				return 'border-l-yellow-400/80';
			default:
				return 'border-l-transparent';
		}
	}

	function activityToneClass(_item: RecentExperimentActivityItem): string {
		return 'hover:bg-white/[0.04]';
	}

	function activityBarClass(item: RecentExperimentActivityItem): string {
		switch ((item.status ?? '').toLowerCase()) {
			case 'failed':
				return 'border-l-red-400/80';
			case 'stale':
				return 'border-l-amber-400/80';
			case 'canceled':
				return 'border-l-yellow-300/80';
			case 'completed':
			case 'complete':
				return 'border-l-emerald-400/80';
			default:
				return 'border-l-sky-400/70';
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

	function sourceAvailabilityLabel(source: OverviewSource): string {
		switch ((source.state ?? '').toLowerCase()) {
			case 'live':
				return 'LIVE';
			case 'cached':
				return 'CACHED';
			case 'unavailable':
				return 'UNAVAILABLE';
			default:
				return (source.state ?? 'unknown').toUpperCase();
		}
	}

	function sourceAvailabilityClass(source: OverviewSource): string {
		switch ((source.state ?? '').toLowerCase()) {
			case 'live':
				return 'bg-emerald-400/12 text-emerald-200 ring-1 ring-emerald-400/20';
			case 'cached':
				return 'bg-amber-400/12 text-amber-100 ring-1 ring-amber-400/20';
			case 'unavailable':
				return 'bg-red-400/12 text-red-200 ring-1 ring-red-400/20';
			default:
				return 'bg-white/[0.06] text-slate-300 ring-1 ring-white/8';
		}
	}

	function sourceBootstrapClass(source: OverviewSource): string {
		return (source.bootstrap_status ?? '').toLowerCase() === 'degraded'
			? 'bg-red-500/[0.08] text-red-200 border-red-400/20'
			: 'bg-white/[0.03] text-slate-200 border-white/10';
	}

	function sourceBootstrapLabel(source: OverviewSource): string {
		return (source.bootstrap_status ?? 'ready').toUpperCase();
	}

	function sourceDetail(source: OverviewSource): string {
		if (source.last_bootstrap_error) return source.last_bootstrap_error;
		if ((source.state ?? '').toLowerCase() === 'cached') return 'using last successful snapshot';
		if ((source.state ?? '').toLowerCase() === 'unavailable') return 'snapshot unavailable';
		if (source.last_bootstrap_at) return `bootstrapped ${formatSurfaceTime(source.last_bootstrap_at)}`;
		return 'remote source ready';
	}

	function sourceStatusDotClass(source: OverviewSource | null): string {
		return (source?.state ?? '').toLowerCase() === 'live' ? 'bg-emerald-400' : 'bg-red-400';
	}

	function experimentKey(item: ExperimentOverviewItem): string {
		return `${item.source_kind ?? 'local'}:${item.source_id ?? 'local'}:${item.experiment_id}`;
	}

	function liveExperimentKey(item: LiveExperimentOverview): string {
		return `${item.source_kind ?? 'local'}:${item.source_id ?? 'local'}:${item.experiment_id}`;
	}

	function recentActivityKey(item: RecentExperimentActivityItem, index: number): string {
		return [
			item.source_kind ?? 'local',
			item.source_id ?? 'local',
			item.job_id ?? 'job',
			item.run_id ?? 'run',
			item.experiment_id,
			String(index)
		].join(':');
	}

	function normalizedProgressMode(run: LiveRunOverview): 'exact' | 'estimated' | 'indeterminate' {
		const mode = (run.progress_mode ?? '').toLowerCase();
		if (mode === 'exact' || mode === 'estimated' || mode === 'indeterminate') return mode;
		return run.progress_percent == null ? 'indeterminate' : 'exact';
	}

	function progressModeRank(run: LiveRunOverview): number {
		switch (normalizedProgressMode(run)) {
			case 'exact':
				return 2;
			case 'estimated':
				return 1;
			default:
				return 0;
		}
	}

	function primaryRun(item: LiveExperimentOverview): LiveRunOverview | null {
		if (item.runs.length === 0) return null;
		return [...item.runs].sort((left, right) => {
			const updatedCompare = String(right.updated_at ?? '').localeCompare(String(left.updated_at ?? ''));
			if (updatedCompare !== 0) return updatedCompare;
			const modeCompare = progressModeRank(right) - progressModeRank(left);
			if (modeCompare !== 0) return modeCompare;
			return String(left.run_id).localeCompare(String(right.run_id));
		})[0];
	}

	function additionalActiveRuns(item: LiveExperimentOverview, run: LiveRunOverview | null): number {
		if (!run) return 0;
		return Math.max(0, item.runs.length - 1);
	}

	function primaryRunDetail(run: LiveRunOverview): string {
		if (run.progress_label) return run.progress_label;
		if (run.current_stage) return formatStageLabel(run.current_stage);
		return 'Awaiting progress signal';
	}

	function primaryRunMeta(item: LiveExperimentOverview, run: LiveRunOverview): string {
		const detail = primaryRunDetail(run);
		const status = formatStatusLabel(run.status);
		const parts: string[] = [];
		if (detail && detail.toLowerCase() !== status.toLowerCase()) {
			parts.push(detail);
		}
		parts.push(status);
		const extra = additionalActiveRuns(item, run);
		if (extra > 0) {
			parts.push(`+${extra} more active`);
		}
		return parts.join(' · ');
	}

	function progressValueLabel(run: LiveRunOverview): string {
		if (run.progress_percent == null || Number.isNaN(run.progress_percent)) return '—';
		return `${Math.round(run.progress_percent)}%`;
	}

	function progressAriaValueText(run: LiveRunOverview): string {
		const detail = primaryRunDetail(run);
		switch (normalizedProgressMode(run)) {
			case 'exact':
				return `${progressValueLabel(run)} complete${detail ? `, ${detail}` : ''}`;
			case 'estimated':
				return `${progressValueLabel(run)} complete, estimated${detail ? `, ${detail}` : ''}`;
			default:
				return detail ? `Progress indeterminate, ${detail}` : 'Progress indeterminate';
		}
	}

	function progressAriaLabel(item: LiveExperimentOverview, run: LiveRunOverview): string {
		return `${item.name} ${run.config_label}`;
	}

	function isActiveRunStatus(status: string | null | undefined): boolean {
		return ['queued', 'starting', 'running', 'canceling'].includes((status ?? '').toLowerCase());
	}

	function progressFillClass(run: LiveRunOverview): string {
		const classes = ['mission-progress-fill'];
		if (normalizedProgressMode(run) === 'estimated') {
			classes.push('mission-progress-fill-estimated');
		}
		if (isActiveRunStatus(run.status)) {
			classes.push('mission-progress-fill-active');
		}
		return classes.join(' ');
	}

	function recentActivityHint(item: RecentExperimentActivityItem): string {
		if (item.terminal_reason) return item.terminal_reason.replaceAll('_', ' ');
		if (item.progress_label) return item.progress_label;
		if (item.current_stage) return formatStageLabel(item.current_stage);
		return 'No terminal detail recorded';
	}
</script>

<div class="mission-shell -mx-8 -mt-14 -mb-8 flex h-screen min-h-0 flex-col overflow-x-hidden overflow-y-auto md:-mt-8 xl:overflow-hidden">
		<section class="mission-panel mission-surface cv-section flex min-h-0 flex-1 flex-col overflow-hidden border-t border-white/8">
			<header class="mission-toolbar px-5 py-4 sm:px-6">
			<div class="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
				<div class="min-w-0">
					<p class="font-mono text-[11px] uppercase tracking-[0.32em] text-slate-400">Mission Control</p>
					<div class="mt-2 flex flex-wrap items-end gap-x-4 gap-y-2">
						<h1 class="text-[clamp(1.95rem,3.2vw,2.7rem)] font-semibold tracking-[-0.05em] text-white">
							Workspace experiments
						</h1>
						<p class="pb-1 font-mono text-[11px] uppercase tracking-[0.18em] text-slate-400">
							{summary.total_experiments} tracked
						</p>
					</div>
				</div>

				<div class="flex w-full flex-col items-end gap-1 xl:w-auto xl:min-w-[12rem]">
					<p class="font-mono text-[11px] uppercase tracking-[0.24em] text-slate-400">Remote Sources</p>

					{#if remoteSourcesLoading}
						<div class="mission-loading-indicator text-left xl:justify-end" role="status" aria-live="polite">
							<span class="mission-loading-spinner" aria-hidden="true"></span>
							<span class="text-sm">Loading sources</span>
						</div>
					{:else if primaryRemoteSource}
							<button
								type="button"
								class="remote-summary-link text-left xl:text-right"
								onclick={() => (remoteSourcesViewOpen = true)}
							>
							<span class="remote-status-dot rounded-full {sourceStatusDotClass(primaryRemoteSource)}"></span>
							<span class="truncate text-sm font-semibold text-white">{primaryRemoteSource.label}</span>
						</button>
					{:else}
						<div class="remote-summary-empty text-sm text-slate-400">No remote sources</div>
					{/if}
				</div>
			</div>
		</header>

		{#if remoteSourcesViewOpen}
			<section class="cv-section flex min-h-0 flex-1 flex-col">
				<div class="border-b border-white/8 px-5 py-4 sm:px-6">
					<div class="flex items-start justify-between gap-4">
						<div>
							<p class="font-mono text-[11px] uppercase tracking-[0.28em] text-slate-400">Remote Sources</p>
							<h2 class="mt-2 text-lg font-semibold text-white">Federated monitors</h2>
							<p class="mt-1 font-mono text-[11px] uppercase tracking-[0.18em] text-slate-400">
								{remoteSources.length} enabled
							</p>
						</div>
						<button
							type="button"
							class="remote-close-button"
							aria-label="Close remote sources"
							onclick={() => (remoteSourcesViewOpen = false)}
						>
							<span aria-hidden="true">X</span>
						</button>
					</div>
				</div>

				{#if remoteSourcesLoading}
					<div class="mission-loading-state flex-1 px-6 py-10 text-center" role="status" aria-live="polite">
						<div>
							<div class="mission-loading-indicator mx-auto justify-center">
								<span class="mission-loading-spinner mission-loading-spinner-lg" aria-hidden="true"></span>
							</div>
							<p class="mt-4 text-sm font-medium text-white">Loading remote sources...</p>
							<p class="mt-2 text-sm text-slate-400">Checking federated monitors and source health.</p>
						</div>
					</div>
				{:else if remoteSources.length === 0}
					<div class="flex flex-1 items-center justify-center px-6 py-10 text-center">
						<div>
							<p class="text-sm font-medium text-white">No remote sources configured.</p>
							<p class="mt-2 text-sm text-slate-400">Add an SSH monitor to populate this view.</p>
						</div>
					</div>
				{:else}
					<div class="flex-1 min-h-0 overflow-y-auto px-5 py-5 sm:px-6">
						<div class="grid gap-4 xl:grid-cols-2 2xl:grid-cols-3">
							{#each remoteSources as source (source.id)}
								<article class="remote-source-detail-card rounded-[1.1rem] border border-white/8 px-4 py-4">
									<div class="flex items-start justify-between gap-3">
										<div class="min-w-0">
											<p class="truncate text-sm font-semibold text-white">{source.label}</p>
											<p class="mt-1 font-mono text-[10px] uppercase tracking-[0.14em] text-slate-400">
												{source.id}{#if source.host} · {source.host}{/if}
											</p>
										</div>
										<span class="rounded-full px-2.5 py-1 text-[10px] font-medium uppercase {sourceAvailabilityClass(source)}">
											{sourceAvailabilityLabel(source)}
										</span>
									</div>

									<div class="mt-3 flex flex-wrap items-center gap-2">
										<span class="rounded-full border px-2.5 py-1 text-[10px] font-medium uppercase {sourceBootstrapClass(source)}">
											{sourceBootstrapLabel(source)}
										</span>
										{#if source.store_root}
											<span class="min-w-0 flex-1 truncate font-mono text-[10px] text-slate-500">{source.store_root}</span>
										{/if}
									</div>

									<p class="mt-3 text-xs leading-5 text-slate-400">{sourceDetail(source)}</p>
								</article>
							{/each}
						</div>
					</div>
				{/if}
			</section>
		{:else}
				<div class="flex min-h-0 flex-1 flex-col md:grid md:grid-cols-2 md:grid-rows-[minmax(0,0.98fr)_minmax(0,1.02fr)] xl:grid-cols-[340px_minmax(0,1fr)] xl:grid-rows-[minmax(0,1fr)] 2xl:grid-cols-[360px_minmax(0,1fr)]">
					<aside class="mission-pane flex min-h-0 flex-col border-b-[1.5px] border-white/12 md:border-r-[1.5px] xl:border-r-[1.5px] xl:border-b-0">
						<div class="mission-pane-header border-y-[1.5px] border-white/12 px-5 py-4">
						<div class="flex items-end justify-between gap-4">
							<div>
								<p class="font-mono text-[11px] uppercase tracking-[0.28em] text-slate-400">Experiment Index</p>
								<h2 class="mt-2 text-lg font-semibold text-white">All experiments</h2>
							</div>
							<p class="font-mono text-[11px] uppercase tracking-[0.18em] text-slate-300/80">
								{summary.total_experiments} tracked
							</p>
						</div>
					</div>

					{#if experiments.length === 0}
						<div class="flex flex-1 items-center justify-center px-5 py-10 text-center text-sm text-slate-400">
							No experiments indexed yet.
						</div>
					{:else}
						<div class="flex-1 min-h-0 overflow-y-auto px-3 py-3">
							<div class="space-y-3">
								{#each experiments as experiment (experimentKey(experiment))}
									<AccentCard
										href={experiment.detail_href ?? null}
										class={`experiment-row transition-all duration-200 ${experimentRowToneClass(experiment)}`}
										paddingClass="px-4 py-3.5"
										roundedClass="rounded-[1.15rem]"
										accentClass={experimentAccentClass(experiment)}
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
									</AccentCard>
								{/each}
							</div>
						</div>
					{/if}
				</aside>

				<div class="flex min-h-0 flex-1 flex-col md:contents xl:flex xl:min-h-0 xl:flex-1 xl:flex-col">
					<div class="flex min-h-0 flex-1 flex-col md:contents xl:grid xl:min-h-0 xl:flex-1 xl:grid-cols-[minmax(0,1.45fr)_minmax(320px,0.95fr)] xl:grid-rows-[minmax(0,1fr)]">
							<section class="mission-pane cv-section flex min-h-0 flex-col border-b-[1.5px] border-white/12 md:col-start-1 md:row-start-2 md:col-span-2 md:border-b-0 xl:col-auto xl:row-auto xl:col-span-1 xl:border-r-[1.5px] xl:border-b-0">
								<div class="mission-pane-header border-y-[1.5px] border-white/12 px-5 py-4">
								<div class="flex flex-col gap-2 lg:flex-row lg:items-end lg:justify-between">
									<div>
										<p class="font-mono text-[11px] uppercase tracking-[0.28em] text-slate-400">Live Operations</p>
										<h2 class="mt-2 text-lg font-semibold text-white">Active systems</h2>
									</div>
									{#if liveOperationsLoading}
										<div
											class="mission-loading-indicator font-mono text-[11px] uppercase tracking-[0.22em] text-slate-300/80"
											role="status"
											aria-live="polite"
										>
											<span class="mission-loading-spinner" aria-hidden="true"></span>
											<span>Loading live state</span>
										</div>
									{:else}
										<div class="font-mono text-[11px] uppercase tracking-[0.22em] text-slate-300/80">
											3s cadence · workspace pulse
										</div>
									{/if}
								</div>
							</div>

							{#if liveOperationsLoading}
								<div class="mission-loading-state flex-1 px-6 py-10 text-center" role="status" aria-live="polite">
									<div>
										<div class="mission-loading-indicator mx-auto justify-center">
											<span class="mission-loading-spinner mission-loading-spinner-lg" aria-hidden="true"></span>
										</div>
										<p class="mt-4 text-sm font-medium text-white">Loading live operations...</p>
										<p class="mt-2 text-sm text-slate-400">Pulling active runs, queue state, and workspace pulse.</p>
									</div>
								</div>
							{:else if liveExperiments.length === 0}
								<div class="flex flex-1 items-center justify-center px-6 py-10 text-center">
									<div>
										<p class="text-sm font-medium text-white">No active experiments right now.</p>
										<p class="mt-2 text-sm text-slate-400">
											The live surface will populate as soon as new runs enter the queue.
										</p>
									</div>
								</div>
							{:else}
								<div class="flex-1 min-h-0 overflow-y-auto">
									<div class="divide-y divide-white/8">
										{#each liveExperiments as item (liveExperimentKey(item))}
											{@const featuredRun = primaryRun(item)}
											<article class="px-5 py-4">
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
														{#if featuredRun}
															<div class="mission-primary-run rounded-[1.15rem] border border-white/8 bg-white/[0.015] px-4 py-4">
																<div class="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
																	<div class="min-w-0">
																		<p class="truncate text-base font-semibold text-white">{featuredRun.config_label}</p>
																		<p class="mt-1 font-mono text-[11px] uppercase tracking-[0.16em] text-slate-300/82">
																			{primaryRunMeta(item, featuredRun)}
																		</p>
																	</div>

																	<span
																		class="inline-flex shrink-0 rounded-full px-2.5 py-1 text-[10px] font-medium uppercase {statusChipClass(featuredRun.status)}"
																	>
																		{formatStatusLabel(featuredRun.status)}
																	</span>
																</div>

																<div class="mission-progress-layout mt-4">
																	<div class="min-w-0">
																		<div
																			class="mission-progress-rail"
																			role="progressbar"
																			aria-label={progressAriaLabel(item, featuredRun)}
																			aria-valuemin="0"
																			aria-valuemax="100"
																			aria-valuenow={normalizedProgressMode(featuredRun) === 'indeterminate'
																				? undefined
																				: Math.round(featuredRun.progress_percent ?? 0)}
																			aria-valuetext={progressAriaValueText(featuredRun)}
																		>
																			{#if normalizedProgressMode(featuredRun) === 'indeterminate'}
																				<span class="mission-progress-indeterminate"></span>
																			{:else}
																				<span
																					class={progressFillClass(featuredRun)}
																					style={`width: ${progressWidth(featuredRun.progress_percent)}`}
																				></span>
																			{/if}
																		</div>
																		<p class="mt-2 font-mono text-[11px] uppercase tracking-[0.14em] text-slate-400">
																			{primaryRunDetail(featuredRun)}
																		</p>
																	</div>

																	<div class="mission-progress-value-wrap">
																		<p class="mission-progress-value">{progressValueLabel(featuredRun)}</p>
																		{#if normalizedProgressMode(featuredRun) === 'estimated'}
																			<span class="mission-estimate-pill">EST</span>
																		{/if}
																	</div>
																</div>
															</div>
														{/if}
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
								</div>
							{/if}
						</section>

							<aside class="mission-pane flex min-h-0 flex-col md:col-start-2 md:row-start-1 md:border-b-[1.5px] md:border-white/12 xl:col-auto xl:row-auto xl:border-b-0">
								<div class="mission-pane-header border-y-[1.5px] border-white/12 px-5 py-4">
								<div class="flex items-end justify-between gap-4">
									<div>
										<p class="font-mono text-[11px] uppercase tracking-[0.28em] text-slate-400">Terminal Feed</p>
										<h2 class="mt-2 text-lg font-semibold text-white">Recent activity</h2>
									</div>
									{#if terminalFeedLoading}
										<div
											class="mission-loading-indicator font-mono text-[11px] uppercase tracking-[0.18em] text-slate-300/80"
											role="status"
											aria-live="polite"
										>
											<span class="mission-loading-spinner" aria-hidden="true"></span>
											<span>Loading events</span>
										</div>
									{:else}
										<p class="font-mono text-[11px] uppercase tracking-[0.18em] text-slate-300/80">Newest 8</p>
									{/if}
								</div>
							</div>

							{#if terminalFeedLoading}
								<div class="mission-loading-state flex-1 px-5 py-10 text-center" role="status" aria-live="polite">
									<div>
										<div class="mission-loading-indicator mx-auto justify-center">
											<span class="mission-loading-spinner mission-loading-spinner-lg" aria-hidden="true"></span>
										</div>
										<p class="mt-4 text-sm font-medium text-white">Loading recent activity...</p>
										<p class="mt-2 text-sm text-slate-400">Collecting recent terminal events and run completions.</p>
									</div>
								</div>
							{:else if recentActivity.length === 0}
								<div class="flex flex-1 items-center justify-center px-5 py-10 text-center">
									<div>
										<p class="text-sm font-medium text-white">No terminal activity yet.</p>
										<p class="mt-2 text-sm text-slate-400">
											Completed, canceled, stale, and failed runs will appear here.
										</p>
									</div>
								</div>
							{:else}
								<div class="flex-1 min-h-0 overflow-y-auto px-3 py-3">
									<div class="space-y-3">
										{#each recentActivity as item, index (recentActivityKey(item, index))}
											<AccentCard
												class={`recent-activity-card ${activityToneClass(item)}`}
												paddingClass="px-4 py-4"
												roundedClass="rounded-[1.05rem]"
												accentClass={activityBarClass(item)}
											>
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
											</AccentCard>
										{/each}
									</div>
								</div>
							{/if}
						</aside>
					</div>
				</div>
			</div>
		{/if}
	</section>
</div>

<style>
	.mission-shell {
		position: relative;
	}

	.mission-panel {
		background: #070707;
		backdrop-filter: blur(18px);
	}

	.mission-surface {
		background: #070707;
		box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.025);
	}

	.mission-toolbar {
		background: #070707;
	}

	.mission-pane {
		background: #070707;
	}

	.mission-pane-header {
		background: #151515;
	}

	.mission-loading-indicator {
		display: inline-flex;
		align-items: center;
		gap: 0.55rem;
		color: rgba(203, 213, 225, 0.84);
	}

	.mission-loading-spinner {
		display: inline-flex;
		width: 0.8rem;
		height: 0.8rem;
		border-radius: 9999px;
		border: 1.5px solid rgba(148, 163, 184, 0.22);
		border-top-color: rgba(248, 250, 252, 0.92);
		animation: mission-spinner 0.82s linear infinite;
	}

	.mission-loading-spinner-lg {
		width: 1.1rem;
		height: 1.1rem;
		border-width: 2px;
	}

	.mission-loading-state {
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.remote-summary-link {
		display: inline-flex;
		align-items: center;
		align-self: flex-end;
		gap: 0.55rem;
		width: fit-content;
		max-width: min(100%, 17rem);
		margin-left: auto;
		border: 1px solid rgba(255, 255, 255, 0.08);
		border-radius: 0.95rem;
		background: #151515;
		cursor: pointer;
		opacity: 0.96;
		padding: 0.75rem 0.95rem;
		color: rgba(226, 232, 240, 0.9);
		box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02);
		transition:
			background-color 160ms ease,
			border-color 160ms ease,
			opacity 160ms ease,
			color 160ms ease;
	}

	.remote-summary-link:hover {
		background: rgba(255, 255, 255, 0.04);
		border-color: rgba(255, 255, 255, 0.14);
		color: rgba(248, 250, 252, 0.96);
		opacity: 1;
	}

	.remote-summary-empty {
		color: rgba(148, 163, 184, 0.82);
	}

	.live-dot {
		display: inline-flex;
		width: 0.5rem;
		height: 0.5rem;
		box-shadow: none;
		animation: telemetry-pulse 1.8s ease-in-out infinite;
	}

	.remote-status-dot {
		display: inline-flex;
		width: 0.5rem;
		height: 0.5rem;
		box-shadow: 0 0 0 0.2rem rgba(255, 255, 255, 0.02);
	}

	.remote-close-button {
		display: inline-flex;
		height: 2.25rem;
		width: 2.25rem;
		align-items: center;
		justify-content: center;
		border-radius: 9999px;
		border: 1px solid rgba(255, 255, 255, 0.1);
		background: rgba(255, 255, 255, 0.03);
		font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace;
		font-size: 0.8rem;
		letter-spacing: 0.1em;
		text-transform: uppercase;
		color: rgba(226, 232, 240, 0.9);
		transition:
			background-color 160ms ease,
			border-color 160ms ease,
			color 160ms ease;
	}

	.remote-close-button:hover {
		border-color: rgba(255, 255, 255, 0.16);
		background: rgba(255, 255, 255, 0.06);
		color: rgba(248, 250, 252, 0.98);
	}

	.remote-source-detail-card {
		background: #151515;
		box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02);
	}

	.experiment-row {
		background: #151515;
		box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02);
	}

	.recent-activity-card {
		background: #151515;
		box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02);
	}

	.mission-primary-run {
		background: #151515;
		box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02);
	}

	.mission-progress-layout {
		display: grid;
		gap: 1rem;
		align-items: center;
		grid-template-columns: minmax(0, 1fr) auto;
	}

	.mission-progress-rail {
		position: relative;
		height: 0.7rem;
		overflow: hidden;
		border-radius: 9999px;
		background: rgba(255, 255, 255, 0.045);
		box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.05);
	}

	.mission-progress-fill {
		position: relative;
		display: block;
		height: 100%;
		border-radius: inherit;
		background: linear-gradient(90deg, rgba(34, 211, 238, 0.72), rgba(56, 189, 248, 0.92));
		transition:
			width 720ms cubic-bezier(0.22, 1, 0.36, 1),
			background-color 240ms ease;
	}

	.mission-progress-fill-estimated {
		background: linear-gradient(90deg, rgba(103, 232, 249, 0.58), rgba(56, 189, 248, 0.78));
	}

	.mission-progress-fill-active::after {
		content: '';
		position: absolute;
		inset: 0 auto 0 -22%;
		width: 22%;
		background: linear-gradient(
			90deg,
			rgba(255, 255, 255, 0),
			rgba(255, 255, 255, 0.18),
			rgba(255, 255, 255, 0)
		);
		transform: skewX(-18deg);
		animation: mission-progress-sheen 2.2s linear infinite;
	}

	.mission-progress-indeterminate {
		position: absolute;
		top: 0;
		bottom: 0;
		left: -24%;
		width: 24%;
		border-radius: inherit;
		background: linear-gradient(90deg, rgba(103, 232, 249, 0.15), rgba(56, 189, 248, 0.72), rgba(103, 232, 249, 0.15));
		animation: mission-progress-travel 1.25s cubic-bezier(0.33, 1, 0.68, 1) infinite;
	}

	.mission-progress-value-wrap {
		display: inline-flex;
		align-items: center;
		justify-content: flex-end;
		gap: 0.55rem;
	}

	.mission-progress-value {
		font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace;
		font-size: 1rem;
		font-weight: 600;
		letter-spacing: 0.02em;
		color: rgba(248, 250, 252, 0.96);
	}

	.mission-estimate-pill {
		display: inline-flex;
		align-items: center;
		border-radius: 9999px;
		border: 1px solid rgba(56, 189, 248, 0.16);
		background: rgba(56, 189, 248, 0.08);
		padding: 0.2rem 0.5rem;
		font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace;
		font-size: 0.68rem;
		letter-spacing: 0.18em;
		text-transform: uppercase;
		color: rgba(186, 230, 253, 0.94);
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

	@keyframes mission-spinner {
		to {
			transform: rotate(360deg);
		}
	}

	@keyframes mission-progress-sheen {
		to {
			transform: translateX(460%) skewX(-18deg);
		}
	}

	@keyframes mission-progress-travel {
		0% {
			transform: translateX(0);
			opacity: 0.55;
		}
		50% {
			opacity: 1;
		}
		100% {
			transform: translateX(520%);
			opacity: 0.55;
		}
	}

	@media (max-width: 640px) {
		.mission-progress-layout {
			grid-template-columns: 1fr;
		}

		.mission-progress-value-wrap {
			justify-content: flex-start;
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.live-dot {
			animation: none !important;
		}

		.mission-loading-spinner {
			animation: none !important;
		}

		.mission-progress-fill,
		.mission-progress-indeterminate {
			transition: none;
			animation: none !important;
		}

		.mission-progress-fill-active::after {
			animation: none !important;
			opacity: 0;
		}
	}
</style>
