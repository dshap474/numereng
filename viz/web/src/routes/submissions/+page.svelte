<!--
	Submissions page: live-round snapshots for submitted Numerai models plus the
	local-vs-live calibration view.

	USAGE:
		route `/submissions`; data comes from `+page.ts` (`/api/submissions`,
		`/api/submissions/calibration`) and per-model detail fetches.

	Read-only surface: nothing here mutates the store or calls Numerai.
-->
<script lang="ts">
	import {
		api,
		type SubmissionCalibrationObservation,
		type SubmissionCalibrationResponse,
		type SubmissionDetail,
		type SubmissionListResponse
	} from '$lib/api/client';
	import LocalLiveCalibrationChart from '$lib/components/charts/LocalLiveCalibrationChart.svelte';
	import SubmissionDetailPanel from '$lib/components/submissions/SubmissionDetailPanel.svelte';
	import SubmissionModelTable from '$lib/components/submissions/SubmissionModelTable.svelte';
	import {
		DASH,
		dateTime,
		formatNumber,
		formatShortDate,
		formatText,
		sortSubmissionItems,
		type SubmissionSortKey
	} from '$lib/components/submissions/format';
	import AccentCard from '$lib/components/ui/AccentCard.svelte';

	// ----------------------------------------------------------------------- //
	// Types and props
	// ----------------------------------------------------------------------- //

	type LocalMetricKey = 'bmc200' | 'bmc' | 'corr' | 'mmc' | 'fnc';
	type LiveMetricKey = 'mmc20' | 'corr20' | 'mmc60' | 'corr60';
	type CalibrationScope = 'all_scored' | 'resolved_only';
	type CalibrationSort = 'live_rank' | 'live_since';
	type CalibrationPoint = {
		id: string;
		modelName: string;
		detailLabel?: string | null;
		target?: string | null;
		confidence?: string | null;
		liveStartedAt?: string | null;
		x: number;
		y: number;
	};

	let {
		data
	}: {
		data: {
			submissions: SubmissionListResponse;
			calibration: SubmissionCalibrationResponse;
		};
	} = $props();

	const routeSubmissions = () => data.submissions;
	const routeCalibration = () => data.calibration;
	let submissions = $state<SubmissionListResponse>(routeSubmissions());
	let calibration = $state<SubmissionCalibrationResponse>(routeCalibration());
	let selectedModel = $state<string | null>(null);
	let selectedDetail = $state<SubmissionDetail | null>(null);
	let detailLoading = $state(false);
	let activeTab = $state<'live' | 'calibration'>('live');
	let submissionSort = $state<SubmissionSortKey>('live_since');
	let calibrationScope = $state<CalibrationScope>('all_scored');
	let calibrationSort = $state<CalibrationSort>('live_rank');
	let localMetric = $state<LocalMetricKey>('bmc200');
	let liveMetric = $state<LiveMetricKey>('mmc20');
	let targetFilter = $state('all');
	let featureFilter = $state('all');
	let recipeFilter = $state('all');
	let confidenceFilter = $state('all');

	// ----------------------------------------------------------------------- //
	// Derived state
	// ----------------------------------------------------------------------- //

	let sortedItems = $derived(sortSubmissionItems(submissions.items, submissionSort));
	let selectedItem = $derived(sortedItems.find((item) => item.model_name === selectedModel) ?? null);
	let calibrationRows = $derived.by(() =>
		[...(calibration.observations ?? [])]
			.filter((row) => row.scope === calibrationScope)
			.sort(compareCalibrationRows)
	);
	let filteredCalibrationItems = $derived.by(() =>
		calibrationRows.filter((row) => {
			const confidence = confidenceForRow(row);
			return (
				(targetFilter === 'all' || row.target === targetFilter) &&
				(featureFilter === 'all' || row.feature_scope === featureFilter) &&
				(recipeFilter === 'all' || row.recipe === recipeFilter) &&
				(confidenceFilter === 'all' || confidence === confidenceFilter)
			);
		})
	);
	let chartPoints = $derived.by(() => buildChartPoints(filteredCalibrationItems));
	let chartStats = $derived.by(() => buildChartStats(chartPoints));

	// ----------------------------------------------------------------------- //
	// Effects
	// ----------------------------------------------------------------------- //

	$effect(() => {
		submissions = routeSubmissions();
		calibration = routeCalibration();
	});

	$effect(() => {
		if (sortedItems.length === 0) {
			selectedModel = null;
			return;
		}
		if (!sortedItems.some((item) => item.model_name === selectedModel)) {
			selectedModel = sortedItems[0].model_name;
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

	// ----------------------------------------------------------------------- //
	// Header helpers
	// ----------------------------------------------------------------------- //

	function tabClass(tab: 'live' | 'calibration'): string {
		return activeTab === tab
			? 'rounded-md bg-white/12 px-3 py-1.5 text-sm text-foreground'
			: 'rounded-md px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-white/[0.05]';
	}

	// ----------------------------------------------------------------------- //
	// Calibration helpers
	// ----------------------------------------------------------------------- //

	const localMetricLabels: Record<LocalMetricKey, string> = {
		bmc200: 'Local BMC200',
		bmc: 'Local BMC',
		corr: 'Local CORR',
		mmc: 'Local MMC',
		fnc: 'Local FNC'
	};
	const liveMetricLabels: Record<LiveMetricKey, string> = {
		mmc20: 'Live MMC20',
		corr20: 'Live CORR20',
		mmc60: 'Live MMC60',
		corr60: 'Live CORR60'
	};

	function compareCalibrationRows(
		a: SubmissionCalibrationObservation,
		b: SubmissionCalibrationObservation
	): number {
		if (calibrationSort === 'live_since') {
			const timeA = dateTime(a.live_started_at);
			const timeB = dateTime(b.live_started_at);
			return timeB - timeA || compareScoredDesc(a, b) || a.model_name.localeCompare(b.model_name);
		}
		const rankA = typeof a.live_rank === 'number' ? a.live_rank : Number.POSITIVE_INFINITY;
		const rankB = typeof b.live_rank === 'number' ? b.live_rank : Number.POSITIVE_INFINITY;
		return rankA - rankB || compareScoredDesc(a, b) || a.model_name.localeCompare(b.model_name);
	}

	function compareScoredDesc(
		a: SubmissionCalibrationObservation,
		b: SubmissionCalibrationObservation
	): number {
		return Number(b.scored_round_count ?? 0) - Number(a.scored_round_count ?? 0);
	}

	function observationRounds(row: SubmissionCalibrationObservation): string {
		const first = row.first_round_number;
		const latest = row.latest_round_number;
		if (typeof first !== 'number' && typeof latest !== 'number') return DASH;
		if (first === latest || typeof first !== 'number') return formatText(latest);
		return `${first}-${latest}`;
	}

	function observationWindow(row: SubmissionCalibrationObservation): string {
		const first = formatShortDate(row.first_close_date ?? row.live_started_at);
		const latest = formatShortDate(row.latest_close_date ?? row.live_ended_at);
		if (first === DASH && latest === DASH) return 'window not recorded';
		if (latest === DASH || first === latest) return first;
		if (first === DASH) return latest;
		return `${first} -> ${latest}`;
	}

	function calibrationRowTone(row: SubmissionCalibrationObservation): string {
		if (row.confidence === 'resolved_signal' || row.confidence === 'stronger_signal') return 'border-l-emerald-400/80';
		if (row.has_live_score) return 'border-l-sky-400/70';
		return 'border-l-transparent';
	}

	function rankText(value: unknown): string {
		if (typeof value !== 'number' || Number.isNaN(value)) return DASH;
		return `#${value}`;
	}

	function deltaText(value: unknown): string {
		if (typeof value !== 'number' || Number.isNaN(value)) return DASH;
		if (value === 0) return '0';
		return value > 0 ? `+${value}` : String(value);
	}

	function confidenceText(value: unknown): string {
		if (typeof value !== 'string' || !value.trim()) return 'waiting';
		return value.replaceAll('_', ' ');
	}

	function confidenceForRow(row: SubmissionCalibrationObservation): string {
		return typeof row.confidence === 'string' && row.confidence ? row.confidence : 'waiting';
	}

	function rankForRow(row: SubmissionCalibrationObservation, key: 'local_rank' | 'live_rank' | 'rank_delta'): unknown {
		return row[key] ?? null;
	}

	function localMetricValue(row: SubmissionCalibrationObservation): number | null {
		const value =
			localMetric === 'bmc200'
				? row.local_bmc200_mean
				: localMetric === 'bmc'
					? row.local_bmc_mean
					: localMetric === 'corr'
						? row.local_corr_mean
						: localMetric === 'mmc'
							? row.local_mmc_mean
							: row.local_fnc_mean;
		return typeof value === 'number' && Number.isFinite(value) ? value : null;
	}

	function liveMetricValue(row: SubmissionCalibrationObservation): number | null {
		const value =
			liveMetric === 'mmc20'
				? row.live_mmc20
				: liveMetric === 'corr20'
					? row.live_corr20
					: liveMetric === 'mmc60'
						? row.live_mmc60
						: row.live_corr60;
		return typeof value === 'number' && Number.isFinite(value) ? value : null;
	}

	function buildChartPoints(rows: SubmissionCalibrationObservation[]): CalibrationPoint[] {
		return rows
			.map((row): CalibrationPoint | null => {
				const x = localMetricValue(row);
				const y = liveMetricValue(row);
				if (x === null || y === null) return null;
				return {
					id: `${row.model_name}:${row.upload_id ?? 'current'}:${row.scope}:${localMetric}:${liveMetric}`,
					modelName: row.model_name,
					detailLabel: `${row.scored_round_count ?? 0} scored rounds`,
					target: row.target,
					confidence: confidenceForRow(row),
					liveStartedAt: row.live_started_at,
					x,
					y
				};
			})
			.filter((point): point is CalibrationPoint => point !== null);
	}

	function buildChartStats(points: CalibrationPoint[]): { n: number; r: number | null; r2: number | null } {
		const n = points.length;
		if (n < 3) return { n, r: null, r2: null };
		const meanX = points.reduce((sum, point) => sum + point.x, 0) / n;
		const meanY = points.reduce((sum, point) => sum + point.y, 0) / n;
		const sxx = points.reduce((sum, point) => sum + (point.x - meanX) ** 2, 0);
		const syy = points.reduce((sum, point) => sum + (point.y - meanY) ** 2, 0);
		const sxy = points.reduce((sum, point) => sum + (point.x - meanX) * (point.y - meanY), 0);
		if (sxx === 0 || syy === 0) return { n, r: null, r2: null };
		const r = sxy / Math.sqrt(sxx * syy);
		return { n, r, r2: r * r };
	}
</script>

<svelte:head>
	<title>Submissions · Numereng</title>
</svelte:head>

<!-- ------------------------------------------------------------------------ -->
<!-- Markup                                                                    -->
<!-- ------------------------------------------------------------------------ -->

<div class="-mx-8 -mt-14 -mb-8 flex h-screen min-h-0 flex-col overflow-x-hidden overflow-y-auto md:-mt-8 xl:overflow-hidden">
	<header class="flex shrink-0 flex-wrap items-end justify-between gap-4 px-8 pt-14 pb-5 md:pt-8">
		<div class="min-w-0">
			<p class="text-[11px] uppercase tracking-[0.22em] text-muted-foreground">Live submission snapshots</p>
			<h1 class="mt-1 text-2xl font-semibold tracking-tight text-foreground">Submissions</h1>
		</div>
		{#if submissions.items.length > 0}
			<div class="flex w-fit items-center gap-1 rounded-lg border border-white/8 bg-white/[0.02] p-1">
				<button type="button" class={tabClass('live')} onclick={() => (activeTab = 'live')}>Live Scores</button>
				<button type="button" class={tabClass('calibration')} onclick={() => (activeTab = 'calibration')}>
					Calibration
				</button>
			</div>
		{/if}
	</header>

	{#if submissions.items.length === 0}
		<div class="px-8 pb-8 xl:min-h-0 xl:flex-1">
			<AccentCard paddingClass="px-5 py-5" roundedClass="rounded-lg">
				<div class="flex flex-col gap-2">
					<p class="text-sm font-medium text-foreground">No submission snapshots found.</p>
					<p class="text-sm text-muted-foreground">
						Expected local folders under <code class="font-mono text-foreground">{submissions.root}</code>.
					</p>
					<p class="text-sm text-muted-foreground">
						Populate them with <code class="font-mono text-foreground">uv run numereng submissions refresh</code>, then
						<code class="font-mono text-foreground">uv run numereng submissions calibration update</code>.
					</p>
				</div>
			</AccentCard>
		</div>
	{:else if activeTab === 'live'}
		<div
			class="grid grid-cols-1 border-t-[1.5px] border-white/12 xl:min-h-0 xl:flex-1 xl:grid-cols-[minmax(0,1fr)_minmax(460px,0.92fr)] xl:overflow-hidden"
		>
			<SubmissionModelTable
				items={sortedItems}
				selected={selectedModel}
				sortKey={submissionSort}
				onSelect={(modelName) => (selectedModel = modelName)}
				onSortChange={(next) => (submissionSort = next)}
			/>
			<SubmissionDetailPanel item={selectedItem} detail={selectedDetail} loading={detailLoading} />
		</div>
	{:else}
		<div class="flex flex-col border-t-[1.5px] border-white/12 xl:min-h-0 xl:flex-1 xl:overflow-y-auto">
			<section class="shrink-0 bg-[#111115]">
				<div
					class="flex flex-col gap-3 border-y-[1.5px] border-white/12 bg-card px-5 py-4 xl:flex-row xl:items-end xl:justify-between"
				>
					<div>
						<p class="font-mono text-[11px] uppercase tracking-[0.28em] text-muted-foreground">Local vs live</p>
						<h2 class="mt-2 text-lg font-semibold text-foreground">
							{localMetricLabels[localMetric]} vs {liveMetricLabels[liveMetric]}
						</h2>
						<p class="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
							<span class="font-mono tabular-nums">{chartStats.n} usable points</span>
							{#if chartStats.r !== null}
								<span class="font-mono tabular-nums">r={formatNumber(chartStats.r, 3)}</span>
								<span class="font-mono tabular-nums">R²={formatNumber(chartStats.r2, 3)}</span>
							{:else}
								<span>not enough points for regression</span>
							{/if}
						</p>
					</div>
					<div class="grid gap-2 sm:grid-cols-2 lg:grid-cols-4 2xl:grid-cols-8">
							<label class="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
								Scope
								<select bind:value={calibrationScope} class="mt-1 w-full rounded-md border border-white/10 bg-background px-2 py-1.5 text-xs normal-case tracking-normal text-foreground">
									<option value="all_scored">all scored</option>
									<option value="resolved_only">resolved only</option>
								</select>
							</label>
							<label class="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
								X
								<select bind:value={localMetric} class="mt-1 w-full rounded-md border border-white/10 bg-background px-2 py-1.5 text-xs normal-case tracking-normal text-foreground">
									<option value="bmc200">Local BMC200</option>
									<option value="bmc">Local BMC</option>
									<option value="corr">Local CORR</option>
									<option value="mmc">Local MMC</option>
									<option value="fnc">Local FNC</option>
								</select>
							</label>
							<label class="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
								Y
								<select bind:value={liveMetric} class="mt-1 w-full rounded-md border border-white/10 bg-background px-2 py-1.5 text-xs normal-case tracking-normal text-foreground">
									<option value="mmc20">Live MMC20</option>
									<option value="corr20">Live CORR20</option>
									<option value="mmc60">Live MMC60</option>
									<option value="corr60">Live CORR60</option>
								</select>
							</label>
							<label class="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
								Target
								<select bind:value={targetFilter} class="mt-1 w-full rounded-md border border-white/10 bg-background px-2 py-1.5 text-xs normal-case tracking-normal text-foreground">
									<option value="all">all</option>
									<option value="ender20">ender20</option>
									<option value="ender60">ender60</option>
									<option value="cyrusd20">cyrusd20</option>
									<option value="cyrusd60">cyrusd60</option>
									<option value="cross_scope">cross_scope</option>
								</select>
							</label>
							<label class="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
								Feature
								<select bind:value={featureFilter} class="mt-1 w-full rounded-md border border-white/10 bg-background px-2 py-1.5 text-xs normal-case tracking-normal text-foreground">
									<option value="all">all</option>
									<option value="small">small</option>
									<option value="medium">medium</option>
									<option value="deep">deep</option>
									<option value="blend">blend</option>
								</select>
							</label>
							<label class="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
								Recipe
								<select bind:value={recipeFilter} class="mt-1 w-full rounded-md border border-white/10 bg-background px-2 py-1.5 text-xs normal-case tracking-normal text-foreground">
									<option value="all">all</option>
									<option value="moderate_lgbm">moderate_lgbm</option>
									<option value="standard_large_lgbm">standard_large_lgbm</option>
									<option value="cross_scope">cross_scope</option>
								</select>
							</label>
							<label class="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
								Confidence
								<select bind:value={confidenceFilter} class="mt-1 w-full rounded-md border border-white/10 bg-background px-2 py-1.5 text-xs normal-case tracking-normal text-foreground">
									<option value="all">all</option>
									<option value="waiting">waiting</option>
									<option value="early">early</option>
									<option value="usable_estimate">usable_estimate</option>
									<option value="resolved_signal">resolved_signal</option>
									<option value="stronger_signal">stronger_signal</option>
								</select>
							</label>
							<label class="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
								Sort
								<select bind:value={calibrationSort} class="mt-1 w-full rounded-md border border-white/10 bg-background px-2 py-1.5 text-xs normal-case tracking-normal text-foreground">
									<option value="live_rank">live rank</option>
									<option value="live_since">live since</option>
								</select>
						</label>
					</div>
				</div>

				<div class="px-5 py-4">
					<LocalLiveCalibrationChart
						points={chartPoints}
						xLabel={localMetricLabels[localMetric]}
						yLabel={liveMetricLabels[liveMetric]}
					/>
				</div>
			</section>

			<section class="bg-[#111115]">
				<div class="border-y-[1.5px] border-white/12 bg-card px-5 py-4">
					<p class="font-mono text-[11px] uppercase tracking-[0.28em] text-muted-foreground">Observation history</p>
					<h2 class="mt-2 text-lg font-semibold text-foreground">Calibration Observations</h2>
				</div>
				<div class="overflow-x-auto">
					<table class="min-w-full text-left text-sm">
						<thead
							class="sticky top-0 z-10 bg-card text-[11px] uppercase tracking-[0.16em] text-muted-foreground shadow-[inset_0_-1px_0_rgba(255,255,255,0.08)]"
						>
								<tr>
									<th class="px-5 py-3 font-medium">Model</th>
									<th class="px-4 py-3 font-medium">Upload / Window</th>
									<th class="px-4 py-3 text-right font-medium">Rounds</th>
									<th class="px-4 py-3 font-medium">Feature</th>
									<th class="px-4 py-3 font-medium">Target</th>
									<th class="px-4 py-3 font-medium">Recipe</th>
									<th class="px-4 py-3 text-right font-medium">Local BMC200</th>
									<th class="px-4 py-3 text-right font-medium">Local CORR</th>
									<th class="px-4 py-3 text-right font-medium">Live MMC20</th>
								<th class="px-4 py-3 text-right font-medium">Live CORR20</th>
								<th class="px-4 py-3 text-right font-medium">Local Rank</th>
								<th class="px-4 py-3 text-right font-medium">Live Rank</th>
								<th class="px-4 py-3 text-right font-medium">Delta</th>
								<th class="px-4 py-3 font-medium">Confidence</th>
							</tr>
							</thead>
							<tbody class="divide-y divide-white/6">
								{#each filteredCalibrationItems as row (`${row.model_name}:${row.upload_id ?? 'current'}:${row.scope}`)}
									<tr class="border-l-2 {calibrationRowTone(row)}">
										<td class="px-5 py-3">
											<div class="font-mono text-sm font-semibold text-foreground">{row.model_name}</div>
											<div class="mt-1 text-xs text-muted-foreground">
												{formatText(row.package_id ?? row.local_metric_source)}
											</div>
										</td>
										<td class="px-4 py-3">
											<div class="font-mono text-foreground">{formatText(row.upload_id)}</div>
											<div class="mt-0.5 text-xs text-muted-foreground">{observationWindow(row)}</div>
										</td>
										<td class="px-4 py-3 text-right font-mono text-foreground">
											<div>{formatText(row.scored_round_count)}</div>
											<div class="mt-0.5 text-xs text-muted-foreground">{observationRounds(row)}</div>
										</td>
										<td class="px-4 py-3 text-muted-foreground">{formatText(row.feature_scope)}</td>
										<td class="px-4 py-3 text-muted-foreground">{formatText(row.target)}</td>
										<td class="px-4 py-3 text-muted-foreground">{formatText(row.recipe)}</td>
										<td class="px-4 py-3 text-right font-mono text-foreground">
											{formatNumber(row.local_bmc200_mean)}
										</td>
									<td class="px-4 py-3 text-right font-mono text-foreground">
										{formatNumber(row.local_corr_mean)}
									</td>
									<td class="px-4 py-3 text-right font-mono text-foreground">
										{formatNumber(row.live_mmc20)}
									</td>
									<td class="px-4 py-3 text-right font-mono text-foreground">
										{formatNumber(row.live_corr20)}
									</td>
									<td class="px-4 py-3 text-right font-mono text-foreground">
										{rankText(rankForRow(row, 'local_rank'))}
									</td>
									<td class="px-4 py-3 text-right font-mono text-foreground">
										{rankText(rankForRow(row, 'live_rank'))}
									</td>
									<td class="px-4 py-3 text-right font-mono text-foreground">
										{deltaText(rankForRow(row, 'rank_delta'))}
									</td>
									<td class="px-4 py-3 text-muted-foreground">
										{confidenceText(confidenceForRow(row))}
									</td>
								</tr>
							{/each}
							{#if filteredCalibrationItems.length === 0}
								<tr>
									<td class="px-5 py-6 text-sm text-muted-foreground" colspan="14">
										No calibration observations match the current filters.
									</td>
								</tr>
							{/if}
						</tbody>
					</table>
				</div>
			</section>
		</div>
	{/if}
</div>
