<script lang="ts">
	import {
		api,
		type SubmissionCalibrationResponse,
		type SubmissionCalibrationRow,
		type SubmissionDetail,
		type SubmissionItem,
		type SubmissionListResponse,
		type SubmissionRound
	} from '$lib/api/client';
	import LocalLiveCalibrationChart from '$lib/components/charts/LocalLiveCalibrationChart.svelte';
	import AccentCard from '$lib/components/ui/AccentCard.svelte';

	type LocalMetricKey = 'bmc200' | 'bmc' | 'corr' | 'mmc' | 'fnc';
	type LiveMetricKey = 'mmc20' | 'corr20' | 'mmc60' | 'corr60';
	type CalibrationSort = 'live_rank' | 'live_since';
	type CalibrationPoint = {
		id: string;
		modelName: string;
		roundNumber?: number | null;
		target?: string | null;
		confidence?: string | null;
		liveStartedAt?: string | null;
		x: number;
		y: number;
	};
	type RoundLike = {
		round?: number | string | null;
		round_number?: number | string | null;
		close_date?: string | null;
		resolve_date?: string | null;
		state?: string | null;
		status?: string | null;
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
	let selectedModel = $state<string | null>(routeSubmissions().items[0]?.model_name ?? null);
	let selectedDetail = $state<SubmissionDetail | null>(null);
	let detailLoading = $state(false);
	let activeTab = $state<'live' | 'calibration'>('live');
	let calibrationSort = $state<CalibrationSort>('live_rank');
	let localMetric = $state<LocalMetricKey>('bmc200');
	let liveMetric = $state<LiveMetricKey>('mmc20');
	let targetFilter = $state('all');
	let featureFilter = $state('all');
	let recipeFilter = $state('all');
	let confidenceFilter = $state('all');

	let selectedItem = $derived(submissions.items.find((item) => item.model_name === selectedModel) ?? null);
	let rounds = $derived(selectedDetail?.rounds ?? []);
	let confidenceByModel = $derived.by(() => buildConfidenceByModel(calibration.rows));
	let rankByModel = $derived.by(() => buildRankByModel(calibration.report));
	let calibrationRows = $derived.by(() => [...calibration.rows].sort(compareCalibrationRows));
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

	$effect(() => {
		submissions = routeSubmissions();
		calibration = routeCalibration();
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

	function compareCalibrationRows(a: SubmissionCalibrationRow, b: SubmissionCalibrationRow): number {
		if (calibrationSort === 'live_since') {
			const timeA = dateTime(a.live_started_at);
			const timeB = dateTime(b.live_started_at);
			return timeB - timeA || compareScoredDesc(a, b) || a.model_name.localeCompare(b.model_name) || compareRoundDesc(a, b);
		}
		const rankA = rankByModel.get(a.model_name)?.live_rank ?? Number.POSITIVE_INFINITY;
		const rankB = rankByModel.get(b.model_name)?.live_rank ?? Number.POSITIVE_INFINITY;
		return rankA - rankB || compareScoredDesc(a, b) || compareRoundDesc(a, b) || a.model_name.localeCompare(b.model_name);
	}

	function compareRoundDesc(a: RoundLike, b: RoundLike): number {
		const roundA = Number(a.round_number ?? a.round ?? 0);
		const roundB = Number(b.round_number ?? b.round ?? 0);
		return (Number.isFinite(roundB) ? roundB : 0) - (Number.isFinite(roundA) ? roundA : 0);
	}

	function compareScoredDesc(a: SubmissionCalibrationRow, b: SubmissionCalibrationRow): number {
		return Number(Boolean(b.has_live_score)) - Number(Boolean(a.has_live_score));
	}

	function formatNumber(value: unknown, digits = 4): string {
		if (typeof value !== 'number' || Number.isNaN(value)) return 'n/a';
		return value.toFixed(digits);
	}

	function formatText(value: unknown): string {
		if (typeof value === 'string' && value.trim()) return value;
		if (typeof value === 'number') return String(value);
		return 'n/a';
	}

	function roundNumber(row: RoundLike | null | undefined): string {
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

	function roundDates(row: RoundLike | null | undefined): string {
		const closeDate = formatDate(row?.close_date);
		const resolveDate = formatDate(row?.resolve_date);
		if (closeDate === 'n/a' && resolveDate === 'n/a') return 'dates n/a';
		if (resolveDate === 'n/a') return `close ${closeDate}`;
		if (closeDate === 'n/a') return `resolve ${resolveDate}`;
		return `${closeDate} -> ${resolveDate}`;
	}

	function roundState(row: RoundLike | null | undefined): string {
		return formatText(row?.state ?? row?.status);
	}

	function rowTone(item: SubmissionItem): string {
		if (item.summary.resolving_round_count > 0) return 'border-l-sky-400/70';
		if (item.summary.resolved_round_count > 0) return 'border-l-emerald-400/80';
		return 'border-l-transparent';
	}

	function calibrationRowTone(row: SubmissionCalibrationRow): string {
		if (row.state === 'resolved') return 'border-l-emerald-400/80';
		if (row.has_live_score) return 'border-l-sky-400/70';
		return 'border-l-transparent';
	}

	function rankText(value: unknown): string {
		if (typeof value !== 'number' || Number.isNaN(value)) return 'n/a';
		return `#${value}`;
	}

	function deltaText(value: unknown): string {
		if (typeof value !== 'number' || Number.isNaN(value)) return 'n/a';
		if (value === 0) return '0';
		return value > 0 ? `+${value}` : String(value);
	}

	function confidenceText(value: unknown): string {
		if (typeof value !== 'string' || !value.trim()) return 'waiting';
		return value.replaceAll('_', ' ');
	}

	function dateTime(value: unknown): number {
		if (typeof value !== 'string' || !value.trim()) return 0;
		const parsed = Date.parse(value);
		return Number.isNaN(parsed) ? 0 : parsed;
	}

	function buildConfidenceByModel(rows: SubmissionCalibrationRow[]): Map<string, string> {
		const counts = new Map<string, { scored: number; resolved: number }>();
		for (const row of rows) {
			const current = counts.get(row.model_name) ?? { scored: 0, resolved: 0 };
			if (row.has_live_score) {
				current.scored += 1;
				if (row.state === 'resolved') current.resolved += 1;
			}
			counts.set(row.model_name, current);
		}
		const confidence = new Map<string, string>();
		for (const [modelName, count] of counts) {
			if (count.scored === 0) confidence.set(modelName, 'waiting');
			else if (count.resolved >= 8) confidence.set(modelName, 'stronger_signal');
			else if (count.resolved >= 3) confidence.set(modelName, 'resolved_signal');
			else if (count.scored >= 5) confidence.set(modelName, 'usable_estimate');
			else confidence.set(modelName, 'early');
		}
		return confidence;
	}

	function buildRankByModel(report: Record<string, unknown>): Map<string, Record<string, unknown>> {
		const scopes = isRecord(report.scopes) ? report.scopes : {};
		const allScored = isRecord(scopes.all_scored) ? scopes.all_scored : {};
		const rows = Array.isArray(allScored.rank_deltas) ? allScored.rank_deltas : [];
		const ranks = new Map<string, Record<string, unknown>>();
		for (const row of rows) {
			if (isRecord(row) && typeof row.model_name === 'string') ranks.set(row.model_name, row);
		}
		return ranks;
	}

	function isRecord(value: unknown): value is Record<string, unknown> {
		return typeof value === 'object' && value !== null && !Array.isArray(value);
	}

	function confidenceForRow(row: SubmissionCalibrationRow): string {
		return confidenceByModel.get(row.model_name) ?? 'waiting';
	}

	function rankForRow(row: SubmissionCalibrationRow, key: 'local_rank' | 'live_rank' | 'rank_delta'): unknown {
		return rankByModel.get(row.model_name)?.[key] ?? null;
	}

	function localMetricValue(row: SubmissionCalibrationRow): number | null {
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

	function liveMetricValue(row: SubmissionCalibrationRow): number | null {
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

	function buildChartPoints(rows: SubmissionCalibrationRow[]): CalibrationPoint[] {
		return rows
			.map((row) => ({
				id: `${row.model_name}:${row.round_number ?? 'na'}:${localMetric}:${liveMetric}`,
				modelName: row.model_name,
				roundNumber: row.round_number,
				target: row.target,
				confidence: confidenceForRow(row),
				liveStartedAt: row.live_started_at,
				x: localMetricValue(row),
				y: liveMetricValue(row)
			}))
			.filter(
				(row): row is CalibrationPoint =>
					row.x !== null && row.y !== null
			);
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
		<div class="flex items-center gap-2">
			<button
				type="button"
				class="rounded-md border px-3 py-1.5 text-sm transition-colors {activeTab === 'live' ? 'border-white/18 bg-white/12 text-foreground' : 'border-white/8 text-muted-foreground hover:bg-white/[0.04]'}"
				onclick={() => (activeTab = 'live')}
			>
				Live Scores
			</button>
			<button
				type="button"
				class="rounded-md border px-3 py-1.5 text-sm transition-colors {activeTab === 'calibration' ? 'border-white/18 bg-white/12 text-foreground' : 'border-white/8 text-muted-foreground hover:bg-white/[0.04]'}"
				onclick={() => (activeTab = 'calibration')}
			>
				Calibration
			</button>
		</div>

		{#if activeTab === 'live'}
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
		{:else}
			<AccentCard paddingClass="px-4 py-4" roundedClass="rounded-lg">
				<div class="flex flex-col gap-4">
					<div class="flex flex-col gap-3 xl:flex-row xl:items-end xl:justify-between">
						<div>
							<h2 class="text-sm font-semibold text-foreground">
								{localMetricLabels[localMetric]} vs {liveMetricLabels[liveMetric]}
							</h2>
							<p class="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
								<span>{chartStats.n} usable points</span>
								{#if chartStats.r !== null}
									<span class="font-mono tabular-nums">r={formatNumber(chartStats.r, 3)}</span>
									<span class="font-mono tabular-nums">R²={formatNumber(chartStats.r2, 3)}</span>
								{:else}
									<span>not enough points for regression</span>
								{/if}
							</p>
						</div>
						<div class="grid gap-2 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-7">
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

					<LocalLiveCalibrationChart
						points={chartPoints}
						xLabel={localMetricLabels[localMetric]}
						yLabel={liveMetricLabels[liveMetric]}
					/>
				</div>
			</AccentCard>

			<AccentCard paddingClass="p-0" roundedClass="rounded-lg" class="overflow-hidden">
				<div class="border-b border-white/8 px-4 py-3">
					<h2 class="text-sm font-semibold text-foreground">Calibration Observations</h2>
				</div>
				<div class="overflow-x-auto">
					<table class="min-w-full text-left text-sm">
						<thead class="bg-white/[0.025] text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
								<tr>
									<th class="px-4 py-3 font-medium">Model</th>
									<th class="px-4 py-3 font-medium">Round / Dates</th>
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
								{#each filteredCalibrationItems as row (`${row.model_name}:${row.round_number ?? 'na'}`)}
									<tr class="border-l {calibrationRowTone(row)}">
										<td class="px-4 py-3">
											<div class="font-mono text-sm font-semibold text-foreground">{row.model_name}</div>
											<div class="mt-1 text-xs text-muted-foreground">
												{formatText(row.package_id ?? row.local_metric_source)}
											</div>
										</td>
										<td class="px-4 py-3">
											<div class="font-mono text-foreground">{roundNumber(row)}</div>
											<div class="mt-0.5 text-xs text-muted-foreground">{roundDates(row)}</div>
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
									<td class="px-4 py-6 text-sm text-muted-foreground" colspan="13">
										No calibration rows match the current filters.
									</td>
								</tr>
							{/if}
						</tbody>
					</table>
				</div>
			</AccentCard>
		{/if}
	{/if}
</div>
