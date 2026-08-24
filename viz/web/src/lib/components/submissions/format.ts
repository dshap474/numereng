/**
 * Formatting and derivation helpers shared by the Submissions page components.
 *
 * USAGE:
 *   import { formatNumber, formatDate, sortSubmissionItems } from '$lib/components/submissions/format';
 *   const label = formatDate(item.summary.pulled_at);
 *
 * Every helper is null-safe: submission snapshots written on another machine can be thin
 * (no uploads/source/hosted_pickle), so missing data must render as a dash, never "undefined".
 */

import type { SubmissionItem, SubmissionRound } from '$lib/api/client';

// --------------------------------------------------------------------------- //
// Types
// --------------------------------------------------------------------------- //

export type SubmissionSortKey = 'live_since' | 'name' | 'mmc20' | 'corr20' | 'rounds';

export type LiveRoundMetricKey =
	| 'mmc20'
	| 'corr20'
	| 'mmc60'
	| 'corr60'
	| 'mmc'
	| 'corr'
	| 'bmc'
	| 'fnc';


export const DASH = '—';

// --------------------------------------------------------------------------- //
// Scalar formatting
// --------------------------------------------------------------------------- //

export function formatNumber(value: unknown, digits = 4): string {
	if (typeof value !== 'number' || !Number.isFinite(value)) return DASH;
	return value.toFixed(digits);
}

export function formatText(value: unknown): string {
	if (typeof value === 'string' && value.trim()) return value;
	if (typeof value === 'number' && Number.isFinite(value)) return String(value);
	return DASH;
}

export function formatPercentile(value: unknown): string | null {
	if (typeof value !== 'number' || !Number.isFinite(value)) return null;
	const scaled = value <= 1 && value >= 0 ? value * 100 : value;
	return `${Math.round(scaled)}%`;
}

export function formatStatusLabel(value: unknown): string {
	if (typeof value !== 'string' || !value.trim()) return 'unknown';
	return value.replaceAll('_', ' ');
}

export function signToneClass(value: unknown): string {
	if (typeof value !== 'number' || !Number.isFinite(value) || value === 0) return 'text-foreground';
	return value > 0 ? 'text-positive' : 'text-negative';
}

// --------------------------------------------------------------------------- //
// Dates and freshness
// --------------------------------------------------------------------------- //

export function dateTime(value: unknown): number {
	if (typeof value !== 'string' || !value.trim()) return 0;
	const parsed = Date.parse(value);
	return Number.isNaN(parsed) ? 0 : parsed;
}

export function formatDate(value: unknown): string {
	if (typeof value !== 'string' || !value.trim()) return DASH;
	const calendarMatch = value.match(/^(\d{4})-(\d{2})-(\d{2})$/);
	const date = calendarMatch
		? new Date(Number(calendarMatch[1]), Number(calendarMatch[2]) - 1, Number(calendarMatch[3]))
		: new Date(value);
	if (Number.isNaN(date.getTime())) return value;
	return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

export function formatShortDate(value: unknown): string {
	const formatted = formatDate(value);
	if (formatted === DASH) return DASH;
	if (typeof value !== 'string') return formatted;
	const date = new Date(value);
	if (Number.isNaN(date.getTime())) return formatted;
	return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

// --------------------------------------------------------------------------- //
// Round helpers
// --------------------------------------------------------------------------- //

export function roundNumber(row: SubmissionRound | null | undefined): string {
	return formatText(row?.round ?? row?.round_number);
}

export function roundSortValue(row: SubmissionRound): number {
	const raw = row.round ?? row.round_number;
	const parsed = Number(raw);
	return Number.isFinite(parsed) ? parsed : 0;
}

export function roundState(row: SubmissionRound | null | undefined): string {
	const state = row?.state ?? row?.status;
	return typeof state === 'string' && state.trim() ? state : 'unknown';
}

export function roundWindow(row: SubmissionRound | null | undefined): string {
	const parts = [row?.open_date, row?.close_date, row?.resolve_date]
		.map((value) => formatShortDate(value))
		.filter((value) => value !== DASH);
	if (parts.length === 0) return 'dates not recorded';
	return parts.join(' → ');
}

export function roundMetric(row: SubmissionRound, key: LiveRoundMetricKey): number | null {
	const value = row[key];
	return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

export function isScoredRound(row: SubmissionRound): boolean {
	return (['mmc20', 'corr20', 'mmc60', 'corr60'] as LiveRoundMetricKey[]).some(
		(key) => roundMetric(row, key) !== null
	);
}

export function mean(values: number[]): number | null {
	if (values.length === 0) return null;
	return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function meanMetric(rows: SubmissionRound[], key: LiveRoundMetricKey): number | null {
	return mean(rows.map((row) => roundMetric(row, key)).filter((value): value is number => value !== null));
}

// --------------------------------------------------------------------------- //
// List sorting
// --------------------------------------------------------------------------- //

function latestScored(item: SubmissionItem, key: 'mmc20' | 'corr20'): number {
	const value = item.summary.latest_scored_round?.[key];
	return typeof value === 'number' && Number.isFinite(value) ? value : Number.NEGATIVE_INFINITY;
}

export function sortSubmissionItems(items: SubmissionItem[], sortKey: SubmissionSortKey): SubmissionItem[] {
	const sorted = [...items];
	sorted.sort((a, b) => {
		switch (sortKey) {
			case 'name':
				return a.model_name.localeCompare(b.model_name);
			case 'mmc20':
				return latestScored(b, 'mmc20') - latestScored(a, 'mmc20');
			case 'corr20':
				return latestScored(b, 'corr20') - latestScored(a, 'corr20');
			case 'rounds':
				return b.summary.round_count - a.summary.round_count;
			case 'live_since':
			default:
				return (
					dateTime(b.summary.live_started_at) - dateTime(a.summary.live_started_at) ||
					a.model_name.localeCompare(b.model_name)
				);
		}
	});
	return sorted;
}
