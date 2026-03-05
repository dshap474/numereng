export const CANONICAL_METRIC_KEYS = [
	'corr20v2_mean',
	'corr20v2_std',
	'corr20v2_sharpe',
	'corr20v2_n_eras',
	'mmc_mean',
	'mmc_std',
	'mmc_sharpe',
	'mmc_n_eras',
	'mmc_coverage_rows',
	'mmc_coverage_eras',
	'mmc_coverage_ratio_rows',
	'mmc_coverage_ratio_eras',
	'payout_estimate_mean',
	'payout_estimate_std',
	'payout_estimate_sharpe',
	'payout_estimate_n_eras',
	'bmc_mean',
	'bmc_std',
	'bmc_sharpe',
	'bmc_n_eras',
	'bmc_coverage_ratio_rows',
	'bmc_coverage_ratio_eras'
] as const;

export const DASHBOARD_KEY_METRICS = [
	'corr20v2_sharpe',
	'corr20v2_mean',
	'mmc_mean',
	'payout_estimate_mean',
	'mmc_coverage_ratio_rows',
	'bmc_mean'
] as const;

export type CanonicalMetricKey =
	| (typeof CANONICAL_METRIC_KEYS)[number]
	| 'validation_profile'
	| 'bmc_model';

export function metricNumber(
	metrics: Record<string, unknown> | null | undefined,
	key: string
): number | null {
	if (!metrics) return null;
	const value = metrics[key];
	return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

export function metricLabel(key: string): string {
	switch (key) {
		case 'corr20v2_mean':
			return 'CORR20v2 Mean';
		case 'corr20v2_std':
			return 'CORR20v2 Std';
		case 'corr20v2_sharpe':
			return 'CORR20v2 Sharpe';
		case 'mmc_mean':
			return 'MMC Mean';
		case 'mmc_std':
			return 'MMC Std';
		case 'mmc_sharpe':
			return 'MMC Sharpe';
		case 'payout_estimate_mean':
			return 'Payout Estimate Mean';
		case 'payout_estimate_std':
			return 'Payout Estimate Std';
		case 'payout_estimate_sharpe':
			return 'Payout Estimate Sharpe';
		case 'mmc_coverage_ratio_rows':
			return 'MMC Coverage Rows';
		case 'bmc_mean':
			return 'BMC Mean (Diagnostic)';
		default:
			return key;
	}
}

export function targetLabel(value: {
	target_payout?: string | null;
	target_train?: string | null;
	target_col?: string | null;
	target?: string | null;
}): string {
	return value.target_payout ?? value.target_train ?? value.target_col ?? value.target ?? 'unknown';
}
