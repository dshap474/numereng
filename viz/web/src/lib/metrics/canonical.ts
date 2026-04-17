export const CANONICAL_METRIC_KEYS = [
	'bmc_last_200_eras_mean',
	'bmc_mean',
	'bmc_std',
	'bmc_sharpe',
	'bmc_n_eras',
	'corr_mean',
	'corr_std',
	'corr_sharpe',
	'corr_n_eras',
	'fnc_mean',
	'fnc_std',
	'fnc_sharpe',
	'mmc_mean',
	'mmc_std',
	'mmc_sharpe',
	'mmc_n_eras',
	'mmc_coverage_rows',
	'mmc_coverage_eras',
	'mmc_coverage_ratio_rows',
	'mmc_coverage_ratio_eras',
	'cwmm_mean',
	'cwmm_std',
	'cwmm_sharpe',
	'max_drawdown'
] as const;

export const RUNOPS_MAIN_METRICS = [
	'bmc_last_200_eras_mean',
	'bmc_mean',
	'corr_sharpe',
	'corr_mean',
	'fnc_mean',
	'mmc_mean',
	'cwmm_mean',
	'max_drawdown'
] as const;

export const RUNOPS_ALL_SCORING_METRICS = [
	'bmc_last_200_eras_mean',
	'bmc_mean',
	'bmc_std',
	'bmc_sharpe',
	'corr_mean',
	'corr_std',
	'corr_sharpe',
	'fnc_mean',
	'fnc_std',
	'fnc_sharpe',
	'mmc_mean',
	'mmc_std',
	'mmc_sharpe',
	'mmc_coverage_ratio_rows',
	'cwmm_mean',
	'cwmm_std',
	'cwmm_sharpe',
	'max_drawdown'
] as const;

export const DASHBOARD_KEY_METRICS = [
	'bmc_last_200_eras_mean',
	'bmc_mean',
	'corr_sharpe',
	'corr_mean',
	'mmc_mean',
	'mmc_coverage_ratio_rows'
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
		case 'bmc_last_200_eras_mean':
			return 'BMC Mean (Last 200 Eras)';
		case 'bmc_mean':
			return 'BMC Mean';
		case 'bmc_std':
			return 'BMC Std';
		case 'bmc_sharpe':
			return 'BMC Sharpe';
		case 'corr_mean':
			return 'CORR Mean';
		case 'corr_std':
			return 'CORR Std';
		case 'corr_sharpe':
			return 'CORR Sharpe';
		case 'fnc_mean':
			return 'FNC Mean';
		case 'fnc_std':
			return 'FNC Std';
		case 'fnc_sharpe':
			return 'FNC Sharpe';
		case 'mmc_mean':
			return 'MMC Mean';
		case 'mmc_std':
			return 'MMC Std';
		case 'mmc_sharpe':
			return 'MMC Sharpe';
		case 'mmc_coverage_ratio_rows':
			return 'MMC Coverage Rows';
		case 'cwmm_mean':
			return 'CWMM Mean';
		case 'cwmm_std':
			return 'CWMM Std';
		case 'cwmm_sharpe':
			return 'CWMM Sharpe';
		case 'max_drawdown':
			return 'Max Drawdown';
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
