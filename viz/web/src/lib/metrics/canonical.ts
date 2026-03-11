export const CANONICAL_METRIC_KEYS = [
	'corr20v2_mean',
	'corr20v2_std',
	'corr20v2_sharpe',
	'corr20v2_n_eras',
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
	'payout_estimate_mean',
	'payout_estimate_std',
	'payout_estimate_sharpe',
	'payout_estimate_n_eras',
	'bmc_mean',
	'bmc_std',
	'bmc_sharpe',
	'bmc_n_eras',
	'bmc_coverage_ratio_rows',
	'bmc_coverage_ratio_eras',
	'bmc_last_200_eras_mean',
	'cwmm_mean',
	'cwmm_std',
	'cwmm_sharpe',
	'feature_exposure_mean',
	'feature_exposure_std',
	'feature_exposure_sharpe',
	'max_feature_exposure',
	'max_drawdown'
] as const;

export const RUNOPS_MAIN_METRICS = [
	'corr20v2_sharpe',
	'corr20v2_mean',
	'fnc_mean',
	'mmc_mean',
	'payout_estimate_mean',
	'bmc_mean',
	'feature_exposure_mean',
	'max_feature_exposure',
	'max_drawdown'
] as const;

export const RUNOPS_ALL_SCORING_METRICS = [
	'corr20v2_mean',
	'corr20v2_std',
	'corr20v2_sharpe',
	'fnc_mean',
	'fnc_std',
	'fnc_sharpe',
	'mmc_mean',
	'mmc_std',
	'mmc_sharpe',
	'payout_estimate_mean',
	'bmc_mean',
	'bmc_std',
	'bmc_sharpe',
	'bmc_last_200_eras_mean',
	'cwmm_mean',
	'cwmm_std',
	'cwmm_sharpe',
	'feature_exposure_mean',
	'feature_exposure_std',
	'feature_exposure_sharpe',
	'max_feature_exposure',
	'max_drawdown'
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
		case 'bmc_std':
			return 'BMC Std';
		case 'bmc_sharpe':
			return 'BMC Sharpe';
		case 'bmc_last_200_eras_mean':
			return 'BMC Mean (Last 200 Eras)';
		case 'cwmm_mean':
			return 'CWMM Mean';
		case 'cwmm_std':
			return 'CWMM Std';
		case 'cwmm_sharpe':
			return 'CWMM Sharpe';
		case 'feature_exposure_mean':
			return 'Feature Exposure Mean';
		case 'feature_exposure_std':
			return 'Feature Exposure Std';
		case 'feature_exposure_sharpe':
			return 'Feature Exposure Sharpe';
		case 'max_feature_exposure':
			return 'Max Feature Exposure';
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
