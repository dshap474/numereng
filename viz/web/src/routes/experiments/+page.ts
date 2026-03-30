import type { ExperimentOverviewItem, ExperimentOverviewResponse } from '$lib/api/client';
import type { PageLoad } from './$types';

function fallbackOverview(experiments: ExperimentOverviewItem[]): ExperimentOverviewResponse {
	return {
		generated_at: null,
		summary: {
			total_experiments: experiments.length,
			active_experiments: experiments.filter((item) => item.status === 'active').length,
			completed_experiments: experiments.filter((item) => item.status === 'complete').length,
			live_experiments: 0,
			live_runs: 0,
			queued_runs: 0,
			attention_count: 0
		},
		experiments,
		live_experiments: [],
		recent_activity: [],
		sources: []
	};
}

export const load: PageLoad = async ({ parent }) => {
	const parentData = await parent();
	const experiments = Array.isArray(parentData.experiments)
		? (parentData.experiments as ExperimentOverviewItem[])
		: [];
	const overview =
		parentData.experimentsOverview && typeof parentData.experimentsOverview === 'object'
			? (parentData.experimentsOverview as ExperimentOverviewResponse)
			: fallbackOverview(experiments);
	return {
		overview,
		overviewPending: !parentData.experimentsOverview
	};
};
