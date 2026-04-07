import { createApi, type ExperimentOverviewResponse } from '$lib/api/client';
import type { PageLoad } from './$types';

function fallbackOverview(): ExperimentOverviewResponse {
	return {
		generated_at: null,
		summary: {
			total_experiments: 0,
			active_experiments: 0,
			completed_experiments: 0,
			live_experiments: 0,
			live_runs: 0,
			queued_runs: 0,
			attention_count: 0
		},
		experiments: [],
		live_experiments: [],
		recent_activity: [],
		sources: []
	};
}

export const load: PageLoad = async ({ fetch }) => {
	const api = createApi(fetch);
	const overview = await api.getExperimentsOverview({ include_remote: false }).catch(() => fallbackOverview());
	return {
		overview,
		// The route SSRs a fast local-only overview first, then the page hydrates and
		// performs the remote-aware refresh. Keep mission-control in a bootstrap
		// loading state until that first SSH-backed refresh settles.
		overviewPending: true
	};
};
