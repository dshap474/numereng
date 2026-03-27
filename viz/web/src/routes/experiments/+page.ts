import { createApi, type Experiment, type ExperimentOverviewResponse } from '$lib/api/client';
import type { PageLoad } from './$types';

function fallbackOverview(experiments: Experiment[]): ExperimentOverviewResponse {
	return {
		summary: {
			total_experiments: experiments.length,
			active_experiments: experiments.filter((item) => item.status === 'active').length,
			completed_experiments: experiments.filter((item) => item.status === 'complete').length,
			live_experiments: 0,
			live_runs: 0,
			queued_runs: 0,
			attention_count: 0
		},
		experiments: experiments.map((item) => ({
			experiment_id: item.experiment_id,
			name: item.name,
			status: item.status,
			created_at: item.created_at,
			updated_at: item.updated_at,
			run_count: item.run_count ?? (Array.isArray(item.runs) ? item.runs.length : 0),
			tags: item.tags ?? [],
			has_live: false,
			live_run_count: 0,
			attention_state: 'none',
			latest_activity_at: item.updated_at ?? item.created_at ?? null,
			source_kind: 'local',
			source_id: 'local',
			source_label: 'Local store',
			detail_href: `/experiments/${item.experiment_id}`
		})),
		live_experiments: [],
		recent_activity: []
	};
}

export const load: PageLoad = async ({ fetch, parent }) => {
	const api = createApi(fetch);
	const parentData = await parent();
	const experiments = Array.isArray(parentData.experiments)
		? (parentData.experiments as Experiment[])
		: [];

	try {
		const overview = await api.getExperimentsOverview();
		return { overview };
	} catch {
		return { overview: fallbackOverview(experiments) };
	}
};
