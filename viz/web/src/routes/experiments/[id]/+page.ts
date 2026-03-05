import { createApi } from '$lib/api/client';
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ params, fetch }) => {
	const api = createApi(fetch);
	const [experiment, runs, roundResults, configs, runJobs, studies, ensembles, capabilities] = await Promise.all([
		api.getExperiment(params.id),
		api.getExperimentRuns(params.id),
		api.getExperimentRoundResults(params.id).catch(() => []),
		api.getExperimentConfigs(params.id, { limit: 500, offset: 0 }),
		api.listRunJobs({ experiment_id: params.id, limit: 200, offset: 0 }),
		api.getExperimentStudies(params.id).catch(() => []),
		api.getExperimentEnsembles(params.id).catch(() => []),
		api.getSystemCapabilities().catch(() => ({ read_only: true, write_controls: false }))
	]);
	return { experiment, runs, roundResults, configs, runJobs, studies, ensembles, capabilities };
};
