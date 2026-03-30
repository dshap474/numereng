import { createApi } from '$lib/api/client';
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ params, fetch, url }) => {
	const api = createApi(fetch);
	const source = {
		source_kind: url.searchParams.get('source_kind'),
		source_id: url.searchParams.get('source_id')
	};
	const [experiment, runs, roundResults, configs, runJobs, studies, ensembles, capabilities] = await Promise.all([
		api.getExperiment(params.id, source),
		api.getExperimentRuns(params.id, source),
		api.getExperimentRoundResults(params.id, source).catch(() => []),
		api.getExperimentConfigs(params.id, { limit: 500, offset: 0, ...source }),
		api.listRunJobs({ experiment_id: params.id, limit: 200, offset: 0, ...source }),
		api.getExperimentStudies(params.id, source).catch(() => []),
		api.getExperimentEnsembles(params.id, source).catch(() => []),
		api.getSystemCapabilities().catch(() => ({ read_only: true, write_controls: false }))
	]);
	return { experiment, runs, roundResults, configs, runJobs, studies, ensembles, capabilities, source };
};
