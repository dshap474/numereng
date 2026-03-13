import { createApi } from '$lib/api/client';
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ params, fetch }) => {
	const api = createApi(fetch);
	const [experiment, runs, capabilities] = await Promise.all([
		api.getExperiment(params.id),
		api.getExperimentRuns(params.id),
		api.getSystemCapabilities().catch(() => ({ read_only: true, write_controls: false }))
	]);

	return {
		runId: params.runId,
		experimentId: params.id,
		experiment,
		runs,
		capabilities
	};
};
