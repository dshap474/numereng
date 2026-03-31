import { createApi } from '$lib/api/client';
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ params, fetch, url }) => {
	const api = createApi(fetch);
	const source = {
		source_kind: url.searchParams.get('source_kind'),
		source_id: url.searchParams.get('source_id')
	};
	const [experiment, capabilities] = await Promise.all([
		api.getExperiment(params.id, source),
		api.getSystemCapabilities().catch(() => ({ read_only: true, write_controls: false }))
	]);

	return {
		runId: params.runId,
		experimentId: params.id,
		experiment,
		capabilities,
		source
	};
};
