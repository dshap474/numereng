import { createApi } from '$lib/api/client';
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ params, fetch }) => {
	const api = createApi(fetch);
	const [ensemble, correlations, artifacts] = await Promise.all([
		api.getEnsemble(params.ensembleId),
		api.getEnsembleCorrelations(params.ensembleId).catch(() => ({ labels: [], matrix: [] })),
		api.getEnsembleArtifacts(params.ensembleId).catch(() => null)
	]);
	return { experimentId: params.id, ensemble, correlations, artifacts };
};
