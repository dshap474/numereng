import { createApi } from '$lib/api/client';
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ params, fetch, url }) => {
	const api = createApi(fetch);
	const source = {
		source_kind: url.searchParams.get('source_kind'),
		source_id: url.searchParams.get('source_id')
	};
	const [ensemble, correlations, artifacts] = await Promise.all([
		api.getEnsemble(params.ensembleId, source),
		api.getEnsembleCorrelations(params.ensembleId, source).catch(() => ({ labels: [], matrix: [] })),
		api.getEnsembleArtifacts(params.ensembleId, source).catch(() => null)
	]);
	return { experimentId: params.id, ensemble, correlations, artifacts, source };
};
