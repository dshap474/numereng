import { createApi } from '$lib/api/client';
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ params, fetch, url }) => {
	const api = createApi(fetch);
	const source = {
		source_kind: url.searchParams.get('source_kind'),
		source_id: url.searchParams.get('source_id')
	};
	const [study, trials] = await Promise.all([
		api.getStudy(params.studyId, source),
		api.getStudyTrials(params.studyId, source).catch(() => [])
	]);
	return { experimentId: params.id, study, trials, source };
};
