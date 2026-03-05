import { createApi } from '$lib/api/client';
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ params, fetch }) => {
	const api = createApi(fetch);
	const [study, trials] = await Promise.all([
		api.getStudy(params.studyId),
		api.getStudyTrials(params.studyId).catch(() => [])
	]);
	return { experimentId: params.id, study, trials };
};
