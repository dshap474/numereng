import { createApi, type SubmissionListResponse } from '$lib/api/client';
import type { PageLoad } from './$types';

function fallbackSubmissions(): SubmissionListResponse {
	return {
		items: [],
		total: 0,
		root: '.numereng/submissions'
	};
}

export const load: PageLoad = async ({ fetch }) => {
	const api = createApi(fetch);
	const submissions = await api.listSubmissions().catch(() => fallbackSubmissions());
	return { submissions };
};
