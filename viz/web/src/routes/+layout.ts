import { createApi } from '$lib/api/client';
import type { LayoutLoad } from './$types';

export const load: LayoutLoad = async ({ fetch }) => {
	const api = createApi(fetch);
	const [experiments, capabilities] = await Promise.all([
		api.listExperiments().catch(() => []),
		api.getSystemCapabilities().catch(() => ({ read_only: true, write_controls: false }))
	]);
	return { experiments, capabilities };
};
