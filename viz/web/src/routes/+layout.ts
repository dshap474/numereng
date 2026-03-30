import { createApi, type ExperimentOverviewItem } from '$lib/api/client';
import type { LayoutLoad } from './$types';

export const ssr = false;

function fallbackExperiments(items: Array<{ experiment_id: string; name: string; status: string; created_at: string; updated_at: string; run_count?: number; runs?: string[]; tags?: string[] }>): ExperimentOverviewItem[] {
	return items.map((item) => ({
		experiment_id: item.experiment_id,
		name: item.name,
		status: item.status,
		created_at: item.created_at,
		updated_at: item.updated_at,
		run_count: item.run_count ?? (Array.isArray(item.runs) ? item.runs.length : 0),
		tags: item.tags ?? [],
		has_live: false,
		live_run_count: 0,
		attention_state: 'none',
		latest_activity_at: item.updated_at ?? item.created_at ?? null,
		source_kind: 'local',
		source_id: 'local',
		source_label: 'Local store',
		detail_href: `/experiments/${item.experiment_id}`
	}));
}

export const load: LayoutLoad = async ({ fetch }) => {
	const api = createApi(fetch);
	const [overview, localExperiments, capabilities] = await Promise.all([
		api.getExperimentsOverview().catch(() => null),
		api.listExperiments().catch(() => []),
		api.getSystemCapabilities().catch(() => ({ read_only: true, write_controls: false }))
	]);
	const experiments =
		overview && Array.isArray(overview.experiments)
			? overview.experiments
			: fallbackExperiments(localExperiments);
	return { experiments, experimentsOverview: overview, capabilities };
};
