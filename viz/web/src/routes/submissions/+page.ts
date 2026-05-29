import {
	createApi,
	type SubmissionCalibrationResponse,
	type SubmissionListResponse
} from '$lib/api/client';
import type { PageLoad } from './$types';

function fallbackSubmissions(): SubmissionListResponse {
	return {
		items: [],
		total: 0,
		root: '.numereng/submissions'
	};
}

function fallbackCalibration(): SubmissionCalibrationResponse {
	return {
		rows: [],
		total: 0,
		root: '.numereng/analysis/live_calibration',
		report: {},
		manifest: {}
	};
}

export const load: PageLoad = async ({ fetch }) => {
	const api = createApi(fetch);
	const [submissions, calibration] = await Promise.all([
		api.listSubmissions().catch(() => fallbackSubmissions()),
		api.getSubmissionCalibration().catch(() => fallbackCalibration())
	]);
	return { submissions, calibration };
};
