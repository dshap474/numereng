import type { RunJobSample } from '$lib/api/client';

export interface MonitorSampleAverages {
	process_cpu_percent: number | null;
	process_rss_gb: number | null;
	host_cpu_percent: number | null;
	host_ram_available_gb: number | null;
	host_ram_used_gb: number | null;
	host_gpu_percent: number | null;
	host_gpu_mem_used_gb: number | null;
}

const avg = (values: Array<number | null | undefined>): number | null => {
	const usable = values.filter((value): value is number => value != null);
	if (usable.length === 0) return null;
	return usable.reduce((acc, value) => acc + value, 0) / usable.length;
};

export const processCpu = (sample: RunJobSample | null | undefined): number | null =>
	sample?.process_cpu_percent ?? sample?.cpu_percent ?? null;

export const processRssGb = (sample: RunJobSample | null | undefined): number | null =>
	sample?.process_rss_gb ?? sample?.rss_gb ?? null;

export const hostCpu = (sample: RunJobSample | null | undefined): number | null =>
	sample?.host_cpu_percent ?? null;

export const hostRamAvailableGb = (sample: RunJobSample | null | undefined): number | null =>
	sample?.host_ram_available_gb ?? sample?.ram_available_gb ?? null;

export const hostRamUsedGb = (sample: RunJobSample | null | undefined): number | null =>
	sample?.host_ram_used_gb ?? null;

export const hostGpu = (sample: RunJobSample | null | undefined): number | null =>
	sample?.host_gpu_percent ?? sample?.gpu_percent ?? null;

export const hostGpuMemUsedGb = (sample: RunJobSample | null | undefined): number | null =>
	sample?.host_gpu_mem_used_gb ?? sample?.gpu_mem_gb ?? null;

export function latestSample(samples: RunJobSample[]): RunJobSample | null {
	if (samples.length === 0) return null;
	return samples[samples.length - 1];
}

export function sampleAverages(samples: RunJobSample[], windowSize = 30): MonitorSampleAverages | null {
	const window = samples.slice(-windowSize);
	if (window.length === 0) return null;
	return {
		process_cpu_percent: avg(window.map((item) => processCpu(item))),
		process_rss_gb: avg(window.map((item) => processRssGb(item))),
		host_cpu_percent: avg(window.map((item) => hostCpu(item))),
		host_ram_available_gb: avg(window.map((item) => hostRamAvailableGb(item))),
		host_ram_used_gb: avg(window.map((item) => hostRamUsedGb(item))),
		host_gpu_percent: avg(window.map((item) => hostGpu(item))),
		host_gpu_mem_used_gb: avg(window.map((item) => hostGpuMemUsedGb(item))),
	};
}

export function scopeLabel(scope: RunJobSample['scope'] | null | undefined): string {
	switch (scope) {
		case 'launcher_process_tree':
			return 'launcher process tree';
		case 'launcher_wrapper_only':
			return 'launcher wrapper only';
		case 'launcher_host_only':
			return 'launcher host only';
		case 'unavailable':
			return 'unavailable';
		default:
			return scope ?? 'unknown';
	}
}

export function sampleStatusMessage(sample: RunJobSample | null): string | null {
	if (!sample || sample.status === 'ok') return null;
	if (sample.status === 'unavailable') return 'Telemetry unavailable for this job.';
	if (sample.scope === 'launcher_wrapper_only') {
		return 'Partial telemetry: showing launcher process metrics only.';
	}
	if (sample.scope === 'launcher_host_only') {
		return 'Partial telemetry: showing host-level metrics only.';
	}
	return 'Partial telemetry: some metrics are unavailable.';
}
