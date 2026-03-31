import { dev } from '$app/environment';

let observersStarted = false;

function canMeasure(): boolean {
	return dev && typeof window !== 'undefined' && typeof performance !== 'undefined';
}

export function mark(name: string): void {
	if (!canMeasure()) return;
	performance.mark(name);
}

export function measure(name: string, start: string, end: string): void {
	if (!canMeasure()) return;
	try {
		performance.measure(name, start, end);
	} catch {
		// Ignore missing mark pairs in local dev instrumentation.
	}
}

export function ensureClientPerfObservers(): void {
	if (!canMeasure() || observersStarted || typeof PerformanceObserver === 'undefined') return;
	observersStarted = true;

	try {
		const measureObserver = new PerformanceObserver((list) => {
			for (const entry of list.getEntries()) {
				console.debug(`[viz-perf] measure ${entry.name}: ${entry.duration.toFixed(1)}ms`);
			}
		});
		measureObserver.observe({ entryTypes: ['measure'] });
	} catch {
		// Ignore unsupported entry types.
	}

	try {
		const longTaskObserver = new PerformanceObserver((list) => {
			for (const entry of list.getEntries()) {
				console.debug(`[viz-perf] longtask ${entry.name || 'main'}: ${entry.duration.toFixed(1)}ms`);
			}
		});
		longTaskObserver.observe({ entryTypes: ['longtask'] });
	} catch {
		// Ignore unsupported entry types.
	}
}
