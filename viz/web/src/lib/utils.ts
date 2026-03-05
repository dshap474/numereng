import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
	return twMerge(clsx(inputs));
}

export function fmt(value: number | undefined | null): string {
	if (value == null) return 'N/A';
	return value.toFixed(4);
}

export function fmtPercent(value: number | null | undefined): string {
	if (value == null) return '—';
	return `${value.toFixed(1)}%`;
}

export function fmtGb(value: number | null | undefined): string {
	if (value == null) return '—';
	return `${value.toFixed(2)} GB`;
}

export function badgeClass(status: string): string {
	switch (status) {
		case 'draft':
			return 'bg-champion/15 text-champion';
		case 'active':
			return 'bg-blue-500/15 text-blue-400';
		case 'complete':
			return 'bg-positive/15 text-positive';
		default:
			return 'bg-muted text-muted-foreground';
	}
}
