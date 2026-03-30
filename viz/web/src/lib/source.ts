export interface SourceContext {
	source_kind?: string | null;
	source_id?: string | null;
	source_label?: string | null;
}

export function normalizedSource(source?: SourceContext): { source_kind: string; source_id: string } {
	return {
		source_kind: source?.source_kind?.trim() || 'local',
		source_id: source?.source_id?.trim() || 'local'
	};
}

export function isLocalSource(source?: SourceContext): boolean {
	const normalized = normalizedSource(source);
	return normalized.source_kind === 'local' && normalized.source_id === 'local';
}

export function sourceQueryParams(source?: SourceContext): Record<string, string> {
	if (isLocalSource(source)) {
		return {};
	}
	const normalized = normalizedSource(source);
	return {
		source_kind: normalized.source_kind,
		source_id: normalized.source_id
	};
}

export function withSourceHref(path: string, source?: SourceContext): string {
	const params = new URLSearchParams(sourceQueryParams(source));
	const query = params.toString();
	return query ? `${path}?${query}` : path;
}
