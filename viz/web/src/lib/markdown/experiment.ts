export interface ExperimentDocHeaderEntry {
	label: string;
	value: string;
}

export interface ExperimentDocParseResult {
	title: string | null;
	headerEntries: ExperimentDocHeaderEntry[];
	body: string;
}

export interface ExperimentDocStats {
	totalRuns: number;
	completedRuns: number;
	roundCount: number;
	studyCount: number;
	ensembleCount: number;
}

export interface ExperimentDocContext {
	experimentId: string;
	name: string;
	status: string | null;
	createdAt: string | null;
	updatedAt: string | null;
	championRunId: string | null;
	tags: string[];
	stats: ExperimentDocStats;
}

const HEADER_LINE_RE = /^\*\*([^*]+)\*\*:\s*(.+?)\s*$/;

export function extractExperimentDocHeader(content: string): ExperimentDocParseResult {
	const normalized = content.replaceAll('\r\n', '\n');
	const lines = normalized.split('\n');
	let cursor = 0;

	while (cursor < lines.length && lines[cursor].trim().length === 0) cursor += 1;

	let title: string | null = null;
	if (cursor < lines.length) {
		const match = lines[cursor].match(/^#\s+(.+?)\s*$/);
		if (match) {
			title = match[1] ?? null;
			cursor += 1;
		}
	}

	while (cursor < lines.length && lines[cursor].trim().length === 0) cursor += 1;

	const headerEntries: ExperimentDocHeaderEntry[] = [];
	while (cursor < lines.length) {
		const line = lines[cursor].trim();
		if (!line) break;
		const match = line.match(HEADER_LINE_RE);
		if (!match) break;
		headerEntries.push({ label: match[1].trim(), value: match[2].trim() });
		cursor += 1;
	}

	if (title !== null || headerEntries.length > 0) {
		while (cursor < lines.length && lines[cursor].trim().length === 0) cursor += 1;
	}

	return {
		title,
		headerEntries,
		body: lines.slice(cursor).join('\n').trimStart()
	};
}

export function normalizedHeaderLabel(label: string): string {
	return label.trim().toLowerCase().replaceAll(/\s+/g, ' ');
}

export function splitTagList(raw: string): string[] {
	return raw
		.split(',')
		.map((item) => item.trim())
		.map((item) => item.replace(/^`(.+)`$/, '$1').trim())
		.filter(Boolean);
}
