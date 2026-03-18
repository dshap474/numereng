import hljs from 'highlight.js/lib/common';
import { marked, type Tokens } from 'marked';

export type MarkdownSurface = 'notes' | 'docs' | 'docs_numerai';
export interface MarkdownRenderOptions {
	surface?: MarkdownSurface | null;
	currentPath?: string | null;
}

function dirnamePosix(path: string): string {
	const idx = path.lastIndexOf('/');
	if (idx === -1) return '';
	return path.slice(0, idx);
}

function normalizeHrefRaw(href: string): string {
	const trimmed = href.trim();
	if (trimmed.startsWith('<') && trimmed.endsWith('>')) {
		return trimmed.slice(1, -1).trim();
	}
	return trimmed;
}

function splitHref(href: string): { path: string; query: string; fragment: string } {
	const hashIdx = href.indexOf('#');
	const withoutFragment = hashIdx === -1 ? href : href.slice(0, hashIdx);
	const fragment = hashIdx === -1 ? '' : href.slice(hashIdx);
	const queryIdx = withoutFragment.indexOf('?');
	if (queryIdx === -1) {
		return { path: withoutFragment, query: '', fragment };
	}
	return {
		path: withoutFragment.slice(0, queryIdx),
		query: withoutFragment.slice(queryIdx),
		fragment
	};
}

function resolveWithinRoot(currentPath: string, hrefPath: string): { resolved: string; escaped: boolean } {
	const baseDir = dirnamePosix(currentPath);
	const rootRelative = hrefPath.startsWith('/') && !hrefPath.startsWith('//');
	const raw = rootRelative ? hrefPath.slice(1) : baseDir ? `${baseDir}/${hrefPath}` : hrefPath;

	const stack: string[] = [];
	let escaped = false;

	for (const seg of raw.split('/')) {
		if (!seg || seg === '.') continue;
		if (seg === '..') {
			if (stack.length === 0) escaped = true;
			else stack.pop();
			continue;
		}
		stack.push(seg);
	}

	return { resolved: stack.join('/'), escaped };
}

function isExternalHref(href: string): boolean {
	const lower = href.toLowerCase();
	return (
		lower.startsWith('http://') ||
		lower.startsWith('https://') ||
		lower.startsWith('mailto:') ||
		lower.startsWith('tel:') ||
		lower.startsWith('data:') ||
		lower.startsWith('blob:') ||
		lower.startsWith('javascript:') ||
		href.startsWith('//')
	);
}

function isAlreadyRoutedHref(href: string): boolean {
	return (
		href.startsWith('/notes?path=') ||
		href.startsWith('/docs?path=') ||
		href.startsWith('/docs/numerai?path=') ||
		href.startsWith('/api/docs/numereng/asset?path=') ||
		href.startsWith('/api/docs/numerai/asset?path=')
	);
}

function surfaceDocHref(surface: MarkdownSurface, path: string, fragment: string): string {
	const encoded = encodeURIComponent(path);
	switch (surface) {
		case 'notes':
			return `/notes?path=${encoded}${fragment}`;
		case 'docs':
			return `/docs?path=${encoded}${fragment}`;
		case 'docs_numerai':
			return `/docs/numerai?path=${encoded}${fragment}`;
	}
}

function surfaceAssetHref(surface: MarkdownSurface, path: string, fragment: string): string | null {
	const encoded = encodeURIComponent(path);
	switch (surface) {
		case 'docs':
			return `/api/docs/numereng/asset?path=${encoded}${fragment}`;
		case 'docs_numerai':
			return `/api/docs/numerai/asset?path=${encoded}${fragment}`;
		case 'notes':
			return null;
	}
}

function rewriteHref(href: string, surface: MarkdownSurface, currentPath: string): string | null {
	const normalizedHref = normalizeHrefRaw(href);
	if (!normalizedHref) return null;
	if (normalizedHref.startsWith('#')) return null;
	if (isExternalHref(normalizedHref)) return null;
	if (isAlreadyRoutedHref(normalizedHref)) return null;

	const { path: hrefPath, query: _query, fragment } = splitHref(normalizedHref);
	if (!hrefPath) return null;

	const { resolved, escaped } = resolveWithinRoot(currentPath, hrefPath);
	if (escaped || !resolved || resolved.includes('\x00')) return null;

	if (resolved.toLowerCase().endsWith('.md')) {
		return surfaceDocHref(surface, resolved, fragment);
	}

	// For docs surfaces, rewrite local assets/files to API asset endpoint.
	if (surface === 'docs' || surface === 'docs_numerai') {
		return surfaceAssetHref(surface, resolved, fragment);
	}

	return null;
}

function rewriteHtmlAttributes(
	html: string,
	opts: { surface: MarkdownSurface; currentPath: string }
): string {
	if (opts.surface === 'notes') return html;

	return html.replace(/\b(href|src)\s*=\s*("([^"]*)"|'([^']*)')/gi, (match, attr, _quoted, dq, sq) => {
		const original = (dq ?? sq ?? '').trim();
		const nextHref = rewriteHref(original, opts.surface, opts.currentPath);
		if (!nextHref) return match;
		const quote = dq !== undefined ? '"' : "'";
		return `${attr}=${quote}${nextHref}${quote}`;
	});
}

function escapeHtml(text: string): string {
	return text
		.replaceAll('&', '&amp;')
		.replaceAll('<', '&lt;')
		.replaceAll('>', '&gt;')
		.replaceAll('"', '&quot;')
		.replaceAll("'", '&#39;');
}

function decodeHtmlEntities(text: string): string {
	return text.replaceAll(/&(?:#(\d+)|#x([0-9a-f]+)|([a-z]+));/gi, (entity, dec, hex, named) => {
		if (dec) {
			const codePoint = Number.parseInt(dec, 10);
			return Number.isFinite(codePoint) ? String.fromCodePoint(codePoint) : entity;
		}
		if (hex) {
			const codePoint = Number.parseInt(hex, 16);
			return Number.isFinite(codePoint) ? String.fromCodePoint(codePoint) : entity;
		}

		switch ((named ?? '').toLowerCase()) {
			case 'amp':
				return '&';
			case 'lt':
				return '<';
			case 'gt':
				return '>';
			case 'quot':
				return '"';
			case 'apos':
			case '#39':
				return "'";
			case 'nbsp':
				return '\u00a0';
			default:
				return entity;
		}
	});
}

function extractClassNames(attrs: string): string[] {
	const match = attrs.match(/\bclass\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/i);
	const rawClasses = match?.[1] ?? match?.[2] ?? match?.[3] ?? '';
	return rawClasses
		.split(/\s+/)
		.map((value) => value.trim())
		.filter(Boolean);
}

function detectCodeLanguage(attrs: string[]): string | undefined {
	for (const value of attrs) {
		const match = value.match(/^(?:language|lang)-([a-z0-9_+#.-]+)$/i);
		if (match?.[1]) return match[1];
	}
	return undefined;
}

function normalizeRawHtmlCode(code: string): string {
	return decodeHtmlEntities(code.replaceAll(/<br\s*\/?>/gi, '\n').replaceAll(/<\/?[^>]+>/g, ''));
}

function extractRawHtmlCodeBlock(html: string): { code: string; language?: string } | null {
	const match = html.match(/^\s*<pre\b([^>]*)>\s*<code\b([^>]*)>([\s\S]*?)<\/code>\s*<\/pre>\s*$/i);
	if (!match) return null;

	const [, preAttrs, codeAttrs, innerCode] = match;
	const language =
		detectCodeLanguage(extractClassNames(codeAttrs)) ?? detectCodeLanguage(extractClassNames(preAttrs));

	return {
		code: normalizeRawHtmlCode(innerCode),
		language
	};
}

function classifyCodespan(text: string): string[] {
	const normalized = text.trim().toLowerCase();
	if (!normalized) return ['md-inline-code'];

	const collapsedMetric = normalized.replaceAll('.', '_');
	if (
		/^(bmc|corr|fnc|mmc|cwmm|feature_exposure|max_feature_exposure|max_drawdown)(?:_[a-z0-9]+)*$/.test(
			collapsedMetric
		)
	) {
		return ['md-inline-code', 'md-token-metric'];
	}
	if (normalized === '20d') {
		return ['md-inline-code', 'md-token-horizon-20'];
	}
	if (normalized === '60d') {
		return ['md-inline-code', 'md-token-horizon-60'];
	}
	if (/^[0-9a-f]{12}$/i.test(normalized)) {
		return ['md-inline-code', 'md-token-run-id'];
	}
	if (/^(?:target_)?[a-z][a-z0-9]*(?:_[a-z0-9]+)*_(?:20|60)$/.test(normalized)) {
		return ['md-inline-code', 'md-token-target'];
	}

	return ['md-inline-code'];
}

function highlightCodeBlock(code: string, lang: string | undefined): { html: string; languageClass: string } {
	const normalizedLang = lang?.trim().toLowerCase().split(/\s+/, 1)[0] ?? '';
	if (normalizedLang && hljs.getLanguage(normalizedLang)) {
		return {
			html: hljs.highlight(code, { language: normalizedLang, ignoreIllegals: true }).value,
			languageClass: `language-${normalizedLang}`
		};
	}

	const auto = hljs.highlightAuto(code);
	return {
		html: auto.value,
		languageClass: auto.language ? `language-${auto.language}` : ''
	};
}

function renderHighlightedCodeBlock(code: string, lang: string | undefined): string {
	const highlighted = highlightCodeBlock(code, lang);
	const classes = ['hljs', highlighted.languageClass].filter(Boolean).join(' ');
	return `<pre class="md-code-block"><code class="${classes}">${highlighted.html}</code></pre>`;
}

function createRenderer(opts: MarkdownRenderOptions) {
	const renderer = new marked.Renderer();
	const baseLink = renderer.link.bind(renderer);
	const baseHtml = renderer.html.bind(renderer);
	const baseTable = renderer.table.bind(renderer);
	const currentPath = opts.currentPath ?? '';

	renderer.link = (token: Tokens.Link): string => {
		if (!opts.surface) return baseLink(token);
		const nextHref = rewriteHref(token.href, opts.surface, currentPath);
		if (!nextHref) return baseLink(token);
		return baseLink({ ...token, href: nextHref });
	};

	renderer.codespan = ({ text }: Tokens.Codespan): string => {
		const classes = classifyCodespan(text).join(' ');
		return `<code class="${classes}">${escapeHtml(text)}</code>`;
	};

	renderer.code = ({ text, lang }: Tokens.Code): string => {
		return renderHighlightedCodeBlock(text, lang);
	};

	renderer.html = (token: Tokens.HTML | Tokens.Tag): string => {
		const rawCodeBlock = extractRawHtmlCodeBlock(token.text);
		if (!rawCodeBlock) return baseHtml(token);
		return renderHighlightedCodeBlock(rawCodeBlock.code, rawCodeBlock.language);
	};

	renderer.table = (token: Tokens.Table): string => {
		return `<div class="md-table-wrap">${baseTable(token).replace('<table>', '<table class="md-table">')}</div>`;
	};

	return renderer;
}

export function renderMarkdown(content: string, opts: MarkdownRenderOptions = {}): string {
	const resolved = { surface: opts.surface ?? null, currentPath: opts.currentPath ?? '' };
	const rendered = marked.parse(content, { renderer: createRenderer(resolved) }) as string;
	if (!resolved.surface) return rendered;
	return rewriteHtmlAttributes(rendered, { surface: resolved.surface, currentPath: resolved.currentPath });
}

export function renderMarkdownInline(content: string, opts: MarkdownRenderOptions = {}): string {
	const resolved = { surface: opts.surface ?? null, currentPath: opts.currentPath ?? '' };
	const rendered = marked.parseInline(content, { renderer: createRenderer(resolved) }) as string;
	if (!resolved.surface) return rendered;
	return rewriteHtmlAttributes(rendered, { surface: resolved.surface, currentPath: resolved.currentPath });
}
