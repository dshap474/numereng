import { marked, type Tokens } from 'marked';

export type MarkdownSurface = 'notes' | 'docs' | 'docs_numerai';

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

export function renderMarkdown(content: string, opts: { surface: MarkdownSurface; currentPath: string }): string {
	const renderer = new marked.Renderer();
	const baseLink = renderer.link.bind(renderer);

	renderer.link = (token: Tokens.Link): string => {
		const nextHref = rewriteHref(token.href, opts.surface, opts.currentPath);
		if (!nextHref) return baseLink(token);
		return baseLink({ ...token, href: nextHref });
	};

	const rendered = marked.parse(content, { renderer }) as string;
	return rewriteHtmlAttributes(rendered, opts);
}
