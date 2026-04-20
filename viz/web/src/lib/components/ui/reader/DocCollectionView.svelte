<script lang="ts">
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';
	import { api, type NumeraiDocTree, type NumeraiDocNode } from '$lib/api/client';
	import { renderMarkdown } from '$lib/markdown/render';
	import ReaderSidebar from './ReaderSidebar.svelte';
	import ReaderWorkspace from './ReaderWorkspace.svelte';

	type DocVariant = 'numereng' | 'numerai';

	let { variant } = $props<{ variant: DocVariant }>();

	let tree = $state<NumeraiDocTree | null>(null);
	let content = $state('');
	let loading = $state(true);
	let docLoading = $state(false);
	let error = $state('');
	let docNotice = $state('');
	let expandedSections = $state<Set<string>>(new Set());
	let expandedForumYears = $state<Set<string>>(new Set());

	const DEFAULT_DOC_PATH = 'README.md';

	const variantConfig = $derived.by(() => {
		if (variant === 'numerai') {
			return {
				baseHref: '/docs/numerai',
				label: 'Numerai',
				description: 'Official reference docs and forum archive.',
				emptyHint:
					'Run `uv run numereng docs sync numerai` to download the official docs into this workspace.'
			};
		}
		return {
			baseHref: '/docs',
			label: 'Numereng',
			description: 'Project guides, workflows, and local docs.',
			emptyHint: 'Numereng guide docs not found at docs/numereng/.'
		};
	});

	let selectedPath = $derived($page.url.searchParams.get('path') || DEFAULT_DOC_PATH);

	async function loadTree() {
		try {
			tree = variant === 'numerai' ? await api.getNumeraiDocTree() : await api.getNumerengDocTree();
			if (tree) {
				expandedSections = new Set(tree.sections.map((s) => s.heading));
				expandedForumYears = new Set();
			}
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load doc tree';
		}
	}

	async function loadContent(path: string) {
		docLoading = true;
		const requestedPath = (path || DEFAULT_DOC_PATH).trim() || DEFAULT_DOC_PATH;

		try {
			const res =
				variant === 'numerai'
					? await api.getNumeraiDocContent(requestedPath)
					: await api.getNumerengDocContent(requestedPath);

			if (res.exists) {
				content = res.content;
				docNotice = '';
				return;
			}

			if (variant === 'numerai' && res.missing_reason === 'docs_not_downloaded') {
				content =
					'*Numerai docs are not preinstalled in this workspace.*\n\nRun `uv run numereng docs sync numerai` to download the official docs locally.';
				docNotice = '';
				return;
			}

			if (requestedPath !== DEFAULT_DOC_PATH) {
				const fallback =
					variant === 'numerai'
						? await api.getNumeraiDocContent(DEFAULT_DOC_PATH)
						: await api.getNumerengDocContent(DEFAULT_DOC_PATH);
				if (fallback.exists) {
					content = fallback.content;
					docNotice = `Document not found: ${requestedPath}. Showing ${DEFAULT_DOC_PATH} instead.`;
					goto(`${variantConfig.baseHref}?path=${encodeURIComponent(DEFAULT_DOC_PATH)}`, {
						replaceState: true,
						noScroll: true
					});
					return;
				}
			}

			content = `*Document not found: ${requestedPath}*`;
		} catch (e) {
			content = `*Error loading document: ${e instanceof Error ? e.message : 'unknown'}*`;
		} finally {
			docLoading = false;
		}
	}

	function selectDoc(path: string) {
		docNotice = '';
		goto(`${variantConfig.baseHref}?path=${encodeURIComponent(path)}`, {
			replaceState: true,
			noScroll: true
		});
	}

	function toggleSection(heading: string) {
		const next = new Set(expandedSections);
		if (next.has(heading)) next.delete(heading);
		else next.add(heading);
		expandedSections = next;
	}

	function toggleForumYear(year: string) {
		const next = new Set(expandedForumYears);
		if (next.has(year)) next.delete(year);
		else next.add(year);
		expandedForumYears = next;
	}

	function isSelected(nodePath: string | null): boolean {
		return nodePath === selectedPath;
	}

	function findNodeTitle(nodes: NumeraiDocNode[], path: string): string | null {
		for (const node of nodes) {
			if (node.path === path) return node.title;
			if (node.children) {
				const childTitle = findNodeTitle(node.children, path);
				if (childTitle) return childTitle;
			}
		}
		return null;
	}

	let selectedTitle = $derived.by(() => {
		if (!tree) return selectedPath.split('/').pop()?.replace(/\.md$/i, '') || variantConfig.label;
		for (const section of tree.sections) {
			const found = findNodeTitle(section.items, selectedPath);
			if (found) return found;
		}
		return selectedPath.split('/').pop()?.replace(/\.md$/i, '') || variantConfig.label;
	});

	let rendered = $derived.by(() => {
		try {
			return renderMarkdown(content, {
				surface: variant === 'numerai' ? 'docs_numerai' : 'docs',
				currentPath: selectedPath
			});
		} catch {
			return '<p>Failed to parse markdown.</p>';
		}
	});

	$effect(() => {
		void variant;
		loadTree().then(() => {
			loading = false;
		});
	});

	$effect(() => {
		void variant;
		if (selectedPath) {
			loadContent(selectedPath);
		}
	});
</script>

<ReaderWorkspace>
	{#snippet sidebar()}
		<ReaderSidebar
			kicker="Docs"
			title={variantConfig.label}
			description={variantConfig.description}
		>
			{#if loading}
				<div class="px-2 py-3 text-sm text-muted-foreground">Loading collection…</div>
			{:else if tree}
				{#each tree.sections as section}
					{@const isArchive = section.heading === 'Forum Archive'}
					<section class="reader-section-block">
						<button
							type="button"
							class={`reader-section-label ${isArchive ? 'reader-section-label-archive' : ''}`}
							onclick={() => toggleSection(section.heading)}
						>
							<span>{section.heading}</span>
							<svg
								class={`reader-chevron ${expandedSections.has(section.heading) ? 'rotate-90' : ''}`}
								fill="none"
								viewBox="0 0 24 24"
								stroke="currentColor"
								stroke-width="2"
							>
								<path stroke-linecap="round" stroke-linejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
							</svg>
						</button>

						{#if expandedSections.has(section.heading)}
							<ul class="mt-2 space-y-1">
								{#each section.items as item}
									<li>
										{#if isArchive && item.children}
											<button
												type="button"
												class="reader-archive-year"
												onclick={() => toggleForumYear(item.title)}
											>
												<span>{item.title}</span>
												<svg
													class={`reader-chevron ${expandedForumYears.has(item.title) ? 'rotate-90' : ''}`}
													fill="none"
													viewBox="0 0 24 24"
													stroke="currentColor"
													stroke-width="2"
												>
													<path stroke-linecap="round" stroke-linejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
												</svg>
											</button>
										{:else if item.path}
											<button
												type="button"
												class={`reader-nav-row ${isSelected(item.path) ? 'reader-nav-row-active' : ''}`}
												onclick={() => item.path && selectDoc(item.path)}
											>
												<span class="reader-nav-title">{item.title}</span>
											</button>
										{:else}
											<div class="reader-nav-label">{item.title}</div>
										{/if}

										{#if item.children && (!isArchive || expandedForumYears.has(item.title))}
											<ul class="mt-1 space-y-1 pl-3">
												{#each item.children as child}
													<li>
														{#if child.path}
															<button
																type="button"
																class={`reader-nav-row reader-nav-row-child ${isSelected(child.path) ? 'reader-nav-row-active' : ''}`}
																onclick={() => child.path && selectDoc(child.path)}
															>
																<span class="reader-nav-title">{child.title}</span>
															</button>
														{:else}
															<div class="reader-nav-label reader-nav-label-child">{child.title}</div>
														{/if}
													</li>
												{/each}
											</ul>
										{/if}
									</li>
								{/each}
							</ul>
						{/if}
					</section>
				{/each}
			{/if}
		</ReaderSidebar>
	{/snippet}

	<div class="reader-content-header">
		<div>
			<div class="reader-content-kicker">Docs workspace</div>
			<h1 class="reader-content-title">{selectedTitle}</h1>
			<div class="reader-content-meta">{selectedPath}</div>
		</div>

		<div class="pill-tabs" aria-label="Docs collection">
			<a
				href="/docs"
				class={`pill-tab ${variant === 'numereng' ? 'pill-tab-active' : ''}`}
			>
				Numereng
			</a>
			<a
				href="/docs/numerai"
				class={`pill-tab ${variant === 'numerai' ? 'pill-tab-active' : ''}`}
			>
				Numerai
			</a>
		</div>
	</div>

	{#if error}
		<div class="reader-alert reader-alert-danger">
			{error}
			<p class="mt-2 text-muted-foreground">{variantConfig.emptyHint}</p>
		</div>
	{:else}
		{#if docNotice}
			<div class="reader-alert reader-alert-warning">{docNotice}</div>
		{/if}
		{#if docLoading && !content}
			<div class="py-6 text-sm text-muted-foreground">Loading document…</div>
		{:else}
			<div class="prose-dark reader-prose text-sm" class:reader-prose-loading={docLoading}>
				{@html rendered}
			</div>
		{/if}
	{/if}
</ReaderWorkspace>
