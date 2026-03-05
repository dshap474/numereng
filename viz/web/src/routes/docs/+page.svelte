<script lang="ts">
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';
	import { api, type NumeraiDocTree, type NumeraiDocNode } from '$lib/api/client';
	import { renderMarkdown } from '$lib/markdown/render';

	let tree = $state<NumeraiDocTree | null>(null);
	let content = $state('');
	let loading = $state(true);
	let docLoading = $state(false);
	let error = $state('');
	let docNotice = $state('');
	let expandedSections = $state<Set<string>>(new Set());
	const DEFAULT_DOC_PATH = 'README.md';

	let selectedPath = $derived($page.url.searchParams.get('path') || DEFAULT_DOC_PATH);

	async function loadTree() {
		try {
			tree = await api.getNumerengDocTree();
			if (tree) {
				expandedSections = new Set(tree.sections.map((s) => s.heading));
			}
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load doc tree';
		}
	}

	async function loadContent(path: string) {
		docLoading = true;
		const requestedPath = (path || DEFAULT_DOC_PATH).trim() || DEFAULT_DOC_PATH;
		try {
			const res = await api.getNumerengDocContent(requestedPath);
			if (res.exists) {
				content = res.content;
			} else {
				if (requestedPath !== DEFAULT_DOC_PATH) {
					const fallback = await api.getNumerengDocContent(DEFAULT_DOC_PATH);
					if (fallback.exists) {
						content = fallback.content;
						docNotice = `Document not found: ${requestedPath}. Showing ${DEFAULT_DOC_PATH} instead.`;
						goto(`/docs?path=${encodeURIComponent(DEFAULT_DOC_PATH)}`, {
							replaceState: true,
							noScroll: true
						});
						return;
					}
				}
				content = `*Document not found: ${requestedPath}*`;
			}
		} catch (e) {
			content = `*Error loading document: ${e instanceof Error ? e.message : 'unknown'}*`;
		} finally {
			docLoading = false;
		}
	}

	function selectDoc(path: string) {
		docNotice = '';
		goto(`/docs?path=${encodeURIComponent(path)}`, { replaceState: true, noScroll: true });
	}

	function toggleSection(heading: string) {
		const next = new Set(expandedSections);
		if (next.has(heading)) {
			next.delete(heading);
		} else {
			next.add(heading);
		}
		expandedSections = next;
	}

	function isSelected(nodePath: string | null): boolean {
		return nodePath === selectedPath;
	}

	let rendered = $derived.by(() => {
		try {
			return renderMarkdown(content, { surface: 'docs', currentPath: selectedPath });
		} catch {
			return '<p>Failed to parse markdown.</p>';
		}
	});

	$effect(() => {
		loadTree().then(() => {
			loading = false;
		});
	});

	$effect(() => {
		if (selectedPath) {
			loadContent(selectedPath);
		}
	});
</script>

<div class="-m-8 -mt-14 md:-mt-8 flex h-screen">
	<nav class="w-64 flex-shrink-0 overflow-y-auto bg-card border-r border-border px-4 pb-4 pt-[25.6px]">
		{#if loading}
			<div class="px-2 text-muted-foreground text-sm">Loading...</div>
		{:else if tree}
			{#each tree.sections as section}
				<div class="mb-3">
					<button
						type="button"
						class="flex items-center justify-between w-full px-2 text-left text-xs font-semibold uppercase tracking-wider text-foreground/70 hover:text-foreground transition-colors"
						onclick={() => toggleSection(section.heading)}
					>
						{section.heading}
						<svg class="w-3 h-3 flex-shrink-0 transition-transform {expandedSections.has(section.heading) ? 'rotate-90' : ''}" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
							<path stroke-linecap="round" stroke-linejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
						</svg>
					</button>
					{#if expandedSections.has(section.heading)}
						<ul class="mt-1 space-y-0.5">
							{#each section.items as item}
								<li>
									{#if item.path}
										<button
											type="button"
											class="block w-full text-left px-2 py-1 text-sm rounded transition-colors {isSelected(item.path)
												? 'bg-accent text-accent-foreground font-medium'
												: 'text-muted-foreground hover:text-foreground hover:bg-accent/50'}"
											onclick={() => item.path && selectDoc(item.path)}
										>
											{item.title}
										</button>
									{:else}
										<span class="block px-2 py-1 text-sm text-muted-foreground">{item.title}</span>
									{/if}
									{#if item.children}
										<ul class="ml-3 space-y-0.5">
											{#each item.children as child}
												<li>
													{#if child.path}
														<button
															type="button"
															class="block w-full text-left px-2 py-1 text-xs rounded transition-colors {isSelected(child.path)
																? 'bg-accent text-accent-foreground font-medium'
																: 'text-muted-foreground hover:text-foreground hover:bg-accent/50'}"
															onclick={() => child.path && selectDoc(child.path)}
														>
															{child.title}
														</button>
													{:else}
														<span class="block px-2 py-1 text-xs text-muted-foreground">{child.title}</span>
													{/if}
												</li>
											{/each}
										</ul>
									{/if}
								</li>
							{/each}
						</ul>
					{/if}
				</div>
			{/each}
		{/if}
	</nav>

	<div class="flex-1 min-w-0 overflow-y-auto px-8 pb-8 pt-[18px]">
		{#if error}
			<div class="border border-destructive/50 bg-destructive/10 rounded-lg p-4 text-sm text-destructive">
				{error}
				<p class="mt-2 text-muted-foreground">Numereng guide docs not found at docs/numereng/</p>
			</div>
		{:else if docLoading}
			<div class="text-muted-foreground text-sm">Loading document...</div>
		{:else}
			{#if docNotice}
				<div class="mb-3 rounded-md border border-amber-400/25 bg-amber-400/10 px-3 py-2 text-xs text-amber-300">
					{docNotice}
				</div>
			{/if}
			<div class="prose-dark text-sm">
				{@html rendered}
			</div>
		{/if}
	</div>
</div>
