<script lang="ts">
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';
	import { api, type NumeraiDocTree, type NumeraiDocNode } from '$lib/api/client';
	import { renderMarkdown } from '$lib/markdown/render';

	const HIDDEN_NOTE_STEMS = new Set(['CLAUDE', 'AGENTS']);

	let tree = $state<NumeraiDocTree | null>(null);
	let content = $state('');
	let loading = $state(true);
	let docLoading = $state(false);
	let error = $state('');
	let expandedSections = $state<Set<string>>(new Set());
	let expandedFolders = $state<Set<string>>(new Set());
	let copiedPath = $state<string | null>(null);
	let copyStatus = $state<'idle' | 'copied' | 'error'>('idle');

	let selectedPath = $derived($page.url.searchParams.get('path') || '');
	let apiFolderSections = $derived.by(() => (tree ? tree.sections.filter((s) => s.heading !== 'Notes') : []));
	let apiRootItems = $derived.by(() => (tree ? (tree.sections.find((s) => s.heading === 'Notes')?.items ?? []) : []));
	let renderSections = $derived.by(() => {
		if (apiFolderSections.length > 0) return apiFolderSections;
		// Back-compat: if the API returns a single "Notes" section, treat its top-level
		// folders as collapsible sections (so "titles" are visually distinct).
		return apiRootItems
			.filter((n) => n.children && n.children.length > 0)
			.map((n) => ({ heading: n.title, items: n.children ?? [] }));
	});
	let rootFiles = $derived.by(() => {
		if (apiFolderSections.length > 0) return apiRootItems;
		return apiRootItems.filter((n) => !n.children);
	});
	let hasSections = $derived.by(() => renderSections.length > 0);

	function isHiddenNoteFile(node: NumeraiDocNode): boolean {
		if (node.children && node.children.length > 0) return false;
		const name = node.path ? node.path.split('/').pop() ?? node.path : node.title;
		const stem = name.replace(/\.md$/i, '').toUpperCase();
		return HIDDEN_NOTE_STEMS.has(stem);
	}

	function filterNoteNodes(nodes: NumeraiDocNode[]): NumeraiDocNode[] {
		const out: NumeraiDocNode[] = [];
		for (const node of nodes) {
			if (isHiddenNoteFile(node)) continue;
			if (node.children && node.children.length > 0) {
				const children = filterNoteNodes(node.children);
				if (children.length === 0) continue;
				out.push({ ...node, children });
				continue;
			}
			out.push(node);
		}
		return out;
	}

	function filterNotesTree(t: NumeraiDocTree): NumeraiDocTree {
		return {
			sections: t.sections
				.map((s) => ({ ...s, items: filterNoteNodes(s.items) }))
				.filter((s) => s.items.length > 0)
		};
	}

	async function loadTree() {
		try {
			const raw = await api.getNotesTree();
			tree = filterNotesTree(raw);
			if (tree) {
				const nonNotesHeadings = tree.sections.map((s) => s.heading).filter((h) => h !== 'Notes');
				if (nonNotesHeadings.length > 0) {
					expandedSections = new Set(nonNotesHeadings);
				} else {
					const items = tree.sections.find((s) => s.heading === 'Notes')?.items ?? [];
					expandedSections = new Set(items.filter((n) => n.children && n.children.length > 0).map((n) => n.title));
				}
			}
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load notes tree';
		}
	}

	async function loadContent(path: string) {
		if (!path) {
			content = '';
			return;
		}
		docLoading = true;
		try {
			const res = await api.getNotesContent(path);
			if (res.exists) {
				content = res.content;
			} else {
				content = '';
			}
		} catch (e) {
			content = `*Error loading note: ${e instanceof Error ? e.message : 'unknown'}*`;
		} finally {
			docLoading = false;
		}
	}

	function selectNote(path: string) {
		goto(`/notes?path=${encodeURIComponent(path)}`, { replaceState: true, noScroll: true });
	}

	function toggleSection(heading: string) {
		const next = new Set(expandedSections);
		if (next.has(heading)) next.delete(heading);
		else next.add(heading);
		expandedSections = next;
	}

	function toggleFolder(folderKey: string) {
		const next = new Set(expandedFolders);
		if (next.has(folderKey)) next.delete(folderKey);
		else next.add(folderKey);
		expandedFolders = next;
	}

	async function copyNoteContent(path: string) {
		try {
			const res = await api.getNotesContent(path);
			if (!res.exists) {
				throw new Error('Note not found');
			}
			await navigator.clipboard.writeText(res.content);
			copiedPath = path;
			copyStatus = 'copied';
			setTimeout(() => {
				copyStatus = 'idle';
				copiedPath = null;
			}, 1500);
		} catch {
			copiedPath = path;
			copyStatus = 'error';
			setTimeout(() => {
				copyStatus = 'idle';
				copiedPath = null;
			}, 1500);
		}
	}

	function isSelected(nodePath: string | null): boolean {
		return nodePath === selectedPath;
	}

	let rendered = $derived.by(() => {
		try {
			return renderMarkdown(content, { surface: 'notes', currentPath: selectedPath });
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
	<nav class="w-64 flex-shrink-0 bg-card border-r border-border px-4 pb-4 pt-[25.6px] flex flex-col overflow-hidden">
		<div class="flex items-center justify-between pb-2 mb-2 border-b border-border/50 flex-shrink-0">
			<span class="px-2 text-xs font-semibold uppercase tracking-wider text-foreground/80">Notes</span>
		</div>

		<div class="flex-1 overflow-y-auto">
			{#if loading}
				<div class="px-2 text-muted-foreground text-sm">Loading...</div>
			{:else if tree}
				{#if renderSections.length === 0 && rootFiles.length === 0}
					<div class="px-2 text-muted-foreground text-sm">No notes yet. Create one below.</div>
				{:else}
					{#each renderSections as section}
						<div class="mb-3">
							<button
								type="button"
								class="flex items-center justify-between w-full px-2 text-left text-xs font-semibold uppercase tracking-wider text-foreground/80 hover:text-foreground transition-colors"
								onclick={() => toggleSection(section.heading)}
							>
								{section.heading}
								<svg class="w-3 h-3 flex-shrink-0 transition-transform {expandedSections.has(section.heading) ? 'rotate-90' : ''}" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
									<path stroke-linecap="round" stroke-linejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
								</svg>
							</button>
							{#if expandedSections.has(section.heading)}
								<ul class="mt-1 space-y-0.5">
									{#each section.items as item (item.path || item.title)}
										<li>
											{@render noteNode(item, 0)}
										</li>
									{/each}
								</ul>
							{/if}
						</div>
					{/each}

					{#if rootFiles.length > 0}
						<div class="{hasSections ? 'mt-3 pt-3 border-t border-border/50' : 'mt-1'}">
							<ul class="space-y-0.5">
								{#each rootFiles as item (item.path || item.title)}
									<li>
										{@render noteNode(item, 0)}
									</li>
								{/each}
							</ul>
						</div>
					{/if}
				{/if}
			{/if}
		</div>

		<div class="pt-3 border-t border-border/50 flex-shrink-0">
			<p class="text-[11px] text-amber-300 px-2">Read-only mode: notes are view-only in the dashboard.</p>
		</div>
	</nav>

	<div class="flex-1 min-w-0 overflow-y-auto px-8 pb-8 pt-[18px]">
		{#if error}
			<div class="border border-destructive/50 bg-destructive/10 rounded-lg p-4 text-sm text-destructive mb-4">
				{error}
				<button type="button" class="ml-2 underline" onclick={() => (error = '')}>dismiss</button>
			</div>
		{/if}

		{#if !selectedPath}
			<div class="text-muted-foreground text-sm">Select a note from the sidebar.</div>
		{:else if docLoading}
			<div class="text-muted-foreground text-sm">Loading...</div>
		{:else}
			<div class="flex items-center gap-2 mb-4">
				<span class="text-sm text-muted-foreground font-mono">{selectedPath}</span>
				<span class="text-[11px] text-amber-300">Read-only</span>
			</div>
			<div class="prose-dark text-sm">
				{@html rendered}
			</div>
		{/if}
	</div>
</div>

{#snippet noteNode(item: NumeraiDocNode, depth: number)}
	{#if item.children}
		<!-- Folder -->
		{@const folderKey = item.path ?? item.title}
		<div>
			<button
				type="button"
				class="flex items-center justify-between w-full text-left px-2 py-1 rounded transition-colors text-muted-foreground hover:text-foreground hover:bg-accent/50 {depth === 0 ? 'text-sm' : 'text-xs'}"
				style="padding-left: {depth * 0.75 + 0.5}rem"
				onclick={() => toggleFolder(folderKey)}
			>
				<span class="truncate">{item.title}</span>
				<svg class="w-3 h-3 flex-shrink-0 transition-transform {expandedFolders.has(folderKey) ? 'rotate-90' : ''}" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path stroke-linecap="round" stroke-linejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
				</svg>
			</button>
			{#if expandedFolders.has(folderKey) && item.children}
				<ul class="mt-0.5 space-y-0.5">
					{#each item.children as child (child.path || child.title)}
						<li>
							{@render noteNode(child, depth + 1)}
						</li>
					{/each}
				</ul>
			{/if}
		</div>
	{:else if item.path}
		<!-- File -->
		<div
			class="flex items-center w-full rounded transition-colors px-2 py-1 {isSelected(item.path)
				? 'bg-accent text-accent-foreground font-medium'
				: 'text-muted-foreground hover:text-foreground hover:bg-accent/50'} {depth === 0 ? 'text-sm' : 'text-xs'}"
			style="padding-left: {depth * 0.75 + 0.5}rem"
		>
			<button type="button" class="flex-1 text-left truncate" onclick={() => item.path && selectNote(item.path)}>
				{item.title}
			</button>
			<button
				type="button"
				class="ml-2 p-1 rounded hover:bg-accent/60 transition-colors flex-shrink-0 opacity-70 hover:opacity-100 {copiedPath === item.path && copyStatus === 'copied'
					? 'text-positive'
					: copiedPath === item.path && copyStatus === 'error'
						? 'text-negative'
				: isSelected(item.path)
					? 'text-accent-foreground/70'
					: 'text-muted-foreground/70 hover:text-foreground/70'}"
				title="Copy note content"
				aria-label="Copy note content"
				onclick={(e) => { e.preventDefault(); item.path && void copyNoteContent(item.path); }}
			>
				{#if copiedPath === item.path && copyStatus === 'copied'}
					<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
						<path stroke-linecap="round" stroke-linejoin="round" d="M4.5 12.75l6 6 9-13.5" />
					</svg>
				{:else if copiedPath === item.path && copyStatus === 'error'}
					<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
						<path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
					</svg>
				{:else}
					<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8">
						<rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
						<path stroke-linecap="round" stroke-linejoin="round" d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
					</svg>
				{/if}
			</button>
		</div>
	{/if}
{/snippet}
