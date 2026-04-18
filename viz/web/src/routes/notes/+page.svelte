<script lang="ts">
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';
	import { api, type NumeraiDocTree, type NumeraiDocNode } from '$lib/api/client';
	import NoteTreeItem from '$lib/components/ui/reader/NoteTreeItem.svelte';
	import ReaderSidebar from '$lib/components/ui/reader/ReaderSidebar.svelte';
	import ReaderWorkspace from '$lib/components/ui/reader/ReaderWorkspace.svelte';
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
	let selectedTitle = $derived.by(() =>
		selectedPath ? selectedPath.split('/').pop()?.replace(/\.md$/i, '') || 'Selected note' : 'Notes'
	);
	let selectedBreadcrumb = $derived.by(() =>
		selectedPath ? selectedPath.replace(/\.md$/i, '').split('/').join(' / ') : 'Choose a note from the collection.'
	);

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

<ReaderWorkspace>
	{#snippet sidebar()}
		<ReaderSidebar
			kicker="Notes"
			title="Workspace notes"
			description="Research memory, notes, and local collections."
		>
			{#if loading}
				<div class="px-2 py-3 text-sm text-muted-foreground">Loading notes…</div>
			{:else if tree}
				{#if renderSections.length === 0 && rootFiles.length === 0}
					<div class="px-2 py-3 text-sm text-muted-foreground">No notes yet. Create one below.</div>
				{:else}
					{#each renderSections as section}
						<section class="reader-section-block">
							<button
								type="button"
								class="reader-section-label"
								onclick={() => toggleSection(section.heading)}
							>
								<span>{section.heading}</span>
								<svg class="reader-chevron {expandedSections.has(section.heading) ? 'rotate-90' : ''}" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
									<path stroke-linecap="round" stroke-linejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
								</svg>
							</button>

							{#if expandedSections.has(section.heading)}
								<ul class="mt-2 space-y-1">
									{#each section.items as item (item.path || item.title)}
										<li>
											<NoteTreeItem
												{item}
												depth={0}
												{selectedPath}
												{expandedFolders}
												onToggleFolder={toggleFolder}
												onSelect={selectNote}
												onCopy={copyNoteContent}
												{copyStatus}
												{copiedPath}
											/>
										</li>
									{/each}
								</ul>
							{/if}
						</section>
					{/each}

					{#if rootFiles.length > 0}
						<section class={`reader-section-block ${hasSections ? 'border-t border-white/6 pt-4' : ''}`}>
							<div class="reader-section-label reader-section-label-static">
								<span>Loose notes</span>
							</div>
							<ul class="mt-2 space-y-1">
								{#each rootFiles as item (item.path || item.title)}
									<li>
										<NoteTreeItem
											{item}
											depth={0}
											{selectedPath}
											{expandedFolders}
											onToggleFolder={toggleFolder}
											onSelect={selectNote}
											onCopy={copyNoteContent}
											{copyStatus}
											{copiedPath}
										/>
									</li>
								{/each}
							</ul>
						</section>
					{/if}
				{/if}
			{/if}
		</ReaderSidebar>
	{/snippet}

	<div class="reader-content-header">
		<div>
			<div class="reader-content-kicker">Notes workspace</div>
			<h1 class="reader-content-title">{selectedTitle}</h1>
			<div class="reader-content-meta">{selectedBreadcrumb}</div>
		</div>

		{#if selectedPath}
			<button type="button" class="reader-secondary-button" onclick={() => void copyNoteContent(selectedPath)}>
				{#if copiedPath === selectedPath && copyStatus === 'copied'}
					Copied
				{:else if copiedPath === selectedPath && copyStatus === 'error'}
					Copy failed
				{:else}
					Copy markdown
				{/if}
			</button>
		{/if}
	</div>

	<div class="reader-document-region">
		{#if error}
			<div class="reader-alert reader-alert-danger mb-4">
				{error}
				<button type="button" class="ml-2 underline" onclick={() => (error = '')}>dismiss</button>
			</div>
		{/if}

		{#if !selectedPath}
			<div class="py-6 text-sm text-muted-foreground">Select a note from the sidebar.</div>
		{:else if docLoading}
			<div class="py-6 text-sm text-muted-foreground">Loading note…</div>
		{:else}
			<div class="prose-dark reader-prose text-sm">
				{@html rendered}
			</div>
		{/if}
	</div>
</ReaderWorkspace>
