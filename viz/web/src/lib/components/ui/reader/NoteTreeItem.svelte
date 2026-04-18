<script lang="ts">
	import type { NumeraiDocNode } from '$lib/api/client';
	import SelfTree from './NoteTreeItem.svelte';

	let {
		item,
		depth = 0,
		selectedPath,
		expandedFolders,
		onToggleFolder,
		onSelect,
		onCopy,
		copyStatus = 'idle',
		copiedPath = null
	} = $props<{
		item: NumeraiDocNode;
		depth?: number;
		selectedPath: string;
		expandedFolders: Set<string>;
		onToggleFolder: (folderKey: string) => void;
		onSelect: (path: string) => void;
		onCopy: (path: string) => void;
		copyStatus?: 'idle' | 'copied' | 'error';
		copiedPath?: string | null;
	}>();

	function isSelected(path: string | null): boolean {
		return path === selectedPath;
	}
</script>

{#if item.children}
	{@const folderKey = item.path ?? item.title}
	<div class="reader-folder-block">
		<button
			type="button"
			class="reader-folder-row"
			style={`padding-left: calc(0.8rem + ${depth} * 0.72rem)`}
			onclick={() => onToggleFolder(folderKey)}
		>
			<span class="reader-folder-title">{item.title}</span>
			<svg
				class={`reader-chevron ${expandedFolders.has(folderKey) ? 'rotate-90' : ''}`}
				fill="none"
				viewBox="0 0 24 24"
				stroke="currentColor"
				stroke-width="2"
			>
				<path stroke-linecap="round" stroke-linejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
			</svg>
		</button>

		{#if expandedFolders.has(folderKey)}
			<ul class="mt-1 space-y-1">
				{#each item.children as child (child.path || child.title)}
					<li>
						<SelfTree
							item={child}
							depth={depth + 1}
							{selectedPath}
							{expandedFolders}
							{onToggleFolder}
							{onSelect}
							{onCopy}
							{copyStatus}
							{copiedPath}
						/>
					</li>
				{/each}
			</ul>
		{/if}
	</div>
{:else if item.path}
	<div
		class={`reader-note-row ${isSelected(item.path) ? 'reader-note-row-active' : ''}`}
		style={`padding-left: calc(0.8rem + ${depth} * 0.72rem)`}
	>
		<button type="button" class="min-w-0 flex-1 text-left" onclick={() => onSelect(item.path!)}>
			<div class="reader-note-title">{item.title}</div>
		</button>
		<button
			type="button"
			class={`reader-note-copy ${
				copiedPath === item.path && copyStatus === 'copied'
					? 'text-positive'
					: copiedPath === item.path && copyStatus === 'error'
						? 'text-negative'
						: ''
			}`}
			title="Copy note content"
			aria-label="Copy note content"
			onclick={(e) => {
				e.preventDefault();
				onCopy(item.path!);
			}}
		>
			{#if copiedPath === item.path && copyStatus === 'copied'}
				<svg class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path stroke-linecap="round" stroke-linejoin="round" d="M4.5 12.75l6 6 9-13.5" />
				</svg>
			{:else if copiedPath === item.path && copyStatus === 'error'}
				<svg class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
				</svg>
			{:else}
				<svg class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8">
					<rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
					<path stroke-linecap="round" stroke-linejoin="round" d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
				</svg>
			{/if}
		</button>
	</div>
{/if}
