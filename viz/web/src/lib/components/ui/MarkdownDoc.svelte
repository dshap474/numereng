<script lang="ts">
	import ExperimentDocHeader from '$lib/components/ui/ExperimentDocHeader.svelte';
	import {
		extractExperimentDocHeader,
		type ExperimentDocContext
	} from '$lib/markdown/experiment';
	import { renderMarkdown, type MarkdownSurface } from '$lib/markdown/render';

	let {
		label,
		load,
		save = null,
		tabs = [],
		activeTab = null,
		onTabChange = null,
		borderless = false,
		linkSurface = null,
		linkCurrentPath = null,
		variant = 'default',
		experimentContext = null,
		readOnly = false,
		readOnlyMessage = 'Read-only mode: editing is disabled.'
	}: {
		label: string;
		load: () => Promise<{ content: string; exists: boolean }>;
		save?: ((content: string) => Promise<void>) | null;
		tabs?: Array<{ key: string; label: string }>;
		activeTab?: string | null;
		onTabChange?: ((key: string) => void) | null;
		borderless?: boolean;
		linkSurface?: MarkdownSurface | null;
		linkCurrentPath?: string | null;
		variant?: 'default' | 'experiment';
		experimentContext?: ExperimentDocContext | null;
		readOnly?: boolean;
		readOnlyMessage?: string;
	} = $props();

	let content = $state('');
	let exists = $state(false);
	let editing = $state(false);
	let draft = $state('');
	let loading = $state(true);
	let saving = $state(false);
	let error = $state<string | null>(null);

	let experimentDoc = $derived.by(() => {
		if (variant !== 'experiment') return null;
		return extractExperimentDocHeader(content);
	});

	let rendered = $derived.by(() => {
		try {
			const markdownBody = experimentDoc?.body ?? content;
			return renderMarkdown(markdownBody, {
				surface: linkSurface,
				currentPath: linkCurrentPath ?? label
			});
		} catch {
			return '<p>Failed to parse markdown.</p>';
		}
	});

	$effect(() => {
		loading = true;
		error = null;
		load()
			.then((result) => {
				content = result.content;
				exists = result.exists;
			})
			.catch((err) => {
				error = err instanceof Error ? err.message : 'Failed to load document.';
			})
			.finally(() => {
				loading = false;
			});
	});

	function startEditing() {
		if (readOnly) return;
		draft = content;
		editing = true;
	}

	function cancelEditing() {
		editing = false;
		draft = '';
	}

	async function saveDoc() {
		if (readOnly || save == null) return;
		saving = true;
		error = null;
		try {
			await save(draft);
			content = draft;
			exists = true;
			editing = false;
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to save document.';
		} finally {
			saving = false;
		}
	}
</script>

<div class="{borderless ? 'bg-background overflow-hidden flex-1 min-h-0 flex flex-col' : 'border border-border rounded-lg bg-card overflow-hidden h-[700px] flex flex-col'}">
	{#if tabs.length > 0}
		<div class="flex items-center justify-between px-4 py-2.5 border-b border-border bg-muted/20">
			<div class="flex items-center gap-1">
				{#each tabs as tab (tab.key)}
					<button
						type="button"
						class="px-2.5 py-1 rounded text-xs font-medium transition-colors {activeTab === tab.key
							? 'bg-card text-foreground border border-border'
							: 'text-muted-foreground hover:text-foreground'}"
						onclick={() => onTabChange?.(tab.key)}
					>
						{tab.label}
					</button>
				{/each}
			</div>
			{#if !readOnly}
				<div class="flex gap-2">
					{#if editing}
						<button
							type="button"
							class="px-2.5 py-1 rounded border border-border text-xs text-muted-foreground hover:text-foreground"
							disabled={saving}
							onclick={cancelEditing}
						>Cancel</button>
						<button
							type="button"
							class="px-2.5 py-1 rounded bg-primary text-primary-foreground text-xs font-medium disabled:opacity-50"
							disabled={saving}
							onclick={() => void saveDoc()}
						>{saving ? 'Saving...' : 'Save'}</button>
					{:else}
						<button
							type="button"
							class="px-2.5 py-1 rounded border border-border text-xs text-muted-foreground hover:text-foreground"
							disabled={loading}
							onclick={startEditing}
						>Edit</button>
					{/if}
				</div>
			{/if}
		</div>
	{/if}

	<div class="px-6 py-5 flex-1 min-h-0 overflow-y-auto">
		{#if readOnly && !editing && readOnlyMessage.trim().length > 0}
			<p class="mb-3 text-xs text-amber-300">{readOnlyMessage}</p>
		{/if}
		{#if tabs.length === 0}
			{#if !readOnly}
				<div class="float-right sticky top-0 z-10 flex gap-2 ml-4">
					{#if editing}
						<button
							type="button"
							class="px-2.5 py-1 rounded border border-border text-xs text-muted-foreground hover:text-foreground bg-card"
							disabled={saving}
							onclick={cancelEditing}
						>Cancel</button>
						<button
							type="button"
							class="px-2.5 py-1 rounded bg-primary text-primary-foreground text-xs font-medium disabled:opacity-50"
							disabled={saving}
							onclick={() => void saveDoc()}
						>{saving ? 'Saving...' : 'Save'}</button>
					{:else}
						<button
							type="button"
							class="px-2.5 py-1 rounded border border-border text-xs text-muted-foreground hover:text-foreground bg-card"
							disabled={loading}
							onclick={startEditing}
						>Edit</button>
					{/if}
				</div>
			{/if}
		{/if}
		{#if loading}
			<p class="text-sm text-muted-foreground">Loading...</p>
		{:else if error}
			<p class="text-sm text-negative">{error}</p>
		{:else if editing}
			<textarea
				class="w-full min-h-[300px] rounded-md border border-border bg-background p-3 text-sm font-mono resize-y focus:outline-none focus:ring-1 focus:ring-ring"
				bind:value={draft}
			></textarea>
		{:else if content}
			{#if variant === 'experiment' && experimentContext}
				<ExperimentDocHeader
					title={experimentDoc?.title ?? null}
					headerEntries={experimentDoc?.headerEntries ?? []}
					context={experimentContext}
				/>
			{/if}
			{#if rendered.trim().length > 0}
				<div class="prose-dark text-sm">
					{@html rendered}
				</div>
			{/if}
		{:else}
			<p class="text-sm text-muted-foreground italic">
				No {label.toLowerCase()} yet.{readOnly ? '' : ' Click Edit to create one.'}
			</p>
		{/if}
	</div>
</div>
