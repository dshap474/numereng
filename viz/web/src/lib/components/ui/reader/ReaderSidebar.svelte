<script lang="ts">
	import type { Snippet } from 'svelte';

	let {
		kicker,
		title,
		description = '',
		toolbar,
		children
	} = $props<{
		kicker: string;
		title: string;
		description?: string;
		toolbar?: Snippet;
		children?: Snippet;
	}>();
</script>

<div class="flex h-full flex-col">
	<div class="reader-sidebar-header">
		<div class="reader-sidebar-kicker">{kicker}</div>
		<h2 class="reader-sidebar-title">{title}</h2>
		{#if description}
			<p class="reader-sidebar-description">{description}</p>
		{/if}
		{#if toolbar}
			<div class="reader-sidebar-toolbar">
				{@render toolbar()}
			</div>
		{/if}
	</div>

	<div class="min-h-0 flex-1 overflow-y-auto px-4 pt-4 pb-5">
		{@render children?.()}
	</div>
</div>
