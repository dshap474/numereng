<script lang="ts">
	import type { Snippet } from 'svelte';

	let { sidebarWidthClass = 'w-[19rem]', contentClass = '', sidebar, children } = $props<{
		sidebarWidthClass?: string;
		contentClass?: string;
		sidebar?: Snippet;
		children?: Snippet;
	}>();
</script>

<div class="-m-8 -mt-14 md:-mt-8 flex h-screen bg-background">
	<aside class={`reader-local-rail ${sidebarWidthClass} flex-shrink-0 overflow-hidden`}>
		{@render sidebar?.()}
	</aside>

	<div class={`reader-content-region ${contentClass}`}>
		<div class="reader-content-shell">
			{@render children?.()}
		</div>
	</div>
</div>
