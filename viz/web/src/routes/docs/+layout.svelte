<script lang="ts">
	import { page } from '$app/stores';
	import DocCollectionView from '$lib/components/ui/reader/DocCollectionView.svelte';

	let { children } = $props();

	let variant = $derived<'numereng' | 'numerai'>(
		$page.url.pathname.startsWith('/docs/numerai') ? 'numerai' : 'numereng'
	);
</script>

<DocCollectionView {variant} />
{@render children?.()}
