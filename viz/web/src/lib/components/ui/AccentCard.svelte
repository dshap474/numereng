<script lang="ts">
	let {
		href = null,
		class: className = '',
		accentClass = 'border-l-transparent',
		paddingClass = 'px-4 py-4',
		roundedClass = 'rounded-[1.1rem]',
		children
	}: {
		href?: string | null;
		class?: string;
		accentClass?: string;
		paddingClass?: string;
		roundedClass?: string;
		children: import('svelte').Snippet;
	} = $props();
</script>

<svelte:element
	this={href ? 'a' : 'div'}
	href={href ?? undefined}
	class={`accent-card block border border-white/7 border-l ${roundedClass} ${paddingClass} ${accentClass} ${className}`}
>
	{@render children()}
</svelte:element>

<style>
	.accent-card {
		background: var(--color-card);
		box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02);
	}
</style>
