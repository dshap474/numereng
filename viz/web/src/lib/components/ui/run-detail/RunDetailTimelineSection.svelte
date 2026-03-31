<script lang="ts">
	import type { RunEvent } from '$lib/api/client';

	let {
		loading = false,
		error = null,
		events = []
	}: {
		loading?: boolean;
		error?: string | null;
		events?: RunEvent[];
	} = $props();

	function formatTime(value: string | undefined | null): string {
		if (!value) return '—';
		const date = new Date(value);
		if (Number.isNaN(date.getTime())) return value;
		return date.toLocaleString();
	}

	function formatPayload(payload: Record<string, unknown> | undefined): string {
		if (!payload) return '';
		const text = JSON.stringify(payload);
		if (text.length <= 200) return text;
		return `${text.slice(0, 200)}...`;
	}
</script>

<div class="bg-card border border-border rounded-lg p-5">
	<h3 class="text-sm font-semibold mb-3">Run Timeline</h3>
	{#if loading}
		<p class="text-sm text-muted-foreground" aria-busy="true">Loading timeline...</p>
	{:else if error}
		<p class="text-sm text-negative">{error}</p>
	{:else if events.length > 0}
		<div class="space-y-2 max-h-[540px] overflow-y-auto pr-1">
			{#each [...events].reverse() as event (event.id)}
				<div class="rounded-md border border-border bg-muted/20 px-3 py-2">
					<div class="flex items-center justify-between gap-3">
						<span class="text-xs font-medium text-foreground">{event.event_type}</span>
						<span class="text-[11px] text-muted-foreground">{formatTime(event.created_at)}</span>
					</div>
					{#if event.payload && Object.keys(event.payload).length > 0}
						<p class="mt-1 text-[11px] text-muted-foreground font-mono break-all">{formatPayload(event.payload)}</p>
					{/if}
				</div>
			{/each}
		</div>
	{:else}
		<p class="text-sm text-muted-foreground">No events recorded.</p>
	{/if}
</div>
