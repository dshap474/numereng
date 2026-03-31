<script lang="ts">
	import RunDetailPanel from '$lib/components/ui/RunDetailPanel.svelte';

	let { data } = $props();
</script>

<div class="space-y-6">
	{#key data.runId}
		<RunDetailPanel
			runId={data.runId}
			experimentId={data.experimentId}
			experimentName={data.experiment?.name ?? null}
			source={data.source}
			readOnly={Boolean(data.capabilities?.read_only)}
		/>
	{/key}
</div>
