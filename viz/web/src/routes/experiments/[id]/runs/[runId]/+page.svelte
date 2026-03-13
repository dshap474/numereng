<script lang="ts">
	import RunDetailPanel from '$lib/components/ui/RunDetailPanel.svelte';

	let { data } = $props();
</script>

<div class="space-y-6">
	<RunDetailPanel
		runId={data.runId}
		experimentId={data.experimentId}
		experimentName={data.experiment?.name ?? null}
		runs={Array.isArray(data.runs) ? data.runs : []}
		readOnly={Boolean(data.capabilities?.read_only)}
	/>
</div>
