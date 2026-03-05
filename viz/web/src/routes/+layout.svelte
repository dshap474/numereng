<script lang="ts">
	import '../app.css';
	import { page } from '$app/stores';

	let { data, children } = $props();

	let currentPath = $derived($page.url.pathname);
	let sidebarOpen = $state(false);
	let experimentsExpanded = $state(false);
	let docsExpanded = $state(false);

	let onExperimentsRoute = $derived(currentPath.startsWith('/experiments'));
	let onDocsRoute = $derived(currentPath.startsWith('/docs'));

	$effect(() => { if (onDocsRoute) docsExpanded = true; });
	$effect(() => { if (onExperimentsRoute) experimentsExpanded = true; });

	let showDocsList = $derived(docsExpanded);
	let showExperimentList = $derived(experimentsExpanded);

	let activeExperimentId = $derived.by(() => {
		const match = currentPath.match(/\/experiments\/([^/]+)/);
		return match ? match[1] : null;
	});

	function isActive(experimentId: string): boolean {
		return activeExperimentId === experimentId;
	}

	function closeSidebar() {
		sidebarOpen = false;
	}
</script>

<!-- Skip to content -->
<a href="#main-content" class="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:bg-primary focus:text-primary-foreground focus:px-4 focus:py-2 focus:rounded-md focus:text-sm">
	Skip to content
</a>

<div class="flex h-screen">
	<!-- Mobile hamburger -->
	<button
		type="button"
		class="fixed top-3 left-3 z-40 p-2 rounded-md bg-card border border-border md:hidden"
		aria-label="Toggle sidebar"
		onclick={() => (sidebarOpen = !sidebarOpen)}
	>
		<svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
			{#if sidebarOpen}
				<path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
			{:else}
				<path stroke-linecap="round" stroke-linejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
			{/if}
		</svg>
	</button>

	<!-- Backdrop -->
	{#if sidebarOpen}
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<div class="fixed inset-0 z-30 bg-black/50 md:hidden" onclick={closeSidebar} onkeydown={(e) => e.key === 'Escape' && closeSidebar()}></div>
	{/if}

	<aside
		class="w-60 border-r border-sidebar-border bg-sidebar-background flex flex-col flex-shrink-0 h-screen fixed z-30 transition-transform md:relative md:translate-x-0 {sidebarOpen ? 'translate-x-0' : '-translate-x-full'}"
	>
		<div class="px-4 pb-4 pt-[20px] flex-shrink-0">
			<a
				href="/"
				class="block px-3 mb-2 text-lg font-semibold tracking-tight text-sidebar-primary rounded-md transition-all hover:text-white active:scale-[0.97]"
				onclick={closeSidebar}
			>
				Numereng
			</a>
			<nav aria-label="Main navigation">
				<!-- Docs nav item with expandable chevron -->
				<div class="mt-1 group relative">
					<a
						href="/docs"
						aria-current={currentPath === '/docs' ? 'page' : undefined}
						class="flex items-center gap-2 px-3 py-2 text-sm transition-colors {showDocsList ? 'rounded-t-md bg-sidebar-accent/30' : 'rounded-md'} {onDocsRoute
							? 'bg-sidebar-accent text-sidebar-accent-foreground font-medium'
							: 'text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground'}"
						onclick={closeSidebar}
					>
						<svg class="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
							<path stroke-linecap="round" stroke-linejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
						</svg>
						Docs
						<button
							type="button"
							aria-label={showDocsList ? 'Collapse docs' : 'Expand docs'}
							class="ml-auto p-0.5 rounded hover:bg-sidebar-border transition-all opacity-60 hover:opacity-100"
							onclick={(e) => { e.preventDefault(); e.stopPropagation(); docsExpanded = !docsExpanded; }}
						>
							<svg class="w-3.5 h-3.5 transition-transform {showDocsList ? 'rotate-90' : ''}" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
								<path stroke-linecap="round" stroke-linejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
							</svg>
						</button>
					</a>
					{#if showDocsList}
						<div class="bg-sidebar-accent/30 rounded-b-md p-1 space-y-1">
							<a
								href="/docs/numerai"
								class="block px-3 py-2 rounded-md transition-colors {currentPath === '/docs/numerai'
									? 'bg-sidebar-accent text-sidebar-accent-foreground'
									: 'text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground'}"
								onclick={closeSidebar}
							>
								<span class="text-sm font-medium block pl-4">Numerai</span>
								<span class="text-[10px] text-sidebar-foreground/50 block pl-4">Official docs</span>
							</a>
							<a
								href="/docs"
								class="block px-3 py-2 rounded-md transition-colors {currentPath === '/docs'
									? 'bg-sidebar-accent text-sidebar-accent-foreground'
									: 'text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground'}"
								onclick={closeSidebar}
							>
								<span class="text-sm font-medium block pl-4">Numereng</span>
								<span class="text-[10px] text-sidebar-foreground/50 block pl-4">Project docs</span>
							</a>
						</div>
					{/if}
				</div>
				<!-- Notes nav item -->
				<a
					href="/notes"
					aria-current={currentPath.startsWith('/notes') ? 'page' : undefined}
					class="mt-1 flex items-center gap-2 px-3 py-2 rounded-md text-sm transition-colors {currentPath.startsWith('/notes')
						? 'bg-sidebar-accent text-sidebar-accent-foreground font-medium'
						: 'text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground'}"
					onclick={closeSidebar}
				>
					<svg class="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
						<path stroke-linecap="round" stroke-linejoin="round" d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0115.75 21H5.25A2.25 2.25 0 013 18.75V8.25A2.25 2.25 0 015.25 6H10" />
					</svg>
					Notes
				</a>
				<!-- Experiments nav item with expandable chevron -->
				<div class="mt-1 group relative">
					<a
						href="/experiments"
						aria-current={currentPath === '/experiments' ? 'page' : undefined}
						class="flex items-center gap-2 px-3 py-2 text-sm transition-colors {showExperimentList ? 'rounded-t-md bg-sidebar-accent/30' : 'rounded-md'} {onExperimentsRoute
							? 'bg-sidebar-accent text-sidebar-accent-foreground font-medium'
							: 'text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground'}"
						onclick={closeSidebar}
					>
						<svg class="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
							<path stroke-linecap="round" stroke-linejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />
						</svg>
						Experiments
						{#if data.experiments.length > 0}
							<button
								type="button"
								aria-label={showExperimentList ? 'Collapse experiments' : 'Expand experiments'}
								class="ml-auto p-0.5 rounded hover:bg-sidebar-border transition-all opacity-60 hover:opacity-100"
								onclick={(e) => { e.preventDefault(); e.stopPropagation(); experimentsExpanded = !experimentsExpanded; }}
							>
								<svg class="w-3.5 h-3.5 transition-transform {showExperimentList ? 'rotate-90' : ''}" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
									<path stroke-linecap="round" stroke-linejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
								</svg>
							</button>
						{/if}
					</a>
					{#if showExperimentList && data.experiments.length > 0}
						<div class="bg-sidebar-accent/30 rounded-b-md p-1 space-y-1 max-h-[60vh] overflow-y-auto">
							<nav aria-label="Experiments">
								<ul class="space-y-1">
									{#each data.experiments as exp (exp.experiment_id)}
										<li>
											<a
												href="/experiments/{exp.experiment_id}"
												aria-current={isActive(exp.experiment_id) ? 'page' : undefined}
												class="block px-3 py-3 rounded-md transition-colors {isActive(exp.experiment_id)
													? 'bg-sidebar-accent text-sidebar-accent-foreground'
													: 'text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground'}"
												onclick={closeSidebar}
											>
												<div class="flex items-center justify-between mb-1">
													<span class="text-[9px] text-sidebar-foreground/40 tabular-nums">{exp.created_at?.slice(0, 10) ?? ''}</span>
													</div>
												<span class="text-sm font-medium truncate block">{exp.name}</span>
											</a>
										</li>
									{/each}
								</ul>
							</nav>
						</div>
					{/if}
				</div>
			</nav>
		</div>
	</aside>
	<main id="main-content" class="flex-1 overflow-y-auto overscroll-contain bg-background md:ml-0">
		<div class="p-8 pt-14 md:pt-8">
			{@render children()}
		</div>
	</main>
</div>
