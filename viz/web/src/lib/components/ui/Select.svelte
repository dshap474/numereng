<script lang="ts">
	import { onMount, tick } from 'svelte';

	export interface SelectOption {
		value: string;
		label: string;
		disabled?: boolean;
	}

	interface Props {
		options: SelectOption[];
		value?: string;
		placeholder?: string;
		disabled?: boolean;
		size?: 'xs' | 'sm' | 'md';
		tone?: 'default' | 'sidebar';
		align?: 'left' | 'right';
		ariaLabel?: string;
		class?: string;
		menuClass?: string;
	}

	let {
		options,
		value = $bindable(''),
		placeholder = 'Select',
		disabled = false,
		size = 'sm',
		tone = 'default',
		align = 'left',
		ariaLabel,
		class: className = '',
		menuClass = ''
	}: Props = $props();

	let rootEl = $state<HTMLElement | null>(null);
	let buttonEl = $state<HTMLButtonElement | null>(null);
	let menuEl = $state<HTMLDivElement | null>(null);
	let open = $state(false);
	let highlightedIndex = $state(-1);
	let menuPlacement = $state<'bottom' | 'top'>('bottom');
	let menuMaxHeight = $state(256);
	let menuShiftX = $state(0);
	let menuMaxWidth = $state(360);

	let selectedIndex = $derived.by(() => options.findIndex((option) => option.value === value));
	let selectedOption = $derived.by(() =>
		selectedIndex >= 0 ? options[selectedIndex] : null
	);

	let triggerSizeClass = $derived.by(() => {
		switch (size) {
			case 'xs':
				return 'h-7 px-2 py-1 text-[11px]';
			case 'md':
				return 'h-10 px-3 py-2 text-sm';
			case 'sm':
			default:
				return 'h-8 px-2.5 py-1.5 text-xs';
		}
	});

	let triggerToneClass = $derived.by(() => {
		if (tone === 'sidebar') {
			return 'border-sidebar-border bg-sidebar-background text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground';
		}
		return 'border-border bg-background text-foreground hover:bg-muted/30';
	});

	let menuToneClass = $derived.by(() => {
		if (tone === 'sidebar') {
			return 'border-sidebar-border bg-sidebar-background text-sidebar-foreground shadow-[0_10px_28px_rgba(0,0,0,0.45)]';
		}
		return 'border-border bg-card text-card-foreground shadow-[0_12px_30px_rgba(0,0,0,0.45)]';
	});

	let optionSelectedClass = $derived.by(() => {
		if (tone === 'sidebar') {
			return 'bg-sidebar-accent text-sidebar-accent-foreground';
		}
		return 'bg-muted text-foreground';
	});

	let optionHoverClass = $derived.by(() => {
		if (tone === 'sidebar') {
			return 'hover:bg-sidebar-accent/80 hover:text-sidebar-accent-foreground';
		}
		return 'hover:bg-muted/50';
	});

	function closeMenu() {
		open = false;
		highlightedIndex = -1;
	}

	function firstEnabledIndex(): number {
		return options.findIndex((option) => !option.disabled);
	}

	function nextEnabledIndex(start: number, direction: 1 | -1): number {
		if (options.length === 0) return -1;
		let current = start;
		for (let step = 0; step < options.length; step += 1) {
			current = (current + direction + options.length) % options.length;
			if (!options[current]?.disabled) return current;
		}
		return -1;
	}

	async function openMenu() {
		if (disabled || options.length === 0) return;
		open = true;
		if (selectedIndex >= 0 && !options[selectedIndex]?.disabled) {
			highlightedIndex = selectedIndex;
		} else {
			highlightedIndex = firstEnabledIndex();
		}
		await tick();
		updateMenuPlacement();
	}

	function toggleMenu() {
		if (open) {
			closeMenu();
			return;
		}
		void openMenu();
	}

	function selectIndex(index: number) {
		const option = options[index];
		if (!option || option.disabled) return;
		value = option.value;
		closeMenu();
		buttonEl?.focus();
	}

	function handleButtonKeydown(event: KeyboardEvent) {
		if (disabled) return;
			if (event.key === 'ArrowDown') {
				event.preventDefault();
				if (!open) {
					void openMenu();
					return;
				}
			highlightedIndex = nextEnabledIndex(highlightedIndex, 1);
			return;
		}
			if (event.key === 'ArrowUp') {
				event.preventDefault();
				if (!open) {
					void openMenu();
					return;
				}
			highlightedIndex = nextEnabledIndex(highlightedIndex < 0 ? 0 : highlightedIndex, -1);
			return;
		}
			if (event.key === 'Enter' || event.key === ' ') {
				event.preventDefault();
				if (!open) {
					void openMenu();
					return;
				}
			if (highlightedIndex >= 0) selectIndex(highlightedIndex);
		}
		if (event.key === 'Escape') {
			event.preventDefault();
			closeMenu();
		}
	}

	function updateMenuPlacement() {
		if (!buttonEl) return;
		const rect = buttonEl.getBoundingClientRect();
		const viewportHeight = window.innerHeight;
		const viewportWidth = window.innerWidth;
		const gap = 6;
		const horizontalGap = 8;
		const minMenuHeight = 120;
		const preferredMenuHeight = 256;
		const spaceBelow = viewportHeight - rect.bottom - gap;
		const spaceAbove = rect.top - gap;
		const canFitBelow = spaceBelow >= minMenuHeight;
		const canFitAbove = spaceAbove >= minMenuHeight;
		if (!canFitBelow && canFitAbove) {
			menuPlacement = 'top';
		} else if (!canFitAbove && canFitBelow) {
			menuPlacement = 'bottom';
		} else {
			menuPlacement = spaceBelow >= spaceAbove ? 'bottom' : 'top';
		}
		const available = menuPlacement === 'bottom' ? spaceBelow : spaceAbove;
		menuMaxHeight = Math.max(minMenuHeight, Math.min(preferredMenuHeight, Math.floor(available)));

		const measuredMenuWidth =
			menuEl?.offsetWidth ?? Math.max(Math.ceil(rect.width), 180);
		const preferredLeft =
			align === 'right' ? rect.right - measuredMenuWidth : rect.left;
		const minLeft = horizontalGap;
		const maxLeft = Math.max(minLeft, viewportWidth - horizontalGap - measuredMenuWidth);
		const clampedLeft = Math.max(minLeft, Math.min(maxLeft, preferredLeft));
		menuShiftX = Math.round(clampedLeft - rect.left);
		menuMaxWidth = Math.max(180, Math.floor(viewportWidth - horizontalGap * 2));
	}

	$effect(() => {
		if (!open) return;
		void tick().then(() => {
			updateMenuPlacement();
		});
	});

	function handleWindowKeydown(event: KeyboardEvent) {
		if (!open) return;
		if (event.key === 'Escape') {
			event.preventDefault();
			closeMenu();
			buttonEl?.focus();
			return;
		}
		if (event.key === 'ArrowDown') {
			event.preventDefault();
			highlightedIndex = nextEnabledIndex(highlightedIndex, 1);
			return;
		}
		if (event.key === 'ArrowUp') {
			event.preventDefault();
			highlightedIndex = nextEnabledIndex(highlightedIndex < 0 ? 0 : highlightedIndex, -1);
			return;
		}
		if (event.key === 'Enter') {
			event.preventDefault();
			if (highlightedIndex >= 0) selectIndex(highlightedIndex);
			return;
		}
		if (event.key === 'Tab') {
			closeMenu();
		}
	}

	onMount(() => {
		const handlePointerDown = (event: PointerEvent) => {
			if (!open || !rootEl) return;
			const target = event.target as Node | null;
			if (target && !rootEl.contains(target)) {
				closeMenu();
			}
		};
		const handleViewportChange = () => {
			if (!open) return;
			updateMenuPlacement();
		};

		document.addEventListener('pointerdown', handlePointerDown, true);
		window.addEventListener('resize', handleViewportChange);
		window.addEventListener('scroll', handleViewportChange, { capture: true, passive: true });
		return () => {
			document.removeEventListener('pointerdown', handlePointerDown, true);
			window.removeEventListener('resize', handleViewportChange);
			window.removeEventListener('scroll', handleViewportChange, true);
		};
	});
</script>

<svelte:window onkeydown={handleWindowKeydown} />

<div class={`relative w-full ${className}`} bind:this={rootEl}>
	<button
		type="button"
		class={`w-full rounded-md border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50 disabled:cursor-not-allowed ${triggerSizeClass} ${triggerToneClass}`}
		aria-haspopup="listbox"
		aria-expanded={open}
		aria-label={ariaLabel ?? placeholder}
		disabled={disabled}
		onclick={toggleMenu}
		onkeydown={handleButtonKeydown}
		bind:this={buttonEl}
	>
		<span class="flex w-full items-center gap-2">
			<span class="min-w-0 flex-1 truncate text-left">{selectedOption?.label ?? placeholder}</span>
			<svg
				class={`ml-auto h-3.5 w-3.5 shrink-0 transition-transform ${open ? 'rotate-180' : ''}`}
				viewBox="0 0 20 20"
				fill="none"
				aria-hidden="true"
			>
				<path
					d="M5 7.5L10 12.5L15 7.5"
					stroke="currentColor"
					stroke-width="1.5"
					stroke-linecap="round"
					stroke-linejoin="round"
				/>
			</svg>
		</span>
	</button>

	{#if open}
		<div
			class={`absolute z-40 min-w-full rounded-md border p-1 ${menuToneClass} left-0 ${menuPlacement === 'bottom' ? 'top-full mt-1' : 'bottom-full mb-1'} ${menuClass}`}
			style:transform={`translateX(${menuShiftX}px)`}
			style:max-width={`${menuMaxWidth}px`}
			bind:this={menuEl}
		>
			<ul role="listbox" aria-label={ariaLabel ?? placeholder} class="overflow-auto" style:max-height={`${menuMaxHeight}px`}>
				{#each options as option, index (option.value)}
					<li role="presentation">
						<button
							type="button"
							role="option"
							aria-selected={option.value === value}
							class={`w-full rounded px-2 py-1.5 text-left text-sm transition-colors ${optionHoverClass} ${
								option.value === value ? optionSelectedClass : ''
							} ${highlightedIndex === index && option.value !== value ? 'bg-muted/35' : ''} ${
								option.disabled ? 'cursor-not-allowed opacity-40' : ''
							}`}
							disabled={option.disabled}
							onmouseenter={() => {
								if (!option.disabled) highlightedIndex = index;
							}}
							onclick={() => selectIndex(index)}
						>
							<span class="flex items-center justify-between gap-2">
								<span class="truncate">{option.label}</span>
								{#if option.value === value}
									<span class="text-xs opacity-80">✓</span>
								{/if}
							</span>
						</button>
					</li>
				{/each}
			</ul>
		</div>
	{/if}
</div>
