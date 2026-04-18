<script lang="ts">
	import { onMount } from 'svelte';

	let terminalEl: HTMLElement | null = null;
	let noiseCanvas: HTMLCanvasElement | null = null;
	let barHeights: number[] = [];

	const footerMetrics = [
		{ label: 'CORR_NATIVE', value: '0.1214' },
		{ label: 'CORR_ENDER20', value: '0.1182' },
		{ label: 'BMC_LAST_200 | BMC_MEAN', value: '0.1042 | 0.0965' },
		{ label: 'FNC_MEAN | MMC_MEAN', value: '0.0441 | 0.0187' }
	];

	function randomBarHeight() {
		return 20 + Math.random() * 80;
	}

	onMount(() => {
		barHeights = Array.from({ length: 40 }, () => Math.random() * 100);

		const intervalId = window.setInterval(() => {
			barHeights = barHeights.map(() => randomBarHeight());
		}, 150);

		if (!noiseCanvas || !terminalEl) {
			return () => {
				window.clearInterval(intervalId);
			};
		}

		const gl = noiseCanvas.getContext('webgl');
		if (!gl) {
			return () => {
				window.clearInterval(intervalId);
			};
		}

		const resize = () => {
			if (!noiseCanvas || !terminalEl) return;
			noiseCanvas.width = terminalEl.clientWidth;
			noiseCanvas.height = terminalEl.clientHeight;
			gl.viewport(0, 0, noiseCanvas.width, noiseCanvas.height);
		};

		const compileShader = (type: number, source: string): WebGLShader | null => {
			const shader = gl.createShader(type);
			if (!shader) return null;
			gl.shaderSource(shader, source);
			gl.compileShader(shader);
			if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
				gl.deleteShader(shader);
				return null;
			}
			return shader;
		};

		const vertexSource = 'attribute vec2 p; void main() { gl_Position = vec4(p, 0.0, 1.0); }';
		const fragmentSource = `
			precision highp float;
			uniform vec2 res;
			uniform float t;

			float rnd(vec2 s) {
				return fract(sin(dot(s, vec2(12.9898, 78.233))) * 43758.5453);
			}

			void main() {
				vec2 s = gl_FragCoord.xy / res.xy * 2.0;
				gl_FragColor = vec4(vec3(rnd(s + t * 0.01)), 1.0);
			}
		`;

		const vertexShader = compileShader(gl.VERTEX_SHADER, vertexSource);
		const fragmentShader = compileShader(gl.FRAGMENT_SHADER, fragmentSource);
		if (!vertexShader || !fragmentShader) {
			return () => {
				window.clearInterval(intervalId);
			};
		}

		const program = gl.createProgram();
		if (!program) {
			gl.deleteShader(vertexShader);
			gl.deleteShader(fragmentShader);
			return () => {
				window.clearInterval(intervalId);
			};
		}

		gl.attachShader(program, vertexShader);
		gl.attachShader(program, fragmentShader);
		gl.linkProgram(program);

		if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
			gl.deleteProgram(program);
			gl.deleteShader(vertexShader);
			gl.deleteShader(fragmentShader);
			return () => {
				window.clearInterval(intervalId);
			};
		}

		gl.useProgram(program);

		const buffer = gl.createBuffer();
		if (!buffer) {
			gl.deleteProgram(program);
			gl.deleteShader(vertexShader);
			gl.deleteShader(fragmentShader);
			return () => {
				window.clearInterval(intervalId);
			};
		}

		gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
		gl.bufferData(
			gl.ARRAY_BUFFER,
			new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
			gl.STATIC_DRAW
		);

		const position = gl.getAttribLocation(program, 'p');
		if (position !== -1) {
			gl.enableVertexAttribArray(position);
			gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
		}

		const resolution = gl.getUniformLocation(program, 'res');
		const time = gl.getUniformLocation(program, 't');

		let animationFrame = 0;

		const loop = (frameTime: number) => {
			if (!noiseCanvas) return;
			if (resolution) {
				gl.uniform2f(resolution, noiseCanvas.width, noiseCanvas.height);
			}
			if (time) {
				gl.uniform1f(time, frameTime);
			}
			gl.drawArrays(gl.TRIANGLES, 0, 6);
			animationFrame = window.requestAnimationFrame(loop);
		};

		window.addEventListener('resize', resize);
		resize();
		animationFrame = window.requestAnimationFrame(loop);

		return () => {
			window.clearInterval(intervalId);
			window.removeEventListener('resize', resize);
			window.cancelAnimationFrame(animationFrame);
			gl.deleteBuffer(buffer);
			gl.deleteProgram(program);
			gl.deleteShader(vertexShader);
			gl.deleteShader(fragmentShader);
		};
	});
</script>

<svelte:head>
	<link rel="preconnect" href="https://fonts.googleapis.com" />
	<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin="anonymous" />
	<link
		href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap"
		rel="stylesheet"
	/>
</svelte:head>

<div
	class="launch-mirror-shell -mx-8 -mt-14 -mb-8 flex h-screen min-h-0 flex-col overflow-x-hidden overflow-y-auto md:-mt-8"
>
	<main class="terminal" bind:this={terminalEl}>
		<canvas id="noise-canvas" bind:this={noiseCanvas}></canvas>

		<nav class="header-nav">
			<span class="sys-label">LIVE_TELEMETRY</span>
		</nav>

		<div class="display-header">NUMERENG</div>

		<div class="launch-subhead">// A NUMERAI AGENTIC DEVELOPMENT ENGINE</div>

		<section class="full-bleed-graphic">
			<svg class="sun-core-svg" viewBox="0 0 400 400" aria-hidden="true">
				<circle cx="200" cy="200" r="40" fill="var(--ink)" opacity="0.9"></circle>
				<circle cx="200" cy="200" r="60" fill="none" stroke="var(--ink)" stroke-width="0.5"></circle>
				<circle
					cx="200"
					cy="200"
					r="85"
					fill="none"
					stroke="var(--ink)"
					stroke-width="1"
					stroke-dasharray="4 8"
				></circle>
				<circle cx="200" cy="200" r="110" fill="none" stroke="var(--ink)" stroke-width="0.5"></circle>
				<circle
					cx="200"
					cy="200"
					r="140"
					fill="none"
					stroke="var(--ink)"
					stroke-width="2"
					stroke-dasharray="1 15"
				></circle>
				<circle
					cx="200"
					cy="200"
					r="180"
					fill="none"
					stroke="var(--ink)"
					stroke-width="0.5"
					opacity="0.4"
				></circle>

				<line
					x1="0"
					y1="200"
					x2="400"
					y2="200"
					stroke="var(--ink)"
					stroke-width="0.5"
					opacity="0.3"
				></line>
				<line
					x1="200"
					y1="0"
					x2="200"
					y2="400"
					stroke="var(--ink)"
					stroke-width="0.5"
					opacity="0.3"
				></line>

				<g>
					<circle cx="280" cy="120" r="3" fill="var(--ink)"></circle>
					<text class="point-label" x="288" y="120">ender_20</text>
					<circle cx="100" cy="300" r="3" fill="var(--ink)"></circle>
					<text class="point-label" x="108" y="300">cyrus_20</text>
				</g>
			</svg>

			<div class="overlay-data">
				<div class="readout-box">
					<div class="sys-label">LGBM</div>
					<div class="readout-val readout-config">255L | LR .03</div>
				</div>
				<div style="align-self: flex-end">
					<div class="readout-box" style="margin-bottom: 8px">
						<div class="sys-label">XGBOOST</div>
						<div class="readout-val readout-config">D6 | ETA .05</div>
					</div>
					<div
						class="capsule"
						style="background: var(--ink); color: var(--bg); font-weight: bold; text-align: center"
					>
						RUN_ARTIFACTS_READY
					</div>
				</div>
			</div>
		</section>

		<section class="spectrum-viz">
			<span class="sys-label">RUN_METRIC_SERIES</span>
			<div class="spectrum-bars" id="bars">
				{#each barHeights as height, index (`bar-${index}`)}
					<div class="bar" style={`height: ${height}%`}></div>
				{/each}
			</div>
		</section>

		<div class="footer-metrics">
			{#each footerMetrics as metric (metric.label)}
				<div class="metric-item">
					<div class="metric-label">{metric.label}</div>
					<div class="metric-value">{metric.value}</div>
				</div>
			{/each}
		</div>
	</main>
</div>

<style>
	.launch-mirror-shell {
		--bg: #0f0f11;
		--ink: #dadada;
		--font-display: Impact, 'Arial Black', sans-serif;
		--font-mono: 'Space Mono', monospace;
		--border-fine: 1px solid var(--ink);
		background-color: var(--bg);
	}

	.launch-mirror-shell,
	.launch-mirror-shell * {
		box-sizing: border-box;
		user-select: none;
	}

	#noise-canvas {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		z-index: 5;
		pointer-events: none;
		mix-blend-mode: overlay;
		opacity: 0.6;
	}

	.terminal {
		position: relative;
		width: 100%;
		height: 100%;
		padding: 1.5rem 1rem;
		display: flex;
		flex-direction: column;
		z-index: 1;
		background-color: var(--bg);
		color: var(--ink);
		font-family: var(--font-mono);
		overflow-x: hidden;
		-webkit-font-smoothing: antialiased;
	}

	.header-nav {
		display: flex;
		justify-content: flex-end;
		align-items: center;
		position: absolute;
		top: 1.5rem;
		right: 1rem;
		left: 1rem;
		z-index: 6;
	}

	.display-header {
		font-family: var(--font-display);
		font-size: 12vw;
		line-height: 0.8;
		letter-spacing: -0.02em;
		text-transform: uppercase;
		margin-bottom: 1rem;
		position: relative;
		z-index: 3;
	}

	.full-bleed-graphic {
		flex-grow: 1;
		position: relative;
		border: var(--border-fine);
		margin: 1rem 0;
		overflow: hidden;
		display: flex;
		justify-content: center;
		align-items: center;
		background: radial-gradient(circle, rgba(218, 218, 218, 0.05) 0%, transparent 70%);
		z-index: 3;
	}

	.sun-core-svg {
		width: 140%;
		height: 140%;
		animation: slow-rotate 120s linear infinite;
	}

	.point-label {
		fill: var(--ink);
		font-family: var(--font-mono);
		font-size: 8px;
		text-anchor: start;
		dominant-baseline: middle;
	}

	.overlay-data {
		position: absolute;
		top: 10px;
		left: 10px;
		right: 10px;
		bottom: 10px;
		pointer-events: none;
		display: flex;
		flex-direction: column;
		justify-content: space-between;
	}

	.sys-label {
		font-size: 0.55rem;
		letter-spacing: 0.2em;
		text-transform: uppercase;
		position: relative;
		z-index: 3;
	}

	.launch-subhead {
		font-size: 0.82rem;
		letter-spacing: 0.16em;
		text-transform: uppercase;
		position: relative;
		z-index: 3;
	}

	.readout-box {
		border: var(--border-fine);
		background: var(--bg);
		padding: 8px;
		width: fit-content;
	}

	.readout-val {
		font-size: 1.2rem;
		font-weight: bold;
	}

	.readout-config {
		font-size: 0.92rem;
		letter-spacing: 0.08em;
		white-space: nowrap;
	}

	.spectrum-viz {
		height: 60px;
		border: var(--border-fine);
		margin-top: 1rem;
		display: flex;
		flex-direction: column;
		padding: 4px;
		position: relative;
		z-index: 3;
	}

	.spectrum-bars {
		flex-grow: 1;
		display: flex;
		align-items: flex-end;
		gap: 2px;
	}

	.bar {
		flex: 1;
		background: var(--ink);
		opacity: 0.6;
	}

	.capsule {
		border: var(--border-fine);
		border-radius: 50px;
		padding: 0.3rem 0.8rem;
		font-size: 0.6rem;
		text-transform: uppercase;
	}

	.footer-metrics {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 10px;
		margin-top: 1rem;
		position: relative;
		z-index: 3;
	}

	.metric-item {
		border-top: var(--border-fine);
		padding-top: 4px;
	}

	.metric-label {
		font-size: 0.5rem;
		opacity: 0.7;
	}

	.metric-value {
		font-size: 0.75rem;
	}

	@keyframes slow-rotate {
		from {
			transform: rotate(0deg);
		}

		to {
			transform: rotate(360deg);
		}
	}

	@media (max-width: 900px) {
		.display-header {
			font-size: 16vw;
		}
	}

	@media (max-width: 640px) {
		.launch-mirror-shell {
			overflow-y: auto;
		}

		.terminal {
			height: auto;
			min-height: 100%;
		}

		.footer-metrics {
			grid-template-columns: 1fr;
		}

		.full-bleed-graphic {
			min-height: 20rem;
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.sun-core-svg {
			animation: none;
		}
	}
</style>
