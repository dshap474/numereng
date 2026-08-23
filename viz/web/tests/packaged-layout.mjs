#!/usr/bin/env node
/**
 * Verify the shipped Viz bundle renders the desktop experiment workspace.
 *
 * USAGE:
 *   npm run test:packaged
 */

import assert from 'node:assert/strict';
import { once } from 'node:events';
import { mkdtemp, rm } from 'node:fs/promises';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { chromium } from 'playwright';

// --------------------------------------------------------------------------- //
// Configuration
// --------------------------------------------------------------------------- //

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const VIEWPORT = { width: 1440, height: 900 };
const OVERVIEW_FIXTURE = {
	generated_at: '2026-08-22T12:00:00Z',
	summary: {
		total_experiments: 1,
		active_experiments: 1,
		completed_experiments: 0,
		live_experiments: 0,
		live_runs: 0,
		queued_runs: 0,
		attention_count: 0
	},
	experiments: [
		{
			experiment_id: 'packaged-layout-smoke',
			name: 'Packaged layout smoke',
			status: 'active',
			created_at: '2026-08-22T12:00:00Z',
			updated_at: '2026-08-22T12:00:00Z',
			run_count: 1,
			tags: ['smoke'],
			has_live: false,
			live_run_count: 0,
			attention_state: 'none',
			latest_activity_at: '2026-08-22T12:00:00Z',
			source_kind: 'local',
			detail_href: null
		}
	],
	live_experiments: [],
	recent_activity: [],
	sources: []
};

// --------------------------------------------------------------------------- //
// Process helpers
// --------------------------------------------------------------------------- //

function delay(milliseconds) {
	return new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));
}

async function reservePort() {
	const server = createServer();
	server.listen(0, '127.0.0.1');
	await once(server, 'listening');
	const address = server.address();
	assert(address && typeof address !== 'string', 'Expected an ephemeral TCP port');
	const { port } = address;
	server.close();
	await once(server, 'close');
	return port;
}

async function waitForHealth(url, processHandle, processOutput) {
	for (let attempt = 0; attempt < 100; attempt += 1) {
		if (processHandle.exitCode !== null) {
			throw new Error(`Packaged Viz exited before readiness.\n${processOutput()}`);
		}
		try {
			const response = await fetch(url);
			if (response.ok) return;
		} catch {
			// The server is still starting.
		}
		await delay(100);
	}
	throw new Error(`Packaged Viz did not become healthy.\n${processOutput()}`);
}

async function stopProcess(processHandle) {
	if (processHandle.exitCode !== null) return;
	processHandle.kill('SIGTERM');
	await Promise.race([once(processHandle, 'exit'), delay(2_000)]);
	if (processHandle.exitCode === null) {
		processHandle.kill('SIGKILL');
		await once(processHandle, 'exit');
	}
}

// --------------------------------------------------------------------------- //
// Browser assertion
// --------------------------------------------------------------------------- //

async function inspectPackagedLayout(page, url) {
	await page.route('**/api/experiments/overview**', (route) =>
		route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(OVERVIEW_FIXTURE)
		})
	);
	await page.goto(url, { waitUntil: 'networkidle' });
	await page.locator('.experiment-row').waitFor({ state: 'visible' });

	return page.evaluate(() => {
		const heading = [...document.querySelectorAll('h2')].find(
			(element) => element.textContent?.trim() === 'All experiments'
		);
		const experimentIndex = heading?.closest('aside');
		const workspace = experimentIndex?.parentElement;
		const title = document.querySelector('h1');
		const row = document.querySelector('.experiment-row');
		const paneHeadings = ['All experiments', 'Active systems', 'Recent activity'].map((label) => {
			const element = [...document.querySelectorAll('h2')].find(
				(candidate) => candidate.textContent?.trim() === label
			);
			const box = element?.getBoundingClientRect();
			return {
				label,
				visible: Boolean(
					box &&
					box.width > 0 &&
					box.height > 0 &&
					box.top >= 0 &&
					box.bottom <= window.innerHeight &&
					box.left >= 0 &&
					box.right <= window.innerWidth
				)
			};
		});

		return {
			workspaceDisplay: workspace ? getComputedStyle(workspace).display : null,
			workspaceColumns: workspace ? getComputedStyle(workspace).gridTemplateColumns : null,
			titleFontSize: title ? Number.parseFloat(getComputedStyle(title).fontSize) : 0,
			rowBorderRadius: row ? Number.parseFloat(getComputedStyle(row).borderRadius) : 0,
			rowPaddingTop: row ? Number.parseFloat(getComputedStyle(row).paddingTop) : 0,
			paneHeadings
		};
	});
}

// --------------------------------------------------------------------------- //
// Entry point
// --------------------------------------------------------------------------- //

async function main() {
	const workspaceRoot = await mkdtemp(join(tmpdir(), 'numereng-viz-layout-'));
	const port = await reservePort();
	const origin = `http://127.0.0.1:${port}`;
	const serverOutput = [];
	const apiProcess = spawn(
		'uv',
		['run', 'python', '-m', 'uvicorn', 'viz.api:app', '--host', '127.0.0.1', '--port', String(port)],
		{
			cwd: REPO_ROOT,
			env: { ...process.env, NUMERENG_WORKSPACE_ROOT: workspaceRoot },
			stdio: ['ignore', 'pipe', 'pipe']
		}
	);
	apiProcess.stdout.on('data', (chunk) => serverOutput.push(chunk.toString()));
	apiProcess.stderr.on('data', (chunk) => serverOutput.push(chunk.toString()));

	let browser;
	try {
		await waitForHealth(`${origin}/healthz`, apiProcess, () => serverOutput.join(''));
		browser = await chromium.launch({ headless: true });
		const page = await browser.newPage({ viewport: VIEWPORT });
		const layout = await inspectPackagedLayout(page, `${origin}/experiments`);

		assert.equal(layout.workspaceDisplay, 'grid', `Expected desktop grid, received ${layout.workspaceDisplay}`);
		assert.match(layout.workspaceColumns ?? '', /^340px\s/, 'Expected the desktop experiment index column');
		assert(layout.titleFontSize >= 30, `Expected title size >= 30px, received ${layout.titleFontSize}px`);
		assert(layout.rowBorderRadius > 0, 'Expected packaged experiment rows to retain rounded corners');
		assert(layout.rowPaddingTop > 0, 'Expected packaged experiment rows to retain vertical padding');
		for (const pane of layout.paneHeadings) {
			assert(pane.visible, `Expected ${pane.label} to be visible in the desktop viewport`);
		}

		console.log('PASS: packaged experiment workspace renders the desktop grid and all three panes.');
	} finally {
		if (browser) await browser.close();
		await stopProcess(apiProcess);
		await rm(workspaceRoot, { recursive: true, force: true });
	}
}

main().catch((error) => {
	console.error(error);
	process.exitCode = 1;
});
