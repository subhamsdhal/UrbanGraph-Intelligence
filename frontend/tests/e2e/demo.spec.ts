import { expect, test } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import path from 'node:path';

const VALID_WORKER_TOKEN = 'ug_worker_staging_sec_1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d';
const EVIDENCE_DIR = path.resolve('..', 'URBANGRAPH_LOCALHOST_FINAL', 'test-evidence', 'screenshots');
const ACCEPTANCE_DIR = path.resolve('..', 'acceptance', 'screenshots');

test('01: RBAC 401 enforcement and interactive authentication', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));

  // 1. Navigate without token
  await page.goto('/');

  // 2. Verify 401 alert and uncredentialed state
  await expect(page.getByRole('alert')).toBeVisible();
  await expect(page.getByText(/Unable to load this view/i)).toBeVisible();
  await expect(page.getByText(/Bearer token authorization/i)).toBeVisible();

  // Capture uncredentialed 401 enforcement screenshot
  await page.screenshot({ path: path.join(EVIDENCE_DIR, '01_login_401_enforcement.png') });
  await page.screenshot({ path: path.join(ACCEPTANCE_DIR, '01_login_401_enforcement.png') });

  // 3. Authenticate session via token injection and retry
  await page.evaluate((tok) => {
    localStorage.setItem('urbangraph_auth_token', tok);
  }, VALID_WORKER_TOKEN);

  const retryBtn = page.getByRole('button', { name: /Retry/i });
  await expect(retryBtn).toBeVisible();
  await retryBtn.click();

  // 4. Verify successful authentication and dashboard hydration
  await expect(page.getByRole('heading', { name: 'Day-ahead outlook' })).toBeVisible({ timeout: 10_000 });
  await expect(page.getByText(/ENERGY SYSTEMS \/ RESEARCH INTERFACE/i)).toBeVisible();
  await expect(page.getByText(/RESEARCH DEMONSTRATION/i)).toBeVisible();

  // Capture hydrated dashboard screenshot
  await page.screenshot({ path: path.join(EVIDENCE_DIR, '02_dashboard_forecast.png') });
  await page.screenshot({ path: path.join(ACCEPTANCE_DIR, '02_dashboard_forecast.png') });

  expect(errors).toEqual([]);
});

test('02: synthetic forecast, graph, and battery journey', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));

  // Pre-seed authenticated localStorage session
  await page.addInitScript((token) => {
    localStorage.setItem('urbangraph_auth_token', token);
  }, VALID_WORKER_TOKEN);

  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Day-ahead outlook' })).toBeVisible({ timeout: 10_000 });
  await expect(page.getByRole('img', { name: /forecast versus truth/i })).toBeVisible();
  await expect(page.getByText('Uncalibrated band')).toBeVisible();
  await expect(page.getByRole('img', { name: /schematic links/i })).toBeVisible();
  await expect(page.getByText('invented battery capacity')).toBeVisible();

  const audit = await new AxeBuilder({ page }).exclude('.simulation-panel').analyze();
  expect(audit.violations.filter(item => ['critical', 'serious'].includes(item.impact ?? ''))).toEqual([]);

  await page.getByRole('combobox', { name: 'REGION' }).selectOption('SYN_B');
  await expect(page.getByRole('combobox', { name: 'REGION' })).toHaveValue('SYN_B');
  await expect(page.getByText(/No battery example for this region/)).toBeVisible();

  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(2);
  expect(errors).toEqual([]);
});

test('03: geospatial command center interactive journey', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));

  await page.addInitScript((token) => {
    localStorage.setItem('urbangraph_auth_token', token);
  }, VALID_WORKER_TOKEN);

  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Day-ahead outlook' })).toBeVisible({ timeout: 10_000 });

  // Verify Geospatial Command Center is mounted and visible
  await expect(page.getByText('R22 ACTIVE COMMAND CENTER')).toBeVisible();
  await expect(page.getByText('Geospatial Transmission & Regional Load Grid')).toBeVisible();

  // Verify interactive outage injection
  const dropBtn = page.getByRole('button', { name: 'Drop SYN_A' });
  await expect(dropBtn).toBeVisible();
  await dropBtn.click();

  // Restore sensor button should now appear
  const restoreBtn = page.getByRole('button', { name: 'Restore SYN_A' });
  await expect(restoreBtn).toBeVisible();
  await restoreBtn.click();

  // Verify it restored back to Drop SYN_A
  await expect(page.getByRole('button', { name: 'Drop SYN_A' })).toBeVisible();

  // Accessibility audit on command center
  const audit = await new AxeBuilder({ page }).include('.geospatial-command-center').analyze();
  expect(audit.violations.filter(item => ['critical', 'serious'].includes(item.impact ?? ''))).toEqual([]);
  expect(errors).toEqual([]);
});

test('04: deck.gl webgl, maplibre vector style, and svg fallback toggling', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));

  await page.addInitScript((token) => {
    localStorage.setItem('urbangraph_auth_token', token);
  }, VALID_WORKER_TOKEN);

  await page.goto('/');
  await expect(page.getByText('R22 ACTIVE COMMAND CENTER')).toBeVisible({ timeout: 10_000 });

  // Find the renderer toggle button
  const toggleBtn = page.getByRole('button', { name: /DECK\.GL|MAPLIBRE|SVG\/CANVAS/i });
  await expect(toggleBtn).toBeVisible();

  const initialText = await toggleBtn.textContent();

  if (initialText?.includes('DECK.GL')) {
    // 1. WebGL is active: verify Deck.gl canvas
    await expect(page.locator('#deckgl-overlay-canvas')).toBeVisible();

    // 2. Toggle to MapLibre offline vector style
    await toggleBtn.click();
    await expect(page.getByRole('button', { name: /MAPLIBRE \(VECTOR SPEC\)/i })).toBeVisible();
    await expect(page.locator('#maplibre-offline-container')).toBeVisible();
    await page.waitForTimeout(500);

    await page.screenshot({ path: path.join(EVIDENCE_DIR, '03_geospatial_maplibre.png') });
    await page.screenshot({ path: path.join(ACCEPTANCE_DIR, '03_geospatial_maplibre.png') });

    // 3. Toggle to SVG/Canvas fallback
    await toggleBtn.click();
    await expect(page.getByRole('button', { name: /SVG\/CANVAS \(FALLBACK\)/i })).toBeVisible();
    await expect(page.locator('svg.geo-svg')).toBeVisible();

    // 4. Toggle back to WebGL
    await toggleBtn.click();
    await expect(page.getByRole('button', { name: /DECK\.GL \(WEBGL\)/i })).toBeVisible();
    await expect(page.locator('#deckgl-overlay-canvas')).toBeVisible();
  } else {
    await expect(page.locator('svg.geo-svg, #maplibre-offline-container')).toBeVisible();
    await page.screenshot({ path: path.join(EVIDENCE_DIR, '03_geospatial_maplibre.png') });
    await page.screenshot({ path: path.join(ACCEPTANCE_DIR, '03_geospatial_maplibre.png') });
  }

  expect(errors).toEqual([]);
});

test('05: grid simulation lab (dc power flow & highs milp optimization)', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));

  await page.addInitScript((token) => {
    localStorage.setItem('urbangraph_auth_token', token);
  }, VALID_WORKER_TOKEN);

  await page.goto('/');
  await expect(page.getByText('DC Power Flow & HiGHS Storage Engine')).toBeVisible({ timeout: 10_000 });

  // 1. Test DC Power Flow with Contingency
  const outageBtn = page.getByRole('button', { name: /SYN_A-SYN_B/i });
  await expect(outageBtn).toBeVisible();
  await outageBtn.click(); // Outage line

  const simBtn = page.getByRole('button', { name: /EXECUTE REAL GRID SIMULATION/i });
  await expect(simBtn).toBeVisible();
  await simBtn.click();

  // Verify power flow convergence and results display
  await expect(page.getByText(/GRID CONVERGED & FEASIBLE|GRID CONSTRAINT VIOLATIONS/i)).toBeVisible({ timeout: 10_000 });
  await expect(page.getByText('TRANSMISSION BRANCH FLOWS & THERMAL LIMITS')).toBeVisible();

  await page.screenshot({ path: path.join(EVIDENCE_DIR, '04_dc_power_flow_simulation.png') });
  await page.screenshot({ path: path.join(ACCEPTANCE_DIR, '04_dc_power_flow_simulation.png') });

  // 2. Restore line contingency and test HiGHS MILP Battery Optimization
  await outageBtn.click(); // Restore line to IN SERVICE

  const scenarioSelect = page.locator('select').filter({ hasText: /DC Power Flow/i });
  await scenarioSelect.selectOption('battery_optimization');

  await expect(page.getByText(/BESS CAPACITY RATIO/i)).toBeVisible();
  await expect(page.getByText(/ROUNDTRIP EFFICIENCY/i)).toBeVisible();

  await simBtn.click();

  await expect(page.getByText(/GRID CONVERGED & FEASIBLE/i)).toBeVisible({ timeout: 10_000 });
  await expect(page.getByText(/HiGHS MILP STORAGE ARBITRAGE FINANCIAL REALIZATION/i)).toBeVisible();
  await expect(page.getByText(/Net Economic Benefit:/i)).toBeVisible();

  await page.screenshot({ path: path.join(EVIDENCE_DIR, '05_highs_battery_optimization.png') });
  await page.screenshot({ path: path.join(ACCEPTANCE_DIR, '05_highs_battery_optimization.png') });

  expect(errors).toEqual([]);
});
