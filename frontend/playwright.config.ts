import { defineConfig, devices } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(__dirname, '..');
const venvPy = path.join(repo, '.venv', 'Scripts', 'python.exe');
const python = process.env.EG006_PYTHON ?? (process.platform === 'win32' && fs.existsSync(venvPy) ? venvPy : 'python');
const packagePath = path.join(repo, 'tests', 'fixtures', 'eg006');
function port(name: string, fallback: number): number {
  const value = Number(process.env[name] ?? fallback);
  if (!Number.isInteger(value) || value < 1 || value > 65535) throw new Error(`${name} must be a valid TCP port`);
  return value;
}
const apiPort = port('EG006_API_PORT', 8000);
const baseUrl = process.env.PLAYWRIGHT_BASE_URL ?? `http://127.0.0.1:${apiPort}`;

const cachedChromium = path.join(
  process.env.LOCALAPPDATA ?? (process.env.USERPROFILE ? path.join(process.env.USERPROFILE, 'AppData', 'Local') : '/tmp'),
  'ms-playwright',
  'chromium-1243',
  'chrome-win64',
  'chrome.exe'
);
const localBrowserPath = fs.existsSync(cachedChromium) ? cachedChromium : undefined;

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 30_000,
  use: {
    baseURL: baseUrl,
    trace: 'retain-on-failure',
    ...(localBrowserPath ? { launchOptions: { executablePath: localBrowserPath } } : {}),
  },
  projects: [
    { name: 'desktop-chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'mobile-chromium', use: { ...devices['Pixel 7'] } },
  ],
  webServer: process.env.EG006_EXTERNAL_SERVERS ? undefined : {
    command: `"${python}" -B -m urbangraph.production_server --host 127.0.0.1 --port ${apiPort}`,
    cwd: repo,
    url: `http://127.0.0.1:${apiPort}/health`,
    reuseExistingServer: true,
    env: {
      PYTHONPATH: path.join(repo, 'src'),
      ENERGRAPH_DEMO_PACKAGE: packagePath,
      ENERGRAPH_STATIC_DIR: path.join(__dirname, 'dist'),
      PYTHONDONTWRITEBYTECODE: '1',
    },
  },
});
