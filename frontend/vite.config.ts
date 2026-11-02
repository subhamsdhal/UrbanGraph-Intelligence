import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

declare const process: { env: Record<string, string | undefined> };

const apiPort = Number(process.env.ENERGRAPH_DEMO_API_PORT ?? '8000');
if (!Number.isInteger(apiPort) || apiPort < 1 || apiPort > 65535) {
  throw new Error('ENERGRAPH_DEMO_API_PORT must be a valid TCP port');
}
const apiTarget = `http://127.0.0.1:${apiPort}`;

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    proxy: { '/api': apiTarget, '/health': apiTarget },
  },
  test: { environment: 'jsdom', setupFiles: ['./tests/setup.ts'], include: ['tests/*.test.tsx'] },
});
