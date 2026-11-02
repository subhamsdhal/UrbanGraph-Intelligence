import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../src/App';

const origins = Array.from({ length: 7 }, (_, day) => new Date(Date.UTC(2030, 2, 31 + day, 23)).toISOString().replace('.000Z', 'Z'));
const regions = ['SYN_A', 'SYN_B', 'SYN_C'].map(id => ({ id, label: `Fictional ${id.slice(-1)}`, origins_utc: origins }));
const base = { schema_version: 'eg006-demo-v1', dataset_kind: 'synthetic', manifest_sha256: 'a'.repeat(64) };
const responses: Record<string, unknown> = {
  '/api/v1/meta': { ...base, regions: 3, origins_per_region: 7, first_utc: '2030-01-01T00:00:00Z', last_utc: '2030-04-10T23:00:00Z', disclaimer: 'Invented' },
  '/api/v1/regions': { ...base, regions },
  '/api/v1/graph': { ...base, coordinate_system: 'synthetic_unit_square', meaning: 'invented', regions: regions.map((r, i) => ({ id: r.id, label: r.label, x: .2 + i * .3, y: .4 })), edges: [{source: 'SYN_A', target: 'SYN_B', weight: .7}] },
  '/api/v1/series': { ...base, points: Array.from({length: 168}, (_, i) => ({timestamp_utc: new Date(Date.UTC(2030, 2, 24) + i * 3600000).toISOString(), load_MW: 180 + i % 24})), count: 168 },
  '/api/v1/forecasts': { ...base, interval_kind: 'illustrative_uncalibrated', points: Array.from({length: 24}, (_, i) => ({horizon_hours: i + 1, target_utc: '2030-04-01T00:00:00Z', point_MW: 200 + i, truth_MW: 201 + i, lower_MW: 190 + i, upper_MW: 210 + i})) },
  '/api/v1/dispatch': { ...base, battery: {capacity_MWh: 100, power_limit_MW: 50, initial_terminal_soc_MWh: 50, roundtrip_efficiency: .9}, points: Array.from({length:24}, (_, i) => ({target_utc: '2030-04-01T00:00:00Z', load_MW: 200, import_MW: 200, charge_MW: 0, discharge_MW: 0, soc_MWh_end: 50, price_synthetic_CU_per_MWh: 35, hour: i})) },
};

describe('synthetic observatory', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn(async (path: string) => {
      const key = path.split('?')[0];
      return { ok: true, json: async () => responses[key] };
    }));
  });
  afterEach(() => vi.unstubAllGlobals());

  it('labels synthetic forecasting, uncertainty, graph and battery views', async () => {
    render(<App />);
    expect(await screen.findByText('Day-ahead outlook')).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText('Battery scenario')).toBeInTheDocument());
    expect(screen.getAllByText(/uncalibrated/i).length).toBeGreaterThan(0);
    expect(screen.getByRole('img', {name: /schematic links/i})).toBeInTheDocument();
    expect(await screen.findByRole('img', {name: /forecast versus truth/i})).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('REGION'), {target: {value: 'SYN_B'}});
    expect((screen.getByLabelText('REGION') as HTMLSelectElement).value).toBe('SYN_B');
  });

  it('shows an actionable error and retry', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => { throw new Error('API offline'); }));
    render(<App />);
    expect(await screen.findByRole('alert')).toHaveTextContent('API offline');
    expect(screen.getByRole('button', {name: 'Retry'})).toBeInTheDocument();
  });
});
