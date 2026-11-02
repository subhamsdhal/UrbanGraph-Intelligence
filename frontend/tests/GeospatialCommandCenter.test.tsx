import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { GeospatialCommandCenter } from '../src/components/GeospatialCommandCenter';
import type { ForecastPoint, Region } from '../src/api';

const mockRegions: Region[] = [
  { id: 'SYN_A', label: 'Region A', origins_utc: ['2030-04-01T00:00:00Z'] },
  { id: 'SYN_B', label: 'Region B', origins_utc: ['2030-04-01T00:00:00Z'] },
  { id: 'SYN_C', label: 'Region C', origins_utc: ['2030-04-01T00:00:00Z'] },
];

const mockForecast: ForecastPoint[] = Array.from({ length: 24 }, (_, i) => ({
  horizon_hours: i + 1,
  target_utc: `2030-04-01T${String(i).padStart(2, '0')}:00:00Z`,
  point_MW: 210 + i * 2,
  truth_MW: 212 + i * 2,
  lower_MW: 190 + i * 2,
  upper_MW: 230 + i * 2,
}));

describe('GeospatialCommandCenter', () => {
  afterEach(() => {
    cleanup();
  });

  it('renders the command center header and geographic map', () => {
    const onSelect = vi.fn();
    render(
      <GeospatialCommandCenter
        graph={null}
        selectedRegionId="SYN_A"
        onSelectRegion={onSelect}
        forecastPoints={mockForecast}
        regions={mockRegions}
      />
    );

    expect(screen.getByText('R22 ACTIVE COMMAND CENTER')).toBeInTheDocument();
    expect(screen.getByText('Geospatial Transmission & Regional Load Grid')).toBeInTheDocument();
    expect(screen.getByRole('img', { name: /Interactive Geographic Transmission Map/i })).toBeInTheDocument();
  });

  it('allows scrubbing through forecast horizons', () => {
    const onSelect = vi.fn();
    render(
      <GeospatialCommandCenter
        graph={null}
        selectedRegionId="SYN_A"
        onSelectRegion={onSelect}
        forecastPoints={mockForecast}
        regions={mockRegions}
      />
    );

    const slider = screen.getByRole('slider');
    fireEvent.change(slider, { target: { value: '12' } });

    expect(screen.getByText(/\+12h/)).toBeInTheDocument();
  });

  it('toggles sensor outage simulation and updates uncertainty', () => {
    const onSelect = vi.fn();
    render(
      <GeospatialCommandCenter
        graph={null}
        selectedRegionId="SYN_A"
        onSelectRegion={onSelect}
        forecastPoints={mockForecast}
        regions={mockRegions}
      />
    );

    const dropBtn = screen.getByRole('button', { name: 'Drop SYN_A' });
    fireEvent.click(dropBtn);

    expect(screen.getByRole('button', { name: 'Restore SYN_A' })).toBeInTheDocument();
    expect(screen.getByText(/⚠ SENSOR DROPOUT/)).toBeInTheDocument();
  });
});
