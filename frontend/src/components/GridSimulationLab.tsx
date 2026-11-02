import React, { useState } from 'react';
import { api, type SimulationRequest, type SimulationResponse } from '../api';

interface Props {
  selectedRegionId: string;
}

export const GridSimulationLab: React.FC<Props> = ({ selectedRegionId }) => {
  const [scenarioType, setScenarioType] = useState<'dc_power_flow' | 'battery_optimization'>('dc_power_flow');
  const [loadScale, setLoadScale] = useState<number>(1.0);
  const [lineOutages, setLineOutages] = useState<string[]>([]);
  const [batteryCapacityRatio, setBatteryCapacityRatio] = useState<number>(0.10);
  const [batteryEfficiency, setBatteryEfficiency] = useState<number>(0.90);
  const [tariffPeak, setTariffPeak] = useState<number>(15.0);
  const [tariffDegradation, setTariffDegradation] = useState<number>(5.0);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SimulationResponse | null>(null);

  const availableLines = ['SYN_A-SYN_B', 'SYN_B-SYN_C', 'SYN_A-SYN_C'];

  const toggleLineOutage = (line: string) => {
    setLineOutages((prev) =>
      prev.includes(line) ? prev.filter((l) => l !== line) : [...prev, line]
    );
  };

  const runSimulation = async () => {
    setLoading(true);
    setError(null);
    try {
      const payload: SimulationRequest = {
        scenario_type: scenarioType,
        region_id: selectedRegionId,
        load_scale: loadScale,
        line_outages: lineOutages,
        battery_capacity_ratio: batteryCapacityRatio,
        battery_efficiency: batteryEfficiency,
        tariff_peak: tariffPeak,
        tariff_degradation: tariffDegradation,
      };
      const resp = await api.simulate(payload);
      setResult(resp);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Simulation failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="panel simulation-panel" style={{ marginTop: '20px' }}>
      <div className="panel-head">
        <div>
          <p className="eyebrow">GRID SIMULATION & BATTERY OPTIMIZATION LAB</p>
          <h2>DC Power Flow & HiGHS Storage Engine</h2>
        </div>
        <span className="panel-tag">MILP OPTIMIZER · KIRCHHOFF LAW</span>
      </div>
      <p className="panel-desc">
        Evaluate transmission branch thermal loadings under contingency scenarios using genuine DC power flow equations,
        or solve multi-period battery state-of-charge arbitrage using the HiGHS Mixed-Integer Linear Programming (MILP) solver.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '16px', background: '#091c26', padding: '16px', borderRadius: '6px', margin: '16px 0' }}>
        <div>
          <label style={{ fontSize: '11px', fontWeight: 'bold', color: 'var(--teal)', display: 'block', marginBottom: '6px' }}>SCENARIO TYPE</label>
          <select
            value={scenarioType}
            onChange={(e) => setScenarioType(e.target.value as 'dc_power_flow' | 'battery_optimization')}
            style={{ width: '100%', padding: '8px', background: '#102b39', color: '#fff', border: '1px solid #34505d', borderRadius: '4px' }}
          >
            <option value="dc_power_flow">DC Power Flow (Thermal & Islanding)</option>
            <option value="battery_optimization">HiGHS MILP Battery Optimization</option>
          </select>
        </div>

        <div>
          <label style={{ fontSize: '11px', fontWeight: 'bold', color: 'var(--teal)', display: 'block', marginBottom: '6px' }}>
            REGIONAL LOAD SCALE: {(loadScale * 100).toFixed(0)}%
          </label>
          <input
            type="range"
            min="0.5"
            max="2.5"
            step="0.1"
            value={loadScale}
            onChange={(e) => setLoadScale(parseFloat(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--teal)' }}
          />
        </div>

        <div>
          <label style={{ fontSize: '11px', fontWeight: 'bold', color: 'var(--teal)', display: 'block', marginBottom: '6px' }}>
            TRANSMISSION LINE CONTINGENCIES
          </label>
          <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
            {availableLines.map((line) => {
              const active = lineOutages.includes(line);
              return (
                <button
                  key={line}
                  type="button"
                  onClick={() => toggleLineOutage(line)}
                  style={{
                    padding: '4px 8px',
                    fontSize: '10px',
                    borderRadius: '4px',
                    border: active ? '1px solid #f07167' : '1px solid #34505d',
                    background: active ? '#4d1d19' : '#102b39',
                    color: active ? '#ffcbc7' : '#a2bdc7',
                    cursor: 'pointer',
                  }}
                >
                  {active ? `⚠ OUTAGE: ${line}` : `IN SERVICE: ${line}`}
                </button>
              );
            })}
          </div>
        </div>

        {scenarioType === 'battery_optimization' && (
          <>
            <div>
              <label style={{ fontSize: '11px', fontWeight: 'bold', color: 'var(--amber)', display: 'block', marginBottom: '6px' }}>
                BESS CAPACITY RATIO: {(batteryCapacityRatio * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0.05"
                max="0.40"
                step="0.05"
                value={batteryCapacityRatio}
                onChange={(e) => setBatteryCapacityRatio(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--amber)' }}
              />
            </div>
            <div>
              <label style={{ fontSize: '11px', fontWeight: 'bold', color: 'var(--amber)', display: 'block', marginBottom: '6px' }}>
                ROUNDTRIP EFFICIENCY: {(batteryEfficiency * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0.70"
                max="0.98"
                step="0.02"
                value={batteryEfficiency}
                onChange={(e) => setBatteryEfficiency(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--amber)' }}
              />
            </div>
          </>
        )}
      </div>

      <div style={{ display: 'flex', gap: '12px', alignItems: 'center', marginBottom: '16px' }}>
        <button
          type="button"
          onClick={runSimulation}
          disabled={loading}
          className="btn-command"
          style={{ padding: '10px 20px', fontSize: '12px', background: 'var(--teal)', color: '#04151f', border: 'none', cursor: 'pointer' }}
        >
          {loading ? 'SOLVING POWER FLOW & DISPATCH...' : '⚡ EXECUTE REAL GRID SIMULATION'}
        </button>
        {result && (
          <span
            style={{
              padding: '6px 12px',
              borderRadius: '4px',
              fontSize: '11px',
              fontWeight: 'bold',
              background: result.feasible ? '#113438' : '#56312f',
              color: result.feasible ? 'var(--teal)' : '#ffcbc7',
              border: `1px solid ${result.feasible ? 'var(--teal)' : '#f07167'}`,
            }}
          >
            {result.feasible ? '✓ GRID CONVERGED & FEASIBLE' : '⚠ GRID CONSTRAINT VIOLATIONS DETECTED'}
          </span>
        )}
      </div>

      {error && (
        <div role="alert" className="state error" style={{ marginBottom: '16px' }}>
          <strong>Simulation Error:</strong> {error}
        </div>
      )}

      {result && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '16px' }}>
          {/* Branch Flows */}
          <div style={{ background: '#0a1d27', border: '1px solid #20414f', borderRadius: '6px', padding: '16px' }}>
            <h4 style={{ margin: '0 0 10px', fontSize: '14px', color: 'var(--teal)' }}>
              TRANSMISSION BRANCH FLOWS & THERMAL LIMITS
            </h4>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #294c5b', color: '#8faab6' }}>
                  <th style={{ padding: '6px 4px' }}>BRANCH</th>
                  <th style={{ padding: '6px 4px' }}>FLOW (MW)</th>
                  <th style={{ padding: '6px 4px' }}>RATING</th>
                  <th style={{ padding: '6px 4px' }}>LOADING</th>
                  <th style={{ padding: '6px 4px' }}>STATUS</th>
                </tr>
              </thead>
              <tbody>
                {result.branch_flows.map((b) => (
                  <tr key={b.branch} style={{ borderBottom: '1px solid #173340' }}>
                    <td style={{ padding: '6px 4px', fontWeight: 'bold' }}>{b.branch}</td>
                    <td style={{ padding: '6px 4px' }}>{b.flow_MW.toFixed(1)}</td>
                    <td style={{ padding: '6px 4px' }}>{b.rating_MW.toFixed(0)} MW</td>
                    <td style={{ padding: '6px 4px' }}>
                      <span style={{ color: b.overloaded ? '#f07167' : b.loading_pct > 80 ? 'var(--amber)' : '#56dbc4' }}>
                        {b.loading_pct.toFixed(1)}%
                      </span>
                    </td>
                    <td style={{ padding: '6px 4px' }}>
                      {b.overloaded ? (
                        <span style={{ color: '#f07167', fontWeight: 'bold' }}>OVERLOAD</span>
                      ) : (
                        <span style={{ color: '#56dbc4' }}>NORMAL</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Bus Phase Angles */}
          <div style={{ background: '#0a1d27', border: '1px solid #20414f', borderRadius: '6px', padding: '16px' }}>
            <h4 style={{ margin: '0 0 10px', fontSize: '14px', color: 'var(--teal)' }}>
              BUS VOLTAGE PHASE ANGLES (θ)
            </h4>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #294c5b', color: '#8faab6' }}>
                  <th style={{ padding: '6px 4px' }}>BUS ID</th>
                  <th style={{ padding: '6px 4px' }}>ANGLE (RAD)</th>
                  <th style={{ padding: '6px 4px' }}>ANGLE (DEG)</th>
                  <th style={{ padding: '6px 4px' }}>ROLE</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(result.bus_angles).map(([bus, rad]) => (
                  <tr key={bus} style={{ borderBottom: '1px solid #173340' }}>
                    <td style={{ padding: '6px 4px', fontWeight: 'bold' }}>{bus}</td>
                    <td style={{ padding: '6px 4px' }}>{rad.toFixed(4)}</td>
                    <td style={{ padding: '6px 4px' }}>{((rad * 180) / Math.PI).toFixed(2)}°</td>
                    <td style={{ padding: '6px 4px', color: rad === 0 ? 'var(--amber)' : '#8faab6' }}>
                      {rad === 0 ? 'Slack Bus' : 'Load Bus'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Violations / Realized Financials */}
          <div style={{ background: '#0a1d27', border: '1px solid #20414f', borderRadius: '6px', padding: '16px', gridColumn: '1 / -1' }}>
            {result.constraint_violations && result.constraint_violations.length > 0 && (
              <div style={{ marginBottom: '12px' }}>
                <h4 style={{ margin: '0 0 8px', fontSize: '13px', color: '#f07167' }}>
                  ⚠ DETECTED CONSTRAINT VIOLATIONS ({result.constraint_violations.length})
                </h4>
                <ul style={{ margin: 0, paddingLeft: '18px', color: '#ffcbc7', fontSize: '11px' }}>
                  {result.constraint_violations.map((v, i) => (
                    <li key={i}>{v}</li>
                  ))}
                </ul>
              </div>
            )}

            {result.realized_cost_CU && (
              <div>
                <h4 style={{ margin: '0 0 8px', fontSize: '13px', color: 'var(--amber)' }}>
                  ★ HiGHS MILP STORAGE ARBITRAGE FINANCIAL REALIZATION
                </h4>
                <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap', fontSize: '12px' }}>
                  <div>
                    <span style={{ color: '#8faab6' }}>Gross Energy Cost: </span>
                    <strong style={{ color: '#fff' }}>{result.realized_cost_CU.gross_energy_cost_CU?.toFixed(2)} CU</strong>
                  </div>
                  <div>
                    <span style={{ color: '#8faab6' }}>Arbitrage Savings: </span>
                    <strong style={{ color: 'var(--teal)' }}>{result.realized_cost_CU.arbitrage_savings_CU?.toFixed(2)} CU</strong>
                  </div>
                  <div>
                    <span style={{ color: '#8faab6' }}>Cell Degradation Cost: </span>
                    <strong style={{ color: '#f07167' }}>{result.realized_cost_CU.degradation_cost_CU?.toFixed(2)} CU</strong>
                  </div>
                  <div>
                    <span style={{ color: '#8faab6' }}>Net Economic Benefit: </span>
                    <strong style={{ color: 'var(--teal)' }}>{result.realized_cost_CU.net_economic_benefit_CU?.toFixed(2)} CU</strong>
                  </div>
                </div>
              </div>
            )}

            <p style={{ margin: '12px 0 0', fontSize: '10px', color: '#648391', fontStyle: 'italic' }}>
              {result.advisory_notice}
            </p>
          </div>
        </div>
      )}
    </section>
  );
};

