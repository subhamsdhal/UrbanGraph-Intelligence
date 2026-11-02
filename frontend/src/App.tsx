import { useEffect, useState } from 'react';
import { api, ApiError, setAuthToken, type Dispatch, type ForecastPoint, type Graph, type Meta, type Region, type SeriesPoint } from './api';
import { GeospatialCommandCenter } from './components/GeospatialCommandCenter';
import { GridSimulationLab } from './components/GridSimulationLab';

function line(values: number[], minimum: number, maximum: number): string {
  const range = maximum - minimum || 1;
  return values.map((value, index) => `${32 + index * 676 / Math.max(values.length - 1, 1)},${198 - (value - minimum) * 150 / range}`).join(' ');
}

function ForecastChart({ points }: {points: ForecastPoint[]}) {
  const values = points.flatMap(point => [point.lower_MW, point.upper_MW, point.point_MW, point.truth_MW]);
  const min = Math.floor(Math.min(...values) - 5);
  const max = Math.ceil(Math.max(...values) + 5);
  const upper = line(points.map(point => point.upper_MW), min, max).split(' ');
  const lower = line(points.map(point => point.lower_MW), min, max).split(' ').reverse();
  return <div className="chart-wrap">
    <svg viewBox="0 0 740 230" role="img" aria-label="Twenty-four hour synthetic forecast versus truth in megawatts with illustrative uncalibrated interval">
      <line x1="32" y1="198" x2="708" y2="198" className="axis" />
      <line x1="32" y1="48" x2="32" y2="198" className="axis" />
      <polygon points={[...upper, ...lower].join(' ')} className="band" />
      <polyline points={line(points.map(point => point.point_MW), min, max)} className="forecast-line" />
      <polyline points={line(points.map(point => point.truth_MW), min, max)} className="truth-line" />
      <text x="32" y="220">00:00 UTC</text><text x="646" y="220">23:00 UTC</text>
      <text x="38" y="39">{max} MW</text><text x="38" y="190">{min} MW</text>
    </svg>
    <div className="legend"><span className="key truth">Truth</span><span className="key predicted">Illustrative forecast</span><span className="key uncertainty">Uncalibrated band</span></div>
    <p className="chart-summary">Forecast range {Math.min(...points.map(p => p.point_MW)).toFixed(0)}–{Math.max(...points.map(p => p.point_MW)).toFixed(0)} MW. Truth range {Math.min(...points.map(p => p.truth_MW)).toFixed(0)}–{Math.max(...points.map(p => p.truth_MW)).toFixed(0)} MW.</p>
  </div>;
}

function HistoryChart({ points }: {points: SeriesPoint[]}) {
  const values = points.map(point => point.load_MW);
  const min = Math.floor(Math.min(...values) - 5);
  const max = Math.ceil(Math.max(...values) + 5);
  return <div className="chart-wrap"><svg viewBox="0 0 740 230" role="img" aria-label="Seven days of synthetic hourly energy load in megawatts">
    <line x1="32" y1="198" x2="708" y2="198" className="axis" />
    <polyline points={line(values, min, max)} className="history-line" />
    <text x="32" y="220">7 days before origin</text><text x="630" y="220">Origin UTC</text>
    <text x="38" y="39">{max} MW</text><text x="38" y="190">{min} MW</text>
  </svg><p className="chart-summary">{points.length} hourly observations. Range {Math.min(...values).toFixed(0)}–{Math.max(...values).toFixed(0)} MW.</p></div>;
}

function Network({ graph, selected }: {graph: Graph; selected: string}) {
  const nodes = new Map(graph.regions.map(region => [region.id, region]));
  return <div className="network-wrap"><svg viewBox="0 0 440 250" role="img" aria-label="Schematic links among three invented regions; not real electrical topology">
    {graph.edges.map(edge => { const from = nodes.get(edge.source)!; const to = nodes.get(edge.target)!; return <g key={`${edge.source}-${edge.target}`}>
      <line x1={40 + from.x * 350} y1={25 + from.y * 190} x2={40 + to.x * 350} y2={25 + to.y * 190} className="edge" />
      <text x={40 + (from.x + to.x) * 175} y={20 + (from.y + to.y) * 95}>{edge.weight.toFixed(1)}</text>
    </g>; })}
    {graph.regions.map(region => <g key={region.id}>
      <circle cx={40 + region.x * 350} cy={25 + region.y * 190} r="26" className={region.id === selected ? 'node selected' : 'node'} />
      <text x={40 + region.x * 350} y={30 + region.y * 190} textAnchor="middle" className="node-label">{region.id.replace('SYN_', '')}</text>
    </g>)}
  </svg><p>Fictional unit-square locations and hand-set display links. These are not electrical grid connections.</p></div>;
}

function BatteryView({ scenario }: {scenario: Dispatch}) {
  const first = scenario.points[0];
  const last = scenario.points[scenario.points.length - 1];
  return <div className="battery-grid">
    <div className="battery-hero"><strong>{scenario.battery.capacity_MWh.toFixed(0)} <small>MWh</small></strong><span>invented battery capacity</span></div>
    <div className="battery-metrics"><div><span>Power limit</span><strong>{scenario.battery.power_limit_MW.toFixed(0)} MW</strong></div><div><span>State of charge</span><strong>{first.soc_MWh_end.toFixed(1)} → {last.soc_MWh_end.toFixed(1)} MWh</strong></div><div><span>Tariff</span><strong>synthetic CU/MWh</strong></div></div>
    <div className="battery-timeline" aria-label="Twenty-four hour illustrative charge and discharge schedule">
      {scenario.points.map((point, index) => <div className="battery-hour" key={`${point.target_utc}-${index}`} title={`${point.target_utc}: charge ${point.charge_MW} MW; discharge ${point.discharge_MW} MW; SoC ${point.soc_MWh_end} MWh`}>
        <span className="hour-bar" style={{height: `${Math.max(point.charge_MW, point.discharge_MW) * 2 + 3}px`, background: point.charge_MW > 0 ? 'var(--teal)' : point.discharge_MW > 0 ? 'var(--amber)' : 'var(--muted)'}} />
        {index % 6 === 0 && <small>{String(index).padStart(2, '0')}</small>}
      </div>)}
    </div><p className="fine-print">Checked fixture schedule only. Synthetic prices, no real asset, no optimization or savings claim.</p>
  </div>;
}

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [regions, setRegions] = useState<Region[]>([]);
  const [graph, setGraph] = useState<Graph | null>(null);
  const [region, setRegion] = useState('');
  const [origin, setOrigin] = useState('');
  const [forecast, setForecast] = useState<ForecastPoint[]>([]);
  const [history, setHistory] = useState<SeriesPoint[]>([]);
  const [dispatch, setDispatch] = useState<Dispatch | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectionLoading, setSelectionLoading] = useState(false);
  const [error, setError] = useState('');
  const [retry, setRetry] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setError('');
    Promise.all([api.meta(controller.signal), api.regions(controller.signal), api.graph(controller.signal)])
      .then(([m, r, g]) => { setMeta(m); setRegions(r.regions); setGraph(g); setRegion(r.regions[0]?.id ?? ''); setOrigin(r.regions[0]?.origins_utc[0] ?? ''); })
      .catch(cause => { if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : 'Could not load synthetic data'); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [retry]);

  useEffect(() => {
    if (!region || !origin) return;
    const controller = new AbortController();
    setSelectionLoading(true); setError(''); setForecast([]); setHistory([]); setDispatch(null);
    const end = new Date(origin);
    const start = new Date(end.getTime() - 167 * 60 * 60 * 1000);
    Promise.all([
      api.forecasts(region, origin, controller.signal),
      api.series(region, start.toISOString(), end.toISOString(), controller.signal),
      api.dispatch(region, origin, controller.signal).catch(cause => {
        if (cause instanceof ApiError && cause.status === 404) return null;
        throw cause;
      }),
    ]).then(([f, s, d]) => { setForecast(f.points); setHistory(s.points); setDispatch(d); })
      .catch(cause => { if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : 'Could not load selection'); })
      .finally(() => { if (!controller.signal.aborted) setSelectionLoading(false); });
    return () => controller.abort();
  }, [region, origin, retry]);

  const selected = regions.find(item => item.id === region);
  return <div className="app-shell">
    <header className="topbar"><div className="brand"><span className="brand-mark">E</span><span>EnerGraph <em>AI</em></span></div><span className="top-status"><i /> LOCAL SYNTHETIC DEMO</span></header>
    <main>
      <section className="hero"><div><p className="eyebrow">ENERGY SYSTEMS / RESEARCH INTERFACE</p><h1>See the shape of<br /><span>tomorrow’s demand.</span></h1><p className="hero-copy">Explore a reproducible, fictional energy network. Forecasts, uncertainty and storage are illustrations—not live grid intelligence.</p></div><div className="hero-index"><span>01 / 03</span><strong>Energy<br />observatory</strong><small>UTC · MW · SYNTHETIC CU</small></div></section>
      <section className="notice" aria-label="Synthetic data disclaimer"><strong>RESEARCH DEMONSTRATION</strong><span>All regions and measurements are invented. Intervals are uncalibrated. No real-world performance or financial result is implied.</span></section>
      {loading && <div role="status" className="state">Loading verified synthetic fixtures…</div>}
      {error && (
        <div role="alert" className="state error">
          <strong>Unable to load this view.</strong> {error}
          {error.includes("401") || error.toLowerCase().includes("unauthorized") ? (
            <div style={{ marginTop: '8px', display: 'flex', gap: '8px', alignItems: 'center' }}>
              <input
                type="password"
                placeholder="Enter Bearer Token"
                id="auth-token-input"
                style={{ padding: '6px 10px', borderRadius: '4px', border: '1px solid #73a5af', background: '#0a1d27', color: '#fff' }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    setAuthToken((e.target as HTMLInputElement).value);
                    setRetry((v) => v + 1);
                  }
                }}
              />
              <button
                type="button"
                className="btn-command"
                onClick={() => {
                  const input = document.getElementById('auth-token-input') as HTMLInputElement;
                  if (input) {
                    setAuthToken(input.value);
                    setRetry((v) => v + 1);
                  }
                }}
              >
                Authenticate & Retry
              </button>
            </div>
          ) : (
            <button onClick={() => setRetry((value) => value + 1)}>Retry</button>
          )}
        </div>
      )}
      {meta && regions.length > 0 && <>
        <section className="controls" aria-label="Scenario selection"><div><label htmlFor="region">REGION</label><select id="region" value={region} onChange={event => { const next = regions.find(item => item.id === event.target.value); setRegion(event.target.value); setOrigin(next?.origins_utc[0] ?? ''); }}>{regions.map(item => <option key={item.id} value={item.id}>{item.label} · {item.id}</option>)}</select></div><div><label htmlFor="origin">FORECAST ORIGIN</label><select id="origin" value={origin} onChange={event => setOrigin(event.target.value)}>{selected?.origins_utc.map(item => <option key={item} value={item}>{item.replace('T', ' · ').replace('Z', ' UTC')}</option>)}</select></div><span className="controls-note">24-hour horizon<br />7 origin dates per region</span></section>
        {selectionLoading && <div role="status" className="state">Loading forecast and regional history…</div>}
        <GeospatialCommandCenter
          graph={graph}
          selectedRegionId={region}
          originUtc={origin}
          onSelectRegion={(newId) => {
            const next = regions.find((item) => item.id === newId);
            setRegion(newId);
            if (next?.origins_utc[0]) setOrigin(next.origins_utc[0]);
          }}
          forecastPoints={forecast}
          regions={regions}
        />
        <div className="dashboard-grid">
          <section className="panel forecast-panel"><div className="panel-head"><div><p className="eyebrow">01 / FORECAST</p><h2>Day-ahead outlook</h2></div><span className="panel-tag">24 HOURS · MW</span></div><p className="panel-desc">Point forecast versus synthetic truth. The shaded range is an illustrative, uncalibrated interval.</p>{forecast.length ? <ForecastChart points={forecast} /> : !selectionLoading && <p className="empty">No forecast available for this selection.</p>}</section>
          <section className="panel history-panel"><div className="panel-head"><div><p className="eyebrow">02 / HISTORY</p><h2>Regional load</h2></div><span className="panel-tag">HOURLY · UTC</span></div><p className="panel-desc">Seven days of invented observations preceding the selected origin.</p>{history.length ? <HistoryChart points={history} /> : !selectionLoading && <p className="empty">No hourly observations available.</p>}</section>
          <section className="panel graph-panel"><div className="panel-head"><div><p className="eyebrow">03 / NETWORK</p><h2>Regional connections</h2></div><span className="panel-tag">SCHEMATIC</span></div>{graph && <Network graph={graph} selected={region} />}</section>
          <section className="panel dispatch-panel"><div className="panel-head"><div><p className="eyebrow">04 / STORAGE</p><h2>Battery scenario</h2></div><span className="panel-tag">ILLUSTRATIVE</span></div>{dispatch ? <BatteryView scenario={dispatch} /> : <p className="empty">No battery example for this region and origin. The saved illustration covers one fictional scenario only.</p>}</section>
        </div>
        <GridSimulationLab selectedRegionId={region} />
        <section className="limitations"><span>INTERPRETATION NOTES</span><p>Historical EG-004 intervals undercovered in a later rolling fold. These fixture bands are not derived from that calibration and are not certified risk bounds. The tariff and asset here are fictional; the application cannot control equipment or estimate real savings.</p></section>
        <footer><span>URBANGRAPH INTELLIGENCE · ENERGRAPH AI</span><span>FIXTURE SHA-256 {meta.manifest_sha256.slice(0, 12)}… · {meta.dataset_kind.toUpperCase()}</span></footer>
      </>}
    </main>
  </div>;
}
