export type Identity = { schema_version: string; dataset_kind: 'synthetic'; manifest_sha256: string };
export type Region = { id: string; label: string; origins_utc: string[] };
export type Meta = Identity & { regions: number; origins_per_region: number; first_utc: string; last_utc: string; disclaimer: string };
export type SeriesPoint = { timestamp_utc: string; load_MW: number };
export type ForecastPoint = { horizon_hours: number; target_utc: string; point_MW: number; truth_MW: number; lower_MW: number; upper_MW: number };
export type Graph = Identity & { coordinate_system: string; meaning: string; regions: Array<{id: string; label: string; x: number; y: number}>; edges: Array<{source: string; target: string; weight: number}> };
export type DispatchPoint = { target_utc: string; load_MW: number; import_MW: number; charge_MW: number; discharge_MW: number; soc_MWh_end: number; price_synthetic_CU_per_MWh: number };
export type Dispatch = Identity & { region_id: string; origin_utc: string; battery: {capacity_MWh: number; power_limit_MW: number; initial_terminal_soc_MWh: number; roundtrip_efficiency: number}; points: DispatchPoint[]; feasibility: string; disclaimer: string };

export type OutageReport = {
  total_sensors: number;
  active_sensors: number;
  dropped_sensors: string[];
  dropout_ratio: number;
  degradation_level: 'NORMAL' | 'MODERATE_OUTAGE' | 'SEVERE_OUTAGE' | 'CRITICAL_ISOLATION';
  imputation_applied: string;
  uncertainty_inflation_factor: number;
  metadata?: Record<string, unknown>;
};

export type ResilientForecastPoint = {
  horizon_hours: number;
  target_utc: string;
  point_MW: number;
  truth_MW: number;
  lower_MW: number;
  upper_MW: number;
  raw_point_MW: number;
  uncertainty_inflation_factor: number;
  is_imputed: boolean;
};

export type ResilientPredictResponse = {
  schema_version: string;
  region_id: string;
  origin_utc: string;
  outage_report: OutageReport;
  points: ResilientForecastPoint[];
  drift_detected: boolean;
  drift_status: string;
  model_id: string;
};

export type DriftMonitoringStatus = {
  schema_version: string;
  overall_status: string;
  drift_alerts_total: number;
  sensor_outages_total: number;
  reference_features: string[];
  sample_count: number;
  latest_report: Record<string, unknown>;
};

export type SimulationRequest = {
  scenario_type?: 'dc_power_flow' | 'battery_optimization';
  region_id?: string;
  load_scale?: number;
  line_outages?: string[];
  battery_capacity_ratio?: number;
  battery_efficiency?: number;
  tariff_peak?: number;
  tariff_degradation?: number;
};

export type SimulationResponse = {
  schema_version: string;
  scenario_type: string;
  region_id: string;
  feasible: boolean;
  status: string;
  branch_flows: Array<{
    branch: string;
    flow_MW: number;
    rating_MW: number;
    loading_pct: number;
    overloaded: boolean;
  }>;
  bus_angles: Record<string, number>;
  constraint_violations: string[];
  battery_schedule?: Array<{
    step: number;
    charge_MW: number;
    discharge_MW: number;
    soc_MWh: number;
    cost_CU: number;
  }>;
  realized_cost_CU?: Record<string, number>;
  advisory_notice: string;
};

export class ApiError extends Error {
  constructor(message: string, readonly status: number) { super(message); }
}

let authToken: string | null = null;

export function setAuthToken(token: string | null): void {
  authToken = token;
  if (token) {
    try {
      localStorage.setItem('urbangraph_auth_token', token);
    } catch {}
  } else {
    try {
      localStorage.removeItem('urbangraph_auth_token');
    } catch {}
  }
}

export function getAuthToken(): string | null {
  if (authToken) return authToken;
  try {
    // Tokens are ONLY read from in-memory state or localStorage.
    // Reading tokens from URL query parameters is PROHIBITED: doing so
    // would expose credentials in browser history, server access logs,
    // proxy logs, and HTTP Referer headers.
    const fromStorage = localStorage.getItem('urbangraph_auth_token');
    if (fromStorage) {
      authToken = fromStorage;
      return fromStorage;
    }
  } catch {}
  return null;
}

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const headers: Record<string, string> = { Accept: 'application/json' };
  const token = getAuthToken();
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  const response = await fetch(path, { signal, headers });
  if (!response.ok) {
    const error = await response.json().catch(() => null) as {error?: {message?: string}} | null;
    throw new ApiError(error?.error?.message ?? `Request failed (${response.status})`, response.status);
  }
  return response.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  };
  const token = getAuthToken();
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  const response = await fetch(path, {
    method: 'POST',
    signal,
    headers,
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => null) as {error?: {message?: string}} | null;
    throw new ApiError(error?.error?.message ?? `Request failed (${response.status})`, response.status);
  }
  return response.json() as Promise<T>;
}

const query = (params: Record<string, string>) => new URLSearchParams(params).toString();
export const api = {
  meta: (signal?: AbortSignal) => get<Meta>('/api/v1/meta', signal),
  regions: (signal?: AbortSignal) => get<Identity & {regions: Region[]}>('/api/v1/regions', signal),
  graph: (signal?: AbortSignal) => get<Graph>('/api/v1/graph', signal),
  series: (region_id: string, start_utc: string, end_utc: string, signal?: AbortSignal) => get<Identity & {points: SeriesPoint[]; count: number}>('/api/v1/series?' + query({region_id, start_utc, end_utc}), signal),
  forecasts: (region_id: string, origin_utc: string, signal?: AbortSignal) => get<Identity & {points: ForecastPoint[]; interval_kind: string}>('/api/v1/forecasts?' + query({region_id, origin_utc}), signal),
  dispatch: (region_id: string, origin_utc: string, signal?: AbortSignal) => get<Dispatch>('/api/v1/dispatch?' + query({region_id, origin_utc}), signal),
  predictResilient: (
    region_id: string,
    origin_utc: string,
    injected_outages: string[] = [],
    confidence_level: number = 0.90,
    signal?: AbortSignal,
  ) =>
    post<ResilientPredictResponse>(
      '/api/v1/predict',
      { region_id, origin_utc, injected_outages, confidence_level },
      signal,
    ),
  driftStatus: (signal?: AbortSignal) => get<DriftMonitoringStatus>('/api/v1/monitoring/drift', signal),
  simulate: (payload: SimulationRequest = {}, signal?: AbortSignal) =>
    post<SimulationResponse>('/api/v1/simulate', payload, signal),
};
