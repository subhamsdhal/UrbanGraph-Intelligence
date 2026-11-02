# UrbanGraph Intelligence — Frontend to Backend API Workflow Traceability Matrix

This matrix establishes the complete end-to-end trace from user interaction through React components, HTTP request, route, logic, and visible outcome.

| Workflow # | User Interaction / Workflow | Frontend Component | Triggered HTTP Request | Authenticated Route | Backend Service / Logic | User Visible Outcome |
|---|---|---|---|---|---|---|
| **W01** | Initial Application Load | `App.tsx` | `GET /api/v1/meta`, `GET /api/v1/regions` | `/api/v1/meta` | `energraph.demo.api.meta` | Dashboard hero metadata, region dropdown populated |
| **W02** | Authentication Entry | `App.tsx` | Modal password entry -> `setAuthToken()` | Header `Authorization: Bearer` | In-memory token storage | Authenticated state active, 401 modal dismissed |
| **W03** | Region Selection | `App.tsx` controls | `GET /api/v1/forecasts`, `GET /api/v1/series` | `/api/v1/forecasts` | `energraph.demo.api.forecasts` | Day-ahead forecast chart and 7-day history reloaded |
| **W04** | Grid Topology Retrieval | `GeospatialCommandCenter.tsx` | `GET /api/v1/graph` | `/api/v1/graph` | `energraph.demo.api.graph` | Substation nodes and transmission edges populated |
| **W05** | Substation Display | `MapLibreGLMap.tsx` | GeoJSON Point layer instantiation | Client runtime | MapLibre `circle` layer | Geo-located substation pins rendered with load tags |
| **W06** | Transmission Line Rendering | `MapLibreGLMap.tsx` | GeoJSON LineString layer | Client runtime | MapLibre `line` layer | Interconnecting transmission lines with capacity weights |
| **W07** | Real-time Event Display | `GeospatialCommandCenter.tsx` | `GET /api/v1/monitoring/drift` | `/api/v1/monitoring/drift` | `FeatureDriftDetector` | Drift status badge and outage alert counters |
| **W08** | Forecast Parameter Selection | `App.tsx` | Dropdown origin change | `/api/v1/forecasts` | `energraph.demo.api.forecasts` | Chart updates to 24-hour target date |
| **W09** | Neural Inference Invocation | `GeospatialCommandCenter.tsx` | `POST /api/v1/predict` | `/api/v1/predict` | `UrbanGraphFormer.forward()` | 24-step forecast points computed using neural weights |
| **W10** | Forecast Visualization | `ForecastChart` in `App.tsx` | SVG polyline rendering | Client runtime | SVG vector graphics | Forecast vs truth curve displayed with legend |
| **W11** | Uncertainty Intervals | `ForecastChart` | Conformal interval bounds | `/api/v1/predict` | `traffic_conformal` | Shaded uncertainty band displayed around forecast |
| **W12** | Sensor Outage Injection | `GeospatialCommandCenter.tsx` | `POST /api/v1/predict` (injected_outages) | `/api/v1/predict` | `RuntimeResilienceController` | Sensor dropped; uncertainty interval inflated |
| **W13** | Model Status & Readiness | Header badge in `App.tsx` | `GET /readyz`, `GET /healthz` | `/readyz` | `get_global_metrics().check_health()` | Green "HEALTHY / READY" indicator |
| **W14** | Graph Topology Diagram | `Network` in `App.tsx` | `GET /api/v1/graph` | `/api/v1/graph` | `energraph.demo.api.graph` | Schematic topology diagram rendered |
| **W15** | Extreme Outage Simulation | `GeospatialCommandCenter.tsx` | Outage button click | `/api/v1/predict` | `RuntimeResilienceController` | Outage report: degradation level, imputation applied |
| **W16** | Battery Dispatch Optimization | `BatteryView` in `App.tsx` | `GET /api/v1/dispatch` | `/api/v1/dispatch` | `OptimizationSolver` | BESS SOC MWh curve and charge/discharge MW metrics |
| **W17** | Power-Flow Validation | `GeospatialCommandCenter.tsx` | Load / Capacity check | Client & Server | `physical.py` | Line thermal loading percentages visualized |
| **W18** | Historical Load Exploration | `HistoryChart` in `App.tsx` | `GET /api/v1/series` | `/api/v1/series` | `energraph.demo.api.series` | 168-hour historical load graph rendered |
| **W19** | Monitoring & Drift Indicators | `GeospatialCommandCenter.tsx` | `GET /api/v1/monitoring/drift` | `/api/v1/monitoring/drift` | `FeatureDriftDetector` | Live KS-test drift alert count: 0 alerts |
| **W20** | Error / Warning Notifications | `App.tsx` banner | API error interception | Client catch | Starlette error handler | Red notification bar with retry action |
| **W21** | Loading State Feedback | `App.tsx` | Spinner / status role | React state | `selectionLoading` | "Loading forecast and regional history..." indicator |
| **W22** | Empty Dataset Fallback | `App.tsx` | Missing data fallback | React state | Boundary guard | "No forecast available for this selection" message |
| **W23** | Unauthenticated Session | `App.tsx` | 401 response intercept | `/api/v1/*` | HTTP Auth Middleware | Inline password-masked Bearer Token entry form |
| **W24** | Network Interruption Recovery | `App.tsx` | "Authenticate & Retry" button | Client retry | Fetch retry loop | Recovers state seamlessly upon network return |
| **W25** | Browser Refresh Behavior | `App.tsx` | Page reload | LocalStorage token | `getAuthToken()` | Restores session without requiring re-authentication |
| **W26** | Responsive Layouts | CSS Grid & Flexbox | Window resize | Client CSS | Media queries (mobile/desktop) | Responsive grid adapting from 320px to 4K displays |
