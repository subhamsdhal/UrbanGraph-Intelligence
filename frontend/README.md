# EnerGraph synthetic dashboard

EG-006 adds a local React/TypeScript/Vite dashboard for three invented regions. It uses only the seven read-only FastAPI routes; forecasts, bands, graph and battery values are synthetic illustrations. Run `npm ci`, then `npm run dev` after starting the API, or use `scripts/run_eg006_demo.ps1` from the repository root. Run `npm run build`, `npm test`, and `npm run e2e` for frontend checks. The full setup and limitations are in `docs/energraph/eg006/walkthrough.md`.

The longer-term MapLibre/deck.gl command-center concept remains planned. No real geospatial topology is claimed here.
