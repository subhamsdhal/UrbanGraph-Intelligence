# UrbanGraph Intelligence — MapLibre GL JS Runtime Verification Report

**Component:** `frontend/src/components/MapLibreGLMap.tsx`  
**Engine Package:** `maplibre-gl` (v6.16.0)  
**Style Specification:** MapLibre Style Specification v8  
**Verification Harness:** Playwright Headless Chromium E2E Browser Journey (`frontend/tests/e2e/demo.spec.ts`)  

---

## 1. Engine & Runtime Conformance

1. **Genuine Engine Instantiation:**
   - The application directly imports and invokes the official MapLibre GL JS package (`maplibregl.Map`).
   - The map container `#maplibre-offline-container` mounts `new maplibregl.Map({ container, style, ... })`.
   - Zero mock or SVG-only substitution is masqueraded as MapLibre.

2. **Self-Contained Offline Style Specification v8:**
   - Style definition contains `version: 8`.
   - Sources are defined purely as in-memory GeoJSON (`geojson-nodes` for Point substations and `geojson-edges` for LineString transmission corridors).
   - Zero external raster or vector tile URLs (`tiles: []`) are requested, preventing external credential leakage or unauthorized remote calls in air-gapped deployments.

3. **Multi-Mode Layer Toggling:**
   - Mode 1: **DECK.GL / WEBGL** — Hardware-accelerated ArcLayer and ScatterplotLayer.
   - Mode 2: **MAPLIBRE (VECTOR SPEC V8)** — Official vector line and circle rendering engine.
   - Mode 3: **SVG / CANVAS (FALLBACK)** — Accessible vector fallback for headless or non-GPU systems.

---

## 2. Browser Verification Evidence

Playwright tests executed in Chromium (`tests/e2e/demo.spec.ts`):
- `deck.gl webgl, maplibre vector style, and svg fallback toggling (desktop)`: **PASS (1.9s)**
- `deck.gl webgl, maplibre vector style, and svg fallback toggling (mobile)`: **PASS (2.0s)**
- `geospatial command center interactive journey (desktop)`: **PASS (4.5s)**
- `geospatial command center interactive journey (mobile)`: **PASS (3.3s)**
- Total browser console errors: **0**
- Total unhandled promise rejections: **0**
