# 06 - Frontend Architecture: UrbanGraph Intelligence

## 1. Technology Stack
- **Framework**: React 18
- **Language**: TypeScript (strict mode enabled)
- **Build Tool**: Vite 5
- **Mapping Engine**: MapLibre GL JS
- **State Management**: TanStack Query (React Query) + Zustand
- **Styling**: Tailwind CSS + Lucide Icons

---

## 2. Key Components

- `GridMap`: Vector-based spatial canvas rendering transmission lines and substation buses. Utilizes WebGL shaders for smooth 60fps rendering of large topologies.
- `SimulationPanel`: Interactive control drawer for configuring simulation runs, load scaling, and parameter overrides.
- `MetricsDashboard`: Real-time telemetry widgets showing peak active power, system losses, and branch loading histograms.
- `StorageOptimizerView`: Time-series visualization of battery state-of-charge, charge/discharge power, and market price curves.
