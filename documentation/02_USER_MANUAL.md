# 02 - User Manual: UrbanGraph Intelligence

## 1. Introduction
UrbanGraph Intelligence provides an intuitive web interface for power system engineers, analysts, and municipal planners to interact with synthetic urban electric grids.

---

## 2. Navigating the Web Interface

### 2.1 The Global Navigation Bar
The top header provides:
- **Topology Selector**: Switch between available synthetic networks (`eg006` 6-bus default).
- **System Status Indicator**: Displays live connection status and backend latency.
- **Simulation Mode**: Toggle between baseline power flow, scenario analysis, and storage dispatch.
- **Authentication Badge**: Shows the active user role (`viewer`, `operator`, `admin`).

### 2.2 Geospatial Grid View (MapLibre GL JS)
The central panel renders an interactive vector map showing:
- **Substations (Nodes)**: Color-coded circles indicating bus voltage level and active power injection.
- **Transmission Lines (Branches)**: Directed line strings showing power flow direction and percentage thermal loading:
  - Green: Loading < 60% of rated capacity.
  - Amber: Loading between 60% and 85%.
  - Red: Loading > 85% (potential branch congestion).
- **Interactive Inspection**: Clicking any substation or line opens an inspector card displaying active ($P$), reactive ($Q$), and voltage angle ($	heta$) values.

---

## 3. Core Functional Workflows

### 3.1 Running a DC Power Flow Simulation
1. Select the **Power Flow** tab in the control panel.
2. Adjust the **Load Scaling Factor** slider (0.5x to 1.5x nominal load).
3. Select the **Slack Bus** reference (default: Bus 1).
4. Click **Run Simulation**.
5. The map updates dynamically to display nodal voltage angles and branch power transfers.

### 3.2 Spatiotemporal Load Forecasting
1. Navigate to the **Forecasting** tab.
2. Select the forecast horizon (1h to 24h ahead).
3. Click **Generate Neural Forecast**.
4. The chart renders forecasted active power demand per bus with 95% confidence bands derived from the `UrbanGraphFormer` neural model.

### 3.3 Battery Energy Storage Optimization
1. Navigate to the **Battery Storage** tab.
2. Define storage parameters:
   - Energy Capacity ($E_{\max}$): e.g., 20 MWh.
   - Power Rating ($P_{\max}$): e.g., 5 MW.
   - Initial State of Charge ($	ext{SoC}_0$): e.g., 50%.
3. Choose the optimization objective: **Peak Shaving** or **Cost Arbitrage**.
4. Click **Solve Storage Schedule**.
5. The `HiGHS` solver generates the 24-hour charge/discharge schedule and projected state-of-charge trajectory.
