# 08 - Data and Fixtures Guide: EG006 Network

## 1. Network Specification
The `eg006` fixture is a 6-bus synthetic transmission network operating at 230 kV nominal voltage.

### 1.1 Buses
| Bus ID | Type | Base kV | Pd (MW) | Qd (MVAr) | Description |
|:-------|:-----|:--------|:--------|:----------|:------------|
| 1 | Slack | 230 | 0.0 | 0.0 | Hydro Generation Facility |
| 2 | PV | 230 | 0.0 | 0.0 | Combined Cycle Gas Turbine |
| 3 | PV | 230 | 0.0 | 0.0 | Solar Photovoltaic Farm |
| 4 | PQ | 230 | 100.0 | 15.0 | Urban Commercial Center |
| 5 | PQ | 230 | 100.0 | 15.0 | Industrial Manufacturing Park |
| 6 | PQ | 230 | 100.0 | 15.0 | Residential Metro Area |

### 1.2 Transmission Lines
| From | To | Resistance (pu) | Reactance (pu) | Thermal Limit (MVA) |
|:-----|:---|:----------------|:---------------|:--------------------|
| 1 | 2 | 0.005 | 0.040 | 150.0 |
| 1 | 4 | 0.010 | 0.060 | 120.0 |
| 2 | 3 | 0.005 | 0.030 | 150.0 |
| 2 | 4 | 0.010 | 0.070 | 100.0 |
| 3 | 5 | 0.012 | 0.090 | 100.0 |
| 4 | 5 | 0.008 | 0.050 | 100.0 |
| 5 | 6 | 0.010 | 0.060 | 120.0 |
