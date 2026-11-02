# UrbanGraph Intelligence — Energy System Physics & Dispatch Validation

**Validation Engine:** Analytical Power Flow & Convex BESS Dispatch Formulations  

---

## 1. Physics-Informed Power Flow Validation

- **DC Power Flow Approximation:** Bus power angle calculations enforce Kirchhoff's Current and Voltage Laws.
- **Thermal Line Limits:** Real-time branch flows are evaluated against line thermal ratings ($S_{ij} \le S_{max}$).
- **Voltage Limits:** Bus voltages constrained within $\pm 5\%$ nominal bounds ($0.95 \le V_i \le 1.05	ext{ p.u.}$).

---

## 2. Battery Energy Storage System (BESS) Dispatch Bounds

- **State of Charge (SOC) Bounds:** $0 \le 	ext{SOC}_t \le 	ext{Capacity}_{	ext{MWh}}$ enforced across all 24 horizons.
- **Power Rating Limits:** $-P_{\max} \le P_{discharge, t} - P_{charge, t} \le P_{\max}$ strictly maintained.
- **Roundtrip Efficiency:** Quadratic losses parameterized with $\eta_{rt} = 90\%$.
- **Arbitrage Feasibility:** Solver verified optimal dispatch against diurnal price curves without physical constraint violations.
