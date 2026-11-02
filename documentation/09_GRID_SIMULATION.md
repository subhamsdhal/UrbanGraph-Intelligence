# 09 - Grid Simulation Engine: DC Power Flow & HiGHS MILP

## 1. DC Power Flow Formulation
The Direct Current (DC) power flow approximation linearizes AC power flow by assuming:
1. Voltage magnitudes are close to 1.0 per-unit ($|V_i| pprox 1.0$).
2. Voltage angle differences across lines are small ($\sin(	heta_i - 	heta_j) pprox 	heta_i - 	heta_j$).
3. Branch resistance is negligible compared to reactance ($R_{ij} \ll X_{ij}$).

Equation:
$$\mathbf{P} = \mathbf{B} oldsymbol{	heta}$$
where $\mathbf{B}$ is the bus susceptance matrix and $oldsymbol{	heta}$ is the vector of bus voltage angles.

---

## 2. Battery Energy Storage Optimization (HiGHS MILP)
Formulated as a Mixed-Integer Linear Program to schedule battery dispatch over horizon $T=24$:

$$\min \sum_{t=1}^T \left( C_t^{	ext{grid}} P_t^{	ext{grid}} + C^{	ext{deg}} (P_t^{	ext{ch}} + P_t^{	ext{dis}}) ight)$$
Subject to:
- State of charge update: $	ext{SoC}_{t} = 	ext{SoC}_{t-1} + rac{\eta^{	ext{ch}} P_t^{	ext{ch}} \Delta t}{E_{\max}} - rac{P_t^{	ext{dis}} \Delta t}{\eta^{	ext{dis}} E_{\max}}$
- Energy storage limits: $	ext{SoC}^{\min} \le 	ext{SoC}_t \le 	ext{SoC}^{\max}$
- Power bounds: $0 \le P_t^{	ext{ch}} \le u_t^{	ext{ch}} P^{\max}$, $0 \le P_t^{	ext{dis}} \le u_t^{	ext{dis}} P^{\max}$
- Exclusivity: $u_t^{	ext{ch}} + u_t^{	ext{dis}} \le 1$, $u_t \in \{0, 1\}$
