# UrbanGraph Intelligence — Simulation vs Live Operational Boundaries

**CRITICAL SAFETY DIRECTIVE:**
UrbanGraph Intelligence software is an analytical forecasting and advisory decision-support system.
It does **NOT** possess automated actuation authority over physical electrical grid infrastructure.

---

## 1. Non-Actuation Safeguards

1. **SCADA Isolation:** Software interfaces are simulated or analytical. No network commands (DNP3, IEC 60870-5-104, IEC 61850, Modbus) are dispatched to real RTUs, PLCs, or substation breaker relays.
2. **User Interface Demarcation:** Every chart, topology map, and dispatch forecast explicitly displays the banner:
   `"SYNTHETIC DEMO SCENARIO — NOT REAL ELECTRICAL TOPOLOGY — CANNOT CONTROL EQUIPMENT"`
3. **Safety Fallback:** If any external control command is attempted without cryptographically signed field engineering authorization, the software halts with `403 Forbidden` / Safety Interlock Rejection.

---

## 2. Field Engineering Prerequisites for Live Integration

- Independent electrical engineer-of-record sign-off.
- Dedicated hardware safety relay interlocks (physical trip-to-isolate).
- Dedicated out-of-band watchdog timer.
- Proprietary utility dataset licensing (R26 Gate).
