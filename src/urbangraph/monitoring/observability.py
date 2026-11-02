"""Structured Observability, Telemetry, and Prometheus Metrics for UrbanGraph.

Satisfies Requirement R21:
- Structured application and model metrics.
- Request, error, outage, and drift counters.
- Latency percentile tracking (p50, p90, p99).
- Model artifact and version traceability.
- Secret-safe redaction for all telemetry and diagnostic outputs.
- Standard Prometheus text format exposition.
- Health (/healthz) and readiness (/readyz) diagnostics.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from urbangraph.data import DataError

SECRET_PATTERN = re.compile(
    r"(bearer\s+[\w\.\-]+|api_?key[\s:=]+[\w\-]+|secret[\s:=]+[\w\-]+|token[\s:=]+[\w\-]+|password[\s:=]+[\w\-]+)",
    re.IGNORECASE,
)
SENSITIVE_KEYS = frozenset({"authorization", "token", "secret", "password", "api_key", "hmac_key"})


def redact_secrets(data: Any) -> Any:
    """Recursively scrub authorization tokens, passwords, and API keys from data."""
    if isinstance(data, dict):
        cleaned: dict[str, Any] = {}
        for k, v in data.items():
            if str(k).lower() in SENSITIVE_KEYS:
                cleaned[k] = "[REDACTED]"
            else:
                cleaned[k] = redact_secrets(v)
        return cleaned
    if isinstance(data, list):
        return [redact_secrets(item) for item in data]
    if isinstance(data, str):
        return SECRET_PATTERN.sub("[REDACTED]", data)
    return data


@dataclass
class HealthCheckReport:
    """Operational health report for liveness and readiness probes."""

    status: str  # "HEALTHY", "DEGRADED", "UNHEALTHY"
    ready: bool
    timestamp_utc: str
    uptime_seconds: float
    active_models: list[str]
    checks: dict[str, bool]
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return redact_secrets(asdict(self))


class MetricsCollector:
    """Thread-safe telemetry accumulator for real-time model and operational monitoring."""

    def __init__(self, max_latency_samples: int = 1000) -> None:
        self._lock = threading.RLock()
        self._start_time = time.time()
        self._max_latency_samples = max_latency_samples

        # Counters
        self.requests_total: dict[str, int] = {}  # endpoint -> count
        self.errors_total: dict[str, int] = {}    # error_code -> count
        self.predictions_by_model: dict[str, int] = {}
        self.sensor_outages_total: int = 0
        self.drift_alerts_total: int = 0

        # Latencies in milliseconds (reservoir for percentiles)
        self.latencies: deque[float] = deque(maxlen=max_latency_samples)

        # Artifact metadata
        self.model_metadata: dict[str, dict[str, str]] = {}

    def record_request(self, endpoint: str, status_code: int, latency_ms: float, model_id: str | None = None) -> None:
        """Record an incoming prediction or API request."""
        with self._lock:
            key = f"{endpoint}:{status_code}"
            self.requests_total[key] = self.requests_total.get(key, 0) + 1

            if status_code >= 400:
                err_key = f"HTTP_{status_code}"
                self.errors_total[err_key] = self.errors_total.get(err_key, 0) + 1

            if model_id:
                self.predictions_by_model[model_id] = self.predictions_by_model.get(model_id, 0) + 1

            self.latencies.append(float(latency_ms))

    def record_outage(self, dropped_count: int) -> None:
        """Increment detected sensor outage counter."""
        with self._lock:
            self.sensor_outages_total += dropped_count

    def record_drift_alert(self) -> None:
        """Increment distribution drift alert counter."""
        with self._lock:
            self.drift_alerts_total += 1

    def register_model_artifact(self, model_id: str, version: str, sha256: str) -> None:
        """Register active model artifact and provenance hash."""
        with self._lock:
            self.model_metadata[model_id] = {
                "version": version,
                "sha256": sha256,
                "registered_utc": datetime.now(UTC).isoformat(),
            }

    def get_latency_percentiles(self) -> dict[str, float]:
        """Compute p50, p90, p99 latency in milliseconds."""
        with self._lock:
            if not self.latencies:
                return {"p50_ms": 0.0, "p90_ms": 0.0, "p99_ms": 0.0}
            sorted_lat = sorted(self.latencies)
            n = len(sorted_lat)
            p50 = sorted_lat[int(0.50 * (n - 1))]
            p90 = sorted_lat[int(0.90 * (n - 1))]
            p99 = sorted_lat[int(0.99 * (n - 1))]
            return {
                "p50_ms": round(float(p50), 2),
                "p90_ms": round(float(p90), 2),
                "p99_ms": round(float(p99), 2),
            }

    def get_summary(self) -> dict[str, Any]:
        """Return clean structured summary of operational metrics."""
        with self._lock:
            uptime = time.time() - self._start_time
            total_reqs = sum(self.requests_total.values())
            total_errs = sum(self.errors_total.values())
            error_rate = (total_errs / total_reqs) if total_reqs > 0 else 0.0

            return {
                "uptime_seconds": round(uptime, 1),
                "total_requests": total_reqs,
                "total_errors": total_errs,
                "error_rate": round(error_rate, 4),
                "requests_breakdown": dict(self.requests_total),
                "errors_breakdown": dict(self.errors_total),
                "predictions_by_model": dict(self.predictions_by_model),
                "sensor_outages_total": self.sensor_outages_total,
                "drift_alerts_total": self.drift_alerts_total,
                "latency": self.get_latency_percentiles(),
                "registered_models": dict(self.model_metadata),
            }

    def check_health(self) -> HealthCheckReport:
        """Evaluate operational health."""
        summary = self.get_summary()
        checks = {
            "service_alive": True,
            "error_rate_acceptable": summary["error_rate"] < 0.20,
            "latency_acceptable": summary["latency"]["p90_ms"] < 2000.0,
        }
        healthy = all(checks.values())
        degraded = summary["error_rate"] >= 0.05 or summary["sensor_outages_total"] > 10

        status = "HEALTHY" if healthy and not degraded else ("DEGRADED" if healthy else "UNHEALTHY")

        return HealthCheckReport(
            status=status,
            ready=healthy,
            timestamp_utc=datetime.now(UTC).isoformat(),
            uptime_seconds=summary["uptime_seconds"],
            active_models=list(self.model_metadata.keys()),
            checks=checks,
            details={"summary": summary},
        )


class PrometheusExporter:
    """Formats metrics in standard Prometheus text exposition format."""

    @staticmethod
    def to_prometheus_text(collector: MetricsCollector) -> str:
        summary = collector.get_summary()
        lines: list[str] = [
            "# HELP urbangraph_uptime_seconds Total application uptime in seconds",
            "# TYPE urbangraph_uptime_seconds gauge",
            f"urbangraph_uptime_seconds {summary['uptime_seconds']}",
            "",
            "# HELP urbangraph_requests_total Total number of HTTP prediction requests",
            "# TYPE urbangraph_requests_total counter",
        ]

        for key, count in summary["requests_breakdown"].items():
            ep, status = key.split(":") if ":" in key else (key, "unknown")
            lines.append(f'urbangraph_requests_total{{endpoint="{ep}",status="{status}"}} {count}')

        lines.extend([
            "",
            "# HELP urbangraph_errors_total Total number of errors",
            "# TYPE urbangraph_errors_total counter",
        ])
        for code, count in summary["errors_breakdown"].items():
            lines.append(f'urbangraph_errors_total{{code="{code}"}} {count}')

        lines.extend([
            "",
            "# HELP urbangraph_sensor_outages_total Total number of detected missing sensor observations",
            "# TYPE urbangraph_sensor_outages_total counter",
            f"urbangraph_sensor_outages_total {summary['sensor_outages_total']}",
            "",
            "# HELP urbangraph_drift_alerts_total Total distribution drift alerts triggered",
            "# TYPE urbangraph_drift_alerts_total counter",
            f"urbangraph_drift_alerts_total {summary['drift_alerts_total']}",
            "",
            "# HELP urbangraph_latency_p50_milliseconds 50th percentile request latency",
            "# TYPE urbangraph_latency_p50_milliseconds gauge",
            f"urbangraph_latency_p50_milliseconds {summary['latency']['p50_ms']}",
            "",
            "# HELP urbangraph_latency_p90_milliseconds 90th percentile request latency",
            "# TYPE urbangraph_latency_p90_milliseconds gauge",
            f"urbangraph_latency_p90_milliseconds {summary['latency']['p90_ms']}",
            "",
            "# HELP urbangraph_latency_p99_milliseconds 99th percentile request latency",
            "# TYPE urbangraph_latency_p99_milliseconds gauge",
            f"urbangraph_latency_p99_milliseconds {summary['latency']['p99_ms']}",
            "",
        ])

        return "\n".join(lines)


# Global singleton instance
_GLOBAL_METRICS = MetricsCollector()


def get_global_metrics() -> MetricsCollector:
    return _GLOBAL_METRICS
