"""Bounded, deterministic local replay of approved synthetic forecast origins.

This is an in-process batch replay, not Kafka/live ingestion or delayed-label scoring.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from urbangraph.serving.local_synthetic import LocalServingError, LocalSyntheticEngine

MAX_EVENTS = 256


@dataclass(frozen=True)
class ReplayOutcome:
    forecasts: tuple[dict[str, object], ...]
    accepted: int
    duplicates: int
    late: int
    rejected: int

    def safe_metrics(self) -> dict[str, int]:
        """Return counters only; never expose tokens, paths or observation values."""
        return {
            "accepted": self.accepted,
            "duplicates": self.duplicates,
            "late": self.late,
            "rejected": self.rejected,
        }


def replay_approved_origins(
    engine: LocalSyntheticEngine,
    arrivals: Sequence[str],
    *,
    allowed_lateness: timedelta = timedelta(hours=1),
) -> ReplayOutcome:
    """Replay once in arrival order; drop duplicates and events past watermark.

    Only origins already in the independently verified EG-006 fixture are usable.
    No arbitrary observations, file paths or model selection enter this function.
    """
    if len(arrivals) > MAX_EVENTS or allowed_lateness < timedelta(0):
        raise ValueError("Replay bounds invalid")
    seen: set[str] = set()
    latest: datetime | None = None
    forecasts: list[dict[str, object]] = []
    duplicates = late = rejected = 0
    for origin in arrivals:
        if not isinstance(origin, str) or origin not in engine.data.origins:
            rejected += 1
            continue
        if origin in seen:
            duplicates += 1
            continue
        seen.add(origin)
        stamp = datetime.fromisoformat(origin.replace("Z", "+00:00"))
        if latest is not None and stamp < latest - allowed_lateness:
            late += 1
            continue
        if latest is None or stamp > latest:
            latest = stamp
        try:
            forecasts.append(engine.forecast(origin))
        except LocalServingError:
            rejected += 1
    return ReplayOutcome(tuple(forecasts), len(forecasts), duplicates, late, rejected)
