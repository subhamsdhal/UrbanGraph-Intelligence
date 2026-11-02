"""Exercise real persistence forecasts through the bounded synthetic replay."""

import asyncio
import logging
from datetime import timedelta
from pathlib import Path

import httpx
import pytest

from urbangraph.serving.local_synthetic import LocalSyntheticEngine, create_local_app
from urbangraph.serving.synthetic_replay import replay_approved_origins

ROOT = Path(__file__).resolve().parents[2]


def _engine() -> LocalSyntheticEngine:
    return LocalSyntheticEngine.load(
        ROOT / "tests" / "fixtures" / "p3_synthetic_bundle",
        ROOT / "tests" / "fixtures" / "eg006",
    )


def test_replay_runs_actual_24_hour_baseline_and_counts_events() -> None:
    engine = _engine()
    origins = tuple(engine.data.origins)
    result = replay_approved_origins(engine, (origins[0], origins[0], "external", origins[1]))
    assert result.safe_metrics() == {"accepted": 2, "duplicates": 1, "late": 0, "rejected": 1}
    assert len(result.forecasts) == 2
    assert all(len(item["result"]["predictions"]) == 24 for item in result.forecasts)  # type: ignore[index]
    assert result == replay_approved_origins(
        engine, (origins[0], origins[0], "external", origins[1])
    )


def test_late_origin_dropped_without_forecast() -> None:
    engine = _engine()
    origins = tuple(engine.data.origins)
    result = replay_approved_origins(
        engine, (origins[-1], origins[0]), allowed_lateness=timedelta(0)
    )
    assert result.safe_metrics() == {"accepted": 1, "duplicates": 0, "late": 1, "rejected": 0}


def test_replay_api_requires_local_token_and_returns_actual_forecast(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "s" * 48
    engine = _engine()
    origin = tuple(engine.data.origins)[0]
    app = create_local_app(
        secret,
        ROOT / "tests" / "fixtures" / "p3_synthetic_bundle",
        ROOT / "tests" / "fixtures" / "eg006",
    )

    async def request(token: str, client: str) -> httpx.Response:
        transport = httpx.ASGITransport(app=app, client=(client, 12345))
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as session:
            return await session.get(
                "/api/v1/synthetic-replay",
                params=[("origin_utc", origin), ("origin_utc", origin)],
                headers={"x-local-token": token},
            )

    caplog.set_level(logging.INFO, logger="urbangraph.serving.audit")
    logging.getLogger("urbangraph").propagate = True
    logging.getLogger("urbangraph.serving.audit").propagate = True
    assert asyncio.run(request("bad", "127.0.0.1")).status_code == 401
    assert asyncio.run(request(secret, "203.0.113.1")).status_code == 403
    response = asyncio.run(request(secret, "127.0.0.1"))
    assert response.status_code == 200
    assert response.json()["metrics"] == {
        "accepted": 1,
        "duplicates": 1,
        "late": 0,
        "rejected": 0,
    }
    assert len(response.json()["forecasts"][0]["result"]["predictions"]) == 24
    assert "x-local-token" not in response.text
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "synthetic_replay_denied reason=unauthorized" in messages
    assert "synthetic_replay_completed accepted=1 duplicates=1" in messages
    assert secret not in messages and origin not in messages
