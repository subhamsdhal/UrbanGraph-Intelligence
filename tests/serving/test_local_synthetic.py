"""Observable synthetic model execution and local ASGI security boundary."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import httpx
import pytest

from energraph.demo.fixtures import FixtureError
from urbangraph.serving.bundle_preflight import BundlePreflightError
from urbangraph.serving.local_synthetic import (
    APPROVED_BUNDLE_SHA256,
    LocalServingError,
    LocalSyntheticEngine,
    create_local_app,
)

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "tests" / "fixtures" / "p3_synthetic_bundle"
FIXTURE = ROOT / "tests" / "fixtures" / "eg006"
TEST_SECRET = "x" * 48


def _request(
    app: object, origin: str, *, token: str = TEST_SECRET, client: str = "127.0.0.1"
) -> httpx.Response:
    async def execute() -> httpx.Response:
        transport = httpx.ASGITransport(app=app, client=(client, 12345))  # type: ignore[arg-type]
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as session:
            return await session.get(
                "/api/v1/synthetic-inference",
                params={"origin_utc": origin},
                headers={"x-local-token": token},
            )

    return asyncio.run(execute())


def test_existing_persistence_model_forecasts_all_approved_origins() -> None:
    engine = LocalSyntheticEngine.load(BUNDLE, FIXTURE)
    assert engine.trusted.bundle_sha256 == APPROVED_BUNDLE_SHA256
    for origin in engine.data.origins:
        response = engine.forecast(origin)
        assert response["dataset_kind"] == "synthetic"
        assert isinstance(response["disclaimer"], str)
        assert "no real-world accuracy" in response["disclaimer"]
        result = response["result"]
        assert isinstance(result, dict)
        assert result["bundle_sha256"] == APPROVED_BUNDLE_SHA256
        assert result["model_id"] == "synthetic-persistence-v1"
        predictions = result["predictions"]
        assert isinstance(predictions, list)
        assert len(predictions) == 24
        sensor_ids = result["sensor_ids"]
        assert isinstance(sensor_ids, list)
        expected = [
            engine.data.series[sensor][
                next(
                    i
                    for i, row in enumerate(engine.data.series[sensor])
                    if row["timestamp_utc"] == origin
                )
            ]["load_MW"]
            for sensor in sensor_ids
        ]
        assert all(row["values"] == expected for row in predictions)
        assert engine.forecast(origin) == response
    assert engine.forecast(engine.data.origins[0]) != engine.forecast(engine.data.origins[1])


@pytest.mark.parametrize("defect", ["missing", "tamper", "extra", "manifest", "linked"])
def test_artifact_integrity_fails_before_model_execution(tmp_path: Path, defect: str) -> None:
    root = tmp_path / "bundle"
    shutil.copytree(BUNDLE, root)
    if defect == "missing":
        (root / "model.bin").unlink()
    elif defect == "tamper":
        (root / "model.bin").write_bytes((root / "model.bin").read_bytes() + b"x")
    elif defect == "extra":
        (root / "extra.txt").write_text("invented", encoding="utf-8")
    elif defect == "manifest":
        document = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        document["contract"]["model_id"] = "unsupported"
        (root / "manifest.json").write_text(
            json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
        )
    else:
        (root / "model.bin").unlink()
        try:
            (root / "model.bin").symlink_to(BUNDLE / "model.bin")
        except (OSError, NotImplementedError):
            pytest.skip("File symlinks are unavailable on this host")
    with pytest.raises((LocalServingError, BundlePreflightError)):
        LocalSyntheticEngine.load(root, FIXTURE)


def test_fixture_tamper_and_unsupported_model_semantics_fail(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    shutil.copytree(FIXTURE, fixture)
    (fixture / "fixtures" / "hourly_energy.csv").write_bytes(b"corrupted")
    with pytest.raises(FixtureError):
        LocalSyntheticEngine.load(BUNDLE, fixture)

    from urbangraph.serving import local_synthetic

    payloads = {
        name: (BUNDLE / name).read_bytes()
        for name in ("model.bin", "graph.json", "transform.json", "calibration.json")
    }
    payloads["model.bin"] = b'{"algorithm":"unknown"}\n'
    with pytest.raises(LocalServingError, match="semantics"):
        local_synthetic._semantic_roles(payloads)
    payloads["model.bin"] = (
        b'{"algorithm":"persistence","history_hours":24,'
        b'"schema_version":"ug-synthetic-baseline-1","version":"2.0"}\n'
    )
    with pytest.raises(LocalServingError, match="semantics"):
        local_synthetic._semantic_roles(payloads)


def test_local_api_auth_replay_and_selectors() -> None:
    app = create_local_app(TEST_SECRET, BUNDLE, FIXTURE)
    origin = LocalSyntheticEngine.load(BUNDLE, FIXTURE).data.origins[0]
    good = _request(app, origin)
    assert good.status_code == 200
    assert good.headers["cache-control"] == "no-store"
    assert good.json() == _request(app, origin).json()
    assert len(good.json()["result"]["predictions"]) == 24
    assert _request(app, origin, token="wrong").status_code == 401
    assert _request(app, origin, client="192.0.2.1").status_code == 403
    assert _request(app, "2026-01-01T00:00:00Z").status_code == 404
    assert _request(app, "not-a-time").status_code == 404


def test_unsupported_configuration_and_no_external_observations() -> None:
    with pytest.raises(LocalServingError, match="secret"):
        create_local_app("short", BUNDLE, FIXTURE)
    engine = LocalSyntheticEngine.load(BUNDLE, FIXTURE)
    app = create_local_app(TEST_SECRET, BUNDLE, FIXTURE)
    origin = engine.data.origins[0]

    async def extra_query() -> httpx.Response:
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as session:
            return await session.get(
                "/api/v1/synthetic-inference",
                params={"origin_utc": origin, "source_dir": "untrusted"},
                headers={"x-local-token": TEST_SECRET},
            )

    assert asyncio.run(extra_query()).status_code == 422
