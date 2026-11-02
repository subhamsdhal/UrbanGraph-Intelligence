"""Executable local synthetic persistence forecast; no real-data or promotion path."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from energraph.demo.fixtures import BUILTIN_MANIFEST_SHA256, DemoData, load_demo
from urbangraph.baselines.contracts import ModelConfig
from urbangraph.baselines.history import ForecastBatch, HistoricalAccess
from urbangraph.baselines.models import Baseline
from urbangraph.evaluation.contracts import TargetValidityPolicy
from urbangraph.inference_contract import SCHEMA as REQUEST_SCHEMA
from urbangraph.inference_contract import TrustedContract, parse_request, validate_response
from urbangraph.serving.bundle_preflight import ROLE_FILES, ExpectedBundle, validate_bundle

APPROVED_BUNDLE_SHA256 = "0864c1993288a5460c771211332bda622aacbca31156c10f2f62f3d0f4bc4405"
BASELINE_CODE_COMMIT = "f7a6119115e0afc82efc35c42f66533871d30ced"
SENSORS = ("SYN_A", "SYN_B", "SYN_C")
HISTORY = 24
HORIZON = 24
MAX_BUNDLE_FILE = 64 * 1024
DISCLAIMER = (
    "Invented synthetic data and a persistence baseline; no real-world accuracy, "
    "calibrated uncertainty or operational dispatch claim."
)
_AUDIT = logging.getLogger("urbangraph.serving.audit")


class LocalServingError(ValueError):
    """The pinned local synthetic model or fixture is unavailable or incompatible."""


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _real_file(path: Path) -> bool:
    return (
        path.is_file()
        and not path.is_symlink()
        and not getattr(path, "is_junction", lambda: False)()
    )


def _read_bundle(root: Path) -> tuple[bytes, dict[str, bytes]]:
    if root.is_symlink() or getattr(root, "is_junction", lambda: False)() or not root.is_dir():
        raise LocalServingError("Trusted bundle directory is unavailable")
    names = {entry.name for entry in root.iterdir()}
    if names != set(ROLE_FILES.values()) | {"manifest.json"}:
        raise LocalServingError("Bundle inventory differs from approved model")
    items: dict[str, bytes] = {}
    for name in (*ROLE_FILES.values(), "manifest.json"):
        path = root / name
        if not _real_file(path) or path.stat().st_size > MAX_BUNDLE_FILE:
            raise LocalServingError("Bundle file is missing, linked or oversized")
        items[name] = path.read_bytes()
    return items.pop("manifest.json"), items


def _semantic_roles(payloads: dict[str, bytes]) -> None:
    intended: dict[str, object] = {
        "model.bin": {
            "algorithm": "persistence",
            "history_hours": HISTORY,
            "schema_version": "ug-synthetic-baseline-1",
            "version": "1.0",
        },
        "transform.json": {
            "kind": "identity",
            "schema_version": "ug-synthetic-transform-1",
            "unit": "MW",
        },
        "graph.json": {
            "kind": "schematic-only",
            "schema_version": "ug-synthetic-graph-1",
            "sensor_ids": list(SENSORS),
        },
        "calibration.json": {
            "intervals": "unavailable",
            "kind": "none",
            "schema_version": "ug-synthetic-calibration-1",
        },
    }
    if any(payloads[name] != _canonical(document) for name, document in intended.items()):
        raise LocalServingError("Bundle model semantics are unsupported")


@dataclass(frozen=True)
class LocalSyntheticEngine:
    data: DemoData
    trusted: TrustedContract
    baseline: Baseline

    @classmethod
    def load(cls, bundle_root: Path, fixture_root: Path) -> LocalSyntheticEngine:
        """Load only trusted local paths; no request can choose an artifact or source."""
        manifest, payloads = _read_bundle(bundle_root)
        expected = ExpectedBundle(
            source_manifest_sha256=BUILTIN_MANIFEST_SHA256,
            code_commit=BASELINE_CODE_COMMIT,
            model_id="synthetic-persistence-v1",
            graph_id="schematic-graph-v1",
            transform_id="identity-transform-v1",
            calibration_id="no-calibration-v1",
            sensor_ids=SENSORS,
            horizon_hours=HORIZON,
            unit="MW",
        )
        digest = validate_bundle(manifest, payloads, expected)
        if digest != APPROVED_BUNDLE_SHA256:
            raise LocalServingError("Bundle is not the independently pinned local model")
        _semantic_roles(payloads)
        data = load_demo(fixture_root)
        if data.manifest_sha256 != BUILTIN_MANIFEST_SHA256:
            raise LocalServingError("Only the built-in synthetic fixture is approved")
        trusted = TrustedContract(digest, expected.model_id, SENSORS, HISTORY, HORIZON, "MW")
        baseline = Baseline(
            ModelConfig(model="persistence", history_fallback=HISTORY), interval_seconds=3600
        )
        return cls(data, trusted, baseline)

    def forecast(self, origin_utc: str) -> dict[str, object]:
        """Predict from verified synthetic observations using the existing Baseline."""
        if origin_utc not in self.data.origins:
            raise LocalServingError("Unknown approved synthetic origin")
        first = self.data.series[SENSORS[0]]
        positions = [i for i, row in enumerate(first) if row["timestamp_utc"] == origin_utc]
        if len(positions) != 1 or positions[0] < HISTORY - 1:
            raise LocalServingError("Approved synthetic history is unavailable")
        end = positions[0]
        history: list[dict[str, object]] = []
        for i in range(end - HISTORY + 1, end + 1):
            stamp = first[i]["timestamp_utc"]
            if any(self.data.series[sensor][i]["timestamp_utc"] != stamp for sensor in SENSORS):
                raise LocalServingError("Synthetic sensor histories are misaligned")
            history.append(
                {
                    "timestamp_utc": stamp,
                    "values": [self.data.series[sensor][i]["load_MW"] for sensor in SENSORS],
                }
            )
        request_id = "syn-" + hashlib.sha256(origin_utc.encode("ascii")).hexdigest()[:24]
        request = parse_request(
            _canonical(
                {
                    "schema_version": REQUEST_SCHEMA,
                    "bundle_sha256": self.trusted.bundle_sha256,
                    "request_id": request_id,
                    "origin_utc": origin_utc,
                    "sensor_ids": list(SENSORS),
                    "unit": "MW",
                    "history": history,
                }
            ),
            self.trusted,
        )
        values = np.asarray([row[1] for row in request.history], dtype=np.float64)
        stamps = pd.DatetimeIndex([row[0] for row in request.history])
        access = HistoricalAccess(
            values,
            stamps,
            np.array([HISTORY - 1], dtype=np.int64),
            delay=0,
            lower=0,
            validity=TargetValidityPolicy(),
        )
        targets = pd.date_range(request.origin_utc + timedelta(hours=1), periods=HORIZON, freq="h")
        batch = ForecastBatch(
            values[None, :, :, None],
            np.array([HISTORY - 1], dtype=np.int64),
            np.arange(HISTORY, HISTORY + HORIZON, dtype=np.int64)[None, :],
            targets,
            SENSORS,
            access,
            "p3-synthetic-hourly-v1",
            0,
        )
        prediction = self.baseline.predict(batch)
        if not prediction.available.all() or prediction.fallback.any():
            raise LocalServingError("Synthetic baseline returned unavailable values")
        rows = prediction.values[0, :, :, 0]
        core: dict[str, object] = {
            "schema_version": REQUEST_SCHEMA,
            "bundle_sha256": self.trusted.bundle_sha256,
            "model_id": self.trusted.model_id,
            "request_id": request.request_id,
            "origin_utc": origin_utc,
            "sensor_ids": list(SENSORS),
            "unit": "MW",
            "predictions": [
                {
                    "target_utc": target.isoformat().replace("+00:00", "Z"),
                    "values": [float(number) for number in rows[i]],
                }
                for i, target in enumerate(targets)
            ],
        }
        validate_response(_canonical(core), self.trusted, request)
        return {
            "dataset_kind": "synthetic",
            "source_manifest_sha256": self.data.manifest_sha256,
            "algorithm": "persistence",
            "disclaimer": DISCLAIMER,
            "result": core,
        }


def create_local_app(secret: str, bundle_root: Path, fixture_root: Path) -> FastAPI:
    """Build an in-process API; only a future reviewed runner may bind a socket."""
    if (
        not isinstance(secret, str)
        or not 32 <= len(secret) <= 128
        or not secret.isprintable()
        or not secret.isascii()
    ):
        raise LocalServingError("A strong local secret must be supplied privately")
    engine = LocalSyntheticEngine.load(bundle_root, fixture_root)
    app = FastAPI(title="UrbanGraph local synthetic persistence", version="p3-local-v1")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "dataset_kind": "synthetic"}

    @app.get("/api/v1/synthetic-inference")
    async def infer(request: Request) -> JSONResponse:
        if request.client is None or request.client.host not in {"127.0.0.1", "::1"}:
            return JSONResponse({"error": "local_only"}, status_code=403)
        token = request.headers.get("x-local-token", "")
        if len(token) > 128 or not hmac.compare_digest(token, secret):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        params = list(request.query_params.multi_items())
        if len(params) != 1 or params[0][0] != "origin_utc" or len(params[0][1]) > 32:
            return JSONResponse({"error": "invalid_selector"}, status_code=422)
        try:
            result = engine.forecast(params[0][1])
        except LocalServingError:
            return JSONResponse({"error": "unknown_synthetic_origin"}, status_code=404)
        return JSONResponse(result, headers={"Cache-Control": "no-store"})

    @app.get("/api/v1/synthetic-replay")
    async def replay(request: Request) -> JSONResponse:
        from urbangraph.serving.synthetic_replay import MAX_EVENTS, replay_approved_origins

        if request.client is None or request.client.host not in {"127.0.0.1", "::1"}:
            _AUDIT.warning("synthetic_replay_denied reason=local_only")
            return JSONResponse({"error": "local_only"}, status_code=403)
        token = request.headers.get("x-local-token", "")
        if len(token) > 128 or not hmac.compare_digest(token, secret):
            _AUDIT.warning("synthetic_replay_denied reason=unauthorized")
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        params = list(request.query_params.multi_items())
        if (
            not params
            or len(params) > MAX_EVENTS
            or any(key != "origin_utc" or len(value) > 32 for key, value in params)
        ):
            _AUDIT.warning("synthetic_replay_denied reason=invalid_selector")
            return JSONResponse({"error": "invalid_selector"}, status_code=422)
        result = replay_approved_origins(engine, tuple(value for _, value in params))
        metrics = result.safe_metrics()
        _AUDIT.info(
            "synthetic_replay_completed accepted=%d duplicates=%d late=%d rejected=%d",
            metrics["accepted"],
            metrics["duplicates"],
            metrics["late"],
            metrics["rejected"],
        )
        return JSONResponse(
            {
                "dataset_kind": "synthetic",
                "metrics": metrics,
                "forecasts": result.forecasts,
            },
            headers={"Cache-Control": "no-store"},
        )

    return app
