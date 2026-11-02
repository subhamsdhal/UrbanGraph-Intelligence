"""Local Docker-scoped broker HTTP boundary; no Windows host custody claim."""

from __future__ import annotations

import hmac
import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from energraph.broker_registry import BrokerRegistry
from urbangraph.data import DataError


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Admission(_Strict):
    stage_id: str
    source_id: str
    expected_sha256: str
    directory: bool = False


class Preparation(_Strict):
    stage_id: str
    parent_ids: list[str] = Field(min_length=1)
    transformation: str
    purpose: str
    operation: str
    directory: bool = False


class Approval(_Strict):
    artifact_id: str
    expected_sha256: str


class Use(_Strict):
    artifact_id: str
    purpose: str
    operation: str


class FileUse(Use):
    relative_path: str


class Revocation(_Strict):
    source_id: str


def _secret(path: Path) -> bytes:
    if not path.is_absolute() or not path.is_file():
        raise RuntimeError("Broker credential file unavailable")
    raw = path.read_bytes().strip()
    if not 32 <= len(raw) <= 256 or any(ch < 33 or ch > 126 for ch in raw):
        raise RuntimeError("Broker credential invalid")
    return raw


def create_app(
    registry: BrokerRegistry,
    *,
    admin_secret_file: Path,
    worker_secret_file: Path,
    reader_secret_file: Path,
) -> FastAPI:
    secrets = {
        "admin": _secret(admin_secret_file),
        "worker": _secret(worker_secret_file),
        "reader": _secret(reader_secret_file),
    }
    if len(set(secrets.values())) != 3:
        raise RuntimeError("Broker roles require distinct credentials")
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def require(role: str, authorization: str | None) -> None:
        if authorization is None or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Broker authentication required")
        try:
            token = authorization[7:].encode("ascii")
        except UnicodeEncodeError as exc:
            raise HTTPException(status_code=403, detail="Broker role denied") from exc
        if not hmac.compare_digest(token, secrets[role]):
            raise HTTPException(status_code=403, detail="Broker role denied")

    def guarded(action: str, invoke: object) -> object:
        try:
            assert callable(invoke)
            return invoke()
        except DataError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=503, detail=f"Broker {action} unavailable") from exc

    @app.get("/v1/health")
    def health() -> dict[str, object]:
        return {"version": 1, "status": "ready"}

    @app.post("/v1/admit-source")
    def admit(body: Admission, authorization: str | None = Header(default=None)) -> dict[str, str]:
        require("admin", authorization)
        identity = guarded(
            "admission",
            lambda: registry.admit_source(
                stage_id=body.stage_id,
                source_id=body.source_id,
                expected_sha256=body.expected_sha256,
                issuer="docker-policy-approver-v1",
                directory=body.directory,
            ),
        )
        return {"artifact_id": str(identity)}

    @app.post("/v1/prepare")
    def prepare(
        body: Preparation, authorization: str | None = Header(default=None)
    ) -> dict[str, object]:
        require("worker", authorization)
        identity = guarded(
            "preparation",
            lambda: registry.prepare(
                stage_id=body.stage_id,
                parent_ids=tuple(body.parent_ids),
                transformation=body.transformation,
                purpose=body.purpose,
                operation=body.operation,
                directory=body.directory,
            ),
        )
        details = guarded("preparation identity", lambda: registry.prepared_identity(str(identity)))
        if not isinstance(details, dict):
            raise HTTPException(status_code=503, detail="Broker preparation identity unavailable")
        return details

    @app.post("/v1/approve")
    def approve(body: Approval, authorization: str | None = Header(default=None)) -> dict[str, str]:
        require("admin", authorization)
        guarded(
            "approval",
            lambda: registry.approve(
                body.artifact_id,
                issuer="docker-policy-approver-v1",
                expected_sha256=body.expected_sha256,
            ),
        )
        return {"status": "approved"}

    @app.post("/v1/commit")
    def commit(body: Use, authorization: str | None = Header(default=None)) -> object:
        require("worker", authorization)
        return guarded(
            "commit",
            lambda: registry.commit(
                body.artifact_id, purpose=body.purpose, operation=body.operation
            ),
        )

    @app.post("/v1/authorize-read")
    def authorize_read(body: Use, authorization: str | None = Header(default=None)) -> object:
        require("reader", authorization)
        return guarded(
            "read",
            lambda: registry.authorize_read(
                body.artifact_id, purpose=body.purpose, operation=body.operation
            ),
        )

    @app.post("/v1/read-directory-file")
    def read_directory_file(
        body: FileUse, authorization: str | None = Header(default=None)
    ) -> Response:
        require("reader", authorization)
        content = guarded(
            "read",
            lambda: registry.read_directory_file(
                body.artifact_id,
                body.relative_path,
                purpose=body.purpose,
                operation=body.operation,
            ),
        )
        assert isinstance(content, bytes)
        return Response(content=content, media_type="application/octet-stream")

    @app.post("/v1/revoke")
    def revoke(
        body: Revocation, authorization: str | None = Header(default=None)
    ) -> dict[str, str]:
        require("admin", authorization)
        guarded("revocation", lambda: registry.revoke(body.source_id))
        return {"status": "revoked"}

    return app


def from_environment() -> FastAPI:
    """Uvicorn factory; all credentials and broker-owned paths are injected."""
    if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("Broker runtime requires the approved Linux container filesystem")
    required = (
        "UG_BROKER_DB",
        "UG_BROKER_STAGING",
        "UG_BROKER_ARTIFACTS",
        "UG_BROKER_ADMIN_SECRET_FILE",
        "UG_BROKER_WORKER_SECRET_FILE",
        "UG_BROKER_READER_SECRET_FILE",
    )
    if any(not os.environ.get(key) for key in required):
        raise RuntimeError("Broker deployment configuration incomplete")
    return create_app(
        BrokerRegistry(
            Path(os.environ["UG_BROKER_DB"]),
            Path(os.environ["UG_BROKER_STAGING"]),
            Path(os.environ["UG_BROKER_ARTIFACTS"]),
        ),
        admin_secret_file=Path(os.environ["UG_BROKER_ADMIN_SECRET_FILE"]),
        worker_secret_file=Path(os.environ["UG_BROKER_WORKER_SECRET_FILE"]),
        reader_secret_file=Path(os.environ["UG_BROKER_READER_SECRET_FILE"]),
    )
