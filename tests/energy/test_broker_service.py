"""HTTP role checks use invented test credentials, not deployment custody."""

import hashlib
import io
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from energraph.artifacts import (
    finalize_transactional_publication,
    governed_payload,
    prepare_transactional_publication,
    resolve_broker_reference,
)
from energraph.broker_adapter import BrokerAdapter
from energraph.broker_registry import BrokerRegistry
from energraph.broker_service import create_app
from energraph.broker_tree import inspect_tree
from urbangraph.data import DataError


def test_versioned_broker_roles_and_revocation(
    tmp_path: Path, invented_research_rights: Any
) -> None:
    registry = BrokerRegistry(
        tmp_path / "private/registry.sqlite", tmp_path / "staging", tmp_path / "private/artifacts"
    )
    secrets: dict[str, Path] = {}
    for role, letter in (("admin", "a"), ("worker", "b"), ("reader", "c")):
        path = tmp_path / f"{role}.secret"
        path.write_text(letter * 48)
        secrets[role] = path
    client = TestClient(
        create_app(
            registry,
            admin_secret_file=secrets["admin"],
            worker_secret_file=secrets["worker"],
            reader_secret_file=secrets["reader"],
        )
    )

    def headers(role: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {secrets[role].read_text()}"}

    assert client.get("/v1/health").json() == {"version": 1, "status": "ready"}
    (registry.staging / "invented-raw").write_bytes(b"invented source")
    digest = hashlib.sha256(b"invented source").hexdigest()
    for purpose, operation in (
        ("local_research", "broker_admit"),
        ("research", "create"),
        ("research", "read"),
    ):
        invented_research_rights._grant(
            "invented-source",
            registry.staging / "invented-raw",
            digest,
            operation,
            registry.artifacts,
            purpose,
        )
    admission = {
        "stage_id": "invented-raw",
        "source_id": "invented-source",
        "expected_sha256": digest,
    }
    assert client.post("/v1/admit-source", json=admission).status_code == 401
    assert (
        client.post("/v1/admit-source", json=admission, headers=headers("worker")).status_code
        == 403
    )
    admitted = client.post("/v1/admit-source", json=admission, headers=headers("admin"))
    assert admitted.status_code == 200
    raw_id = admitted.json()["artifact_id"]
    (registry.staging / "invented-derived").write_bytes(b"invented derived")
    prepared = client.post(
        "/v1/prepare",
        json={
            "stage_id": "invented-derived",
            "parent_ids": [raw_id],
            "transformation": "invented-v1",
            "purpose": "research",
            "operation": "create",
        },
        headers=headers("worker"),
    )
    assert prepared.status_code == 200
    artifact_id = prepared.json()["artifact_id"]
    approval = {
        "artifact_id": artifact_id,
        "expected_sha256": hashlib.sha256(b"invented derived").hexdigest(),
    }
    assert client.post("/v1/approve", json=approval, headers=headers("worker")).status_code == 403
    use = {"artifact_id": artifact_id, "purpose": "research", "operation": "read"}
    assert client.post("/v1/authorize-read", json=use, headers=headers("reader")).status_code == 403
    assert client.post("/v1/approve", json=approval, headers=headers("admin")).status_code == 200
    assert (
        client.post(
            "/v1/commit", json={**use, "operation": "create"}, headers=headers("worker")
        ).status_code
        == 200
    )
    assert client.post("/v1/authorize-read", json=use, headers=headers("worker")).status_code == 403
    assert client.post("/v1/authorize-read", json=use, headers=headers("reader")).status_code == 200
    assert (
        client.post(
            "/v1/revoke", json={"source_id": "invented-source"}, headers=headers("admin")
        ).status_code
        == 200
    )
    assert client.post("/v1/authorize-read", json=use, headers=headers("reader")).status_code == 403


def test_invented_directory_admission_requires_separate_admin_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = BrokerRegistry(
        tmp_path / "private/registry.sqlite", tmp_path / "staging", tmp_path / "private/artifacts"
    )
    stage = registry.staging / "invented-source"
    stage.mkdir()
    (stage / "values.txt").write_bytes(b"invented only")
    digest = inspect_tree(stage).digest
    now = datetime.now(UTC)
    grants = [
        {
            "source_id": "invented-directory",
            "source_sha256": digest,
            "source_path": str(stage),
            "purpose": purpose,
            "operation": operation,
            "destination_root": str(registry.artifacts),
            "evidence_reference": "invented-directory-http-test-curator",
            "status": "approved",
            "effective_utc": (now - timedelta(minutes=1)).isoformat(),
            "expires_utc": (now + timedelta(minutes=10)).isoformat(),
            "retention_days": 1,
        }
        for purpose, operation in (
            ("local_research", "broker_admit"),
            ("research", "create"),
            ("research", "read"),
            ("graph_analysis", "energy_graph_create"),
            ("graph_analysis", "energy_graph_read"),
        )
    ]
    policy = tmp_path / "invented-policy.json"
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    monkeypatch.setenv("URBANGRAPH_RIGHTS_POLICY", str(policy))
    secrets: dict[str, Path] = {}
    for role, letter in (("admin", "a"), ("worker", "b"), ("reader", "c")):
        secret = tmp_path / f"{role}.secret"
        secret.write_text(letter * 48)
        secrets[role] = secret
    client = TestClient(
        create_app(
            registry,
            admin_secret_file=secrets["admin"],
            worker_secret_file=secrets["worker"],
            reader_secret_file=secrets["reader"],
        )
    )
    body = {
        "stage_id": "invented-source",
        "source_id": "invented-directory",
        "expected_sha256": digest,
        "directory": True,
    }
    worker = {"Authorization": "Bearer " + secrets["worker"].read_text()}
    admin = {"Authorization": "Bearer " + secrets["admin"].read_text()}
    reader = {"Authorization": "Bearer " + secrets["reader"].read_text()}
    assert client.post("/v1/admit-source", json=body, headers=worker).status_code == 403
    assert not list(registry.artifacts.iterdir())
    admitted = client.post("/v1/admit-source", json=body, headers=admin)
    assert admitted.status_code == 200
    artifact_id = admitted.json()["artifact_id"]
    response = client.post(
        "/v1/authorize-read",
        json={"artifact_id": artifact_id, "purpose": "research", "operation": "read"},
        headers=reader,
    )
    assert response.status_code == 200
    assert response.json()["content_sha256"] == digest
    delivered = client.post(
        "/v1/read-directory-file",
        json={
            "artifact_id": artifact_id,
            "relative_path": "values.txt",
            "purpose": "research",
            "operation": "read",
        },
        headers=reader,
    )
    assert delivered.status_code == 200 and delivered.content == b"invented only"
    assert (
        client.post(
            "/v1/read-directory-file",
            json={
                "artifact_id": artifact_id,
                "relative_path": "values.txt",
                "purpose": "research",
                "operation": "read",
            },
            headers=worker,
        ).status_code
        == 403
    )

    class LocalResponse(io.BytesIO):
        status = 200

    def local_transport(request: Any, timeout: float) -> LocalResponse:
        assert timeout == 10
        response = client.post(
            urlsplit(request.full_url).path,
            content=request.data,
            headers=dict(request.headers),
        )
        if response.status_code != 200:
            raise HTTPError(request.full_url, response.status_code, "denied", None, None)
        return LocalResponse(response.content)

    monkeypatch.setattr("energraph.broker_adapter.urlopen", local_transport)
    adapter = BrokerAdapter(
        "http://broker:8080",
        worker_token_file=secrets["worker"],
        reader_token_file=secrets["reader"],
        staging_root=registry.staging,
    )
    record = adapter.authorize_directory(artifact_id, purpose="research", operation="read")
    private_root = tmp_path / "materialized"
    private_root.mkdir(mode=0o700)
    restored = adapter.materialize_directory(
        artifact_id, private_root, purpose="research", operation="read"
    )
    assert (restored / "values.txt").read_bytes() == b"invented only"
    assert inspect_tree(restored).digest == record.content_sha256
    derived_stage = registry.staging / "invented-derived"
    derived_stage.mkdir()
    (derived_stage / "result.txt").write_bytes(b"derived only")
    pending = adapter.prepare_directory(
        stage_id="invented-derived",
        parent_ids=(artifact_id,),
        transformation="invented-transform-v1",
        purpose="research",
        operation="create",
    )
    with pytest.raises(DataError, match="Broker authorization or artifact unavailable"):
        adapter.commit_directory(pending)
    assert (
        client.post(
            "/v1/approve",
            json={
                "artifact_id": pending.artifact_id,
                "expected_sha256": inspect_tree(derived_stage).digest,
            },
            headers=admin,
        ).status_code
        == 200
    )
    committed = adapter.commit_directory(pending)
    assert committed.artifact_id == pending.artifact_id
    materialized = adapter.materialize_directory(
        committed.artifact_id, private_root, purpose="research", operation="read"
    )
    assert (materialized / "result.txt").read_bytes() == b"derived only"
    scientific_stage = tmp_path / "invented-scientific"
    scientific_stage.mkdir()
    (scientific_stage / "result.txt").write_bytes(b"invented graph output")
    (scientific_stage / "governance.json").write_text("{}", encoding="utf-8")
    raw_ref = resolve_broker_reference(
        adapter, artifact_id, purpose="graph_analysis", operation="energy_graph_create"
    )
    scientific_pending = prepare_transactional_publication(
        adapter, kind="energy-graph", stage=scientific_stage, parents=(raw_ref,)
    )
    assert scientific_pending.state == "AWAITING_APPROVAL"
    assert scientific_pending.prepared.parents == ((raw_ref.artifact_id, raw_ref.content_sha256),)
    with pytest.raises(DataError, match="Broker authorization or artifact unavailable"):
        finalize_transactional_publication(adapter, scientific_pending)

    def independent_approval(pending: Any) -> None:
        response = client.post(
            "/v1/approve",
            json={
                "artifact_id": pending.publication_id,
                "expected_sha256": pending.content_sha256,
            },
            headers=admin,
        )
        assert response.status_code == 200

    independent_approval(scientific_pending)
    committed_publication = finalize_transactional_publication(adapter, scientific_pending)
    scientific_ref = committed_publication.reference
    scientific_payload = governed_payload(
        scientific_ref,
        {"result.txt"},
        kind="energy-graph",
        broker=adapter,
        broker_parents=(raw_ref,),
        materialization_root=private_root,
    )
    assert (scientific_payload / "result.txt").read_bytes() == b"invented graph output"
    with pytest.raises(DataError, match="Broker artifact identity changed"):
        governed_payload(
            replace(scientific_ref, content_sha256="0" * 64),
            {"result.txt"},
            kind="energy-graph",
            broker=adapter,
            broker_parents=(raw_ref,),
            materialization_root=private_root,
        )
    with pytest.raises(DataError, match="Broker parent set differs"):
        governed_payload(
            scientific_ref,
            {"result.txt"},
            kind="energy-graph",
            broker=adapter,
            broker_parents=(),
            materialization_root=private_root,
        )
    with pytest.raises(DataError, match="Broker authority requires"):
        governed_payload(
            scientific_payload,
            {"result.txt"},
            kind="energy-graph",
            broker=adapter,
            broker_parents=(raw_ref,),
            materialization_root=private_root,
        )
    assert (
        client.post(
            "/v1/revoke", json={"source_id": "invented-directory"}, headers=admin
        ).status_code
        == 200
    )
    with pytest.raises(DataError, match="Broker authorization or artifact unavailable"):
        adapter.authorize_directory(artifact_id, purpose="research", operation="read")
    with pytest.raises(DataError, match="Broker authorization or artifact unavailable"):
        adapter.authorize_directory(committed.artifact_id, purpose="research", operation="read")
    with pytest.raises(DataError, match="Broker authorization or artifact unavailable"):
        governed_payload(
            scientific_ref,
            {"result.txt"},
            kind="energy-graph",
            broker=adapter,
            broker_parents=(raw_ref,),
            materialization_root=private_root,
        )
