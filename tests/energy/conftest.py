"""Scoped broker-backed contexts for invented EnerGraph integration fixtures."""

from __future__ import annotations

import io
import shutil
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from energraph.artifacts import (
    BrokerPublicationContext,
    resolve_broker_reference,
)
from energraph.broker_adapter import BrokerAdapter
from energraph.broker_registry import BrokerRegistry
from energraph.broker_service import create_app
from energraph.broker_tree import inspect_tree
from urbangraph.data import DataError, verify_snapshot


@pytest.fixture
def broker_context_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Create a test-only broker with explicit source-specific permissions."""

    def create(
        raw_path: Path,
        rights: Any,
        operations: tuple[tuple[str, str], ...],
        *,
        additional_raw_sources: tuple[tuple[Path, tuple[tuple[str, str], ...]], ...] = (),
    ) -> tuple[Any, ...]:
        registry = BrokerRegistry(
            tmp_path / "broker/registry.sqlite",
            tmp_path / "broker/staging",
            tmp_path / "broker/artifacts",
        )
        source_specs = ((raw_path, operations), *additional_raw_sources)
        admitted_ids: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {}
        staged_sources: list[tuple[str, Path, str]] = []
        for index, (source_path, source_operations) in enumerate(source_specs):
            checked = verify_snapshot(source_path)
            if not checked.valid or checked.manifest is None:
                raise AssertionError("Invented fixture raw snapshot must verify")
            source_id = checked.manifest.dataset_id
            if source_id in admitted_ids:
                raise AssertionError("Invented broker source identities must be distinct")
            stage = registry.staging / f"invented-energy-source-{index}"
            shutil.copytree(source_path, stage)
            digest = inspect_tree(stage).digest
            rights.broker_tree(
                source_id,
                stage,
                digest,
                (("local_research", "broker_admit"), *source_operations),
                registry.artifacts,
            )
            staged_sources.append((source_id, stage, digest))
        secret_paths = {}
        for role, character in (("admin", "a"), ("worker", "b"), ("reader", "c")):
            path = tmp_path / f"broker/{role}.secret"
            path.write_text(character * 48, encoding="ascii")
            secret_paths[role] = path
        client = TestClient(
            create_app(
                registry,
                admin_secret_file=secret_paths["admin"],
                worker_secret_file=secret_paths["worker"],
                reader_secret_file=secret_paths["reader"],
            )
        )

        class LocalResponse(io.BytesIO):
            status = 200

        def local_transport(request: Any, timeout: float) -> LocalResponse:
            if timeout != 10:
                raise AssertionError("Unexpected broker request timeout")
            response = client.post(
                urlsplit(request.full_url).path,
                content=request.data,
                headers=dict(request.headers),
            )
            if response.status_code != 200:
                raise HTTPError(request.full_url, response.status_code, "denied", Message(), None)
            return LocalResponse(response.content)

        monkeypatch.setattr("energraph.broker_adapter.urlopen", local_transport)
        for source_id, stage, digest in staged_sources:
            admitted = client.post(
                "/v1/admit-source",
                json={
                    "stage_id": stage.name,
                    "source_id": source_id,
                    "expected_sha256": digest,
                    "directory": True,
                },
                headers={
                    "Authorization": "Bearer " + secret_paths["admin"].read_text(encoding="ascii")
                },
            )
            if admitted.status_code != 200:
                raise AssertionError("Invented source admission failed")
            admitted_ids[source_id] = (admitted.json()["artifact_id"], source_operations)
        adapter = BrokerAdapter(
            "http://broker:8080",
            worker_token_file=secret_paths["worker"],
            reader_token_file=secret_paths["reader"],
            staging_root=registry.staging,
        )
        admitted_references = {
            source_id: resolve_broker_reference(
                adapter,
                artifact_id,
                purpose=source_operations[0][0],
                operation=source_operations[0][1],
            )
            for source_id, (artifact_id, source_operations) in admitted_ids.items()
        }
        primary = verify_snapshot(raw_path)
        if not primary.valid or primary.manifest is None:
            raise AssertionError("Invented fixture raw snapshot must verify")
        source_id = primary.manifest.dataset_id
        reference = admitted_references[source_id]

        def approve(pending: Any) -> None:
            result = client.post(
                "/v1/approve",
                json={
                    "artifact_id": pending.publication_id,
                    "expected_sha256": pending.content_sha256,
                },
                headers={
                    "Authorization": "Bearer " + secret_paths["admin"].read_text(encoding="ascii")
                },
            )
            if result.status_code != 200:
                raise DataError("Separate test policy actor rejected publication")

        materialized = tmp_path / "broker/materialized"
        materialized.mkdir(mode=0o700)
        context = BrokerPublicationContext(adapter, materialized)
        additional_references = {
            key: value for key, value in admitted_references.items() if key != source_id
        }
        if additional_references:
            return context, reference, approve, additional_references
        return context, reference, approve

    return create
