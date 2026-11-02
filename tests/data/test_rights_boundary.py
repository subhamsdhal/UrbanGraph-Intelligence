"""Observable default-deny raw-ingestion behavior on invented local sources."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from urbangraph.benchmarks.pipeline import import_local
from urbangraph.benchmarks.sources import Acquisition
from urbangraph.data import DataError, RawFileRecord, content_identity, ingest_dataset
from urbangraph.data.filesystem import stream_file
from urbangraph.data.rights import (
    AuthorizationRequest,
    require_authorization_decisions,
    require_authorizations,
    require_raw_authorization,
)


def _grant(
    source: Path, root: Path, *, status: str = "approved", operation: str = "raw_ingest"
) -> dict[str, object]:
    now = datetime.now(UTC)
    return {
        "source_id": "invented-source",
        "source_sha256": content_identity(
            (
                RawFileRecord(
                    "invented.txt",
                    len(b"invented only\n"),
                    hashlib.sha256(b"invented only\n").hexdigest(),
                ),
            )
        ),
        "source_path": str(source),
        "purpose": "local_research",
        "operation": operation,
        "destination_root": str(root),
        "evidence_reference": "invented-test-only",
        "status": status,
        "effective_utc": (now - timedelta(days=1)).isoformat(),
        "expires_utc": (now + timedelta(days=1)).isoformat(),
        "retention_days": 1,
    }


def test_missing_policy_denies_before_source_read_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("URBANGRAPH_RIGHTS_POLICY", raising=False)
    source, destination = tmp_path / "missing-source", tmp_path / "raw"
    with pytest.raises(DataError, match="policy unavailable"):
        ingest_dataset(source, "invented-source", raw_root=destination)
    assert not destination.exists()


def test_hold_policy_denies_before_source_read_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, destination = tmp_path / "missing-source", tmp_path / "raw"
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps({"version": 1, "grants": [_grant(source, destination, status="HOLD")]})
    )
    monkeypatch.setenv("URBANGRAPH_RIGHTS_POLICY", str(policy))
    with pytest.raises(DataError, match="not approved"):
        ingest_dataset(source, "invented-source", raw_root=destination)
    assert not destination.exists()


def test_authorized_invented_source_executes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, destination = tmp_path / "source", tmp_path / "raw"
    source.mkdir()
    (source / "invented.txt").write_bytes(b"invented only\n")
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"version": 1, "grants": [_grant(source, destination)]}))
    monkeypatch.setenv("URBANGRAPH_RIGHTS_POLICY", str(policy))
    result = ingest_dataset(source, "invented-source", raw_root=destination)
    assert (result.directory / "payload" / "invented.txt").read_text() == "invented only\n"


def test_shared_research_fixture_grant_is_tmp_scoped_and_content_bound(
    tmp_path: Path, invented_research_rights: object
) -> None:
    rights = invented_research_rights
    source = tmp_path / "source"
    source.mkdir()
    (source / "fixture.csv").write_text("invented fixture only\n", encoding="utf-8")
    rights.source(source, "invented-fixture", tmp_path / "raw")
    grant = rights.grants[-1]
    assert grant["source_path"] == str(source.resolve())
    assert grant["source_sha256"] == content_identity(
        (RawFileRecord("fixture.csv", *stream_file(source / "fixture.csv")),)
    )
    assert grant["evidence_reference"] == "invented-offline-research-fixture-only"
    with pytest.raises(ValueError, match="under this test's tmp_path"):
        rights.source(tmp_path.parent / "outside", "invented-fixture", tmp_path / "raw")
    with pytest.raises(DataError, match="Parent traversal"):
        rights.source(source / ".." / "source", "invented-fixture", tmp_path / "raw")


def test_shared_research_fixture_grant_rejects_digest_mismatch(
    tmp_path: Path, invented_research_rights: object
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "fixture.csv").write_text("invented fixture only\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest does not match"):
        invented_research_rights._grant(
            "invented-fixture", source, "0" * 64, "raw_ingest", tmp_path / "raw"
        )


def test_shared_research_fixture_grant_rejects_linked_source_when_supported(
    tmp_path: Path, invented_research_rights: object
) -> None:
    outside = tmp_path.parent / (tmp_path.name + "-shared-link-target")
    outside.mkdir()
    (outside / "fixture.csv").write_text("invented fixture only\n", encoding="utf-8")
    link = tmp_path / "source"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Host cannot create directory symlinks: {type(exc).__name__}")
    with pytest.raises(DataError, match="Symlinks and Windows reparse points"):
        invented_research_rights.source(link, "invented-fixture", tmp_path / "raw")


def test_no_caller_policy_parameter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("URBANGRAPH_RIGHTS_POLICY", raising=False)
    with pytest.raises(TypeError):
        ingest_dataset(
            tmp_path / "src",
            "invented-source",
            raw_root=tmp_path / "raw",
            policy_path=tmp_path / "forged",
        )  # type: ignore[call-arg]


def test_duplicate_grant_denied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, destination = tmp_path / "source", tmp_path / "raw"
    grant = _grant(source, destination)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"version": 1, "grants": [grant, grant]}))
    monkeypatch.setenv("URBANGRAPH_RIGHTS_POLICY", str(policy))
    with pytest.raises(DataError, match="Conflicting"):
        require_raw_authorization("invented-source", source, "local_research", destination)


def test_relative_policy_path_denied_before_input_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("URBANGRAPH_RIGHTS_POLICY", "caller-policy.json")
    with pytest.raises(DataError, match="must be absolute"):
        ingest_dataset(tmp_path / "missing", "invented-source", raw_root=tmp_path / "raw")
    assert not (tmp_path / "raw").exists()


def test_benchmark_import_denies_before_file_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("URBANGRAPH_RIGHTS_POLICY", raising=False)
    with pytest.raises(DataError, match="policy unavailable"):
        import_local(
            tmp_path / "not-present.h5",
            "metr-la",
            Acquisition("invented", "invented", "real", "test only"),
            raw_root=tmp_path / "raw",
        )
    assert not (tmp_path / "raw").exists()


def test_complete_source_set_uses_one_policy_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from urbangraph.data import rights

    first, second, destination = tmp_path / "first", tmp_path / "second", tmp_path / "out"
    first_grant = _grant(first, destination)
    second_grant = _grant(second, destination)
    second_grant["source_id"] = "other-invented-source"
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"version": 1, "grants": [first_grant, second_grant]}))
    monkeypatch.setenv("URBANGRAPH_RIGHTS_POLICY", str(policy))
    original = rights._load_policy
    calls = 0

    def counted() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(rights, "_load_policy", counted)
    requests = (
        AuthorizationRequest("invented-source", first, "local_research", destination, "raw_ingest"),
        AuthorizationRequest(
            "other-invented-source", second, "local_research", destination, "raw_ingest"
        ),
    )
    assert len(require_authorizations(requests)) == 2
    assert calls == 1
    second_grant["status"] = "revoked"
    policy.write_text(json.dumps({"version": 1, "grants": [first_grant, second_grant]}))
    with pytest.raises(DataError, match="not approved"):
        require_authorizations(requests)
    with pytest.raises(DataError, match="Duplicate upstream"):
        require_authorizations((requests[0], requests[0]))


def test_detailed_decision_retains_policy_scope_without_granting_new_rights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, destination = tmp_path / "source", tmp_path / "derived"
    grant = _grant(source, destination, operation="derived_create")
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"version": 1, "grants": [grant]}))
    monkeypatch.setenv("URBANGRAPH_RIGHTS_POLICY", str(policy))
    request = AuthorizationRequest(
        "invented-source", source, "local_research", destination, "derived_create"
    )
    decision = require_authorization_decisions((request,))[0]
    assert decision.source_sha256 == grant["source_sha256"]
    assert decision.source_path == source.resolve()
    assert decision.destination_root == destination.resolve()
    assert decision.operation == "derived_create"
    assert decision.evidence_reference == "invented-test-only"
    assert decision.retention_days == 1
    assert decision.effective_utc < decision.expires_utc
    with pytest.raises(DataError, match="No matching"):
        require_authorization_decisions(
            (
                AuthorizationRequest(
                    "invented-source", source, "local_research", destination, "derived_commit"
                ),
            )
        )
