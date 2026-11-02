"""Broker core tests with invented bytes; no claim of independent custody."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from energraph.broker_registry import BrokerRegistry
from urbangraph.data import DataError


def _broker(tmp_path: Path) -> BrokerRegistry:
    return BrokerRegistry(
        tmp_path / "private/registry.sqlite", tmp_path / "staging", tmp_path / "private/artifacts"
    )


def _invented_tree_rights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    broker: BrokerRegistry,
    sources: tuple[tuple[str, str], ...],
) -> dict[str, str]:
    """Test curator grants exact invented trees; the worker receives no policy role."""
    from energraph.broker_tree import inspect_tree

    now = datetime.now(UTC)
    grants: list[dict[str, object]] = []
    digests: dict[str, str] = {}
    for source_id, stage_id in sources:
        stage = broker.staging / stage_id
        assert stage.is_relative_to(tmp_path) and stage.is_dir()
        digest = inspect_tree(stage).digest
        digests[source_id] = digest
        for purpose, operation in (
            ("local_research", "broker_admit"),
            ("research", "create"),
            ("research", "read"),
        ):
            grants.append(
                {
                    "source_id": source_id,
                    "source_sha256": digest,
                    "source_path": str(stage),
                    "purpose": purpose,
                    "operation": operation,
                    "destination_root": str(broker.artifacts),
                    "evidence_reference": "invented-broker-tree-test-curator",
                    "status": "approved",
                    "effective_utc": (now - timedelta(minutes=1)).isoformat(),
                    "expires_utc": (now + timedelta(minutes=10)).isoformat(),
                    "retention_days": 1,
                }
            )
    policy = tmp_path / "invented-tree-policy.json"
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    monkeypatch.setenv("URBANGRAPH_RIGHTS_POLICY", str(policy))
    return digests


def test_two_invented_raw_directories_admit_and_derive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from energraph.broker_tree import inspect_tree

    broker = _broker(tmp_path)
    for stage_id, content in (
        ("first-tree", b"first invented"),
        ("second-tree", b"second invented"),
    ):
        stage = broker.staging / stage_id
        stage.mkdir()
        (stage / "invented.txt").write_bytes(content)
    digests = _invented_tree_rights(
        tmp_path,
        monkeypatch,
        broker,
        (("invented-first", "first-tree"), ("invented-second", "second-tree")),
    )
    first = broker.admit_source(
        stage_id="first-tree",
        source_id="invented-first",
        expected_sha256=digests["invented-first"],
        issuer="invented-curator",
        directory=True,
    )
    second = broker.admit_source(
        stage_id="second-tree",
        source_id="invented-second",
        expected_sha256=digests["invented-second"],
        issuer="invented-curator",
        directory=True,
    )
    assert (
        broker.authorize_read(first, purpose="research", operation="read")["content_sha256"]
        == digests["invented-first"]
    )
    assert (
        broker.read_directory_file(first, "invented.txt", purpose="research", operation="read")
        == b"first invented"
    )
    with pytest.raises(DataError, match="No matching rights grant"):
        broker.read_directory_file(first, "invented.txt", purpose="wrong", operation="read")
    with pytest.raises(DataError):
        broker.read_directory_file(first, "../invented.txt", purpose="research", operation="read")
    stage = broker.staging / "derived-tree"
    stage.mkdir()
    (stage / "result.txt").write_bytes(b"synthetic result")
    derived = broker.prepare(
        stage_id="derived-tree",
        parent_ids=(first, second),
        transformation="invented-transform",
        purpose="research",
        operation="create",
        directory=True,
    )
    broker.approve(derived, issuer="invented-curator", expected_sha256=inspect_tree(stage).digest)
    record = broker.commit(derived, purpose="research", operation="create")
    assert [s["source_id"] for s in record["sources"]] == ["invented-first", "invented-second"]
    assert (
        broker.authorize_read(derived, purpose="research", operation="read")["artifact_id"]
        == derived
    )
    restarted = BrokerRegistry(broker.db_path, broker.staging, broker.artifacts)
    assert restarted.authorize_read(derived, purpose="research", operation="read") == record
    with pytest.raises(DataError, match="No matching rights grant"):
        restarted.authorize_read(derived, purpose="other", operation="read")
    broker.revoke("invented-second")
    with pytest.raises(DataError, match="rights unavailable"):
        broker.authorize_read(derived, purpose="research", operation="read")


def test_raw_directory_admission_rejects_missing_or_wrong_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broker = _broker(tmp_path)
    stage = broker.staging / "invented-tree"
    stage.mkdir()
    (stage / "data.txt").write_bytes(b"invented")
    from energraph.broker_tree import inspect_tree

    digest = inspect_tree(stage).digest
    with pytest.raises(DataError, match="rights policy unavailable"):
        broker.admit_source(
            stage_id="invented-tree",
            source_id="invented-one",
            expected_sha256=digest,
            issuer="invented-curator",
            directory=True,
        )
    _invented_tree_rights(tmp_path, monkeypatch, broker, (("invented-one", "invented-tree"),))
    with pytest.raises(DataError, match="approver-pinned"):
        broker.admit_source(
            stage_id="invented-tree",
            source_id="invented-one",
            expected_sha256="0" * 64,
            issuer="invented-curator",
            directory=True,
        )
    (stage / "data.txt").write_bytes(b"changed")
    with pytest.raises(DataError, match="approver-pinned"):
        broker.admit_source(
            stage_id="invented-tree",
            source_id="invented-one",
            expected_sha256=digest,
            issuer="invented-curator",
            directory=True,
        )


def test_raw_directory_expired_grant_denies_without_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from energraph.broker_tree import inspect_tree

    broker = _broker(tmp_path)
    stage = broker.staging / "invented-tree"
    stage.mkdir()
    (stage / "data.txt").write_bytes(b"invented")
    digest = _invented_tree_rights(
        tmp_path, monkeypatch, broker, (("invented-one", "invented-tree"),)
    )["invented-one"]
    policy = tmp_path / "invented-tree-policy.json"
    document = json.loads(policy.read_text(encoding="utf-8"))
    for grant in document["grants"]:
        grant["expires_utc"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    policy.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(DataError, match="outside validity window"):
        broker.admit_source(
            stage_id="invented-tree",
            source_id="invented-one",
            expected_sha256=digest,
            issuer="invented-curator",
            directory=True,
        )
    assert inspect_tree(stage).digest == digest
    assert not list(broker.artifacts.iterdir())


def _admit(broker: BrokerRegistry, rights: Any, source_id: str = "invented-one") -> str:
    source = broker.staging / source_id
    source.write_bytes(b"invented source only")
    digest = hashlib.sha256(b"invented source only").hexdigest()
    for purpose, operation in (
        ("local_research", "broker_admit"),
        ("research", "create"),
        ("research", "read"),
    ):
        rights._grant(source_id, source, digest, operation, broker.artifacts, purpose)
    return broker.admit_source(
        stage_id=source_id,
        source_id=source_id,
        expected_sha256=digest,
        issuer="invented-curator",
    )


def test_approved_derived_publication_read_and_revocation(
    tmp_path: Path, invented_research_rights: Any
) -> None:
    broker = _broker(tmp_path)
    parent = _admit(broker, invented_research_rights)
    (broker.staging / "derived-stage").write_bytes(b"invented derived only")
    artifact = broker.prepare(
        stage_id="derived-stage",
        parent_ids=(parent,),
        transformation="invented-transform",
        purpose="research",
        operation="create",
    )
    with pytest.raises(DataError, match="unavailable"):
        broker.authorize_read(artifact, purpose="research", operation="read")
    with pytest.raises(DataError, match="not approved"):
        broker.commit(artifact, purpose="research", operation="create")
    broker.approve(
        artifact,
        issuer="invented-curator",
        expected_sha256=hashlib.sha256(b"invented derived only").hexdigest(),
    )
    record = broker.commit(artifact, purpose="research", operation="create")
    assert isinstance(record["sources"], list)
    assert len(record["sources"]) == 1
    assert record["sources"][0]["source_id"] == "invented-one"
    assert broker.authorize_read(artifact, purpose="research", operation="read") == record
    with pytest.raises(DataError, match="No matching rights grant"):
        broker.authorize_read(artifact, purpose="other", operation="read")
    broker.revoke("invented-one")
    with pytest.raises(DataError, match="rights unavailable"):
        broker.authorize_read(artifact, purpose="research", operation="read")


def test_missing_parent_and_changed_stage_fail_closed(
    tmp_path: Path, invented_research_rights: Any
) -> None:
    broker = _broker(tmp_path)
    parent = _admit(broker, invented_research_rights)
    (broker.staging / "derived-stage").write_bytes(b"first")
    with pytest.raises(DataError, match="unknown"):
        broker.prepare(
            stage_id="derived-stage",
            parent_ids=("forged-parent",),
            transformation="invented-transform",
            purpose="research",
            operation="create",
        )
    artifact = broker.prepare(
        stage_id="derived-stage",
        parent_ids=(parent,),
        transformation="invented-transform",
        purpose="research",
        operation="create",
    )
    broker.approve(
        artifact, issuer="invented-curator", expected_sha256=hashlib.sha256(b"first").hexdigest()
    )
    (broker.staging / "derived-stage").write_bytes(b"changed")
    with pytest.raises(DataError, match="content changed"):
        broker.commit(artifact, purpose="research", operation="create")
    assert not (broker.artifacts / artifact).exists()


def test_source_tampering_denies_derived_read(
    tmp_path: Path, invented_research_rights: Any
) -> None:
    broker = _broker(tmp_path)
    parent = _admit(broker, invented_research_rights)
    (broker.staging / "derived-stage").write_bytes(b"derived")
    artifact = broker.prepare(
        stage_id="derived-stage",
        parent_ids=(parent,),
        transformation="invented-transform",
        purpose="research",
        operation="create",
    )
    broker.approve(
        artifact, issuer="invented-curator", expected_sha256=hashlib.sha256(b"derived").hexdigest()
    )
    broker.commit(artifact, purpose="research", operation="create")
    (broker.artifacts / parent).write_bytes(b"tampered")
    with pytest.raises(DataError, match="source content changed"):
        broker.authorize_read(artifact, purpose="research", operation="read")


def test_two_source_transitive_closure_and_revocation(
    tmp_path: Path, invented_research_rights: Any
) -> None:
    broker = _broker(tmp_path)
    first = _admit(broker, invented_research_rights, "invented-one")
    second = _admit(broker, invented_research_rights, "invented-two")
    (broker.staging / "combined").write_bytes(b"combined")
    combined = broker.prepare(
        stage_id="combined",
        parent_ids=(second, first),
        transformation="invented-combine",
        purpose="research",
        operation="create",
    )
    broker.approve(
        combined,
        issuer="invented-curator",
        expected_sha256=hashlib.sha256(b"combined").hexdigest(),
    )
    broker.commit(combined, purpose="research", operation="create")
    (broker.staging / "nested").write_bytes(b"nested")
    nested = broker.prepare(
        stage_id="nested",
        parent_ids=(combined,),
        transformation="invented-nest",
        purpose="research",
        operation="create",
    )
    broker.approve(
        nested,
        issuer="invented-curator",
        expected_sha256=hashlib.sha256(b"nested").hexdigest(),
    )
    record = broker.commit(nested, purpose="research", operation="create")
    sources = record["sources"]
    assert isinstance(sources, list)
    assert [item["source_id"] for item in sources] == ["invented-one", "invented-two"]
    broker.revoke("invented-two")
    with pytest.raises(DataError, match="rights unavailable"):
        broker.authorize_read(nested, purpose="research", operation="read")


def test_wrong_purpose_denies_before_protected_artifact_read(
    tmp_path: Path, invented_research_rights: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    broker = _broker(tmp_path)
    parent = _admit(broker, invented_research_rights)
    protected_reads: list[Path] = []
    original = Path.read_bytes

    def observed(path: Path) -> bytes:
        if path.is_relative_to(broker.artifacts):
            protected_reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", observed)
    with pytest.raises(DataError, match="No matching rights grant"):
        broker.authorize_read(parent, purpose="unauthorized", operation="read")
    assert protected_reads == []


def test_commit_rejects_context_changed_after_prepare(
    tmp_path: Path, invented_research_rights: Any
) -> None:
    broker = _broker(tmp_path)
    parent = _admit(broker, invented_research_rights)
    (broker.staging / "derived-stage").write_bytes(b"invented")
    artifact = broker.prepare(
        stage_id="derived-stage",
        parent_ids=(parent,),
        transformation="invented-transform",
        purpose="research",
        operation="create",
    )
    broker.approve(
        artifact, issuer="invented-curator", expected_sha256=hashlib.sha256(b"invented").hexdigest()
    )
    with pytest.raises(DataError, match="context differs"):
        broker.commit(artifact, purpose="research", operation="read")
    assert not (broker.artifacts / artifact).exists()
    assert broker.commit(artifact, purpose="research", operation="create")["state"] == "committed"


def test_derived_parent_content_tampering_denies_nested_read(
    tmp_path: Path, invented_research_rights: Any
) -> None:
    broker = _broker(tmp_path)
    raw = _admit(broker, invented_research_rights)
    parent = raw
    for stage_id in ("first-derived", "second-derived"):
        content = stage_id.encode()
        (broker.staging / stage_id).write_bytes(content)
        artifact = broker.prepare(
            stage_id=stage_id,
            parent_ids=(parent,),
            transformation="invented-transform",
            purpose="research",
            operation="create",
        )
        broker.approve(
            artifact, issuer="invented-curator", expected_sha256=hashlib.sha256(content).hexdigest()
        )
        broker.commit(artifact, purpose="research", operation="create")
        parent = artifact
    record = broker.authorize_read(parent, purpose="research", operation="read")
    parents = record["parents"]
    assert isinstance(parents, list)
    first = parents[0]["artifact_id"]
    (broker.artifacts / first).write_bytes(b"tampered")
    with pytest.raises(DataError, match="direct parent content changed"):
        broker.authorize_read(parent, purpose="research", operation="read")


def test_directory_publication_and_revocation(
    tmp_path: Path, invented_research_rights: Any
) -> None:
    broker = _broker(tmp_path)
    raw = _admit(broker, invented_research_rights)
    stage = broker.staging / "directory-stage"
    stage.mkdir()
    (stage / "descriptor.json").write_bytes(b"{}")
    (stage / "trial-0").mkdir()
    (stage / "trial-0" / "weights.bin").write_bytes(b"invented")
    from energraph.broker_tree import inspect_tree

    expected = inspect_tree(stage)
    artifact = broker.prepare(
        stage_id="directory-stage",
        parent_ids=(raw,),
        transformation="invented-directory-transform",
        purpose="research",
        operation="create",
        directory=True,
    )
    broker.approve(artifact, issuer="invented-curator", expected_sha256=expected.digest)
    committed = broker.commit(artifact, purpose="research", operation="create")
    assert committed["version"] == 2
    snapshot = committed["snapshot"]
    assert isinstance(snapshot, dict)
    assert snapshot["schema"] == "broker-tree-v1"
    assert committed["content_sha256"] == expected.digest
    assert inspect_tree(broker.artifacts / artifact) == expected
    assert broker.authorize_read(artifact, purpose="research", operation="read") == committed
    assert (
        broker.read_directory_file(
            artifact, "trial-0/weights.bin", purpose="research", operation="read"
        )
        == b"invented"
    )
    (broker.artifacts / artifact / "trial-0" / "weights.bin").write_bytes(b"tampered")
    with pytest.raises(DataError, match="snapshot content changed"):
        broker.authorize_read(artifact, purpose="research", operation="read")
    with pytest.raises(DataError, match="snapshot content changed"):
        broker.read_directory_file(
            artifact, "trial-0/weights.bin", purpose="research", operation="read"
        )
    broker.revoke("invented-one")
    with pytest.raises(DataError, match="rights unavailable"):
        broker.authorize_read(artifact, purpose="research", operation="read")


def test_directory_stage_change_denies_commit(
    tmp_path: Path, invented_research_rights: Any
) -> None:
    broker = _broker(tmp_path)
    raw = _admit(broker, invented_research_rights)
    stage = broker.staging / "directory-stage"
    stage.mkdir()
    (stage / "descriptor.json").write_bytes(b"{}")
    from energraph.broker_tree import inspect_tree

    digest = inspect_tree(stage).digest
    artifact = broker.prepare(
        stage_id="directory-stage",
        parent_ids=(raw,),
        transformation="invented-directory-transform",
        purpose="research",
        operation="create",
        directory=True,
    )
    broker.approve(artifact, issuer="invented-curator", expected_sha256=digest)
    (stage / "extra.txt").write_bytes(b"unexpected")
    with pytest.raises(DataError, match="directory changed"):
        broker.commit(artifact, purpose="research", operation="create")
    assert not (broker.artifacts / artifact).exists()


def test_directory_parent_transitive_two_source_closure(
    tmp_path: Path, invented_research_rights: Any
) -> None:
    broker = _broker(tmp_path)
    first = _admit(broker, invented_research_rights, "invented-one")
    second = _admit(broker, invented_research_rights, "invented-two")
    (broker.staging / "parent-tree").mkdir()
    (broker.staging / "parent-tree" / "payload.txt").write_bytes(b"parent")
    from energraph.broker_tree import inspect_tree

    parent = broker.prepare(
        stage_id="parent-tree",
        parent_ids=(first, second),
        transformation="invented-parent-tree",
        purpose="research",
        operation="create",
        directory=True,
    )
    broker.approve(
        parent,
        issuer="invented-curator",
        expected_sha256=inspect_tree(broker.staging / "parent-tree").digest,
    )
    broker.commit(parent, purpose="research", operation="create")
    (broker.staging / "child-tree").mkdir()
    (broker.staging / "child-tree" / "payload.txt").write_bytes(b"child")
    child = broker.prepare(
        stage_id="child-tree",
        parent_ids=(parent,),
        transformation="invented-child-tree",
        purpose="research",
        operation="create",
        directory=True,
    )
    broker.approve(
        child,
        issuer="invented-curator",
        expected_sha256=inspect_tree(broker.staging / "child-tree").digest,
    )
    record = broker.commit(child, purpose="research", operation="create")
    sources = record["sources"]
    assert isinstance(sources, list)
    assert [item["source_id"] for item in sources] == ["invented-one", "invented-two"]
    (broker.artifacts / parent / "payload.txt").write_bytes(b"tampered")
    with pytest.raises(DataError, match="direct parent content changed"):
        broker.authorize_read(child, purpose="research", operation="read")


def test_prepare_verifies_every_parent_not_only_last(
    tmp_path: Path, invented_research_rights: Any
) -> None:
    broker = _broker(tmp_path)
    first = _admit(broker, invented_research_rights, "invented-one")
    second = _admit(broker, invented_research_rights, "invented-two")
    (broker.staging / "derived-stage").write_bytes(b"invented")
    (broker.artifacts / first).write_bytes(b"tampered first parent")
    with pytest.raises(DataError, match="parent content changed"):
        broker.prepare(
            stage_id="derived-stage",
            parent_ids=(first, second),
            transformation="invented-transform",
            purpose="research",
            operation="create",
        )


def test_directory_recovery_after_complete_copy_before_registry_commit(
    tmp_path: Path, invented_research_rights: Any
) -> None:
    from energraph.broker_tree import copy_verified_tree, inspect_tree

    broker = _broker(tmp_path)
    raw = _admit(broker, invented_research_rights)
    stage = broker.staging / "recoverable-tree"
    stage.mkdir()
    (stage / "descriptor.json").write_bytes(b"invented")
    identity = inspect_tree(stage)
    artifact = broker.prepare(
        stage_id="recoverable-tree",
        parent_ids=(raw,),
        transformation="invented-recoverable",
        purpose="research",
        operation="create",
        directory=True,
    )
    broker.approve(artifact, issuer="invented-curator", expected_sha256=identity.digest)
    copy_verified_tree(stage, broker.artifacts / artifact, identity)
    with pytest.raises(DataError, match="unavailable"):
        broker.authorize_read(artifact, purpose="research", operation="read")
    # A restarted broker recovers the approved record only after rechecking
    # current rights, staged bytes and the complete already-copied tree.
    restarted = _broker(tmp_path)
    committed = restarted.commit(artifact, purpose="research", operation="create")
    assert committed["content_sha256"] == identity.digest
    assert inspect_tree(restarted.artifacts / artifact) == identity


def test_incomplete_orphan_directory_remains_untrusted(
    tmp_path: Path, invented_research_rights: Any
) -> None:
    broker = _broker(tmp_path)
    _admit(broker, invented_research_rights)
    orphan = broker.artifacts / "derived-orphan"
    orphan.mkdir()
    (orphan / "incomplete.txt").write_bytes(b"invented")
    with pytest.raises(DataError, match="unavailable"):
        broker.authorize_read("derived-orphan", purpose="research", operation="read")
