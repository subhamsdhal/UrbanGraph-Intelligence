"""Wire-shape checks are necessary, but cannot establish broker custody."""

from copy import deepcopy

import pytest

from energraph.broker_contract import parse_committed_record
from urbangraph.data import DataError


def _record() -> dict[str, object]:
    return {
        "version": 1,
        "state": "committed",
        "artifact_id": "invented-derived-1",
        "content_sha256": "a" * 64,
        "kind": "derived",
        "parents": [{"artifact_id": "invented-parent-1", "content_sha256": "b" * 64}],
        "sources": [{"source_id": "invented-only", "content_sha256": "c" * 64}],
        "revision": 1,
        "transformation": "invented-v1",
        "issuer": "invented-curator",
        "audit_id": "invented-audit",
    }


def test_strict_committed_record() -> None:
    result = parse_committed_record(_record())
    assert result.parents == (("invented-parent-1", "b" * 64),)
    assert result.sources == (("invented-only", "c" * 64),)
    assert result.revision == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", 2),
        ("state", "prepared"),
        ("revision", 0),
        ("revision", True),
        ("content_sha256", "bad"),
        ("sources", []),
        ("sources", [{"source_id": "invented-only", "content_sha256": "bad"}]),
    ],
)
def test_broker_record_rejects_invalid_fields(field: str, value: object) -> None:
    record = _record()
    record[field] = value
    with pytest.raises(DataError):
        parse_committed_record(record)


def test_broker_record_rejects_extra_field_and_duplicate_parent() -> None:
    record = _record()
    record["caller_claimed_authority"] = True
    with pytest.raises(DataError):
        parse_committed_record(record)
    duplicate = _record()
    parents = deepcopy(duplicate["parents"])
    assert isinstance(parents, list)
    duplicate["parents"] = parents + parents
    with pytest.raises(DataError):
        parse_committed_record(duplicate)
