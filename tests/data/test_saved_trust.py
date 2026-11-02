"""Saved-run integrity must not become a scientific endorsement."""

import pytest

from urbangraph.data import DataError
from urbangraph.data.saved_trust import classify_for_publication, verification_status


def test_legacy_has_no_implicit_authorship() -> None:
    result = verification_status({"model": "legacy"})
    assert result["integrity"] == "pass"
    assert result["classification"] == "legacy_unknown"
    assert result["authorship_assurance"] == "unverified"
    assert result["scientific_validation"] == "not_established"
    assert result["production_approval"] == "not_established"


@pytest.mark.parametrize(
    "claim",
    [
        {"trusted": True},
        {"validated": True},
        {"production_ready": True},
        {"saved_run_trust": {"authorship_assurance": "verified"}},
    ],
)
def test_direct_caller_cannot_assert_trust(claim: dict[str, object]) -> None:
    with pytest.raises(DataError):
        classify_for_publication(claim)


def test_unsupported_legacy_claim_fails_closed() -> None:
    with pytest.raises(DataError):
        verification_status({"saved_run_trust": {"schema_version": "future"}})
