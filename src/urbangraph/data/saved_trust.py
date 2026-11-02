"""Versioned, deliberately limited trust claims for saved scientific runs.

An in-process publisher cannot attest who computed caller-supplied payloads.
"""

from __future__ import annotations

from urbangraph.data import DataError

TRUST_KEY = "saved_run_trust"
UNVERIFIED = {
    "schema_version": "1.0",
    "submission": "caller_supplied",
    "authorship_assurance": "unverified",
    "scientific_validation": "not_established",
    "production_approval": "not_established",
}


def classify_for_publication(descriptor: dict[str, object]) -> dict[str, object]:
    """Reject attempts to assert trust; stamp the only supportable status."""
    forbidden = {
        "valid",
        "verified",
        "trusted",
        "validated",
        "production_ready",
        "promotion_approved",
        "authorship_assurance",
        "scientific_validation",
        "production_approval",
    }
    if forbidden.intersection(descriptor) or descriptor.get(TRUST_KEY, UNVERIFIED) != UNVERIFIED:
        raise DataError("Caller cannot assert saved-run trust or endorsement")
    return descriptor | {TRUST_KEY: dict(UNVERIFIED)}


def verification_status(descriptor: dict[str, object]) -> dict[str, object]:
    """Integrity is checked by caller; missing legacy claims never imply trust."""
    claim = descriptor.get(TRUST_KEY)
    if claim is None:
        classification = "legacy_unknown"
    elif claim == UNVERIFIED:
        classification = "caller_supplied_unverified"
    else:
        raise DataError("Unsupported saved-run trust assertion")
    return {
        "valid": True,  # compatibility: byte/content integrity only
        "integrity": "pass",
        "classification": classification,
        "authorship_assurance": "unverified",
        "scientific_validation": "not_established",
        "production_approval": "not_established",
    }
