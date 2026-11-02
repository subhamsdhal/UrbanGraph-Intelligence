"""Immutable opaque-file datasets; parsing and preprocessing are separate stages."""

from urbangraph.data.contracts import (
    METR_LA,
    PEMS_BAY,
    DataError,
    DatasetIdentity,
    DatasetManifest,
    DatasetSnapshot,
    DatasetSource,
    RawFileRecord,
    content_identity,
)
from urbangraph.data.ingestion import ingest_dataset
from urbangraph.data.verification import VerificationResult, verify_snapshot

__all__ = [
    "METR_LA",
    "PEMS_BAY",
    "DataError",
    "DatasetIdentity",
    "DatasetManifest",
    "DatasetSnapshot",
    "DatasetSource",
    "RawFileRecord",
    "VerificationResult",
    "content_identity",
    "ingest_dataset",
    "verify_snapshot",
]
